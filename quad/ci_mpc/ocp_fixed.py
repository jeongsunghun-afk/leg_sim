"""Fixed-schedule whole-body trot OCP for the 17-DOF 02_Leg (Phase 0.3).

Crocoddyl Box-FDDP over the waist-locked 16-DOF Pinocchio model (model_bridge.py).
The contact schedule (which feet touch in which node) is PRE-DEFINED here - this is
the fixed-schedule baseline that Phase 2 replaces with a contact-implicit model.

Structure mirrors crocoddyl's SimpleQuadrupedalGaitProblem: each node is a
DifferentialActionModelContactFwdDynamics with ContactModel3D on stance feet and a
CostModelSum (state reg, control reg, friction cone on stance feet, swing-foot
position/velocity tracking, base linear/angular velocity command). Wrapped in
IntegratedActionModelEuler(dt) and solved with SolverBoxFDDP.
"""
import numpy as np
import pinocchio as pin
import crocoddyl
from model_bridge import MjPinBridge, FEET


def standing_ik(br, base_z=0.42, foot_xy=None, foot_z=0.018, tol=1e-6, iters=200):
    """IK a symmetric standing pose: feet under nominal hips at ground, base at base_z.

    foot_z = contact-frame (sphere center) height when the sphere rests on the ground,
    i.e. the sphere radius.
    """
    m, d = br.model, br.data
    if foot_xy is None:
        foot_xy = {'FL': (0.30, 0.16), 'FR': (0.30, -0.16),
                   'HL': (-0.28, 0.16), 'HR': (-0.28, -0.16)}
    targets = {L: np.array([foot_xy[L][0], foot_xy[L][1], foot_z]) for L in FEET}
    q = pin.neutral(m); q[2] = base_z
    fid = br.foot_fid
    for _ in range(iters):
        pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
        pin.computeJointJacobians(m, d, q)
        err = np.zeros(12); J = np.zeros((12, m.nv))
        for i, L in enumerate(FEET):
            err[3*i:3*i+3] = targets[L] - d.oMf[fid[L]].translation
            J[3*i:3*i+3] = pin.getFrameJacobian(m, d, fid[L], pin.LOCAL_WORLD_ALIGNED)[:3]
        J[:, :6] = 0.0
        if np.linalg.norm(err) < tol:
            break
        q = pin.integrate(m, q, 0.5 * np.linalg.lstsq(J, err, rcond=None)[0])
    return q


def ik_feet(br, base_z, foot_targets, q_init=None, tol=1e-5, iters=120):
    """IK leg joints so each foot reaches its target (dict L->xyz); base fixed at base_z."""
    m, d = br.model, br.data
    q = pin.neutral(m) if q_init is None else q_init.copy()
    q[2] = base_z
    for _ in range(iters):
        pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
        pin.computeJointJacobians(m, d, q)
        err = np.zeros(12); J = np.zeros((12, m.nv))
        for i, L in enumerate(FEET):
            err[3*i:3*i+3] = foot_targets[L] - d.oMf[br.foot_fid[L]].translation
            J[3*i:3*i+3] = pin.getFrameJacobian(m, d, br.foot_fid[L], pin.LOCAL_WORLD_ALIGNED)[:3]
        J[:, :6] = 0.0
        if np.linalg.norm(err) < tol:
            break
        q = pin.integrate(m, q, 0.6 * np.linalg.lstsq(J, err, rcond=None)[0])
    return q


class TrotGaitOCP:
    """Builds fixed-schedule trot shooting problems for receding-horizon MPC."""

    def __init__(self, br: MjPinBridge, dt=2.5e-2, base_z=0.42, foot_r=0.018):
        self.br = br
        self.m = br.model
        self.dt = dt
        self.foot_r = foot_r                 # contact sphere radius (stance frame height)
        self.state = crocoddyl.StateMultibody(self.m)
        self.actu = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actu.nu
        self.fid = br.foot_fid
        self.mu = 0.7
        # per-joint peak torque bounds (Pinocchio actuation order: FL,FR,HL,HR legs;
        # each leg = hip,thigh,calf,foot). Matches actuatorfrcrange / motor peaks.
        self.tau_lim = np.tile([84.0, 84.0, 126.0, 100.8], 4)   # 16
        self.q_stand = standing_ik(br, base_z, foot_z=foot_r)
        self.x_stand = np.concatenate([self.q_stand, np.zeros(self.m.nv)])
        # nominal foot placements at stance, stored as offsets from the base xy so
        # swing targets can be re-anchored to the CURRENT base each MPC solve.
        self.foot0 = br.foot_positions(self.q_stand)
        base_xy0 = self.q_stand[0:2]
        self.foot_off = {L: (self.foot0[L][0:2] - base_xy0) for L in FEET}
        # state regulation weights (length 2*nv). Base x,y untracked (free forward
        # motion) but base z IS tracked so vertical dynamics stay regulated; base
        # orientation + joints + all velocities tracked.
        self.wx = np.array([0., 0., 30.] + [50.]*3 + [1.]*self.nu
                           + [10.]*3 + [10.]*3 + [1.]*self.nu)

    # ---- per-node cost/dynamics ----
    def _make_node(self, stance, swing_targets, v_cmd, is_terminal=False):
        state, nu = self.state, self.nu
        contacts = crocoddyl.ContactModelMultiple(state, nu)
        costs = crocoddyl.CostModelSum(state, nu)

        for L in stance:
            ci = crocoddyl.ContactModel3D(state, self.fid[L], np.zeros(3),
                                          pin.LOCAL_WORLD_ALIGNED, nu, np.array([0., 50.]))
            contacts.addContact(L + "_c", ci)
            # friction cone on stance foot
            cone = crocoddyl.FrictionCone(np.eye(3), self.mu, 4, False)
            act = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub))
            res = crocoddyl.ResidualModelContactFrictionCone(state, self.fid[L], cone, nu)
            costs.addCost(L + "_cone", crocoddyl.CostModelResidual(state, act, res), 5.0)

        # swing-foot position tracking (lift over an arc target)
        for L, ptgt in swing_targets.items():
            res = crocoddyl.ResidualModelFrameTranslation(state, self.fid[L], ptgt, nu)
            costs.addCost(L + "_track", crocoddyl.CostModelResidual(state, res), 1e5)

        # base velocity command via state reg toward a moving target
        x_ref = self.x_stand.copy()
        x_ref[self.m.nq + 0] = v_cmd[0]   # vx (local base linear)
        x_ref[self.m.nq + 1] = v_cmd[1]   # vy
        x_ref[self.m.nq + 5] = v_cmd[2]   # wz
        sref = crocoddyl.ResidualModelState(state, x_ref, nu)
        act = crocoddyl.ActivationModelWeightedQuad(self.wx**2)
        costs.addCost("xreg", crocoddyl.CostModelResidual(state, act, sref),
                      0.2 if not is_terminal else 10.0)
        if not is_terminal:
            costs.addCost("ureg", crocoddyl.CostModelResidual(
                state, crocoddyl.ResidualModelControl(state, nu)), 1e-3)

        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actu, contacts, costs, 0., True)
        iam = crocoddyl.IntegratedActionModelEuler(dam, 0. if is_terminal else self.dt)
        if not is_terminal:
            iam.u_lb = -self.tau_lim     # Box-FDDP respects these torque bounds
            iam.u_ub = self.tau_lim
        return iam

    # ---- trot schedule ----
    def _phase(self, k, T_half):
        """Return (stance feet, swing feet) for global node index k. Trot = diagonal pairs."""
        block = (k // T_half) % 2
        if block == 0:
            return ['FL', 'HR'], ['FR', 'HL']    # FR,HL swing
        else:
            return ['FR', 'HL'], ['FL', 'HR']    # FL,HR swing

    def _swing_target(self, L, phase_frac, step_len, step_h, base_xy):
        """Arc target for a swinging foot, anchored to the CURRENT base xy (Raibert-like).

        Foot lands a half-step ahead of its nominal hip offset so it tracks forward
        with the body instead of being pulled back to a fixed world point.
        """
        off = self.foot_off[L]
        px = base_xy[0] + off[0] + step_len * (phase_frac - 0.5) + 0.5 * step_len
        py = base_xy[1] + off[1]
        pz = self.foot_r + step_h * np.sin(np.pi * phase_frac)
        return np.array([px, py, pz])

    def create_problem(self, x0, v_cmd=(0.4, 0., 0.), N=20, T_half=8, step_h=0.08, k0=0,
                       foot_adjust=None):
        """Build an N-node receding-horizon trot problem starting at x0.

        k0 anchors the gait schedule to the global gait clock so the contact
        sequence advances with real time across successive MPC solves.
        foot_adjust(xyz)->xyz : optional terrain-aware nudge of each swing landing
        target (e.g. shift off a gap onto a platform). This is the hybrid's
        perceptive foothold layer on top of the stable OCP gait.
        """
        v_cmd = np.asarray(v_cmd, float)
        step_len = float(v_cmd[0]) * (T_half * self.dt)   # distance covered per swing
        base_xy0 = np.asarray(x0[0:2], float)             # anchor swing targets to current base
        models = []
        for k in range(N):
            g = k0 + k
            stance, swing = self._phase(g, T_half)
            phase_frac = ((g % T_half) + 0.5) / T_half
            # base advances at commanded vx over the horizon
            base_xy = base_xy0 + np.array([v_cmd[0] * k * self.dt, 0.0])
            swing_targets = {L: self._swing_target(L, phase_frac, step_len, step_h, base_xy)
                             for L in swing}
            if foot_adjust is not None:                   # terrain-aware foothold nudge
                swing_targets = {L: foot_adjust(L, t, phase_frac) for L, t in swing_targets.items()}
            models.append(self._make_node(stance, swing_targets, v_cmd))
        # terminal
        stance, swing = self._phase(k0 + N, T_half)
        term = self._make_node(stance, {}, v_cmd, is_terminal=True)
        return crocoddyl.ShootingProblem(x0, models, term)


def _selftest():
    br = MjPinBridge()
    ocp = TrotGaitOCP(br)
    print(f"nu={ocp.nu}  q_stand base_z={ocp.q_stand[2]:.3f}")
    prob = ocp.create_problem(ocp.x_stand, v_cmd=(0.4, 0., 0.), N=20)
    solver = crocoddyl.SolverBoxFDDP(prob)
    xs = [ocp.x_stand] * (prob.T + 1)
    us = prob.quasiStatic([ocp.x_stand] * prob.T)
    ok = solver.solve(xs, us, 100)
    print(f"trot OCP solved={ok} iters={solver.iter} cost={solver.cost:.3f}")
    xN = solver.xs[-1]
    print(f"base x drift over horizon = {xN[0]-ocp.x_stand[0]:+.3f} m "
          f"(expect ~{0.4*prob.T*ocp.dt:.3f})")
    print(f"final base z = {xN[2]:.3f}  vx = {xN[br.model.nq]:.3f}")
    umax = max(np.abs(u).max() for u in solver.us)
    print(f"max |tau| = {umax:.1f} N·m")


if __name__ == "__main__":
    _selftest()
