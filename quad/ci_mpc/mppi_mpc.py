"""Sampling-based Contact-Implicit MPC (§15 path B) for the 17-DOF 02_Leg.

Receding-horizon MPPI / predictive sampling. Rollouts run in MuJoCo itself (the
deployment simulator) so contacts are handled exactly and there is ZERO sim2sim gap
- no complementarity, relaxation, QCQP or analytical gradient. This promotes the
offline getup_mppi.py to an online locomotion MPC.

Parameterization: a short horizon of joint-position setpoints q_ref[H, nu], PD +
gravity-comp tracked. Physics decides the actual contact sequence/timing (this is
the "contact-implicit" property - timing emerges from the rollout, not a schedule).

Each control step: sample Nsamp smoothed perturbations around the (warm-started)
nominal, roll each out, exp(-cost/lambda) weighted update, apply the first setpoint,
shift. CPU validation first; MJX GPU parallel rollout is the real-time path (§15).

Usage: MJPY mppi_mpc.py  [VX=0.3] [STEPS=1500] [H=15] [NSAMP=24] [VIEW=1]
"""
import os, sys, time
import numpy as np
import mujoco
from scipy.ndimage import gaussian_filter1d
from model_bridge import MJCF, MJ_WAIST_JIDX, apply_gearbox, set_foot_sphere

VX = float(os.environ.get("VX", "0.3"))
STEPS = int(os.environ.get("STEPS", "1500"))
H = int(os.environ.get("H", "15"))              # horizon (control nodes)
NSAMP = int(os.environ.get("NSAMP", "24"))
ITERS = int(os.environ.get("ITERS", "2"))       # MPPI iters per replan
SIGMA = float(os.environ.get("SIGMA", "0.10"))  # joint-target sample std [rad]
LAM = float(os.environ.get("LAM", "40.0"))      # MPPI temperature
DT = float(os.environ.get("DT", "0.02"))        # control node dt
FOOT_R = float(os.environ.get("FOOT_R", "0.024"))
VIEW = os.environ.get("VIEW", "0") == "1"
KP = float(os.environ.get("KP", "120"))
KD = float(os.environ.get("KD", "4"))

# physical peak torque per MJCF joint (HL,HR,waist,FL,FR): hip84 thigh84 calf126 foot100.8
TAU = np.array([84, 84, 126, 100.8] * 2 + [84.0] + [84, 84, 126, 100.8] * 2)
LEGS = ['HL', 'HR', 'FL', 'FR']


STEP_H = float(os.environ.get("STEP_H", "0.07"))    # gait-prior swing lift [m]
NCYC = int(os.environ.get("NCYC", "20"))            # control nodes per gait cycle


class SamplingMPC:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(MJCF)
        apply_gearbox(self.m)
        set_foot_sphere(self.m, FOOT_R)
        self.nu = self.m.nu
        self.sub = max(1, round(DT / self.m.opt.timestep))
        self.fg = {L: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, L + '_sphere')
                   for L in LEGS}
        self.wid_q = 7 + MJ_WAIST_JIDX          # waist qpos index
        self.wid_v = 6 + MJ_WAIST_JIDX
        self._d = mujoco.MjData(self.m)         # scratch reused across rollouts
        self._build_gait_prior()

    def _build_gait_prior(self):
        """Precompute a stepping-in-place trot joint reference over one gait cycle.

        This only SEEDS the sampler - the actual contact timing emerges from the
        MuJoCo rollout, so it stays contact-implicit. MPPI shapes it (and the
        forward-velocity cost) into forward locomotion.
        """
        from ocp_fixed import standing_ik, ik_feet
        from model_bridge import MjPinBridge
        br = MjPinBridge()
        q_stand = standing_ik(br, 0.40, foot_z=FOOT_R)
        foot0 = br.foot_positions(q_stand)
        self.q_stand_pin = q_stand
        self.qstand_j = br.pin_to_mj_qpos(q_stand)[7:].copy()
        table = np.zeros((NCYC, self.nu))
        for c in range(NCYC):
            frac = c / NCYC
            # diagonal trot: first half FR,HL swing; second half FL,HR swing
            swing = ['FR', 'HL'] if frac < 0.5 else ['FL', 'HR']
            sfrac = (frac % 0.5) / 0.5
            tgt = {L: foot0[L].copy() for L in ['FL', 'FR', 'HL', 'HR']}
            for L in swing:
                tgt[L][2] = FOOT_R + STEP_H * np.sin(np.pi * sfrac)
            q = ik_feet(br, 0.40, tgt, q_init=q_stand)
            table[c] = br.pin_to_mj_qpos(q)[7:]
        self.prior = table

    def prior_node(self, gk):
        return self.prior[gk % NCYC]

    def prior_horizon(self, phase):
        return np.array([self.prior_node(phase + k) for k in range(H)])

    # ---- one PD-tracked rollout from state (qpos,qvel); returns cost ----
    def rollout(self, qpos0, qvel0, qref):
        d = self._d
        d.qpos[:] = qpos0; d.qvel[:] = qvel0
        mujoco.mj_forward(self.m, d)
        nu = self.nu
        cost = 0.0
        for k in range(H):
            tgt = qref[k]
            for _ in range(self.sub):
                tau = d.qfrc_bias[6:6 + nu] + KP * (tgt - d.qpos[7:7 + nu]) - KD * d.qvel[6:6 + nu]
                tau[MJ_WAIST_JIDX] = 200 * (0.0 - d.qpos[self.wid_q]) - 5 * d.qvel[self.wid_v]
                d.ctrl[:] = np.clip(tau, -TAU, TAU)
                mujoco.mj_step(self.m, d)
            cost += self.step_cost(d)
        return cost

    def step_cost(self, d):
        z = d.qpos[2]
        w, x, y, zq = d.qpos[3:7]
        tilt = 1.0 - 2.0 * (x * x + y * y)            # cos(tilt), 1=upright
        vx = d.qvel[0]; vy = d.qvel[1]
        c = 0.0
        c += 12.0 * (vx - VX) ** 2                    # track forward speed
        c += 8.0 * vy ** 2                            # no sideways drift
        c += 30.0 * (z - 0.40) ** 2                   # hold height
        c += 20.0 * (1.0 - tilt)                      # stay upright
        c += 2.0 * d.qvel[5] ** 2                     # no yaw spin
        if z < 0.24:
            c += 500.0                                # near-fall penalty
        return c

    def replan(self, qpos0, qvel0, nominal):
        """MPPI: sample around nominal, roll out, weighted update. Returns new nominal."""
        for _ in range(ITERS):
            eps = np.random.randn(NSAMP, H, self.nu) * SIGMA
            for s in range(NSAMP):
                eps[s] = gaussian_filter1d(eps[s], sigma=3, axis=0)   # smooth in time
            costs = np.array([self.rollout(qpos0, qvel0, nominal + eps[s])
                              for s in range(NSAMP)])
            beta = costs.min()
            w = np.exp(-(costs - beta) / LAM); w /= w.sum()
            nominal = nominal + np.einsum('s,shu->hu', w, eps)
            nominal = gaussian_filter1d(nominal, sigma=1, axis=0)
        return nominal, beta


def main():
    mpc = SamplingMPC()
    m = mpc.m
    d = mujoco.MjData(m)
    from model_bridge import MjPinBridge
    br = MjPinBridge()
    q_mj = br.pin_to_mj_qpos(mpc.q_stand_pin)
    qstand_j = mpc.qstand_j

    d.qpos[:] = q_mj; d.qvel[:] = 0.0; mujoco.mj_forward(m, d)
    for _ in range(300):                              # settle
        tau = d.qfrc_bias[6:] + KP * (qstand_j - d.qpos[7:]) - KD * d.qvel[6:]
        tau[MJ_WAIST_JIDX] = 200 * (0.0 - d.qpos[mpc.wid_q]) - 5 * d.qvel[mpc.wid_v]
        d.ctrl[:] = np.clip(tau, -TAU, TAU); mujoco.mj_step(m, d)

    phase = 0
    nominal = mpc.prior_horizon(phase)               # seed with gait prior
    viewer = None
    if VIEW:
        import mujoco.viewer as _mjv
        viewer = _mjv.launch_passive(m, d)

    falls = 0; t_plan = 0.0; n_plan = 0
    for step in range(STEPS):
        t0 = time.perf_counter()
        nominal, best = mpc.replan(d.qpos.copy(), d.qvel.copy(), nominal)
        t_plan += time.perf_counter() - t0; n_plan += 1
        tgt = nominal[0]
        # apply first setpoint for one control node (sub sim steps)
        for _ in range(mpc.sub):
            tau = d.qfrc_bias[6:] + KP * (tgt - d.qpos[7:]) - KD * d.qvel[6:]
            tau[MJ_WAIST_JIDX] = 200 * (0.0 - d.qpos[mpc.wid_q]) - 5 * d.qvel[mpc.wid_v]
            d.ctrl[:] = np.clip(tau, -TAU, TAU); mujoco.mj_step(m, d)
        # shift horizon; append fresh gait-prior tail so the horizon keeps gait shape
        phase += 1
        nominal = np.vstack([nominal[1:], mpc.prior_node(phase + H - 1)])
        if d.qpos[2] < 0.20:
            falls = 1; break
        if viewer is not None and step % 2 == 0:
            viewer.sync()
    print(f"VX={VX} steps={step+1} falls={falls} x={d.qpos[0]:+.3f} z={d.qpos[2]:.3f} "
          f"vx={d.qvel[0]:+.3f} plan_ms={1e3*t_plan/max(1,n_plan):.1f} "
          f"(H={H} Nsamp={NSAMP} iters={ITERS})")
    if viewer is not None:
        viewer.close()
    return falls


if __name__ == "__main__":
    sys.exit(main())
