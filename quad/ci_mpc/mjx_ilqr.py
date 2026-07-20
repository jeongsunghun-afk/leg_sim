"""MJX-based contact-implicit iLQR/DDP (§15 C-2: MuJoCo-dialect of the HOUND recipe).

The gradient-CI insight the hybrid study forced: robust crossing needs contact-TIMING
adaptation (gradient flows through contact) AND stabilizing feedback (Riccati K) at once.
Reverse-mode autodiff through mjx.step is blocked (dynamic while_loop in the constraint
solver), but FORWARD-mode Jacobians (fx, fu) work - and those are exactly what iLQR/DDP
needs. So we linearise the MJX dynamics with jax.jacfwd and run a standard LQR backward
pass, giving feed-forward controls + feedback gains K.

v1 pragmatics: the base orientation quaternion is treated in the raw 47-dim state
(qpos24+qvel23); the forward rollout (mjx.step) re-normalises it. This Euclidean-quat
approximation is fine for near-upright locomotion / short horizons; a proper tangent-space
version is a later refinement.

This module currently provides the dynamics-linearisation core; the backward/forward
iLQR passes are built on top of it.
"""
import os
import numpy as np
import mujoco
import jax
import jax.numpy as jnp
from mujoco import mjx
from model_bridge import MJCF, apply_gearbox, set_foot_sphere, strip_mesh_collision

FOOT_R = float(os.environ.get("FOOT_R", "0.024"))
DT = float(os.environ.get("DT", "0.02"))          # iLQR control-node dt
DT_SIM = float(os.environ.get("DT_SIM", "0.002"))  # stable MuJoCo sim step (contact needs small dt)


def build_mjx(mjcf=MJCF, foot_r=FOOT_R):
    mm = mujoco.MjModel.from_xml_path(mjcf)
    apply_gearbox(mm)
    set_foot_sphere(mm, foot_r)
    strip_mesh_collision(mm)             # sphere feet + terrain only (fast MJX)
    mm.opt.timestep = DT_SIM             # ★ small sim step; contact is unstable at dt=0.02
    return mm, mjx.put_model(mm)


class MjxDynamics:
    """One-step MJX dynamics x_{t+1}=f(x_t,u_t) with forward-mode Jacobians fx, fu.

    State x = [qpos(nq), qvel(nv)] (raw, quat in place). Control u = ctrl(nu).
    """

    def __init__(self, mm, mx):
        self.mm, self.mx = mm, mx
        self.nq, self.nv, self.nu = mm.nq, mm.nv, mm.nu
        self.nx = self.nq + self.nv
        self._d0 = mjx.make_data(mx)
        self.sub = max(1, round(DT / mm.opt.timestep))   # sim sub-steps per control node

        def f(x, u):
            # one control node = `sub` small sim steps (holding u), so the node dt is DT
            # while the contact sim runs at the stable DT_SIM. Forward-differentiable
            # (lax.scan) - which is all jacfwd needs.
            d = self._d0.replace(qpos=x[:self.nq], qvel=x[self.nq:], ctrl=u)
            def sstep(d, _):
                return mjx.step(mx, d.replace(ctrl=u)), None
            d, _ = jax.lax.scan(sstep, d, None, length=self.sub)
            return jnp.concatenate([d.qpos, d.qvel])
        self.f = f
        # forward-mode Jacobians (reverse-mode is blocked by the solver's while_loop)
        self._fx = jax.jit(jax.jacfwd(f, argnums=0))
        self._fu = jax.jit(jax.jacfwd(f, argnums=1))
        self._f = jax.jit(f)
        # batched over a horizon
        self._fx_b = jax.jit(jax.vmap(jax.jacfwd(f, argnums=0)))
        self._fu_b = jax.jit(jax.vmap(jax.jacfwd(f, argnums=1)))
        self._f_b = jax.jit(jax.vmap(f))

    def step(self, x, u):
        return np.asarray(self._f(jnp.asarray(x), jnp.asarray(u)))

    def linearize(self, x, u):
        """Return (fx, fu) at (x,u): fx=[nx,nx], fu=[nx,nu]."""
        return np.asarray(self._fx(jnp.asarray(x), jnp.asarray(u))), \
               np.asarray(self._fu(jnp.asarray(x), jnp.asarray(u)))

    def linearize_traj(self, xs, us):
        """Batched linearisation over a horizon. xs:[N,nx], us:[N,nu] -> fx:[N,nx,nx], fu:[N,nx,nu]."""
        xs = jnp.asarray(xs); us = jnp.asarray(us)
        return np.asarray(self._fx_b(xs, us)), np.asarray(self._fu_b(xs, us))

    def rollout(self, x0, us):
        """Sequential rollout of a control sequence. us:[N,nu] -> xs:[N+1,nx]."""
        x = jnp.asarray(x0); xs = [np.asarray(x)]
        for u in us:
            x = self._f(x, jnp.asarray(u)); xs.append(np.asarray(x))
        return np.array(xs)


class QuadCost:
    """Quadratic tracking cost l = 0.5 dxᵀQ dx + 0.5 uᵀR u (terminal uses Qf).

    v1 treats the base quaternion Euclidean (penalise x,y,z components = tilt).
    """
    def __init__(self, nx, nu, nq, x_ref, q_diag, r_diag, qf_scale=20.0):
        self.x_ref = np.asarray(x_ref, float)
        self.Q = np.diag(np.asarray(q_diag, float))
        self.R = np.diag(np.asarray(r_diag, float))
        self.Qf = self.Q * qf_scale
        self.nx, self.nu = nx, nu

    def stage(self, x, u):
        dx = x - self.x_ref
        return 0.5 * dx @ self.Q @ dx + 0.5 * u @ self.R @ u

    def terminal(self, x):
        dx = x - self.x_ref
        return 0.5 * dx @ self.Qf @ dx

    def total(self, xs, us):
        return sum(self.stage(xs[t], us[t]) for t in range(len(us))) + self.terminal(xs[-1])


def ilqr(dyn, x0, us, cost, iters=12, reg0=1e-3, verbose=True):
    """Contact-implicit iLQR via MJX forward-mode Jacobians. Returns xs, us, K."""
    N, nx, nu = len(us), dyn.nx, dyn.nu
    xs = dyn.rollout(x0, us)
    J = cost.total(xs, us)
    reg = reg0
    for it in range(iters):
        FX, FU = dyn.linearize_traj(xs[:-1], us)          # [N,nx,nx], [N,nx,nu]
        # ---- backward pass ----
        Vx = cost.Qf @ (xs[-1] - cost.x_ref)
        Vxx = cost.Qf.copy()
        K = np.zeros((N, nu, nx)); kff = np.zeros((N, nu))
        ok = True
        for t in range(N - 1, -1, -1):
            dx = xs[t] - cost.x_ref
            lx = cost.Q @ dx; lu = cost.R @ us[t]
            fx, fu = FX[t], FU[t]
            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = cost.Q + fx.T @ Vxx @ fx
            Quu = cost.R + fu.T @ Vxx @ fu + reg * np.eye(nu)
            Qux = fu.T @ Vxx @ fx
            try:
                L = np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError:
                ok = False; break
            Quu_inv = np.linalg.inv(Quu)
            K[t] = -Quu_inv @ Qux; kff[t] = -Quu_inv @ Qu
            Vx = Qx + K[t].T @ Quu @ kff[t] + K[t].T @ Qu + Qux.T @ kff[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            Vxx = 0.5 * (Vxx + Vxx.T)
        if not ok:
            reg *= 4.0
            if verbose: print(f"  it{it}: backward non-PD, reg->{reg:.1e}")
            continue
        # ---- forward pass w/ line search ----
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
            xn = np.zeros((N + 1, nx)); xn[0] = x0; un = np.zeros((N, nu))
            for t in range(N):
                un[t] = us[t] + alpha * kff[t] + K[t] @ (xn[t] - xs[t])
                xn[t + 1] = dyn.step(xn[t], un[t])
            Jn = cost.total(xn, un)
            if Jn < J:
                xs, us, J = xn, un, Jn; improved = True
                reg = max(reg / 2, 1e-6); break
        if verbose:
            print(f"  it{it}: J={J:.3f} alpha={'-' if not improved else alpha} reg={reg:.1e}", flush=True)
        if not improved:
            reg *= 4.0
            if reg > 1e3:
                break
    return xs, us, K


def _stand_test():
    """Regulate to standing: iLQR must find controls+feedback that hold the robot up."""
    from ocp_fixed import standing_ik
    from model_bridge import MjPinBridge
    mm, mx = build_mjx()
    dyn = MjxDynamics(mm, mx)
    from model_bridge import MJ_WAIST_JIDX
    br = MjPinBridge()
    q_stand = standing_ik(br, 0.42, foot_z=FOOT_R)
    q_mj = br.pin_to_mj_qpos(q_stand)
    # settle so the feet actually penetrate/contact (raw IK pose has feet exactly at
    # ground -> ncon=0 -> no contact force -> free fall). Use the settled in-contact
    # state as both x0 and the regulation target.
    md0 = mujoco.MjData(mm); md0.qpos[:] = q_mj; mujoco.mj_forward(mm, md0)
    qh = q_mj[7:].copy()
    for _ in range(400):                 # gentle settle to a real in-contact rest state
        u = md0.qfrc_bias[6:] + 150.0 * (qh - md0.qpos[7:]) - 12.0 * md0.qvel[6:]
        u[MJ_WAIST_JIDX] = 200 * (0 - md0.qpos[7 + MJ_WAIST_JIDX]) - 5 * md0.qvel[6 + MJ_WAIST_JIDX]
        md0.ctrl[:] = np.clip(u, -200, 200); mujoco.mj_step(mm, md0)
    q_mj = md0.qpos.copy()
    u_settle = np.clip(md0.qfrc_bias[6:] + 150.0 * (qh - md0.qpos[7:]) - 12.0 * md0.qvel[6:], -200, 200)
    print(f"settled: base z={q_mj[2]:.3f} ncon={md0.ncon} jointvel={np.linalg.norm(md0.qvel[6:]):.3f}")
    x_ref = np.concatenate([q_mj, np.zeros(dyn.nv)])
    # cost weights: state [pos3, quat4, joints17, vlin3, vang3, vjoint17]
    qd = ([0, 0, 60] + [0, 120, 120, 120] + [3.0]*17
          + [2, 2, 2] + [2, 2, 2] + [0.2]*17)
    rd = [1e-3]*dyn.nu
    cost = QuadCost(dyn.nx, dyn.nu, dyn.nq, x_ref, qd, rd, qf_scale=20.0)
    N = 15
    x0 = x_ref.copy()                    # start at standing
    # warm-start controls with the gravity-comp (bias) torque at the standing config -
    # a far better initial guess than zero (which free-falls), so iLQR converges fast.
    us = np.tile(u_settle, (N, 1))       # warm-start = the settle's stance-holding control
    print(f"iLQR standing regulation (settle warm-start |u|={np.linalg.norm(u_settle):.1f}):")
    xs, us, K = ilqr(dyn, x0, us, cost, iters=20)
    print(f"final base z = {xs[-1][2]:.3f} (target {x_ref[2]:.3f}), |K0|={np.linalg.norm(K[0]):.1f}")
    print("base z along horizon:", np.round([x[2] for x in xs], 3))

    # ---- closed-loop feedback test: apply u0 + K0·(x_meas - xs0) in MuJoCo ----
    # validates that the iLQR feedback gain actually STABILISES standing over time.
    md = mujoco.MjData(mm); md.qpos[:] = q_mj; md.qvel[:] = 0.0; mujoco.mj_forward(mm, md)
    u0 = us[0].copy(); K0 = K[0].copy(); xs0 = xs[0].copy()
    print("closed-loop feedback (u0 + K0·dx):")
    for step in range(1000):
        x_meas = np.concatenate([md.qpos, md.qvel])
        u = u0 + K0 @ (x_meas - xs0)
        md.ctrl[:] = np.clip(u, -200, 200)
        mujoco.mj_step(mm, md)
        if step % 200 == 0 or step == 999:
            print(f"  step{step:4d} base z={md.qpos[2]:.3f} tilt_qxy={np.linalg.norm(md.qpos[4:6]):.3f} "
                  f"jv={np.linalg.norm(md.qvel[6:]):.2f}")
        if md.qpos[2] < 0.18:
            print(f"  FELL at step {step}"); break


def _settle_state(mm, q_mj, MJ_WAIST_JIDX, steps=400, kp=150.0, kd=12.0):
    """Settle to an in-contact rest state; return (q_settled, u_hold)."""
    md = mujoco.MjData(mm); md.qpos[:] = q_mj; mujoco.mj_forward(mm, md)
    qh = q_mj[7:].copy(); tau = np.array([84, 84, 126, 100.8]*2 + [84.] + [84, 84, 126, 100.8]*2)
    for _ in range(steps):
        u = md.qfrc_bias[6:] + kp*(qh - md.qpos[7:]) - kd*md.qvel[6:]
        u[MJ_WAIST_JIDX] = 200*(0 - md.qpos[7+MJ_WAIST_JIDX]) - 5*md.qvel[6+MJ_WAIST_JIDX]
        md.ctrl[:] = np.clip(u, -tau, tau); mujoco.mj_step(mm, md)
    u_hold = np.clip(md.qfrc_bias[6:] + kp*(qh - md.qpos[7:]) - kd*md.qvel[6:], -tau, tau)
    return md.qpos.copy(), u_hold


def _mpc_run():
    """Closed-loop iLQR-MPC forward-walking attempt (HOUND-style regulating cost)."""
    from ocp_fixed import standing_ik
    from model_bridge import MjPinBridge, MJ_WAIST_JIDX
    VX = float(os.environ.get("VX", "0.2"))
    NCTRL = int(os.environ.get("NCTRL", "40"))     # closed-loop control steps
    Nh = int(os.environ.get("NH", "12"))           # iLQR horizon
    ITERS = int(os.environ.get("ILQR_ITERS", "3"))
    mm, mx = build_mjx(); dyn = MjxDynamics(mm, mx)
    br = MjPinBridge()
    q0 = br.pin_to_mj_qpos(standing_ik(br, 0.42, foot_z=FOOT_R))
    q_mj, u_hold = _settle_state(mm, q0, MJ_WAIST_JIDX)
    nq, nv, nx, nu = dyn.nq, dyn.nv, dyn.nx, dyn.nu
    # regulating cost: base x FREE, track height/upright + forward velocity vx
    x_ref = np.concatenate([q_mj, np.zeros(nv)]); x_ref[nq + 0] = VX   # world vx target
    qd = ([0, 8, 50] + [0, 100, 100, 100] + [1.0]*17          # pos: x free, y, z, quat, joints
          + [25, 10, 5] + [5, 5, 8] + [0.05]*17)              # vel: vx-track, vy, vz, ang, jvel
    cost = QuadCost(nx, nu, nq, x_ref, qd, [2e-3]*nu, qf_scale=8.0)
    md = mujoco.MjData(mm); md.qpos[:] = q_mj; md.qvel[:] = 0.0; mujoco.mj_forward(mm, md)
    us = np.tile(u_hold, (Nh, 1))
    print(f"iLQR-MPC walk: VX={VX} Nh={Nh} iters={ITERS}")
    falls = 0
    for c in range(NCTRL):
        x_meas = np.concatenate([md.qpos, md.qvel])
        xs, us, K = ilqr(dyn, x_meas, us, cost, iters=ITERS, verbose=False)
        u0, K0, xs0 = us[0].copy(), K[0].copy(), xs[0].copy()
        for _ in range(dyn.sub):                    # apply node control for sub sim steps
            xm = np.concatenate([md.qpos, md.qvel])
            md.ctrl[:] = np.clip(u0 + K0 @ (xm - xs0), -200, 200)
            mujoco.mj_step(mm, md)
        us = np.vstack([us[1:], us[-1]])            # shift warm-start
        if c % 5 == 0:
            print(f"  c{c:3d} x={md.qpos[0]:+.3f} z={md.qpos[2]:.3f} vx={md.qvel[0]:+.3f}", flush=True)
        if md.qpos[2] < 0.20:
            falls = 1; print(f"  FELL at c{c}"); break
    print(f"RESULT VX={VX} ctrl={c+1} falls={falls} x={md.qpos[0]:+.3f} z={md.qpos[2]:.3f} vx={md.qvel[0]:+.3f}")


def _selftest():
    mm, mx = build_mjx()
    dyn = MjxDynamics(mm, mx)
    x0 = np.concatenate([mm.qpos0, np.zeros(dyn.nv)])
    u0 = np.zeros(dyn.nu)
    print(f"nx={dyn.nx} nu={dyn.nu}")
    x1 = dyn.step(x0, u0)
    print(f"step OK, base z {x0[2]:.3f} -> {x1[2]:.3f}")
    fx, fu = dyn.linearize(x0, u0)
    print(f"linearize OK  fx{fx.shape} |fx|={np.linalg.norm(fx):.2f}  fu{fu.shape} |fu|={np.linalg.norm(fu):.3f}")
    # batched over a short horizon
    N = 8
    xs = np.tile(x0, (N, 1)); us = np.zeros((N, dyn.nu))
    FX, FU = dyn.linearize_traj(xs, us)
    print(f"batched linearize_traj OK  FX{FX.shape}  FU{FU.shape}")
    print("PASS")


if __name__ == "__main__":
    _selftest()
