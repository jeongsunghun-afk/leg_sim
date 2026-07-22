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

FOOT_R = float(os.environ.get("FOOT_R", "0.025"))
DT = float(os.environ.get("DT", "0.02"))          # iLQR control-node dt
DT_SIM = float(os.environ.get("DT_SIM", "0.002"))  # stable MuJoCo sim step (contact needs small dt)


MJCF_PATH = os.environ.get("MJCF_PATH", MJCF)          # override to a terrain scene
DISABLE_FLOOR = os.environ.get("DISABLE_FLOOR", "0") == "1"
BRAKE = float(os.environ.get("BRAKE", "1.0"))          # base-x 참조 제동: 1=VX(제동, lunge억제) · 0=vx_pred(4차 크로싱 거동)


def build_mjx(mjcf=None, foot_r=FOOT_R):
    mm = mujoco.MjModel.from_xml_path(mjcf or MJCF_PATH)
    apply_gearbox(mm)
    set_foot_sphere(mm, foot_r)
    strip_mesh_collision(mm)             # sphere feet + terrain only (fast MJX)
    if DISABLE_FLOOR:                    # gaps between platforms become true voids
        fg = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if fg >= 0:
            mm.geom_contype[fg] = 0; mm.geom_conaffinity[fg] = 0
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

    x_ref may be a single target [nx] OR a per-node reference trajectory [N+1, nx]
    (for gait tracking - a trot joint pattern makes the legs step). v1 treats the base
    quaternion Euclidean (penalise x,y,z components = tilt).
    """
    def __init__(self, nx, nu, nq, x_ref, q_diag, r_diag, qf_scale=20.0):
        self.x_ref = np.asarray(x_ref, float)      # [nx] or [N+1, nx]
        self.per_node = self.x_ref.ndim == 2
        self.Q = np.diag(np.asarray(q_diag, float))
        self.R = np.diag(np.asarray(r_diag, float))
        self.Qf = self.Q * qf_scale
        self.nx, self.nu = nx, nu

    def ref(self, t):
        return self.x_ref[t] if self.per_node else self.x_ref

    def stage(self, x, u, t):
        dx = x - self.ref(t)
        return 0.5 * dx @ self.Q @ dx + 0.5 * u @ self.R @ u

    def terminal(self, x, t):
        dx = x - self.ref(t)
        return 0.5 * dx @ self.Qf @ dx

    def total(self, xs, us):
        return (sum(self.stage(xs[t], us[t], t) for t in range(len(us)))
                + self.terminal(xs[-1], len(us)))


def ilqr(dyn, x0, us, cost, iters=12, reg0=1e-3, verbose=True):
    """Contact-implicit iLQR via MJX forward-mode Jacobians. Returns xs, us, K."""
    N, nx, nu = len(us), dyn.nx, dyn.nu
    xs = dyn.rollout(x0, us)
    J = cost.total(xs, us)
    reg = reg0
    for it in range(iters):
        FX, FU = dyn.linearize_traj(xs[:-1], us)          # [N,nx,nx], [N,nx,nu]
        # ---- backward pass ----
        Vx = cost.Qf @ (xs[-1] - cost.ref(N))
        Vxx = cost.Qf.copy()
        K = np.zeros((N, nu, nx)); kff = np.zeros((N, nu))
        ok = True
        for t in range(N - 1, -1, -1):
            dx = xs[t] - cost.ref(t)
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


def gait_joint_table(br, base_z, foot_r, ncyc, step_h, vx, dt):
    """Trot joint reference over one gait cycle (diagonal pairs lift + sweep forward).

    A gait REFERENCE the iLQR tracks in joint space -> the legs step; the contact
    gradient + feedback then stabilise and adapt the timing. Reuses ik_feet.
    """
    from ocp_fixed import standing_ik, ik_feet, FEET
    q_stand = standing_ik(br, base_z, foot_z=foot_r)
    foot0 = br.foot_positions(q_stand)
    T_stance = 0.5 * ncyc * dt
    half = 0.5 * vx * T_stance
    pair = {'FR': (0, 0.5), 'HL': (0, 0.5), 'FL': (0.5, 1.0), 'HR': (0.5, 1.0)}
    nmj = len(br.pin_to_mj_qpos(q_stand)) - 7      # MJCF joint count (17, incl. waist)
    table = np.zeros((ncyc, nmj))
    for c in range(ncyc):
        p = c / ncyc
        tgt = {L: foot0[L].copy() for L in FEET}
        for L in FEET:
            s0, s1 = pair[L]
            if s0 <= p < s1:
                sp = (p - s0) / (s1 - s0)
                tgt[L][0] += half * (2 * sp - 1); tgt[L][2] = foot_r + step_h * np.sin(np.pi * sp)
            else:
                st = ((p - s1) % 1.0) / (1.0 - (s1 - s0)); tgt[L][0] += half * (1 - 2 * st)
        q = ik_feet(br, base_z, tgt, q_init=q_stand)
        table[c] = br.pin_to_mj_qpos(q)[7:]
    return table


def _mpc_run():
    """Closed-loop iLQR-MPC forward walk with a trot gait reference (joint-space)."""
    from ocp_fixed import standing_ik
    from model_bridge import MjPinBridge, MJ_WAIST_JIDX
    VX = float(os.environ.get("VX", "0.2"))
    NCTRL = int(os.environ.get("NCTRL", "40"))
    Nh = int(os.environ.get("NH", "12"))
    ITERS = int(os.environ.get("ILQR_ITERS", "3"))
    NCYC = int(os.environ.get("NCYC", "20"))
    STEP_H = float(os.environ.get("STEP_H", "0.07"))
    GAIT = os.environ.get("GAIT", "trot")            # trot(2-foot) | crawl(3-foot static)
    GAP_AWARE = os.environ.get("GAP_AWARE", "0") == "1"   # high-plateau swing over a void
    TERRAIN = DISABLE_FLOOR                          # terrain-aware footholds on gap course
    mm, mx = build_mjx(); dyn = MjxDynamics(mm, mx)
    br = MjPinBridge()
    q0 = br.pin_to_mj_qpos(standing_ik(br, 0.42, foot_z=FOOT_R))
    q_mj, u_hold = _settle_state(mm, q0, MJ_WAIST_JIDX)
    nq, nv, nx, nu = dyn.nq, dyn.nv, dyn.nx, dyn.nu
    # --- terrain-aware trot gait reference (the last piece for gap crossing) ---
    # A fixed gait pattern (flat) makes the iLQR step but places swing feet blindly - a
    # foot landing in a gap falls into the void. Here the swing LANDING xy is checked
    # against the terrain (mj_ray, group 2) and nudged onto the nearest platform, then
    # IK'd to joints. Combined with the contact-implicit iLQR (no GRF-bounce) this is the
    # full solution: gait-shaping (step) + perceptive foothold (avoid gap) + CI (stabilise).
    # --- world-frame foothold scheduler (the fix over the body-relative table) ---
    # The body-relative table let the STANCE-foot reference drift forward with the base
    # (foot = base + offset), so near a gap the planted foot got dragged into the void.
    # Here, every control step, stance feet are pinned to their ACTUAL world position
    # (read from FK), so as the base advances the reference keeps them planted; only
    # swing feet move, to a base-anchored landing nudged off gaps (mj_ray, group 2).
    from ocp_fixed import ik_feet, FEET
    q_stand = standing_ik(br, 0.42, foot_z=FOOT_R)
    foot0 = br.foot_positions(q_stand)
    FOFF = {L: (foot0[L][:2] - q_stand[:2]) for L in FEET}   # body-frame hip offset
    # gait swing windows (phase fraction each foot is airborne). TROT = diagonal pairs
    # (2-foot support, dynamic). CRAWL/wave = one foot at a time (3-foot support, static)
    # — the CoM stays inside the support triangle, which is what a careful gap crossing
    # needs (trot tips forward once the CoM passes over the void).
    if GAIT == "crawl":
        pair = {'HL': (0.0, 0.25), 'FL': (0.25, 0.5), 'HR': (0.5, 0.75), 'FR': (0.75, 1.0)}
        duty = 0.25
    else:
        pair = {'FR': (0, 0.5), 'HL': (0, 0.5), 'FL': (0.5, 1.0), 'HR': (0.5, 1.0)}
        duty = 0.5
    T_stance = (1.0 - duty) * NCYC * DT; half = 0.5 * VX * T_stance
    _grp2 = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
    # base x weight 6 (was 0/free): tracks a controlled advancing line (build_ref sets
    # ref x = base_x0 + VX·t) so a forward lunge — CoM running ahead of the line — is
    # braked, while steady forward progress at VX is unpenalised. Position feedback on
    # top of the vx (velocity) damping = PD on CoM forward motion, catches the runaway.
    qd = ([6, 8, 120] + [0, 120, 120, 120] + [40.0]*17
          + [20, 10, 5] + [5, 5, 8] + [0.05]*17)
    rd = [2e-3]*nu

    _gid1 = np.zeros(1, dtype=np.int32)
    def _supported(x, y):                             # terrain (group 2) under (x,y)?
        mujoco.mj_ray(mm, md, np.array([x, y, 0.6]), np.array([0., 0., -1.]), _grp2, 1, -1, _gid1)
        return _gid1[0] >= 0

    def _nudge(x, y):                                 # shift x forward off a gap onto terrain
        if not TERRAIN or _supported(x, y):
            return x
        for d in (0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, -0.03, -0.06):
            if _supported(x + d, y):
                return x + d
        return x

    def build_ref(phase, base_x0, base_y0, fworld, vx_meas):
        # Anchor the horizon's base prediction to a capture-point velocity: blend the
        # command with the MEASURED forward speed. During a forward lunge (vx_meas >> VX)
        # this places the swing landing further ahead (Raibert-like) so the front foot
        # catches the body on the far platform instead of diving short into the gap.
        vx_pred = 0.5 * VX + 0.5 * float(np.clip(vx_meas, 0.0, 0.9))
        ref = np.zeros((Nh + 1, nx))
        q_prev = q_stand                             # warm-start IK chain: adjacent horizon
        for k in range(Nh + 1):                      # nodes differ little → few IK iters each
            g = phase + k; p = (g % NCYC) / NCYC
            base_xk = base_x0 + vx_pred * k * DT
            tgt = {}
            for L in FEET:
                s0, s1 = pair[L]
                if s0 <= p < s1:                      # swing: capture-point landing, terrain-aware
                    sp = (p - s0) / (s1 - s0)
                    cap = 0.5 * (vx_meas - VX) * T_stance   # Raibert capture offset (extra reach when lunging)
                    fy = base_y0 + FOFF[L][1]
                    nom_x = base_xk + FOFF[L][0] + half * (2 * sp - 1) + cap
                    crossing = GAP_AWARE and TERRAIN and not _supported(nom_x, fy)  # landing over a void
                    xw = _nudge(nom_x, fy)
                    if crossing:                      # gap-aware(opt): high plateau, descend past far edge
                        zc = 1.5 * STEP_H * min(1.0, np.sin(np.pi * sp) / np.sin(0.15 * np.pi))
                    else:
                        zc = STEP_H * np.sin(np.pi * sp)
                    tgt[L] = np.array([xw - base_xk, FOFF[L][1], FOOT_R + zc])
                else:                                 # stance: pinned to ACTUAL planted world pos
                    tgt[L] = np.array([fworld[L][0] - base_xk, fworld[L][1] - base_y0, FOOT_R])
            q_prev = ik_feet(br, 0.42, tgt, q_init=q_prev)
            ref[k, :nq] = q_mj
            ref[k, 0] = base_x0 + (BRAKE * VX + (1.0 - BRAKE) * vx_pred) * k * DT  # BRAKE=1 제동·0=vx_pred(4차)
            ref[k, 7:nq] = br.pin_to_mj_qpos(q_prev)[7:]
            ref[k, nq + 0] = VX
        return ref

    fgid_all = {L: mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, L + '_sphere') for L in FEET}
    fgid = {L: fgid_all[L] for L in ['FL', 'HL']}
    md = mujoco.MjData(mm); md.qpos[:] = q_mj; md.qvel[:] = 0.0; mujoco.mj_forward(mm, md)
    us = np.tile(u_hold, (Nh, 1)); phase = 0
    PROFILE = os.environ.get("PROFILE", "0") == "1"   # break wall-clock into ref/ilqr/sim
    import time as _time
    t_ref = t_ilqr = t_sim = 0.0
    print(f"iLQR-MPC gait walk: VX={VX} Nh={Nh} iters={ITERS} NCYC={NCYC} sub={dyn.sub}")
    falls = 0; fmax = {}
    for c in range(NCTRL):
        x_meas = np.concatenate([md.qpos, md.qvel])
        fworld = {L: md.geom_xpos[fgid_all[L]].copy() for L in FEET}   # actual planted foot positions
        t0 = _time.perf_counter()
        rref = build_ref(phase, md.qpos[0], md.qpos[1], fworld, md.qvel[0])
        cost = QuadCost(nx, nu, nq, rref, qd, rd, qf_scale=8.0)
        t1 = _time.perf_counter()
        xs, us, K = ilqr(dyn, x_meas, us, cost, iters=ITERS, verbose=False)
        u0, K0, xs0 = np.asarray(us[0]).copy(), np.asarray(K[0]).copy(), np.asarray(xs[0]).copy()
        t2 = _time.perf_counter()
        for _ in range(dyn.sub):                    # apply node control for sub sim steps
            xm = np.concatenate([md.qpos, md.qvel])
            md.ctrl[:] = np.clip(u0 + K0 @ (xm - xs0), -200, 200)
            mujoco.mj_step(mm, md)
        if PROFILE:
            t_ref += t1 - t0; t_ilqr += t2 - t1; t_sim += _time.perf_counter() - t2
        us = np.vstack([us[1:], us[-1]]); phase += 1   # shift warm-start + advance gait clock
        for L in fgid:
            fmax[L] = max(fmax.get(L, 0), md.geom_xpos[fgid[L]][2])
        if c % 5 == 0:
            print(f"  c{c:3d} x={md.qpos[0]:+.3f} z={md.qpos[2]:.3f} vx={md.qvel[0]:+.3f} "
                  f"FLz={md.geom_xpos[fgid['FL']][2]:.3f} HLz={md.geom_xpos[fgid['HL']][2]:.3f}", flush=True)
        if md.qpos[2] < 0.20:
            falls = 1; print(f"  FELL at c{c}"); break
    print(f"RESULT VX={VX} ctrl={c+1} falls={falls} x={md.qpos[0]:+.3f} z={md.qpos[2]:.3f} vx={md.qvel[0]:+.3f} "
          f"foot_lift_max FL={fmax.get('FL',0):.3f} HL={fmax.get('HL',0):.3f} (>0.04=stepping, ~0.024=sliding)")
    if PROFILE:
        n = c + 1; tot = t_ref + t_ilqr + t_sim
        print(f"PROFILE/step[ms]: build_ref(CPU IK) {1e3*t_ref/n:.1f} | ilqr(GPU) {1e3*t_ilqr/n:.1f} | "
              f"sim {1e3*t_sim/n:.1f} | total {1e3*tot/n:.1f}  (ref {100*t_ref/tot:.0f}% ilqr {100*t_ilqr/tot:.0f}% sim {100*t_sim/tot:.0f}%)")


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
