"""MJX-accelerated sampling CI-MPC (§15 path B, GPU parallel rollouts).

Same idea as mppi_mpc.py but the Nsamp rollouts run in parallel on the GPU via
MJX + jax.vmap, with the whole H*sub-step rollout fused into one jitted lax.scan
(no per-step Python dispatch). Contacts are handled by MJX exactly -> zero sim2sim
gap vs the deployment MuJoCo model.

Hybrid loop: the ACTUAL sim + applied control run on CPU MuJoCo; each replan ships
the current state to the GPU, evaluates Nsamp sampled joint-reference trajectories,
and does the MPPI weighted update on the host.

Usage: MJPY mjx_mppi.py  [VX=0.3] [STEPS=300] [H=15] [NSAMP=64] [VIEW=1]
"""
import os, sys, time, functools
import numpy as np
import mujoco
import jax
import jax.numpy as jnp
from mujoco import mjx
from scipy.ndimage import gaussian_filter1d
from model_bridge import (MJCF, MJ_WAIST_JIDX, apply_gearbox, set_foot_sphere,
                          strip_mesh_collision, MjPinBridge)
from ocp_fixed import standing_ik, ik_feet, FEET

VX = float(os.environ.get("VX", "0.3"))
STEPS = int(os.environ.get("STEPS", "300"))
H = int(os.environ.get("H", "15"))
NSAMP = int(os.environ.get("NSAMP", "64"))
ITERS = int(os.environ.get("ITERS", "2"))
SIGMA = float(os.environ.get("SIGMA", "0.10"))
LAM = float(os.environ.get("LAM", "40.0"))
DT = float(os.environ.get("DT", "0.02"))
FOOT_R = float(os.environ.get("FOOT_R", "0.024"))
STEP_H = float(os.environ.get("STEP_H", "0.07"))
NCYC = int(os.environ.get("NCYC", "20"))
PRIOR_GAIN = float(os.environ.get("PRIOR_GAIN", "1.3"))   # prior step over-drive (slippage comp)
VIEW = os.environ.get("VIEW", "0") == "1"
KP = float(os.environ.get("KP", "300"))     # stiff enough to hold stance (PD has no base-gravity FF)
KD = float(os.environ.get("KD", "8"))
IKZ = float(os.environ.get("IKZ", "0.42"))  # IK base height; PD sags to ~0.38

TAU_NP = np.array([84, 84, 126, 100.8] * 2 + [84.0] + [84, 84, 126, 100.8] * 2)
WID = MJ_WAIST_JIDX


MJCF_PATH = os.environ.get("MJCF_PATH", MJCF)         # override to a terrain scene
DISABLE_FLOOR = os.environ.get("DISABLE_FLOOR", "0") == "1"   # gaps become real voids


def build_model():
    mm = mujoco.MjModel.from_xml_path(MJCF_PATH)
    apply_gearbox(mm)
    set_foot_sphere(mm, FOOT_R)
    strip_mesh_collision(mm)             # only sphere feet + terrain collide (fast MJX)
    if DISABLE_FLOOR:                    # so gaps between platforms are true voids
        fg = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if fg >= 0:
            mm.geom_contype[fg] = 0
            mm.geom_conaffinity[fg] = 0
    return mm


def make_rollout(mx, sub):
    """Return jitted rollout_batch(qpos0, qvel0, qref_batch) -> costs[N]."""
    TAU = jnp.array(TAU_NP)

    def step_cost(dx, y0):
        z = dx.qpos[2]
        qw, qx, qy, qz = dx.qpos[3], dx.qpos[4], dx.qpos[5], dx.qpos[6]
        tilt = 1.0 - 2.0 * (qx * qx + qy * qy)      # 1 = upright
        pitch = 2.0 * (qw * qy - qz * qx)           # sin(pitch); <0 = nose-down (dive)
        vx, vy = dx.qvel[0], dx.qvel[1]
        # Stable forward gait: velocity tracking (weight ~30 is the stable sweet spot),
        # lateral line hold, dive-blocking pitch penalty, collapse barrier + soft floor.
        # No height TRACKING (freezes gait bob). Stable ~0.12 m/s - ideal for careful
        # foothold crossing where slow is better than fast.
        c = 30.0 * (vx - VX) ** 2 + 6.0 * vy ** 2
        c += 40.0 * (dx.qpos[1] - y0) ** 2          # hold lateral line (no side drift)
        c += 15.0 * (1.0 - tilt) + 3.0 * dx.qvel[5] ** 2
        c += 40.0 * pitch ** 2 + 80.0 * jnp.maximum(0.0, -pitch) ** 2   # block dive
        c += 200.0 * jnp.maximum(0.0, vx - 0.40) ** 2   # anti-lunge overshoot barrier (soft, not a state clip)
        c += jnp.where(z < 0.30, 2000.0, 0.0)           # collapse barrier
        c += 500.0 * jnp.maximum(0.0, 0.33 - z) ** 2    # soft floor
        return c

    def rollout_one(dx0, qref):                     # qref [H, nu]
        y0 = dx0.qpos[1]
        def node(dx, tgt):
            def substep(dx, _):
                tau = dx.qfrc_bias[6:] + KP * (tgt - dx.qpos[7:]) - KD * dx.qvel[6:]
                tau = tau.at[WID].set(200.0 * (0.0 - dx.qpos[7 + WID]) - 5.0 * dx.qvel[6 + WID])
                tau = jnp.clip(tau, -TAU, TAU)
                dx = dx.replace(ctrl=tau)
                dx = mjx.step(mx, dx)
                return dx, None
            dx, _ = jax.lax.scan(substep, dx, None, length=sub)
            return dx, step_cost(dx, y0)
        dx, costs = jax.lax.scan(node, dx0, qref)
        return costs.sum()

    dx_template = mjx.make_data(mx)

    @jax.jit
    def rollout_batch(qpos0, qvel0, qref_batch):
        dx0 = dx_template.replace(qpos=qpos0, qvel=qvel0)
        dx0 = mjx.forward(mx, dx0)
        return jax.vmap(rollout_one, in_axes=(None, 0))(dx0, qref_batch)

    return rollout_batch


class MjxMPPI:
    def __init__(self):
        self.mm = build_model()
        self.mx = mjx.put_model(self.mm)
        self.nu = self.mm.nu
        self.sub = max(1, round(DT / self.mm.opt.timestep))
        self.rollout = make_rollout(self.mx, self.sub)
        self.fg = {L: mujoco.mj_name2id(self.mm, mujoco.mjtObj.mjOBJ_GEOM, L + '_sphere')
                   for L in ['HL', 'HR', 'FL', 'FR']}
        self._build_prior()

    def _build_prior(self):
        """Forward-trot kinematic reference (periodic in body frame).

        Stance feet sweep BACKWARD relative to the body (propulsion); swing feet
        arc forward from liftoff to landing (Raibert). Half-step = 0.5*VX*T_stance.
        This is only a seed - MuJoCo physics decides the real contact timing.
        """
        br = MjPinBridge()
        base_z = IKZ
        q_stand = standing_ik(br, base_z, foot_z=FOOT_R)
        foot0 = br.foot_positions(q_stand)
        self.q_stand_mj = br.pin_to_mj_qpos(q_stand)
        self.qstand_j = self.q_stand_mj[7:].copy()
        T_stance = 0.5 * NCYC * DT
        half = 0.5 * VX * PRIOR_GAIN * T_stance
        pair_swing = {'FR': (0.0, 0.5), 'HL': (0.0, 0.5),   # (swing_start, swing_end) in cycle frac
                      'FL': (0.5, 1.0), 'HR': (0.5, 1.0)}
        table = np.zeros((NCYC, self.nu))
        for c in range(NCYC):
            p = c / NCYC
            tgt = {L: foot0[L].copy() for L in FEET}
            for L in FEET:
                s0, s1 = pair_swing[L]
                in_swing = (s0 <= p < s1)
                if in_swing:
                    sp = (p - s0) / (s1 - s0)                 # 0..1 swing progress
                    tgt[L][0] += half * (2 * sp - 1)         # back -> front
                    tgt[L][2] = FOOT_R + STEP_H * np.sin(np.pi * sp)
                else:
                    st = ((p - s1) % 1.0) / (1.0 - (s1 - s0))  # 0..1 stance progress
                    tgt[L][0] += half * (1 - 2 * st)         # front -> back (propulsion)
            q = ik_feet(br, base_z, tgt, q_init=q_stand)
            table[c] = br.pin_to_mj_qpos(q)[7:]
        self.prior = table

    def prior_node(self, gk):
        return self.prior[gk % NCYC]

    def prior_horizon(self, phase):
        return np.array([self.prior_node(phase + k) for k in range(H)])

    def replan(self, qpos, qvel, nominal):
        qp = jnp.array(qpos); qv = jnp.array(qvel)
        for _ in range(ITERS):
            eps = np.random.randn(NSAMP, H, self.nu) * SIGMA
            eps = gaussian_filter1d(eps, sigma=3, axis=1)
            batch = jnp.array(nominal[None] + eps)
            costs = np.array(self.rollout(qp, qv, batch))
            beta = costs.min()
            w = np.exp(-(costs - beta) / LAM); w /= w.sum()
            nominal = nominal + np.einsum('s,shu->hu', w, eps)
            nominal = gaussian_filter1d(nominal, sigma=1, axis=0)
        return nominal, beta


def main():
    mpc = MjxMPPI()
    mm, m = mpc.mm, mpc.mm
    d = mujoco.MjData(mm)
    d.qpos[:] = mpc.q_stand_mj; d.qvel[:] = 0.0; mujoco.mj_forward(mm, d)
    qs = mpc.qstand_j
    for _ in range(300):
        tau = d.qfrc_bias[6:] + KP * (qs - d.qpos[7:]) - KD * d.qvel[6:]
        tau[WID] = 200 * (0 - d.qpos[7 + WID]) - 5 * d.qvel[6 + WID]
        d.ctrl[:] = np.clip(tau, -TAU_NP, TAU_NP); mujoco.mj_step(mm, d)

    phase = 0
    nominal = mpc.prior_horizon(phase)
    # warm up JIT
    t = time.time(); mpc.replan(d.qpos.copy(), d.qvel.copy(), nominal)
    print(f"[jit warmup {time.time()-t:.1f}s]", flush=True)

    viewer = None
    if VIEW:
        import mujoco.viewer as _mjv
        viewer = _mjv.launch_passive(mm, d)
    falls = 0; t_plan = 0.0; n = 0
    for step in range(STEPS):
        t0 = time.perf_counter()
        nominal, best = mpc.replan(d.qpos.copy(), d.qvel.copy(), nominal)
        t_plan += time.perf_counter() - t0; n += 1
        tgt = nominal[0]
        for _ in range(mpc.sub):
            tau = d.qfrc_bias[6:] + KP * (tgt - d.qpos[7:]) - KD * d.qvel[6:]
            tau[WID] = 200 * (0 - d.qpos[7 + WID]) - 5 * d.qvel[6 + WID]
            d.ctrl[:] = np.clip(tau, -TAU_NP, TAU_NP); mujoco.mj_step(mm, d)
        phase += 1
        nominal = np.vstack([nominal[1:], mpc.prior_node(phase + H - 1)])
        if step % 50 == 0:
            print(f"  step{step:4d} x={d.qpos[0]:+.3f} y={d.qpos[1]:+.3f} z={d.qpos[2]:.3f} "
                  f"vx={d.qvel[0]:+.3f}", flush=True)
        if d.qpos[2] < 0.20:
            falls = 1; break
        if viewer is not None and step % 2 == 0:
            viewer.sync()
    print(f"VX={VX} steps={step+1} falls={falls} x={d.qpos[0]:+.3f} z={d.qpos[2]:.3f} "
          f"vx={d.qvel[0]:+.3f} plan_ms={1e3*t_plan/max(1,n):.1f} "
          f"(H={H} Nsamp={NSAMP} iters={ITERS})")
    if viewer is not None:
        viewer.close()
    return falls


if __name__ == "__main__":
    sys.exit(main())
