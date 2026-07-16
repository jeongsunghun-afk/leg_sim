"""Closed-loop MuJoCo run of the fixed-schedule trot OCP (Phase 0.4).

MPC (Box-FDDP, dt=25ms, N=20) re-solves at 40 Hz, warm-started from the shifted
previous solution. Between solves a low-level PD+feed-forward law (paper Kim2023
eq 6.3) tracks the first planned node at the 500 Hz MuJoCo rate:
    u = u0*_ff + Kp (q_des - q) + Kd (qdot_des - qdot)
The waist is held at 0 by its own stiff PD (locked in the Pinocchio model).

Usage: MJPY ci_mpc_run.py  [VX=0.4] [STEPS=2000] [VIEW=1]
"""
import os, sys, time
import numpy as np
import mujoco
from model_bridge import (MjPinBridge, MJCF, MJ_WAIST_JIDX, MJ2PIN_LEG,
                          apply_gearbox, set_foot_sphere)
from ocp_fixed import TrotGaitOCP
import crocoddyl

VX = float(os.environ.get("VX", "0.4"))
STEPS = int(os.environ.get("STEPS", "2000"))
VIEW = os.environ.get("VIEW", "0") == "1"
KP = float(os.environ.get("KP", "60"))
KD = float(os.environ.get("KD", "2.0"))
FF = float(os.environ.get("FF", "1.0"))     # feed-forward torque scale
FOOT_R = float(os.environ.get("FOOT_R", "0.024"))   # contact sphere radius (ci_mpc only)
WAIST_KP, WAIST_KD = 200.0, 5.0
N = 20
DT_MPC = 2.5e-2
MAXIT = int(os.environ.get("MAXIT", "20"))


def main():
    br = MjPinBridge()
    ocp = TrotGaitOCP(br, dt=DT_MPC, foot_r=FOOT_R)
    mm = mujoco.MjModel.from_xml_path(MJCF)
    apply_gearbox(mm)                                    # reflected rotor inertia (C++ parity)
    set_foot_sphere(mm, FOOT_R)                          # enlarge foot spheres (ci_mpc only)
    md = mujoco.MjData(mm)
    sim_per_mpc = int(round(DT_MPC / mm.opt.timestep))   # 12–13

    # init to standing pose, settle with PD hold
    q_mj = br.pin_to_mj_qpos(ocp.q_stand)
    md.qpos[:] = q_mj
    md.qvel[:] = 0.0
    mujoco.mj_forward(mm, md)
    qstand_mj = q_mj[7:].copy()
    for _ in range(300):
        u = md.qfrc_bias[6:] + KP * (qstand_mj - md.qpos[7:]) - KD * md.qvel[6:]
        u[MJ_WAIST_JIDX] = WAIST_KP * (0.0 - md.qpos[7 + MJ_WAIST_JIDX]) - WAIST_KD * md.qvel[6 + MJ_WAIST_JIDX]
        md.ctrl[:] = np.clip(u, -200, 200)
        mujoco.mj_step(mm, md)

    # warm-start buffers
    x0 = br.mj_to_pin_x(md.qpos, md.qvel)
    prob = ocp.create_problem(x0, v_cmd=(VX, 0., 0.), N=N)
    solver = crocoddyl.SolverBoxFDDP(prob)
    xs = [x0] * (N + 1)
    us = prob.quasiStatic([x0] * N)
    solver.solve(xs, us, 100)
    xs, us = list(solver.xs), list(solver.us)

    viewer = None
    if VIEW:
        import mujoco.viewer as _mjv
        viewer = _mjv.launch_passive(mm, md)

    state = ocp.state
    node = 0
    falls = 0
    t_solve = 0.0
    n_solve = 0
    gait_acc = 0.0
    q_des = qstand_mj.copy(); qd_des = np.zeros(17); u_ff = np.zeros(17)
    tau_pk = np.zeros(17); vx_sum = 0.0; vx_n = 0; sat_n = 0
    # physical peak per MJCF joint (HL,HR,waist,FL,FR order): hip84 thigh84 calf126 foot100.8
    tau_cap = np.array([84, 84, 126, 100.8] * 2 + [84.0] + [84, 84, 126, 100.8] * 2)
    for step in range(STEPS):
        # ---- MPC solve at 40 Hz ----
        if step % sim_per_mpc == 0:
            x0 = br.mj_to_pin_x(md.qpos, md.qvel)
            gait_acc += sim_per_mpc * mm.opt.timestep / DT_MPC   # advance gait clock in real time
            node = int(round(gait_acc))
            prob = ocp.create_problem(x0, v_cmd=(VX, 0., 0.), N=N, k0=node)
            xs_ws = [x0] + xs[2:] + [xs[-1]]
            us_ws = us[1:] + [us[-1]]
            solver = crocoddyl.SolverBoxFDDP(prob)
            t0 = time.perf_counter()
            solver.solve(xs_ws, us_ws, MAXIT)
            t_solve += time.perf_counter() - t0
            n_solve += 1
            xs, us = list(solver.xs), list(solver.us)
            q_des = br.scatter_leg(xs[1][7:br.nq], waist=0.0)
            qd_des = br.scatter_leg(xs[1][br.nq + 6:], waist=0.0)
            u_ff = br.pin_tau_to_mj_ctrl(us[0])

        # ---- low-level: joint PD around plan node-1 + feed-forward torque (500 Hz) ----
        q = md.qpos[7:]; qd = md.qvel[6:]
        u = FF * u_ff + KP * (q_des - q) + KD * (qd_des - qd)
        u[MJ_WAIST_JIDX] = WAIST_KP * (0.0 - q[MJ_WAIST_JIDX]) - WAIST_KD * qd[MJ_WAIST_JIDX]
        md.ctrl[:] = np.clip(u, -200, 200)
        mujoco.mj_step(mm, md)

        # metrics (skip 0.5 s startup transient). qfrc_actuator is the generalized
        # joint torque AFTER MuJoCo clamps to jnt_actfrcrange (physical peak).
        if step > 250:
            ja = np.abs(md.qfrc_actuator[6:])
            tau_pk = np.maximum(tau_pk, ja)
            if (ja / tau_cap).max() > 0.98:      # any leg joint at >=98% of peak
                sat_n += 1
            vx_sum += md.qvel[0]; vx_n += 1
        # fall check
        if md.qpos[2] < 0.18:
            falls = 1
            break
        if viewer is not None and step % 4 == 0:
            viewer.sync()

    # peak torque by joint group (MJCF order HL,HR,waist,FL,FR; foot=idx 3,7,12,16 etc.)
    g = {'hip': [0, 4, 9, 13], 'thigh': [1, 5, 10, 14], 'calf': [2, 6, 11, 15], 'foot': [3, 7, 12, 16]}
    tpk = {k: tau_pk[v].max() for k, v in g.items()}
    vx_mean = vx_sum / max(1, vx_n)
    sat_pct = 100.0 * sat_n / max(1, vx_n)
    print(f"VX={VX} steps={step+1} falls={falls} "
          f"x={md.qpos[0]:+.3f} z={md.qpos[2]:.3f} vx_mean={vx_mean:+.3f} "
          f"tau_thigh={tpk['thigh']:.0f} tau_calf={tpk['calf']:.0f} tau_foot={tpk['foot']:.0f} "
          f"sat_pct={sat_pct:.0f} solve_ms={1e3*t_solve/max(1,n_solve):.1f}")
    if viewer is not None:
        viewer.close()
    return falls


if __name__ == "__main__":
    sys.exit(main())
