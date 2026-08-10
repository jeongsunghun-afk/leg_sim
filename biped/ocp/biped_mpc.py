"""Biped closed-loop MPC (crocoddyl OCP ↔ MuJoCo sim) — pitch 검증.

단발 OCP는 접촉 스위칭 gap으로 미수렴 → **매 제어틱을 실제 sim 상태에서 solve**(near-feasible)하고
u[0]만 적용, 게이트 클럭 전진. ci_mpc 방식. "호라이즌 OCP가 flat 단일지지 backward-pitch를
잡는지"를 실제 폐루프로 확인.

pinocchio(URDF)↔MuJoCo: URDF가 MJCF서 생성돼 **관절 순서 동일(HL4,HR4)=순열 불필요**.
base quat wxyz↔xyzw, base lin-vel world↔local(RᵀV)만 변환.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, mujoco, crocoddyl, pinocchio as pin
from biped_ocp import BipedWalkOCP, MJCF, TAU_LIM

# GEARBOX(반사관성) — 폐루프 수치안정 필수(ci_mpc 교훈). C++ 컨트롤러 parity.
GEAR = [7., 7., 10.5, 8.]; I_ROTOR = 1e-4; J_DAMP = 0.1; J_FRIC = 0.5


def apply_gearbox(m):
    for j in range(m.nu):
        dof = 6 + j; N = GEAR[j % 4]
        m.dof_armature[dof] = I_ROTOR * N * N
        m.dof_damping[dof] = J_DAMP; m.dof_frictionloss[dof] = J_FRIC


def quat2mat(w, x, y, z):
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def mj_to_pin_x(m, d, nq, nv):
    p = d.qpos[0:3].copy(); w, x, y, z = d.qpos[3:7]
    R = quat2mat(w, x, y, z)
    q = np.concatenate([p, [x, y, z, w], d.qpos[7:7+nv-6]])
    v = np.concatenate([R.T @ d.qvel[0:3], d.qvel[3:6], d.qvel[6:6+nv-6]])
    return np.concatenate([q, v])


def pitch_deg(d):
    w, x, y, z = d.qpos[3:7]
    return np.degrees(np.arcsin(np.clip(2*(w*y - z*x), -1, 1)))


def run(v_cmd=0.15, T=4.0, T_ss=12, N=24, replan=1, view=False):
    ocp = BipedWalkOCP(dt=2e-2)
    ocp.w_cone = 0.5
    m = mujoco.MjModel.from_xml_path(MJCF); m.opt.timestep = 2e-3
    apply_gearbox(m)
    # 메시 충돌 끔(구 접촉만) — biped_flatfoot는 이미 mesh contype=0이나 안전차원
    d = mujoco.MjData(m)
    nq, nv = ocp.model.nq, ocp.model.nv
    # 초기 자세 = q_stand (MJCF 순서로)
    d.qpos[0:3] = ocp.q_stand[0:3]; d.qpos[2] = ocp.q_stand[2]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:7+nv-6] = ocp.q_stand[7:]
    mujoco.mj_forward(m, d)

    sim_per_mpc = int(round(ocp.dt / m.opt.timestep))   # 10
    xs = ocp.warm_start((v_cmd, 0, 0), N, T_ss, 0.05)   # 첫 warm-start
    us = None
    nsteps = int(T / m.opt.timestep)
    falls = 0; k_gait = 0; u_hold = np.zeros(ocp.nu)
    log = []
    viewer = mujoco.viewer.launch_passive(m, d) if view else None
    for it in range(nsteps):
        if us is None or it % (replan * sim_per_mpc) == 0:   # replan(낙상 후 즉시 포함)
            x0 = mj_to_pin_x(m, d, nq, nv)
            prob = ocp.create_problem(x0, v_cmd=(v_cmd, 0, 0), N=N, T_ss=T_ss, T_ds=0, k0=k_gait)
            solver = crocoddyl.SolverBoxFDDP(prob)
            if us is None:
                xs_ws = ocp.warm_start((v_cmd, 0, 0), N, T_ss, 0.05); xs_ws[0] = x0
                us_ws = prob.quasiStatic(xs_ws[:-1])
            else:                                        # 이전 해 shift
                xs_ws = list(xs[replan:]) + [xs[-1]] * replan; xs_ws[0] = x0
                us_ws = list(us[replan:]) + [us[-1]] * replan
                xs_ws = xs_ws[:N+1] + [xs_ws[-1]] * max(0, N+1-len(xs_ws)); xs_ws = xs_ws[:N+1]
                us_ws = us_ws[:N] + [us_ws[-1]] * max(0, N-len(us_ws)); us_ws = us_ws[:N]
            solver.solve(xs_ws, us_ws, 30)
            xs = list(solver.xs); us = list(solver.us)
            K0 = np.asarray(solver.K[0]); x_ref0 = np.asarray(xs[0])   # ★피드백 게인·기준상태
            k_gait += replan
        # ★MPC 제어법 = 피드포워드 + LQR 피드백: u = us[0] − K[0]·diff(xs[0], x_now)
        x_now = mj_to_pin_x(m, d, nq, nv)
        dx = ocp.state.diff(x_ref0, x_now)
        u = np.asarray(us[0]) - K0 @ dx
        d.ctrl[:] = np.clip(u, -TAU_LIM, TAU_LIM)         # pin u(HL,HR)=mujoco ctrl 동일순서
        mujoco.mj_step(m, d)
        if viewer and it % 5 == 0: viewer.sync()
        if it % 50 == 0:
            log.append((it*m.opt.timestep, d.qpos[0], d.qpos[2], pitch_deg(d)))
        if d.qpos[2] < 0.25 or abs(pitch_deg(d)) > 40:   # 낙상
            falls += 1
            d.qpos[:] = 0; d.qpos[2] = ocp.q_stand[2]; d.qpos[3] = 1
            d.qpos[7:7+nv-6] = ocp.q_stand[7:]; d.qvel[:] = 0
            mujoco.mj_forward(m, d); us = None; k_gait = 0
    if viewer: viewer.close()
    print("t     base_x  base_z  pitch°")
    for t, x, z, p in log[::2]:
        print(f"{t:4.1f}  {x:+.3f}  {z:.3f}  {p:+.1f}")
    print(f"\nvx={v_cmd} T={T}s · falls={falls} · 최종 base_x={d.qpos[0]:+.3f} pitch={pitch_deg(d):+.1f}°")


if __name__ == "__main__":
    import sys
    run(v_cmd=float(sys.argv[1]) if len(sys.argv) > 1 else 0.15,
        T=float(sys.argv[2]) if len(sys.argv) > 2 else 4.0)
