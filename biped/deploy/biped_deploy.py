"""biped 배포 러너 — 컨트롤러(MPC+WBIC)를 RobotInterface 통해 구동.

플랜트만 교체:  --backend sim (기본, MuJoCo)  /  --backend hw (실모터, 스텁)
GUI(teleop_gui_biped)와 동일 JSON 채널(/tmp/biped_cmd.json) 사용 → sim/실 조종 동일.
배포 루프(mature quad와 동형):
    st = iface.read()  →  iface.apply_state(d, st)  →  [GUI 명령 반영]  →  c.control(dt)
    →  LowCmd(tau=d.ctrl)  →  iface.write(cmd)         # sim=mj_step / HW=setTorqueRef

실행:
  sim 검증 : python biped_deploy.py --backend sim  [--view]      # biped_run과 동일 결과
  실배포   : python biped_deploy.py --backend hw                # HardwareInterface 구현 후
"""
from __future__ import annotations
import os, sys, time, json, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # biped/
import biped_mpc_wbic as BM
from biped_wbic import base_rpy
from robot_interface import SimInterface, HardwareInterface, LowCmd, StateEstimator, FOOT_NAMES

CMD   = os.environ.get('QUAD_CMD',   '/tmp/biped_cmd.json')
STATE = os.environ.get('QUAD_STATE', '/tmp/biped_state.json')


def read_cmd():
    try:
        with open(CMD) as f:
            return json.load(f)
    except Exception:
        return None


def make_interface(backend, c):
    if backend == 'sim':
        return SimInterface(c.m, c.d)
    if backend == 'hw':
        return HardwareInterface(BM.BS.BW.MJCF)   # ★NotImplementedError까지 = SDK 연결 대기
    raise ValueError(backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='sim', choices=['sim', 'hw'])
    ap.add_argument('--view', action='store_true')
    ap.add_argument('--est-ctrl', action='store_true',
                    help='폐루프 검증: 추정 base로 제어(물리는 GT). 추정기 품질 검증(sim)')
    ap.add_argument('--T', type=float, default=1e9)
    args = ap.parse_args()

    c = BM.BipedMPCWBIC(); c.reset(); c.setup_mpc()
    m, d = c.m, c.d; dt = m.opt.timestep
    iface = make_interface(args.backend, c)
    z_home = float(c.com_ref[2]); body_h = z_home
    mode, prev_mode = 'stand', 'stand'

    # ── 상태추정기(leg-odometry) — 센서만으로 base pose/vel 복원, GT 대비 오차 추출 ──
    import mujoco
    sph = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
    rad = [float(m.geom_size[g][0]) for g in sph]
    est = StateEstimator(m, sph, rad, dt); est.reset(d.qpos[0:3])
    est_perr = est_verr = 0.0
    def est_reset():
        est.reset(d.qpos[0:3])

    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(m, d)
    print(f"biped_deploy · backend={args.backend} · est-ctrl={args.est_ctrl} · CMD={CMD} · 기본높이 {z_home:.3f}")

    k = 0; t0 = time.perf_counter()
    while ((viewer is None) or viewer.is_running()) and k*dt < args.T:
        st = iface.read()                              # ① 센서 → LowState
        iface.apply_state(d, st)                       # ② 측정 주입(HW만 실효, sim=no-op)
        if k % 20 == 0:                                # ③ GUI 명령 폴링(50Hz)
            cmd = read_cmd()
            if cmd:
                mode = cmd.get('mode', mode); body_h = float(cmd.get('body_h', body_h))
                if mode == 'reset':
                    c.reset(); c.setup_mpc(); c.com_ref[2] = body_h; c._k = 0; mode = 'stand'; est_reset()
                if prev_mode == 'off' and mode != 'off':   # 전원 재투입
                    c.reset(); c.setup_mpc(); c.com_ref[2] = body_h; c._k = 0; est_reset()
                prev_mode = mode
                walking = mode == 'walk'
                c.vx_cmd = float(cmd.get('v', 0.0))  if walking else 0.0
                c.wz_cmd = float(cmd.get('w', 0.0))  if walking else 0.0
                c.vy_cmd = float(cmd.get('vy', 0.0)) if walking else 0.0
                c.com_ref[2] = body_h
        off = (mode == 'off')                          # ④ 모터 on/off
        iface.enable_motors(not off)
        # ── 상태추정(센서만): 관절 q/dq + IMU quat/gyro + 접촉 → base pose/vel ──
        if not off:
            if mode == 'walk':
                est.estimate(st.q, st.dq, st.quat, st.gyro, st.foot_contact)
            else:
                est.hold(); est.p[:] = est.p + est.v * dt   # 정지: 속도0 유지(드리프트 억제)
            est_perr = float(np.linalg.norm(est.p - d.qpos[0:3]))   # GT 대비 위치오차[m] (sim만 의미)
            est_verr = float(np.linalg.norm(est.v - d.qvel[0:3]))   # GT 대비 속도오차[m/s]
        if off:
            iface.write(LowCmd())                      # tau=0(limp) + step
        elif args.est_ctrl and args.backend == 'sim':  # ⑤' 폐루프 검증: 추정 base로 제어, 물리는 GT
            gp, gv = d.qpos[0:3].copy(), d.qvel[0:3].copy()
            d.qpos[0:3] = est.p; d.qvel[0:3] = est.v; mujoco.mj_forward(m, d)
            c.control(dt)                              # tau ← 추정 base(드리프트 포함)
            d.qpos[0:3] = gp; d.qvel[0:3] = gv; mujoco.mj_forward(m, d)   # 물리는 GT 복원
            iface.write(LowCmd(tau=d.ctrl.copy()))
        else:
            c.control(dt)                              # ⑤ WBIC → d.ctrl (tau)
            iface.write(LowCmd(tau=d.ctrl.copy()))     # ⑥ 토크 명령 → 플랜트
        k += 1
        tilt = np.hypot(*base_rpy(d.qpos[3:7])[:2])
        if not off and (d.qpos[2] < 0.2 or tilt > 45):  # 낙상 자동리셋(sim; HW는 안전정지로 교체)
            if viewer is not None: time.sleep(0.3)
            c.reset(); c.setup_mpc(); c.com_ref[2] = body_h; c._k = 0; est_reset()
        if k % 20 == 0:
            vx_act = float((c.Jc_cache @ d.qvel)[0]) if hasattr(c, 'Jc_cache') else 0.0
            yaw = float(base_rpy(d.qpos[3:7])[2])
            try:
                tmp = STATE + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump({'mode': mode, 'base_z': float(d.qpos[2]), 'vx_cmd': float(c.vx_cmd),
                               'vx_act': vx_act, 'wz_cmd': float(c.wz_cmd), 'yaw': yaw,
                               'tilt': float(tilt), 'x': float(d.qpos[0]), 'y': float(d.qpos[1]),
                               'est_perr': est_perr, 'est_verr': est_verr,          # ★GT 대비 추정오차
                               'est_x': float(est.p[0]), 'est_y': float(est.p[1])}, f)
                os.replace(tmp, STATE)
            except Exception:
                pass
        if k % 500 == 0 and mode == 'walk':            # 1s마다 추정오차 리포트(헤드리스)
            print(f"  t={k*dt:5.1f}s  est err pos={est_perr*100:5.1f}cm vel={est_verr:.3f}m/s"
                  f"  GT xy=({d.qpos[0]:+.2f},{d.qpos[1]:+.2f}) EST xy=({est.p[0]:+.2f},{est.p[1]:+.2f})")
        if viewer is not None and k % 8 == 0:
            viewer.sync()
        lag = t0 + k*dt - time.perf_counter()          # 실시간 페이싱
        if lag > 0:
            time.sleep(lag)


if __name__ == '__main__':
    main()
