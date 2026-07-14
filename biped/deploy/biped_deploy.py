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

    if os.environ.get('K_RETURN'):                 # 진단: 절대 원점복귀 게인 오버라이드
        BM.BS.K_RETURN = float(os.environ['K_RETURN'])
    c = BM.BipedMPCWBIC(); c.reset(); c.setup_mpc()
    m, d = c.m, c.d; dt = m.opt.timestep
    NOISE = float(os.environ.get('NOISE', '0')); np.random.seed(0)   # 센서 노이즈 배율(재현용 시드)
    from collections import deque
    dt0 = m.opt.timestep
    SLAT = int(round(float(os.environ.get('SENSE_LAT_MS', '0')) / 1000 / dt0))  # 센서→제어 지연[step]
    ALAT = int(round(float(os.environ.get('ACT_LAT_MS', '0')) / 1000 / dt0))    # 제어→구동 지연[step]
    LCOMP = int(round(float(os.environ.get('LAT_COMP_MS', '0')) / 1000 / dt0))  # ★지연보상 예측[step]
    sense_buf = deque(); act_buf = deque()
    iface = make_interface(args.backend, c)
    z_home = float(c.com_ref[2]); body_h = z_home
    mode, prev_mode = 'stand', 'stand'

    # ── 상태추정기(leg-odometry) — 센서만으로 base pose/vel 복원, GT 대비 오차 추출 ──
    import mujoco
    sph = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
    rad = [float(m.geom_size[g][0]) for g in sph]
    est = StateEstimator(m, sph, rad, dt,                          # 튜닝: env로 오버라이드
                         alpha=float(os.environ.get('EST_ALPHA', '0.4')),
                         dwell_steps=int(os.environ.get('EST_DWELL', '15')),
                         k_anchor=float(os.environ.get('EST_ANCHOR', '0.05')),
                         use_accel=os.environ.get('EST_ACCEL', '0') == '1',
                         contact_height=os.environ.get('EST_NOCH') is None)  # 접촉높이(기본 on)
    est.reset(d.qpos[0:3])
    dpred = mujoco.MjData(m) if (LCOMP > 0 and os.environ.get('LAT_COMP_KIN') is None) else None  # 롤아웃 scratch
    est_perr = est_verr = 0.0
    def est_reset():
        est.reset(d.qpos[0:3])

    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(m, d)
    print(f"biped_deploy · backend={args.backend} · est-ctrl={args.est_ctrl} · CMD={CMD} · 기본높이 {z_home:.3f}")

    k = 0; falls = 0; t0 = time.perf_counter()
    while ((viewer is None) or viewer.is_running()) and k*dt < args.T:
        st = iface.read()                              # ① 센서 → LowState
        if NOISE > 0:                                  # ★실 센서 노이즈 주입(배포 강건성 스트레스)
            st.q = st.q + np.random.normal(0, NOISE*0.005, 8)     # 엔코더 σ≈0.005rad·NOISE
            st.dq = st.dq + np.random.normal(0, NOISE*0.05, 8)    # 관절속도 σ≈0.05
            st.gyro = st.gyro + np.random.normal(0, NOISE*0.02, 3)# 자이로 σ≈0.02rad/s
            st.acc = st.acc + np.random.normal(0, NOISE*0.5, 3)   # 가속도 σ≈0.5m/s²
        sense_buf.append(st)                           # ★센서→제어 지연(링버퍼): 지연된 상태로 추정·제어
        while len(sense_buf) > SLAT + 1: sense_buf.popleft()
        st = sense_buf[0]                              # SLAT step 전 상태(warmup=가장 오래된 것)
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
        # ── 상태추정(센서만): 관절 q/dq + IMU quat/gyro/accel + 접촉 → base pose/vel (실로봇 동일, 항상) ──
        if not off:
            est.estimate(st.q, st.dq, st.quat, st.gyro, st.foot_contact, acc=st.acc)
            est_perr = float(np.linalg.norm(est.p - d.qpos[0:3]))   # GT 대비 위치오차[m] (sim만 의미)
            est_verr = float(np.linalg.norm(est.v - d.qvel[0:3]))   # GT 대비 속도오차[m/s]
        if off:
            cmd_tau = np.zeros(m.nu)                    # 전원 off = limp
        elif args.est_ctrl and args.backend == 'sim':  # ⑤' 폐루프 검증: 지연·추정 제어상태로 제어, 물리는 GT
            _pf = os.environ.get('EST_PERFECT')
            # 컨트롤러가 보는 '제어 상태'(실로봇 동일): base=추정기(지연센서) · attitude/gyro·관절=지연센서 st
            cbp = est.p.copy(); cbv = est.v.copy()
            cq, cdq = st.q.copy(), st.dq.copy(); cquat, cgyro = st.quat.copy(), st.gyro.copy()
            if _pf or os.environ.get('EST_POSGT'): cbp = d.qpos[0:3].copy()   # 진단
            if _pf or os.environ.get('EST_VELGT'): cbv = d.qvel[0:3].copy()
            if os.environ.get('EST_ZGT'): cbp[2] = d.qpos[2]
            if LCOMP > 0 and dpred is not None:        # ★지연보상2: 모델 롤아웃(동역학+in-flight 토크로 전진)
                dpred.qpos[0:3] = cbp; dpred.qpos[3:7] = cquat; dpred.qpos[7:] = cq
                dpred.qvel[0:3] = cbv; dpred.qvel[3:6] = cgyro; dpred.qvel[6:] = cdq
                tau_hold = act_buf[-1] if act_buf else np.zeros(m.nu)   # in-flight(마지막 명령) 유지
                for _ in range(LCOMP):
                    dpred.ctrl[:] = tau_hold; mujoco.mj_step(m, dpred)
                cbp = dpred.qpos[0:3].copy(); cquat = dpred.qpos[3:7].copy(); cq = dpred.qpos[7:].copy()
                cbv = dpred.qvel[0:3].copy(); cgyro = dpred.qvel[3:6].copy(); cdq = dpred.qvel[6:].copy()
            elif LCOMP > 0:                            # ★지연보상1: 속도 외삽(kinematic, dpred 없을때)
                tt = LCOMP * dt0
                cbp = cbp + cbv * tt; cq = cq + cdq * tt
                mujoco.mju_quatIntegrate(cquat, cgyro, tt)
            sqp, sqv = d.qpos.copy(), d.qvel.copy()    # GT(물리) 저장
            d.qpos[0:3] = cbp; d.qpos[3:7] = cquat; d.qpos[7:] = cq
            d.qvel[0:3] = cbv; d.qvel[3:6] = cgyro; d.qvel[6:] = cdq
            mujoco.mj_forward(m, d)
            c.control(dt)                              # tau ← (예측된) 제어상태
            d.qpos[:] = sqp; d.qvel[:] = sqv; mujoco.mj_forward(m, d)   # 물리는 GT 복원
            cmd_tau = d.ctrl.copy()
        else:
            c.control(dt)                              # ⑤ WBIC → d.ctrl (tau)
            cmd_tau = d.ctrl.copy()
        act_buf.append(cmd_tau)                        # ★제어→구동 지연(링버퍼): ALAT step 전 토크 인가
        while len(act_buf) > ALAT + 1: act_buf.popleft()
        iface.write(LowCmd() if off else LowCmd(tau=act_buf[0]))   # ⑥ 토크 명령 → 플랜트
        k += 1
        tilt = np.hypot(*base_rpy(d.qpos[3:7])[:2])
        if not off and (d.qpos[2] < 0.2 or tilt > 45):  # 낙상 자동리셋(sim; HW는 안전정지로 교체)
            if viewer is not None: time.sleep(0.3)
            c.reset(); c.setup_mpc(); c.com_ref[2] = body_h; c._k = 0; est_reset(); falls += 1
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
        if viewer is None:                             # 헤드리스=최대속(페이싱 생략, 스윕용)
            continue
        lag = t0 + k*dt - time.perf_counter()          # 실시간 페이싱
        if lag > 0:
            time.sleep(lag)
    print(f"[요약] backend={args.backend} est-ctrl={args.est_ctrl} T={k*dt:.1f}s "
          f"falls={falls}  최종 est err pos={est_perr*100:.1f}cm vel={est_verr:.3f}m/s")


if __name__ == '__main__':
    main()
