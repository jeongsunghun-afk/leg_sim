"""biped 실행기 — MPC+WBIC 컨트롤러 + 뷰어 + JSON 명령채널.

GUI(teleop_gui_biped.py)가 /tmp/biped_cmd.json 발행 → 이 프로세스가 소비(mature quad CMDFILE 방식).
상태는 /tmp/biped_state.json 발행(GUI 오버레이용). 낙상 시 자동 리셋.
실행: python biped_run.py        (뷰어 ON)  ·  헤드리스: VIEW=0 python biped_run.py
명령 포맷: {"v":전진m/s, "vy":좌우, "w":선회, "body_h":몸통높이m, "mode":"stand"|"walk"|"off"|"reset"}
"""
import os, time, json, numpy as np, mujoco, mujoco.viewer
import biped_mpc_wbic as BM
from biped_wbic import base_rpy

CMD   = os.environ.get('QUAD_CMD',   '/tmp/biped_cmd.json')
STATE = os.environ.get('QUAD_STATE', '/tmp/biped_state.json')


def read_cmd():
    try:
        with open(CMD) as f:
            return json.load(f)
    except Exception:
        return None


# ★접촉 모드별 모델: 1pt=점발(biped_from_quad, 동적보행)·2pt=평발(biped_flatfoot, 정적 양발지지)
MODELS = {'1pt': None, '2pt': os.path.join(os.path.dirname(__file__), 'biped_flatfoot.mjcf')}


def build(contact):
    mjcf = MODELS.get(contact) or os.environ.get('BIPED_MJCF')
    c = BM.BipedMPCWBIC(mjcf=mjcf) if mjcf else BM.BipedMPCWBIC()
    c.reset(); c.setup_mpc()
    tag = '2점 평발(정적지지)' if c.flatfoot else ('2점' if c.two_contact else '1점 점발')
    print(f"[biped_run] 접촉모드={contact} → {tag} · 발당 접촉구 {len(c.foot_spheres[0])}개 · 기본높이 {c.com_ref[2]:.3f}")
    return c


def main():
    contact = os.environ.get('CONTACT', '1pt')     # 시작 접촉모드
    c = build(contact)
    m, d = c.m, c.d; dt = m.opt.timestep
    z_home = float(c.com_ref[2])
    mode, body_h, prev_mode = 'stand', z_home, 'stand'
    view = os.environ.get('VIEW', '1') != '0'
    viewer = mujoco.viewer.launch_passive(m, d) if view else None
    rt = os.environ.get('RT', '1') != '0'                  # 실시간 페이싱(뷰어 자연스러움·테스트 정확)
    print(f"biped_run · CMD={CMD} · 뷰어={'ON' if view else 'OFF'} · RT={'ON' if rt else 'OFF'} · 기본높이 {z_home:.3f}")
    k = 0
    t_end = float(os.environ.get('T', '1e9'))
    t0 = time.perf_counter()
    while ((viewer is None) or viewer.is_running()) and k * dt < t_end:
        if k % 20 == 0:                                    # 명령 폴링(50Hz)
            cmd = read_cmd()
            if cmd:
                if cmd.get('contact', contact) != contact:      # ★접촉모드 전환(1pt↔2pt): 컨트롤러+뷰어 재생성
                    contact = cmd.get('contact')
                    if viewer is not None: viewer.close()
                    c = build(contact); m, d = c.m, c.d
                    z_home = float(c.com_ref[2]); body_h = z_home
                    viewer = mujoco.viewer.launch_passive(m, d) if view else None
                    mode = 'stand'; prev_mode = 'stand'; t0 = time.perf_counter(); k = 0
                mode = cmd.get('mode', mode)
                body_h = float(cmd.get('body_h', body_h))
                if mode == 'reset':                        # ★RESET: 초기 자세로 리셋(모터 재활성)
                    c.reset(); c.setup_mpc(); c.com_ref[2] = body_h; c._k = 0
                    mode = 'stand'
                if prev_mode == 'off' and mode != 'off':    # ★전원 재투입(Off→Stand/Walk): 낙상 자세서 리셋 후 기립
                    c.reset(); c.setup_mpc(); c.com_ref[2] = body_h; c._k = 0
                prev_mode = mode
                walking = mode == 'walk'
                c.vx_cmd = float(cmd.get('v', 0.0))  if walking else 0.0
                c.wz_cmd = float(cmd.get('w', 0.0))  if walking else 0.0   # ★선회
                c.vy_cmd = float(cmd.get('vy', 0.0)) if walking else 0.0   # ★좌우
                c.com_ref[2] = body_h                      # 몸통높이 라이브 조절(crouch)
        if mode == 'off':                                  # ★모터 전원 off = 토크 0(limp). 실HW=motor disable
            d.ctrl[:] = 0.0
        else:
            c.control(dt)
        mujoco.mj_step(m, d); k += 1
        tilt = np.hypot(*base_rpy(d.qpos[3:7])[:2])
        if mode != 'off' and (d.qpos[2] < 0.2 or tilt > 45):   # 낙상 → 자동 리셋 (off는 전원차단이라 리셋 안함)
            if viewer is not None: time.sleep(0.3)
            c.reset(); c.setup_mpc(); c.com_ref[2] = body_h; c._k = 0
        if k % 20 == 0:                                    # 상태 발행
            vx_act = float((c.Jc_cache @ d.qvel)[0]) if hasattr(c, 'Jc_cache') else 0.0
            yaw = float(base_rpy(d.qpos[3:7])[2])
            try:
                tmp = STATE + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump({'mode': mode, 'base_z': float(d.qpos[2]), 'vx_cmd': float(c.vx_cmd),
                               'vx_act': vx_act, 'wz_cmd': float(c.wz_cmd), 'yaw': yaw,
                               'tilt': float(tilt), 'x': float(d.qpos[0]), 'y': float(d.qpos[1])}, f)
                os.replace(tmp, STATE)
            except Exception:
                pass
        if viewer is not None and k % 8 == 0:
            viewer.sync()
        if rt:                                             # 실시간 페이싱
            lag = t0 + k * dt - time.perf_counter()
            if lag > 0:
                time.sleep(lag)


if __name__ == '__main__':
    main()
