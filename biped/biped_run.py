"""biped 실행기 — MPC+WBIC 컨트롤러 + 뷰어 + JSON 명령채널.

GUI(teleop_gui_biped.py)가 /tmp/biped_cmd.json 발행 → 이 프로세스가 소비(mature quad CMDFILE 방식).
상태는 /tmp/biped_state.json 발행(GUI 오버레이용). 낙상 시 자동 리셋.
실행: python biped_run.py        (뷰어 ON)  ·  헤드리스: VIEW=0 python biped_run.py
명령 포맷: {"v":전진m/s, "body_h":몸통높이m, "mode":"stand"|"walk"|"stop"}
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


def main():
    c = BM.BipedMPCWBIC(); c.reset(); c.setup_mpc()
    m, d = c.m, c.d; dt = m.opt.timestep
    z_home = float(c.com_ref[2])
    mode, body_h = 'stand', z_home
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
                mode = cmd.get('mode', mode)
                body_h = float(cmd.get('body_h', body_h))
                if mode == 'reset':                        # ★RESET: 초기 자세로 리셋
                    c.reset(); c.setup_mpc(); c.com_ref[2] = body_h; c._k = 0
                    mode = 'stand'
                walking = mode == 'walk'
                c.vx_cmd = float(cmd.get('v', 0.0))  if walking else 0.0
                c.wz_cmd = float(cmd.get('w', 0.0))  if walking else 0.0   # ★선회
                c.vy_cmd = float(cmd.get('vy', 0.0)) if walking else 0.0   # ★좌우
                c.com_ref[2] = body_h                      # 몸통높이 라이브 조절(crouch)
        c.control(dt); mujoco.mj_step(m, d); k += 1
        tilt = np.hypot(*base_rpy(d.qpos[3:7])[:2])
        if d.qpos[2] < 0.2 or tilt > 45:                   # 낙상 → 자동 리셋
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
