#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rollout qpos(.npy) 뷰어 replay. mjx_ilqr가 QPOS_OUT로 저장한 궤적을 실시간 재생.
gap 씬(DISABLE_FLOOR)서 발판만 남기고 floor 숨겨 진짜 void 표시.

실행: QPOS=crossing.npy MJCF_PATH=../mjcf/ci_mpc_gap.mjcf DISABLE_FLOOR=1 \
      /home/jsh/miniforge3/envs/proxddp/bin/python replay_viewer.py
"""
import numpy as np, mujoco, mujoco.viewer, os, time
from model_bridge import MJCF, apply_gearbox, set_foot_sphere, strip_mesh_collision

QPOS = os.environ.get("QPOS", "crossing.npy")
MJCF_PATH = os.environ.get("MJCF_PATH", MJCF)
DISABLE_FLOOR = os.environ.get("DISABLE_FLOOR", "0") == "1"
FOOT_R = float(os.environ.get("FOOT_R", "0.025"))
SPEED = float(os.environ.get("SPEED", "1.0"))          # 재생 배속
DT_SIM = float(os.environ.get("DT_SIM", "0.002"))

def main():
    mm = mujoco.MjModel.from_xml_path(MJCF_PATH)
    apply_gearbox(mm); set_foot_sphere(mm, FOOT_R); strip_mesh_collision(mm)
    if DISABLE_FLOOR:
        fg = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if fg >= 0: mm.geom_rgba[fg][3] = 0.0            # floor 투명(void 표시)
    md = mujoco.MjData(mm)
    Q = np.load(QPOS)
    print(f"[replay] {QPOS}: {Q.shape[0]} 프레임, x {Q[0,0]:+.2f}→{Q[-1,0]:+.2f} z {Q[:,2].min():.2f}~{Q[:,2].max():.2f}")

    with mujoco.viewer.launch_passive(mm, md) as v:
        v.cam.distance = 2.2; v.cam.azimuth = 90; v.cam.elevation = -12
        for i in range(Q.shape[0]):
            md.qpos[:] = Q[i]; md.qvel[:] = 0.0
            mujoco.mj_forward(mm, md)
            v.cam.lookat[0] = md.qpos[0]                 # 카메라 로봇 추적
            v.sync()
            if not v.is_running(): break
            time.sleep(DT_SIM / max(SPEED, 1e-3))
        print("[replay] 종료. 창 닫으면 끝.")
        while v.is_running():
            time.sleep(0.1)

if __name__ == "__main__":
    main()
