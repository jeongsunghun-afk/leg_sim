#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
컨트롤러 비교 뷰어 (GUI). MuJoCo 뷰어에서 **A 제어기 vs CI-MPC**를 같은 동작으로 나란히 비교.
  동작:  1=walk(보행) · 2=crouch(웅크리기) · 3=sit(앉기) · 4=lie(눕기)
  제어기: A=A 제어기(WBIC+MPC) · C=CI-MPC · SPACE=정지 · R=리셋

각 (제어기, 동작) = qpos 궤적(.npy). CI-MPC는 ci_mpc_walk.py가, A는 A 컨트롤러가 QPOS_OUT로 저장.
  CI-MPC: walk.npy · crouch.npy · sit.npy · lie.npy   (env HARD=1 [POSE_Z=..] QPOS_OUT=.. python ci_mpc_walk.py)
  A 제어기: a_walk.npy · a_crouch.npy · a_sit.npy · a_lie.npy  (A 컨트롤러 실행 시 저장 — 없으면 CI만 표시)
실행: /home/jsh/miniforge3/envs/proxddp/bin/python ci_mpc_viewer.py
"""
import numpy as np, mujoco, mujoco.viewer, os, time
from model_bridge import MJCF, apply_gearbox, set_foot_sphere, strip_mesh_collision

HERE = os.path.dirname(os.path.abspath(__file__))
MOTIONS = {49:'walk', 50:'crouch', 51:'sit', 52:'lie'}       # 키 1~4 → 동작
CTRL    = {'CI':'', 'A':'a_'}                                 # 제어기 → 파일 prefix
DT_SIM  = float(os.environ.get("DT_SIM","0.02"))
SPEED   = float(os.environ.get("SPEED","0.5"))

def load_all():
    traj={}
    for ctrl,pre in CTRL.items():
        for _,motion in MOTIONS.items():
            p=os.path.join(HERE, "%s%s.npy"%(pre,motion))
            if os.path.exists(p): traj[(ctrl,motion)]=np.load(p)
    return traj

def main():
    mm = mujoco.MjModel.from_xml_path(MJCF)
    apply_gearbox(mm); set_foot_sphere(mm, 0.025); strip_mesh_collision(mm)
    md = mujoco.MjData(mm)
    traj=load_all()
    if not traj: print("재생할 .npy 없음"); return
    print("=== 컨트롤러 비교 뷰어 ===")
    for ctrl in CTRL:
        have=[m for _,m in MOTIONS.items() if (ctrl,m) in traj]
        print("  %s 제어기: %s"%(ctrl, ", ".join(have) if have else "(없음)"))
    ctrl=['CI' if any(k[0]=='CI' for k in traj) else 'A']; motion=['walk']; frame=[0]; paused=[False]

    def cur_key(): return (ctrl[0], motion[0])
    def key_cb(keycode):
        if keycode in MOTIONS: motion[0]=MOTIONS[keycode]; frame[0]=0
        elif keycode in (65,97): ctrl[0]='A'; frame[0]=0                 # A/a
        elif keycode in (67,99): ctrl[0]='CI'; frame[0]=0               # C/c
        elif keycode==32: paused[0]=not paused[0]
        elif keycode in (82,114): frame[0]=0
        if cur_key() in traj: print("▶ [%s] %s"%(ctrl[0],motion[0]))
        else: print("✗ [%s] %s 궤적 없음"%(ctrl[0],motion[0]))

    with mujoco.viewer.launch_passive(mm, md, key_callback=key_cb) as v:
        print("\n동작 1=walk 2=crouch 3=sit 4=lie · 제어기 A / C(CI) · SPACE=정지 R=리셋\n")
        while v.is_running():
            k=cur_key()
            if k in traj:
                Q=traj[k]; md.qpos[:]=Q[min(frame[0],len(Q)-1)]; mujoco.mj_forward(mm,md)
                if not paused[0]:
                    frame[0]+=1
                    if frame[0]>=len(Q): frame[0]=len(Q)-1              # 끝에서 홀드
            v.sync(); time.sleep(DT_SIM/max(SPEED,1e-3))

if __name__=="__main__":
    main()
