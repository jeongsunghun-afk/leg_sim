#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CI-MPC / A 제어기 비교 GUI + 뷰어 (단일 프로세스).
  Tkinter 버튼 창 + MuJoCo 뷰어를 같이 띄우고, 버튼을 누르면 뷰어가 그 모션을 replay.
  제어기: [CI-MPC] [A 제어기]  ·  동작: [Walk][Crouch][Sit][Lie]  ·  [일시정지][리셋]

각 (제어기,동작)=qpos 궤적(.npy). CI-MPC=ci_mpc_walk.py(QPOS_OUT), A=quad_mpc_wbic_17dof(QPOS_OUT).
실행: /home/jsh/miniforge3/envs/proxddp/bin/python ci_mpc_gui.py
"""
import numpy as np, mujoco, mujoco.viewer, os, time, threading
import tkinter as tk
from model_bridge import MJCF, apply_gearbox, set_foot_sphere, strip_mesh_collision

HERE = os.path.dirname(os.path.abspath(__file__))
MOTIONS = ['walk','crouch','sit','lie']
CTRLS   = {'CI-MPC':'', 'A 제어기':'a_'}
DT_SIM  = float(os.environ.get("DT_SIM","0.02"))
SPEED   = float(os.environ.get("SPEED","0.5"))

sel = {'ctrl':'CI-MPC', 'motion':'walk', 'paused':False, 'reset':False, 'quit':False}
lock = threading.Lock()

def load_traj():
    traj={}
    for cname,pre in CTRLS.items():
        for m in MOTIONS:
            p=os.path.join(HERE, "%s%s.npy"%(pre,m))
            if os.path.exists(p): traj[(cname,m)]=np.load(p)
    return traj

def gui_thread(traj):
    root=tk.Tk(); root.title("CI-MPC / A 제어기 비교"); root.geometry("360x320")
    status=tk.StringVar(value="▶ CI-MPC · walk")
    def refresh():
        with lock: status.set("▶ %s · %s%s"%(sel['ctrl'],sel['motion']," (없음)" if (sel['ctrl'],sel['motion']) not in traj else ""))
    def set_motion(m):
        with lock: sel['motion']=m; sel['reset']=True
        refresh()
    def set_ctrl(c):
        with lock: sel['ctrl']=c; sel['reset']=True
        refresh()
    def toggle_pause():
        with lock: sel['paused']=not sel['paused']
    def do_reset():
        with lock: sel['reset']=True

    tk.Label(root, text="제어기", font=("",11,"bold")).pack(pady=(12,2))
    fc=tk.Frame(root); fc.pack()
    for c in CTRLS: tk.Button(fc, text=c, width=12, command=lambda c=c:set_ctrl(c)).pack(side=tk.LEFT, padx=4)
    tk.Label(root, text="동작", font=("",11,"bold")).pack(pady=(14,2))
    fm=tk.Frame(root); fm.pack()
    names={'walk':'Walk 보행','crouch':'Crouch 웅크리기','sit':'Sit 앉기','lie':'Lie 눕기'}
    for i,m in enumerate(MOTIONS):
        tk.Button(fm, text=names[m], width=14, command=lambda m=m:set_motion(m)).grid(row=i//2,column=i%2,padx=4,pady=4)
    ff=tk.Frame(root); ff.pack(pady=14)
    tk.Button(ff, text="일시정지/재생", width=12, command=toggle_pause).pack(side=tk.LEFT,padx=4)
    tk.Button(ff, text="리셋", width=8, command=do_reset).pack(side=tk.LEFT,padx=4)
    tk.Label(root, textvariable=status, font=("",11), fg="#1a7").pack(pady=8)
    avail=", ".join(sorted(set(c for c,_ in traj)))
    tk.Label(root, text="로드된 제어기: %s"%(avail or "없음"), font=("",9), fg="#888").pack()
    def on_close():
        with lock: sel['quit']=True
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

def main():
    traj=load_traj()
    if not traj: print("재생할 .npy 없음 — 먼저 생성하세요"); return
    print("로드:", ", ".join("%s/%s"%(c,m) for c,m in sorted(traj)))
    mm=mujoco.MjModel.from_xml_path(MJCF)
    apply_gearbox(mm); set_foot_sphere(mm,0.025); strip_mesh_collision(mm)
    md=mujoco.MjData(mm)
    threading.Thread(target=gui_thread, args=(traj,), daemon=True).start()
    frame=0; cur=None
    with mujoco.viewer.launch_passive(mm, md) as v:
        while v.is_running():
            with lock:
                if sel['quit']: break
                key=(sel['ctrl'],sel['motion']); paused=sel['paused']
                if sel['reset']: frame=0; sel['reset']=False
            if key!=cur: cur=key; frame=0
            if key in traj:
                Q=traj[key]; md.qpos[:]=Q[min(frame,len(Q)-1)]; mujoco.mj_forward(mm,md)
                if not paused:
                    frame+=1
                    if frame>=len(Q): frame=len(Q)-1               # 끝에서 홀드
            v.sync(); time.sleep(DT_SIM/max(SPEED,1e-3))

if __name__=="__main__":
    main()
