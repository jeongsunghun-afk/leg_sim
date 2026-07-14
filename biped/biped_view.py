"""biped 뷰어 — 낙상 시 자동 리셋 반복(계속 관찰). VX=0.3 등으로 보행."""
import os, time, numpy as np, mujoco, mujoco.viewer
import biped_mpc_wbic as BM
from biped_wbic import base_rpy

c = BM.BipedMPCWBIC(); c.reset(); c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
vx = c.vx_cmd
print(f"biped 뷰어 · vx={vx} · 낙상 시 자동 리셋. 창 닫으면 종료.")
with mujoco.viewer.launch_passive(m, d) as v:
    step = 0
    while v.is_running():
        c.control(dt); mujoco.mj_step(m, d); step += 1
        if step % 8 == 0:
            v.sync()
        tilt = np.hypot(*base_rpy(d.qpos[3:7])[:2])
        if d.qpos[2] < 0.25 or tilt > 40:          # 낙상 → 리셋
            time.sleep(0.4)
            c.reset(); c.vx_cmd = vx; c.setup_mpc(); c._k = 0
        time.sleep(dt)
