import os, sys, subprocess
BIPED='/home/jsh/simulation/biped'
code = '''
import os, sys
sys.path.insert(0, "/home/jsh/simulation/biped")
import numpy as np, mujoco
import biped_mpc_wbic as BM
c = BM.BipedMPCWBIC(mjcf="/home/jsh/simulation/biped/biped_flatfoot.mjcf")
c.set_contact_mode("1pt"); c.reset(); c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
swaps=0; last=c.stance
for k in range(int(6.0/dt)):
    t=k*dt
    c.vx_cmd = 0.10 if t>2 else 0.0
    c.control(dt); mujoco.mj_step(m,d)
    if c.stance!=last: swaps+=1; last=c.stance
    if d.qpos[2]<0.2: print("FELL", t); break
print("RESULT swaps=", swaps, " T_STEP=", os.environ.get("T_STEP"))
'''
for ts in ('0.26','0.40'):
    env=dict(os.environ, ALPHA_AXIS='0.85', FOOT_FRIC_EXTRA='0.36', FRIC_COMP='0', T_STEP=ts)
    r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=900)
    out=[l for l in r.stdout.splitlines() if 'RESULT' in l or 'FELL' in l]
    print(ts, out, r.stderr[-200:] if not out else '')
