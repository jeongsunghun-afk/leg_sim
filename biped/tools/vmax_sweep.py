import os, sys, subprocess, json
BIPED='/home/jsh/simulation/biped'
def run(vx):
    env=dict(os.environ, ALPHA_AXIS='0.85', FOOT_FRIC_EXTRA='0.36', FRIC_COMP='0',
             FOOT_COMP_NM='0', T_STEP='0.30')
    code=f'''
import os,sys,json
sys.path.insert(0,{BIPED!r})
import numpy as np, mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy
c=BM.BipedMPCWBIC(mjcf=os.path.join({BIPED!r},"biped_flatfoot.mjcf"))
c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
m,d=c.m,c.d; dt=m.opt.timestep; fell=None; tilts=[]; x0=d.qpos[0]
for k in range(int(14.0/dt)):
    t=k*dt
    c.vx_cmd={vx} if t>2 else 0.0; c.vy_cmd=c.wz_cmd=0.0
    c.control(dt); mujoco.mj_step(m,d)
    tilt=float(np.hypot(*base_rpy(d.qpos[3:7])[:2])); tilts.append(tilt)
    if d.qpos[2]<0.2 or tilt>45: fell=t; break
print("RESULT "+json.dumps(dict(fell=fell,tilt_p95=float(np.percentile(tilts,95)),dist=float(d.qpos[0]-x0))))
'''
    r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=1800)
    for line in r.stdout.splitlines():
        if line.startswith('RESULT '): return json.loads(line[7:])
    return dict(error=(r.stderr or r.stdout)[-200:])
for vx in (0.20,0.25,0.30,0.35,0.40):
    r=run(vx)
    if 'error' in r: print(f"vx={vx:.2f} ⚠ {r['error'][:150]}"); continue
    print(f"vx={vx:.2f}  {'✅완주' if r['fell'] is None else '❌낙상 t=%.1fs'%r['fell']}"
          f"  tilt_p95 {r['tilt_p95']:.1f}°  전진 {r['dist']:+.2f} m", flush=True)
