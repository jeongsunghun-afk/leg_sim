import os, sys, subprocess, json
from concurrent.futures import ThreadPoolExecutor
BIPED='/home/jsh/simulation/biped'
def run(args):
    h, vx = args
    env=dict(os.environ, ALPHA_AXIS='0.85', FOOT_FRIC_EXTRA='0.36', FRIC_COMP='0',
             FOOT_COMP_NM='0', T_STEP='0.30')
    code=f'''
import os,sys,json
sys.path.insert(0,{BIPED!r})
import numpy as np, mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy
c=BM.BipedMPCWBIC(mjcf=os.path.join({BIPED!r},"biped_flatfoot.mjcf"))
c.CZ_1PT={h}
c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
m,d=c.m,c.d; dt=m.opt.timestep; fell=None; tilts=[]; kn=[]; x0=d.qpos[0]
for k in range(int(14.0/dt)):
    t=k*dt
    c.vx_cmd={vx} if t>2 else 0.0; c.vy_cmd=c.wz_cmd=0.0
    c.control(dt); mujoco.mj_step(m,d)
    tilt=float(np.hypot(*base_rpy(d.qpos[3:7])[:2])); tilts.append(tilt)
    if t>2.5 and k%4==0: kn.append(max(abs(d.ctrl[2]),abs(d.ctrl[6])))
    if d.qpos[2]<0.15 or tilt>45: fell=t; break
print("RESULT "+json.dumps(dict(fell=fell,tilt_p95=float(np.percentile(tilts,95)),
      dist=float(d.qpos[0]-x0), knee_p95=float(np.percentile(kn,95)) if kn else 0)))
'''
    r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=1700)
    for line in r.stdout.splitlines():
        if line.startswith('RESULT '): return h,vx,json.loads(line[7:])
    return h,vx,dict(error=(r.stderr or r.stdout)[-150:])
grid=[(h,vx) for h in (0.44,0.46,0.48,0.50,0.52) for vx in (0.10,0.30,0.35)]
with ThreadPoolExecutor(4) as ex:
    res=list(ex.map(run,grid))
print(f"{'h[m]':5s} {'vx':5s} {'결과':14s} {'tilt_p95':>8s} {'전진m':>6s} {'무릎p95Nm':>9s}")
for h,vx,r in res:
    if 'error' in r: print(f"{h:.2f} {vx:.2f} ⚠ {r['error'][:100]}"); continue
    s='✅완주' if r['fell'] is None else f"❌낙상 {r['fell']:.1f}s"
    print(f"{h:.2f} {vx:.2f}  {s:12s} {r['tilt_p95']:8.1f} {r['dist']:6.2f} {r['knee_p95']:9.1f}")
