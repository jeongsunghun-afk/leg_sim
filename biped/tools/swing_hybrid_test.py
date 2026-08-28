import os, sys, subprocess, json
from concurrent.futures import ThreadPoolExecutor
BIPED='/home/jsh/simulation/biped'
def run(args):
    kp, kd, vx = args
    env=dict(os.environ, ALPHA_AXIS='0.85', FOOT_FRIC_EXTRA='0.36', FRIC_COMP='0',
             FOOT_COMP_NM='0', T_STEP='0.30', SWING_KP=str(kp), SWING_KD=str(kd))
    code=f'''
import os,sys,json
sys.path.insert(0,{BIPED!r})
import numpy as np, mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy
c=BM.BipedMPCWBIC(mjcf=os.path.join({BIPED!r},"biped_flatfoot.mjcf"))
c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
m,d=c.m,c.d; dt=m.opt.timestep; fell=None; tilts=[]; x0=d.qpos[0]; kdmax=0.0; vmax=0.0
KD_CH=[6.0,4.0,3.5,2.0]; GEARK=[1.0,1.0,1.5,1.2]
for k in range(int(14.0/dt)):
    t=k*dt
    c.vx_cmd={vx} if t>2 else 0.0; c.vy_cmd=c.wz_cmd=0.0
    c.control(dt); mujoco.mj_step(m,d)
    tilt=float(np.hypot(*base_rpy(d.qpos[3:7])[:2])); tilts.append(tilt)
    if t>2.5 and k%4==0:
        dq=np.rad2deg(d.qvel[6:14])
        for leg in range(2):
            b=4*leg
            ch=[abs(dq[b])*1.0,abs(dq[b+1])*1.0,abs(dq[b+2])*1.5,abs(dq[b+2]+dq[b+3])*1.2]
            for i,v in enumerate(ch):
                vmax=max(vmax,v); kdmax=max(kdmax, v*KD_CH[i]*np.pi/180)
    if d.qpos[2]<0.15 or tilt>45: fell=t; break
print("RESULT "+json.dumps(dict(fell=fell,tilt_p95=float(np.percentile(tilts,95)),
      dist=float(d.qpos[0]-x0),vmax=vmax,kdmax=kdmax)))
'''
    r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=1700)
    for line in r.stdout.splitlines():
        if line.startswith('RESULT '): return kp,kd,vx,json.loads(line[7:])
    return kp,kd,vx,dict(error=(r.stderr or r.stdout)[-160:])
grid=[(0,0,0.10),(0,0,0.30),(20,1.0,0.10),(20,1.0,0.30),(50,2.0,0.10),(50,2.0,0.30),(100,4.0,0.30)]
with ThreadPoolExecutor(4) as ex: res=list(ex.map(run,grid))
print(f"{'SWING_KP':>9s}{'KD':>5s}{'vx':>6s}  {'결과':14s}{'tilt_p95':>9s}{'전진m':>7s}{'ch속도max':>10s}{'kd토크max':>10s}")
for kp,kd,vx,r in res:
    if 'error' in r: print(f"{kp:9.0f}{kd:5.1f}{vx:6.2f}  ⚠ {r['error'][:90]}"); continue
    st='✅완주' if r['fell'] is None else f"❌낙상{r['fell']:.1f}s"
    print(f"{kp:9.0f}{kd:5.1f}{vx:6.2f}  {st:14s}{r['tilt_p95']:9.1f}{r['dist']:7.2f}{r['vmax']:10.0f}{r['kdmax']:10.1f}")
