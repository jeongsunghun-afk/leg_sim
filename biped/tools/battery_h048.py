import os, sys, subprocess, json
from concurrent.futures import ThreadPoolExecutor
BIPED='/home/jsh/simulation/biped'
def run(args):
    name,vx,vy,wz = args
    env=dict(os.environ, ALPHA_AXIS='0.85', FOOT_FRIC_EXTRA='0.36', FRIC_COMP='0',
             FOOT_COMP_NM='0', T_STEP='0.30')
    code=f'''
import os,sys,json
sys.path.insert(0,{BIPED!r})
import numpy as np, mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy
c=BM.BipedMPCWBIC(mjcf=os.path.join({BIPED!r},"biped_flatfoot.mjcf"))
c.CZ_1PT=0.48
c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
m,d=c.m,c.d; dt=m.opt.timestep; fell=None; tilts=[]; x0=d.qpos[0]
for k in range(int(12.0/dt)):
    t=k*dt
    c.vx_cmd={vx} if t>2 else 0.0
    c.vy_cmd={vy} if t>2 else 0.0
    c.wz_cmd={wz} if t>2 else 0.0
    c.control(dt); mujoco.mj_step(m,d)
    tilt=float(np.hypot(*base_rpy(d.qpos[3:7])[:2])); tilts.append(tilt)
    if d.qpos[2]<0.15 or tilt>45: fell=t; break
print("RESULT "+json.dumps(dict(fell=fell,tilt_p95=float(np.percentile(tilts,95)),dist=float(d.qpos[0]-x0))))
'''
    r=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=1700)
    for line in r.stdout.splitlines():
        if line.startswith('RESULT '): return name,json.loads(line[7:])
    return name,dict(error=(r.stderr or r.stdout)[-120:])
proto=[('정지',0,0,0),('전진0.05',0.05,0,0),('전진0.10',0.10,0,0),('전진0.20',0.20,0,0),
       ('전진0.30',0.30,0,0),('후진0.10',-0.10,0,0),('측방0.05',0,0.05,0),('선회0.2',0,0,0.2),
       ('전진0.40',0.40,0,0)]
with ThreadPoolExecutor(4) as ex:
    res=list(ex.map(run,proto))
ok=0
for name,r in res:
    if 'error' in r: print(f"{name:8s} ⚠ {r['error']}"); continue
    good=r['fell'] is None; ok+=good
    print(f"{name:8s} {'✅' if good else '❌낙상 %.1fs'%r['fell']}  tilt_p95 {r['tilt_p95']:.1f}°  전진 {r['dist']:+.2f} m")
print(f"→ {ok}/{len(proto)}")
