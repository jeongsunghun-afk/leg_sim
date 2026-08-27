#!/usr/bin/env python3
"""실측 플랜트에서 현 파라미터 세트 검증 + T_STEP 재스윕.

사용자 지적(08-27): T_STEP=0.30 최적은 공칭 플랜트의 유산 — 실측 플랜트(α 0.85 ·
foot 결손 0.36 · 조립마찰)에서 ①현 세트로 stand/walk 가 정말 되는지 검증하고
②안정성 지도를 다시 그린 뒤 트립/q̇_cmd 를 확정한다.

PART A — 검증 프로토콜 (T_STEP 0.30 · 8/8 스타일): 정지 · 전진 0.05/0.10/0.15/0.20 ·
         후진 −0.10 · 측방 0.05 · 선회 0.2  (각 12s · 1점)
PART B — T_STEP 재스윕: 0.26~0.40 × vx 0.10, 낙상/tilt/calf 채널속도/kd 토크

플랜트: ALPHA_AXIS=0.85 · FOOT_FRIC_EXTRA=0.36 · FRIC_COMP=0(sim 규약)
"""
import os, sys, subprocess, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(HERE)

def run(t_step, vx, vy, wz, T=12.0):
    env = dict(os.environ, ALPHA_AXIS='0.85', FOOT_FRIC_EXTRA='0.36', FRIC_COMP='0',
               FOOT_COMP_NM='0', T_STEP=str(t_step))   # ★상속 오염 차단(08-27)
    code = f'''
import os, sys, json
sys.path.insert(0, {BIPED!r})
import numpy as np, mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy
c = BM.BipedMPCWBIC(mjcf=os.path.join({BIPED!r}, "biped_flatfoot.mjcf"))
c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
GEARK=[1.0,1.0,1.5,1.2]; KD_CH=[6.0,4.0,3.5,2.0]
tilts=[]; vmax=0.0; tkmax=0.0; fell=None; x0=d.qpos[0]
for k in range(int({T}/dt)):
    t=k*dt
    c.vx_cmd={vx} if t>2 else 0.0
    c.vy_cmd={vy} if t>2 else 0.0
    c.wz_cmd={wz} if t>2 else 0.0
    c.control(dt)
    mujoco.mj_step(m,d)
    tilt=float(np.hypot(*base_rpy(d.qpos[3:7])[:2])); tilts.append(tilt)
    if d.qpos[2]<0.2 or tilt>45: fell=t; break
    if t>2.5 and k%4==0:
        dq=np.rad2deg(d.qvel[6:14])
        for leg in range(2):
            b=4*leg
            ch=[abs(dq[b])*1.0,abs(dq[b+1])*1.0,abs(dq[b+2])*1.5,abs(dq[b+2]+dq[b+3])*1.2]
            for i,v in enumerate(ch):
                vmax=max(vmax,v); tkmax=max(tkmax, v*KD_CH[i]*np.pi/180)
print("RESULT "+json.dumps(dict(fell=fell,tilt_p95=float(np.percentile(tilts,95)),
      dist=float(d.qpos[0]-x0), vmax=vmax, tkmax=tkmax)))
'''
    r = subprocess.run([sys.executable,'-c',code], env=env, capture_output=True, text=True, timeout=1800)
    for line in r.stdout.splitlines():
        if line.startswith('RESULT '): return json.loads(line[7:])
    return dict(error=(r.stderr or r.stdout)[-250:])

if __name__ == '__main__':
    print("■ PART A — 현 세트 검증 (T_STEP 0.30 · 실측 플랜트 · 각 12s)")
    proto = [('정지',0,0,0), ('전진0.05',0.05,0,0), ('전진0.10',0.10,0,0),
             ('전진0.15',0.15,0,0), ('전진0.20',0.20,0,0), ('후진0.10',-0.10,0,0),
             ('측방0.05',0,0.05,0), ('선회0.2',0,0,0.2)]
    ok=0
    for name,vx,vy,wz in proto:
        r=run(0.30,vx,vy,wz)
        if 'error' in r: print(f"  {name:8s} ⚠ {r['error'][:120]}"); continue
        good = r['fell'] is None
        ok += good
        if good:
            print(f"  {name:8s} ✅  tilt_p95 {r['tilt_p95']:.1f}°  전진 {r['dist']:+.2f} m")
        else:
            print(f"  {name:8s} ❌낙상 t={r['fell']:.1f}s")
    print(f"  → {ok}/{len(proto)} 무낙상")

    print("\n■ PART B — T_STEP 재스윕 (vx 0.10 · 실측 플랜트)")
    print(f"  {'T':5s} {'결과':16s} {'tilt_p95':>8s} {'ch속도max':>9s} {'kd토크max':>9s}")
    for ts in (0.26,0.28,0.30,0.32,0.34,0.36,0.40):
        r=run(ts,0.10,0,0)
        if 'error' in r: print(f"  {ts:.2f} ⚠ {r['error'][:100]}"); continue
        res = '✅완주' if r['fell'] is None else f"❌낙상 {r['fell']:.1f}s"
        print(f"  {ts:.2f}  {res:14s} {r['tilt_p95']:8.1f} {r['vmax']:9.0f} {r['tkmax']:9.1f}")
