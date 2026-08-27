#!/usr/bin/env python3
"""walk(T_STEP 0.30)가 요구하는 채널속도·kd 감쇠토크 정량화 → 트립 설정 제안.

배경: T_STEP 은 안정성 스윕으로 0.30 고정(0.34+ 낙상 — biped_step.py 주석).
따라서 "스윙 감쇠토크 2.1×트립" 문제의 해법은 T_STEP 이 아니라
①vel/tau 트립 상향 ②q̇_cmd 전송(감쇠토크 제거)이고, 이 도구가 그 설정치를 잰다.

측정(실측 플랜트: α 0.85 + foot 마찰 +0.36 · FRIC_COMP=0):
  · 채널 속도 |dq_ch| p95/max — 채널 = 관절×gear_k · 발목은 (q̇f+q̇c)×1.2 합산
  · kd 감쇠토크 |kd_ch·dq_ch| p95/max — dq_des=0 유지 시 드라이버가 낼 제동토크
  → vel_trip(현 200dps)·tau_trip(15Nm) 대비 여유/초과 표 + 제안치
"""
import os, sys, subprocess, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(HERE)

def run_walk(vx, T=12.0):
    env = dict(os.environ, ALPHA_AXIS='0.85', FOOT_FRIC_EXTRA='0.36', FRIC_COMP='0',
               FOOT_COMP_NM='0', T_STEP='0.30')   # ★상속 오염 차단(08-27)
    code = f'''
import os, sys, json
sys.path.insert(0, {BIPED!r})
import numpy as np, mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy
c = BM.BipedMPCWBIC(mjcf=os.path.join({BIPED!r}, "biped_flatfoot.mjcf"))
c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
GEARK = [1.0, 1.0, 1.5, 1.2]
KD_CH = [6.0, 4.0, 3.5, 2.0]
rec = []
fell = None
for k in range(int({T}/dt)):
    t = k*dt
    c.vx_cmd = {vx} if t > 2.0 else 0.0
    c.wz_cmd = c.vy_cmd = 0.0
    c.control(dt)
    mujoco.mj_step(m, d)
    tilt = float(np.hypot(*base_rpy(d.qpos[3:7])[:2]))
    if d.qpos[2] < 0.2 or tilt > 45: fell = t; break
    if t > 2.5 and k % 4 == 0:                      # 보행 정상부만 샘플
        dq = np.rad2deg(d.qvel[6:14])               # 관절 dps
        row = []
        for leg in range(2):
            b = 4*leg
            ch = [dq[b+0]*GEARK[0], dq[b+1]*GEARK[1], dq[b+2]*GEARK[2],
                  (dq[b+2]+dq[b+3])*GEARK[3]]       # 발목 채널 = (calf+foot)×1.2
            row += ch
        rec.append(row)
rec = np.array(rec) if rec else np.zeros((1,8))
kdch = np.array(KD_CH*2)
tau_kd = np.abs(rec) * kdch * np.pi/180.0           # kd·dq_ch [Nm] (rad/s 환산)
out = dict(fell=fell,
           v_p95=[float(v) for v in np.percentile(np.abs(rec), 95, axis=0)],
           v_max=[float(v) for v in np.abs(rec).max(axis=0)],
           t_p95=[float(v) for v in np.percentile(tau_kd, 95, axis=0)],
           t_max=[float(v) for v in tau_kd.max(axis=0)])
print("RESULT " + json.dumps(out))
'''
    r = subprocess.run([sys.executable, '-c', code], env=env,
                       capture_output=True, text=True, timeout=1800)
    for line in r.stdout.splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[7:])
    return dict(error=(r.stderr or r.stdout)[-300:])

if __name__ == '__main__':
    CH = ['HLhip','HLth','HLcalf','HLfoot','HRhip','HRth','HRcalf','HRfoot']
    for vx in (0.10, 0.20):
        res = run_walk(vx)
        if 'error' in res:
            print(f"vx={vx}: ⚠ {res['error'][:200]}"); continue
        print(f"\n■ 1점 walk vx={vx} (실측 플랜트 · T_STEP 0.30)"
              + (f"  ❌낙상 t={res['fell']:.2f}s" if res['fell'] else "  ✅완주"))
        print(f"  {'채널':7s} {'|dq|p95':>8s} {'|dq|max':>8s}  vs 트립200   {'kd·dq p95':>9s} {'max':>6s}  vs 트립15Nm")
        for i, n in enumerate(CH):
            vp, vm = res['v_p95'][i], res['v_max'][i]
            tp, tm = res['t_p95'][i], res['t_max'][i]
            print(f"  {n:7s} {vp:8.0f} {vm:8.0f}  {'⚠초과' if vm>200 else 'ok  '}      "
                  f"{tp:9.1f} {tm:6.1f}  {'⚠초과' if tm>15 else 'ok'}")
        print(f"  제안: vel_trip ≥ {1.3*max(res['v_max']):.0f} dps (max×1.3) · "
              f"q̇_cmd 전송 시 kd 감쇠토크 {max(res['t_max']):.1f} Nm → ~0")
