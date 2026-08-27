#!/usr/bin/env python3
"""실측 전달비를 주입한 sim2sim 검토 — 2점 stand · walk 가 약해진 플랜트에서 버티는가.

플랜트 주입은 ALPHA_AXIS(biped_wbic.setup_gearbox → actuator_gear) 로 하고,
제어기는 α 를 모른다(실기와 동일 — 보상 없는 최악조건).

시나리오 (환경변수 SCEN 로 하나만 실행, 기본 전부):
  stand2_base   2점 stand · α=1.0            (기준 — 8/8 시절 파리티)
  stand2_meas   2점 stand · α=0.80 공통       (평발 실측 플랜트)
  walk2_base    2점 walk vx0.10 · α=1.0
  walk2_meas    2점 walk vx0.10 · α=0.80
  walk1_base    1점 walk vx0.10 · α=1.0
  walk1_hypo    1점 walk vx0.10 · α=[0.80,0.80,0.80,0.45]  (Qhome8 가설: foot 발목각 손실)

판정: 낙상 시각 · tilt p95 · 전진거리(명령 대비) · 자동리셋 없음(첫 낙상에서 종료)
"""
import os, sys, subprocess, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(HERE)

SCENARIOS = {
    'stand2_base': dict(contact='2pt', alpha='1.0', vx=0.0, T=10),
    'stand2_meas': dict(contact='2pt', alpha='0.80', vx=0.0, T=10),
    'walk2_base':  dict(contact='2pt', alpha='1.0', vx=0.10, T=15),
    'walk2_meas':  dict(contact='2pt', alpha='0.80', vx=0.10, T=15),
    'walk1_base':  dict(contact='1pt', alpha='1.0', vx=0.10, T=15),
    'walk1_meas':  dict(contact='1pt', alpha='0.80', vx=0.10, T=15),
    'walk1_hypo':  dict(contact='1pt', alpha='0.80,0.80,0.80,0.45', vx=0.10, T=15),
}

def run_one(name, sc):
    """자식 프로세스에서 한 시나리오 실행 (ALPHA_AXIS 는 import 시점 주입이라 프로세스 분리)."""
    # ★FRIC_COMP=0 — 실기용 마찰 전방보상이 sim 에선 림보사이클을 만든다
    #   (08-27 실증: 기본 1.0 이면 기준 stand 가 8s 주기 낙상-리셋, 0 이면 tilt 0.1°)
    env = dict(os.environ, ALPHA_AXIS=sc['alpha'], FRIC_COMP='0')
    code = f'''
import os, sys, json
sys.path.insert(0, {BIPED!r})
import numpy as np, mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy
c = BM.BipedMPCWBIC(mjcf=os.path.join({BIPED!r}, "biped_flatfoot.mjcf"))
c.set_contact_mode({sc['contact']!r}); c.reset(); c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
T, VX = {sc['T']}, {sc['vx']}
tilts, x0 = [], d.qpos[0]
fell = None
for k in range(int(T/dt)):
    t = k*dt
    c.vx_cmd = VX if t > 2.0 else 0.0     # 2s 정착 후 보행 명령
    c.wz_cmd = c.vy_cmd = 0.0
    c.control(dt)
    mujoco.mj_step(m, d)
    tilt = float(np.hypot(*base_rpy(d.qpos[3:7])[:2]))
    tilts.append(tilt)
    if d.qpos[2] < 0.2 or tilt > 45:
        fell = t; break
out = dict(fell=fell, T=T, tilt_p95=float(np.percentile(tilts, 95)),
           dist=float(d.qpos[0]-x0), z=float(d.qpos[2]))
print("RESULT " + json.dumps(out))
'''
    r = subprocess.run([sys.executable, '-c', code], env=env,
                       capture_output=True, text=True, timeout=1800)
    for line in r.stdout.splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[7:])
    return dict(error=(r.stderr or r.stdout)[-400:])

if __name__ == '__main__':
    only = os.environ.get('SCEN')
    names = [only] if only else list(SCENARIOS)
    print(f"{'시나리오':14s} {'α':22s} {'결과':44s}")
    for n in names:
        sc = SCENARIOS[n]
        res = run_one(n, sc)
        if 'error' in res:
            print(f"{n:14s} {sc['alpha']:22s} ⚠에러: {res['error'][:120]}")
            continue
        exp = sc['vx'] * max(0, sc['T'] - 2.0)
        verdict = (f"❌낙상 t={res['fell']:.2f}s" if res['fell'] is not None else
                   f"✅완주 tilt_p95 {res['tilt_p95']:.1f}° · 전진 {res['dist']:+.2f} m"
                   + (f" (명령 {exp:.2f})" if sc['vx'] else ''))
        print(f"{n:14s} {sc['alpha']:22s} {verdict}")
