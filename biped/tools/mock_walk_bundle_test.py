#!/usr/bin/env python3
"""walk 묶음(모드 한정 트립 + WALK_KD_FLOOR) mock 검증.

시나리오 (각각 별도 프로세스, MockHw 고장주입):
  A) walk + FAULT_VEL_DPS=500 @6s  → 트립 없어야 함 (walk 900)
  B) stand + FAULT_VEL_DPS=500 @6s → 트립 (cfg 200, "(cfg)" 표기)
  C) walk + FAULT_TAU_NM=30 @6s    → 트립 (walk 25, "(walk)" 표기)
  D) walk + FAULT_VEL_DPS=500 @6s + WALK_VEL_TRIP_DPS=400 → 트립 (env 덮기)
공통 확인: 기동 배너 "walk 한정: 트립 900dps/25.0Nm · kd×0.15"
"""
import os, subprocess, time, json, sys, threading

CPP = '/home/jsh/simulation/biped/cpp'
BIN = CPP + '/build/biped_deploy'
SP = os.path.dirname(os.path.abspath(__file__))

def run_case(name, target_mode, fault_env, extra_env=None, T=20.0, exit_to=None, exit_at=1e9):
    cmd_p = f'{SP}/cmd_{name}.json'
    stt_p = f'{SP}/stt_{name}.json'
    env = dict(os.environ, GROUND_RATIO='0', FAULT_AT_S='15', **fault_env)
    env.update(extra_env or {})
    seq = [0]
    def write(mode):
        seq[0] += 1
        with open(cmd_p + '.tmp', 'w') as f:
            json.dump({'mode': mode, 'seq': seq[0], 'v': 0.05, 'vy': 0, 'w': 0}, f)
        os.replace(cmd_p + '.tmp', cmd_p)
    write('off')
    p = subprocess.Popen([BIN, '--mock', '--cmd', cmd_p, '--state', stt_p,
                          '--config', '/home/jsh/simulation/biped/emb/config/biped_emb.yaml',
                          '--mjcf', '/home/jsh/simulation/biped/biped_from_quad.mjcf',
                          '--T', str(T)],
                         cwd=CPP + '/build', env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # 20Hz 시퀀스: hold 2s → home 8s(mock 60dps 로 자세가드 통과) → stand 4s → 12s~ target
    t0 = time.time(); stop = [False]
    def feeder():
        while not stop[0]:
            t = time.time() - t0
            m = ('hold' if t < 2 else 'home' if t < 8 else
                 'stand' if t < 12 else (exit_to if (exit_to and t >= exit_at) else target_mode))
            write(m)
            time.sleep(0.05)
    th = threading.Thread(target=feeder, daemon=True); th.start()
    out, _ = p.communicate(timeout=T + 30)
    stop[0] = True; th.join(timeout=1)
    return out

def check(name, out, expect_trip, expect_sub=''):
    tripped = 'E-STOP' in out and ('속도' in out or '토크' in out)
    banner = 'walk 한정: 트립' in out
    ok = (tripped == expect_trip) and banner and (expect_sub in out if expect_sub else True)
    trip_lines = [l for l in out.splitlines() if 'E-STOP' in l or 'walk 한정' in l]
    print(f"[{name}] {'✅' if ok else '❌'} 트립={tripped}(기대 {expect_trip})"
          + (f" · 표기'{expect_sub}' {'있음' if expect_sub in out else '없음'}" if expect_sub else ''))
    for l in trip_lines[:3]: print('   ', l.strip())
    if not ok:
        print('   --- 꼬리 15줄 ---')
        for l in out.splitlines()[-15:]: print('   ', l)
    return ok

results = []
out = run_case('A', 'walk', {'FAULT_VEL_DPS': '500'})
results.append(check('A walk vel500 → 무트립', out, False))
out = run_case('B', 'stand', {'FAULT_VEL_DPS': '500'})
results.append(check('B stand vel500 → 트립(cfg)', out, True, '(cfg)'))
out = run_case('C', 'walk', {'FAULT_TAU_NM': '30'})
results.append(check('C walk tau30 → 트립(walk)', out, True, '(walk)'))
out = run_case('D', 'walk', {'FAULT_VEL_DPS': '500'}, {'WALK_VEL_TRIP_DPS': '400'})
results.append(check('D walk vel500·env400 → 트립', out, True, '(walk)'))
# E: ★상태 기반(2026-08-28) — 고장 주입은 배포기 시각, 모드 전환은 벽시계라 부팅 지연만큼
#    어긋난다. 상태에서 "walk + 과속 주입됨" 을 **관측한 뒤** 이탈시켜 유예를 정확히 잰다.
def run_exit_grace(T=45.0):
    cmd_p = f'{SP}/cmd_E.json'; stt_p = f'{SP}/stt_E.json'
    env = dict(os.environ, GROUND_RATIO='0', FAULT_AT_S='22', FAULT_VEL_DPS='500')
    seq = [0]
    def write(mode):
        seq[0] += 1
        with open(cmd_p + '.tmp', 'w') as f:
            json.dump({'mode': mode, 'seq': seq[0], 'v': 0.05, 'vy': 0, 'w': 0}, f)
        os.replace(cmd_p + '.tmp', cmd_p)
    write('off')
    p = subprocess.Popen([BIN, '--mock', '--cmd', cmd_p, '--state', stt_p,
                          '--config', '/home/jsh/simulation/biped/emb/config/biped_emb.yaml',
                          '--mjcf', '/home/jsh/simulation/biped/biped_from_quad.mjcf',
                          '--T', str(T)], cwd=CPP + '/build', env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t0 = time.time(); stop = [False]; phase = ['hold']
    def feeder():
        while not stop[0]:
            write(phase[0]); time.sleep(0.05)
    th = threading.Thread(target=feeder, daemon=True); th.start()
    def st():
        try:    return json.load(open(stt_p))
        except Exception: return {}
    for wait_s, m in ((2.0,'home'), (8.0,'stand'), (4.0,'walk')):
        time.sleep(wait_s); phase[0] = m
    # walk 상태에서 과속 주입이 상태에 보일 때까지 대기 → 그때 이탈
    seen = False
    t1 = time.time()
    while time.time() - t1 < 25.0:
        s = st()
        if s.get('mode') == 'walk' and max(abs(x) for x in (s.get('dq_leg_dps') or [0])) > 400:
            seen = True; break
        time.sleep(0.02)
    phase[0] = 'stand'
    time.sleep(0.35)                       # 유예 0.5s 안
    stop[0] = True
    p.terminate()
    out = p.communicate(timeout=20)[0]
    return out, seen
out, seen = run_exit_grace()
ok_e = seen and ('E-STOP' not in out)
print(f"[E walk→stand 이탈 유예 0.35s → 무트립] {'✅' if ok_e else '❌'} 주입관측 {seen} · 트립 {'E-STOP' in out}")
if not ok_e:
    for l in out.splitlines():
        if 'E-STOP' in l: print('   ', l.strip()[:150])
results.append(ok_e)
# F: env 오타 → 경고 + 기본 900 폴백 (500dps 무트립 유지)
out = run_case('F', 'walk', {'FAULT_VEL_DPS': '500'}, {'WALK_VEL_TRIP_DPS': 'abc'})
results.append(check('F env 오타 → 폴백·무트립', out, False, "무효"))
print('\n결과:', sum(results), '/', len(results))
sys.exit(0 if all(results) else 1)
