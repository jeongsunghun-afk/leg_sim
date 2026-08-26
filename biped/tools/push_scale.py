#!/usr/bin/env python3
"""push_scale.py — 발밀기(push) 저울 스윕을 자동화하고 **축별로** 판독한다.

왜 이 도구인가 (2026-08-25, 사용자 제안):
  손으로 저울만 적으면 "그때의 자세" 가정이 다 뭉개진다. 정착 순간의 상태
  스냅샷(q_leg_deg · tau_cmd_nm · push_fz)과 저울값을 **짝지어** 기록한다.

  ★정직한 한계 (설계 중 정정): τ_cmd 는 우리가 모델로 계산해 보낸 값이라
    τ_cmd vs N 회귀는 −Jz_모델/α̂ 를 되돌려줄 뿐 — **한 자세에서는 축별 분리가
    원리적으로 안 된다** (실측 토크가 없다: fCurrent=fTorque 에코).
  ⇒ 축별을 가르는 지렛대는 **자세 다양성**이다:
      α̂(자세) = Σ w_j(자세)·r_j     w_j = Jz² 가중치 · r_j = 경로별 전달비
      0°(calf 74%) · 평발(thigh 지배) 등 자세 2~3개의 α̂ 를 연립하면 r_j 가 풀린다.
    그래서 이 도구는 매 점의 **q 를 기록**한다 — 자세별 JSON 을 모아 연립하는 게 본론.
  축별 회귀는 **일관성 검사**로만 쓴다: 기울기가 −Jz_모델/α̂ 에서 벗어나면
  스윕 중 자세가 흘렀다는(버클링) 신호다.

★C++ 수정이 필요 없다 — 필요한 값은 상태파일에 이미 다 있다.

사용:
  1) 배포기:  ./run_deploy_hw.sh   (GUI 는 닫을 것 — 20ms 마다 명령을 덮는다)
  2) 저울을 측정할 발 밑에 (F=0 에서 살짝 닿게)
  3) python3 tools/push_scale.py --leg HL            # 0→50→0, 10N 계단
     python3 tools/push_scale.py --leg HR --max 40
  각 정착점에서 저울값[g]을 물어본다. 엔터만 치면 그 점은 건너뛴다(무효 표시).

⚠워치독: 저울값 입력 동안에도 **하트비트 스레드**가 명령을 계속 쏜다 —
  input() 이 막는 동안 명령이 끊기면 워치독(500ms, 내용변화 기준)이 limp 를 만든다.
⚠분석의 모델 대조부(Jz_모델)는 mujoco 가 있으면 하고 없으면(Pi) 건너뛴다 —
  원자료 JSON 을 노트북으로 가져가면 전체 대조를 돌릴 수 있다.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import threading
import time

CMD = os.environ.get('QUAD_CMD', '/tmp/biped_cmd.json')
STT = os.environ.get('QUAD_STATE', '/tmp/biped_state.json')
NJ = 8
NAMES = ['HL_hip', 'HL_thigh', 'HL_calf', 'HL_foot',
         'HR_hip', 'HR_thigh', 'HR_calf', 'HR_foot']

_seq = [0]
_cur = {'mode': 'push', 'push_fz': 0.0, 'push_leg': 0}
_stop = [False]


def _send():
    _seq[0] += 1
    c = {'v': 0., 'vy': 0., 'w': 0., 'body_h': 0.38, 'jog_deg': [0.] * NJ,
         'pos_kp_scale': 1.0, 'seq': _seq[0]}
    c.update(_cur)
    t = '%s.%d.tmp' % (CMD, os.getpid())
    open(t, 'w').write(json.dumps(c))
    os.replace(t, CMD)


def _heartbeat():
    """★input() 이 막는 동안에도 명령을 계속 쏜다 — 워치독(내용변화 500ms) 대비."""
    while not _stop[0]:
        try:
            _send()
        except Exception:
            pass
        time.sleep(0.05)


def st():
    try:
        return json.load(open(STT))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', choices=['HL', 'HR'], required=True)
    ap.add_argument('--max', type=float, default=50.0, help='최대 힘[N]')
    ap.add_argument('--step', type=float, default=10.0)
    ap.add_argument('--settle', type=float, default=1.0,
                    help='정지 판정 창[s] — 이 시간 동안 전 축이 안 움직여야 정착')
    ap.add_argument('--still-deg', type=float, default=0.10,
                    help='정지 판정 문턱[°] — 창 안 최대 |Δq|')
    ap.add_argument('--settle-timeout', type=float, default=8.0,
                    help='이 시간까지 정지 안 되면 ⚠표시하고 진행(버클링 신호)')
    ap.add_argument('--no-home', action='store_true',
                    help='시작 시 home 정렬을 건너뛴다(이미 원하는 자세일 때)')
    a = ap.parse_args()

    # ── 가드 — float_gstar 와 같은 이유들 ──────────────────────────────────
    try:
        r = subprocess.run(['pgrep', '-af', 'teleop_gui_biped'],
                           capture_output=True, text=True, timeout=3)
        if [l for l in r.stdout.splitlines() if 'pgrep' not in l]:
            print('✗ teleop GUI 가 떠 있다 — 20ms 마다 명령을 덮어써 스윕이 성립하지 않는다.')
            print('  → GUI 를 닫고 다시 실행할 것.')
            return 1
    except Exception:
        pass
    s0 = st()
    if not s0.get('q_leg_deg'):
        print('✗ 상태파일을 못 읽는다 — 배포기(./run_deploy_hw.sh)가 떠 있는가.')
        return 1
    if s0.get('push_fz') is None:
        print('✗ 상태에 push_fz 가 없다 — 옛 바이너리다. 재빌드+재시작할 것.')
        return 1

    _cur['push_leg'] = 0 if a.leg == 'HL' else 1
    grid = [0.0]
    f = a.step
    while f <= a.max + 1e-9:
        grid.append(round(f, 1)); f += a.step
    grid = grid + grid[-2::-1]          # 0→max→0 왕복

    print('■ push 저울 스윕 — %s · %s' % (a.leg, ' → '.join('%g' % g for g in grid)))
    print('  각 점: 램프 도달 → **엔코더 정지**(%.1fs 창 · |Δq|<%.2f°) → 저울값[g] 입력' % (a.settle, a.still_deg))
    print('  ⚠크레인 줄 팽팽하게 · 측정할 발 하나만 저울 위에.\n')

    th = threading.Thread(target=_heartbeat, daemon=True)
    th.start()

    # ── ★시작 시 home 정렬 (2026-08-25) — GUI 를 끈 뒤에는 home 을 시킬 방법이
    #   없고, GUI 종료~도구 시작 사이 워치독 limp 로 다리가 처져 있을 수도 있다.
    #   ⇒ 도구가 직접 home 을 잡고, 엔코더 정지(같은 기준)로 도달을 확인한 뒤 push 로 간다.
    if not a.no_home:
        print('  home 정렬 중… (엔코더 정지로 도달 판정)')
        _cur['mode'] = 'home'
        t0 = time.time(); buf = []
        while time.time() - t0 < 20.0:
            q = st().get('q_leg_deg')
            if q:
                buf.append((time.time(), q))
                buf = [(t, v) for t, v in buf if time.time() - t <= 1.5]
                if len(buf) >= 2 and buf[0][0] <= time.time() - 1.4:
                    span = max(max(v[j] for _, v in buf) - min(v[j] for _, v in buf)
                               for j in range(NJ))
                    if span < 0.15:
                        break
            if st().get('estop'):
                print('⛔ E-stop — 중단'); _stop[0] = True; return 1
            time.sleep(0.1)
        print('  home 도달. ★저울 위치·발 접촉을 확인하고 엔터를 누르면 push 로 넘어간다.')
        try:
            input('  준비되면 엔터: ')
        except EOFError:
            pass
        _cur['mode'] = 'push'

    rows = []
    # ★점마다 즉시저장 — 축사망/터미널 사망 등 어떤 형태로 끊겨도 데이터가 남는다
    #   (드라이버 과도→축사망 전조 확인 후 보강, 2026-08-26)
    out = '/tmp/push_scale_%s_%s.json' % (a.leg, time.strftime('%Y%m%d-%H%M%S'))
    def _save():
        json.dump({'leg': a.leg, 'rows': rows}, open(out, 'w'), indent=1)
    try:
        for tgt in grid:
            _cur['push_fz'] = float(tgt)
            # 램프 도달 대기 (+ E-stop 감시 — 래치되면 이후 전부 쓰레기다: float_gstar 교훈)
            t0 = time.time()
            while time.time() - t0 < 30.0:
                s = st()
                if s.get('estop'):
                    print('\n⛔ E-stop 래치 — 중단. 이 뒤 점들은 잴 수 없다.')
                    raise KeyboardInterrupt
                if abs(float(s.get('push_fz', 0.0)) - tgt) < 0.2:
                    break
                time.sleep(0.1)
            # ★정착 판정 = 엔코더 정지 (사용자 제안). 고정 대기가 아니라
            #   "창(--settle s) 동안 전 축 |Δq| < --still-deg" 를 요구한다.
            #   calib_zero 의 정지 게이트와 같은 원리 — 그리고 이 판정이
            #   고하중 버클링(무릎이 계속 밀림 = 영원히 정착 안 됨)을 자동 검출한다.
            t0 = time.time(); buf = []
            settled = False; span = float('nan')
            while time.time() - t0 < a.settle_timeout:
                q = st().get('q_leg_deg')
                if q:
                    now = time.time()
                    buf.append((now, q))
                    buf = [(t, v) for t, v in buf if now - t <= a.settle * 2.0]
                    win = [(t, v) for t, v in buf if now - t <= a.settle]
                    if len(win) >= 2:
                        span = max(max(v[j] for _, v in win) - min(v[j] for _, v in win)
                                   for j in range(NJ))
                        # ★판정창(settle)이 표본으로 채워졌을 때만 — 가지치기 창(2×settle)을
                        #   더 넉넉히 둬서 "오래된 표본이 판정 직전에 잘리는" 엡실론 오탐을 없앤다
                        #   (2026-08-25 실기: 0.02° 정지인데 매 점 '정지 안 됨' 오탐이 났었다)
                        if buf[0][0] <= now - a.settle and span < a.still_deg:
                            settled = True; break
                time.sleep(0.1)
            s = st()
            if not settled:
                print('  ⚠F=%.0fN — %.0fs 안에 정지 안 됨(최근 창 최대 %.2f°) — '
                      '자세가 밀리는 중(버클링?). 이 점은 참고용.'
                      % (tgt, a.settle_timeout,
                         span if buf else float('nan')))
            try:
                raw = input('  F=%5.1f N  저울[g]: ' % tgt).strip()
            except EOFError:
                raw = ''
            rows.append({
                't': time.time(), 'F_cmd': tgt,
                'scale_g': float(raw) if raw else None,
                'push_fz': s.get('push_fz'),
                'q_leg_deg': s.get('q_leg_deg'),
                'tau_cmd_nm': s.get('tau_cmd_nm'),
                'mode': s.get('mode'),
                'settled': settled,
            })
            _save()
    except KeyboardInterrupt:
        print('\n  (중단 — 지금까지 점으로 계산한다)')
    finally:
        _cur['push_fz'] = 0.0
        time.sleep(1.0)
        _cur['mode'] = 'float'          # 끝나면 무중력으로 안전 대기
        time.sleep(0.3)
        _stop[0] = True

    _save()
    ok = [r for r in rows if r['scale_g'] is not None and r['q_leg_deg']]
    print('\n  원자료 → %s  (유효 %d/%d점 — 이 파일을 그대로 전달하면 된다)' % (out, len(ok), len(rows)))
    if len(ok) < 4:
        print('  점이 부족해 회귀 생략.')
        return 0

    # ── ① 혼합 기울기 (저울 vs 명령힘) ─────────────────────────────────────
    import statistics
    X = [r['F_cmd'] / 9.81 for r in ok]                    # 명령 [kgf]
    Y = [r['scale_g'] / 1000.0 for r in ok]                # 저울 [kg]
    n = len(X); mx = sum(X) / n; my = sum(Y) / n
    sxx = sum((x - mx) ** 2 for x in X)
    sxy = sum((x - mx) * (y - my) for x, y in zip(X, Y))
    slope = sxy / sxx if sxx > 0 else float('nan')
    print('\n■ 혼합 판독:  저울 = %.3f × 명령 %+.3f kg   → α̂(혼합) = %.3f'
          % (slope, my - slope * mx, slope))

    # ── ② 축별 일관성 검사 — 기울기는 −Jz_모델/α̂ 여야 한다 (새 정보 아님).
    #     이탈 = 스윕 중 자세가 흐른 것(버클링). 축별 '분리'는 자세 여러 개의
    #     혼합 α̂ 를 연립해서 한다(파일 머리 주석).
    print('\n■ 축별 일관성 검사 — 기울기 ≈ −Jz_모델/α̂ 이탈 여부 [m]')
    base = 0 if a.leg == 'HL' else 4
    NN = [r['scale_g'] / 1000.0 * 9.81 for r in ok]        # 저울 실측힘 [N]
    mN = sum(NN) / n
    snn = sum((v - mN) ** 2 for v in NN)
    slopes = {}
    print('  %-9s %14s' % ('축', '기울기 [m]'))
    for j in range(base, base + 4):
        T = [r['tau_cmd_nm'][j] for r in ok]
        mT = sum(T) / n
        sl = sum((v - mN) * (t - mT) for v, t in zip(NN, T)) / snn if snn > 0 else float('nan')
        slopes[NAMES[j]] = sl
        print('  %-9s %+14.4f' % (NAMES[j], sl))

    # ── ③ 모델 대조 (mujoco 있을 때만 — Pi 는 건너뛴다) ────────────────────
    try:
        import mujoco
        import numpy as np
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        m = mujoco.MjModel.from_xml_path(os.path.join(here, 'biped_flatfoot.mjcf'))
        d = mujoco.MjData(m)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                'HL_sphere' if a.leg == 'HL' else 'HR_sphere')
        qm = [statistics.mean(r['q_leg_deg'][j] for r in ok) for j in range(NJ)]
        d.qpos[:] = 0; d.qpos[2] = 0.5; d.qpos[3] = 1
        for j in range(NJ):
            d.qpos[7 + j] = qm[j] * 3.141592653589793 / 180.0
        mujoco.mj_forward(m, d)
        jacp = np.zeros((3, m.nv)); mujoco.mj_jacGeom(m, d, jacp, None, gid)
        print('\n■ 모델 대조 — 비가 1/α̂(≈%.2f)에서 벗어난 축 = 스윕 중 자세 흐름 신호' % (1/slope if slope else 0))
        print('  %-9s %10s %10s %14s' % ('축', '모델 Jz', '−기울기', '비'))
        for j in range(base, base + 4):
            jz = float(jacp[2, 6 + j]); sl = -slopes[NAMES[j]]
            print('  %-9s %+10.4f %+10.4f %14s'
                  % (NAMES[j], jz, sl, ('%.3f' % (sl / jz)) if abs(jz) > 5e-3 else '(레버 미소)'))
        print('  ⇒ ★경로별 분리는 자세 2~3개(0°=calf 지배 · 평발=thigh 지배)의 혼합 α̂ 연립으로.')
    except ImportError:
        print('\n  (mujoco 없음 — 모델 대조는 노트북에서. 원자료 JSON 을 가져갈 것)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
