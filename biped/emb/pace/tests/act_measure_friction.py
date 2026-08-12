#!/usr/bin/env python3
"""act_measure_friction.py — 관절 마찰(정지/쿨롱/점성) 실측.

레퍼런스: motorcortex-python-tools/automatic_testing_examples/motorcortex_tests/
          act_measure_friction.py (VECTIONEER) — "저속 사인으로 동적력을 최소화하고
          위치·속도 대비 토크를 로깅, midstroke 히스테리시스로 마찰을 읽는다" 개념을 계승.
차이점:
  - 신호발생기/DataLogger 대신 SHM(위치+게인) 직접 가진·로깅.
  - **양방향 상쇄를 추가**했다. 레퍼런스는 midstroke 히스테리시스 폭 하나만 보지만,
    실기 관절은 중력토크와 토크센서 바이어스가 함께 섞여 들어온다. 같은 위치를
    +방향/−방향으로 각각 통과시키면
        tau⁺ = +f(v) + g(q) + bias ,  tau⁻ = −f(v) + g(q) + bias
        → f(v)      = (tau⁺ − tau⁻)/2      (마찰만)
        → g(q)+bias = (tau⁺ + tau⁻)/2      (중력+바이어스, 속도에 무관해야 함=검증지표)
    로 분리된다. 이게 없으면 중력이 쿨롱마찰로 둔갑한다.

세 가지 측정:
  (A) breakaway  : 목표각을 아주 천천히 밀어 토크를 키우다 움직이는 순간의 토크 = 정지마찰
  (B) 등속 스윕  : 여러 속도 × 양방향 → f(v) = tau_c + b·v 회귀 → JFRIC, JDAMP
  (C) 저속 사인  : 레퍼런스 방식의 마찰 루프 플롯(위치-토크, 속도-토크)

산출: JFRIC[Nm](쿨롱), JDAMP[Nm·s/rad](점성), tau_static[Nm](정지), Stribeck v_s.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template

from hwio import DEG, Sample, samples_to_arrays

TEMPLATE = Template("""
<h2>{{ title }}</h2>
<p>관절 마찰을 <b>정지마찰(breakaway)</b> · <b>등속 스윕</b> · <b>저속 사인</b> 세 방법으로 측정한다.
등속 스윕은 같은 구간을 양방향으로 통과시켜 중력·센서 바이어스를 상쇄한다.</p>

<table>
  <tr><th colspan="2">시험 조건</th></tr>
  <tr><td>일시</td><td>{{ datetime }}</td></tr>
  <tr><td>축</td><td>{{ joint }} (SHM ch{{ ch }}, 감속비 N={{ gear }})</td></tr>
  <tr><td>게인</td><td class="numeric">Kp={{ kp }} Kd={{ kd }}</td></tr>
  <tr><td>흔든 폭·빠르기</td><td class="numeric">{{ swings }}
      <span class="dim">(= {{ speeds }} deg/s. 저장값은 dps 다 — 회귀변수가 속도라서.
      Hz 는 같은 왕복을 삼각파로 돌렸을 때의 <b>등가</b> 주파수)</span></td></tr>
  <tr><td>사인</td><td class="numeric">{{ '%.1f' % sine_amp }} deg @ {{ sine_f }} Hz</td></tr>

  <tr><th colspan="2">측정 결과 <span class="dim">(보고 토크 단위 기준{{ frame_note }})</span></th></tr>
  <tr><td>정지마찰 τ_s</td><td class="numeric">{{ '%0.4f' % tau_static }} Nm
      <span class="dim">(+dir {{ '%0.4f' % tau_break_pos }} / −dir {{ '%0.4f' % tau_break_neg }}, 산포 ±{{ '%0.4f' % tau_static_sd }})</span></td></tr>
  <tr><td>쿨롱마찰 τ_c → <b>JFRIC</b></td><td class="numeric">{{ '%0.4f' % jfric }} Nm</td></tr>
  <tr><td>점성감쇠 b → <b>JDAMP</b></td><td class="numeric">{{ '%0.4f' % jdamp }} N·m·s/rad</td></tr>
  <tr><td>Stribeck 속도 v_s</td><td class="numeric">{{ vs_str }}</td></tr>
  <tr><td>회귀 적합도 R²</td><td class="numeric">{{ '%0.4f' % r2 }}</td></tr>
  <tr><td>τ_s / τ_c 비</td><td class="numeric">{{ '%0.2f' % stribeck_ratio }}
      <span class="dim">(1보다 크면 Stribeck 효과 있음 = 저속에서 더 무겁다)</span></td></tr>
  <tr><td>중력+바이어스</td><td class="numeric">{{ '%0.4f' % grav_bias }} Nm
      <span class="dim">(속도 무관해야 정상 — 속도별 산포 ±{{ '%0.4f' % grav_bias_sd }})</span></td></tr>
  <tr><td>사인 midstroke 히스테리시스</td><td class="numeric">{{ '%0.4f' % hysteresis }} Nm
      <span class="dim">(≈ 2·τ_c = {{ '%0.4f' % (2 * jfric) }} 와 비교)</span></td></tr>

  <tr><th colspan="2">관절축 환산 <span class="dim">(×N)</span></th></tr>
  <tr><td>JFRIC (관절축)</td><td class="numeric">{{ jfric_joint }}</td></tr>
  <tr><td>JDAMP (관절축)</td><td class="numeric">{{ jdamp_joint }}</td></tr>
</table>

{{ warnings }}

<p><img src="{{ plot_fv }}"></p>
<p><img src="{{ plot_loop }}"></p>
<p><img src="{{ plot_break }}"></p>
""")


# ── (A) 정지마찰 ────────────────────────────────────────────────────────────
def _breakaway(hw, ch, cfg, kp, kd, log, ff=None,
               q_ref0=None) -> tuple[list[float], list[float], list]:
    """목표각을 ramp_dps 로 밀며 |dq|>thresh 가 되는 순간의 토크를 기록.

    ★ff 는 **인자로 받는다** (2026-08-12 실기에서 NameError 로 터진 뒤).
      중력 FF 를 넣을 때 `_ff` 를 여기서 그냥 호출하게 뒀는데, `_ff` 는
      measure_actuator_friction 안의 클로저이고 이 함수는 모듈 수준이라 **안 보인다.**
      _sweeps 는 처음부터 `ff=` 로 받고 있었다 — 같은 값을 두 함수가 다른 방식으로
      집어오던 구조였고, 한쪽만 틀렸다. 이제 둘 다 인자로 받는다.
      ⚠오프라인 테스트가 _breakaway 를 안 밟아서 3커밋 동안 안 걸렸다. 아래
        test_hwio_offline.py 에 이 경로를 넣었다.
    """
    ff = ff or (lambda q: 0.0)
    pos, neg, traces = [], [], []
    # ★모든 시행을 **같은 자리에서** 시작한다 (2026-08-12).
    #   파단토크는 중력을 포함한 생값이다. 시행마다 자리가 옮겨가면 그만큼 밀린다.
    #     HL_thigh 중력 기울기 **+0.10 Nm/°**, 파단푸시가 최대 8° 옮긴다 → 0.83 Nm.
    #     실기 +dir 3시행이 −2.35 → −1.77 → −1.16 으로 **한 방향으로 밀렸다**
    #     (증분 +0.578, +0.609 — 위 계산과 일치). 산포가 아니라 드리프트다.
    #   ⚠이건 brake 도입의 부작용이다. limp 시절엔 매번 같은 처짐자리로 떨어져
    #     기준점이 저절로 생겼는데, brake 는 멈춘 자리에 그대로 붙든다.
    #     limp 로 되돌릴 수는 없다 — thigh 가 146dps 로 자유낙하한다(hwio.brake 주석).
    #   ⇒ 기준자리를 명시적으로 잡고 매 시행 전에 되돌아온다. 정착은 마찰 데드밴드
    #     (τ_s/kp) 안이므로 thigh 기준 ±0.7° = ±0.07 Nm — 종전 ±0.8 Nm 의 1/10 이다.
    for direction in (+1.0, -1.0):
        for trial in range(int(cfg["trials"])):
            if q_ref0 is None:
                q_ref0 = float(hw.read(ch)[0])
                log(f"    기준자리 {q_ref0:+.2f}° — 모든 시행을 여기서 시작한다")
            else:
                hw.goto(ch, q_ref0, kp, kd, speed_dps=10.0, tau_ff_fn=ff)
            q0 = hw.arm(ch, kp, kd)
            time.sleep(0.2)
            tau_at_move, samples, t_ref = None, [], None
            t_max = cfg["max_push_deg"] / cfg["ramp_dps"]

            def qcmd(t, q0=q0, d=direction):
                return q0 + d * cfg["ramp_dps"] * t

            # ★검출은 **위치 변위** 기준. 속도로 판정하면 안 된다 —
            #   이 로봇의 속도 노이즈가 ±15 deg/s 라 임계 2 deg/s 는 정지 중에도 상시 초과한다
            #   (그래서 초기 구현은 τ_s < τ_c 라는 물리적으로 불가능한 값을 냈다).
            #   위치 노이즈는 ~0.01 deg 이므로 0.25 deg 면 25배 여유.
            thr_deg = float(cfg.get("move_thresh_deg", 0.25))
            # ★푸시 **토크 상한** (2026-08-12). 파단푸시는 스톨 감지를 끄고 도는데
            #   (안 움직이는 축에 토크를 키우는 게 측정법 자체다), 그러면 상한이
            #   max_push_deg×kp 라는 **변위 상한 하나뿐**이 된다. hip 이 그게 위험하다:
            #     2.5° × kp 100Nm/rad = 4.36Nm, 중력 4.85 를 더해 피크 9.2Nm 을
            #     최대 4.2초(=2.5/0.6dps) 동안 문다. 실제로 2026-08-12 에 10.6Nm 로
            #     밀다 드라이버 파워단을 잃었다.
            #   ⇒ 중력 대비 초과분이 이 값을 넘으면 **그 시행만** 버리고 나온다.
            #     마찰의 몇 배로 잡으면 정상 파단은 절대 못 건드린다(hip 0.88 vs 상한 2.5).
            cap = cfg.get("tau_cap_nm")
            cap = float(cap) if cap is not None else None
            jammed = False
            q_ref = None
            t0 = time.monotonic()
            k = 0
            while time.monotonic() - t0 < t_max:
                t = time.monotonic() - t0
                ff_now = ff(float(hw._q[ch]))
                s = hw.step(ch, qcmd(t), kp, kd, tau_ff=ff_now)
                samples.append(s)
                if cap is not None and (s.tau - ff_now) * direction > cap:
                    jammed = True
                    break
                if t > 0.3 and q_ref is None:
                    q_ref = s.q_deg                      # 인가 정착 후의 기준 위치
                    t_ref = s.t                          # ★peak-hold 시작 시각도 함께 래치
                # 미는 방향으로 thr_deg 이상 실제로 이동했을 때만 breakaway 로 인정
                if q_ref is not None and (s.q_deg - q_ref) * direction > thr_deg:
                    # ★검출 시점의 순간토크가 아니라 **그때까지의 최대토크**를 쓴다.
                    #   관절이 풀리는 순간 가속하면서 추종오차가 줄어 토크가 이미 떨어져 있다.
                    #   정지마찰의 정의는 "파단 직전 버틴 최대토크" 다.
                    # ★t_ref 이후만 본다. 이전 코드의 `x.t >= samples[0].t` 는
                    #   Sample.t 가 monotonic 절대시각이라 **항상 참**이어서 필터가 없었고,
                    #   인가 램프 과도까지 peak-hold 에 섞였다.
                    seg = [x.tau * direction for x in samples if x.t >= t_ref]
                    tau_at_move = max(seg) * direction
                    break
                k += 1
                slp = t0 + k * hw.dt - time.monotonic()
                if slp > 0:
                    time.sleep(slp)

            # ★limp 하지 않는다 — thigh 는 여기서 146dps 로 자유낙하했다(brake 주석).
            #   파단 판정은 이미 끝났고, 다음 시행의 arm() 이 현재각을 다시 래치한다.
            hw.brake(ch, kp, kd, 0.3, tau_ff_fn=ff)
            if jammed:
                log(f"    ⚠ {'+' if direction > 0 else '−'}dir trial{trial}: "
                    f"중력 대비 초과토크가 상한 {cap:.2f}Nm 에 걸려 중단 — **막힘**이다"
                    f"(파단이면 마찰 근처에서 풀렸어야 한다). 이 시행 제외.")
                continue
            if tau_at_move is None:
                log(f"    ⚠ {'+' if direction > 0 else '−'}dir trial{trial}: "
                    f"{cfg['max_push_deg']}deg 밀어도 미동 — 막힘/한계 의심(제외)")
                continue
            (pos if direction > 0 else neg).append(tau_at_move)
            traces.append((direction, samples_to_arrays(samples)))
            log(f"    {'+' if direction > 0 else '−'}dir trial{trial}: "
                f"breakaway tau={tau_at_move:+.4f}")
    return pos, neg, traces


def _free_reference(hw, ch, kp, kd, ff, log, tau_s_hint=0.8, probe_deg=2.0,
                    step_deg=6.0, tries=4, q_hi=None) -> float:
    """파단을 **양방향 다 움직이는 자리**에서 시작하도록 기준점을 고른다.

    ★왜 필요한가 (2026-08-12 HL_thigh) — 파단은 ±방향 차로 중력을 상쇄하므로
      **한쪽만 막혀도 값이 통째로 안 나온다.** 실기에서 정확히 그랬다:
        +dir 3시행 정상(파단 −2.60, 마찰 0.25) · −dir 3시행 전부 상한 2.5Nm 에 걸림
      중력모델은 무죄로 확인됐다 — FF −2.791 vs MuJoCo(실제 foot +60° 반영) −2.847,
      **0.056Nm 차이**다. 필요한 −dir 힘은 0.30Nm 인데 2.5 에서 막혔으니 진짜 간섭이다.
      가장 유력한 정체: 홈 근처로 내려오면 **늘어진 발이 바닥/스탠드에 닿아** 다리가
      버팀대가 된다. 아래로는 막히고 위로는 자유롭다 — 관측된 비대칭과 일치한다.
    ⇒ 지금 자리에서 ±probe_deg 를 짧게 밀어 보고, 막힌 쪽이 있으면 **반대로 step_deg
      옮겨서** 다시 본다. 마찰 측정은 자리를 안 가린다(±상쇄) — 자유로운 데면 된다.
    ⚠아무 데서도 양방향이 안 열리면 중단한다. 그때는 사람이 볼 문제다.
    """
    cap = max(0.8, 2.0 * float(tau_s_hint))
    q = float(hw.read(ch)[0])
    with hw.intentional_push():
        for k in range(tries):
            blocked = 0
            for d in (+1.0, -1.0):
                hw.arm(ch, kp, kd)
                q0 = float(hw.read(ch)[0])
                t0, moved, since = time.monotonic(), False, None
                while time.monotonic() - t0 < probe_deg / 0.6 + 1.0:
                    t = time.monotonic() - t0
                    fv = ff(float(hw._q[ch]))
                    smp = hw.step(ch, q0 + d * min(0.6 * t, probe_deg), kp, kd, tau_ff=fv)
                    if (smp.q_deg - q0) * d > 0.25:
                        moved = True
                        break
                    if abs(smp.tau - fv) > cap:
                        since = since or time.monotonic()
                        if (time.monotonic() - since) * 1e3 > 200:
                            break
                    else:
                        since = None
                    time.sleep(hw.dt)
                hw.brake(ch, kp, kd, 0.2, tau_ff_fn=ff)
                if not moved:
                    blocked += int(d)          # +1 이면 위쪽이, −1 이면 아래쪽이 막힘
            if blocked == 0:
                log(f"    기준자리 {q:+.2f}° — 양방향 자유(±{probe_deg}° 확인)")
                return q
            away = -1.0 if blocked > 0 else +1.0
            q_new = q + away * step_deg
            if q_hi is not None and not (q_hi[0] <= q_new <= q_hi[1]):
                break
            log(f"    ⚠{q:+.2f}° 에서 {'위' if blocked > 0 else '아래'}쪽이 막혔다 — "
                f"{q_new:+.2f}° 로 옮겨 다시 본다 ({k + 1}/{tries})")
            hw.goto(ch, q_new, kp, kd, speed_dps=10.0, tau_ff_fn=ff)
            q = float(hw.read(ch)[0])
    raise RuntimeError(
        f"양방향이 다 열리는 자리를 {tries}번 시도해도 못 찾았다 (마지막 {q:+.2f}°).\n"
        f"  파단은 ±상쇄가 전제라 한쪽만 막혀도 값이 안 나온다.\n"
        f"  **기구를 볼 것** — 늘어진 발이 바닥·스탠드에 닿아 다리가 버팀대가 되면\n"
        f"  아래로만 막힌다(2026-08-12 HL_thigh 가 그랬다). 다리를 살짝 들어 줄 것.")


# ── (A½) 스윕 전 **범위 확인** ──────────────────────────────────────────────
def _probe_span(hw, ch, kp, kd, q_lo, q_hi, ff, log, tau_s,
                speed_dps=10.0, stuck_ms=250.0) -> tuple[float, float]:
    """[q_lo, q_hi] 를 실제로 갈 수 있는지 저속으로 확인하고 **도달 가능한 구간**을 낸다.

    ★왜 필요한가 (2026-08-12 HL_thigh) — **상자 안인데 못 가는 자리가 있다.**
      무여자로 늘어진 하위 링크(calf·foot)가 바닥·프레임·반대 다리에 닿으면 거기서
      막힌다. 관절한계(상자)로는 절대 알 수 없고, 자세가 바뀔 때마다 위치도 바뀐다.
      실기: 상자 [-60,+40] **한가운데인 +6.71°** 에서 중력 대비 2.17Nm 를 넘기며 스톨.
      상자를 아무리 정확히 적어도 이건 안 잡힌다 — 실제로 가 보는 수밖에 없다.
    ⇒ 스윕 전에 양끝을 **저속으로 한 번 다녀온다.** 막히면 그 자리를 기록하고 구간을
      줄인다. 시험을 죽이지 않는다 — 막힘은 실패가 아니라 **정보**다.
    ⚠전역 스톨 감지는 끄고(임계가 달라서) 여기서 자체 판정한다. 임계 cap_nm 은
      실측 마찰 최대(hip 0.88Nm)보다 위, 스톨 감지 2.0Nm 보다 아래로 잡는다 —
      전역 감지가 터지기 **전에** 우리가 먼저 알아채고 물러나야 하기 때문이다.
    """
    # ★"도달" 문턱은 **정지마찰 데드밴드**보다 넉넉해야 한다 (2026-08-12).
    #   위치제어는 kp·err = τ_s 에서 멈춘다 — 그게 물리다. 그 밖으로는 원래 못 간다.
    #     foot τ_s 0.59 ÷ kp 30 = 1.13° · thigh 0.60 ÷ 50 = 0.69° · calf 0.75 ÷ 80 = 0.54°
    #   0.5° 를 요구하면 **막히지 않았는데도 전부 '도달 실패'** 가 된다(오프라인에서 재현).
    #   ⇒ 데드밴드의 1.5배 + 0.5°. 이건 정밀 위치확인이 아니라 **간섭 탐지**다.
    cap_nm = max(1.0, 2.5 * float(tau_s))
    tol = float(np.rad2deg(float(tau_s) / kp) * 1.5 + 0.5)
    log(f"    (막힘 임계 {cap_nm:.2f}Nm = 정지마찰 {tau_s:.2f} 의 2.5배 · "
        f"도달 문턱 {tol:.2f}° = 데드밴드 {np.rad2deg(tau_s / kp):.2f}° 의 1.5배 +0.5)")
    out = [q_lo, q_hi]
    q_now = float(hw.read(ch)[0])
    # 가까운 끝부터 — 멀리 있는 끝으로 먼저 가면 지나가며 막힐 자리를 두 번 지난다
    ends = sorted(((abs(q_now - q), q, i) for i, q in enumerate((q_lo, q_hi))))
    with hw.intentional_push():
        for _, tgt, idx in ends:
            hw.arm(ch, kp, kd)
            q0 = float(hw.read(ch)[0])
            d = 1.0 if tgt > q0 else -1.0
            t0, k, since = time.monotonic(), 0, None
            while True:
                t = time.monotonic() - t0
                cmd = q0 + d * min(speed_dps * t, abs(tgt - q0))
                fv = ff(float(hw._q[ch]))
                smp = hw.step(ch, cmd, kp, kd, tau_ff=fv)
                if (smp.q_deg - tgt) * d >= -tol:              # 도달(데드밴드 감안)
                    break
                if abs(smp.tau - fv) > cap_nm and abs(smp.dq_dps) < 3.0:
                    since = since or time.monotonic()
                    if (time.monotonic() - since) * 1e3 > stuck_ms:
                        out[idx] = float(smp.q_deg) - d * 2.0    # 2° 물러선 자리
                        log(f"    ⚠{'+' if d > 0 else '−'}쪽 {tgt:+.2f}° 로 못 간다 — "
                            f"{smp.q_deg:+.2f}° 에서 막혔다(중력 대비 "
                            f"{abs(smp.tau - fv):.2f}Nm > {cap_nm}Nm, {smp.dq_dps:+.1f}dps). "
                            f"상자가 아니라 **간섭**이다. 여기까지만 쓴다: {out[idx]:+.2f}°")
                        break
                else:
                    since = None
                if t > abs(tgt - q0) / speed_dps + 5.0:          # 시간 초과
                    # ⚠시간 초과는 **막힘이 아니다** — 느리게라도 가고 있을 수 있다.
                    #   물러나는 폭을 막힘(2°)보다 작게 준다.
                    out[idx] = float(smp.q_deg) - d * 0.5
                    log(f"    ⚠{tgt:+.2f}° 도달 실패(시간 초과, {smp.q_deg:+.2f}° 까지) — "
                        f"{out[idx]:+.2f}° 로 잡는다")
                    break
                k += 1
                slp = t0 + k * hw.dt - time.monotonic()
                if slp > 0:
                    time.sleep(slp)
            hw.brake(ch, kp, kd, 0.2, tau_ff_fn=ff)
    return out[0], out[1]


# ── (B) 등속 스윕 ───────────────────────────────────────────────────────────
def swing_str(stroke_deg: float, v_dps: float) -> str:
    """등속 스윕 속도를 **±각도 · Hz** 로 바꿔 적는다 (2026-08-12, 사용자 요청:
    "dps 란 표현보다 몇 deg 를 몇 Hz 로 움직인다는 표현이 이해하기 쉽다").

    ★저장값은 dps 그대로 둔다 — 바꾸지 말 것. 마찰 회귀의 **회귀변수가 속도**다
      (f = τ_c + b·v). Hz 로 저장하면 stroke 를 건드릴 때마다 실제 속도가 조용히
      따라 변해, "같은 값을 두 곳에서 다르게 다루는" 부류의 버그가 된다.
      ⇒ dps 는 단일 진실원, Hz 는 **표시 전용 파생값**. 변환은 이 함수 하나뿐이다.

    ⚠"등가" Hz 다. 실제 스윕은 한 방향씩 따로 돌고 사이에 goto·정착 대기가 있어
      연속 왕복이 아니다. 같은 왕복을 삼각파로 돌렸을 때의 주파수를 적는 것이다.
    """
    return f"±{stroke_deg / 2:.4g}°·{v_dps / (2.0 * stroke_deg):.3g}Hz"


def _sweeps(hw, ch, cfg, kp, kd, q_center, log, ff=None) -> dict[float, tuple[float, float, float, float]]:
    """속도별 (tau_plus, tau_minus, dq_plus, dq_minus). 양방향 상쇄용."""
    half = cfg["stroke_deg"] / 2.0
    frac = cfg["dwell_frac"]
    out = {}
    # ★스트로크에 안 맞는 속도는 **자동으로 뺀다** (2026-08-12).
    #   통과시간 T=stroke/v 에서 양끝 accel_skip 을 빼면 정착구간이 남아야 한다.
    #   2026-08-12 hip: 스트로크를 16→8° 로 줄이자 35dps 가 0.229s 만에 지나가는데
    #   양끝 accel_skip 0.30s 라 **정착이 음수**가 됐다. 그 점의 '마찰 1.364' 는
    #   가속토크였고, 직선적합의 기울기를 끌어올려 절편 JFRIC 을 **−0.35(음수)** 로,
    #   JDAMP 를 3.08(hip 기대값의 30배)로 만들었다. 물리적으로 불가능한 값이다.
    #   ⚠손으로 속도를 빼면 다음에 스트로크를 또 바꿀 때 같은 일이 난다. 계산으로 건다.
    #   ★뺀 속도는 **반드시 찍는다** — 조용히 줄이면 "다 쟀다" 로 읽힌다.
    _rate = 1.0 / hw.dt
    _need = 2.0 * cfg["accel_skip_s"] + cfg["min_dwell_samples"] / _rate
    _use, _drop = [], []
    for v in cfg["speeds_dps"]:
        (_use if cfg["stroke_deg"] / float(v) > _need else _drop).append(float(v))
    if _drop:
        _ds = ", ".join(f"{swing_str(cfg['stroke_deg'], v)}({v:g}dps)" for v in _drop)
        log(f"    ⚠제외: {_ds} — 스트로크 {cfg['stroke_deg']}° 로는 통과시간이 "
            f"{_need:.3f}s(가속 {2*cfg['accel_skip_s']:.2f} + 최소정착 "
            f"{cfg['min_dwell_samples']/_rate:.3f})보다 짧아 정착 데이터가 안 남는다.")
    log(f"    흔드는 폭·빠르기: "
        + ", ".join(f"{swing_str(cfg['stroke_deg'], v)}" for v in _use)
        + f"   (= {[float(v) for v in _use]} deg/s)")
    if len(_use) < 2:
        raise RuntimeError(
            f"스윕 가능한 속도가 {len(_use)}개뿐이다({_use}) — 회귀가 안 된다. "
            f"stroke_deg 를 키우거나 speeds_dps 에 더 느린 속도를 넣을 것.")
    for v in _use:
        v = float(v)
        T = cfg["stroke_deg"] / v
        res = {}
        for d in (+1.0, -1.0):
            hw.arm(ch, kp, kd)
            hw.goto(ch, q_center - d * half, kp, kd, speed_dps=min(20.0, 3 * v), tau_ff_fn=ff)
            time.sleep(0.3)
            q_start = q_center - d * half
            s = hw.run(ch, lambda t, a=q_start, d=d: a + d * v * t, T, kp, kd, tau_ff_fn=ff)
            # ★limp 하지 않는다 — 120dps 에서 21.8° 관성주행해 상자를 넘었다(brake 주석).
            #   샘플은 이미 s 에 다 들어 있다. 브레이크 구간은 분석에 안 쓴다.
            hw.brake(ch, kp, kd, 0.25, tau_ff_fn=ff)

            a = samples_to_arrays(s)
            # 가감속 구간을 빼고 중앙 dwell 만 사용.
            # ★추가로 "기동 과도" 를 명시적으로 배제한다 — 정지에서 v 로 튀어오르는 구간은
            #   가속토크가 섞여 마찰을 과대평가한다(초기 구현의 R² 저하 원인).
            lo, hi = (1 - frac) / 2, (1 + frac) / 2
            t_settle = float(cfg.get("accel_skip_s", 0.15))
            m = (a["t"] > max(lo * T, t_settle)) & (a["t"] < hi * T)
            need = int(cfg.get("min_dwell_samples", 15))
            if m.sum() < need:
                log(f"    ⚠ {swing_str(cfg['stroke_deg'], v)}: 정상구간 샘플 "
                    f"{m.sum()} < {need} — 제외(스트로크를 늘리거나 더 천천히 흔들 것)")
                res = {}
                break
            # 등속 도달 확인: dwell 구간 속도의 변동이 크면 아직 과도상태다
            v_sd = float(np.std(a["dq"][m]))
            if v_sd > max(0.35 * v, 8.0):
                log(f"    ⚠ {swing_str(cfg['stroke_deg'], v)}: dwell 속도 산포 "
                    f"{v_sd:.1f} deg/s 과대 — 등속 미도달")
            res[d] = (float(np.mean(a["tau"][m])), float(np.mean(a["dq"][m])))
        if len(res) == 2:
            out[v] = (res[+1.0][0], res[-1.0][0], res[+1.0][1], res[-1.0][1])
            f = (res[+1.0][0] - res[-1.0][0]) / 2
            g = (res[+1.0][0] + res[-1.0][0]) / 2
            log(f"    {swing_str(cfg['stroke_deg'], v):>16} ({v:5.1f}dps): "
                f"tau+={res[+1.0][0]:+.4f} tau−={res[-1.0][0]:+.4f} "
                f"→ 마찰 {f:+.4f} · 중력+bias {g:+.4f}")
    return out


def _fit_friction(v_rad, f_meas, tau_s_hint):
    """f(v) = tau_c + b·v (기본) / Stribeck 항은 데이터가 충분할 때만 시도."""
    A = np.column_stack([np.ones_like(v_rad), v_rad])
    coef, *_ = np.linalg.lstsq(A, f_meas, rcond=None)
    tau_c, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(np.sum((f_meas - pred) ** 2))
    ss_tot = float(np.sum((f_meas - np.mean(f_meas)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    v_s = None
    if len(v_rad) >= 5 and tau_s_hint and tau_s_hint > tau_c:
        try:
            from scipy.optimize import curve_fit

            def stribeck(v, tc, bb, ts, vs):
                return tc + bb * v + (ts - tc) * np.exp(-(v / max(vs, 1e-6)) ** 2)

            p0 = [tau_c, b, tau_s_hint, max(np.min(v_rad), 1e-3) * 2]
            popt, _ = curve_fit(stribeck, v_rad, f_meas, p0=p0, maxfev=20000)
            resid = f_meas - stribeck(v_rad, *popt)
            r2s = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 1e-12 else -np.inf
            if r2s > r2 and popt[3] > 0:          # 더 잘 맞을 때만 채택
                tau_c, b, v_s, r2 = float(popt[0]), float(popt[1]), float(popt[3]), r2s
        except Exception:
            pass
    return tau_c, b, v_s, r2


def measure_actuator_friction(hw, spec, joint, plotdir, log=print) -> str:
    ch = int(joint["ch"])
    name, gear = joint["name"], joint["gear"]
    # ★축별 배포게인을 쓴다 (2026-08-12). spec.gains.kp 는 **없어진 키**다 —
    #   게인이 스칼라 하나였던 시절의 잔재이고, 축별 dict(safety.hold_kp)로 바뀌면서
    #   지워졌는데 이 시험만 옛 키를 보고 있어 KeyError 로 죽었다.
    #   ⚠오늘 세 번째 같은 부류다(actuator_test·probe 는 이미 고쳤다). 게인의 단일 출처는
    #     safety.hold_kp/hold_kd 이며, 없으면 gains 의 스칼라로 떨어진다.
    _kp = spec.get("safety", {}).get("hold_kp", spec["gains"].get("kp"))
    _kd = spec.get("safety", {}).get("hold_kd", spec["gains"].get("kd"))
    kp = float(_kp[ch] if isinstance(_kp, dict) else _kp)
    kd = float(_kd[ch] if isinstance(_kd, dict) else _kd)
    # ★축별 축소값을 병합한다 (2026-08-12). hip 은 크게 움직이면 두 다리가 부딪힌다 —
    #   스톨 → 과전류 → 드라이버 보호 → EtherCAT OP 이탈로 이어진다(오늘 3회 동결).
    #   ⚠MuJoCo 충돌 판정에는 발 구 4개만 들어 있어 이 간섭을 못 본다. 값은 실물 기준이다.
    # ★중력 피드포워드 (2026-08-12) — hip 처럼 중력이 큰 축은 kp 가 중력을 감당하느라
    #   축을 밀 여력이 없다(실기: hip 6시행 전부 미동). τ_ff=G(q) 를 실어 kp 를 해방한다.
    #   ⚠표가 없으면 0 이 되어 종전 동작 그대로다(하위호환).
    _gt = (spec.get("torque_mode", {}).get("tau_grav_table") or {}).get(ch)
    _gq = np.asarray(_gt["q_ch"], float) if _gt else None
    _gv = np.asarray(_gt["tau"], float) if _gt else None
    # ★표에 **실측 보정**을 얹는다 (2026-08-12 HL_thigh 오진).
    #   표는 gen_grav_table.py 가 "다른 관절 = hold_pose.neutral_deg" 로 뽑은 것이다
    #   (그 독스트링이 "자세를 바꾸면 다시 뽑아야 한다" 고 이미 경고하고 있다).
    #   그런데 --solo 는 하위 관절이 **무여자로 늘어져** 있다 — calf −61°·foot +60°.
    #   thigh 는 다리 전체를 드는 축이라 여기에 가장 민감하다: 표 1.05 vs 실측 1.90 Nm.
    #   그 0.85Nm 오차가 스톨 감지에서 **가짜 초과토크**가 되어 시험을 세 번 죽였다.
    #   ⇒ 파단이 끝나면 (τ⁺+τ⁻)/2 로 **그 자리의 진짜 중력**을 알 수 있다(마찰이 상쇄됨).
    #     표와의 차이를 상수 오프셋으로 얹는다. 곡선 모양은 표를, 높이는 실측을 믿는다.
    #   ⚠마찰 값 자체는 이 보정과 무관하다 — (τ⁺−τ⁻)/2 에서 중력은 어차피 빠진다.
    #     보정이 고치는 것은 **FF 여력과 스톨 판정**이다.
    #   ⚠actuator_test 가 **홈복귀 전에 이미** 실측 보정을 걸어 hw.grav_fn 에 넣어 뒀다.
    #     여기서 표를 새로 읽으면 그 보정이 사라진다 — 같은 값을 두 곳이 따로 만드는,
    #     오늘 여러 번 나온 그 구조다. ⇒ hw.grav_fn 이 있으면 **그걸 쓴다.**
    #     _gb 는 파단 뒤 **추가 미세보정**(더 정확한 ± 평균)만 담는다.
    _gb = [0.0]
    _base = ((lambda q: float(hw.grav_fn(ch, q))) if hw.grav_fn is not None
             else (lambda q: float(np.interp(q, _gq, _gv)) if _gt else 0.0))
    def _ff(q_ch):
        return _base(q_ch) + _gb[0]
    if _gt:
        log(f"  [{name}] ★중력 피드포워드 켜짐 — 현재 위치 G={_ff(hw.read(ch)[0]):+.3f} Nm "
            f"(kp 가 중력을 감당할 필요가 없어져 push 여력이 그만큼 는다)")

    fr = {k: (dict(v) if isinstance(v, dict) else v) for k, v in spec["friction"].items()}
    _ov = (fr.pop("by_ch", None) or {}).get(ch, {})
    for _sec, _kv in _ov.items():
        fr.setdefault(_sec, {}).update(_kv)
    if _ov:
        log(f"  [{name}] ★축별 축소 적용: " + " · ".join(
            f"{a}.{b}={c}" for a, d in _ov.items() for b, c in d.items()))
    warn: list[str] = []

    log(f"  [{name}] 마찰 측정 시작 (ch{ch}, Kp={kp} Kd={kd})")

    # (A) 정지마찰
    log("  (A) breakaway — 목표각 저속 램프")
    # ★스톨 감지는 여기서만 끈다 — "안 움직이는 축에 토크를 키운다" 가 측정법 자체다.
    #   대신 _breakaway 안의 tau_cap_nm 이 상한을 쥔다(그 주석 참조). 둘은 한 쌍이다:
    #   cap 없이 이 with 만 쓰면 hip 보호가 통째로 사라진다.
    # ★파단 전에 **양방향이 열리는 자리**를 고른다 (_free_reference 주석).
    _q0 = _free_reference(hw, ch, kp, kd, _ff, log,
                          tau_s_hint=max(0.8, 2.0 * float(fr["breakaway"].get(
                              "tau_cap_nm", 2.5)) / 2.5),
                          q_hi=(joint["q_min"] + 3.0, joint["q_max"] - 3.0))
    with hw.intentional_push():
        pos, neg, btraces = _breakaway(hw, ch, fr["breakaway"], kp, kd, log,
                                       ff=_ff, q_ref0=_q0)
    if not pos or not neg:
        raise RuntimeError("breakaway 양방향 데이터 부족 — 축이 막혔거나 게인 부족")
    tp, tn = float(np.mean(pos)), float(np.mean(neg))
    tau_static = (tp - tn) / 2.0                      # 마찰만(중력·bias 상쇄)
    tau_static_sd = float(np.std(pos + [-x for x in neg]))

    # 기준 중심각: breakaway 후 현재 위치
    hw.wait_fresh(ch=ch)
    q_center = hw.read(ch)[0]

    # ★중력 실측 보정 (위 _gb 주석). 파단 ± 평균이 그 자리의 중력이다.
    _g_meas = (tp + tn) / 2.0
    _g_cur = _ff(q_center) - _gb[0]
    _gb[0] = _g_meas - _g_cur
    log(f"    중력 미세보정 {_gb[0]:+.3f} Nm — 현재값 {_g_cur:+.3f} vs 파단 ± 평균 "
        f"{_g_meas:+.3f} Nm @ {q_center:+.2f}°  (0 근처면 홈복귀 전 실측이 맞았다는 뜻)")
    if abs(_gb[0]) > 3.0:
        warn.append(f"중력 보정이 {_gb[0]:+.2f} Nm 로 크다 — 표를 다시 뽑을 것"
                    f"(tools/gen_grav_table.py)")
        log(f"    ⚠보정이 3Nm 을 넘는다 — 자세가 표 생성 시점과 많이 다르다는 뜻이다.")
    # 스톨 감지도 **같은 값**을 보게 한다. 두 곳이 다른 중력을 쓰면 그게 곧 오진이다.
    _prev_gfn = hw.grav_fn
    hw.grav_fn = lambda c, qq, _o=_prev_gfn: (_ff(qq) if c == ch else
                                              (_o(c, qq) if _o else 0.0))
    # ★상자 끝에 **여유를 남긴다** (2026-08-12 실기 ch1 스톨).
    #   종전엔 중심을 [q_min+half, q_max-half] 로 클립했다 — 그러면 스윕 끝이 상자
    #   경계와 **정확히 일치**한다. 실기에서 그대로 터졌다:
    #     HL_thigh 상자 [-60, +40], 파단 후 위치가 +20 이상 → 중심 +20.00 으로 클립
    #     → 스윕 [0, +40]. 그 +40 에서 기구 스톱을 밀며 스톨(초과 2.06Nm, -1.5dps).
    #   ⇒ 양끝 MARGIN 을 비우고, 그래도 안 들어가면 **스트로크를 줄인다.**
    #     조용히 줄이지 않는다 — 줄인 사실과 실제 구간을 로그에 찍는다.
    # ★여유폭 = 기본 3° + **제어정지 거리**. 상수로 두면 최고속도를 올릴 때 또 터진다
    #   (2026-08-12: MARGIN 3° 로 calf 를 +37 까지 보냈고 거기서 넘어갔다).
    #   속도계단 v0 를 kp·kd 로 세울 때의 최대 변위 = (v0/ω_n)·exp(−ζ·φ/√(1−ζ²)).
    #     thigh 120dps → 3.2° · calf 120dps → 2.3° · foot 60dps → 1.1° · hip 10dps → 0.2°
    _I = next((float(x["I_total_pred"]) for x in spec["joints"]
               if int(x["ch"]) == ch and "I_total_pred" in x), None)
    _vmax = max(float(v) for v in fr["sweep"]["speeds_dps"])
    if _I:
        _wn, _z = np.sqrt(kp / _I), kd / (2.0 * np.sqrt(kp * _I))
        _d = np.deg2rad(_vmax) / _wn
        if _z < 1.0:
            _d *= np.exp(-_z / np.sqrt(1 - _z ** 2)
                         * np.arctan2(np.sqrt(1 - _z ** 2), _z))
        MARGIN = 3.0 + float(np.rad2deg(_d))
    else:
        MARGIN = 6.0                       # I 를 모르면 보수적으로
    log(f"    상자 여유 {MARGIN:.1f}° (기본 3.0 + 최고 {_vmax:.0f}dps 제어정지 "
        f"{MARGIN - 3.0:.1f}°)")
    lo_b, hi_b = joint["q_min"] + MARGIN, joint["q_max"] - MARGIN
    half = fr["sweep"]["stroke_deg"] / 2
    if 2 * half > hi_b - lo_b:
        old = fr["sweep"]["stroke_deg"]
        fr["sweep"]["stroke_deg"] = float(hi_b - lo_b)
        half = fr["sweep"]["stroke_deg"] / 2
        msg = (f"스윕 스트로크 축소 {old:.1f}° → {fr['sweep']['stroke_deg']:.1f}° — "
               f"상자 [{joint['q_min']:.1f}, {joint['q_max']:.1f}] 에 여유 {MARGIN}° 를 "
               f"빼면 그만큼밖에 안 들어간다")
        log(f"    ⚠{msg}"); warn.append(msg)
    if not (lo_b + half <= q_center <= hi_b - half):
        was = q_center
        q_center = float(np.clip(q_center, lo_b + half, hi_b - half))
        msg = (f"스윕 중심각 이동 {was:.2f}° → {q_center:.2f}° "
               f"(상자 여유 {MARGIN:.1f}° 확보)")
        log(f"    ⚠{msg}"); warn.append(msg)

    # ★상자만 믿지 않고 **실제로 가 본다** (_probe_span 주석 참조).
    log(f"    범위 확인 — [{q_center - half:+.2f}, {q_center + half:+.2f}]° 를 "
        f"저속으로 다녀온다")
    # ★임계는 **방금 잰 그 축의 마찰**에서 뽑는다 — 고정값은 축마다 틀린다.
    #   "정지마찰의 2.5배를 넘겨도 안 움직이면 그건 마찰이 아니다."
    #   고정 1.2Nm 였다면 hip(τ_s 0.72)에서 여유가 0.5Nm 뿐이라 중력모델 오차만으로도
    #   '간섭' 으로 오탐했을 것이다. 축별로 hip 1.80 · calf 1.88 · foot 1.71 · thigh 1.0.
    r_lo, r_hi = _probe_span(hw, ch, kp, kd, q_center - half, q_center + half,
                             _ff, log, tau_s=tau_static)
    if (r_hi - r_lo) < 2 * half - 0.5:
        half = (r_hi - r_lo) / 2.0
        q_center = (r_hi + r_lo) / 2.0
        fr["sweep"]["stroke_deg"] = float(2 * half)
        msg = (f"간섭으로 스윕 축소 → 중심 {q_center:+.2f}° · ±{half:.1f}° "
               f"(스트로크 {2 * half:.1f}°)")
        log(f"    ⚠{msg}"); warn.append(msg)
        if 2 * half < 3.0:
            raise RuntimeError(
                f"쓸 수 있는 구간이 {2 * half:.1f}° 뿐이다 — 축이 거의 갇혀 있다.\n"
                f"  도달 가능 [{r_lo:+.2f}, {r_hi:+.2f}]° · 상자 "
                f"[{joint['q_min']:+.1f}, {joint['q_max']:+.1f}]°.\n"
                f"  **기구를 눈으로 볼 것** — 늘어진 하위 링크가 바닥·프레임·반대 다리에\n"
                f"  닿아 있을 가능성이 높다. 그 다리를 살짝 들어 주고 다시 실행할 것.")

    # (B) 등속 스윕
    log(f"  (B) 등속 스윕 — 중심 {q_center:.2f}° · ±{half:.1f}° → 실제 구간 "
        f"[{q_center - half:+.2f}, {q_center + half:+.2f}]° "
        f"(상자 [{joint['q_min']:+.1f}, {joint['q_max']:+.1f}]°, 양끝 여유 "
        f"{min(q_center - half - joint['q_min'], joint['q_max'] - q_center - half):+.2f}°)")
    sw = _sweeps(hw, ch, fr["sweep"], kp, kd, q_center, log, ff=_ff)
    if len(sw) < 2:
        raise RuntimeError("등속 스윕 유효 속도 2개 미만 — 측정 불가")

    speeds = np.array(sorted(sw.keys()))
    tau_p = np.array([sw[v][0] for v in speeds])
    tau_n = np.array([sw[v][1] for v in speeds])
    dq_p = np.array([sw[v][2] for v in speeds])
    dq_n = np.array([sw[v][3] for v in speeds])
    f_meas = (tau_p - tau_n) / 2.0
    g_meas = (tau_p + tau_n) / 2.0
    # 회귀에는 명령속도가 아니라 **측정 평균속도**를 쓴다(추종지연 반영)
    v_rad = (np.abs(dq_p) + np.abs(dq_n)) / 2.0 * DEG

    jfric, jdamp, v_s, r2 = _fit_friction(v_rad, f_meas, tau_static)
    grav_bias, grav_bias_sd = float(np.mean(g_meas)), float(np.std(g_meas))
    if grav_bias_sd > 0.15 * max(abs(grav_bias), 1e-6) and abs(grav_bias) > 1e-3:
        warn.append(f"중력+바이어스가 속도에 따라 변한다(±{grav_bias_sd:.4f}) — "
                    f"자세 이동/추종지연/열드리프트 의심. 스트로크를 줄여 재측정 권장")
    if jfric < 0:
        warn.append(f"쿨롱마찰이 음수({jfric:.4f}) — 부호·상쇄 전제 위반. 결과 신뢰 불가")
    # ★점성감쇠는 물리적으로 음수일 수 없다. 이 스윕 속도범위(≤35 deg/s)에서는 Stribeck
    #   감소가 점성 증가를 압도해 선형적합이 음의 기울기를 내는 일이 실제로 발생한다
    #   (2026-08-05 HR_hip: b=-0.0415, R²=0.9992 — 적합도는 좋은데 값이 무의미).
    #   음수를 그대로 내보내면 config 에 붙여넣어질 수 있으므로 여기서 차단한다.
    if jdamp < 0:
        warn.append(
            f"<b>점성감쇠가 음수({jdamp:.4f}) → 이 값을 쓰지 말 것.</b> "
            f"스윕 속도범위(≤{max(fr['sweep']['speeds_dps']):.0f} deg/s)에서는 Stribeck 감소가 "
            f"점성 증가를 가려 기울기가 음수로 나온다. <b>JDAMP 는 PACE 처프 결과를 쓸 것</b> "
            f"(처프는 ~95 deg/s 까지 여기시켜 점성이 신호에 잡힌다).")
        jdamp = float("nan")          # 숫자로 새어나가지 않게 무효화
    elif jdamp * (max(v_rad) if len(v_rad) else 0) < 0.05 * max(abs(jfric), 1e-9):
        warn.append(
            f"점성 기여가 최고속도에서도 쿨롱마찰의 5% 미만 → <b>JDAMP 사실상 미식별</b>. "
            f"PACE 처프 결과를 쓸 것.")

    # (C) 저속 사인 (레퍼런스 방식)
    log("  (C) 저속 사인 — 마찰 루프")
    sn = fr["sine"]
    amp, f_hz = sn["amplitude_deg"], sn["frequency_hz"]
    amp = min(amp, (joint["q_max"] - joint["q_min"]) / 2 - 1.0)
    hw.arm(ch, kp, kd)
    hw.goto(ch, q_center, kp, kd, tau_ff_fn=_ff)   # ★`ff` 아니다 — 클로저 이름은 _ff
    T = sn["cycles"] / f_hz
    s_sine = hw.run(ch, lambda t: q_center + amp * np.sin(2 * np.pi * f_hz * t),
                    T, kp, kd, progress="sine", tau_ff_fn=_ff)
    hw.limp()
    A = samples_to_arrays(s_sine)

    # midstroke 히스테리시스 = 중심각 부근 토크의 최대−최소 ≈ 2·τ_c
    delta = max(0.02 * amp, 0.15)
    mid = np.abs(A["q"] - q_center) < delta
    hysteresis = float(np.max(A["tau"][mid]) - np.min(A["tau"][mid])) if mid.sum() > 3 else float("nan")

    # ── 플롯 ────────────────────────────────────────────────────────────────
    p_fv = f"{plotdir}/friction_fv_ch{ch:02d}.png"
    plt.figure()
    plt.plot(v_rad, f_meas, "o", label="measured f(v)=(tau+ - tau-)/2")
    vv = np.linspace(0, max(v_rad) * 1.05, 200)
    if v_s:
        plt.plot(vv, jfric + jdamp * vv + (tau_static - jfric) * np.exp(-(vv / v_s) ** 2),
                 "-", label=f"Stribeck fit (v_s={v_s:.3f})")
    else:
        plt.plot(vv, jfric + jdamp * vv, "-", label="linear fit  tau_c + b*v")
    plt.axhline(tau_static, ls=":", label=f"static tau_s={tau_static:.3f}")
    plt.xlabel("|angular velocity| (rad/s)"), plt.ylabel("friction torque (Nm)")
    plt.title(f"Friction vs Velocity — {name}"), plt.legend(), plt.grid(alpha=.3)
    plt.savefig(p_fv, dpi=110, bbox_inches="tight"), plt.close()

    p_loop = f"{plotdir}/friction_loop_ch{ch:02d}.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(A["q"], A["tau"], lw=.8)
    ax[0].set_xlabel("position (deg)"), ax[0].set_ylabel("torque (Nm)")
    ax[0].set_title("Friction loop (position)"), ax[0].grid(alpha=.3)
    ax[1].plot(A["dq"] * DEG, A["tau"], lw=.6)
    ax[1].set_xlabel("velocity (rad/s)"), ax[1].set_ylabel("torque (Nm)")
    ax[1].set_title("Torque vs velocity"), ax[1].grid(alpha=.3)
    fig.suptitle(f"Slow sine {amp:.1f}deg @ {f_hz}Hz — {name}")
    plt.savefig(p_loop, dpi=110, bbox_inches="tight"), plt.close()

    p_brk = f"{plotdir}/friction_break_ch{ch:02d}.png"
    plt.figure()
    for d, a in btraces:
        plt.plot(a["t"], a["tau"], lw=.9, label=f"{'+' if d > 0 else '−'}dir")
    plt.axhline(tp, ls="--", c="C0", alpha=.6), plt.axhline(tn, ls="--", c="C1", alpha=.6)
    plt.xlabel("time (s)"), plt.ylabel("torque (Nm)")
    plt.title(f"Breakaway ramp — {name}"), plt.grid(alpha=.3)
    h, l = plt.gca().get_legend_handles_labels()
    if l:
        plt.legend(dict(zip(l, h)).values(), dict(zip(l, h)).keys())
    plt.savefig(p_brk, dpi=110, bbox_inches="tight"), plt.close()

    # ── 축 환산 ─────────────────────────────────────────────────────────────
    frame = spec["units"].get("torque_frame")
    if frame == "motor":
        jf_j, jd_j = f"{jfric * gear:.4f} Nm", f"{jdamp * gear:.4f} N·m·s/rad"
        frame_note = " · 보고=모터축"
    elif frame == "joint":
        jf_j, jd_j = f"{jfric:.4f} Nm (동일)", f"{jdamp:.4f} N·m·s/rad (동일)"
        frame_note = " · 보고=관절축"
    else:
        jf_j = f"미확정 — 모터축이면 {jfric * gear:.4f}, 관절축이면 {jfric:.4f} Nm"
        jd_j = f"미확정 — 모터축이면 {jdamp * gear:.4f}, 관절축이면 {jdamp:.4f}"
        frame_note = " · <b>축 미확정</b>"
        warn.append("spec.yaml <code>units.torque_frame</code> 이 TODO 다 — "
                    "관절축 환산이 확정되지 않는다(모터축이면 ×N).")

    log(f"  [{name}] → JFRIC={jfric:.4f} Nm · JDAMP={jdamp:.4f} Nm·s/rad · "
        f"τ_s={tau_static:.4f} Nm · R²={r2:.4f}")

    warnings_html = ""
    if warn:
        warnings_html = ('<div class="warn"><b>주의</b><ul>'
                         + "".join(f"<li>{w}</li>" for w in warn) + "</ul></div>")

    return TEMPLATE.render(
        title=f"Actuator Friction — {name}", datetime=time.strftime("%Y-%m-%d %H:%M:%S"),
        joint=name, ch=ch, gear=gear, kp=kp, kd=kd,
        speeds=list(fr["sweep"]["speeds_dps"]),
        swings=", ".join(swing_str(fr["sweep"]["stroke_deg"], float(v))
                         for v in fr["sweep"]["speeds_dps"]),
        sine_amp=amp, sine_f=f_hz,
        tau_static=tau_static, tau_static_sd=tau_static_sd,
        tau_break_pos=tp, tau_break_neg=tn,
        jfric=jfric, jdamp=jdamp, r2=r2,
        vs_str=(f"{v_s:.4f} rad/s" if v_s else "미검출(선형이 더 적합)"),
        stribeck_ratio=(tau_static / jfric if jfric > 1e-9 else float("nan")),
        grav_bias=grav_bias, grav_bias_sd=grav_bias_sd, hysteresis=hysteresis,
        jfric_joint=jf_j, jdamp_joint=jd_j, frame_note=frame_note,
        warnings=warnings_html,
        plot_fv=p_fv.replace(plotdir, "plots"), plot_loop=p_loop.replace(plotdir, "plots"),
        plot_break=p_brk.replace(plotdir, "plots"),
    ), {"jfric": jfric, "jdamp": jdamp, "tau_static": tau_static, "r2": r2,
        "grav_bias": grav_bias, "v_s": v_s, "ch": ch, "name": name}
