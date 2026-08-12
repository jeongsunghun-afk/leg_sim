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
def _breakaway(hw, ch, cfg, kp, kd, log) -> tuple[list[float], list[float], list]:
    """목표각을 ramp_dps 로 밀며 |dq|>thresh 가 되는 순간의 토크를 기록."""
    pos, neg, traces = [], [], []
    for direction in (+1.0, -1.0):
        for trial in range(int(cfg["trials"])):
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
                ff = _ff(float(hw._q[ch]))
                s = hw.step(ch, qcmd(t), kp, kd, tau_ff=ff)
                samples.append(s)
                if cap is not None and (s.tau - ff) * direction > cap:
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

            hw.limp()
            time.sleep(0.3)
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
            hw.limp()
            time.sleep(0.25)

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
    def _ff(q_ch):
        return float(np.interp(q_ch, _gq, _gv)) if _gt else 0.0
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
    with hw.intentional_push():
        pos, neg, btraces = _breakaway(hw, ch, fr["breakaway"], kp, kd, log)
    if not pos or not neg:
        raise RuntimeError("breakaway 양방향 데이터 부족 — 축이 막혔거나 게인 부족")
    tp, tn = float(np.mean(pos)), float(np.mean(neg))
    tau_static = (tp - tn) / 2.0                      # 마찰만(중력·bias 상쇄)
    tau_static_sd = float(np.std(pos + [-x for x in neg]))

    # 기준 중심각: breakaway 후 현재 위치
    hw.wait_fresh(ch=ch)
    q_center = hw.read(ch)[0]
    half = fr["sweep"]["stroke_deg"] / 2
    if not (joint["q_min"] + half <= q_center <= joint["q_max"] - half):
        q_center = float(np.clip(q_center, joint["q_min"] + half, joint["q_max"] - half))
        warn.append(f"스윕 중심각을 한계 안으로 이동: {q_center:.2f} deg")

    # (B) 등속 스윕
    log(f"  (B) 등속 스윕 — 중심 {q_center:.2f}deg, ±{half:.1f}deg")
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
    hw.goto(ch, q_center, kp, kd, tau_ff_fn=ff)
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
