#!/usr/bin/env python3
"""act_measure_inertia_torque.py — **순환 없는** 관성 측정. 2단(다단) 토크법.

★왜 이 방법인가 (2026-08-11)
  위치처프 식별은 **순환**이다. 드라이버가 돌려주는 τ 가 `kp·err + kd·derr` 로
  R² 0.97 재구성되므로(지연 10ms 정렬), 그 τ 로 회귀하면 우리 게인의 그림자를
  식별하게 된다. `kp=kd=0` 이면 재구성할 항이 없고 **우리가 넣은 τ_ff 가 곧 입력**이다.
  2026-08-11 실기에서 HL_foot 이 평균 0.674 Nm(채널)에 파단 → **순수 토크모드 지원 확인**.
  그 경로가 열렸으니 관성도 같은 경로로 잰다.

원리 — **공통속도법(matched-speed)**. 운동방정식:
      τ = I·q̈ + b·q̇ + τ_c·sgn(q̇) + τ_g

  여러 토크 준위를 걸되, q̈ 를 **모든 준위에서 같은 속도 q̇_ref 에서** 평가한다.
  그러면 `b·q̇_ref` 가 준위마다 **같은 상수**가 되어 τ_c·τ_g 와 함께 절편으로 빠진다:
        τ_i = I · q̈_i|_(q̇=q̇_ref)  +  [b·q̇_ref + τ_c + τ_g]
                                      └── 준위 무관 상수 = 절편 ──┘
  ⇒ τ 를 q̈ 에 회귀하면 **기울기가 곧 I**. 점성 모델도 마찰 모델도 필요 없다.

★쓰지 않는 방법 셋 (전부 실패를 확인했다):
  ① 단순 "τ vs 평균 q̈" — **+20.3% 편향**(합성). 준위가 높을수록 속도가 빨라져
     b·q̇ 손실이 커지고 q̈ 가 눌린다. 그 압축이 기울기를 부풀린다.
  ② 4모수 전역회귀(I·q̈ + b·q̇ + τ_c·sgn + τ_g) — **cond 1e15**. q̈·q̇·sgn 이 전부
     방향과 함께 부호가 뒤집혀 수치적으로 특이하다.
  ③ **2단 회귀(런 내부 q̈~q̇ → 런 사이)** — 2026-08-11 실기에서 **실패**했다.
     런 내부 회귀는 q̈ 가 런 안에서 변해야 성립하는데, 감쇠 시상수 I/b ≈ 1.9s 에 비해
     실제 런은 **0.13~0.27s**(이동 12°·속도상한 120dps) 로 시상수의 7~14% 뿐이다.
     ⇒ q̈ 가 사실상 상수 → 회귀가 노이즈를 맞춘다. 실측 1단 R² 0.004~0.61,
       방향간 I 편차 71%, 같은 시험 2회가 −74.0% / −13.4% 로 갈렸다.
     ⚠내 합성검증이 dur=0.6s 였다 — **실제 런보다 3배 길어서** 통과했던 것이다.
       검증은 반드시 **실제 이동·속도 한계가 만드는 런 길이**로 해야 한다.

합성 검증(실제 런 길이로 재현: 이동 12°·상한 120dps, 참값 I=0.0376):
      준위 [0.90,1.10,1.30] → I 오차 +0.5% / +0.8%  (양방향)
      준위 [0.80,0.95,1.10,1.25] → +0.7% / +0.9%

⚠ 순수 토크는 **자기제한이 없다.** 위치피드백이 0 이라 파단을 넘는 토크는 계속 가속한다.
  방어는 넷:
    ① 이동 상한(`travel_deg`) 도달 즉시 토크 0 → 위치게인으로 **제동**
    ② `hwio._check` 가 매 틱 위치·속도·토크 한계를 강제
    ③ 준위마다 시작점으로 복귀(램프)
    ④ 어떤 경로로 끝나든 limp
⚠ 여기서 나오는 I 는 **채널공간**이다. 관절공간은 `I_joint = I_ch · k²`
  (τ_joint=τ_ch·k, q̈_joint=q̈_ch/k). foot 은 k=1.2 → k²=1.44.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.signal import savgol_filter

from hwio import DEG, SafetyAbort


def _ddq_at_speed(t, q, dq, dt, vref_dps: float, skip_s: float,
                  half_s: float = 0.05, sg_win: int = 31, tol_dps: float = 15.0):
    """**공통 속도 q̇_ref 에서의 q̈** 를 뽑는다. 이게 이 시험의 핵심이다.

    ★왜 공통 속도인가 — 준위마다 속도가 다르면 b·q̇ 손실이 달라져 기울기에 섞인다.
      같은 속도에서 읽으면 그 항이 준위 무관 상수가 되어 **절편으로 빠진다**.
    ★q̈ 는 q(t) 국소 2차식으로 뽑는다(q̇ 미분 아님) — q̇ 노이즈가 실측 15 dps 라
      미분하면 증폭된다. q 는 적분값이라 매끈하다.
    ★반환 None 은 "안 움직였다" 가 아니라 대개 **런이 너무 짧다**는 뜻이다.
    """
    m = t >= skip_s
    n = int(m.sum())
    if n < 20:
        return {"fail": f"skip 후 표본 {n}개 — 런이 너무 짧다"}
    tt, qq, vv = t[m], q[m], dq[m]
    w = min(sg_win if sg_win % 2 else sg_win + 1, n - (1 - n % 2))
    vf = savgol_filter(vv, w, 3) if w >= 7 else vv
    i = int(np.argmin(np.abs(np.abs(vf) - vref_dps)))
    reached = float(np.max(np.abs(vf)))
    if abs(abs(vf[i]) - vref_dps) > tol_dps:
        return {"fail": f"q̇_ref {vref_dps:.0f}dps 미도달(최대 {reached:.0f}dps)"}
    sel = np.abs(tt - tt[i]) <= half_s
    if sel.sum() < 15:
        return {"fail": f"q̇_ref 주변 표본 {int(sel.sum())}개 — 런이 너무 짧다"}
    ddq = float(np.polyfit(tt[sel], qq[sel], 2)[0] * 2.0 * DEG)   # rad/s²
    return {"ddq": ddq, "v_at": float(vf[i]), "t_at": float(tt[i]),
            "n_win": int(sel.sum()), "v_max": reached}


def measure_inertia_torque(hw, spec, joint, plotdir, log=print) -> tuple[str, dict]:
    ch = int(joint["ch"])
    name = joint["name"]
    cfg = spec.get("inertia_torque", {})
    levels = [float(x) for x in cfg.get("tau_levels_nm", [0.9, 1.1, 1.3])]
    tau_max = float(cfg.get("tau_max_nm", 1.6))
    travel = float(cfg.get("travel_deg", 12.0))
    skip_s = float(cfg.get("skip_s", 0.04))
    vref = float(cfg.get("vref_dps", 60.0))
    half_s = float(cfg.get("win_half_s", 0.05))
    v_cap = float(cfg.get("vel_cap_dps", 120.0))
    brake_kp = float(cfg.get("brake_kp", 30.0))
    brake_kd = float(cfg.get("brake_kd", 2.0))
    k_gear = float(joint.get("gear_k", 1.0))
    I_pred_joint = joint.get("I_link_total_pred", joint.get("I_total_pred"))

    log(f"  [{name}] 관성 측정(공통속도법) — 준위 {levels} Nm · 이동 {travel}° · "
        f"속도상한 {v_cap:.0f}dps · **q̇_ref {vref:.0f}dps**")
    log(f"           모든 준위에서 **같은 속도**의 q̈ 를 읽는다 ⇒ b·q̇ 가 절편으로 빠지고"
        f" 기울기가 곧 I")

    q_start = hw.read(ch)[0]
    runs = []
    try:
        for direction in (+1.0, -1.0):
            for tau in levels:
                hw.goto(ch, q_start, brake_kp, brake_kd, speed_dps=10.0)  # 시작점 복귀
                time.sleep(0.3)
                hw.arm(ch, 0.0, 0.0)                  # kp=kd=0 (게인 램프 0→0, 무해)
                q0 = hw.read(ch)[0]
                t0 = time.monotonic()
                T, Q, V, TAU = [], [], [], []
                hit = None
                while True:
                    t = time.monotonic() - t0
                    if t > 4.0:
                        hit = "시간초과"
                        break
                    s = hw.step_torque(ch, direction * tau, tau_max)
                    T.append(t); Q.append(s.q_deg); V.append(s.dq_dps); TAU.append(s.tau)
                    if abs(s.q_deg - q0) >= travel:
                        hit = "이동상한"
                        break
                    if abs(s.dq_dps) >= v_cap:
                        hit = "속도상한"
                        break
                    time.sleep(hw.dt)
                # ★즉시 제동 — τ_ff 를 끊고 위치게인으로 잡는다. 놔두면 계속 가속한다.
                #   제동목표는 **제동 시작 시점의 위치**로 고정한다(매 틱 현재값으로
                #   갱신하면 오차가 0 이라 kp 가 일을 안 하고 kd 감쇠만 남는다).
                q_brake = hw.read(ch)[0]
                for _ in range(int(0.5 / hw.dt)):
                    hw.step(ch, q_brake, brake_kp, brake_kd)
                    time.sleep(hw.dt)
                t = np.array(T); q = np.array(Q) - q0; v = np.array(V)
                dt = float(np.median(np.diff(t))) if t.size > 2 else hw.dt
                f1 = _ddq_at_speed(t, q, v, dt, vref, skip_s, half_s)
                sgn = '+' if direction > 0 else '−'
                if "fail" in f1:
                    log(f"    {sgn} τ={tau:.2f}: ✗ {f1['fail']}  "
                        f"(런 {t[-1]:.3f}s/{len(T)}표본 · {hit})")
                    continue
                runs.append({"dir": direction, "tau_cmd": tau, "signed_tau": direction * tau,
                             "ddq": f1["ddq"], "v_at": f1["v_at"], "t_at": f1["t_at"],
                             "n_win": f1["n_win"], "v_max": f1["v_max"],
                             "travel": float(q[-1]), "stop": hit, "dur": float(t[-1]),
                             "t": t, "q": q, "dq": v, "tau": np.array(TAU)})
                log(f"    {sgn} τ={tau:.2f}Nm → q̈|q̇={f1['v_at']:5.1f}dps = "
                    f"{f1['ddq']:+7.2f} rad/s²  (t={f1['t_at']:.3f}s · 창표본 {f1['n_win']} · "
                    f"런 {t[-1]:.3f}s · {hit})")
    finally:
        hw.goto(ch, q_start, brake_kp, brake_kd, speed_dps=10.0)

    res = {"ch": ch, "name": name, "levels": levels, "runs":
           [{kk: vv for kk, vv in r.items() if kk not in ("t", "q", "dq", "tau")}
            for r in runs], "gear_k": k_gear}

    # ── 회귀: τ = I·q̈|q̇_ref + [b·q̇_ref + τ_c + τ_g] ────────────────────
    #   기울기가 곧 I. 절편은 준위 무관 상수(점성·마찰·중력의 합)다.
    MIN_PTS = 3          # ★2점이면 직선이 항상 완벽히 맞는다 — R²=1 이 아무 의미 없다
    fits, warn = {}, []
    for d, lbl in ((+1.0, "+"), (-1.0, "−")):
        sel = [r for r in runs if r["dir"] == d]
        if len(sel) < MIN_PTS:
            warn.append(f"{lbl}방향 유효표본 {len(sel)}개 < {MIN_PTS} — 회귀 생략"
                        f"(2점 회귀는 R²=1 이 나오지만 검증력이 없다)")
            continue
        x = np.array([r["ddq"] * d for r in sel])
        y = np.array([r["tau_cmd"] for r in sel])
        A = np.column_stack([x, np.ones_like(x)])
        th, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1.0 - np.sum((y - A @ th) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        fits[lbl] = {"I_ch": float(th[0]), "intercept": float(th[1]), "r2": float(r2),
                     "n": len(sel), "cond": float(np.linalg.cond(A))}
        log(f"  [{name}] {lbl}방향: I_ch={th[0]:.5f} kg·m²(채널) · "
            f"절편={th[1]:+.3f} Nm · R²={r2:.4f} · cond={np.linalg.cond(A):.1f} (n={len(sel)})")
    res["fits"] = fits
    for w in warn:
        log(f"  [{name}] ⚠ {w}")
    res["warnings"] = warn

    # ── 유효성 판정 — 못 믿을 값은 **숫자를 내지 않는다** ──────────────────
    ok = len(fits) == 2
    if ok:
        sp = abs(fits["+"]["I_ch"] - fits["−"]["I_ch"]) / max(
            abs(fits["+"]["I_ch"] + fits["−"]["I_ch"]) / 2, 1e-12)
        res["dir_spread"] = float(sp)
        if sp > 0.15:
            ok = False
            warn.append(f"방향간 I 편차 {sp*100:.0f}% > 15% — 비대칭이 크다")
    res["valid"] = bool(ok)

    if fits:
        I_ch = float(np.mean([f["I_ch"] for f in fits.values()]))
        I_joint = I_ch * k_gear ** 2
        res.update(I_ch=I_ch, I_joint=I_joint)
        tag = "" if ok else "  ⚠**신뢰 못 함**"
        log(f"\n  [{name}] I_ch = {I_ch:.5f} → I_joint = I_ch·k² = "
            f"{I_ch:.5f}×{k_gear**2:.2f} = **{I_joint:.5f} kg·m²**{tag}")
        if len(fits) == 2:
            log(f"  [{name}] 방향간 I 편차 {res['dir_spread']*100:.1f}% "
                f"(기준 15%){'  ✓' if res['dir_spread'] <= 0.15 else '  ✗'}")
            ic = float(np.mean([f["intercept"] for f in fits.values()]))
            res["intercept_mean"] = ic
            log(f"  [{name}] 절편 {ic:+.3f} Nm = b·q̇_ref + τ_c + τ_g "
                f"— 파단토크 0.674 Nm(채널) 과 같은 크기여야 맞다")
        if I_pred_joint:
            dd = (I_joint / float(I_pred_joint) - 1.0) * 100.0
            res.update(I_pred_joint=float(I_pred_joint), err_pct=dd)
            log(f"  [{name}] MJCF 예측 {float(I_pred_joint):.5f} 대비 **{dd:+.1f}%**{tag}")
    else:
        log(f"\n  [{name}] ✗ 유효 표본이 부족해 **관성을 산출하지 않는다**. "
            f"준위/이동/속도상한을 조정해 런을 길게 만들 것.")

    html = _html(name, ch, runs, res, k_gear)
    return html, res


def _html(name, ch, runs, res, k):
    rows = "".join(
        f"<tr><td>{'+' if r['dir']>0 else '−'}</td>"
        f"<td class=numeric>{r['tau_cmd']:.2f}</td>"
        f"<td class=numeric>{r['v_at']:.1f}</td>"
        f"<td class=numeric>{r['ddq']:+.2f}</td>"
        f"<td class=numeric>{r['t_at']:.3f}</td>"
        f"<td class=numeric>{r['n_win']}</td>"
        f"<td class=numeric>{r['dur']:.3f}</td>"
        f"<td class=numeric>{r['travel']:+.1f}</td><td>{r['stop']}</td></tr>"
        for r in runs)
    fit = "".join(
        f"<tr><td>{l}</td><td class=numeric>{f['I_ch']:.5f}</td>"
        f"<td class=numeric>{f['intercept']:+.3f}</td>"
        f"<td class=numeric>{f['r2']:.4f}</td>"
        f"<td class=numeric>{f['cond']:.1f}</td><td class=numeric>{f['n']}</td></tr>"
        for l, f in res.get("fits", {}).items())
    valid = res.get("valid", False)
    concl = ""
    if "I_joint" in res:
        badge = ("<b style='color:#1a7f37'>유효</b>" if valid
                 else "<b style='color:#a8620a'>⚠신뢰 못 함</b>")
        concl = (f"<p>{badge} · <b>I_ch = {res['I_ch']:.5f}</b> kg·m²(채널) → "
                 f"<b>I_joint = I_ch·k² = {res['I_joint']:.5f}</b> kg·m² (k={k})")
        if "err_pct" in res:
            concl += (f" · MJCF 예측 {res['I_pred_joint']:.5f} 대비 "
                      f"<b>{res['err_pct']:+.1f}%</b>")
        concl += "</p>"
        if "dir_spread" in res:
            concl += (f"<p>방향간 I 편차 <b>{res['dir_spread']*100:.1f}%</b> (기준 15%) · "
                      f"절편 {res.get('intercept_mean', float('nan')):+.3f} Nm "
                      f"= b·q̇_ref + τ_c + τ_g — 파단토크와 대조할 것</p>")
    warns = "".join(f"<li>{w}</li>" for w in res.get("warnings", []))
    warns = f"<div class=warn><b>경고</b><ul>{warns}</ul></div>" if warns else ""
    return f"""
<h2>{name} (ch{ch}) — 관성 (공통속도법, 순수 τ_ff)</h2>
<p class=dim>Kp=Kd=0 이라 드라이버가 τ 를 되돌려 계산할 항이 없다 —
<b>순환이 원천 소멸</b>한다. 입력은 우리가 넣은 τ_ff, 출력은 엔코더 q 다.</p>
<p class=dim>모든 준위에서 <b>같은 속도 q̇_ref</b> 의 q̈ 를 읽는다. 그러면 b·q̇_ref 가
준위 무관 상수가 되어 τ_c·τ_g 와 함께 절편으로 빠지고,
<b>τ–q̈ 기울기가 곧 I</b> 다. 점성/마찰 모델이 필요 없다.</p>
<table><tr><th>방향</th><th>τ_cmd[Nm]</th><th>읽은 q̇[dps]</th><th>q̈[rad/s²]</th>
<th>시각[s]</th><th>창표본</th><th>런[s]</th><th>이동[°]</th><th>종료</th></tr>{rows}</table>
<table><tr><th>방향</th><th>I_ch[kg·m²]</th><th>절편[Nm]</th><th>R²</th><th>cond</th>
<th>n</th></tr>{fit}</table>
{concl}{warns}
<div class=warn><b>해석 주의</b><ul>
<li>여기 I 는 <b>채널공간</b>이다. 관절공간은 I·k² (드라이버 기어비 오설정 보정).</li>
<li>준위가 <b>3개 미만</b>이면 회귀하지 않는다 — 2점은 항상 R²=1 이라 검증력이 없다.</li>
<li>폐기된 방법들: ① 평균 q̈ 회귀(+20.3% 편향) ② 4모수 전역회귀(cond 1e15)
③ 런 내부 q̈~q̇ 2단회귀(실기 실패 — 런이 감쇠 시상수의 7~14%라 q̈ 가 상수).</li>
<li>이 시험은 τ_ff <b>스케일</b>을 검증하지 않는다. 명령 τ 가 실제로 그만큼 나오는지는
독립 전류측정이 있어야 안다(현재 fCurrent 는 fTorque 복제).</li>
</ul></div>
"""
