#!/usr/bin/env python3
"""act_measure_inertia_torque.py — **순환 없는** 관성 측정. 2단(다단) 토크법.

★왜 이 방법인가 (2026-08-11)
  위치처프 식별은 **순환**이다. 드라이버가 돌려주는 τ 가 `kp·err + kd·derr` 로
  R² 0.97 재구성되므로(지연 10ms 정렬), 그 τ 로 회귀하면 우리 게인의 그림자를
  식별하게 된다. `kp=kd=0` 이면 재구성할 항이 없고 **우리가 넣은 τ_ff 가 곧 입력**이다.
  2026-08-11 실기에서 HL_foot 이 평균 0.674 Nm(채널)에 파단 → **순수 토크모드 지원 확인**.
  그 경로가 열렸으니 관성도 같은 경로로 잰다.

원리 — **2단 회귀**. 마찰(τ_c)도 점성(b)도 모델 가정 없이 분리된다.
      운동방정식:  τ = I·q̈ + b·q̇ + τ_c·sgn(q̇) + τ_g

  **1단 (런 내부)** — τ 가 상수인 구간에서 정리하면
        q̈ = (τ − τ_c − τ_g)/I  −  (b/I)·q̇
      q̈ 를 q̇ 에 회귀한다. τ 가 안 변하는 동안 **q̇ 만 크게 변하므로 조건이 좋다**
      (실측 cond ≈ 14). 절편 a = (τ−τ_c−τ_g)/I, 기울기 = −b/I.
  **2단 (런 사이)** — 여러 준위의 절편을 τ 에 회귀하면
        a_i = τ_i/I − (τ_c+τ_g)/I        ⇒  **I = 1/기울기**

★왜 단순한 "τ vs q̈ 직선"이 아닌가 — 그건 **+20.3% 편향**이 난다(합성검증 확인).
  준위가 높을수록 속도가 빨라져 b·q̇ 손실이 커지고, 그만큼 q̈ 가 눌린다.
  그 압축이 기울기를 키워 I 를 과대평가한다. 점성을 따로 빼야 한다.
★4모수 전역회귀(τ = I·q̈ + b·q̇ + τ_c·sgn + τ_g)도 안 된다 — **cond 1e15**.
  q̈·q̇·sgn 이 전부 방향과 함께 부호가 뒤집혀 수치적으로 특이하다.
  2단으로 나누면 각 단이 잘 조건화된다.

합성 검증(참값 I=0.0376, τ_c=0.674, b=0.02):
      속도노이즈  0 dps → I 오차 −0.05%
      속도노이즈 15 dps → I 오차 −0.16%   ← 실측 노이즈 수준
      속도노이즈 30 dps → I 오차 −0.47%

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


def _fit_run(t, q, dq, dt, skip_s: float, sg_win: int = 31):
    """1단 — 런 내부에서 q̈ 를 q̇ 에 회귀. 반환: 절편 a=(τ−τ_c−τ_g)/I, 기울기 −b/I.

    ★q̈ 는 q̇ 를 savgol 미분해 얻는다. q 를 두 번 미분하면 노이즈가 두 배로 증폭된다.
    ★skip_s 로 파단 직후 과도구간을 버린다 — 정지→운동 마찰 전환이 거기서 일어난다.
    """
    m = t >= skip_s
    n = int(m.sum())
    if n < sg_win + 5:
        return None
    w = sg_win if sg_win % 2 else sg_win + 1
    w = min(w, n - (1 - n % 2))                     # 표본보다 크면 줄인다(홀수 유지)
    if w < 7:
        return None
    v = savgol_filter(dq[m], w, 3) * DEG            # rad/s
    a = savgol_filter(dq[m], w, 3, deriv=1, delta=dt) * DEG   # rad/s²
    slope, icept = np.polyfit(v, a, 1)
    pred = slope * v + icept
    r2 = 1.0 - np.sum((a - pred) ** 2) / max(np.sum((a - a.mean()) ** 2), 1e-12)
    return {"icept": float(icept), "slope": float(slope), "r2": float(r2),
            "n": n, "dq_span": float(np.ptp(v) / DEG), "dq_mean": float(np.mean(np.abs(v)) / DEG),
            "ddq_mean": float(np.mean(a))}


def measure_inertia_torque(hw, spec, joint, plotdir, log=print) -> tuple[str, dict]:
    ch = int(joint["ch"])
    name = joint["name"]
    cfg = spec.get("inertia_torque", {})
    levels = [float(x) for x in cfg.get("tau_levels_nm", [0.9, 1.1, 1.3])]
    tau_max = float(cfg.get("tau_max_nm", 1.6))
    travel = float(cfg.get("travel_deg", 12.0))
    skip_s = float(cfg.get("skip_s", 0.10))
    v_cap = float(cfg.get("vel_cap_dps", 120.0))
    brake_kp = float(cfg.get("brake_kp", 30.0))
    brake_kd = float(cfg.get("brake_kd", 2.0))
    k_gear = float(joint.get("gear_k", 1.0))
    I_pred_joint = joint.get("I_link_total_pred", joint.get("I_total_pred"))

    log(f"  [{name}] 관성 측정(2단 토크법) — 준위 {levels} Nm · 이동 {travel}° · "
        f"속도상한 {v_cap:.0f}dps")
    log(f"           마찰·중력은 τ–q̈ 직선의 **절편**으로 빠진다 ⇒ 기울기가 I")

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
                f1 = _fit_run(t, q, v, dt, skip_s)      # ★1단 — 런 내부 q̈~q̇
                if f1 is None:
                    log(f"    {'+' if direction>0 else '−'} τ={tau:.2f}: 표본 부족({len(T)}) "
                        f"— 움직이지 않았을 수 있다(파단 미만?)")
                    continue
                runs.append({"dir": direction, "tau_cmd": tau, "signed_tau": direction * tau,
                             "icept": f1["icept"], "slope": f1["slope"], "r2_run": f1["r2"],
                             "dq_span": f1["dq_span"], "dq_mean": f1["dq_mean"],
                             "ddq_mean": f1["ddq_mean"],
                             "travel": float(q[-1]), "stop": hit, "n": f1["n"],
                             "t": t, "q": q, "dq": v, "tau": np.array(TAU)})
                log(f"    {'+' if direction>0 else '−'} τ={tau:.2f}Nm → "
                    f"절편 {f1['icept']:+7.2f} rad/s² · 기울기 {f1['slope']:+6.3f}(=−b/I) "
                    f"· q̇범위 {f1['dq_span']:5.1f}dps · R²={f1['r2']:.3f} · {hit}")
    finally:
        hw.goto(ch, q_start, brake_kp, brake_kd, speed_dps=10.0)

    res = {"ch": ch, "name": name, "levels": levels, "runs":
           [{kk: vv for kk, vv in r.items() if kk not in ("t", "q", "dq", "tau")}
            for r in runs], "gear_k": k_gear}

    # ── 2단 회귀 — 절편 a_i = τ_i/I − (τ_c+τ_g)/I  ⇒ I = 1/기울기 ──────────
    #   ★단순 "τ vs q̈" 직선은 쓰지 않는다. 점성이 기울기로 새어들어 **+20.3% 편향**이
    #     난다(합성검증). 4모수 전역회귀는 cond 1e15 로 특이하다. 2단이 유일하게 맞다.
    fits = {}
    for d, lbl in ((+1.0, "+"), (-1.0, "−")):
        sel = [r for r in runs if r["dir"] == d]
        if len(sel) < 2:
            continue
        tau_i = np.array([r["tau_cmd"] for r in sel])
        a_i = np.array([r["icept"] * d for r in sel])     # 방향 부호 제거
        M = np.column_stack([tau_i, np.ones_like(tau_i)])
        th, *_ = np.linalg.lstsq(M, a_i, rcond=None)
        if abs(th[0]) < 1e-9:
            continue
        I_ch = 1.0 / th[0]
        pred = M @ th
        r2 = 1.0 - np.sum((a_i - pred) ** 2) / max(np.sum((a_i - a_i.mean()) ** 2), 1e-12)
        b_ch = -float(np.mean([r["slope"] for r in sel])) * I_ch
        fits[lbl] = {"I_ch": float(I_ch), "tau_c_plus_g": float(-th[1] * I_ch),
                     "b_ch": b_ch, "r2": float(r2), "n": len(sel),
                     "cond": float(np.linalg.cond(M))}
        log(f"  [{name}] {lbl}방향 2단: I_ch={I_ch:.5f} kg·m²(채널) · "
            f"τ_c+τ_g={-th[1]*I_ch:+.3f} Nm · b={b_ch:.4f} · R²={r2:.4f} "
            f"· cond={np.linalg.cond(M):.1f} (n={len(sel)})")
    res["fits"] = fits

    if fits:
        I_ch = float(np.mean([f["I_ch"] for f in fits.values()]))
        I_joint = I_ch * k_gear ** 2
        res["I_ch"] = I_ch
        res["I_joint"] = I_joint
        res["b_ch"] = float(np.mean([f["b_ch"] for f in fits.values()]))
        if len(fits) == 2:
            # 양방향 상쇄: (τ_c+τ_g)_+ 와 _− 에서 마찰과 중력을 가른다
            fp, fm = fits["+"]["tau_c_plus_g"], fits["−"]["tau_c_plus_g"]
            res["tau_c"] = float((fp + fm) / 2.0)      # 두 방향 모두 운동 반대 → 공통
            res["tau_g"] = float((fp - fm) / 2.0)
            log(f"  [{name}] 절편 분해 — 쿨롱마찰 {res['tau_c']:+.3f} Nm · "
                f"중력+bias {res['tau_g']:+.3f} Nm (채널)")
            log(f"           ↑ 파단토크와 같은 크기여야 맞다 (HL_foot 실측 0.674 Nm 채널)")
            # 좌우 일관성 — 두 방향 I 가 크게 다르면 무언가 비대칭이다
            sp = abs(fits["+"]["I_ch"] - fits["−"]["I_ch"]) / max(I_ch, 1e-12)
            res["dir_spread"] = float(sp)
            log(f"  [{name}] 방향간 I 편차 {sp*100:.1f}%"
                + ("  ⚠10% 초과 — 비대칭 원인 확인 필요" if sp > 0.10 else ""))
        log(f"\n  [{name}] ★I_ch = {I_ch:.5f} → **I_joint = I_ch·k² = "
            f"{I_ch:.5f}×{k_gear**2:.2f} = {I_joint:.5f} kg·m²**")
        if I_pred_joint:
            dd = (I_joint / float(I_pred_joint) - 1.0) * 100.0
            res["I_pred_joint"] = float(I_pred_joint)
            res["err_pct"] = dd
            log(f"  [{name}] MJCF 예측 {float(I_pred_joint):.5f} 대비 **{dd:+.1f}%**")

    html = _html(name, ch, runs, res, k_gear)
    return html, res


def _html(name, ch, runs, res, k):
    rows = "".join(
        f"<tr><td>{'+' if r['dir']>0 else '−'}</td>"
        f"<td class=numeric>{r['tau_cmd']:.2f}</td>"
        f"<td class=numeric>{r['icept']:+.2f}</td>"
        f"<td class=numeric>{r['slope']:+.3f}</td>"
        f"<td class=numeric>{r['dq_span']:.1f}</td>"
        f"<td class=numeric>{r['r2_run']:.3f}</td>"
        f"<td class=numeric>{r['travel']:+.1f}</td><td>{r['stop']}</td></tr>"
        for r in runs)
    fit = "".join(
        f"<tr><td>{l}</td><td class=numeric>{f['I_ch']:.5f}</td>"
        f"<td class=numeric>{f['tau_c_plus_g']:+.3f}</td>"
        f"<td class=numeric>{f['b_ch']:.4f}</td>"
        f"<td class=numeric>{f['r2']:.4f}</td>"
        f"<td class=numeric>{f['cond']:.1f}</td><td class=numeric>{f['n']}</td></tr>"
        for l, f in res.get("fits", {}).items())
    concl = ""
    if "I_joint" in res:
        concl = (f"<p><b>I_ch = {res['I_ch']:.5f}</b> kg·m²(채널) → "
                 f"<b>I_joint = I_ch·k² = {res['I_joint']:.5f}</b> kg·m² (k={k})")
        if "err_pct" in res:
            concl += (f" · MJCF 예측 {res['I_pred_joint']:.5f} 대비 "
                      f"<b>{res['err_pct']:+.1f}%</b>")
        concl += "</p>"
        if "tau_c" in res:
            concl += (f"<p>절편 분해 — 쿨롱마찰 <b>{res['tau_c']:+.3f} Nm</b> · "
                      f"중력+bias {res['tau_g']:+.3f} Nm (채널). "
                      f"파단토크와 같은 크기여야 <b>교차검증</b>이 된다.</p>")
    return f"""
<h2>{name} (ch{ch}) — 관성 (2단 회귀, 순수 τ_ff)</h2>
<p class=dim>Kp=Kd=0 이라 드라이버가 τ 를 되돌려 계산할 항이 없다 —
<b>순환이 원천 소멸</b>한다. 입력은 우리가 넣은 τ_ff, 출력은 엔코더 q 다.</p>
<p class=dim><b>1단</b>(런 내부): τ 가 상수인 구간에서 q̈ = (τ−τ_c−τ_g)/I − (b/I)·q̇.
q̇ 만 크게 변하므로 조건이 좋다. <b>2단</b>(런 사이): 절편 a_i = τ_i/I − (τ_c+τ_g)/I
⇒ <b>I = 1/기울기</b>.</p>
<table><tr><th>방향</th><th>τ_cmd[Nm]</th><th>1단 절편[rad/s²]</th><th>1단 기울기(−b/I)</th>
<th>q̇범위[dps]</th><th>R²</th><th>이동[°]</th><th>종료</th></tr>{rows}</table>
<table><tr><th>방향</th><th>I_ch[kg·m²]</th><th>τ_c+τ_g[Nm]</th><th>b[Nm·s/rad]</th>
<th>R²</th><th>cond</th><th>n</th></tr>{fit}</table>
{concl}
<div class=warn><b>해석 주의</b><ul>
<li>여기 I 는 <b>채널공간</b>이다. 관절공간은 I·k² (드라이버 기어비 오설정 보정).</li>
<li>단순 "τ vs q̈" 직선은 <b>쓰지 않는다</b> — 점성이 기울기로 새어들어 합성검증에서
<b>+20.3% 편향</b>이 났다. 4모수 전역회귀는 cond 1e15 로 특이하다.</li>
<li>절편의 쿨롱마찰이 <b>파단토크와 다르면</b> 둘 중 하나가 틀렸다. 반드시 대조할 것.</li>
<li>이 시험은 τ_ff <b>스케일</b>을 검증하지 않는다. 명령 τ 가 실제로 그만큼 나오는지는
독립 전류측정이 있어야 알 수 있다(현재 fCurrent 는 fTorque 복제).</li>
</ul></div>
"""
