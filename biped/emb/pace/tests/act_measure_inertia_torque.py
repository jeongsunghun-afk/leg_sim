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
  ★회귀는 반드시 **q̈ 를 τ 에** 한다(x=τ, y=q̈). I = 1/기울기.
    τ 는 우리가 명령한 값이라 **오차가 없고**, 잡음은 전부 q̈ 에 있다. 반대로 놓으면
    (x=q̈) errors-in-variables 감쇠로 기울기가 0 쪽으로 눌린다 —
    합성검증에서 τ_c 산포 12% 일 때 **편향 −19.05%** 였다(올바른 방향은 +1.38%).

★쓰지 않는 방법 넷 (전부 실패를 확인했다):
  ⓪ **τ 를 q̈ 에 회귀**(x=q̈) — 잡음이 x 축에 있어 감쇠한다. τ_c 산포 12% 에서 −19%.
     0% 산포에서는 멀쩡해 보여(편향 +0.02%) 합성검증만으로는 안 드러난다.
     **실기의 지배적 오차원이 마찰 산포**(파단토크 CV 15.6%)라는 걸 알고서야 보였다.
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

import os
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


def design_levels(tau_break, I_ch, vref_dps, travel_deg, vel_cap_dps,
                  skip_s=0.04, half_s=0.05, n=6, u_lo_frac=0.30, margin=0.7):
    """순토크 u=τ−τ_c 의 가용 구간에서 준위를 자동으로 뽑는다.

    ★축마다 파단토크가 다르므로 상수 준위표는 못 쓴다 — HL 0.674 · HR 0.753 이었고,
      HL 기준으로 잡은 0.90 은 HR 에겐 파단의 20% 위밖에 안 돼 Stribeck 구간이었다.

    제약 넷 (하나라도 어기면 창이 못 잡히거나 런이 탈락한다):
      ① 창이 과도구간 뒤:    t_ref = q̇_ref·I/u ≥ skip+half   → u ≤ q̇_ref·I/(skip+half)
      ② 창 끝이 속도상한 전: v(t_ref+half) ≤ vcap            → u ≤ (vcap−q̇_ref)·I/half
      ③ q̇_ref 까지 이동이 범위 안:                            → u ≥ q̇_ref²·I/(2·travel·margin)
      ④ 파단 근처 회피(Stribeck):                             → u ≥ u_lo_frac·τ_c
    """
    v = vref_dps * DEG
    u_hi = min(v * I_ch / (skip_s + half_s),
               (vel_cap_dps * DEG - v) * I_ch / half_s)
    u_lo = max(v * v * I_ch / (2 * travel_deg * DEG * margin), u_lo_frac * tau_break)
    if u_hi <= u_lo:
        return None, (u_lo, u_hi)
    return [tau_break + u for u in np.linspace(u_lo, u_hi, n)], (u_lo, u_hi)


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
    log(f"           모든 준위에서 **같은 속도**의 q̈ 를 읽는다 ⇒ b·q̇ 가 절편으로 빠진다.")
    log(f"           회귀는 q̈~τ (x=τ 정확·y=q̈ 잡음) 로 한다 — 반대로 놓으면 감쇠한다")

    # ── 준위 자동설계 (측정된 파단토크 기준) ──────────────────────────────
    tau_break = joint.get("_tau_break")            # torque 시험이 같은 실행에서 채워준다
    # ★이번 실행에서 토크 프로브를 안 돌렸으면 **spec 의 실측 파단토크**를 쓴다
    #   (2026-08-14). 종전엔 `--tests inertia` 단독이면 tau_break 가 None 이라 상수
    #   준위로 떨어졌는데, 그 상수는 **HL_foot 기준**이다. thigh 에 걸면 이렇게 된다:
    #     준위 [0.9~1.35] · thigh 파단 0.711 → 순토크 0.19~0.64
    #     실측: 8런 중 7런이 "q̇_ref 미도달", +방향 유효표본 1개 → 회귀 불가
    #   파단토크는 이미 8축 다 재서 spec 에 있다. 안 쓸 이유가 없다.
    if tau_break is None:
        _tbs = (spec.get("friction") or {}).get("measured_tau_break_ch") or {}
        if int(ch) in _tbs:
            tau_break = float(_tbs[int(ch)])
            log(f"  [{name}] 파단토크를 spec 실측값에서 가져온다: {tau_break:.3f} Nm "
                f"(이번 실행에서 --tests torque 를 안 돌렸다)")
    if tau_break and cfg.get("auto_levels", True):
        I_des = float(I_pred_joint or 0.054) / max(k_gear ** 2, 1e-9)
        auto, (ulo, uhi) = design_levels(
            tau_break, I_des, vref, travel, v_cap,
            skip_s, half_s, int(cfg.get("n_levels", 6)),
            float(cfg.get("u_lo_frac", 0.30)))
        if auto:
            levels = auto
            log(f"  [{name}] 준위 자동설계 — 파단 {tau_break:.3f} Nm 기준 · "
                f"순토크 u {ulo:.3f}~{uhi:.3f} (비 {uhi/ulo:.2f})")
        else:
            log(f"  [{name}] ⚠준위 자동설계 불가(u 구간 없음 {ulo:.3f}~{uhi:.3f}) — spec 값 사용")
    elif not tau_break:
        log(f"  [{name}] ★★파단토크를 모른다 — spec 상수 준위 {levels} 를 쓴다.\n"
            f"           ⚠이 상수는 **HL_foot 기준**이다. 파단이 더 큰 축에 걸면 순토크가\n"
            f"             거의 0 이 되어 런이 전부 탈락한다(2026-08-14 thigh 에서 8런 중 7런).\n"
            f"           ⇒ `--tests torque,inertia` 로 돌리거나 spec 의\n"
            f"             friction.measured_tau_break_ch 에 그 축을 넣을 것.")

    # ── 방향별 시작점 — 한계상자 끝에서 출발해 **이동거리를 최대로** ────────
    #   HOME 에서 양방향으로 가면 각 방향이 상자의 절반밖에 못 쓴다. 방향마다
    #   반대쪽 끝에서 출발하면 상자 전체를 쓴다(실측 27° → 69°).
    #   ★런이 짧으면 저τ 준위가 q̇_ref 에 못 닿아 **탈락**하고, 탈락은 τ_c 가 큰 쪽에
    #     치우쳐 일어나 회귀를 오염시킨다.
    box = joint.get("_ch_box")
    M = float(cfg.get("box_margin_deg", 3.0))
    q_home = hw.read(ch)[0]
    # ★중력이 큰 축은 **훑는 구간을 중력 0 근처로 좁힌다** (2026-08-14).
    #   상자 전체(70°)를 쓰면 중력이 그만큼 크게 변하고, 그 잔차가 방향 비대칭으로
    #   남아 게이트를 못 넘는다 — HL_thigh 가 실제로 방향편차 15.8%(>15%)로 탈락했다.
    #   ⚠중심만 옮기면 **오히려 나빠진다.** thigh 를 중력 0(+24.5°) 중심으로 70° 훑으면
    #     |τ_g| 변화폭이 4.88 → 6.37 Nm 로 늘어난다(사인의 더 넓은 구간을 지난다).
    #     **폭을 같이 줄여야** 한다: 30° 로 줄이면 2.87 Nm 로 **−41%** 다.
    #   ⇒ `center_by_ch` 와 `travel_by_ch` 를 같이 준다. 둘 중 하나만 주면 안 된다.
    #   ⚠폭을 줄이면 design_levels 의 제약 ③(u ≥ q̇_ref²·I/(2·travel·margin))이 올라
    #     최저준위가 커진다. thigh 30° 에서 1.42~4.01 · 첨두 5.71 < 예산 10.8 로 성립한다.
    _ctr = (cfg.get("center_by_ch") or {}).get(ch)
    _trv = (cfg.get("travel_by_ch") or {}).get(ch)
    starts, travels = {}, {}
    if _ctr is not None and _trv is not None and box:
        lo_b, hi_b = box[0] + M, box[1] - M
        half = float(_trv) / 2.0
        lo, hi = float(_ctr) - half, float(_ctr) + half
        if lo < lo_b or hi > hi_b:                  # 상자를 벗어나면 밀어 넣는다
            sh = max(lo_b - lo, 0.0) + min(hi_b - hi, 0.0)
            lo += sh; hi += sh
            log(f"  [{name}] ⚠탐침 구간이 상자를 벗어나 {sh:+.1f}° 밀었다")
        for d in (+1.0, -1.0):
            starts[d] = lo if d > 0 else hi
            travels[d] = hi - lo
        _g = getattr(hw, "grav_fn", None)
        _sp = (max(_g(ch, x) for x in np.linspace(lo, hi, 40))
               - min(_g(ch, x) for x in np.linspace(lo, hi, 40))) if _g else float("nan")
        log(f"  [{name}] ★중력 최소 구간에서 훑는다 — 중심 {float(_ctr):+.1f}° · "
            f"폭 {hi-lo:.0f}° · |τ_g| 변화폭 {_sp:.2f} Nm")
    else:
        for d in (+1.0, -1.0):
            if box:
                lo, hi = box[0] + M, box[1] - M
                starts[d] = lo if d > 0 else hi
                travels[d] = min(hi - lo, float(cfg.get("travel_max_deg", 70.0)))
            else:
                starts[d] = q_home
                travels[d] = travel
    log(f"  [{name}] 방향별 시작 — +{starts[+1.0]:.1f}°(이동 {travels[+1.0]:.0f}°) · "
        f"−{starts[-1.0]:.1f}°(이동 {travels[-1.0]:.0f}°)")

    # ★첫 goto 전에 **무장한다** (2026-08-14, 실기에서 잡혔다).
    #   루프 안의 `arm(ch, 0, 0)` 은 **토크램프용**(게인을 0 으로 내리는 것)이고,
    #   그 바로 앞 `goto` 는 위치게인으로 움직이므로 **이미 무장돼 있어야** 한다.
    #   순서가 뒤집혀 있었는데 `--tests torque,inertia` 로 돌리면 토크 프로브가 먼저
    #   arm 해서 가려졌다 — `--tests inertia` **단독은 한 번도 돈 적이 없다.**
    #   ⚠arm 은 지금 있는 자리를 래치하고 게인을 0→목표로 램프하므로 충격이 없다.
    hw.arm(ch, brake_kp, brake_kd)

    # ★중력을 **명령에 실어 상쇄한다** (2026-08-14, 실기 HL_calf 에서 잡혔다).
    #   `step_torque` 는 `_raw_write` 를 안 거치므로 hw.tau_ff_fn(중력 FF)이 **안 실린다** —
    #   순수 명령토크다. 그래서 실제 순토크는 방향마다 다르다:
    #       + 방향: τ − τ_c + |τ_g|      − 방향: τ − τ_c − |τ_g|
    #   HL_calf 실측(2026-08-14): 마찰 f≈0.60 · 중력 g≈0.55 Nm 이라
    #     + 는 저항이 0.04 Nm 뿐이라 **0.10s 만에 속도상한**에 닿고(창이 런 전체를 덮는다)
    #     − 는 준위 1.05~1.20 사이에서야 파단해 낮은 준위 셋이 통째로 날아갔다
    #       → −방향 유효표본 2개 < 3 → **회귀 자체를 못 했다.**
    #   ⇒ 명령을 `τ_g(q) + direction·level` 로 준다. 그러면 순토크가 방향대칭이 되고,
    #     회귀의 x 는 **가진준위 그대로**이며 절편에서 τ_g 가 빠진다(b·q̇_ref + τ_c 만 남는다).
    #   ⚠q 마다 다시 읽어야 한다 — 70° 를 움직이는 동안 중력이 크게 변한다.
    _gfn = getattr(hw, "grav_fn", None)

    def grav_at(q):
        """채널각 q 에서의 중력토크[Nm]. actuator_test 가 홈복귀 전에 **실측 보정**을
        얹어 hw.grav_fn 에 넣어 둔다 — 표를 직접 읽으면 solo 자세에서 틀린다."""
        if _gfn is not None:
            return float(_gfn(ch, q))
        return 0.0

    if _gfn is None:
        log(f"  [{name}] ⚠중력함수가 없다 — 상쇄 없이 돈다. 방향 비대칭이 남는다")
    else:
        # ★상한을 상쇄분만큼 올린다. 안 그러면 `step_torque` 의 clip 이 **가진을 먼저**
        #   깎아 준위가 거짓이 된다 — 회귀의 x 가 틀리면 I 가 통째로 틀린다.
        _lo_b, _hi_b = (joint.get("_ch_box") or (-60.0, 60.0))
        _gmax = max(abs(grav_at(q)) for q in np.linspace(_lo_b, _hi_b, 41))
        tau_max = tau_max + _gmax
        log(f"  [{name}] 중력상쇄 켬 — |τ_g| 최대 {_gmax:.3f} Nm ⇒ 토크상한 {tau_max:.2f} Nm")
        # ★**트립 예산**을 미리 확인한다 (2026-08-14). 종전엔 tau_max 만 올리고
        #   안전트립(τ_trip)과 대조하지 않아 **런 도중에** 죽었다:
        #     HL_thigh 중력 6.67 + 파단 0.73 + 가진 3.27 = 10.7Nm > τ_trip 8.0
        #     → "토크 한계 |−10.36| > 8.0 이 50ms 지속" 으로 2번째 준위에서 중단
        #   중력이 작은 축은 예산이 남아 안 걸렸을 뿐이다(foot 0.24+0.6+0.73=1.6).
        #   ⇒ 필요 첨두를 계산해 **준위를 깎는다.** 못 깎으면 그 자리에서 멈춘다 —
        #     런을 반쯤 돌린 뒤 죽는 것보다 낫다.
        _trip = float(getattr(hw.limits_for(ch), "tau_trip", 0.0) or 0.0)
        if _trip > 0 and levels:
            _need = _gmax + max(levels)
            _room = _trip * 0.90 - _gmax          # 가진에 쓸 수 있는 몫
            if _need > _trip * 0.90:
                if _room <= min(levels):
                    raise SafetyAbort(
                        f"{name}: 토크예산 부족 — 중력만 {_gmax:.2f}Nm 인데 τ_trip 이"
                        f" {_trip:.1f}Nm 이다. 최저준위 {min(levels):.2f}Nm 도 못 넣는다.\n"
                        f"  ⇒ spec 의 tau_trip_nm 을 올리거나(시험 전용), 중력이 작은"
                        f" 자세에서 잴 것.")
                _old = list(levels)
                levels = [x for x in levels if x <= _room]
                log(f"  [{name}] ⚠토크예산 — 중력 {_gmax:.2f} + 준위최대 {max(_old):.2f}"
                    f" = {_need:.2f} > τ_trip {_trip:.1f}×0.9")
                log(f"           준위를 {len(_old)}→{len(levels)}개로 깎는다"
                    f" (상한 {_room:.2f}Nm): {[round(x,2) for x in levels]}")
                if len(levels) < 3:
                    raise SafetyAbort(
                        f"{name}: 예산 안에 남는 준위가 {len(levels)}개뿐이다(최소 3)."
                        f" τ_trip 을 올리거나 중력이 작은 자세에서 잴 것.")

    n_rep = int(cfg.get("repeats", 1))
    runs = []
    try:
      for rep in range(n_rep):
        if n_rep > 1:
            log(f"  [{name}] ── 반복 {rep+1}/{n_rep} ──")
        for direction in (+1.0, -1.0):
            q_start, travel_d = starts[direction], travels[direction]
            for tau in levels:
                hw.goto(ch, q_start, brake_kp, brake_kd, speed_dps=20.0)  # 시작점으로
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
                    # ★중력상쇄 + 가진. tau_max 는 상쇄분만큼 여유가 있어야 한다.
                    s = hw.step_torque(ch, grav_at(hw._q[ch]) + direction * tau, tau_max)
                    T.append(t); Q.append(s.q_deg); V.append(s.dq_dps); TAU.append(s.tau)
                    if abs(s.q_deg - q0) >= travel_d:
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
                # ★창이 런의 대부분을 덮으면 **공통속도법이 아니다** (2026-08-14).
                #   창이 런의 대부분을 덮으면 q̈ 는 "q̇_ref 에서의 값" 이 아니라
                #   **런 전체 평균가속도**가 되고, b·q̇_ref 가 절편으로 안 빠진다.
                #   이 파일이 방법 ③ 을 버린 이유(런이 시상수의 7~14%)와 같은 병이다.
                #   ⚠조용히 통과시키면 I 가 그럴듯한 값으로 나와서 더 위험하다.
                # ★2026-08-14 정정 — 이 가드를 넣을 때 든 근거(실기 HL_calf 에서 91%,
                #   107% 가 나왔다)는 **틀렸다.** 표본율을 500Hz 로 가정했는데 실제는
                #   1000Hz(실측 801Hz)다. 다시 계산하면 46표본×1.25ms/0.101s = **57%**,
                #   72표본/0.135s = **67%** 로 전부 정상 범위였다.
                #   ⇒ 그때 실패의 원인은 **중력 비대칭 하나**였고 그 수정이 효과를 냈다
                #     (I_joint MJCF 대비 −29.9% → −0.6%).
                #   가드 자체는 실측 dt 를 쓰므로 유효하고, 관측된 최대 덮음은 71% 다
                #   (설계 한계와 일치). 상한 0.80 은 그대로 두되 **근거를 바로잡아 둔다** —
                #   틀린 숫자를 근거로 나중에 이 값을 조이면 정상 준위를 버리게 된다.
                if "fail" not in f1 and t.size > 2:
                    _cov = f1["n_win"] * float(np.median(np.diff(t))) / max(float(t[-1]), 1e-9)
                    if _cov > float(cfg.get("win_cover_max", 0.60)):
                        f1 = {"fail": f"창이 런의 {_cov*100:.0f}% 를 덮는다"
                                      f"(상한 {float(cfg.get('win_cover_max', 0.60))*100:.0f}%)"
                                      f" — 런이 너무 짧다. 준위를 낮추거나 이동을 늘릴 것"}
                sgn = '+' if direction > 0 else '−'
                if "fail" in f1:
                    log(f"    {sgn} τ={tau:.2f}: ✗ {f1['fail']}  "
                        f"(런 {t[-1]:.3f}s/{len(T)}표본 · {hit})")
                    continue
                runs.append({"rep": rep, "dir": direction, "tau_cmd": tau,
                             "signed_tau": direction * tau,
                             "ddq": f1["ddq"], "v_at": f1["v_at"], "t_at": f1["t_at"],
                             "n_win": f1["n_win"], "v_max": f1["v_max"],
                             "travel": float(q[-1]), "stop": hit, "dur": float(t[-1]),
                             "t": t, "q": q, "dq": v, "tau": np.array(TAU)})
                log(f"    {sgn} τ={tau:.2f}Nm → q̈|q̇={f1['v_at']:5.1f}dps = "
                    f"{f1['ddq']:+7.2f} rad/s²  (t={f1['t_at']:.3f}s · 창표본 {f1['n_win']} · "
                    f"런 {t[-1]:.3f}s · {hit})")
    finally:
        hw.goto(ch, q_home, brake_kp, brake_kd, speed_dps=20.0)   # 끝나면 HOME 으로

    res = {"ch": ch, "name": name, "levels": levels, "runs":
           [{kk: vv for kk, vv in r.items() if kk not in ("t", "q", "dq", "tau")}
            for r in runs], "gear_k": k_gear}

    # ★원시궤적을 남긴다 (2026-08-14). 토크 프로브는 남기는데 여기만 빠져 있었다.
    #   ⚠지금 이게 없어서 **속도 의존성을 검증할 수 없다.** 2026-08-14 HL_foot 에서
    #     q̇_ref 훑기가 I −20%(40→140dps) · 절편기울기 0.20 Nm·s/rad(관절) 을 냈고
    #     두 조건(무릎 자유·고정)에서 재현됐다. 원인 후보가 셋인데(점성마찰 · 토크-속도
    #     derating · 창 평균 편향) **원시 q̈~q̇ 곡선이 있어야** 가른다.
    #   ⇒ 런마다 (t, q, dq, tau) 를 통째로 저장한다. 준위·방향·반복을 같이 넣어야
    #     나중에 어느 런인지 안다.
    try:
        from hwio import raw_trace_dir
        _d = raw_trace_dir(plotdir); os.makedirs(_d, exist_ok=True)
        outp = os.path.join(_d, f"inertia_ch{ch:02d}.npz")
        np.savez(outp, cols=np.array(["t", "q", "dq", "tau"]),
                 gear_k=k_gear, vref=vref, levels=np.array(levels, float),
                 tau_break=float(joint.get("_tau_break") or np.nan),
                 hold=np.array(sorted(getattr(hw, "hold_ch", []) or []), int),
                 # ★**측정 자세**를 남긴다 (2026-08-14). 이게 없으면 MJCF 대조가 안 된다.
                 #   유효관성은 자세에 크게 의존한다 — HL_thigh 를 중립에서 계산하면
                 #   0.155(solo)/0.168(홀드)인데, 실제 solo 자세(calf −61°·foot +62° 로
                 #   접힘)에서는 **0.185/0.221** 이다. 20~31% 차이다.
                 #   ⚠스크립트가 찍는 "MJCF 예측" 은 `I_total_pred`(= I_link+I_rotor·N²)로
                 #     **고립축** 값이다. 아래 subtree 가 작은 foot·calf 에서는 우연히
                 #     가까웠지만 thigh 에서는 안 맞는다(실측 0.261 vs 0.169 → +55%,
                 #     자세 반영 홀드 기준으로는 +18%).
                 #   ⇒ 제대로 대조하려면 **이 자세 + 이 홀드조건**으로 M⁻¹ 을 풀어야 한다.
                 #     그건 mujoco 가 있는 곳(오프라인)에서 한다.
                 q_pose_ch=np.asarray(hw._q[:hw.n], float),
                 **{f"r{i}": np.column_stack([r["t"], r["q"], r["dq"], r["tau"]])
                    for i, r in enumerate(runs)},
                 meta=np.array([[r["rep"], r["dir"], r["tau_cmd"]] for r in runs], float))
        log(f"  [{name}] 원시 궤적 저장: {os.path.relpath(outp)} ({len(runs)} 런)")
    except Exception as e:                       # 저장 실패가 시험을 죽이면 안 된다
        log(f"  [{name}] ⚠원시 궤적 저장 실패: {type(e).__name__}: {e}")

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
        # ★x=τ(정확), y=q̈(잡음). I = 1/기울기.  ⚠반대로 놓으면 감쇠한다(모듈 주석 ⓪)
        x = np.array([r["tau_cmd"] for r in sel])
        y = np.array([r["ddq"] * d for r in sel])
        A = np.column_stack([x, np.ones_like(x)])
        th, *_ = np.linalg.lstsq(A, y, rcond=None)
        if abs(th[0]) < 1e-9:
            warn.append(f"{lbl}방향 기울기 0 — 회귀 불가")
            continue
        r2 = 1.0 - np.sum((y - A @ th) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
        fits[lbl] = {"I_ch": float(1.0 / th[0]),
                     "intercept": float(-th[1] / th[0]),   # τ 축 절편 = b·q̇_ref+τ_c+τ_g
                     "r2": float(r2), "n": len(sel), "cond": float(np.linalg.cond(A))}
        # ⚠th 를 그대로 찍지 말 것 — 회귀를 q̈~τ 로 바꾼 뒤 th[0]=1/I, th[1]=q̈축 절편이다.
        #   dict 에 담은 환산값(I_ch, intercept)을 찍어야 한다.
        f_ = fits[lbl]
        log(f"  [{name}] {lbl}방향: I_ch={f_['I_ch']:.5f} kg·m²(채널) · "
            f"절편={f_['intercept']:+.3f} Nm · R²={r2:.4f} · "
            f"cond={f_['cond']:.1f} (n={len(sel)})")
    res["fits"] = fits
    for w in warn:
        log(f"  [{name}] ⚠ {w}")
    res["warnings"] = warn

    # ── ★q̇_ref 훑기 — **같은 런에서 b(점성)까지 뽑는다** (2026-08-14) ──────
    #   위 회귀의 절편은 설계상 `b·q̇_ref + τ_c(q̇_ref) + τ_g` 라 셋이 뭉쳐 있다.
    #   그런데 그건 **q̇_ref 의 일차식**이다. 같은 런을 여러 q̇_ref 에서 다시 읽어
    #   절편을 q̇_ref 에 회귀하면 기울기가 곧 `b + dτ_c/dq̇` 다.
    #   ⇒ **하드웨어를 다시 안 돌린다.** 런 하나에 여러 속도가 이미 다 들어 있다.
    #
    #   ★★그런데 이 b 는 **값이 아니라 괄호다.** 합성검증(참값 기지, 이 파일의
    #     t_vref_sweep_synth)에서 I 는 ±1% 로 되찾는데 b 는 이렇게 갈렸다:
    #         half_s   0.02      0.03      0.05      0.08
    #         I 오차   −0.0%    −0.4%    +0.5%    +0.9%
    #         b 오차  −57.2%   +39.3%   +22.5%   +36.0%
    #       참값별(half_s=0.03): b=0.004 → **−110%(부호까지 뒤집힘)** · 0.008 → −19%
    #                            0.012 → +39% · 0.020 → +39% · 0.040 → +5%
    #     원인은 `_ddq_at_speed` 가 **고정 시간창**(±half_s)에서 q̈ 를 읽기 때문이다.
    #     창 안에서 속도가 변하므로 `b·q̇_ref` 가 정확히 그 속도의 값이 아니고, 그
    #     잔차가 q̇_ref 마다 달라 기울기에 그대로 실린다. I 는 **준위 간 차이**로
    #     구해지니 이 편향이 상쇄되지만, b 는 **절편의 미세한 기울기**라 안 상쇄된다.
    #     ⚠R² 는 방패가 안 된다 — 위 +51% 사례의 R² 가 **0.958** 이었다.
    #   ⇒ 결론: **JDAMP 를 이걸로 못박지 말 것.** 다만 PACE 의 JDAMP 상자가 지금
    #     ×0.1~10(폭 100배)이므로, ×[0.5, 2](폭 4배)로 **좁히는 근거**로는 쓸 수 있다.
    #     그게 이 값의 유일한 용도다.
    #
    #   ★왜 지금 이게 필요한가 — JDAMP 는 **어느 방법으로도 못 얻고 있었다.**
    #     각축 마찰법: 등속 구간에서 재므로 q̈=0, b 는 τ_c 와 완전히 섞인다(8축 전부 nan).
    #     PACE(다축):  JDAMP↔JFRIC r=+0.93 축퇴라 평탄방향. 2026-08-14 적합에서
    #                  JDAMP.foot 이 상자 **상한 95%**, JDAMP.hip 이 **하한 1%** 로 갈렸다.
    #     이 방법만 q̈ 를 실제로 만들어 놓고 b 를 각축에서 직접 본다.
    #
    #   ⚠기울기는 `b + dτ_c/dq̇` 이지 b 가 아니다. 둘을 가르려면 **마찰-속도 곡선**이
    #     필요한데 우리는 그걸 쟀다(act_measure_friction 의 속도 훑기):
    #       calf 는 2~120dps 에서 평탄(±3%) ⇒ dτ_c/dq̇ ≈ 0 이라 기울기 = b 로 읽어도 된다.
    #       thigh 는 40→60dps 에서 −30% 로 꺾인다 ⇒ **기울기를 b 로 읽으면 안 된다**(음수로 나온다).
    #     그래서 아래는 기울기를 그대로 찍고, 판정은 사람이 그 곡선을 보고 한다.
    #   ⚠I_ch 가 q̇_ref 마다 크게 달라지면 **공통속도법의 전제가 깨진 것**이다
    #     (그 속도에 못 미친 런이 섞였거나 창이 겹쳤다). 그래서 같이 찍는다.
    vsweep = [float(x) for x in cfg.get("vref_sweep_dps", []) if float(x) > 0]
    if vsweep and runs:
        rows = []
        for vr in sorted(vsweep):
            ff = {}
            for d, lbl in ((+1.0, "+"), (-1.0, "−")):
                pts = []
                for r in runs:
                    if r["dir"] != d:
                        continue
                    # dt 는 **그 런의 실측 중앙값** — 위 1차 회귀와 같은 값을 써야 한다
                    dt_r = (float(np.median(np.diff(r["t"]))) if r["t"].size > 2 else hw.dt)
                    g = _ddq_at_speed(r["t"], r["q"], r["dq"], dt_r, vr, skip_s, half_s)
                    if "fail" not in g:
                        pts.append((r["tau_cmd"], g["ddq"] * d))
                if len(pts) < MIN_PTS:
                    continue
                x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
                A = np.column_stack([x, np.ones_like(x)])
                th, *_ = np.linalg.lstsq(A, y, rcond=None)
                if abs(th[0]) > 1e-9:
                    ff[lbl] = (float(1.0 / th[0]), float(-th[1] / th[0]), len(pts))
            if len(ff) == 2:      # 양방향 다 있을 때만 — 한쪽만이면 중력이 안 빠진다
                rows.append({"vref": vr,
                             "I_ch": float(np.mean([v[0] for v in ff.values()])),
                             "intercept": float(np.mean([v[1] for v in ff.values()])),
                             "n": int(sum(v[2] for v in ff.values()))})
        res["vref_sweep"] = rows
        if len(rows) >= 3:
            vv = np.array([r["vref"] for r in rows]); ic = np.array([r["intercept"] for r in rows])
            II = np.array([r["I_ch"] for r in rows])
            A = np.column_stack([vv, np.ones_like(vv)])
            th, *_ = np.linalg.lstsq(A, ic, rcond=None)
            r2 = 1.0 - np.sum((ic - A @ th) ** 2) / max(np.sum((ic - ic.mean()) ** 2), 1e-12)
            b_ch = float(th[0]) / DEG          # Nm/(deg/s) → Nm·s/rad
            res["b_ch"] = b_ch
            res["b_joint"] = b_ch * k_gear ** 2
            res["b_r2"] = float(r2)
            res["tau_c0"] = float(th[1])
            I_spread = float((II.max() - II.min()) / max(abs(II.mean()), 1e-12))
            res["I_vref_spread"] = I_spread
            log(f"  [{name}] ★q̇_ref 훑기 — 같은 런을 {len(rows)} 속도에서 다시 읽었다")
            log(f"           {'q̇_ref[dps]':>11}{'I_ch':>10}{'절편[Nm]':>11}{'n':>5}")
            for r in rows:
                log(f"           {r['vref']:>11.0f}{r['I_ch']:>10.5f}{r['intercept']:>11.3f}{r['n']:>5}")
            log(f"           절편 기울기 = {b_ch:+.4f} Nm·s/rad(채널) "
                f"= {res['b_joint']:+.4f}(관절, ×k²={k_gear**2:.2f}) · R²={r2:.4f}")
            log(f"           절편0(q̇→0) = {th[1]:+.3f} Nm = τ_c(0)+τ_g")
            # ★값이 아니라 **괄호**로 낸다. 합성검증에서 −110%~+39% 로 갈렸다(위 주석).
            ok_b = (b_ch > 0) and (r2 >= 0.8) and (I_spread <= 0.15)
            if ok_b:
                res["b_joint_bracket"] = [res["b_joint"] * 0.5, res["b_joint"] * 2.0]
                log(f"           ⇒ JDAMP 괄호(관절) **[{res['b_joint']*0.5:.4f}, "
                    f"{res['b_joint']*2:.4f}]** — 못박는 값이 **아니다**.")
                log(f"             합성검증 b 오차 −110%~+39% (I 는 ±1%). PACE 의 JDAMP "
                    f"상자를 ×0.1~10 → 이 괄호로 좁히는 데만 쓸 것")
            else:
                why = ("기울기 음수" if b_ch <= 0 else
                       f"직선성 R²={r2:.3f}<0.8" if r2 < 0.8 else
                       f"I 가 속도마다 {I_spread*100:.0f}% 흔들림>15%")
                log(f"           ✗b 괄호 없음 — {why}. **JDAMP 는 이 시험으로 못 얻었다**")
            log(f"           ⚠기울기는 **b + dτ_c/dq̇** 다. 마찰-속도 곡선이 평탄한 축"
                f"(calf: 2~120dps ±3%)만 b 로 읽을 것. thigh 는 40→60dps 에서 −30% 라 안 된다")

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
            tb = joint.get("_tau_break")
            if tb:
                # τ_c(q̇_ref) = 절편 − b·q̇_ref. 정지마찰(파단)보다 낮으면 Stribeck.
                log(f"  [{name}] 절편 {ic:+.3f} Nm = b·q̇_ref + τ_c(q̇={vref:.0f}dps) + τ_g")
                log(f"           vs **자기 파단토크(정지마찰) {tb:.3f} Nm** → "
                    f"{(ic/tb-1)*100:+.0f}%"
                    + ("  ⇒ 운동마찰이 정지마찰보다 낮다(Stribeck)" if ic < tb * 0.95 else ""))
            else:
                log(f"  [{name}] 절편 {ic:+.3f} Nm = b·q̇_ref + τ_c + τ_g "
                    f"(파단토크 미측정 — `--tests torque,inertia` 로 같이 돌리면 대조된다)")
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
