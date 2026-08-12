#!/usr/bin/env python3
"""design_excitation.py — 가진 궤적의 **식별가능성**을 설계 단계에서 잰다 (하드웨어 미접촉).

═══ 참조 원문 (2026-08-12 확인) ═════════════════════════════════════════════
  PACE = **Precise Adaptation through Continuous Evolution** (ETH RSL)
    논문  arXiv:2509.06342 "Towards bridging the gap: Systematic sim-to-real
          transfer for diverse legged robots"  https://arxiv.org/abs/2509.06342
    코드  https://github.com/leggedrobotics/pace-sim2real   문서 https://pace.filipbjelonic.com
  ⚠종전 사내 문서(sim2real_checklist_17dof.html:57)는 "ETH RSL, CMA-ES" 한 줄뿐이고
    가리키던 메모리도 없어서 **궤적 사양이 없는 줄 알았는데 원문에는 있다.**

  ── 원문이 명시한 가진 (§3.2.2 full-robot in-air) ──
      전 축 **동시** chirp · **0.1~10 Hz** · **20~60s**(주로 20~40s)
      **base 매달림(in-air), 다리 자유, 접촉 없음**       ← 우리 설정과 같다
      ★**대칭 궤적으로 net wrench 를 상쇄**한다             ← 우리는 안 한다(아래)
      PD 위치추종(Tytan: Pτ=60 Nm/rad, Dτ=2 Nms/rad) · 로깅 400 Hz
      파라미터 **p = [I_a, d, τ_f, q̃_b, T_d] ∈ R^(4n+1)**  ← bias·지연 **포함**
      목적함수 ℓ = (1/k)Σ‖q_real − q_sim‖²                 ← 우리와 동일
      검증 **unseen PD gains**(Tytan 60/2 → **145/5**) + unseen 궤적
      단계 ① single-drive(0.1~10Hz, 5 부하 × 3 PD) ② in-air ③ on-ground
      설계규칙 *"trajectories cover up to **f_policy/2**(정책 Nyquist)"*
      ⚠구조 한계 인정: **ANYmal 은 2 Hz**, 다른 플랫폼 8~10 Hz

  ── 우리와 갈리는 곳 (실행 전 판단 필요) ──
    1. f_end 1.55Hz vs 10Hz — 단 우리는 **ANYmal 과 같은 처지**(구조·토크·속도 한계).
       아래 f1scale 실험 참조: 올리면 실제로 좋아지지만 q̇·τ 예산이 먼저 터진다.
    2. **q̃_b(관절 bias)·T_d(전역 지연)가 우리 탐색에 없다.** 우리는 지연을 따로 재고
       (Δ=8.39ms) CMA-ES 에 안 넣는다. 원문은 넣는다 — 안 넣으면 그 오차가
       armature/마찰로 흡수될 수 있다.
    3. **net wrench 상쇄** — 원문은 대칭 명령으로 base 반력을 없앤다. 우리는 황금비
       위상이라 상쇄되지 않는다. 시뮬은 base 를 고정해 문제가 없지만 **실기는 크레인에
       매달려 흔들린다.** 그 흔들림은 모델에 없다.
    4. 검증이 **unseen PD gains** 다. 우리 계획(hold-out 궤적)보다 강하다 —
       게인을 바꿔도 같은 θ 가 나오는지가 kp 순환 우려를 직접 친다.

  ── ★원문이 **하지 않은** 것 ──
    correlation matrix 없음 · Cramér-Rao/Fisher 없음 · 민감도 분석 없음.
    *"damping/friction separation assumed but not validated independently"* —
    즉 아래 JDAMP↔JFRIC 축퇴는 **원문도 다루지 않은 빈틈**이다. 이 도구가 그 자리다.
═════════════════════════════════════════════════════════════════════════════

★왜 필요한가 — 지금 설계검사(collect_multichirp --dry)는 **축간 상관**을 본다. 그건 대리지표다.
  PACE 의 목적함수는
        cost(θ) = Σ_t Σ_i ( q_sim,i(t;θ) − q_real,i(t) )²
  이고 θ 근처에서 선형화하면
        Δcost ≈ ‖S·Δθ‖²,      S = ∂q_sim/∂θ        (감도행렬)
  ⇒ 좋은 가진의 조건은 둘이고, **둘 다 파라미터(θ) 기준이지 축 기준이 아니다**:
      (1) 각 파라미터의 ‖S_p‖ 가 **측정 노이즈보다 클 것**  — 아니면 애초에 식별 불가
      (2) S 의 열들이 **서로 평행하지 않을 것**             — 평행하면 두 파라미터가 맞바꿔진다
  축간 상관을 낮추는 건 (2)에 도움이 되지만 (2) 자체가 아니다.
  ★반례가 우리 구성에 실제로 있다: `ROTOR_I` 는 **8축이 공유**하는 하나의 값이라,
    축 궤적을 아무리 비상관화해도 `JDAMP` 와 평행해질 수 있다. 축 상관은 그걸 못 본다.

★여태 확인된 것은 `ROTOR_I` **하나뿐**이다
  2026-08-11 손계산: `ROTOR_I` +10% → 궤적변화 0.058°, 잡음 0.02° 대비 **SNR 3**
  (a348781 커밋 메시지 · NEXT_HW.md:745). 그 계산은 **코드에 없고** 나머지 8개
  (JDAMP×4 · JFRIC×4)는 감도를 재 본 적이 없다. 이 도구가 그 자리를 메운다.

★측정 방식은 CMA-ES 가 실제로 하는 것과 **같아야 한다**
  pace_cmaes 는 `--window`(0.5s)마다 실측 상태로 재초기화한다. 그래서 여기서도 섭동
  롤아웃을 **공칭 궤적 상태로 창마다 재초기화**한다. 30초를 통짜 개루프로 적분한 감도를
  재면 적분 발산분이 섞여 실제 식별력보다 **크게** 나온다(낙관 편향).

★한계 — 이건 **시뮬 안에서의 감도**다
  모델이 틀린 부분(미모델 마찰·백래시·지연)은 여기 안 잡힌다. "SNR 이 충분하다"는
  "이 모델이 맞다면 식별 가능하다"는 뜻이지 "실측이 잘 나온다"가 아니다.

═══ 2026-08-12 이 도구로 얻은 결론 ═══════════════════════════════════════════

★현행 궤적(spec.pace_multi, 30s)의 평가 — 표본 240,000
    ROTOR_I      RMS 0.087° · 분해능 0.0% · 단독대비 1.0x   ← 깨끗하다
    JFRIC×4      RMS 0.009~0.030°
    JDAMP×4      RMS 0.001~0.005°   ← ROTOR_I 의 1/16 ~ 1/87
    조건수 5.1 · 축간 상관 0.149(설계값과 일치)
  ⇒ ROTOR_I 는 이 가진으로 잘 잡힌다. 어제 손계산(SNR 3)과 같은 결론이다.

★★그러나 `JDAMP.foot ↔ JFRIC.foot` 상관이 **+0.926** 이다 — 맞바꿔진다.
  물리적으로 당연하다: foot 최고속(146dps=2.55rad/s)에서 점성토크는
  0.02×2.55 = 0.051 Nm 로 쿨롱 0.44 Nm 의 **12%** 밖에 안 된다.
  둘 다 sign(q̇) 를 따라가는 거의 평행한 신호가 된다.

  ⚠**궤적 모양으로 고치려는 시도 둘 다 실패했다**(재시도 방지용으로 기록):
    ① dual — 느리고 큰 성분을 겹쳐 저속·고속을 동시에 만든다
         비율 0.30 → r 0.946→**0.957** · 조건수 6.2→7.0
         비율 0.45 → r 0.946→**0.965** · 조건수 6.2→7.7
       진폭 예산을 나눠 최고속이 186→106dps 로 떨어졌고, 느린 성분이 계속 움직여
       q̇≈0 체류가 오히려 줄었다. **정확히 반대 방향이었다.**
    ③ f1scale — f_end 를 올린다 (원문의 0.1~10Hz 를 따라가 본다)
         2배(f_end 3.0Hz) → r 0.934→0.922 · 조건수 5.5→5.1 · JDAMP.foot 감도 +59%
         4배(f_end 6.0Hz) → r **0.888**(혼동 해소) · 조건수 **4.4** · 감도 +70%
       ★**방향은 맞다** — 원문이 10Hz 를 쓰는 이유가 이것이다. 그런데 못 쓴다:
             최대 q̇  2배 372 dps · 4배 742 dps   (상한 **200**)
         진폭을 줄여 q̇ 를 맞추면 관성토크 ∝ A·f² 라 2배에서 27.8 Nm(트립 16)로 터진다.
       ⇒ 우리 f_end 1.55Hz 는 게으름이 아니라 **q̇·τ 예산의 결과**다. 원문도 ANYmal 을
         2 Hz 로 제한했다고 명시한다 — 우리가 그 부류다. 올리려면 하드웨어(트립 여유·
         감속비 오설정 보정)를 먼저 풀어야 한다.
    ② f0scale — f_start 를 낮춰 저속 구간을 늘린다
         0.3배 → r 0.936 · 조건수 5.7 · 축간 상관 0.400→**0.562**
         0.1배 → r 0.932 · 조건수 5.5 · 축간 상관 0.400→**0.606**
       상관은 0.946→0.932 로 **거의 안 움직이고** 축간 상관만 나빠진다.
       전 축이 초반을 함께 느리게 돌기 때문이다.

  ⇒ **해법은 궤적이 아니다.** JDAMP 또는 JFRIC 중 하나를 축별 시험
    (q̇_ref 를 훑는 마찰-속도 곡선, NEXT_HW §B)으로 **먼저 못박고** CMA-ES 에서 고정할 것.
    NEXT_HW 가 그걸 "(선택)" 으로 적어 뒀는데, 이 분석 기준으로는 **선택이 아니다.**
    같은 결론이 pace_cmaes 셀프테스트에서도 나왔다 — 6세대에서 틀어지는 건 JDAMP 쪽이다.

★분해능 숫자는 **낙관값**이다(백색·독립 잡음 가정). 믿을 것은 **순위와 혼동쌍**이지
  "분해능 0.1%" 같은 절대값이 아니다. 실제 한계는 잡음이 아니라 모델오차다.

사용:
    ~/.venv-mujoco/bin/python design_excitation.py                # 현재 spec 평가(30s)
    ~/.venv-mujoco/bin/python design_excitation.py --T 10         # 짧게(빠른 확인)
    ~/.venv-mujoco/bin/python design_excitation.py --dual 0.3,0.1 # 변형안 평가
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.dirname(HERE)
BIPED = os.path.dirname(EMB)
sys.path[:0] = [HERE, os.path.join(HERE, "tests"), os.path.join(EMB, "interface")]

import pace_cmaes as P                                    # noqa: E402
from joint_map import JointMap                            # noqa: E402

DEG = np.pi / 180.0


def build_traj(mc, jm, cfg_all, T, rate, dual=None):
    """collect_multichirp 과 **같은 함수**를 쓴다 — 설계와 수집이 갈리면 분석이 무의미해진다.

    dual=(ratio, f_slow): ★JDAMP↔JFRIC 분리를 겨냥한 변형.
      점성 `b·q̇` 와 쿨롱 `τ_c·sign(q̇)` 는 **속도 크기가 한 종류면 구분되지 않는다**
      (한 속도에서는 둘 다 그냥 상수 토크로 보인다). 고정진폭 처프는 각 주파수에서
      속도 분포가 비슷해 이 축퇴가 남는다.
      ⇒ 같은 축에 **느리고 큰** 성분을 겹쳐 저속 구간(쿨롱 지배)과 고속 구간(점성 지배)을
        **동시에** 만든다. 진폭 예산은 나눠 쓴다(합이 종전 진폭을 넘지 않게).
    """
    import collect_multichirp as cm
    n = jm.n_leg
    amps, f0, k, phi = cm.chirp_bank(mc, n, T)
    ramp = float(mc.get("ramp_s", 2.0))
    home = np.array([float(x) for x in cfg_all["home"]["q_deg"]])[:n]
    tt = np.arange(0.0, T, 1.0 / rate)
    if dual is None:
        q_cmd = np.array([home + cm.q_at(t, amps, f0, k, phi, ramp) for t in tt])
        return tt, q_cmd, home, dict(amp=amps, f0=f0, f1=f0 + k * T, phi=phi)
    ratio, f_slow = dual
    a_fast = amps * (1.0 - ratio)
    a_slow = amps * ratio
    # 느린 성분도 축마다 위상을 벌린다(황금비) — 안 그러면 8축이 같이 움직여 상관이 생긴다
    psi = ((np.arange(n) + 0.5) * 0.6180339887) % 1.0 * 2 * np.pi
    rows = []
    for t in tt:
        s = min(t / ramp, 1.0) if ramp > 0 else 1.0
        fast = cm.q_at(t, a_fast, f0, k, phi, ramp)
        slow = a_slow * s * np.sin(2 * np.pi * f_slow * t + psi)
        rows.append(home + fast + slow)
    return tt, np.array(rows), home, dict(amp=amps, f0=f0, f1=f0 + k * T, phi=phi,
                                          dual=(ratio, f_slow))


def rollout_free(m, d, idx, q_cmd, kp, kd, dt, q0):
    """공칭 롤아웃 — 재초기화 없이 끝까지. q·dq 를 함께 돌려준다(섭동 롤아웃의 기준상태)."""
    import mujoco
    N, n = q_cmd.shape
    q = np.empty((N, n)); dq = np.empty((N, n))
    m.opt.timestep = dt
    d.qpos[:] = 0; d.qvel[:] = 0
    for i, (_, qa, dofa, _) in enumerate(idx):
        d.qpos[qa] = q0[i] * DEG
    mujoco.mj_forward(m, d)
    for t in range(N):
        for i, (_, qa, dofa, aid) in enumerate(idx):
            q[t, i] = d.qpos[qa] / DEG
            dq[t, i] = d.qvel[dofa] / DEG
            d.ctrl[aid] = kp[i] * (q_cmd[t, i] - q[t, i]) * DEG - kd[i] * d.qvel[dofa]
        mujoco.mj_step(m, d)
    return q, dq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--config", default=os.path.join(EMB, "config", "biped_emb.yaml"))
    ap.add_argument("--mjcf", default=os.path.join(BIPED, "biped_from_quad.mjcf"))
    ap.add_argument("--T", type=float, default=None, help="길이[s] (기본 spec)")
    ap.add_argument("--window", type=float, default=0.5, help="재초기화 창[s] — pace_cmaes 와 맞출 것")
    ap.add_argument("--pert", type=float, default=0.10, help="섭동 크기(상대). 감도는 이 값 기준으로 보고")
    ap.add_argument("--noise", type=float, default=0.02, help="측정 잡음[°] — SNR 기준")
    ap.add_argument("--dt", type=float, default=None, help="적분 timestep[s] (기본 1/rate)")
    ap.add_argument("--f0scale", type=float, default=1.0,
                    help="f_start 배율. <1 이면 저속 구간이 길어진다(쿨롱↔점성 분리용)")
    ap.add_argument("--f1scale", type=float, default=1.0,
                    help="f_end 배율. PACE 논문은 10Hz 까지 쓴다(우리 1.55Hz)")
    ap.add_argument("--dual", default=None, metavar="비율,f_slow",
                    help="느린 대진폭 성분을 겹친다 (예: 0.35,0.12). JDAMP↔JFRIC 분리용")
    a = ap.parse_args()

    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    cfg_all = yaml.safe_load(open(a.config, encoding="utf-8"))
    jm = JointMap(cfg_all)
    mc = spec["pace_multi"]
    T = float(a.T or mc.get("duration_s", 30.0))
    rate = float(spec["shm"]["rate_hz"])
    dt = float(a.dt or 1.0 / rate)

    dual = None
    if a.dual:
        dual = tuple(float(x) for x in a.dual.split(","))
    if a.f0scale != 1.0 or a.f1scale != 1.0:
        mc = dict(mc)
        mc["f_start_hz"] = list(np.array(mc["f_start_hz"], float) * a.f0scale)
        mc["f_end_hz"] = list(np.array(mc["f_end_hz"], float) * a.f1scale)
    tt, q_cmd, home, des = build_traj(mc, jm, cfg_all, T, rate, dual)
    names = list(jm.names)

    # ★게인은 **관절공간**으로. collect_multichirp 이 npz 에 저장하는 것과 같은 환산이다
    #   (kp_joint = kp_ch·k²). 여기서 갈리면 감도가 통째로 틀린다.
    import actuator_test as at
    kp_ch = at._gain(mc["kp"]); kd_ch = at._gain(mc["kd"])
    kp = np.array([kp_ch[c] * jm.k[i] ** 2 for i, c in enumerate(jm.ch)])
    kd = np.array([kd_ch[c] * jm.k[i] ** 2 for i, c in enumerate(jm.ch)])
    gear_n = np.array([float([x for x in spec["joints"] if x["ch"] == c][0]["gear"])
                       for c in jm.ch])

    m = P.load_fixed_base(a.mjcf)
    import mujoco
    d = mujoco.MjData(m)
    idx = P.joint_index(m, names)
    x0, lo, hi = P.init_bounds(a.spec, names, False)
    plabels = ["ROTOR_I"] + [f"JDAMP.{k}" for k in P.KINDS] + [f"JFRIC.{k}" for k in P.KINDS]

    print("■ 가진 궤적 식별가능성 (하드웨어 미접촉)")
    print(f"  {os.path.basename(a.mjcf)} · {T:.0f}s · {rate:.0f}Hz · dt {dt*1e3:.1f}ms "
          f"· 창 {a.window:.2f}s · 섭동 ±{a.pert*100:.0f}% · 잡음 {a.noise:.3f}°")
    print(f"  진폭[°] {np.round(des['amp'],1)}")
    print(f"  f_start[Hz] {np.round(des['f0'],3)}   f_end[Hz] {np.round(des['f1'],2)}")
    if dual:
        print(f"  ★dual — 느린성분 비율 {dual[0]:.2f} · {dual[1]:.2f}Hz "
              f"(빠른성분 진폭은 {(1-dual[0])*100:.0f}% 로 줄여 예산을 나눈다)")
    vmax = np.abs(np.diff(q_cmd, axis=0)).max(axis=0) / (1.0 / rate)
    print(f"  최대 |q̇|[dps] {np.round(vmax,0)}  (속도상한 {spec['safety']['vel_trip_dps']:.0f})")
    lo_j = np.array(jm.jog_min[:len(names)]); hi_j = np.array(jm.jog_max[:len(names)])
    bad = [names[i] for i in range(len(names))
           if q_cmd[:, i].min() < lo_j[i] or q_cmd[:, i].max() > hi_j[i]]
    if bad:
        print(f"  ❌ jog 한계 밖: {bad} — 이 궤적은 실기에 걸 수 없다")

    win = max(1, int(round(a.window / dt)))
    P.apply_params(m, idx, gear_n, x0, False, names)
    q_nom, dq_nom = rollout_free(m, d, idx, q_cmd, kp, kd, dt, home)

    # ── 감도행렬 S: 열 하나가 파라미터 하나 ────────────────────────────────
    #   중앙차분. 창 재초기화를 켠 채로 재야 CMA-ES 가 보는 것과 같다.
    S = np.empty((q_nom.size, len(x0)))
    for p in range(len(x0)):
        col = []
        for sgn in (+1, -1):
            xp = x0.copy(); xp[p] = x0[p] * (1.0 + sgn * a.pert)
            P.apply_params(m, idx, gear_n, xp, False, names)
            col.append(P.rollout(m, d, idx, q_nom, dq_nom, q_cmd, kp, kd, dt, win))
        S[:, p] = ((col[0] - col[1]) * 0.5).ravel()        # +pert 1단위당 궤적변화[°]
    P.apply_params(m, idx, gear_n, x0, False, names)       # 원복

    rms = np.sqrt(np.mean(S ** 2, axis=0))
    snr = rms / a.noise

    # ★판정은 **표본당 SNR 이 아니라 추정 분해능**으로 한다.
    #   표본당 SNR 이 1 밑이어도 표본이 6만 개면 √N 로 평균되어 충분히 식별된다.
    #   반대로 SNR 이 커도 다른 파라미터와 평행하면(r≈1) 분리가 안 된다.
    #   둘을 한꺼번에 보는 게 공분산이다:  Σ = σ²(SᵀS)⁻¹  (Cramér–Rao, 백색잡음 가정)
    #   S 의 열이 "pert 비율 1단위당" 이므로 √Σ_pp · pert 가 곧 **상대 분해능**이다.
    #   ⚠백색·독립 잡음 가정이라 **낙관값**이다. 실제 잡음은 상관이 있고 모델오차도 있다.
    #     따라서 "여유 10배" 정도를 보고 판단할 것이지 1.5배를 믿지 말 것.
    A = S.T @ S
    try:
        cov = (a.noise ** 2) * np.linalg.inv(A)
        res = np.sqrt(np.clip(np.diag(cov), 0, None)) * a.pert       # 상대 분해능(1=100%)
    except np.linalg.LinAlgError:
        res = np.full(len(x0), np.inf)
    # 다른 파라미터를 **알고 있다고 가정**했을 때의 분해능 — 혼동으로 잃은 양을 드러낸다
    res_alone = a.noise * a.pert / np.sqrt(np.clip(np.diag(A), 1e-300, None))
    loss = res / np.where(res_alone > 0, res_alone, 1)

    print(f"\n  ── (1) 감도와 분해능 (섭동 +{a.pert*100:.0f}% 기준, 표본 {S.shape[0]:,}) ──")
    print(f"  {'파라미터':<14}{'초기값':>11}{'RMS[°]':>9}{'표본SNR':>8}"
          f"{'분해능':>9}{'단독대비':>8}   판정")
    for p in np.argsort(res):
        v = ("식별가능" if res[p] < 0.05 else
             "★경계(반복수집)" if res[p] < 0.15 else "❌식별 불가")
        print(f"  {plabels[p]:<14}{x0[p]:>11.4g}{rms[p]:>9.4f}{snr[p]:>8.1f}"
              f"{res[p]*100:>8.1f}%{loss[p]:>7.1f}x   {v}")
    print(f"  분해능 = 그 파라미터를 몇 % 안에서 구분할 수 있나(작을수록 좋다).")
    print(f"  단독대비 = 다른 파라미터를 모두 안다고 가정했을 때 대비 **몇 배 나빠졌나**"
          f" — 혼동으로 잃은 양이다.")

    # ── (2) 파라미터 간 혼동 ───────────────────────────────────────────────
    Sn = S / np.where(rms > 0, rms, 1.0)
    C = (Sn.T @ Sn) / S.shape[0]
    C /= np.sqrt(np.outer(np.diag(C), np.diag(C)))
    off = np.abs(C - np.eye(len(x0)))
    print(f"\n  ── (2) 파라미터 간 상관 (|r|>0.9 면 그 둘은 맞바꿔진다) ──")
    pairs = [(off[i, j], i, j) for i in range(len(x0)) for j in range(i + 1, len(x0))]
    for r, i, j in sorted(pairs, reverse=True)[:5]:
        mark = "  ★혼동" if r > 0.9 else ("  주의" if r > 0.7 else "")
        print(f"    {plabels[i]:<14}↔ {plabels[j]:<14}  r={r:+.3f}{mark}")
    sv = np.linalg.svd(Sn, compute_uv=False)
    print(f"\n  감도행렬 조건수 {sv[0]/max(sv[-1],1e-30):.1f}  "
          f"(정규화 후. 크면 어떤 파라미터 조합이 궤적에 거의 안 나타난다)")

    # 참고: 축 상관 — 종전 지표. 남겨서 대조할 수 있게 한다.
    Cq = np.corrcoef(q_cmd.T)
    print(f"  [참고] 종전 지표 '축간 상관' 최대 {np.abs(Cq-np.eye(len(names))).max():.3f}")

    weak = [plabels[p] for p in range(len(x0)) if res[p] >= 0.15]
    conf = [f"{plabels[i]}↔{plabels[j]}" for r, i, j in pairs if r > 0.9]
    print()
    if weak:
        print(f"  ❌ 식별 불가(분해능 ≥15%): {', '.join(weak)}")
        print(f"     → 진폭·f_end 를 올리거나(토크예산 확인), 그 파라미터를 축별 시험으로 고정할 것")
    if conf:
        print(f"  ★혼동(|r|>0.9): {', '.join(conf)}")
        print(f"     → 위상·주파수 배치를 바꾸거나, 한쪽을 축별 측정으로 고정할 것")
    if not weak and not conf:
        print("  ✅ 전 파라미터 SNR≥2 · 혼동쌍 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
