#!/usr/bin/env python3
"""act_identify_pace.py — PACE 액추에이터 파라미터 식별 (ROTOR_I / JDAMP / JFRIC).

프로젝트에서의 PACE 정의(docs/sim2real_checklist_17dof.html, 2026-07-21 채택):
  "PACE(ETH RSL, CMA-ES 시스템식별) — 실물 관절궤적(명령+측정 dof) 수집 →
   sim 이 실측을 재현하도록 armature/damping/friction 을 진화탐색(Σ(sim−real−bias)² 최소화)"
  적용 경로: 식별값 → env ROTOR_I / JDAMP / JFRIC → apply_gearbox() 가
             MJCF dof_armature(=I_rot·N²) / dof_damping / dof_frictionloss 에 기입.

★역할 분담 (로봇에 mujoco 미설치이므로):
   - **로봇(여기)**: 처프 가진 → (q, q̇, τ, q_cmd) 수집 → *해석적 회귀*로 1차 식별 +
     데이터셋 `pace_dataset_ch##.npz` 내보내기.
   - **노트북**: 그 npz 로 MuJoCo CMA-ES 재현매칭(원래의 PACE 절차) 수행.
     목적함수는 아래 `CMAES_OBJECTIVE` 주석에 그대로 적어두었다.
   1차 회귀값은 CMA-ES 의 **초기값·탐색범위**로 쓰면 수렴이 훨씬 빨라진다.

식별 모델 (관절축 1자유도 기준):
      τ = I_total·q̈ + b·q̇ + τ_c·tanh(q̇/ε) + A·sin(q) + c
  - I_total : 관절축 등가 관성 = I_link(링크) + ROTOR_I·N² (반사관성)
              → **ROTOR_I = (I_total − I_link) / N²**   (spec.I_link 필요)
  - b       : 점성감쇠            → JDAMP
  - τ_c     : 쿨롱마찰            → JFRIC
  - A       : 중력항. ★cos(q) 는 넣지 않는다 — 가진 진폭이 작으면 상수열과 공선이라
              조건수가 16348 까지 뛰고 중력·바이어스가 ±13 Nm 로 서로 상쇄되는
              무의미한 값이 된다(제거 시 365). 단 A·c 는 편향되므로 절대 중력이
              아니라 '여기범위 내 중력 변화폭' 으로 해석할 것.
  - c       : 토크센서 바이어스
  sign() 대신 tanh(q̇/ε) 를 써서 미분가능·저속 수치안정성을 확보한다.

⚠ 식별 가능성(identifiability): 가속도 성분이 충분히 여기되지 않으면 I_total 이
  b·τ_c 와 강하게 상관된다. 아래에서 **조건수와 파라미터별 표준편차**를 함께 보고하니
  반드시 확인할 것. std 가 값과 같은 자릿수면 그 파라미터는 "식별 안 됨" 이다.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from hwio import DEG, samples_to_arrays

# 노트북에서 수행할 원래 PACE 절차(참고용 — 여기서는 실행하지 않는다)
CMAES_OBJECTIVE = """
# 노트북(mujoco+cma 설치 환경)에서:
#   d = np.load('pace_dataset_ch00.npz')
#   def cost(p):                      # p = [ROTOR_I, JDAMP, JFRIC]
#       m.dof_armature[dof]     = p[0] * N * N
#       m.dof_damping[dof]      = p[1]
#       m.dof_frictionloss[dof] = p[2]
#       q_sim = rollout(m, d['q_cmd'], d['kp'], d['kd'], d['dt'], q0=d['q'][0])
#       return np.sum((q_sim - d['q'] - bias)**2)      # PACE: Σ(sim−real−bias)²
#   cma.fmin(cost, x0=[roti0, jdamp0, jfric0], sigma0=...)
# ★x0 는 이 스크립트가 출력하는 회귀 추정값을 쓸 것.
"""

TEMPLATE = Template("""
<h2>{{ title }}</h2>
<p>처프(chirp) 위치가진으로 넓은 주파수 대역을 여기시키고, 관절 운동방정식
<code>τ = I·q̈ + b·q̇ + τ_c·tanh(q̇/ε) + A·sin q + c</code> 를 회귀해
<b>ROTOR_I · JDAMP · JFRIC</b> 를 식별한다.</p>

<table>
  <tr><th colspan="2">시험 조건</th></tr>
  <tr><td>일시</td><td>{{ datetime }}</td></tr>
  <tr><td>축</td><td>{{ joint }} (SHM ch{{ ch }}, 감속비 N={{ gear }})</td></tr>
  <tr><td>처프</td><td class="numeric">±{{ '%.1f' % amp }} deg · {{ f0 }}→{{ f1 }} Hz · {{ dur }} s</td></tr>
  <tr><td>샘플</td><td class="numeric">{{ nsamp }} @ {{ rate }} Hz</td></tr>

  <tr><th colspan="2">식별 결과 <span class="dim">(±는 회귀 표준편차)</span></th></tr>
  <tr><td>I_total (관절축 등가관성)</td><td class="numeric">{{ '%0.6f' % I_total }} ± {{ '%0.6f' % I_sd }} kg·m²</td></tr>
  <tr><td><b>JDAMP</b> (점성감쇠 b)</td><td class="numeric">{{ '%0.4f' % jdamp }} ± {{ '%0.4f' % jdamp_sd }} N·m·s/rad</td></tr>
  <tr><td><b>JFRIC</b> (쿨롱마찰 τ_c)</td><td class="numeric">{{ '%0.4f' % jfric }} ± {{ '%0.4f' % jfric_sd }} N·m</td></tr>
  <tr><td>중력항 A <span class="dim">(여기범위 내 변화폭)</span></td><td class="numeric">{{ '%0.4f' % grav_mag }} N·m</td></tr>
  <tr><td>토크 바이어스 c</td><td class="numeric">{{ '%0.4f' % bias }} N·m</td></tr>
  <tr><td><b>ROTOR_I</b> = (I_total − I_link)/N²</td><td class="numeric">{{ rotor_i_str }}</td></tr>

  <tr><th colspan="2">품질 지표</th></tr>
  <tr><td>적합도 R²</td><td class="numeric">{{ '%0.4f' % r2 }}</td></tr>
  <tr><td>잔차 RMS</td><td class="numeric">{{ '%0.4f' % rms }} N·m</td></tr>
  <tr><td>회귀행렬 조건수</td><td class="numeric">{{ '%0.1f' % cond }}
      <span class="dim">(수백을 넘으면 파라미터 분리 신뢰도 저하)</span></td></tr>
  <tr><td>가속도 여기 범위</td><td class="numeric">|q̈| ≤ {{ '%0.2f' % acc_max }} rad/s²</td></tr>
</table>

{{ warnings }}

<h3>적용 (그대로 붙여넣기)</h3>
<pre>ROTOR_I={{ rotor_i_env }} JDAMP={{ '%0.4f' % jdamp }} JFRIC={{ '%0.4f' % jfric }}</pre>
<p class="dim">→ apply_gearbox() 가 dof_armature = ROTOR_I·N² / dof_damping / dof_frictionloss 에 기입.
데이터셋 <code>{{ npz }}</code> 을 노트북으로 옮기면 MuJoCo CMA-ES 재현매칭(원래 PACE)을 이 값을 x0 로 실행할 수 있다.</p>

<p><img src="{{ plot_fit }}"></p>
<p><img src="{{ plot_exc }}"></p>
""")


def _chirp_cmd(t, amp, f0, f1, dur):
    """선형 처프. 위상 = 2π(f0·t + (f1−f0)/(2T)·t²)."""
    k = (f1 - f0) / max(dur, 1e-9)
    return amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def identify_pace(hw, spec, joint, plotdir, outdir, log=print) -> tuple[str, dict]:
    ch = int(joint["ch"])
    name, gear = joint["name"], float(joint["gear"])
    kp, kd = spec["gains"]["kp"], spec["gains"]["kd"]
    cp, flt = spec["pace"]["chirp"], spec["pace"]["filter"]
    rate = spec["shm"]["rate_hz"]
    warn: list[str] = []

    amp = min(float(cp["amplitude_deg"]),
              (joint["q_max"] - joint["q_min"]) / 2 - 1.0)
    f0, f1, dur = float(cp["f_start_hz"]), float(cp["f_end_hz"]), float(cp["duration_s"])
    log(f"  [{name}] PACE 처프 ±{amp:.1f}deg {f0}→{f1}Hz {dur:.0f}s")

    # ── 가진 ────────────────────────────────────────────────────────────────
    q0 = hw.arm(ch, kp, kd)
    center = float(np.clip(q0, joint["q_min"] + amp + 1, joint["q_max"] - amp - 1))
    hw.goto(ch, center, kp, kd)
    time.sleep(0.3)
    s = hw.run(ch, lambda t: center + _chirp_cmd(t, amp, f0, f1, dur),
               dur, kp, kd, progress="chirp")
    hw.limp()
    A = samples_to_arrays(s)
    n = len(A["t"])
    if n < 200:
        raise RuntimeError(f"샘플 부족({n}) — 처프 시간을 늘릴 것")

    # ── 미분·평활 (실측 속도노이즈 ±15 deg/s) ───────────────────────────────
    dt = float(np.median(np.diff(A["t"])))
    win = int(flt["savgol_window"]) | 1                      # 홀수 보정
    win = min(win, (n // 2) * 2 - 1)
    poly = int(flt["savgol_poly"])
    q_rad = A["q"] * DEG
    dq_raw = A["dq"] * DEG
    dq = savgol_filter(dq_raw, win, poly)
    # 가속도는 평활된 속도의 미분(savgol 의 해석적 미분 사용 — 노이즈 증폭 억제)
    ddq = savgol_filter(dq_raw, win, poly, deriv=1, delta=dt)
    tau = A["tau"] * float(spec["units"].get("torque_scale", 1.0))

    # 과도구간(초기 0.5s) 제외
    m = A["t"] > 0.5
    q_rad, dq, ddq, tau = q_rad[m], dq[m], ddq[m], tau[m]

    # ── 회귀 ────────────────────────────────────────────────────────────────
    # ★cos(q) 를 넣지 않는다. 가진 진폭이 작으면(±6° → cos q ∈ [0.9915, 1]) cos(q) 열이
    #   상수 열과 사실상 동일해져 공선이 된다. 실제로 넣었을 때 조건수 16348,
    #   중력항 +13.4 Nm / 바이어스 −13.4 Nm 라는 서로 상쇄되는 무의미한 값이 나왔다.
    #   제거하면 조건수 365, 바이어스 +0.025 Nm 로 물리적으로 타당해지고
    #   I_total·JDAMP·JFRIC 추정값은 거의 변하지 않는다(=제거해도 정보 손실 없음).
    #   sin(q) 는 ±0.105 로 상수와 구분되므로 유지한다(중력 성분).
    eps = float(spec["pace"].get("tanh_eps_rad_s", 0.02))    # tanh 폭[rad/s] — ★고정

    def regressor(dqv):
        return np.column_stack([ddq, dqv, np.tanh(dqv / eps),
                                np.sin(q_rad), np.ones_like(dqv)])

    W = regressor(dq)
    theta, *_ = np.linalg.lstsq(W, tau, rcond=None)

    # 비선형 정련 — ★eps 는 자유변수로 두지 않는다.
    #   eps 를 풀어주면 tanh 가 저속에서 완만해지며 점성항을 흉내내 b 와 강하게 상관된다
    #   (실측: eps 자유 → JDAMP 0.107→0.049 로 2배 이상 흔들림). 물리적으로 eps 는
    #   "쿨롱마찰이 부호를 바꾸는 속도폭" 이라 작은 상수여야 하므로 고정이 옳다.
    def resid(p):
        I, b, tc, Aa, c = p
        return (I * ddq + b * dq + tc * np.tanh(dq / eps)
                + Aa * np.sin(q_rad) + c - tau)
    try:
        sol = least_squares(resid, theta, method="trf", max_nfev=6000)
        if sol.success:
            theta = sol.x
    except Exception as e:
        warn.append(f"비선형 정련 실패({type(e).__name__}) — 선형 회귀값 사용")

    I_total, b, tau_c, Aa, c = (float(x) for x in theta)
    Bb = 0.0                                                 # cos 항 미사용
    W = regressor(dq)
    pred = W @ theta
    # ── ★전류 교차검증 (2026-08-10) ─────────────────────────────────────
    #   체크리스트 원칙: 보고토크가 kp(q*−q)+kd(q̇*−q̇) 로 재구성되면 회귀가 순환이 된다.
    #   전류는 그 경로와 **독립**이므로 τ_rep 와 Kt·I 를 비교하면 판별된다.
    #   (기존 hip 데이터 검증: MIT 재구성 R² 0.53 → 보고토크는 측정 기반이었다.
    #    그래도 축마다 다를 수 있으니 매 측정에서 확인한다.)
    kt = float(joint.get("kt_nm_per_a", 0.0) or 0.0)
    cur = A.get("cur")
    if kt > 0 and cur is not None and len(cur) == len(tau) + 0:
        cur_m = cur[m] if len(cur) != len(tau) else cur
        tau_i = kt * cur_m
        if np.std(tau_i) > 1e-9:
            r_ti = float(np.corrcoef(tau, tau_i)[0, 1])
            rms  = float(np.sqrt(np.mean((tau - tau_i) ** 2)))
            log(f"  [{name}] 전류 교차검증: corr(τ_rep, Kt·I)={r_ti:.4f} · RMS차 {rms:.4f} Nm")
            if r_ti < 0.9:
                warn.append(f"τ_rep 와 Kt·I 상관 {r_ti:.3f} — 둘 중 하나가 틀렸다. 전류 쪽을 신뢰할 것")
        else:
            warn.append("전류가 거의 상수 — 전류센싱이 죽었을 수 있다(교차검증 불가)")
    else:
        warn.append("전류 또는 kt_nm_per_a 없음 — τ 순환 여부를 독립 검증하지 못했다")

    res = tau - pred
    ss_tot = float(np.sum((tau - np.mean(tau)) ** 2))
    r2 = 1.0 - float(np.sum(res ** 2)) / ss_tot if ss_tot > 1e-12 else float("nan")
    rms = float(np.sqrt(np.mean(res ** 2)))
    cond = float(np.linalg.cond(W))

    # 파라미터 표준편차 (식별 가능성 판정용)
    dof = max(len(tau) - W.shape[1], 1)
    sigma2 = float(np.sum(res ** 2)) / dof
    try:
        cov = sigma2 * np.linalg.inv(W.T @ W)
        sd = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        sd = np.full(5, np.nan)
        warn.append("공분산 계산 실패 — 회귀행렬 특이(파라미터 분리 불가)")

    if abs(sd[0]) > 0.5 * abs(I_total):
        warn.append(f"I_total 표준편차가 값의 50%% 초과(±{sd[0]:.6f}) — <b>관성이 식별되지 않았다</b>. "
                    f"처프 상한주파수를 올리거나 진폭을 키워 가속도 여기를 늘릴 것")
    if cond > 500:
        warn.append(f"조건수 {cond:.0f} 과대 — 파라미터 간 상관이 크다. 가진 다양성 부족")
    if r2 < 0.8:
        warn.append(f"R²={r2:.3f} 낮음 — 모델 미포착 성분(백래시·시간지연·온도) 가능성")
    if tau_c < 0:
        warn.append(f"쿨롱마찰 음수({tau_c:.4f}) — 부호규약/센서 확인 필요")

    # ── ROTOR_I 분리 ────────────────────────────────────────────────────────
    I_link = joint.get("I_link")
    if isinstance(I_link, (int, float)):
        rotor_i = (I_total - float(I_link)) / (gear ** 2)
        rotor_i_str = f"{rotor_i:.3e} kg·m² (I_link={I_link})"
        rotor_i_env = f"{rotor_i:.3e}"
        if rotor_i <= 0:
            warn.append(f"ROTOR_I 이 음수/0 ({rotor_i:.3e}) — I_link 가 과대하거나 "
                        f"I_total 미식별. MJCF 링크관성 확인 필요")
    else:
        rotor_i = None
        rotor_i_str = (f"미확정 — spec.yaml 의 <code>I_link</code> 가 TODO. "
                       f"I_link 를 넣으면 (I_total−I_link)/N² = ({I_total:.6f}−I_link)/{gear**2:.1f}")
        rotor_i_env = f"<I_link 필요>"
        warn.append("MJCF 링크관성 <code>I_link</code> 미제공 → ROTOR_I 를 분리할 수 없다. "
                    "반사관성만 필요하면 I_total 을 그대로 armature 로 써도 되지만, "
                    "그 경우 링크관성이 이중계상되므로 권장하지 않는다.")

    # ── 데이터셋 내보내기 (노트북 CMA-ES 용) ────────────────────────────────
    npz = f"{outdir}/pace_dataset_ch{ch:02d}.npz"
    np.savez(npz, t=A["t"], q=A["q"], dq=A["dq"], tau=A["tau"], q_cmd=A["q_cmd"],
             cur=A["cur"],   # ★전류 저장(2026-08-10) — τ = Kt·I 교차검증용.
                             #   hwio 는 읽고 있었는데 저장에서 빠져 있었다.
                             #   보고토크가 명령 재구성인지 측정인지 가르는 유일한 독립 근거다.
             kp=kp, kd=kd, dt=dt, gear=gear, ch=ch, name=name,
             units="q,q_cmd:deg dq:deg/s tau:reported",
             theta=np.array([I_total, b, tau_c, Aa, Bb, c]), eps=eps,
             note=CMAES_OBJECTIVE)
    log(f"  [{name}] → I_total={I_total:.6f} JDAMP={b:.4f} JFRIC={tau_c:.4f} "
        f"R²={r2:.4f} cond={cond:.0f}")
    log(f"  [{name}] 데이터셋 저장: {npz}")

    # ── 플롯 ────────────────────────────────────────────────────────────────
    p_fit = f"{plotdir}/pace_fit_ch{ch:02d}.png"
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    tt = A["t"][m]
    ax[0].plot(tt, tau, lw=.7, label="measured")
    ax[0].plot(tt, pred, lw=.7, label="model")
    ax[0].set_ylabel("torque (Nm)"), ax[0].legend(), ax[0].grid(alpha=.3)
    ax[0].set_title(f"PACE 회귀 적합 — {name}  (R²={r2:.4f}, RMS={rms:.4f} Nm)")
    ax[1].plot(tt, res, lw=.7, c="C3")
    ax[1].set_xlabel("time (s)"), ax[1].set_ylabel("residual (Nm)"), ax[1].grid(alpha=.3)
    plt.savefig(p_fit, dpi=110, bbox_inches="tight"), plt.close()

    p_exc = f"{plotdir}/pace_excitation_ch{ch:02d}.png"
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(A["t"], A["q_cmd"], lw=.7, label="command"), ax[0].plot(A["t"], A["q"], lw=.7, label="measured")
    ax[0].set_ylabel("position (deg)"), ax[0].legend(), ax[0].grid(alpha=.3)
    ax[1].plot(tt, dq, lw=.7), ax[1].set_ylabel("velocity (rad/s)"), ax[1].grid(alpha=.3)
    ax[2].plot(tt, ddq, lw=.7), ax[2].set_ylabel("accel (rad/s²)")
    ax[2].set_xlabel("time (s)"), ax[2].grid(alpha=.3)
    fig.suptitle(f"Chirp excitation — {name}")
    plt.savefig(p_exc, dpi=110, bbox_inches="tight"), plt.close()

    warnings_html = ""
    if warn:
        warnings_html = ('<div class="warn"><b>주의</b><ul>'
                         + "".join(f"<li>{w}</li>" for w in warn) + "</ul></div>")

    html = TEMPLATE.render(
        title=f"PACE Actuator Identification — {name}",
        datetime=time.strftime("%Y-%m-%d %H:%M:%S"),
        joint=name, ch=ch, gear=gear, amp=amp, f0=f0, f1=f1, dur=dur,
        nsamp=n, rate=rate,
        I_total=I_total, I_sd=sd[0], jdamp=b, jdamp_sd=sd[1],
        jfric=tau_c, jfric_sd=sd[2],
        grav_mag=float(np.hypot(Aa, Bb)), bias=c,
        rotor_i_str=rotor_i_str, rotor_i_env=rotor_i_env,
        r2=r2, rms=rms, cond=cond, acc_max=float(np.max(np.abs(ddq))),
        warnings=warnings_html, npz=npz.split("/")[-1],
        plot_fit=p_fit.replace(plotdir, "plots"), plot_exc=p_exc.replace(plotdir, "plots"))

    return html, {"I_total": I_total, "jdamp": b, "jfric": tau_c, "rotor_i": rotor_i,
                  "r2": r2, "cond": cond, "ch": ch, "name": name, "npz": npz}
