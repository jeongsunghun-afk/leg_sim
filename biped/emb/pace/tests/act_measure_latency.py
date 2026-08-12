#!/usr/bin/env python3
"""act_measure_latency.py — 왕복지연(latency) · 로스트모션(백래시) 실측.

체크리스트 C절이 요구하는 `SENSE` / `ACT_LAT_MS` 와, A절의 backlash 판단근거를 채운다.

── 무엇을 재는가 ────────────────────────────────────────────────────────────
(1) **왕복지연 T_rt** — 스텝응답으로.
    명령을 1회 쓴 시각부터 "토크가 움직이기 시작한 시각" 까지.
    이 값에는 다음이 **전부 포함**된다:
        쓰기 → SHM → Emb 폴링(1kHz) → EtherCAT 사이클 → MCU → FDCAN → 드라이버 인가
        → 센서 → 역경로 → 우리 read
    즉 제어루프가 실제로 겪는 왕복지연이며, ACT 와 SENSE 를 분리한 값이 아니다.
    (단일 지점에서 관측하는 한 원리적으로 분리 불가 — 분리하려면 외부 계측이 필요하다.)
    토크는 위치보다 훨씬 먼저 반응하므로(관성 때문에 위치는 적분되어 늦다)
    **토크 상승 시점**을 쓰는 것이 지연 추정에 정확하다. 위치 도달시각도 참고로 같이 낸다.

(2) **군지연 T_gd** — 처프의 q_cmd ↔ q 상호상관 최대점.
    이건 순수 전송지연이 아니라 **폐루프 응답 전체의 지연**(전송지연 + 임피던스 제어기
    자체의 동특성)이다. 항상 T_rt 보다 크게 나오는 것이 정상이며, 둘의 차이가
    "제어기 동특성 때문에 늦는 정도" 다.

(3) **로스트모션(lost motion)** — 저속 왕복에서 속도가 반전될 때,
    명령각이 얼마나 더 가야 관절이 다시 움직이기 시작하는가.
    ⚠ 이 값은 **백래시 + 컴플라이언스 + 스틱션 dwell 의 합**이다.
    엔코더가 하나뿐이면(모터축이든 관절축이든 한쪽만 보면) 셋을 분리할 수 없다.
    분리하려면 관절축·모터축 양쪽 엔코더가 동시에 필요하다. 여기서는 합계를 보고하고,
    참고로 반전구간의 토크 변화폭(≈2·τ_c)을 함께 내 스틱션 기여분을 가늠하게 한다.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
from scipy.signal import savgol_filter

from hwio import DEG, samples_to_arrays

TEMPLATE = Template("""
<h2>{{ title }}</h2>
<p>제어 왕복지연과 기계적 로스트모션을 측정한다. 지연은 명령 1회 기록 후
<b>sleep 없는 타이트 폴링</b>으로 관측해 루프주기(5 ms)보다 고운 분해능을 얻는다.</p>

<table>
  <tr><th colspan="2">시험 조건</th></tr>
  <tr><td>일시</td><td>{{ datetime }}</td></tr>
  <tr><td>축</td><td>{{ joint }} (SHM ch{{ ch }})</td></tr>
  <tr><td>스텝</td><td class="numeric">{{ '%.1f' % step_deg }} deg × {{ trials }}회 (양방향)</td></tr>
  <tr><td>폴링 실효주기</td><td class="numeric">{{ '%.2f' % poll_ms }} ms</td></tr>

  <tr><th colspan="2">지연</th></tr>
  <tr><td><b>왕복지연 T_rt</b> (토크 반응)</td><td class="numeric">{{ '%0.2f' % t_rt }} ms
      <span class="dim">(중앙값, 산포 {{ '%0.2f' % t_rt_sd }} ms)</span></td></tr>
  <tr><td>위치 반응 시점</td><td class="numeric">{{ '%0.2f' % t_pos }} ms
      <span class="dim">(관성 적분 때문에 토크보다 늦다 — 지연값으로 쓰지 말 것)</span></td></tr>
  <tr><td>군지연 T_gd (처프 상호상관)</td><td class="numeric">{{ gd_str }}</td></tr>
  <tr><td>제어기 동특성 기여분</td><td class="numeric">{{ ctrl_str }}</td></tr>

  <tr><th colspan="2">로스트모션</th></tr>
  <tr><td><b>lost motion</b> (반전 dead-zone)</td><td class="numeric">{{ lm_str }}</td></tr>
  <tr><td>반전구간 토크 변화폭</td><td class="numeric">{{ '%0.4f' % tau_swing }} Nm
      <span class="dim">(≈2·τ_c — 이만큼이 스틱션 기여)</span></td></tr>
  <tr><td>반전 검출 횟수</td><td class="numeric">{{ n_rev }}</td></tr>
</table>

<div class="warn"><b>해석 주의</b>
<ul>
<li><b>T_rt 는 ACT 와 SENSE 의 합</b>이다. 단일 관측점으로는 분리할 수 없다 —
    체크리스트의 <code>ACT_LAT_MS</code>/<code>SENSE</code> 에 나눠 넣으려면
    합을 어떻게 배분할지 별도 근거(예: EtherCAT 사이클 구조)가 필요하다.
    보수적으로는 전량을 <code>ACT_LAT_MS</code> 로 두는 편이 안전측이다.</li>
<li><b>lost motion 은 백래시 단독이 아니다</b> — 백래시+컴플라이언스+스틱션의 합이다.
    엔코더가 한쪽에만 있으면 분리 불가.</li>
</ul></div>

{{ warnings }}

<p><img src="{{ plot_step }}"></p>
<p><img src="{{ plot_rev }}"></p>
""")


def backlash_from_dual_encoder(q_in_deg, q_out_deg, gear) -> dict:
    """★두 엔코더가 확보되면 이 함수 하나로 백래시가 확정된다 (현재는 미탑재 대기).

    백래시 = 입력축을 감속비로 환산한 각도와 출력축 각도의 차이가 방향전환에서 그리는
    히스테리시스 폭이다.  lash(q) = q_in/N - q_out.
    한 방향으로 밀 때와 반대 방향으로 밀 때의 lash 평탄값 차이가 총 유격이다.

    ⚠ 2026-08-05 현재 SHM PDO(MotorParam16_t, 20B)에는 위치 필드가 **하나뿐**이라
      이 함수를 쓸 수 없다. 실측 확인:
        fAccelrationOrTemperture = +1.000 고정 상수(미사용)
        fCurrent = fTorque 완전 중복 / fGainKp,Kd,Ki = 0
      → MCU 펌웨어에 "비어 있는 fAccelrationOrTemperture 에 출력축 엔코더 각도를
        fPosition 과 동일한 부호·영점 규약으로 실어달라" 고 요청해 둔 상태.
        PDO 크기·Emb 코드·SHM ABI 모두 무변경이라 가장 저렴한 경로다.
        도착하면 diag/shm_dump 가 "값이 변한다" 로 알려준다.
    """
    q_in = np.asarray(q_in_deg, float)
    q_out = np.asarray(q_out_deg, float)
    lash = q_in / float(gear) - q_out
    dq = np.gradient(q_out)
    fwd, rev = lash[dq > 0], lash[dq < 0]
    if fwd.size < 5 or rev.size < 5:
        return {"backlash_deg": None, "reason": "방향별 샘플 부족"}
    return {"backlash_deg": float(np.median(fwd) - np.median(rev)),
            "lash_fwd": float(np.median(fwd)), "lash_rev": float(np.median(rev)),
            "spread": float(np.std(fwd) + np.std(rev))}


def held_output_test_is_valid(tau, tau_coulomb, kp, cmd_span_deg,
                              engage_margin=2.0) -> tuple[bool, str]:
    """"출력축을 고정하고 모터를 돌리는" 백래시 시험이 유효했는지 판정.

    ★이 검사가 없으면 무효 측정을 유효한 것으로 오독한다(2026-08-05 실제로 그랬다).
      출력축을 손으로 잡았지만 실제로는 함께 돌아갔고, 그때 관측된 것은
      "q 가 명령을 65% 로 추종" 이었는데 이는 그냥 마찰에 의한 추종지연
      (오차 = tau_c/kp) 일 뿐 유격이 아니었다.

    유효 조건: 기어가 실제로 물려 출력이 버텼다면 토크가 마찰 수준을 크게 넘어
    kp·(명령앞섬) 까지 올라가야 한다. 그렇지 않으면 그냥 자유회전한 것이다.
    """
    tau_max = float(np.max(np.abs(tau)))
    need = engage_margin * float(tau_coulomb)
    if tau_max < need:
        return False, (f"최대토크 {tau_max:.3f} Nm < 마찰 {tau_coulomb:.3f}×{engage_margin:.0f}"
                       f"={need:.3f} Nm → 출력이 버티지 않았다(자유회전). 클램프로 고정할 것")
    reachable = kp * cmd_span_deg * np.pi / 180.0
    if tau_max < 0.5 * reachable:
        return False, (f"최대토크 {tau_max:.3f} Nm 가 도달가능 {reachable:.3f} Nm 의 절반 미만 "
                       f"→ 고정이 불완전했을 가능성")
    return True, f"유효(최대토크 {tau_max:.3f} Nm)"


def _onset(t: np.ndarray, y: np.ndarray, base: float, base_sd: float,
           k_sigma: float = 6.0, min_abs: float = 0.02) -> float | None:
    """base 대비 유의하게 벗어나기 시작한 첫 시각. 노이즈의 k_sigma 배를 임계로 쓴다.
    연속 3샘플이 임계를 넘어야 인정(단발 스파이크 배제)."""
    thr = max(k_sigma * base_sd, min_abs)
    d = np.abs(y - base)
    over = d > thr
    for i in range(len(over) - 2):
        if over[i] and over[i + 1] and over[i + 2]:
            return float(t[i])
    return None


def _latency(hw, spec, joint, kp, kd, log) -> tuple[dict, list]:
    ch = int(joint["ch"])
    cfg = spec.get("latency", {})
    step_deg = float(cfg.get("step_deg", 1.5))
    trials = int(cfg.get("trials", 4))
    window = float(cfg.get("window_s", 0.30))

    t_rt, t_pos, traces, polls = [], [], [], []
    for i in range(trials):
        d = step_deg if i % 2 == 0 else -step_deg
        hw.arm(ch, kp, kd)
        r = hw.step_response(ch, d, kp, kd, window_s=window)
        hw.limp()
        time.sleep(0.25)
        if len(r["t"]) < 20:
            log(f"    ⚠ trial{i}: 샘플 부족({len(r['t'])})")
            continue
        polls.append(float(np.median(np.diff(r["t"]))) * 1e3)

        on_tau = _onset(r["t"], r["tau"], r["tau_base"], r["tau_base_sd"])
        q_base = float(np.mean(r["q"][:5]))
        on_q = _onset(r["t"], r["q"], q_base, max(float(np.std(r["q"][:5])), 1e-3),
                      k_sigma=6.0, min_abs=0.03)
        if on_tau is not None:
            t_rt.append(on_tau * 1e3)
            traces.append(r)
            log(f"    trial{i} ({'+' if d > 0 else '−'}{abs(d):.1f}deg): "
                f"토크반응 {on_tau*1e3:6.2f} ms" +
                (f" · 위치반응 {on_q*1e3:6.2f} ms" if on_q else " · 위치 미검출"))
        else:
            log(f"    ⚠ trial{i}: 토크 반응 미검출(스텝이 작거나 노이즈 과다)")
        if on_q is not None:
            t_pos.append(on_q * 1e3)

    return ({"t_rt": t_rt, "t_pos": t_pos,
             "poll_ms": float(np.median(polls)) if polls else float("nan")}, traces)


def _group_delay(hw, spec, joint, kp, kd, log) -> float | None:
    """짧은 처프로 q_cmd↔q 상호상관 지연(군지연) 추정."""
    ch = int(joint["ch"])
    cfg = spec.get("latency", {})
    amp = min(float(cfg.get("gd_amp_deg", 4.0)),
              (joint["q_max"] - joint["q_min"]) / 2 - 1.0)
    f0, f1 = float(cfg.get("gd_f0", 0.3)), float(cfg.get("gd_f1", 3.0))
    dur = float(cfg.get("gd_duration_s", 12.0))

    q0 = hw.arm(ch, kp, kd)
    center = float(np.clip(q0, joint["q_min"] + amp + 1, joint["q_max"] - amp - 1))
    hw.goto(ch, center, kp, kd)
    k = (f1 - f0) / dur
    s = hw.run(ch, lambda t: center + amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t)),
               dur, kp, kd, progress="gd-chirp")
    hw.limp()
    A = samples_to_arrays(s)
    if len(A["t"]) < 100:
        return None
    dt = float(np.median(np.diff(A["t"])))
    x = A["q_cmd"] - np.mean(A["q_cmd"])
    y = A["q"] - np.mean(A["q"])
    c = np.correlate(y, x, mode="full")
    lag = int(np.argmax(c)) - (len(x) - 1)
    gd = lag * dt * 1e3
    log(f"    군지연(상호상관) {gd:.2f} ms")
    return gd if gd >= 0 else None


def _lost_motion(A: dict, dq_thresh: float = 1.5) -> tuple[float | None, float, int]:
    """저속 왕복 데이터에서 속도반전 dead-zone 폭[deg] 과 반전구간 토크 변화폭[Nm].

    방법: |q̇| < thresh 로 '멈춤 구간' 을 찾고, 그 구간 동안 **명령각이 얼마나 진행했는지**
    를 잰다. 관절이 멈춰 있는 동안 명령이 진행한 양 = 로스트모션(+추종오차).
    """
    if len(A["t"]) < 50:
        return None, float("nan"), 0
    dq = savgol_filter(A["dq"], min(31, (len(A["dq"]) // 2) * 2 - 1), 3)
    still = np.abs(dq) < dq_thresh
    widths, swings, n = [], [], 0
    i = 0
    while i < len(still):
        if not still[i]:
            i += 1
            continue
        j = i
        while j < len(still) and still[j]:
            j += 1
        # 멈춤 구간 [i, j) — 앞뒤로 실제 방향반전이 있었는지 확인
        if i > 2 and j < len(still) - 2:
            v_before = np.mean(dq[max(0, i - 8):i])
            v_after = np.mean(dq[j:min(len(dq), j + 8)])
            if v_before * v_after < 0 and (j - i) >= 3:      # 부호가 바뀜 = 진짜 반전
                widths.append(abs(A["q_cmd"][j - 1] - A["q_cmd"][i]))
                swings.append(float(np.max(A["tau"][i:j]) - np.min(A["tau"][i:j])))
                n += 1
        i = j + 1
    if not widths:
        return None, float("nan"), 0
    return float(np.median(widths)), float(np.median(swings)), n


def measure_latency_and_backlash(hw, spec, joint, plotdir, log=print) -> tuple[str, dict]:
    ch = int(joint["ch"])
    name = joint["name"]
    # ★축별 배포게인 (2026-08-12). spec.gains.kp 는 **없어진 키**다 — 게인이 스칼라
    #   하나였던 시절의 잔재이고, 축별 dict(safety.hold_kp)로 옮겨가며 지워졌다.
    #   단일 출처는 safety.hold_kp/hold_kd, 없으면 gains 의 스칼라로 떨어진다.
    _kp = spec.get("safety", {}).get("hold_kp", spec["gains"].get("kp"))
    _kd = spec.get("safety", {}).get("hold_kd", spec["gains"].get("kd"))
    kp = float(_kp[ch] if isinstance(_kp, dict) else _kp)
    kd = float(_kd[ch] if isinstance(_kd, dict) else _kd)
    warn: list[str] = []
    log(f"  [{name}] 지연·로스트모션 측정")

    log("  (1) 스텝응답 왕복지연")
    lat, traces = _latency(hw, spec, joint, kp, kd, log)
    if not lat["t_rt"]:
        raise RuntimeError("토크 반응을 한 번도 검출하지 못했다 — "
                           "step_deg 를 키우거나 게인을 올릴 것")
    t_rt = float(np.median(lat["t_rt"]))
    t_rt_sd = float(np.std(lat["t_rt"]))
    t_pos = float(np.median(lat["t_pos"])) if lat["t_pos"] else float("nan")

    log("  (2) 처프 군지연")
    gd = _group_delay(hw, spec, joint, kp, kd, log)

    log("  (3) 로스트모션 — 저속 왕복")
    cfg = spec.get("latency", {})
    amp = min(float(cfg.get("lm_amp_deg", 6.0)),
              (joint["q_max"] - joint["q_min"]) / 2 - 1.0)
    f_lm = float(cfg.get("lm_freq_hz", 0.08))
    q0 = hw.arm(ch, kp, kd)
    center = float(np.clip(q0, joint["q_min"] + amp + 1, joint["q_max"] - amp - 1))
    hw.goto(ch, center, kp, kd)
    s = hw.run(ch, lambda t: center + amp * np.sin(2 * np.pi * f_lm * t),
               2.0 / f_lm, kp, kd, progress="lost-motion")
    hw.limp()
    A = samples_to_arrays(s)
    lm, tau_swing, n_rev = _lost_motion(A)

    if lm is None:
        warn.append("속도반전 구간을 검출하지 못했다 — 진폭/주파수를 조정할 것")
    if t_rt_sd > 0.5 * t_rt:
        warn.append(f"왕복지연 산포가 크다(±{t_rt_sd:.2f} ms) — EtherCAT 사이클과 "
                    f"폴링의 비동기 비팅 가능. 시행횟수를 늘려 중앙값을 쓸 것")
    if gd is not None and gd < t_rt:
        warn.append("군지연이 왕복지연보다 작다 — 상호상관 추정이 불안정하다"
                    "(처프 대역/진폭 조정 필요)")

    # ── 플롯 ────────────────────────────────────────────────────────────────
    p_step = f"{plotdir}/latency_step_ch{ch:02d}.png"
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for r in traces[:4]:
        sgn = np.sign(r["delta"])
        ax[0].plot(r["t"] * 1e3, (r["q"] - r["q"][0]) * sgn, lw=1)
        ax[1].plot(r["t"] * 1e3, (r["tau"] - r["tau_base"]) * sgn, lw=1)
    ax[0].axvline(t_rt, ls="--", c="r", label=f"T_rt={t_rt:.2f} ms")
    ax[1].axvline(t_rt, ls="--", c="r")
    ax[0].set_ylabel("Δposition (deg)"), ax[0].legend(), ax[0].grid(alpha=.3)
    ax[1].set_ylabel("Δtorque (Nm)"), ax[1].set_xlabel("time since command write (ms)")
    ax[1].grid(alpha=.3)
    ax[0].set_title(f"Step response - {name} (sign-normalized overlay)")
    plt.savefig(p_step, dpi=110, bbox_inches="tight"), plt.close()

    p_rev = f"{plotdir}/lostmotion_ch{ch:02d}.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(A["q_cmd"], A["q"], lw=.8)
    ax[0].plot([A["q_cmd"].min(), A["q_cmd"].max()],
               [A["q_cmd"].min(), A["q_cmd"].max()], "k:", lw=.8, label="ideal (no lag)")
    ax[0].set_xlabel("command (deg)"), ax[0].set_ylabel("measured (deg)")
    ax[0].set_title("command vs measured (hysteresis)"), ax[0].legend(), ax[0].grid(alpha=.3)
    ax[1].plot(A["t"], A["dq"], lw=.7)
    ax[1].axhline(0, c="k", lw=.6)
    ax[1].set_xlabel("time (s)"), ax[1].set_ylabel("velocity (deg/s)")
    ax[1].set_title("velocity reversal"), ax[1].grid(alpha=.3)
    fig.suptitle(f"Lost motion — {name}")
    plt.savefig(p_rev, dpi=110, bbox_inches="tight"), plt.close()

    log(f"  [{name}] → T_rt={t_rt:.2f}ms · 군지연={gd if gd else float('nan'):.2f}ms · "
        f"lost motion={lm if lm else float('nan'):.4f}deg")

    warnings_html = ""
    if warn:
        warnings_html = ('<div class="warn"><b>주의</b><ul>'
                         + "".join(f"<li>{w}</li>" for w in warn) + "</ul></div>")

    html = TEMPLATE.render(
        title=f"Latency & Lost Motion — {name}",
        datetime=time.strftime("%Y-%m-%d %H:%M:%S"), joint=name, ch=ch,
        step_deg=float(cfg.get("step_deg", 1.5)), trials=len(lat["t_rt"]),
        poll_ms=lat["poll_ms"], t_rt=t_rt, t_rt_sd=t_rt_sd, t_pos=t_pos,
        gd_str=(f"{gd:.2f} ms" if gd is not None else "추정 실패"),
        ctrl_str=(f"{gd - t_rt:.2f} ms" if gd is not None else "—"),
        lm_str=(f"{lm:.4f} deg" if lm is not None else "검출 실패"),
        tau_swing=tau_swing, n_rev=n_rev, warnings=warnings_html,
        plot_step=p_step.replace(plotdir, "plots"), plot_rev=p_rev.replace(plotdir, "plots"))

    return html, {"t_rt_ms": t_rt, "t_rt_sd": t_rt_sd, "group_delay_ms": gd,
                  "lost_motion_deg": lm, "tau_swing": tau_swing,
                  "ch": ch, "name": name}
