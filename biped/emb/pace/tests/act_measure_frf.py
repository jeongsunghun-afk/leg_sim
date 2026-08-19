#!/usr/bin/env python3
"""act_measure_frf.py — 주파수응답(FRF)으로 드라이브트레인 공진·감쇠·강성 측정.

★왜 필요한가 (2026-08-05 실패에서 배움):
  스텝응답으로 링잉을 보려 했으나 **판정 불가**였다.
    · 2° 스텝 = 1.4 Nm 인데 마찰이 0.7 Nm → 절반이 마찰에 먹혀 과감쇠, 오버슛 0.0%
    · 공진을 보려면 탄성변형을 봐야 하는데 0.45 Nm 에서 겨우 0.045°
    · 그런데 위치 분해능이 float16 기준 ±15° 부근에서 0.0156° → **변형이 분해능의 3배뿐**
    · 속도 채널은 노이즈 ±15 deg/s 라 20 Hz 진동을 덮어버린다("진동횟수" 지표가
      공진이 아니라 노이즈 330 Hz 를 셌다)
  ⇒ 지표를 바꾸는 것으로는 해결이 안 되고 **SNR 자체를 올려야** 한다.

이 구현의 대응:
  ① **순수 토크 처프**로 가진 — 토크가 독립 입력이라 전달함수가 정의된다
     (위치+게인 모드의 tau 는 Kp·err 로 위치에서 계산되는 값이라 순환이 된다)
  ② **동기평균 N회** — 노이즈가 √N 로 준다. 20회면 4.5배
  ③ **타이트 폴링**(sleep 없음) — 명령은 루프속도로, 관측도 같은 속도로. 실제 시각을
     기록해 균일격자로 리샘플한다
  ④ **위치 채널만** 씀 — 속도 채널은 노이즈가 1500배 크다
  ⑤ **코히런스 γ²** 를 함께 낸다 — γ² < 0.8 인 주파수는 신뢰하지 않는다.
     오늘처럼 "SNR 이 부족한데 숫자는 나오는" 상황을 자동으로 걸러준다

산출:
  · 공진주파수 f_n        = |H| 피크
  · 감쇠비 ζ              = 반전력(−3dB) 대역폭 / (2·f_n)
  · 정적 컴플라이언스 1/k = |H| 저주파 점근값  ← 토션 히스테리시스 결과와 교차검증
  · 코히런스 스펙트럼     = 각 주파수의 신뢰도

⚠ 진폭 제약: 출력축이 자유로우면 기동토크(0.62 Nm) 이하로 유지해야 회전하지 않는다.
  **클램프로 고정하면**(spec.frf.clamped: true) 그 제약이 풀려 토크를 2~3배 걸 수 있고
  SNR 이 그만큼 개선된다 — 이것이 가장 효과적인 개선책이다.

⚠ 다리 미장착 상태의 공진은 **배포 조건이 아니다**. 관성이 10배가 되면
  f_n = √(k/J)/2π 가 √10 배 낮아져 20 Hz → 6.5 Hz 가 되고, 200 Hz 샘플링에
  30점/주기라 측정이 훨씬 정확해진다. **다리 장착 후에 하는 것이 맞다.**
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
from scipy import signal as sig

TEMPLATE = Template("""
<h2>{{ title }}</h2>
<p>순수 토크 처프로 가진하고 위치 응답의 전달함수 <code>H(f) = Q(f)/T(f)</code>(컴플라이언스)를
구한다. 피크가 드라이브트레인 공진, 저주파 점근값이 정적 컴플라이언스 1/k 다.
<b>코히런스 γ² 가 낮은 주파수는 신뢰하지 않는다.</b></p>

<table>
  <tr><th colspan="2">시험 조건</th></tr>
  <tr><td>일시</td><td>{{ datetime }}</td></tr>
  <tr><td>축</td><td>{{ joint }} (SHM ch{{ ch }})</td></tr>
  <tr><td>토크 처프</td><td class="numeric">±{{ '%.2f' % amp }} Nm · {{ f0 }}→{{ f1 }} Hz · {{ '%.1f' % dur }} s × {{ reps }}회 평균</td></tr>
  <tr><td>출력축 고정</td><td>{{ '클램프 고정 (고진폭 가능)' if clamped else '자유 (기동토크 이하로 제한)' }}</td></tr>
  <tr><td>실효 샘플링</td><td class="numeric">{{ '%.0f' % fs }} Hz (타이트 폴링)</td></tr>

  <tr><th colspan="2">결과 <span class="dim">(γ²≥{{ gmin }} 구간만)</span></th></tr>
  <tr><td><b>공진주파수 f_n</b></td><td class="numeric">{{ fn_str }}</td></tr>
  <tr><td><b>감쇠비 ζ</b></td><td class="numeric">{{ z_str }}</td></tr>
  <tr><td><b>정적 강성 k</b></td><td class="numeric">{{ k_str }}</td></tr>
  <tr><td>유효 대역</td><td class="numeric">{{ band_str }}</td></tr>
  <tr><td>평균 코히런스(유효대역)</td><td class="numeric">{{ '%0.3f' % gmean }}</td></tr>

  <tr><th colspan="2">제어 함의</th></tr>
  <tr><td>게인 상한 Kp_max ≈ k/9</td><td class="numeric">{{ kpmax_str }}</td></tr>
  <tr><td>권장 위치루프 대역</td><td class="numeric">{{ bw_str }}</td></tr>
</table>

{{ warnings }}
<p><img src="{{ plot }}"></p>
""")


def _chirp_tau(t, amp, f0, f1, T):
    k = (f1 - f0) / max(T, 1e-9)
    return amp * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def estimate_frf(t_list, tau_list, q_list, fs, gmin=0.8):
    """H1 추정 + 코히런스. 반복분을 균일격자로 리샘플해 동기평균한다.

    H1 = Pxy/Pxx 는 **출력 노이즈에 강건**하다(입력 노이즈에는 편향). 우리 경우
    입력=토크명령(노이즈 거의 없음), 출력=위치(양자화·노이즈 있음)이므로 H1 이 맞다.
    """
    n = min(len(x) for x in q_list)
    grid = np.arange(n) / fs
    X = np.vstack([np.interp(grid, t, x[:len(t)]) for t, x in zip(t_list, tau_list)])
    Y = np.vstack([np.interp(grid, t, y[:len(t)]) for t, y in zip(t_list, q_list)])
    x, y = X.mean(0), Y.mean(0)                     # 동기평균 → 노이즈 √N 감소
    nper = min(1024, len(x) // 4)
    f, Pxx = sig.welch(x, fs, nperseg=nper)
    _, Pyy = sig.welch(y, fs, nperseg=nper)
    _, Pxy = sig.csd(x, y, fs, nperseg=nper)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = Pxy / Pxx
        coh = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return f, H, np.nan_to_num(coh), x, y


def fit_second_order(f, H, coh, gmin=0.8):
    """복소 FRF 에 2차계 `H = 1/(J(jw)² + b(jw) + k)` 를 직접 피팅.

    ★−3dB 폭을 읽는 방식은 **경감쇠에서 실패한다**. ζ=0.01 이면 반전력 대역폭이
      2ζf_n = 0.4 Hz 인데 Welch 분해능 fs/nperseg 가 0.39 Hz — 대역폭이 1개 빈이라
      폭을 잴 수가 없다(셀프테스트에서 ζ 를 4배 과대추정했다).
      모델 피팅은 유효대역 전체를 쓰므로 빈 폭에 묶이지 않고, 덤으로 **관성 J** 가
      나와 PACE 처프 추정치와 교차검증된다.
    """
    ok = (coh >= gmin) & (f > 0)
    if ok.sum() < 10:
        return None
    w = 2 * np.pi * f[ok]
    # ★단위 통일: 측정 H 는 deg/Nm 이므로 rad/Nm 로 바꾼 뒤 피팅한다.
    #   이걸 안 하면 피팅된 k 가 이미 Nm/deg 인데 호출부에서 또 변환해 이중변환이 된다
    #   (셀프테스트에서 k 가 10.47 → 0.183 으로 57배 작게 나왔다).
    #   rad 로 통일하면 J[kg·m²]·b[N·m·s/rad]·k[Nm/rad] 가 전부 SI 로 나온다.
    Hm = H[ok] * (np.pi / 180.0)
    # 초기값: 저주파 점근 → k, 피크 → f_n → J
    k0 = 1.0 / max(abs(Hm[:3].mean()), 1e-12)
    fn0 = f[ok][int(np.argmax(np.abs(Hm)))]
    J0 = k0 / (2 * np.pi * fn0) ** 2
    b0 = 0.1 * np.sqrt(k0 * J0)

    def resid(p):
        J, b, k = np.exp(p)                       # 양수 보장
        Hh = 1.0 / (-J * w**2 + 1j * b * w + k)
        # 크기·위상 모두 맞춘다(로그 크기로 광대역 균형)
        return np.concatenate([np.log(np.abs(Hh)) - np.log(np.abs(Hm)),
                               np.angle(Hh * np.conj(Hm))])
    from scipy.optimize import least_squares
    try:
        sol = least_squares(resid, np.log([J0, b0, k0]), method="trf", max_nfev=8000)
    except Exception:
        return None
    J, b, k = np.exp(sol.x)
    return {"J": float(J), "b": float(b), "k_rad": float(k),   # k_rad = Nm/rad (SI)
            "f_n": float(np.sqrt(k / J) / (2 * np.pi)),
            "zeta": float(b / (2 * np.sqrt(k * J)))}


def analyze(f, H, coh, gmin=0.8):
    """공진(|H| 피크) · 감쇠비(반전력 대역폭) · 정적 강성(저주파 점근)."""
    mag = np.abs(H)
    ok = (coh >= gmin) & (f > 0)
    out = {"f_n": None, "zeta": None, "k": None, "band": None,
           "coh_mean": float(np.mean(coh[ok])) if ok.any() else 0.0}
    if ok.sum() < 8:
        return out, "유효 주파수(코히런스 기준) 부족 — SNR 미달"
    fo, mo = f[ok], mag[ok]
    out["band"] = (float(fo.min()), float(fo.max()))
    # 정적 컴플라이언스 = 유효대역 최저 3점 평균 (deg/Nm) → k [Nm/deg]
    lo = mo[:3].mean()
    if lo > 0:
        out["k"] = float(1.0 / lo)
    i = int(np.argmax(mo))
    if i in (0, len(mo) - 1):
        return out, "피크가 대역 경계에 있다 — 주파수 범위를 넓혀 재측정할 것"
    out["f_n"] = float(fo[i])                        # 피크 위치(참고값)

    # ★주 추정은 2차계 모델 피팅 — 빈 폭에 묶이지 않는다
    fit = fit_second_order(f, H, coh, gmin)
    if fit:
        out.update({"f_n": fit["f_n"], "zeta": fit["zeta"],
                    "k": fit["k_rad"] * np.pi / 180.0,   # Nm/rad → Nm/deg
                    "J_fit": fit["J"], "b_fit": fit["b"]})
        return out, None
    return out, "2차계 피팅 실패 — 피크 위치만 유효(감쇠비 미산출)" 


# ── 셀프테스트: 합성 2차계로 추정기 검증 (하드웨어 불필요) ──────────────────
def selftest(log=print) -> bool:
    """알려진 2차계 + 실측 수준 노이즈·양자화를 넣고 추정기가 되찾는지 확인."""
    J, b, k = 0.0363, 0.09, 600.0                    # 실측 기반 (관절축)
    fn_true = np.sqrt(k / J) / (2 * np.pi)
    z_true = b / (2 * np.sqrt(k * J))
    fs, T, reps = 400.0, 12.0, 20
    t = np.arange(0, T, 1 / fs)
    sys2 = sig.TransferFunction([1.0], [J, b, k])    # 토크→각(rad)
    rng = np.random.default_rng(1)
    tl, xl, yl = [], [], []
    for _ in range(reps):
        tau = _chirp_tau(t, 0.45, 1.0, 40.0, T)
        _, q, _ = sig.lsim(sys2, tau, t)
        q_deg = q * 180 / np.pi
        q_deg = q_deg + rng.normal(0, 0.004, t.size)         # 위치 노이즈
        q_deg = np.round(q_deg / 0.0156) * 0.0156            # ★float16 양자화(±15° 부근)
        tl.append(t); xl.append(tau); yl.append(q_deg)
    f, H, coh, _, _ = estimate_frf(tl, xl, yl, fs)
    r, err = analyze(f, H, coh)
    ok = True
    log("=== FRF 셀프테스트 (합성 2차계 + 실측 노이즈·양자화) ===")
    for nm, got, exp, tol in (("f_n [Hz]", r["f_n"], fn_true, 0.10),
                              ("zeta", r["zeta"], z_true, 0.60),
                              ("k [Nm/deg]", r["k"], k * np.pi / 180, 0.25)):
        if got is None:
            log(f"  [FAIL] {nm:12s} 추정 실패 ({err})"); ok = False; continue
        e = abs(got - exp) / abs(exp)
        good = e < tol; ok &= good
        log(f"  [{'OK ' if good else 'FAIL'}] {nm:12s} 참값 {exp:8.3f} → 추정 {got:8.3f} "
            f"(오차 {e*100:5.1f}%, 허용 {tol*100:.0f}%)")
    log(f"  평균 코히런스 {r['coh_mean']:.3f} · 유효대역 "
        f"{r['band'][0]:.1f}~{r['band'][1]:.1f} Hz" if r["band"] else "  유효대역 없음")
    log(f"=== FRF 셀프테스트 {'통과' if ok else '실패'} ===")
    return ok


def measure_frf(hw, spec, joint, plotdir, log=print) -> tuple[str, dict]:
    ch = int(joint["ch"]); name = joint["name"]
    cfg = spec.get("frf", {})
    clamped = bool(cfg.get("clamped", False))
    amp = float(cfg.get("amp_nm_clamped" if clamped else "amp_nm_free", 0.45))
    f0, f1 = float(cfg.get("f0_hz", 1.0)), float(cfg.get("f1_hz", 40.0))
    dur = float(cfg.get("duration_s", 12.0))
    reps = int(cfg.get("reps", 20))
    gmin = float(cfg.get("coherence_min", 0.8))
    warn: list[str] = []

    log(f"  [{name}] FRF — ±{amp} Nm 처프 {f0}→{f1} Hz {dur}s × {reps}회"
        f" ({'클램프 고정' if clamped else '자유(기동 이하)'})")
    if not clamped:
        warn.append("출력축이 자유롭다 — 진폭이 기동토크 이하로 제한되어 SNR 이 낮다. "
                    "<b>클램프로 고정하면 2~3배 큰 토크를 걸 수 있어 가장 효과적</b>이다.")

    tl, xl, yl, rates = [], [], [], []
    for r in range(reps):
        hw.arm(ch, 0.0, 0.0)
        time.sleep(0.15)
        q0 = hw.read(ch)[0]
        ts, xs, ys = [], [], []
        t0 = time.monotonic()
        while True:                                   # ★타이트 폴링(sleep 없음)
            t = time.monotonic() - t0
            if t >= dur:
                break
            tc = _chirp_tau(t, amp, f0, f1, dur)
            s = hw.step_torque(ch, tc, amp)
            ts.append(t); xs.append(tc); ys.append(s.q_deg - q0)
        hw.limp()
        time.sleep(0.2)
        if len(ts) < 200:
            log(f"    ⚠ rep{r}: 샘플 부족({len(ts)}) — 제외"); continue
        rates.append(len(ts) / ts[-1])
        tl.append(np.array(ts)); xl.append(np.array(xs)); yl.append(np.array(ys))
        if (r + 1) % 5 == 0:
            log(f"    {r+1}/{reps} 회 (샘플링 {rates[-1]:.0f} Hz)", )
    if len(tl) < 3:
        raise RuntimeError("유효 반복 3회 미만 — 측정 불가")

    fs = float(np.median(rates))
    f, H, coh, x, y = estimate_frf(tl, xl, yl, fs, gmin)
    res, err = analyze(f, H, coh, gmin)
    if err:
        warn.append(err)

    k_nm_deg = res["k"]
    kpmax = (k_nm_deg * 180 / np.pi / 9.0) if k_nm_deg else None
    bw = (res["f_n"] / 3.0) if res["f_n"] else None
    if res["coh_mean"] < gmin:
        warn.append(f"평균 코히런스 {res['coh_mean']:.2f} < {gmin} — <b>측정 신뢰 불가</b>. "
                    f"반복(reps)을 늘리거나 클램프로 고정해 진폭을 키울 것.")

    log(f"  [{name}] → f_n={res['f_n'] or float('nan'):.2f} Hz · "
        f"ζ={res['zeta'] or float('nan'):.3f} · k={k_nm_deg or float('nan'):.2f} Nm/deg · "
        f"코히런스 {res['coh_mean']:.3f}")

    p = f"{plotdir}/frf_ch{ch:02d}.png"
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    m = f > 0
    ax[0].loglog(f[m], np.abs(H[m]), lw=1)
    if res["f_n"]: ax[0].axvline(res["f_n"], ls="--", c="r", label=f"f_n={res['f_n']:.1f} Hz")
    ax[0].set_ylabel("|H| compliance (deg/Nm)"), ax[0].grid(alpha=.3, which="both")
    ax[0].legend(), ax[0].set_title(f"FRF — {name}")
    ax[1].semilogx(f[m], coh[m], lw=1)
    ax[1].axhline(gmin, ls="--", c="r", label=f"coherence gate {gmin}")
    ax[1].set_xlabel("frequency (Hz)"), ax[1].set_ylabel("coherence")
    ax[1].set_ylim(0, 1.05), ax[1].grid(alpha=.3, which="both"), ax[1].legend()
    plt.savefig(p, dpi=110, bbox_inches="tight"), plt.close()

    wh = ('<div class="warn"><b>주의</b><ul>' + "".join(f"<li>{w}</li>" for w in warn)
          + "</ul></div>") if warn else ""
    html = TEMPLATE.render(
        title=f"Frequency Response — {name}", datetime=time.strftime("%Y-%m-%d %H:%M:%S"),
        joint=name, ch=ch, amp=amp, f0=f0, f1=f1, dur=dur, reps=len(tl),
        clamped=clamped, fs=fs, gmin=gmin,
        fn_str=(f"{res['f_n']:.2f} Hz" if res["f_n"] else "미검출"),
        z_str=(f"{res['zeta']:.3f}" if res["zeta"] else "미검출"),
        k_str=(f"{k_nm_deg:.2f} Nm/deg ({k_nm_deg*180/np.pi:.0f} Nm/rad)" if k_nm_deg else "미검출"),
        band_str=(f"{res['band'][0]:.1f} ~ {res['band'][1]:.1f} Hz" if res["band"] else "없음"),
        gmean=res["coh_mean"],
        kpmax_str=(f"{kpmax:.0f} Nm/rad" if kpmax else "—"),
        bw_str=(f"{bw:.1f} Hz 이하" if bw else "—"),
        warnings=wh, plot=p.replace(plotdir, "plots"))
    return html, {**res, "k_nm_per_deg": k_nm_deg, "kp_max": kpmax,
                  "ch": ch, "name": name, "fs": fs, "reps": len(tl)}
