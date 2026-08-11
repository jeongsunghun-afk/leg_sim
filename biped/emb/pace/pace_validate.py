#!/usr/bin/env python3
"""pace_validate.py — PACE 식별 결과 **정합 검증** (체크리스트 Phase 3·5).

식별(`act_identify_pace.py`)이 낸 θ 가 실제로 실기를 재현하는지 본다.
하드웨어를 건드리지 않는다 — 저장된 `pace_dataset_ch##.npz` 만 읽는다.

★두 단계 검증. **둘의 난이도가 다르다는 게 핵심이다.**

  ① 토크 재현 (약한 검증) — 실측 (q, q̇, q̈) 을 모델에 넣어 τ 를 예측하고 실측 τ 와 비교.
     회귀가 최소화한 바로 그 잔차라 좋게 나오는 게 당연하다. R² 가 높아도 "모델이 맞다"
     는 뜻이 아니다.

  ② 궤적 재현 (강한 검증) — 실측 τ 만 입력으로 주고 **운동방정식을 적분**해 q(t) 를 만든다.
        q̈ = (τ − b·q̇ − τ_c·tanh(q̇/ε) − A·sin(q) − c) / I_total
     여기서 어긋나면 모델이 틀린 것이다. 회귀가 못 보던 오차가 여기서 드러난다.
     ⚠순수 개루프 적분은 작은 편향도 시간에 따라 누적돼 발산한다 — 그건 모델 결함이
       아니라 적분의 성질이다. 그래서 **창(window)마다 실측 상태로 재초기화**해
       "국소 정합도" 와 "누적 드리프트" 를 분리해 본다. 둘 다 그린다.

★잔차를 **속도·위치에 대해** 그린다 — 시간축 잔차만 보면 구조가 안 보인다.
  · 잔차 vs q̇  : 마찰 모델이 틀리면 여기서 부호가 갈린 구조가 나온다(Stribeck 미반영 등)
  · 잔차 vs q   : 위치의존 성분(코깅·기어 편심·중력항 미스핏)이 여기서 주기 구조로 보인다
  R² 하나로는 이 둘을 절대 못 본다 — 체크리스트 Phase 5 "잔차 계통분석" 이 이것이다.

사용법:
  python3 pace_validate.py results/pace_dataset_ch00.npz
  python3 pace_validate.py results/*.npz --outdir results/validate
  python3 pace_validate.py results/pace_dataset_ch00.npz --window 0.5
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

# ★한글 폰트 — 없으면 라벨이 두부(□)로 깨진다. 시스템에 Noto Sans CJK 가 있으므로 지정한다.
#   (fc-list :lang=ko 로 확인. matplotlib 은 CJK JP 로 잡히지만 한글 글리프를 포함한다)
for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "DejaVu Sans"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        matplotlib.rcParams["font.family"] = _f
        break
    except Exception:
        continue
matplotlib.rcParams["axes.unicode_minus"] = False   # 한글 폰트에 U+2212 가 없어 −가 깨진다

# ── 검증된 팔레트 (dataviz 기준 인스턴스, light) ──────────────────────────
#   categorical slot1 blue = 실측 · slot2 orange = 모델. 2계열이라 인접쌍 검사 통과.
#   텍스트는 **계열색을 쓰지 않는다** — ink 토큰만 쓴다.
C_MEAS, C_SIM = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#d8d7d2", "#fcfcfb"


def style(ax, xlabel, ylabel, title=None):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def simulate(t, tau, q0, dq0, th, eps):
    """실측 τ 를 입력으로 운동방정식을 적분(RK4). θ=(I,b,tc,A,c)."""
    I, b, tc, A, c = th
    q = np.empty_like(t); dq = np.empty_like(t)
    q[0], dq[0] = q0, dq0

    def acc(qq, dqq, tt):
        return (tt - b * dqq - tc * np.tanh(dqq / eps) - A * np.sin(qq) - c) / I

    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        tm = 0.5 * (tau[k] + tau[k + 1])
        k1v = acc(q[k], dq[k], tau[k]);                 k1x = dq[k]
        k2v = acc(q[k] + .5*h*k1x, dq[k] + .5*h*k1v, tm); k2x = dq[k] + .5*h*k1v
        k3v = acc(q[k] + .5*h*k2x, dq[k] + .5*h*k2v, tm); k3x = dq[k] + .5*h*k2v
        k4v = acc(q[k] + h*k3x, dq[k] + h*k3v, tau[k+1]); k4x = dq[k] + h*k3v
        q[k+1]  = q[k]  + h/6*(k1x + 2*k2x + 2*k3x + k4x)
        dq[k+1] = dq[k] + h/6*(k1v + 2*k2v + 2*k3v + k4v)
    return q, dq


def one(path, outdir, window_s, log=print) -> dict:
    d = np.load(path, allow_pickle=True)
    name = str(d["name"]); ch = int(d["ch"]); eps = float(d["eps"])
    th = [float(x) for x in d["theta"][:5]]                 # I, b, tc, A, c (Bb 미사용)
    I, b, tc, A, c = th
    t = np.asarray(d["t"], float)
    q = np.radians(np.asarray(d["q"], float))
    dq = np.radians(np.asarray(d["dq"], float))
    tau = np.asarray(d["tau"], float)

    # q̈ — 식별과 **같은** 방식이어야 한다(다르면 비교가 성립하지 않는다)
    from scipy.signal import savgol_filter
    dt = float(np.median(np.diff(t)))
    ddq = savgol_filter(dq, 31, 3, deriv=1, delta=dt)

    # ── ① 토크 재현 ─────────────────────────────────────────────────────
    tau_pred = I*ddq + b*dq + tc*np.tanh(dq/eps) + A*np.sin(q) + c
    r = tau - tau_pred
    ss = float(np.sum((tau - tau.mean())**2))
    r2_tau = 1.0 - float(np.sum(r**2))/ss if ss > 0 else float("nan")

    # ── ② 궤적 재현 ─────────────────────────────────────────────────────
    q_open, _ = simulate(t, tau, q[0], dq[0], th, eps)       # 개루프(누적 드리프트 포함)
    n_w = max(2, int(round(window_s/dt)))                    # 창마다 실측으로 재초기화
    q_win = np.empty_like(q)
    for s0 in range(0, len(t), n_w):
        s1 = min(s0 + n_w, len(t))
        if s1 - s0 < 2:
            q_win[s0:s1] = q[s0:s1]; continue
        qs, _ = simulate(t[s0:s1], tau[s0:s1], q[s0], dq[s0], th, eps)
        q_win[s0:s1] = qs
    e_win = np.degrees(q_win - q)
    e_open = np.degrees(q_open - q)

    # ── 그림 ────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.0), facecolor=SURF)
    fig.suptitle(f"PACE 정합 검증 — {name} (ch{ch})", color=INK, fontsize=12,
                 x=0.012, ha="left", y=0.985)

    # 시간축은 앞 6초만 — 전 구간을 그리면 선이 뭉개져 아무것도 안 보인다
    m = t - t[0] <= min(6.0, t[-1] - t[0])
    ts = t[m] - t[0]

    ax = axs[0, 0]
    ax.plot(ts, tau[m], color=C_MEAS, lw=2.0, label="실측 τ")
    ax.plot(ts, tau_pred[m], color=C_SIM, lw=2.0, ls="--", label="모델 예측 τ")
    style(ax, "시간 [s]", "토크 [Nm]", f"① 토크 재현 (약한 검증)   R² = {r2_tau:.4f}")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    ax = axs[0, 1]
    ax.plot(ts, np.degrees(q[m]), color=C_MEAS, lw=2.0, label="실측 q")
    ax.plot(ts, np.degrees(q_win[m]), color=C_SIM, lw=2.0, ls="--",
            label=f"모델 q ({window_s:g}s 창 재초기화)")
    style(ax, "시간 [s]", "각도 [deg]",
          f"② 궤적 재현 (강한 검증)   RMS {np.sqrt(np.mean(e_win**2)):.3f}°")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")

    ax = axs[1, 0]
    ax.scatter(np.degrees(dq), r, s=4, color=C_MEAS, alpha=0.35, edgecolors="none")
    ax.axhline(0, color=INK2, lw=1.0)
    style(ax, "각속도 q̇ [deg/s]", "잔차 (실측 − 모델) [Nm]",
          "③ 잔차 vs 속도 — 마찰모델 결함이 여기 나온다")
    ax.annotate("구조가 보이면 Stribeck·비대칭 마찰 미반영", xy=(0.02, 0.04),
                xycoords="axes fraction", color=INK2, fontsize=8)

    ax = axs[1, 1]
    ax.scatter(np.degrees(q), r, s=4, color=C_MEAS, alpha=0.35, edgecolors="none")
    ax.axhline(0, color=INK2, lw=1.0)
    style(ax, "각도 q [deg]", "잔차 (실측 − 모델) [Nm]",
          "④ 잔차 vs 위치 — 코깅·기어편심이 여기 나온다")
    ax.annotate("주기 구조가 보이면 위치의존 성분 존재", xy=(0.02, 0.04),
                xycoords="axes fraction", color=INK2, fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, f"validate_ch{ch:02d}.png")
    fig.savefig(png, dpi=120, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)

    # ── 잔차 계통성 판정 (그림을 안 보고도 알 수 있게) ──────────────────
    def corr(x, y):
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    c_dq, c_q = corr(np.abs(dq), np.abs(r)), corr(np.abs(q), np.abs(r))

    log(f"\n  {name} (ch{ch})   θ: I={I:.6f} b={b:.4f} τc={tc:.4f} A={A:.4f} c={c:.4f}")
    log(f"    ① 토크 재현    R² {r2_tau:.4f} · 잔차 RMS {r.std():.4f} Nm")
    log(f"    ② 궤적 재현    창 {window_s:g}s RMS {np.sqrt(np.mean(e_win**2)):.3f}° · "
        f"최대 {np.abs(e_win).max():.3f}°")
    log(f"       개루프 전구간 최종 오차 {e_open[-1]:+.1f}° "
        f"(누적 드리프트 — 모델 결함이 아니라 적분 성질)")
    log(f"    ③④ 잔차 계통성  |r| vs |q̇| {c_dq:+.3f} · |r| vs |q| {c_q:+.3f}"
        f"   {'⚠구조 의심' if max(abs(c_dq), abs(c_q)) > 0.3 else '✅ 무작위에 가깝다'}")
    log(f"    → {png}")
    return {"name": name, "ch": ch, "r2_tau": r2_tau,
            "rms_win_deg": float(np.sqrt(np.mean(e_win**2))), "png": png}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--outdir", default=None, help="기본: npz 폴더/validate")
    ap.add_argument("--window", type=float, default=0.5,
                    help="궤적 재현 시 재초기화 창[s]. 작을수록 국소 정합만 본다")
    a = ap.parse_args()
    rc = 0
    for p in a.npz:
        if not os.path.exists(p):
            print(f"✗ 없다: {p}"); rc = 1; continue
        out = a.outdir or os.path.join(os.path.dirname(os.path.abspath(p)), "validate")
        try:
            one(p, out, a.window)
        except Exception as e:
            print(f"✗ {p}: {type(e).__name__}: {e}"); rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
