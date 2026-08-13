#!/usr/bin/env python3
"""compare_params.py — **각축 회귀 실측**과 **PACE 식별**을 나란히 놓는다.

★왜 별도 도구인가 (2026-08-12)
  두 방법은 **다른 것을 잰다.** 겹치는 건 JFRIC 하나뿐이고, 나머지는 한쪽만 준다.
  그걸 표로 못박아 두지 않으면 "PACE 가 τ_s 를 안 준다" 같은 헛된 비교를 반복한다.

      파라미터   각축 회귀            PACE          비교
      ROTOR_I   ✗ 못 잰다            ✓             불가 — PACE 만의 값
      JDAMP     △ 전부 nan/≈0        ✓             **PACE 가 채우는 칸**
      JFRIC     ✓ 8축 실측           ✓             **직접 비교 가능**
      τ_s       ✓ 8축 실측           ✗ 모델에 항 없음  불가 — MuJoCo dof_frictionloss 는
                                                   쿨롱뿐이라 정지마찰 항이 없다
      bias      ✗                   ✓ 축별         불가
      delay     ✓ 8.39ms 별도실측     ✓ ±2σ 로 묶음   이미 실측을 넣어 뒀다

⚠비교할 때 **두 가지를 맞춰야 한다**
  ① 프레임 — 실측은 **채널토크**, PACE(dof_frictionloss)는 **관절토크**다. ×gear_k.
  ② 묶음   — 실측은 축별 8개, PACE 기본은 kind별 4개(좌우 공유)다. 좌우 평균으로 본다.

사용:
    python3 tools/compare_params.py emb/pace/results/pace_multichirp_f0.4_cmaes.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(HERE)
SPEC = os.path.join(BIPED, "emb", "pace", "spec.yaml")
KINDS = ["hip", "thigh", "calf", "foot"]


def kind_of(name: str) -> str:
    for k in KINDS:
        if k in name:
            return k
    raise ValueError(name)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    z = np.load(sys.argv[1], allow_pickle=True)
    names = [str(x) for x in z["names"]]
    x, x0 = np.asarray(z["x"], float), np.asarray(z["x0"], float)
    per_axis = bool(z["per_axis"])
    sp = yaml.safe_load(open(SPEC, encoding="utf-8"))
    gk = {int(j["ch"]): float(j.get("gear_k", 1.0)) for j in sp["joints"]}
    mc = (sp.get("friction") or {}).get("measured_coulomb_ch") or {}
    ts = (sp.get("friction") or {}).get("measured_tau_static_ch") or {}
    n = len(names)

    # 탐색벡터 배치 — init_bounds/param_labels 와 같은 순서다
    nj = n if per_axis else len(KINDS)
    dmp, frc = x[1:1 + nj], x[1 + nj:1 + 2 * nj]
    dmp0, frc0 = x0[1:1 + nj], x0[1 + nj:1 + 2 * nj]

    print(f"■ {os.path.basename(sys.argv[1])}"
          f"  ·  RMS 적합 {float(z['rms_fit']):.4f}° · hold-out {float(z['rms_holdout']):.4f}°")
    print(f"  ROTOR_I  {x[0]:.4e}  (초기 {x0[0]:.4e} · {x[0] / x0[0] - 1:+.0%})\n")

    # ── JFRIC — 유일하게 직접 비교되는 값 ──────────────────────────────────
    print("■ JFRIC [관절토크 Nm] — **직접 비교 가능한 유일한 파라미터**")
    print(f"  {'kind':<7}{'실측 HL/HR(채널)':>18}{'gear_k':>7}{'실측(관절)':>11}"
          f"{'PACE':>9}{'PACE/실측':>10}  판정")
    for i, k in enumerate(KINDS if not per_axis else names):
        if per_axis:
            c = int([j["ch"] for j in sp["joints"] if j["name"] == k][0])
            meas_ch = [float(mc.get(c, np.nan))]
            g = gk[c]
        else:
            chs = [int(j["ch"]) for j in sp["joints"] if kind_of(j["name"]) == k]
            meas_ch = [float(mc[c]) for c in chs if c in mc]
            g = gk[chs[0]]
        if not meas_ch or np.isnan(meas_ch[0]):
            continue
        mj = float(np.mean(meas_ch)) * g
        r = frc[i] / mj if mj else np.nan
        v = ("✓ 실측과 일치(±15%)" if 0.85 <= r <= 1.15 else
             ("★경계에 붙었다 — 실측과 모델이 어긋난다" if r <= 0.72 or r >= 1.28 else
              "△ 어긋나지만 경계 안"))
        s = "/".join(f"{v_:.3f}" for v_ in meas_ch)
        print(f"  {k:<7}{s:>18}{g:>7.1f}{mj:>11.3f}{frc[i]:>9.3f}{r:>9.2f}x  {v}")

    # ── JDAMP — 각축 회귀가 못 얻은 칸 ────────────────────────────────────
    print("\n■ JDAMP [Nm·s/rad] — **각축 회귀로는 못 얻었다**(8축 전부 nan 또는 ≈0)")
    print(f"  {'kind':<7}{'초기값':>10}{'PACE':>10}{'변화':>9}  비고")
    for i, k in enumerate(KINDS if not per_axis else names):
        d = dmp[i]
        v = "★음수 — 무의미" if d < 0 else ("△ 초기값에 붙어 있다 — 식별 안 됨"
                                          if abs(d / dmp0[i] - 1) < 0.02 else "✓ 움직였다")
        print(f"  {k:<7}{dmp0[i]:>10.4f}{d:>10.4f}{d / dmp0[i] - 1:>+8.0%}  {v}")

    # ── 비교 불가 항목을 **명시**한다 ─────────────────────────────────────
    print("\n■ 비교 불가 (한쪽만 갖고 있다)")
    print("  τ_s [정지마찰] — 각축 회귀만. MuJoCo dof_frictionloss 는 쿨롱뿐이라 항이 없다")
    for k in KINDS:
        pair = [(j["name"], float(ts[int(j["ch"])])) for j in sp["joints"]
                if kind_of(j["name"]) == k and int(j["ch"]) in ts]
        if pair:
            print(f"    {k:<7}" + " · ".join(f"{nm} {v:.3f}" for nm, v in pair)
                  + f"   [채널토크 Nm]")
    print("  ROTOR_I·bias — PACE 만. 각축 회귀는 이 둘을 안 건드린다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
