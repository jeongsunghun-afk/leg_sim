#!/usr/bin/env python3
"""prep_inertia.py — 매단 상태 관성 측정(PACE)의 **준비·해석** 도구.

측정 자체는 `actuator_test.py --tests pace` 가 한다. 이 스크립트는 그 앞뒤를 맡는다:

  (1) 측정 자세(HOME)에서의 기대값 계산 — `spec.yaml` 의 `I_link` 는 **q=0** 값이라
      HOME 에서 재면 hip 이 +34% 어긋난다. 자세별 값을 여기서 뽑는다.
  (2) **안전 진폭 산출** — spec 의 `amplitude_deg 6.0` 은 **다리 미장착** 시 잡은 값이다.
      등가관성이 6.9배가 된 지금 그대로 돌리면 4Hz 에서 16.5Nm 로 트립(8Nm)을 넘는다.
  (3) 홀드 민감도 — 홀드축이 강체가 아니면 유효관성이 달라진다. 축별로 그 폭을 보여준다.
  (4) 측정 후 **단위 복원** — PACE 는 채널각·보고토크로 식별하므로 얻는 값은 관절계가 아니다.

★왜 foot 부터인가 (2026-08-10 결정)
  · 안전   : 같은 처프에서 필요 토크가 hip 의 1/4.6 (I 0.0542 vs 0.2497)
  · 강건   : 홀드 강성 민감도 −1.2% (thigh −56% · calf −19% 와 대조)
  · 자세무관: foot I_link 는 q=0·HOME·기준자세에서 **전부 동일**(0.002017)
  · 직접성 : 등가관성의 96.3% 가 로터라 **ROTOR_I 를 가장 직접 잰다**

⚠foot 에만 있는 함정 둘 (hip 에는 없다):
  ① **단위** — PACE 는 채널각[deg]·보고토크[Nm]로 회귀한다. q_ch = k·q_joint,
     tau_ch = tau_joint/k 이므로  tau_ch = (I_joint/k²)·q̈_ch.
     ⇒ **식별값에 k² 를 곱해야** 관절계 관성이 된다(foot ×1.44, calf ×2.25).
     hip/thigh 는 k=1 이라 이 보정이 없다 — 그래서 여태 드러나지 않았다.
  ② **커플링 반작용** — foot 을 가진하면 전치 관계로 calf 모터가 τ_foot 을 떠안는다
     (τ_raw_calf = τ_calf − coef·τ_foot). calf 홀드가 그만큼 더 버텨야 하므로
     측정 후 `check_hold` 결과를 반드시 함께 볼 것.

사용법:
  python3 prep_inertia.py                      # 전 축 준비표(측정 전에 볼 것)
  python3 prep_inertia.py --axis foot          # 한 축만
  python3 prep_inertia.py --identified 0.0376 --axis foot   # 측정값 → 관절계 복원·판정
"""
from __future__ import annotations
import argparse
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.dirname(HERE)
MJCF = os.path.join(os.path.dirname(EMB), "biped_from_quad.mjcf")

AXES = ["hip", "thigh", "calf", "foot"]


def load_cfg():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(EMB, "config", "biped_emb.yaml")))
    spec = yaml.safe_load(open(os.path.join(HERE, "spec.yaml")))
    return cfg, spec


def mass_matrix(q_deg):
    """지정 자세의 관절 질량행렬(base 고정, armature 제외). MuJoCo 가 단일 출처."""
    import mujoco, numpy as np
    m = mujoco.MjModel.from_xml_path(MJCF)
    d = mujoco.MjData(m)
    # ★armature 를 0 으로 지운다 — 안 지우면 로터 반사관성이 I_link 에 섞이고,
    #   그걸 다시 ROTOR_I 분리에 쓰면 순환논법이 된다(RESULTS.md §11-a 와 동일 이유).
    m.dof_armature[:] = 0.0
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    d.qpos[7:] = np.radians(q_deg)
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    return M[6:, 6:]          # base 6dof 제외 = 매단 상태


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=AXES, help="한 축만")
    ap.add_argument("--identified", type=float,
                    help="PACE 가 낸 관성[kg·m², 채널계] → 관절계로 복원하고 판정")
    ap.add_argument("--trip-nm", type=float, default=None, help="토크 트립[Nm]. 기본 spec.safety")
    ap.add_argument("--margin", type=float, default=0.5, help="트립 대비 사용률 상한(0~1)")
    a = ap.parse_args()

    import numpy as np
    cfg, spec = load_cfg()
    home = [float(x) for x in cfg["home"]["q_deg"]]
    names = [j["name"] for j in cfg["joints"]]
    K = {n.split("_")[1]: float(j.get("gear_k", 1.0)) for n, j in zip(names, cfg["joints"])}
    GEAR = {"hip": 7.0, "thigh": 7.0, "calf": 10.5, "foot": 8.4}
    ROTOR_I = 7.4e-4
    ARM = {k: ROTOR_I * GEAR[k] ** 2 for k in AXES}
    chirp = spec["pace"]["chirp"]
    trip = a.trip_nm if a.trip_nm is not None else float(spec["safety"]["tau_trip_nm"])
    f_end = float(chirp["f_end_hz"])

    J = mass_matrix(home)
    idx = {k: names.index(f"HL_{k}") for k in AXES}
    axes = [a.axis] if a.axis else AXES

    # ── 측정 후 해석 모드 ────────────────────────────────────────────────
    if a.identified is not None:
        if not a.axis:
            sys.exit("✗ --identified 는 --axis 와 함께 써야 한다(축마다 k 가 다르다)")
        k = K[a.axis]
        I_joint = a.identified * k * k
        I_link_exp = float(J[idx[a.axis], idx[a.axis]])
        exp_total = I_link_exp + ARM[a.axis]
        rotor = (I_joint - I_link_exp) / GEAR[a.axis] ** 2
        print(f"\n  {a.axis} — PACE 식별값 해석 (측정 자세 = HOME {home})\n")
        print(f"    식별값(채널계)        {a.identified:.6f} kg·m²")
        print(f"    × k² ({k}²={k*k:.2f})       {I_joint:.6f} kg·m²   ← 관절계 등가관성")
        print(f"    기대 I_total          {exp_total:.6f}  (= I_link {I_link_exp:.6f} + armature {ARM[a.axis]:.6f})")
        print(f"    편차                  {(I_joint/exp_total-1)*100:+.1f}%")
        print(f"\n    역산 ROTOR_I = (I_joint − I_link)/N² = {rotor:.3e} kg·m²")
        print(f"    현재 채택값                          = {ROTOR_I:.3e}   편차 {(rotor/ROTOR_I-1)*100:+.1f}%")
        if a.axis == "foot":
            print(f"\n    ⚠foot 은 링크비중이 {I_link_exp/exp_total*100:.1f}% 뿐이라 이 측정은 사실상 ROTOR_I 직접측정이다.")
            print(f"      반대로 I_link 검증에는 못 쓴다 — 그건 hip(링크 {J[idx['hip'],idx['hip']]/(J[idx['hip'],idx['hip']]+ARM['hip'])*100:.0f}%) 담당.")
        return 0

    # ── 측정 전 준비표 ──────────────────────────────────────────────────
    print(f"\n  측정 자세 = HOME {home}   (spec.yaml 의 I_link 는 q=0 값이라 그대로 쓰면 안 된다)")
    print(f"  처프 {chirp['f_start_hz']}→{f_end}Hz · {chirp['duration_s']}s · 트립 {trip}Nm (사용률 상한 {a.margin:.0%})\n")
    print(f"  {'축':7} {'I_link(HOME)':>13} {'armature':>10} {'I_total':>10} {'링크%':>7} "
          f"{'홀드민감':>9} {'안전진폭':>9} {'spec 6°토크':>11}")
    print("  " + "-" * 88)
    for ax in axes:
        i = idx[ax]
        o = [k2 for k2 in range(len(names)) if k2 != i]
        Joo = J[np.ix_(o, o)]
        Joi = J[np.ix_(o, [i])]
        schur = float(J[i, i] - (Joi.T @ np.linalg.solve(Joo, Joi)).item())
        I_link = float(J[i, i])
        I_tot = I_link + ARM[ax]
        sens = ((schur + ARM[ax]) / I_tot - 1) * 100          # 홀드가 완전히 풀렸을 때 변화
        w = 2 * math.pi * f_end
        A_safe = math.degrees(trip * a.margin / (I_tot * w * w))
        tau6 = I_tot * w * w * math.radians(6.0)
        flag = "❌" if tau6 > trip else "✅"
        print(f"  {ax:7} {I_link:13.6f} {ARM[ax]:10.6f} {I_tot:10.6f} "
              f"{I_link/I_tot*100:6.1f}% {sens:+8.1f}% {A_safe:8.2f}° {tau6:9.2f}Nm{flag}")
    print(f"\n  홀드민감 = 홀드축이 **완전히 풀렸을 때** 유효관성 변화(Schur 보수).")
    print(f"    0 에 가까울수록 홀드 강성과 무관 ⇒ 신뢰. 큰 축은 check_hold 결과를 반드시 함께 볼 것.")
    print(f"  안전진폭 = {f_end}Hz 에서 관성토크가 트립의 {a.margin:.0%} 를 넘지 않는 진폭.")
    print(f"\n  ★권장 순서: foot → hip → calf → thigh")
    print(f"    foot 은 안전(토크 최소)·홀드무관·자세무관·ROTOR_I 직접측정 네 가지를 다 만족한다.")
    print(f"    thigh 는 홀드민감이 가장 커서 마지막이다.")
    print(f"\n  ⚠측정 후 반드시 --identified 로 **k² 복원**할 것 — PACE 값은 채널계다"
          f" (foot ×{K['foot']**2:.2f} · calf ×{K['calf']**2:.2f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
