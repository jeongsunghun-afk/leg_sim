#!/usr/bin/env python3
"""collect_multichirp.py — **전축 동시 처프** 수집. PACE(궤적 재현매칭) 입력용.

★왜 한 축씩이 아니라 전축 동시인가 (2026-08-11)
  한 축씩 재는 해석식은 **대각항만** 본다:
        τ_i = I_ii·q̈_i + b·q̇_i + τ_c·sgn + g
  이게 성립하려면 다른 축의 q̈ 가 0 이어야 한다 — 그래서 지그로 묶었다.
  실제 보행은 전 축이 같이 움직이고, 그때는 비대각항이 산다:
        τ_i = Σ_j M_ij(q)·q̈_j + C_i(q,q̇) + g_i + 마찰_i
  M(q)·C·g 를 해석식으로 다시 쓰는 건 무리다. **시뮬레이터가 이미 정확히 들고 있다.**
  ⇒ PACE 방식: 시뮬을 실기와 **같은 제어기·같은 명령**으로 굴려 q(t) 가 겹치도록
    액추에이터 파라미터만 탐색한다(pace_cmaes.py).

★이 방식이 **순환 문제를 통째로 없앤다**
  목적함수가 Σ(q_sim − q_real)² 라 드라이버가 보고한 τ 를 **아예 쓰지 않는다**.
  τ 가 kp·err 로 재구성되든 말든 상관이 없다.

★축을 **비상관화**해야 한다 — 이게 이 스크립트의 핵심이다.
  전 축에 같은 처프를 넣으면 궤적이 완전히 상관되어 파라미터를 못 가른다
  (한 축의 관성을 키우고 다른 축을 줄여도 같은 궤적이 나온다).
  ⇒ 축마다 **다른 위상 + 다른 주파수 기울기**를 준다. 상관계수를 수집 후 보고한다.

⚠안전
  · 위치+게인 모드다. 토크가 kp·err 로 자기제한되므로 폭주가 구조적으로 불가능하다.
  · **자기충돌 포락선을 MJCF 로 확인했다**(꼭짓점 2^8 전수):
        전축 동일 진폭 ±10° 안전 · ±15° 부터 두 발 충돌(−70mm)
        **hip 을 ±8° 로 묶으면 나머지는 ±30° 까지 전부 안전**
    원인은 hip 이다 — 내전하면 발이 모인다. 그래서 hip 진폭만 따로 작게 잡는다.
  · 진폭을 처음 2초간 램프로 올린다(계단 인가 금지).
  · 매 틱 전 채널 한계검사. 어떤 경로로 끝나든 limp.

사용:
    python3 collect_multichirp.py --jig-off        # 지그를 **뺀** 상태여야 한다
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from dataclasses import replace
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "tests"), os.path.join(EMB, "interface")]

from hwio import DEG, Hardware, Limits, SafetyAbort   # noqa: E402
from homing import goto_home, make_homer             # noqa: E402
from joint_map import JointMap                       # noqa: E402
from state_pub import publish_state, leg_extra                  # noqa: E402
import actuator_test as at                           # noqa: E402


def chirp_bank(cfg, n, t_end):
    """축별 처프 설계. **위상과 주파수 기울기를 다르게** 줘서 상관을 깬다.

    q_i(t) = A_i · sin(2π(f0_i·t + ½k_i t²) + φ_i)

    φ 는 황금비 수열(0.618·i mod 1)로 준다 — 균등분포에 가깝고 재현 가능하다
    (난수를 쓰면 수집마다 달라져 비교가 안 된다).
    """
    amps = np.array(cfg["amp_deg"], float)
    f0 = np.array(cfg["f_start_hz"], float)
    f1 = np.array(cfg["f_end_hz"], float)
    if amps.size == 1:
        amps = np.repeat(amps, n)
    if f0.size == 1:
        f0 = np.repeat(f0, n)
    if f1.size == 1:
        f1 = np.repeat(f1, n)
    phi = (np.arange(n) * 0.6180339887) % 1.0 * 2 * np.pi
    k = (f1 - f0) / t_end
    return amps, f0, k, phi


def q_at(t, amps, f0, k, phi, ramp_s, T=None):
    """★진폭 램프는 **양쪽**이다 (2026-08-12). 종전엔 올리기만 했다.

    가진이 t=T 에서 뚝 끊기면 축이 **최대속도로 달리는 중에** 명령이 사라진다.
    실기 첫 다축주행에서 그 뒤 홈복귀가 오차 9.40° 로 실패했다(HR_thigh) —
    58dps 로 달리던 축을 1.4초 궤적으로 세우려 했으니 당연하다.
    ⇒ 끝 ramp_s 동안 진폭을 0 으로 접는다. 정지 상태로 끝나 홈복귀가 자명해진다.
    ⚠식별에는 무해하다. 시뮬이 **같은 q_cmd 를 재생**하므로 램프도 그대로 재현된다.
      마지막 2초가 저진폭이 될 뿐이다(--holdout 이 거기 걸리면 조금 약해진다).
    """
    s = min(t / ramp_s, 1.0) if ramp_s > 0 else 1.0     # 진폭 램프(계단 금지)
    if T is not None and ramp_s > 0:
        s = min(s, max(0.0, (T - t) / ramp_s))          # ★끝에서 접는다
    ph = 2 * np.pi * (f0 * t + 0.5 * k * t * t) + phi
    return amps * s * np.sin(ph)


def apply_mirror(dev, names, mode):
    """★좌우 대칭으로 base net wrench 를 상쇄한다 (PACE arXiv:2509.06342 §3.2.2).

    원문은 in-air 식별에서 *"symmetric trajectory commands to cancel net wrenches"* 를 쓴다.
    **우리는 크레인에 매달아 잰다** — 반력이 크면 base 가 흔들리고, 그 흔들림은
    시뮬(base 고정)에 없으므로 통째로 식별오차가 된다. IMU 도 없어 검출조차 못 한다.

    ⚠처프는 순시주파수가 변하므로 **시간 이동으로는 역위상이 안 된다.**
      sin(θ+π) = −sin(θ) 이므로 **부호 반전**이 곧 역위상이다.

    mode:
      neg — 전 축 역위상 `q_R = −q_L`. **채택값.**
      hip — hip 만 역위상(기하 거울).

    실측(design_excitation.py, 6s, 체중 136N):
        현행(황금비 위상)  수평력 x 3.69 N (2.7%) · 모멘트 [0.52 0.66 1.11]
        **neg**            수평력 x **1.76 N (1.3%)** · 모멘트 [0.32 0.25 1.16]   ← 절반 이하
        hip                수평력 x 9.60 N (7.0%) ← 두 다리가 시상면에서 같이 흔든다. 나쁘다
      식별력은 사실상 불변(조건수 5.4→5.5 · ROTOR_I 감도 0.1036→0.1002).

    ⚠좌우가 종속이 되므로 **좌우를 따로 식별하는 모드(--per-axis)와는 양립하지 않는다.**
      우리 기본 파라미터화는 kind별 공유(ROTOR_I 1 + JDAMP 4 + JFRIC 4)라 손해가 없다.
      좌우 차이를 보려면(예: HL/HR foot 관성 −4.9/+3.0%) 이 옵션을 끄고 따로 수집할 것.
    """
    if not mode:
        return dev
    half = len(names) // 2
    if mode == "neg":
        sg = -np.ones(half)
    elif mode == "hip":
        sg = np.array([-1.0 if "hip" in names[i] else 1.0 for i in range(half)])
    else:
        raise SystemExit(f"✗ pace_multi.mirror 는 neg 또는 hip (받은 값: {mode})")
    out = np.array(dev, float).copy()
    out[..., half:] = sg * out[..., :half]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--config", default=os.path.join(EMB, "config", "biped_emb.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--T", type=float, default=None, help="수집 길이[s] (기본 spec)")
    ap.add_argument("--gains", choices=["id", "validate"], default="id",
                    help="★id=식별용(기본) · validate=검증용 게인셋. "
                         "원문의 'unseen PD gains' 검증 — 다른 게인에서 같은 θ 가 나오는지 본다")
    # ★단계적 상향용 (2026-08-12, 사용자 제안 "처음부터 높은 건 좀"). f_end 만 곱한다 —
    #   f_start 는 그대로라 **여전히 낮은 데서 시작해 쓸어 올린다**. 도달 상한만 낮춘다.
    #   ⚠양발 간격은 **위치 포락선**이 정하므로 이 값과 무관하다(진폭이 안 바뀐다).
    #     낮추는 건 동적 위험(추종실패·공진·토크)만 줄인다.
    #   ⚠1.0 이 아니면 파일명에 접미사가 붙는다 — 예비주행이 본 데이터를 덮으면 안 된다.
    # ★루프 주기 — Pi 는 1000Hz 를 못 지킨다 (2026-08-12 실측 실효 805Hz).
    #   처프 최고 1.85Hz 라 500Hz 면 270배로 충분하다. 마찰시험이 500Hz 로 잘 돌았다.
    ap.add_argument("--rate", type=float, default=None,
                    help="루프 주기[Hz] (기본 spec.shm.rate_hz). Pi 는 500 권장")
    ap.add_argument("--f-scale", type=float, default=1.0,
                    help="f_end 배율. 첫 주행은 0.4~0.6 으로 예비주행할 것")
    ap.add_argument("--dry", action="store_true",
                    help="하드웨어 없이 궤적만 설계·검사(상관·한계·충돌 여유)")
    a = ap.parse_args()

    spec = at.load_spec(a.spec)
    cfg_all = yaml.safe_load(open(a.config, encoding="utf-8"))
    jm = JointMap(cfg_all)
    mc = spec.get("pace_multi", {})
    T = float(a.T or mc.get("duration_s", 30.0))
    ramp = float(mc.get("ramp_s", 2.0))
    rate = float(a.rate or spec["shm"]["rate_hz"])
    n = jm.n_leg

    if a.f_scale != 1.0:
        mc = {kk: (list(vv) if isinstance(vv, list) else vv) for kk, vv in mc.items()}
        mc["f_end_hz"] = [float(v) * a.f_scale for v in mc["f_end_hz"]]
        print(f"  ★f_end ×{a.f_scale:g} — 예비주행. f_start 는 그대로다"
              f"(도달 상한만 낮춘다). f_end = "
              + " ".join(f"{v:.2f}" for v in mc["f_end_hz"]) + " Hz")
    amps, f0, k, phi = chirp_bank(mc, n, T)
    home = np.array([float(x) for x in cfg_all["home"]["q_deg"]])[:n]

    # ── 설계 검사 (하드웨어 없이) ────────────────────────────────────────
    tt = np.arange(0, T, 1.0 / rate)
    mirror = mc.get("mirror") or None
    Q = np.array([apply_mirror(q_at(t, amps, f0, k, phi, ramp, T), jm.names, mirror)
                  for t in tt]) + home
    print("■ 궤적 설계 검사")
    print(f"  길이 {T:.0f}s · {rate:.0f}Hz · {len(tt)} 표본 · 램프 {ramp:.0f}s")
    print(f"  ★게인셋: {a.gains}" + ("  (검증용 — 다른 게인에서 같은 θ 가 나오는지 본다)"
                                        if a.gains == "validate" else ""))
    print(f"  ★좌우 대칭(net wrench 상쇄, PACE §3.2.2): "
          f"{mirror or '없음 — 반력이 크레인을 흔든다'}")
    _kp = at._gain(mc.get("kp" if a.gains == "id" else "kp_validate", {}))
    _kd = at._gain(mc.get("kd" if a.gains == "id" else "kd_validate", {}))
    if isinstance(_kp, dict):
        print(f"  ★시험 전용 게인 kp " + " ".join(f"ch{c}:{_kp[c]:g}" for c in sorted(_kp)))
        print(f"                 kd " + " ".join(f"ch{c}:{_kd[c]:g}" for c in sorted(_kd)))
    print(f"  ★시험 전용 한계  τ_trip {mc.get('tau_trip_nm', spec['safety']['tau_trip_nm'])} Nm"
          f" · err_max {mc.get('err_max_deg', spec['safety']['err_max_deg'])}°"
          f"   (배포값 {spec['safety']['tau_trip_nm']} / {spec['safety']['err_max_deg']})")
    print(f"  {'축':<10}{'진폭[°]':>8}{'f0':>6}{'f1':>6}{'위상[°]':>8}{'범위[°]':>18}{'한계':>18}")
    bad = []
    for i in range(n):
        lo, hi = Q[:, i].min(), Q[:, i].max()
        jl, jh = jm.jog_min[i], jm.jog_max[i]
        out = lo < jl or hi > jh
        if out:
            bad.append(jm.names[i])
        print(f"  {jm.names[i]:<10}{amps[i]:>8.1f}{f0[i]:>6.2f}{f0[i]+k[i]*T:>6.2f}"
              f"{np.degrees(phi[i]):>8.0f}{lo:>8.1f}~{hi:<9.1f}{jl:>8.1f}~{jh:<9.1f}"
              + ("  ★밖" if out else ""))
    if bad:
        raise SystemExit(f"✗ 궤적이 jog 한계 밖이다: {bad} — spec.pace_multi.amp_deg 를 줄일 것")

    # ★속도·범위는 **채널각**으로 봐야 한다 (2026-08-12, 사용자 지적).
    #   종전엔 모델각(Q)으로만 쟀다. 그런데 드라이버가 보는 것도, vel_trip 이 걸리는
    #   것도 **채널각**이다. foot 은 calf 커플링 때문에 둘이 크게 다르다:
    #       q_ch_foot = (q_foot + q_calf)·s·k   [couple_coef +1 · gear_k 1.2]
    #       모델 ±13° → **채널 ±34.8°** (2.7배) · 속도도 2.4배
    #   전체 처프(--f-scale 1.0)에서 foot 채널속도가 **277 dps** 로 상한 200 을 넘는다.
    #   모델각으로는 115 dps 라 "여유 27%" 로 보였다 — **없던 여유다.**
    #   ⚠커플링 자체는 변환(q_joint_to_ch)에 제대로 들어가 있다. 빠진 건 **검사**다.
    #     같은 양을 두 공간에서 다루면서 한쪽만 본, 오늘 반복된 그 부류다.
    Qc = np.array([jm.q_joint_to_ch(q) for q in Q])
    vmax_j = np.abs(np.diff(Q, axis=0)).max(axis=0) * rate
    vmax = np.abs(np.diff(Qc, axis=0)).max(axis=0) * rate      # ★채널 기준
    vlim = float(spec["safety"]["vel_trip_dps"])
    print("\n  채널각 범위·속도 (커플링 반영 — 드라이버가 보는 값)")
    for i, nm in enumerate(jm.names):
        _m = "  ★상한 초과" if vmax[i] > vlim else ""
        print(f"    {nm:<10}[{Qc[:, i].min():+7.1f}, {Qc[:, i].max():+7.1f}]°"
              f"  {vmax[i]:>5.0f} dps  (모델각 {vmax_j[i]:>4.0f}){_m}")
    print(f"\n  최대 |q̇| (명령) " + " ".join(f"{v:.0f}" for v in vmax)
          + f" dps   상한 {vlim:.0f} · 여유 {(1 - vmax.max()/vlim)*100:.0f}%")
    if vmax.max() > vlim * 0.85:
        print(f"    ⚠여유가 {(1-vmax.max()/vlim)*100:.0f}% 뿐이다 — 실측은 명령보다 크다"
              f"(추종 지연·오버슛). amp_deg 나 f_end_hz 를 줄이거나 vel_trip 을 올릴 것.")

    # ★상관 — 이게 크면 파라미터를 못 가른다. 설계 단계에서 잡아야 한다.
    #   ⚠mirror 를 켜면 좌우가 **의도적으로** 종속(r=1)이 된다 — 그건 결함이 아니다.
    #     그래서 좌우 짝은 검사에서 빼고 **한쪽 다리 안에서만** 본다.
    #     (좌우를 따로 식별하려면 mirror 를 끄고 수집해야 한다. apply_mirror 주석 참조)
    m_ = n // 2 if mirror else n
    C = np.corrcoef(Q[:, :m_].T)
    off = np.abs(C - np.eye(m_))
    scope = "한쪽 다리 안" if mirror else "전 축"
    print(f"\n  축간 상관 최대 {off.max():.3f} ({scope} · 0.5 넘으면 분리 나쁨)")
    i, j = np.unravel_index(off.argmax(), off.shape)
    print(f"    최대쌍 {jm.names[i]} ↔ {jm.names[j]}")
    if off.max() > 0.5:
        print("    ⚠위상/주파수를 더 벌릴 것")
    if mirror:
        print(f"    ℹ좌우는 mirror='{mirror}' 로 종속(r=1)이다 — 의도된 것이다."
              f" 파라미터가 kind별 공유라 손해가 없다.")
    # ★진짜 기준은 축 상관이 아니라 **파라미터 감도**다 — design_excitation.py 를 볼 것.
    print(f"    ★이건 대리지표다. 파라미터 기준 식별가능성은:"
          f" ~/.venv-mujoco/bin/python design_excitation.py")

    if a.dry:
        print("\n--dry — 하드웨어 미접촉. 여기까지가 설계 검사다.")
        return 0

    # ── 실기 수집 ──────────────────────────────────────────────────────
    sf, g = spec["safety"], spec["gains"]
    box = at._mech_limit_box()
    ch_all = sorted(box)
    # ★기본 lim 은 **가장 좁은** 축 기준으로 잡는다 — 채널별 탐색범위를 못 찾은 채널이
    #   느슨한 값으로 새는 것을 막는다. 실제 판정은 아래 hw.lim_ch(채널별)가 한다.
    #   ⚠종전엔 **합집합**(min of mins, max of maxes)이었다: hip 실한계가 ±14.9° 인데
    #     탐색범위가 ±176° 가 되어 **위치 트립이 사실상 없었다.** 30초를 도는 시험에서
    #     어떤 축이 폭주해도 안 걸린다.
    lim = Limits(q_min=max(box[c][0] for c in ch_all), q_max=min(box[c][1] for c in ch_all),
                 tau_trip=float(mc.get("tau_trip_nm", sf["tau_trip_nm"])),
                 tau_trip_ms=float(sf["tau_trip_ms"]),
                 vel_trip=float(sf["vel_trip_dps"]),
                 err_max=float(mc.get("err_max_deg", sf["err_max_deg"])),
                 stale_ms=float(sf["stale_ms"]),
                 kp_max=float(g["kp_max"]), kd_max=float(g["kd_max"]))
    # ★게인·한계는 **이 시험 전용**을 쓴다(spec.pace_multi). 배포값을 쓰면
    #   ① kp 가 높아 궤적이 q_cmd 에 붙고 파라미터 정보가 사라지고
    #   ② 중력이 τ_trip 을 먹어 hip 동적여유가 0.35Nm 밖에 안 남는다.
    _gk = "kp" if a.gains == "id" else "kp_validate"
    _gd = "kd" if a.gains == "id" else "kd_validate"
    if _gk not in mc:
        raise SystemExit(f"✗ spec.pace_multi.{_gk} 가 없다")
    kp = at._gain(mc[_gk]); kd = at._gain(mc[_gd])
    # ★홈복귀는 **배포 게인**으로 한다 (2026-08-12 실기). 종전엔 hold_kp 에 처프 게인을
    #   그대로 넘겼는데, goto_home 이 hold_kp 를 쓰므로 **calf 를 kp 25 로 78° 움직였다**
    #   (배포는 80). 그 결과 오차 35.02° · 명령토크 15.28Nm — τ_trip 16 에 육박했고,
    #   그 상태에서 드라이버가 명령을 안 받는 상태로 래치됐다(실기 1회).
    #   ⚠처프가 게인을 낮추는 건 **식별을 위해서**다. 게인이 높으면 궤적이 q_cmd 에
    #     붙어 파라미터 정보가 사라진다. 그건 **가진 구간에만** 필요한 이야기다 —
    #     홈복귀는 그냥 자세를 옮기는 동작이라 배포 게인이 맞다.
    #   ⚠가진 루프는 _raw_write_all(q_ch, kp, kd, box) 로 **처프 게인을 직접 넘긴다** —
    #     hold_kp 를 안 쓰므로 이 변경이 식별 조건을 건드리지 않는다.
    _kp_home = at._gain(sf.get("hold_kp", 40.0))
    _kd_home = at._gain(sf.get("hold_kd", 2.0))
    print("  ★홈복귀 게인 = 배포값 " + " ".join(
        f"ch{c}:{_kp_home[c]:g}" for c in sorted(_kp_home)) if isinstance(_kp_home, dict)
        else f"  ★홈복귀 게인 = {_kp_home}")
    at.preflight(spec)

    with Hardware(spec["shm"]["lib"], spec["shm"]["n_channel"], rate, lim,
                  int(spec["shm"]["recv_wait_ms"]), float(g["enable_ramp_s"]),
                  hold_channels=ch_all, hold_kp=_kp_home, hold_kd=_kd_home) as hw:
        hw.publish_fn = lambda q_ch, rpy, on, raw=None: publish_state(
            "pace:multichirp", jm.ch_to_q_joint(np.asarray(q_ch, float)),
            np.asarray(rpy, float), rate, on, "pace",
            extra=leg_extra(jm, **(raw or {})))
        # ★오늘(2026-08-12) 만든 안전장치를 **여기에도** 건다. 이 시험이 가장 오래 돈다
        #   (30초 연속 가진). 정작 여기에 안 걸려 있었다.
        #   ㆍlim_ch  : 채널별 위치·토크 한계 (위 합집합 사고 참조)
        #   ㆍgrav_fn : 스톨 감지용 중력 조회 — 없으면 그 검사가 꺼진다
        _tt = float(mc.get("tau_trip_nm", sf["tau_trip_nm"]))
        _tb = {int(c): float(v) for c, v in (sf.get("tau_trip_by_ch") or {}).items()}
        hw.lim_ch = {c: replace(lim, q_min=box[c][0], q_max=box[c][1],
                                tau_trip=_tb.get(c, _tt)) for c in ch_all}
        _gt = spec["torque_mode"].get("tau_grav_table") or {}
        hw.grav_fn = (lambda c, q, _t=_gt:
                      float(np.interp(q, _t[c]["q_ch"], _t[c]["tau"])) if c in _t else 0.0)
        print(f"  ★채널별 한계 적용 — 위치는 축마다, τ_trip "
              + " ".join(f"ch{c}:{hw.lim_ch[c].tau_trip:g}" for c in ch_all))
        hw.arm(ch_all[0], kp[ch_all[0]], kd[ch_all[0]])
        goto_home(hw, jm, make_homer(jm, cfg_all, hw.dt), cfg_all, q_box=box,
                  log=lambda m: print(f"  [multichirp]{m}"))
        print(f"\n  가진 시작 — {T:.0f}s. Ctrl+C 로 언제든 중단(limp).")

        # ★배열을 **미리 잡는다**. 여유 20% (루프가 밀리면 표본이 적어질 뿐 넘지 않는다)
        _N = int(T * rate * 1.2) + 16
        T_ = np.zeros(_N); QC = np.zeros((_N, n)); Qm = np.zeros((_N, n))
        DQ = np.zeros((_N, n)); TAU = np.zeros((_N, n)); CUR = np.zeros((_N, n))
        STT = np.zeros((_N, n), int); CONN = np.zeros((_N, n), int)
        _i = 0
        _ipk = [0.0, 0.0]                 # [최대 총전류, 그 시각]
        _over, _lagmax = 0, 0.0           # 밀린 틱 수 · 최대 지연[ms]
        # ★가진 동안 GC 를 끈다 (2026-08-12). 실기 500Hz 에서 **330.7ms** 정지가 찍혔고
        #   그동안 드라이버에 명령이 안 나갔다 — 워치독 래치오프의 유력 원인이다.
        #   ⚠참조순환이 없으면 참조계수만으로 다 회수된다. 위에서 배열을 미리 잡아
        #     객체가 안 쌓이므로 30초 동안 메모리도 안 는다. 끝나면 반드시 되돌린다.
        import gc as _gc
        _gc_was = _gc.isenabled()
        _gc.disable()
        t0 = time.perf_counter()
        while True:
            t = time.perf_counter() - t0
            if t >= T:
                break
            q_leg = home + apply_mirror(q_at(t, amps, f0, k, phi, ramp, T), jm.names, mirror)
            q_ch = jm.q_joint_to_ch(q_leg)
            hw._raw_write_all(q_ch, kp, kd, box)
            # ★SHM 을 **한 번만** 읽는다 (2026-08-12). 종전엔 `[hw.read(c) for c in
            #   range(hw.n)]` 로 **채널 수만큼(10회)** 읽었다. read() 는 매번 SHM 전체를
            #   복사하고 신선도용 tobytes() 를 4개 만든다 — 1000Hz × 10 = **초당 1만 회**,
            #   할당은 4만 회다. Pi 4 + 파이썬에서 이걸 1ms 안에 못 끝낸다.
            #   ⚠루프가 밀리면 드라이버에 **명령이 안 간다.** 배포 제어기(500Hz)조차
            #     "루프가 36ms 밀렸다" 를 찍는 기기다. 명령 공백이 길어지면 드라이버
            #     워치독이 래치오프한다 — 다축에서만, 매번 다른 축이, 가벼운 부하에서
            #     죽는 관측과 정확히 맞는다(solo 는 1축만 쓰고 500Hz 라 여유가 있었다).
            #   ⇒ read() 한 번이면 self._q/_dq/_tau/_cur 가 **전 채널** 채워진다.
            hw.read(ch_all[0])
            q_m = np.array(hw._q, float)
            for c in ch_all:
                hw._check(c, q_m[c], float(hw._dq[c]), float(hw._tau[c]),
                          float(hw._q_cmd[c]))
            # ★파워단 사망·스톨은 check_hold 에만 있다 — _check 로는 못 잡는다.
            #   오늘 다섯 번 겪은 그 고장을 30초 가진 중에 놓치면 안 된다.
            hw.check_hold()
            # ★리스트 append 대신 **미리 잡은 배열**에 인덱스로 채운다 (2026-08-12).
            #   종전엔 틱마다 numpy 배열 ~6개를 만들어 리스트에 쌓았다. 30초면
            #   **15만 객체**가 살아남고, 파이썬 GC 가 gen2 를 돌 때 그 전부를 훑는다.
            #   실기 500Hz 주행에서 **330.7ms** 정지가 찍혔다(이 기기 유휴 실측 32ms —
            #   로봇이 도는 중이면 그 자릿수가 맞다). 그 동안 드라이버에 명령이 안 간다.
            #   ⇒ 배열은 고정, 객체는 안 늘어난다. gc.disable() 과 짝이다.
            #   ⚠임시 배열은 여전히 만들어지지만 **즉시 복사되고 버려진다** — 참조가
            #     안 남으니 GC 가 훑을 대상이 안 늘어난다. 그게 요점이다.
            T_[_i] = t
            QC[_i] = q_leg
            Qm[_i] = jm.ch_to_q_joint(q_m)
            # ★**단위 변환을 빠뜨렸었다** (2026-08-12). ch_to_dq_ctrl 은 마지막에 D2R 을
            #   곱해 **rad/s** 를 낸다. 그런데 npz 의 q 는 **deg** 이고, pace_cmaes 의
            #   rollout 은 `d.qvel = dq_real * DEG` 로 **deg/s 인 줄 알고 또 변환**한다.
            #   ⇒ 시뮬이 창(0.5s)마다 실측속도의 **1/57.3** 로 재초기화됐다 — 사실상
            #     정지에서 출발한 셈이고, 창 앞부분이 계통적으로 뒤처진다.
            #   ★hw_interface.py 는 같은 함수를 쓰면서 `* R2D` 로 되돌린다 — 그 주석에
            #     "rad/s 로 주므로 deg/s 로 되돌린다" 고 적혀 있다. **여기만 빠졌다.**
            #   실측 확인: dq / q의수치미분 기울기가 8축 전부 0.01734~0.01745 (=1/57.2958).
            DQ[_i] = jm.ch_to_dq_ctrl(np.asarray(hw._dq, float)) / DEG   # → 모델각 deg/s
            TAU[_i] = np.asarray(hw._tau, float)[jm.ch]
            # ★전류를 남기고 **총합을 화면에 띄운다** (2026-08-12).
            #   다축에서 스톨도 과토크도 없이 축이 죽는다 — 매번 다른 축이, τ_trip 의
            #   절반도 안 되는 부하에서. solo 와 다른 건 **8축 동시 전류**뿐이다.
            #   공급전압 새그 → 저전압 보호 → 래치오프 가설을 이걸로 검증한다.
            #   ⚠SHM 에 버스 전압은 안 온다. 전류 총합이 유일한 대리지표다.
            CUR[_i] = np.asarray(hw._cur, float)[jm.ch]
            STT[_i] = np.asarray(hw._stt, int)[jm.ch]
            CONN[_i] = np.asarray(hw._conn, int)[jm.ch]
            _isum = float(np.abs(CUR[_i]).sum())
            _i += 1
            if _isum > _ipk[0]:
                _ipk[0], _ipk[1] = _isum, t
            if int(t) != int(t - hw.dt) and int(t) % 5 == 0:
                _st = np.array(hw._stt[:hw.n], int)
                print(f"    {t:5.1f}/{T:.0f}s  f≈{f0[0]+k[0]*t:.2f}Hz  "
                      f"토크합 {_isum:5.1f}Nm (최대 {_ipk[0]:5.1f} @{_ipk[1]:.1f}s)  "
                      f"밀림 {_over}틱(최대 {_lagmax:.1f}ms)  "
                      f"stt " + " ".join(str(v) for v in _st)
                      + ("" if len(set(_st.tolist())) == 1 else "  ★축마다 다르다"))
            nxt = t0 + _i * hw.dt
            slp = nxt - time.perf_counter()
            if slp > 0:
                time.sleep(slp)
            else:
                # ★밀린 틱을 센다. 명령 공백이 드라이버 워치독을 건드릴 수 있다.
                _over += 1
                _lagmax = max(_lagmax, -slp * 1e3)

        if _gc_was:
            _gc.enable()
        T_ = T_[:_i]; QC = QC[:_i]; Qm = Qm[:_i]; DQ = DQ[:_i]
        TAU = TAU[:_i]; CUR = CUR[:_i]; STT = STT[:_i]; CONN = CONN[:_i]

        # ★수집이 끝나면 **먼저 저장하고** 그 다음에 뒷정리한다 (2026-08-12 실기).
        #   종전엔 여기서 goto_home 을 먼저 했는데 그게 오차 9.40° 로 실패하자 예외가
        #   올라가 **np.savez 까지 못 갔다 — 30초를 멀쩡히 수집하고도 통째로 버렸다.**
        #   수집이 끝난 시점에서 홈복귀는 **뒷정리**일 뿐이다. 뒷정리 실패로 결과를
        #   버리면 안 된다.
        #   ⚠저장은 with 블록 **안**이어야 한다 — 밖으로 빼면 __exit__ 이 limp 한 뒤다.
        # ★dt 는 **실측 중앙값**을 쓴다 (2026-08-12). 종전엔 1/rate 를 그대로 적었다.
        #   pace_cmaes 는 이 값을 `m.opt.timestep` 과 지연 샘플수에 **그대로 쓴다** —
        #   실제와 다르면 시뮬이 다른 속도로 돌아 식별이 통째로 틀어진다.
        #   실기 첫 수집: 적힌 1.000ms vs 실제 **1.142ms** (14% 차이).
        #   ROTOR_I 는 가속(1/dt²)에 걸리므로 그 오차가 **30%** 로 증폭된다.
        _dg = np.diff(T_)
        _dt_real = float(np.median(_dg)) if len(_dg) else 1.0 / rate
        _dev = abs(_dt_real - 1.0 / rate) / (1.0 / rate) * 100
        if _dev > 5.0:
            print(f"\n  ⚠루프가 목표 주기를 못 지켰다 — 목표 {1000/rate:.2f}ms vs "
                  f"실측 {_dt_real*1e3:.3f}ms ({_dev:.0f}% 차이, 실효 {1/_dt_real:.0f}Hz)")
            print(f"    긴 공백: 2ms 초과 {int((_dg>0.002).sum())}회 · "
                  f"5ms 초과 {int((_dg>0.005).sum())}회 · 최대 {_dg.max()*1e3:.1f}ms")
            print(f"    **명령 공백은 드라이버 워치독을 건드릴 수 있다** — --rate 로 낮출 것.")
            print(f"    dt 는 실측값 {_dt_real*1e3:.3f}ms 로 저장한다(식별이 틀어지지 않게).")
        os.makedirs(a.out, exist_ok=True)
        _sfx = "" if a.f_scale == 1.0 else f"_f{a.f_scale:g}"
        path = os.path.join(a.out, (f"pace_multichirp{_sfx}.npz" if a.gains == "id"
                                    else f"pace_multichirp_val{_sfx}.npz"))
        # ★관절공간 게인으로 저장한다 — 시뮬은 모델각으로 돌기 때문이다.
        #   τ_joint = kp_ch·k²·Δq_joint  (부호는 토크에서도 같이 뒤집혀 상쇄된다)
        kp_j = np.array([kp[c] * jm.k[i] ** 2 for i, c in enumerate(jm.ch)])
        kd_j = np.array([kd[c] * jm.k[i] ** 2 for i, c in enumerate(jm.ch)])
        # ⚠이 값이 CMA-ES 롤아웃의 제어법칙이 된다. 수집 때 쓴 게인과 **반드시 같아야** 한다.
        np.savez(path, t=T_, q=Qm, q_cmd=QC,
                 dq=DQ, tau_ch=TAU, cur_ch=CUR,
                 stt=STT, conn=CONN,
                 kp_joint=kp_j, kd_joint=kd_j, gear_k=jm.k, gear_n=np.array(
                     [float([x for x in spec["joints"] if x["ch"] == c][0]["gear"])
                      for c in jm.ch]),
                 names=np.array(jm.names), home=home, dt=_dt_real,
                 amp=amps, f0=f0, f1=f0 + k * T, phi=phi, corr_max=off.max(),
                 mirror=str(mirror))
        print(f"\n  ✓ 저장: {path}  ({_i} 표본)")
        _pct = 100.0 * _over / max(_i, 1)
        print(f"    루프 밀림 {_over}/{_i}틱 ({_pct:.1f}%) · 최대 {_lagmax:.1f}ms"
              + ("   ✓ 실시간 유지" if _pct < 1.0 else
                 "   ★밀린다 — rate_hz 를 낮출 것(명령 공백이 드라이버 워치독을 건드린다)"))
        print(f"    최대 토크합 {_ipk[0]:.1f} Nm @ {_ipk[1]:.1f}s"
              f"  — 축별 최대 "
              + " ".join(f"{v:.1f}" for v in np.abs(TAU).max(axis=0)) + " Nm")

        try:
            goto_home(hw, jm, make_homer(jm, cfg_all, hw.dt), cfg_all, q_box=box,
                      log=lambda m: print(f"  [multichirp]{m}"))
        except Exception as e:
            print(f"\n  ⚠수집 후 홈복귀 실패({type(e).__name__}: {e})")
            print("    **데이터는 이미 저장됐다** — 자세만 정리하면 된다.")

    print(f"    다음: ~/.venv-mujoco/bin/python pace_cmaes.py {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
