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

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "tests"), os.path.join(EMB, "interface")]

from hwio import DEG, Hardware, Limits, SafetyAbort   # noqa: E402
from homing import goto_home, make_homer             # noqa: E402
from joint_map import JointMap                       # noqa: E402
from state_pub import publish_state                  # noqa: E402
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


def q_at(t, amps, f0, k, phi, ramp_s):
    s = min(t / ramp_s, 1.0) if ramp_s > 0 else 1.0     # 진폭 램프(계단 금지)
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
    ap.add_argument("--dry", action="store_true",
                    help="하드웨어 없이 궤적만 설계·검사(상관·한계·충돌 여유)")
    a = ap.parse_args()

    spec = at.load_spec(a.spec)
    cfg_all = yaml.safe_load(open(a.config, encoding="utf-8"))
    jm = JointMap(cfg_all)
    mc = spec.get("pace_multi", {})
    T = float(a.T or mc.get("duration_s", 30.0))
    ramp = float(mc.get("ramp_s", 2.0))
    rate = float(spec["shm"]["rate_hz"])
    n = jm.n_leg

    amps, f0, k, phi = chirp_bank(mc, n, T)
    home = np.array([float(x) for x in cfg_all["home"]["q_deg"]])[:n]

    # ── 설계 검사 (하드웨어 없이) ────────────────────────────────────────
    tt = np.arange(0, T, 1.0 / rate)
    mirror = mc.get("mirror") or None
    Q = np.array([apply_mirror(q_at(t, amps, f0, k, phi, ramp), jm.names, mirror)
                  for t in tt]) + home
    print("■ 궤적 설계 검사")
    print(f"  길이 {T:.0f}s · {rate:.0f}Hz · {len(tt)} 표본 · 램프 {ramp:.0f}s")
    print(f"  ★좌우 대칭(net wrench 상쇄, PACE §3.2.2): "
          f"{mirror or '없음 — 반력이 크레인을 흔든다'}")
    _kp = at._gain(mc.get("kp", {})); _kd = at._gain(mc.get("kd", {}))
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
    lim = Limits(q_min=min(box[c][0] for c in ch_all), q_max=max(box[c][1] for c in ch_all),
                 tau_trip=float(mc.get("tau_trip_nm", sf["tau_trip_nm"])),
                 tau_trip_ms=float(sf["tau_trip_ms"]),
                 vel_trip=float(sf["vel_trip_dps"]),
                 err_max=float(mc.get("err_max_deg", sf["err_max_deg"])),
                 stale_ms=float(sf["stale_ms"]),
                 kp_max=float(g["kp_max"]), kd_max=float(g["kd_max"]))
    # ★게인·한계는 **이 시험 전용**을 쓴다(spec.pace_multi). 배포값을 쓰면
    #   ① kp 가 높아 궤적이 q_cmd 에 붙고 파라미터 정보가 사라지고
    #   ② 중력이 τ_trip 을 먹어 hip 동적여유가 0.35Nm 밖에 안 남는다.
    kp = at._gain(mc.get("kp", sf.get("hold_kp", 40.0)))
    kd = at._gain(mc.get("kd", sf.get("hold_kd", 2.0)))
    at.preflight(spec)

    with Hardware(spec["shm"]["lib"], spec["shm"]["n_channel"], rate, lim,
                  int(spec["shm"]["recv_wait_ms"]), float(g["enable_ramp_s"]),
                  hold_channels=ch_all, hold_kp=kp, hold_kd=kd) as hw:
        hw.publish_fn = lambda q_ch, rpy, on: publish_state(
            "pace:multichirp", jm.ch_to_q_joint(np.asarray(q_ch, float)),
            np.asarray(rpy, float), rate, on, "pace")
        hw.arm(ch_all[0], kp[ch_all[0]], kd[ch_all[0]])
        goto_home(hw, jm, make_homer(jm, cfg_all, hw.dt), cfg_all, q_box=box,
                  log=lambda m: print(f"  [multichirp]{m}"))
        print(f"\n  가진 시작 — {T:.0f}s. Ctrl+C 로 언제든 중단(limp).")

        T_, Qm, DQ, TAU, QC = [], [], [], [], []
        t0 = time.perf_counter()
        while True:
            t = time.perf_counter() - t0
            if t >= T:
                break
            q_leg = home + apply_mirror(q_at(t, amps, f0, k, phi, ramp), jm.names, mirror)
            q_ch = jm.q_joint_to_ch(q_leg)
            hw._raw_write_all(q_ch, kp, kd, box)
            q_m = np.array([hw.read(c)[0] for c in range(hw.n)])
            for c in ch_all:
                hw._check(c, q_m[c], float(hw._dq[c]), float(hw._tau[c]),
                          float(hw._q_cmd[c]))
            T_.append(t); QC.append(q_leg)
            Qm.append(jm.ch_to_q_joint(q_m))
            DQ.append(jm.ch_to_dq_ctrl(np.array(hw._dq, float)))   # 채널 dps → 모델각 dps
            TAU.append(np.array(hw._tau, float)[jm.ch])
            if int(t) != int(t - hw.dt) and int(t) % 5 == 0:
                print(f"    {t:5.1f}/{T:.0f}s  f≈{f0[0]+k[0]*t:.2f}Hz")
            nxt = t0 + len(T_) * hw.dt
            slp = nxt - time.perf_counter()
            if slp > 0:
                time.sleep(slp)

        goto_home(hw, jm, make_homer(jm, cfg_all, hw.dt), cfg_all, q_box=box,
                  log=lambda m: print(f"  [multichirp]{m}"))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "pace_multichirp.npz")
    # ★관절공간 게인으로 저장한다 — 시뮬은 모델각으로 돌기 때문이다.
    #   τ_joint = kp_ch·k²·Δq_joint  (부호는 토크에서도 같이 뒤집혀 상쇄된다)
    kp_j = np.array([kp[c] * jm.k[i] ** 2 for i, c in enumerate(jm.ch)])
    kd_j = np.array([kd[c] * jm.k[i] ** 2 for i, c in enumerate(jm.ch)])
    # ⚠이 값이 CMA-ES 롤아웃의 제어법칙이 된다. 수집 때 쓴 게인과 **반드시 같아야** 한다.
    np.savez(path, t=np.array(T_), q=np.array(Qm), q_cmd=np.array(QC),
             dq=np.array(DQ), tau_ch=np.array(TAU),
             kp_joint=kp_j, kd_joint=kd_j, gear_k=jm.k, gear_n=np.array(
                 [float([x for x in spec["joints"] if x["ch"] == c][0]["gear"]) for c in jm.ch]),
             names=np.array(jm.names), home=home, dt=1.0 / rate,
             amp=amps, f0=f0, f1=f0 + k * T, phi=phi, corr_max=off.max(),
             mirror=str(mirror))
    print(f"\n  ✓ 저장: {path}  ({len(T_)} 표본)")
    print(f"    다음: ~/.venv-mujoco/bin/python pace_cmaes.py {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
