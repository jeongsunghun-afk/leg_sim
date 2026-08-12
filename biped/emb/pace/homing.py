"""homing.py — PACE 하니스의 HOME 복귀. **GUI 와 같은 궤적 구현을 쓴다.**

★왜 새로 안 짜는가 (2026-08-11, 사용자 지적)
  처음엔 hwio.goto_all 로 따로 짰다. 그건 틀린 선택이었다:
    ① **보간 공간이 다르다.** goto_all 은 채널각에서 직선보간했다. 그런데 calf→foot
       커플링이 있어서 채널각 직선은 **모델각 직선이 아니다** — 중간 자세가 뷰어에
       보이는 것과도, 기구가 의도한 것과도 다른 경로가 된다.
    ② **가속도 한계가 없었다.** control/home.py 는 T 를 5차식의 v·a 정규화 극값에서
       구한다(S_VMAX 1.875, S_AMAX 10/√3). goto_all 은 속도만 봤다.
    ③ jog 안전한계 클램프와 "잘렸다" 보고가 없었다.
    ④ 구현이 둘이면 갈라진다. 이번 세션에만 같은 환산식 복사본이 5개 나와서
       전부 틀려 있었다.
  ⇒ control/home.py 의 HomeTrajectory 를 그대로 쓰고, 여기서는 **PACE 쪽 배선**만 한다
    (측정각 읽기 → step → 채널각 변환 → 안전검사 → 도달 판정).

시퀀스는 지그 유무와 무관하게 항상 돈다 — 지그가 물려 있으면 편차가 작아 즉시 끝난다.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from dataclasses import replace

EMB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(EMB, "control"), os.path.join(EMB, "interface")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from home import HomeTrajectory          # noqa: E402  (control/home.py — GUI 와 동일)

from hwio import SafetyAbort             # noqa: E402


def make_homer(jm, cfg: dict, dt: float, q_deg=None) -> HomeTrajectory:
    """biped_emb.py 와 **같은 파라미터**로 궤적기를 만든다(app/biped_emb.py:244 참조).

    q_deg 를 주면 홈 대신 그 자세를 목표로 쓴다(속도·가속도 한계는 동일).
    시험용 홀드 자세(--pose neutral)로 옮길 때 쓴다.
    """
    h = cfg.get("home", {})
    return HomeTrajectory(jm, dt,
                          q_deg if q_deg is not None else h.get("q_deg", [0.0] * jm.n_leg),
                          float(h.get("max_speed_dps", 15.0)),
                          float(h.get("max_acc_dps2", 30.0)),
                          float(h.get("min_time_s", 0.6)))


def goto_home(hw, jm, homer: HomeTrajectory, cfg: dict, q_box=None, log=print,
              speed_dps: float | None = None, tol_deg: float = 5.0) -> float:
    """측정각에서 HOME 까지 S-curve 복귀. 소요시간[s] 반환. 도달 실패면 SafetyAbort.

    ★arm 이 되어 있어야 한다 — enable 없이 쓰면 SHM 에 kp=kd=0 이 나가
      (shm_bridge.cpp:112) 모터가 전혀 안 움직이는데 그게 조용히 지나간다.
    """
    if not hw._armed:
        raise RuntimeError("goto_home 전에 arm() 을 호출할 것 — enable 없이는 무동작이다.")
    if speed_dps is not None:                 # CLI 로 더 느리게 가고 싶을 때
        homer.vmax = max(float(speed_dps), 1e-6)

    q_ch = np.array([hw.read(c)[0] for c in range(hw.n)], float)
    q_ch0 = q_ch.copy()                # ★출발 채널각 — 상자를 여기까지 늘린다(계단 방지)
    # ★확장 상자를 **한 번만** 만들어 쓰기·검사에 **같은 것**을 쓴다.
    #   2026-08-12: _write_leg 만 확장하고 _trip_check 는 원래 상자를 쓰게 뒀더니,
    #   명령은 계단이 없는데 **검사가 트립**했다(ch2 20.34° ∉ [−145.58, 18.52]).
    #   늘어진 calf 모델각이 −60.8° 로 관절한계 −59.6° 밖이라 생긴 일이다.
    #   ⇒ 같은 값을 두 곳에서 다르게 다루면 반드시 어긋난다. 하나로 만든다.
    box_eff = None
    if q_box is not None:
        box_eff = {c: (min(lo, float(q_ch0[c])), max(hi, float(q_ch0[c])))
                   for c, (lo, hi) in q_box.items()}
    q_leg = np.asarray(jm.ch_to_q_joint(q_ch), float)      # ★측정각(모델각)에서 출발
    T = homer.start(q_leg)
    for nm, want, got in homer.clamped:
        log(f"    ⚠ home.q_deg[{nm}] {want:+.1f}° → {got:+.1f}° 로 클램프 — 이 자세로는 "
            f"홈에 도달하지 못한다(jog 안전한계).")
    dmax = float(np.max(np.abs(homer.d)))
    if T <= 0.0:
        log(f"    이미 홈 자세(최대 편차 {dmax:.3f}°) — 이동 생략")
        return 0.0
    log(f"    홈 복귀: 최대 {dmax:.1f}° · {T:.1f}s "
        f"(v≤{homer.vmax:.0f}dps, a≤{homer.amax:.0f}dps²) — GUI 와 동일 궤적")

    t_prev = time.perf_counter()
    while not homer.done:
        now = time.perf_counter()
        dt_meas, t_prev = now - t_prev, now
        # ★경과시간 기준으로 진행한다. 호출 횟수 기준이면 루프가 밀렸다 몰아 돌 때
        #   궤적이 빨리감기 되어 v/a 한계가 무의미해진다(control/home.py 주석 참조).
        q_cmd_leg = homer.step(dt_meas)
        _write_leg(hw, jm, q_cmd_leg, box_eff)
        _trip_check(hw, box_eff)
        time.sleep(hw.dt)

    t_settle = time.perf_counter()                       # 정착
    while time.perf_counter() - t_settle < 0.5:
        _write_leg(hw, jm, homer.q_cmd_leg, box_eff)
        _trip_check(hw, box_eff)
        time.sleep(hw.dt)

    q_now = np.asarray(jm.ch_to_q_joint(
        np.array([hw.read(c)[0] for c in range(hw.n)], float)), float)
    err = q_now - homer.q_home
    emax = float(np.max(np.abs(err)))
    log(f"    도착 — 최대 오차 {emax:.2f}° (모델각: "
        f"{', '.join(f'{jm.names[i]}{err[i]:+.1f}' for i in range(jm.n_leg))})")
    if emax > tol_deg:
        hw.limp()
        raise SafetyAbort(
            f"홈 복귀 실패 — 최대 오차 {emax:.2f}° > {tol_deg}°. limp 함.\n"
            f"  ({', '.join(f'{jm.names[i]}{err[i]:+.2f}' for i in range(jm.n_leg))})\n"
            f"  게인 부족·파워단 래치오프·기구 간섭(지그 포함) 중 하나다.")
    return T


def _write_leg(hw, jm, q_leg_deg, q_box) -> None:
    """모델각 → 채널각 → SHM. 변환은 **JointMap 이 전담**한다(수식 복사 금지).

    ★상자를 **현재 측정각까지 늘려서** 적용한다 — 안 그러면 계단이 나간다.
      2026-08-12 실기: 늘어진 자세의 calf 모델각이 −61.0° 인데 관절한계가 −59.6° 라
      채널상자 밖이었다 → 첫 틱에 2.1° 계단(kp 80 이면 2.9Nm).
      배포 앱(biped_emb)에서 같은 사고가 **34.8° 계단 → 426dps 폭주**로 터졌다
      (hw_interface.write_ramped 주석). 여기도 같은 구조였다.
    ⚠보호는 유지된다: 현재 자리는 허용하되 **더 바깥으로는 못 간다**. 목표(홈)는
      상자 안이므로 궤적이 상자 쪽으로만 데려간다.
    """
    q_ch = jm.q_joint_to_ch(np.asarray(q_leg_deg, float))
    hw._raw_write_all(q_ch, hw.hold_kp, hw.hold_kd, q_box)


def _trip_check(hw, q_box=None) -> None:
    """전 채널 트립 감시. ★위치한계만은 **채널별**로 본다.

    ⚠hw.lim 은 **시험축** 한계다. 그걸 전 채널에 적용하면 다른 축이 자기 범위 안인데도
      트립한다 — 2026-08-11 오프라인 테스트가 잡았다(HR_calf 76.01° 가 HL_foot 상자
      [−180, 75.97] 에 걸렸다). _raw_write_all 은 이미 채널별 상자를 쓰는데
      여기만 안 고쳐져 있었다. 종전 영점에서는 우연히 안 걸렸을 뿐이다.
    ⚠속도·토크·stale·추종오차는 축과 무관한 물리량이라 그대로 hw.lim 을 쓴다.
    """
    saved = hw.lim
    snap = []
    try:
        for c in range(hw.n):
            q, dq, tq, _ = hw.read(c)
            snap.append((c, float(hw._q_cmd[c]), q, dq, tq))
            if q_box is not None and c in q_box:
                lo, hi = q_box[c]
                hw.lim = replace(saved, q_min=lo, q_max=hi)
            else:
                hw.lim = saved
            hw._check(c, q, dq, tq, float(hw._q_cmd[c]))
    except SafetyAbort as e:
        # ★트립 순간의 **전 채널 상태**를 같이 찍는다 (2026-08-12).
        #   한 축의 값만 보면 원인을 못 가린다 — 커플링 때문에 어느 축의 지연이
        #   다른 축의 오차로 나타난다(q_ch_foot 는 calf 관절에도 의존한다).
        #   ⚠검사 도중 트립하면 뒤쪽 채널은 아직 안 읽었다. 그건 읽어서 채운다.
        for c in range(len(snap), hw.n):
            try:
                q, dq, tq, _ = hw.read(c)
                snap.append((c, float(hw._q_cmd[c]), q, dq, tq))
            except Exception:
                break
        rows = "\n".join(
            f"    ch{c}  명령 {cmd:+9.2f}  측정 {q:+9.2f}  오차 {cmd - q:+7.2f}"
            f"  속도 {dq:+8.1f}  토크 {tq:+7.3f}" for c, cmd, q, dq, tq in snap)
        raise SafetyAbort(f"{e}\n  트립 순간 전 채널(채널각):\n{rows}") from None
    finally:
        hw.lim = saved
