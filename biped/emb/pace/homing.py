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
              speed_dps: float | None = None, tol_deg: float = 5.0,
              only_ch: int | None = None, kp: float = 0.0, kd: float = 0.0) -> float:
    """측정각에서 HOME 까지 S-curve 복귀. 소요시간[s] 반환. 도달 실패면 SafetyAbort.

    ★only_ch: **그 채널만** 움직이고 나머지는 손도 대지 않는다 (2026-08-12, 사용자 요청).
      "측정축 외에 제어는 하지 마라 — 반대편은 작업자가 손으로 잡는다."
      · 명령·게인을 그 채널에만 쓴다(나머지는 kp=kd=0 그대로).
      · 트립 검사도 **그 채널만** 본다 — 손으로 잡은 축은 위치오차가 크게 나는 게 정상이고,
        그걸로 시험이 꺼지면 안 된다.
      · 도달 판정도 그 채널만. 다른 축의 모델각은 손 위치라 의미가 없다.
      ⚠하위 관절이 무여자가 되므로 I_link 의 강체가정이 깨진다. 관성·처프 시험에는
        쓰면 안 된다. **마찰·파단은 ±방향 차로 중력이 상쇄되므로** 영향이 작다.

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
        _write_leg(hw, jm, q_cmd_leg, box_eff, only_ch, kp, kd)
        _trip_check(hw, box_eff, only_ch)
        time.sleep(hw.dt)

    t_settle = time.perf_counter()                       # 정착
    while time.perf_counter() - t_settle < 0.5:
        _write_leg(hw, jm, homer.q_cmd_leg, box_eff, only_ch, kp, kd)
        _trip_check(hw, box_eff, only_ch)
        time.sleep(hw.dt)

    q_now = np.asarray(jm.ch_to_q_joint(
        np.array([hw.read(c)[0] for c in range(hw.n)], float)), float)
    err = q_now - homer.q_home
    # ★only_ch 면 그 축만 본다 — 손으로 잡은 축의 모델각은 의미가 없다.
    emax = float(abs(err[int(np.where(np.asarray(jm.ch) == only_ch)[0][0])])) \
        if only_ch is not None \
        else float(np.max(np.abs(err)))
    log(f"    도착 — 최대 오차 {emax:.2f}° (모델각: "
        f"{', '.join(f'{jm.names[i]}{err[i]:+.1f}' for i in range(jm.n_leg))})")
    if emax > tol_deg and only_ch is not None:
        # ★--solo 에서는 목표에 **원리적으로 못 갈 수 있다** (2026-08-12 실기).
        #   커플링 축이 그렇다: q_ch_foot = (q_foot + q_calf)·s·k 이므로 calf 가 −55° 로
        #   늘어져 있으면 foot **모델각** 0 을 만들려면 채널각 +66 이 필요한데
        #   상자가 [−27.8, +48.0] 이라 닿지 못한다. ch3 이 55.16° 오차로 중단했다.
        #   ⇒ solo 는 **자세를 맞추는 게 목적이 아니다** — 마찰 시험은 그 자리를 중심으로
        #     ±stroke 만 움직이면 되고, 중심이 어디든 ±방향 차로 중력이 상쇄된다.
        #   ⇒ 경고만 하고 진행한다. 대신 **왜 못 갔는지**를 숫자로 남긴다.
        cmd_ch = float(hw._q_cmd[only_ch])
        at_edge = (q_box is not None and only_ch in q_box
                   and (abs(cmd_ch - q_box[only_ch][0]) < 0.05
                        or abs(cmd_ch - q_box[only_ch][1]) < 0.05))
        log(f"    ⚠목표 미달 {emax:.2f}° > {tol_deg}° — **진행한다**(solo 는 자세가 목적이 아니다).")
        if at_edge:
            lo, hi = q_box[only_ch]
            log(f"      명령이 상자 끝({cmd_ch:+.2f} ∈ [{lo:+.2f}, {hi:+.2f}])에 붙었다 —"
                f" **도달 불가능한 목표**다. 커플링 축이면 원천축을 먼저 0 으로 보낼 것.")
        else:
            log(f"      명령 {cmd_ch:+.2f} 는 상자 안이다 — 축이 못 따라간 것이다"
                f"(마찰·중력·간섭). 시험은 이 자리를 중심으로 돈다.")
        return T
    if emax > tol_deg:
        # ★limp 가 아니라 **제자리 정지**. limp 하면 매단 다리가 떨어져 좌우 발이
        #   겹친 자세로 착지하고, 다음 실행이 거기서 시작해 또 트립한다
        #   (실측: 늘어진 시작자세에서 발 간격 **−27mm**). 위치는 알고 있으니 잡는다.
        hw.safe_hold()
        raise SafetyAbort(
            f"홈 복귀 실패 — 최대 오차 {emax:.2f}° > {tol_deg}°. limp 함.\n"
            f"  ({', '.join(f'{jm.names[i]}{err[i]:+.2f}' for i in range(jm.n_leg))})\n"
            f"  게인 부족·파워단 래치오프·기구 간섭(지그 포함) 중 하나다.")
    return T


def _write_leg(hw, jm, q_leg_deg, q_box, only_ch=None, kp=0.0, kd=0.0) -> None:
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
    if only_ch is not None:
        # ★그 채널만 쓴다. hold_ch 가 비어 있으므로 나머지는 kp=kd=0 으로 나간다.
        v = float(q_ch[only_ch])
        if q_box is not None and only_ch in q_box:
            lo, hi = q_box[only_ch]
            v = min(max(v, lo), hi)
        hw._raw_write(only_ch, v, kp, kd)
        return
    hw._raw_write_all(q_ch, hw.hold_kp, hw.hold_kd, q_box)


def _trip_check(hw, q_box=None, only_ch=None) -> None:
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
        for c in (range(hw.n) if only_ch is None else (only_ch,)):
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
        # ★먼저 **전원이 들어와 있는지** 본다 (2026-08-12).
        #   2026-08-12 실기: 전 축이 무여자인데 "ch2 추종오차 12.02 — 막힘·게인부족·
        #   기계간섭 의심" 으로 떴다. 원인을 셋으로 흩고, 심지어 **엉뚱한 축**을 지목한다.
        #   판별은 간단하다: 유의미한 오차가 있는 축들이 **전부** 무토크면 전원 문제다.
        #     실측 — ch2 오차 12.02° kp80 → 기대 16.78Nm · 보고 0.070Nm (비 0.004)
        #            6축 전부 비 0.002~0.013
        #   한 축만 그러면 그 드라이버 사망, 전부면 전원/enable 이다. 구분해서 말한다.
        try:
            live = dead = 0
            for c, cmd, q, dq, tq in snap:
                kp_c, _ = hw._hold_gain_of(c) if c in hw.hold_ch else (0.0, 0.0)
                exp = kp_c * abs(cmd - q) * np.pi / 180.0
                if exp > 0.5:
                    dead += abs(tq) < exp * 0.15
                    live += 1
            if live >= 3 and dead == live:
                hw.limp()
                raise SafetyAbort(
                    f"**전 축 무여자** — 유의미한 오차가 있는 {live}축이 전부 무토크다.\n"
                    f"  한 드라이버 사망이 아니라 **모터 전원이 꺼져 있거나 enable 이 안 먹은**\n"
                    f"  상태다. 모터 전원을 확인하고, 켜져 있다면 Emb 를 재기동할 것.\n"
                    f"  (원 증상: {e})") from None
        except SafetyAbort:
            raise
        except Exception:
            pass                          # 진단이 실패해도 원래 예외는 살린다
        rows = "\n".join(
            f"    ch{c}  명령 {cmd:+9.2f}  측정 {q:+9.2f}  오차 {cmd - q:+7.2f}"
            f"  속도 {dq:+8.1f}  토크 {tq:+7.3f}" for c, cmd, q, dq, tq in snap)
        raise SafetyAbort(f"{e}\n  트립 순간 전 채널(채널각):\n{rows}") from None
    finally:
        hw.lim = saved
