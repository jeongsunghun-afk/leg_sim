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
        쓰면 안 된다. **마찰·기동은 ±방향 차로 중력이 상쇄되므로** 영향이 작다.

    ★arm 이 되어 있어야 한다 — enable 없이 쓰면 SHM 에 kp=kd=0 이 나가
      (shm_bridge.cpp:112) 모터가 전혀 안 움직이는데 그게 조용히 지나간다.
    """
    if not hw._armed:
        raise RuntimeError("goto_home 전에 arm() 을 호출할 것 — enable 없이는 무동작이다.")
    if speed_dps is not None:                 # CLI 로 더 느리게 가고 싶을 때
        homer.vmax = max(float(speed_dps), 1e-6)

    q_ch = np.array([hw.read(c)[0] for c in range(hw.n)], float)
    q_ch0 = q_ch.copy()                # ★출발 채널각 — 탐색범위를 여기까지 늘린다(계단 방지)
    # ★확장 탐색범위를 **한 번만** 만들어 쓰기·검사에 **같은 것**을 쓴다.
    #   2026-08-12: _write_leg 만 확장하고 _trip_check 는 원래 탐색범위를 쓰게 뒀더니,
    #   명령은 계단이 없는데 **검사가 트립**했다(ch2 20.34° ∉ [−145.58, 18.52]).
    #   늘어진 calf 모델각이 −60.8° 로 관절한계 −59.6° 밖이라 생긴 일이다.
    #   ⇒ 같은 값을 두 곳에서 다르게 다루면 반드시 어긋난다. 하나로 만든다.
    #   2026-08-14: 시작각까지 넓혀도 **그 뒤로 더 처지면** 트립한다. 전원을 껐다 켠 뒤
    #   HR_hip 이 −15.15° 로 탐색범위 [−14.90, +14.90] 를 0.25° 넘겨 홈복귀가 시작도 못 했다.
    #   q_ch0 를 잡는 순간에는 안에 있었는데 무여자로 흘러내린 것이다 — hip 은 중력이
    #   5.25Nm 이라 계속 처진다. ⇒ 넓힐 때 **여유**를 준다. 이건 한계 완화가 아니라
    #   "출발점이 어디든 안으로 데려온다" 는 홈복귀의 목적 그 자체다(actuator_test 도
    #   같은 이유로 arm 직전에 탐색범위를 현재각까지 넓힌다).
    #   ⚠여유는 **넓히는 쪽에만** 붙는다. 원래 탐색범위가 더 넓으면 그대로 둔다.
    _SAG = 2.0        # [deg] 잡은 뒤 더 처질 수 있는 폭
    box_eff = None
    if q_box is not None:
        box_eff = {c: (min(lo, float(q_ch0[c]) - _SAG), max(hi, float(q_ch0[c]) + _SAG))
                   for c, (lo, hi) in q_box.items()}
        _out = {c: float(q_ch0[c]) for c, (lo, hi) in q_box.items()
                if not (lo <= float(q_ch0[c]) <= hi)}
        if _out:
            log("    ⚠출발각이 탐색범위 밖이다 — 홈복귀 동안만 넓힌다: "
                + " ".join(f"ch{c} {v:+.2f}°" for c, v in sorted(_out.items())))
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
    # ★only_ch 면 그 축만, 그리고 **채널각**으로 본다 (2026-08-12 실기 ch3).
    #   종전엔 모델각 오차를 썼는데, 커플링 축은 그게 추종성능이 아니다:
    #     q_foot_model = q_ch3/(s·k) − coef·q_calf
    #   calf 가 무여자로 −60.7° 늘어져 있으면 **foot 모터가 채널각 0 에 정확히 도착해도**
    #   모델각은 +59.4° 로 뜬다. 실기 로그가 그랬고, 그걸 "축이 못 따라갔다" 로 찍었다.
    #   그 뒤 시험은 R²=0.967 로 멀쩡히 돌았다 — 경고가 틀렸던 것이다.
    #   ⇒ solo 의 추종 판정은 **우리가 명령한 그 값**(채널각)으로 한다. 모델각은 참고로만.
    e_model = float(abs(err[int(np.where(np.asarray(jm.ch) == only_ch)[0][0])])) \
        if only_ch is not None else 0.0
    emax = (abs(float(hw._q_cmd[only_ch]) - float(hw.read(only_ch)[0]))
            if only_ch is not None else float(np.max(np.abs(err))))
    if only_ch is not None:
        log(f"    도착 — 시험축 **채널각** 오차 {emax:.2f}° "
            f"(모델각 오차 {e_model:.2f}° — 커플링 원천축이 무여자면 여기는 안 맞는 게 정상)")
        log(f"      참고 모델각: "
            f"{', '.join(f'{jm.names[i]}{err[i]:+.1f}' for i in range(jm.n_leg))}")
    else:
        log(f"    도착 — 최대 오차 {emax:.2f}° (모델각: "
            f"{', '.join(f'{jm.names[i]}{err[i]:+.1f}' for i in range(jm.n_leg))})")
    if emax > tol_deg and only_ch is not None:
        # ★--solo 에서는 목표에 **원리적으로 못 갈 수 있다** (2026-08-12 실기).
        #   커플링 축이 그렇다: q_ch_foot = (q_foot + q_calf)·s·k 이므로 calf 가 −55° 로
        #   늘어져 있으면 foot **모델각** 0 을 만들려면 채널각 +66 이 필요한데
        #   탐색범위가 [−27.8, +48.0] 이라 닿지 못한다. ch3 이 55.16° 오차로 중단했다.
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
            log(f"      명령이 탐색범위 끝({cmd_ch:+.2f} ∈ [{lo:+.2f}, {hi:+.2f}])에 붙었다 —"
                f" **도달 불가능한 목표**다. 커플링 축이면 원천축을 먼저 0 으로 보낼 것.")
        else:
            log(f"      명령 {cmd_ch:+.2f} 는 탐색범위 안이다 — 축이 못 따라간 것이다"
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

    ★탐색범위를 **현재 측정각까지 늘려서** 적용한다 — 안 그러면 계단이 나간다.
      2026-08-12 실기: 늘어진 자세의 calf 모델각이 −61.0° 인데 관절한계가 −59.6° 라
      채널탐색범위 밖이었다 → 첫 틱에 2.1° 계단(kp 80 이면 2.9Nm).
      배포 앱(biped_emb)에서 같은 사고가 **34.8° 계단 → 426dps 폭주**로 터졌다
      (hw_interface.write_ramped 주석). 여기도 같은 구조였다.
    ⚠보호는 유지된다: 현재 자리는 허용하되 **더 바깥으로는 못 간다**. 목표(홈)는
      탐색범위 안이므로 궤적이 탐색범위 쪽으로만 데려간다.
    """
    q_ch = jm.q_joint_to_ch(np.asarray(q_leg_deg, float))
    if only_ch is not None:
        # ★그 채널만 쓴다. hold_ch 가 비어 있으므로 나머지는 kp=kd=0 으로 나간다.
        v = float(q_ch[only_ch])
        if q_box is not None and only_ch in q_box:
            lo, hi = q_box[only_ch]
            v = min(max(v, lo), hi)
        # ★중력 FF 는 **hw._raw_write 가 알아서 태운다**(hwio.tau_ff_fn 주석).
        #   여기서 손으로 꿰지 않는다 — 한 곳만 꿰면 다음 단계가 빠뜨린다.
        #   실제로 그렇게 터졌다: 홈복귀에만 FF 를 넣었더니 verify_driver_live 가
        #   FF 없이 이어받아 HL_thigh 가 17.5° 튀었다.
        #   ⚠FF 가 있으면 홈복귀 정상상태 오차가 4.0° → 0.7° 로 떨어진다.
        hw._raw_write(only_ch, v, kp, kd)
        return
    hw._raw_write_all(q_ch, hw.hold_kp, hw.hold_kd, q_box)


def _trip_check(hw, q_box=None, only_ch=None) -> None:
    """전 채널 트립 감시. ★위치한계만은 **채널별**로 본다.

    ⚠hw.lim 은 **시험축** 한계다. 그걸 전 채널에 적용하면 다른 축이 자기 범위 안인데도
      트립한다 — 2026-08-11 오프라인 테스트가 잡았다(HR_calf 76.01° 가 HL_foot 탐색범위
      [−180, 75.97] 에 걸렸다). _raw_write_all 은 이미 채널별 탐색범위를 쓰는데
      여기만 안 고쳐져 있었다. 종전 영점에서는 우연히 안 걸렸을 뿐이다.
    ⚠속도·토크·stale·추종오차는 축과 무관한 물리량이라 그대로 hw.lim 을 쓴다.
    """
    # ★`hw.lim` 만 바꾸면 **안 먹는다** (2026-08-14 실기에서 잡혔다).
    #   `hw._check` 는 `limits_for(c)` 를 쓰는데 그건 `lim_ch[c]` 를 **우선**한다.
    #   collect_multichirp 는 채널별 탐색범위를 등록하므로(로그 "채널별 한계 적용") 여기서
    #   준 확장이 통째로 무시됐다 — 전원 재투입 후 HR_hip 이 −15.19° 로 처졌는데
    #   홈복귀가 "∉ [−14.90, 14.90]" 으로 트립했다. 확장 경고는 찍혔는데도 그랬다.
    #   ⚠이 파일 위쪽이 경고한 그 부류다("같은 값을 두 곳에서 다르게 다루면 반드시
    #     어긋난다"). 그때 lim/box_eff 를 맞춰 뒀는데 **lim_ch 가 나중에 생기며 재발**했다.
    #   ⇒ 둘 다 덮고 둘 다 되돌린다.
    saved = hw.lim
    saved_ch = dict(hw.lim_ch)
    snap = []
    try:
        for c in (range(hw.n) if only_ch is None else (only_ch,)):
            q, dq, tq, _ = hw.read(c)
            snap.append((c, float(hw._q_cmd[c]), q, dq, tq))
            if q_box is not None and c in q_box:
                lo, hi = q_box[c]
                hw.lim = replace(saved, q_min=lo, q_max=hi)
                hw.lim_ch[c] = replace(hw.limits_for(c), q_min=lo, q_max=hi)
            else:
                hw.lim = saved
            hw._check(c, q, dq, tq, float(hw._q_cmd[c]))
    except SafetyAbort as e:
        # ★트립 순간의 **전 채널 상태**를 같이 찍는다 (2026-08-12).
        #   한 축의 값만 보면 원인을 못 가린다 — 커플링 때문에 어느 축의 지연이
        #   다른 축의 오차로 나타난다(q_ch_foot 는 calf 관절에도 의존한다).
        #   ⚠검사 도중 트립하면 뒤쪽 채널은 아직 안 읽었다. 그건 읽어서 채운다.
        # ★**빠진 채널**을 채운다 — `range(len(snap), n)` 이 아니다 (2026-08-12).
        #   그 식은 "snap 이 0번부터 순서대로 찼다" 를 가정하는데 --solo 는 시험축
        #   하나만 담는다. only_ch=3 이면 len(snap)=1 → range(1,10) 이 되어
        #   **ch0 이 통째로 빠지고 ch8·ch9(존재하지 않는 채널)가 찍혔다.**
        #   실기 로그에 그대로 나왔다 — 진단하려고 만든 덤프가 hip 을 안 보여줬다.
        have = {c for c, *_ in snap}
        for c in [c for c in range(hw.n) if c not in have]:
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
            dead_ch = []
            for c, cmd, q, dq, tq in snap:
                # ★**실제로 써 보낸** 게인을 쓴다 — hold 게인이 아니다(hwio 주석 참조).
                #   다축처프는 자기 게인셋으로 돌아 hold_kp 와 다르다.
                kp_c, _ = hw._written_gain_of(c)
                exp = kp_c * abs(cmd - q) * np.pi / 180.0
                if exp > 0.5:
                    if abs(tq) < exp * 0.15:
                        dead += 1
                        dead_ch.append((c, cmd - q, exp, tq))
                    live += 1
            if live >= 3 and dead == live:
                hw.limp()
                raise SafetyAbort(
                    f"**전 축 무여자** — 유의미한 오차가 있는 {live}축이 전부 무토크다.\n"
                    f"  한 드라이버 사망이 아니라 **모터 전원이 꺼져 있거나 enable 이 안 먹은**\n"
                    f"  상태다. 모터 전원을 확인하고, 켜져 있다면 Emb 를 재기동할 것.\n"
                    f"  (원 증상: {e})") from None
            # ★**한 축만** 죽은 경우도 이름을 대야 한다 (2026-08-12 실기 ch2).
            #   종전엔 '전 축 무여자' 일 때만 진단했다. 한 축이면 그냥 추종오차로 떠서
            #   "막힘·게인부족·기계간섭 의심" 이라는 **엉뚱한 세 원인**을 제시했다.
            #   정작 같은 덤프에 답이 있었다: ch2 오차 35.04° 에 토크 −0.025Nm(비 0.002)
            #   인데 나머지 축은 0.84~2.66 으로 멀쩡했다. 증상은 하나인데 원인을 셋으로
            #   흩는 그 부류다 — 오늘 여러 번 반복됐다.
            if dead_ch and live > dead:
                hw.limp()
                raise SafetyAbort(
                    "**드라이버가 명령을 안 받는다**(파워단 사망이 아니다) — "
                    + " · ".join(f"ch{c}(오차 {ee:+.2f}° · 명령 {cc:.2f}Nm · 보고 {tt:+.3f}Nm"
                                 f" · 비 {abs(tt)/cc:.3f})" for c, ee, cc, tt in dead_ch)
                    + f"\n  같은 순간 나머지 {live - dead}축은 정상 토크를 낸다 — "
                      f"**그 축만** 죽은 것이다.\n"
                    f"  복구: Emb 종료 → **모터 전원 OFF → 3초 → ON** → Emb 재기동.\n"
                    f"  ⚠Emb 재기동만으로는 안 풀린다. 드라이버 자체 래치오프다.\n"
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
        hw.lim_ch.clear(); hw.lim_ch.update(saved_ch)
