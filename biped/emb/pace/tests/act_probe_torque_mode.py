#!/usr/bin/env python3
"""act_probe_torque_mode.py — 드라이버가 **순수 토크모드**(Kp=Kd=0, fTorque만)를 받는가?

★왜 중요한가: WBIC/MPC 는 관절 **토크**를 출력한다. 드라이버가 순수 토크를 받으면
  컨트롤러 출력을 그대로 넣을 수 있다. 안 받으면 위치+게인으로 변환해야 하는데,
  그건 근본적으로 다른(그리고 열등한) 인터페이스다 — 임피던스 뒤에 토크를 숨기는 꼴이라
  접촉힘 제어의 권한이 드라이버로 넘어간다.

측정 방법 — 토크 램프로 파단(breakaway)을 찾는다:
  Kp=Kd=0 으로 두고 tau_ff 를 **중력 바이어스**부터 아주 천천히 올린다.

  ★왜 0 이 아니라 바이어스인가 (2026-08-12)
    0 부터 올리면 **중력이 큰 축은 아예 못 잰다**:
      hip  중력 5.25 Nm(채널) — 파단 0.65 를 재려면 +dir 에 5.90 Nm 이 필요한데
           tau_max 가 1.4 라 도달 자체가 불가능하다. 자세를 바꿔도 안 준다
           (다리를 좌우로 여는 방향이라 thigh·calf 각과 무관).
      calf 중력 0.81 Nm > 파단 0.65 — **kp=kd=0 을 주는 순간 이미 미끄러진다.**
           τ_ff=0 인데 파단이 잡히니 측정이 성립하지 않고, 하위 링크가 떨어진다.
    ⇒ 램프 시작점을 그 축의 **중력토크**로 옮긴다. 그러면 양방향 모두
        τ_break± = bias ± τ_friction  이고 마찰은 (τ⁺−τ⁻)/2 로 깨끗이 분리된다.
      부수효과: 지금까지 +dir/−dir 비대칭에 섞여 있던 중력이 사라진다.
    ⚠바이어스는 **먼저 램프로 올린 뒤** 정착시키고 프로브를 시작한다. 계단으로 주면
      그 자체가 축을 움직여 파단으로 오검출된다.
    ⚠모델 중력이 틀리면 축이 표류한다. 바이어스 인가 후 이동량을 확인하고,
      move_thresh 를 넘으면 **바이어스가 틀린 것**이라 보고 중단한다.
    · 어느 지점에서 관절이 움직이기 시작하면 → **순수 토크모드 지원**.
      그 파단토크는 위치모드로 잰 정지마찰(τ_s)과 일치해야 한다 → 교차검증까지 된다.
    · tau_max 까지 올려도 미동이면 → fTorque 가 무시된다(= 순수 토크모드 미지원).

⚠ 안전 — 위치+게인 모드와 근본적으로 다르다:
  임피던스 모드는 토크가 `Kp·err` 로 자기제한되지만, 순수 토크는 **제한이 없다**.
  마찰(정지 0.71 Nm)을 넘는 순간 관절이 계속 가속한다. 다리 미장착 관성 0.0375 kg·m²
  에서 1 Nm 면 α=26.7 rad/s². 그래서:
    · tau_max 를 정지마찰의 2배 이내로 (기본 1.4 Nm)
    · 움직임 감지 즉시 토크 0 (한 틱 = 5 ms 안에)
    · 위치·속도 한계는 hwio._check 가 매 틱 강제
    · 어떤 경로로 끝나든 limp
  **다리가 없는 지금이 이 시험을 하기에 가장 안전한 시점이다.**
"""
from __future__ import annotations

import time

import numpy as np

from hwio import DEG


def probe_torque_mode(hw, spec, joint, log=print) -> dict:
    ch = int(joint["ch"])
    name = joint["name"]
    cfg = spec.get("torque_mode", {})
    ramp = float(cfg.get("ramp_nm_per_s", 0.25))
    move_deg = float(cfg.get("move_thresh_deg", 0.30))
    trials = int(cfg.get("trials", 2))
    # ★중력 바이어스 — 램프 시작점(위 주석 참조). 채널토크[Nm], 부호 포함.
    bias = float((cfg.get("tau_bias_by_ch") or {}).get(ch, 0.0))
    swing = float((cfg.get("swing_by_ch") or {}).get(ch, cfg.get("tau_max_nm", 1.4)))
    swing = float(swing)                               # 바이어스 **주변** 진폭(축별 가능)
    tau_max = abs(bias) + swing                        # 절대 상한(클립용)

    log(f"  [{name}] 순수 토크모드 프로브 — tau {bias:+.3f}±{swing} Nm @ {ramp} Nm/s, "
        f"방향당 {trials}회" + (f"  (중력 바이어스 {bias:+.3f})" if bias else ""))
    log(f"           (위치모드로 잰 정지마찰 τ_s 와 비교하면 교차검증이 된다)")

    results, raw = [], []
    for direction in (+1.0, -1.0):
        for k in range(trials):
            # ★중력이 큰 축은 **게인을 먼저 놓으면 안 된다** — 바이어스를 올리기 전에
            #   떨어진다(드라이런: hip 207dps 트립, calf 1.58° 표류).
            #   ⇒ 홀드게인으로 잡고 시작해 kp·kd→0 / τ→bias 로 **핸드오프**한다.
            handoff = abs(bias) > 0.05
            kp_h, kd_h = (hw._hold_gain_of(ch) if handoff else (0.0, 0.0))
            hw.arm(ch, kp_h, kd_h)
            time.sleep(0.2)
            q_latch = float(hw._q_cmd[ch])   # arm 이 래치한 명령각
            q_pre = hw.read(ch)[0]
            if bias != 0.0:
                # ★목표각은 **arm 이 래치한 명령각 그대로** 다.
                #   arm 직후 축은 이미 처져서 kp·(q_cmd−q) = bias 로 균형을 이루고 있다.
                #   핸드오프는 그 균형을 (1−r) 비율로 토크에 넘기는 것뿐이라,
                #   q_cmd 를 유지하면 kp·err 가 저절로 (1−r)·bias 가 되어 축이 안 움직인다.
                #   ⚠측정각 q_pre 에 처짐분을 더하면 **이중 계산**이다 —
                #     드라이런에서 hip 이 err 6.02° → 10.46 Nm 로 토크 트립했다.
                nb = max(1, int(0.6 / hw.dt))
                for b in range(nb):
                    r = (b + 1) / nb
                    hw.step_torque(ch, bias * r, tau_max,
                                   kp=kp_h * (1 - r), kd=kd_h * (1 - r), q_des=q_latch)
                    time.sleep(hw.dt)
                # ⚠정착은 stale_ms(150) 보다 짧게. 실기는 잡음(σ_q 0.0044°)이 있어 값이
                #   늘 변하지만, 완전히 정지한 구간이 길면 stale 이 오탐한다.
                for _ in range(int(0.10 / hw.dt)):
                    hw.step_torque(ch, bias, tau_max)
                    time.sleep(hw.dt)
                # ★판정 기준은 "**계속 흐르는가**" 지 "움직였나" 가 아니다.
                #   핸드오프 중 축은 그 축의 정상 처짐(bias/kp_h — hip 3.0° · calf 0.58°)
                #   만큼 내려앉는다. 그건 **무해**하다 — 끝나면 τ=bias 가 중력과 정확히
                #   상쇄해 그 자리에 선다. 그걸 오류로 보면 hip·calf 를 영영 못 잰다.
                #   바이어스가 **틀렸을 때만** 정착 후에도 계속 흐른다.
                q_s = hw.read(ch)[0]
                for _ in range(int(0.15 / hw.dt)):
                    hw.step_torque(ch, bias, tau_max)
                    time.sleep(hw.dt)
                creep = hw.read(ch)[0] - q_s          # 정착 후 0.15s 동안의 표류
                settle = hw.read(ch)[0] - q_pre       # 핸드오프 총 이동(정보용)
                drift = creep
                if abs(creep) > move_deg * 0.5:
                    hw.release_test_axis(ch)
                    gk = float(joint.get("gear_k", 1.0))
                    raise RuntimeError(
                        f"{name}: 중력 바이어스 {bias:+.3f} Nm 인가 후에도 축이 **계속 흐른다** "
                        f"— 정착 뒤 0.15s 에 {creep:+.3f}° (한계 {move_deg*0.5:.2f}°). "
                        f"핸드오프 총 이동은 {settle:+.3f}° (정상 처짐 "
                        f"{np.rad2deg(bias/kp_h) if kp_h else 0:+.2f}° 와 비교할 것).\n"
                        f"    **바이어스가 틀렸다** — 이 상태로 재면 파단토크가 오염된다.\n"
                        f"    표류 부호가 바이어스와 **같으면 과다**, 반대면 **부족**이다.\n"
                        f"    토크 규약 후보:  채널 {bias:+.3f}  ·  관절 {bias * gk:+.3f} Nm "
                        f"(gear_k={gk})\n"
                        f"    spec.torque_mode.tau_bias_by_ch[{ch}] 를 고칠 것.")
            q0 = hw.read(ch)[0]
            t0 = time.monotonic()
            tau_at_move, tau_peak, moved = None, 0.0, False
            traj = []
            while True:
                t = time.monotonic() - t0
                tau_cmd = bias + direction * min(ramp * t, swing)
                if ramp * t >= swing and t > swing / ramp + 1.5:
                    break                       # 상한에서 1.5초 더 버텨보고 종료
                s = hw.step_torque(ch, tau_cmd, tau_max)
                traj.append((t, tau_cmd, s.q_deg, s.dq_dps, s.tau))
                tau_peak = max(tau_peak, abs(tau_cmd - bias))
                if (s.q_deg - q0) * direction > move_deg:
                    # ★기록은 **바이어스를 뺀** 값 = 순수 마찰분이다.
                    tau_at_move = abs(tau_cmd - bias); moved = True
                    break                       # ★즉시 중단 → 다음 줄에서 토크 0
                time.sleep(hw.dt)
            # ★limp() 가 아니라 시험축만 푼다 (2026-08-11).
            #   limp 는 **전 채널** kp=kd=0 이라 홀드축까지 놓는다. 다리가 없던 시절엔
            #   그게 곧 안전이었지만 지금은 매단 다리가 떨어진다. 지그가 물려 있으면
            #   안 떨어지지만, 지그 없이 --tests torque 를 돌리면 시행마다 다리가 주저앉는다.
            hw.release_test_axis(ch)
            time.sleep(0.4)
            a = np.array(traj) if traj else np.zeros((1, 5))
            raw.append(a)                       # ★원시 궤적 보존(아래서 npz 로 저장)
            results.append({"dir": direction, "moved": moved, "tau_break": tau_at_move,
                            "tau_peak": tau_peak, "dq_max": float(np.abs(a[:, 3]).max()),
                            "dq_end": float(a[-1, 2] - q0), "n": len(traj)})
            log(f"    {'+' if direction > 0 else '−'}dir #{k}: "
                + (f"파단 {tau_at_move:.3f} Nm 에서 움직임" if moved
                   else f"{tau_peak:.3f} Nm 까지 올려도 미동(이동 {a[-1,2]-q0:+.3f}°)"))

    moved_any = [r for r in results if r["moved"]]
    supported = len(moved_any) >= 2            # 양방향 최소 1회씩은 움직여야 인정
    out = {"supported": supported, "trials": results, "ch": ch, "name": name,
           # ★기어정보를 결과에 실어 보낸다 — 파단토크는 **채널토크**라 리포트에서
           #   관절토크(τ_ch·k)·모터축토크(÷N)로 환산해야 축끼리 비교가 된다.
           "gear_k": float(joint.get("gear_k", 1.0)),
           "gear": float(joint.get("gear", 7.0)),
           "tau_break_mean": (float(np.mean([r["tau_break"] for r in moved_any]))
                              if moved_any else None)}

    # ★원시 궤적을 npz 로 남긴다(2026-08-11). 종전엔 통계만 남기고 버려서, 사후에
    #   "파단 직후 가속도가 얼마였나" 같은 걸 확인할 방법이 없었다.
    try:
        import os
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results")
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f"torque_probe_ch{ch:02d}.npz"),
                 **{f"trial{i}": a for i, a in enumerate(raw)},
                 cols=np.array(["t", "tau_cmd", "q_deg", "dq_dps", "tau_meas"]),
                 tau_break=np.array([r["tau_break"] if r["tau_break"] else np.nan
                                     for r in results]),
                 dirs=np.array([r["dir"] for r in results]),
                 gear_k=float(joint.get("gear_k", 1.0)), gear=float(joint.get("gear", 7.0)))
        log(f"  원시 궤적 저장: results/torque_probe_ch{ch:02d}.npz ({len(raw)} 시행)")
    except Exception as e:
        log(f"  ⚠ 원시 궤적 저장 실패({type(e).__name__}: {e}) — 측정 자체는 유효하다")

    log("")
    if supported:
        tb = out["tau_break_mean"]
        log(f"  ✅ 순수 토크모드 **지원됨** — 평균 파단토크 {tb:.3f} Nm")
        log(f"     → WBIC/MPC 토크를 fTorque 로 직접 넣을 수 있다(Kp=Kd=0).")
    elif moved_any:
        log(f"  ⚠ 한쪽 방향만 움직였다 — 중력/편향 의심. 재시험 필요")
    else:
        log(f"  ❌ 순수 토크모드 **미지원** — {tau_max} Nm 까지 올려도 미동.")
        log(f"     fTorque 가 무시되는 것으로 보인다. 배포는 위치+게인 경로로 가야 하며,")
        log(f"     WBIC 토크를 임피던스로 변환하는 계층이 필요하다.")
    return out
