#!/usr/bin/env python3
"""act_probe_torque_mode.py — 드라이버가 **순수 토크모드**(Kp=Kd=0, fTorque만)를 받는가?

★왜 중요한가: WBIC/MPC 는 관절 **토크**를 출력한다. 드라이버가 순수 토크를 받으면
  컨트롤러 출력을 그대로 넣을 수 있다. 안 받으면 위치+게인으로 변환해야 하는데,
  그건 근본적으로 다른(그리고 열등한) 인터페이스다 — 임피던스 뒤에 토크를 숨기는 꼴이라
  접촉힘 제어의 권한이 드라이버로 넘어간다.

측정 방법 — 토크 램프로 파단(breakaway)을 찾는다:
  Kp=Kd=0 으로 두고 tau_ff 를 0 부터 아주 천천히 올린다.
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
    tau_max = float(cfg.get("tau_max_nm", 1.4))
    ramp = float(cfg.get("ramp_nm_per_s", 0.25))
    move_deg = float(cfg.get("move_thresh_deg", 0.30))
    trials = int(cfg.get("trials", 2))

    log(f"  [{name}] 순수 토크모드 프로브 — tau 0→{tau_max} Nm @ {ramp} Nm/s, 방향당 {trials}회")
    log(f"           (위치모드로 잰 정지마찰 τ_s 와 비교하면 교차검증이 된다)")

    results = []
    for direction in (+1.0, -1.0):
        for k in range(trials):
            hw.arm(ch, 0.0, 0.0)               # 게인 0 으로 인가(램프도 0→0 이라 무해)
            time.sleep(0.2)
            q0 = hw.read(ch)[0]
            t0 = time.monotonic()
            tau_at_move, tau_peak, moved = None, 0.0, False
            traj = []
            while True:
                t = time.monotonic() - t0
                tau_cmd = direction * min(ramp * t, tau_max)
                if abs(tau_cmd) >= tau_max and t > tau_max / ramp + 1.5:
                    break                       # 상한에서 1.5초 더 버텨보고 종료
                s = hw.step_torque(ch, tau_cmd, tau_max)
                traj.append((t, tau_cmd, s.q_deg, s.dq_dps, s.tau))
                tau_peak = max(tau_peak, abs(tau_cmd))
                if (s.q_deg - q0) * direction > move_deg:
                    tau_at_move = abs(tau_cmd); moved = True
                    break                       # ★즉시 중단 → 다음 줄에서 토크 0
                time.sleep(hw.dt)
            hw.limp()
            time.sleep(0.4)
            a = np.array(traj) if traj else np.zeros((1, 5))
            results.append({"dir": direction, "moved": moved, "tau_break": tau_at_move,
                            "tau_peak": tau_peak, "dq_max": float(np.abs(a[:, 3]).max()),
                            "dq_end": float(a[-1, 2] - q0), "n": len(traj)})
            log(f"    {'+' if direction > 0 else '−'}dir #{k}: "
                + (f"파단 {tau_at_move:.3f} Nm 에서 움직임" if moved
                   else f"{tau_peak:.3f} Nm 까지 올려도 미동(이동 {a[-1,2]-q0:+.3f}°)"))

    moved_any = [r for r in results if r["moved"]]
    supported = len(moved_any) >= 2            # 양방향 최소 1회씩은 움직여야 인정
    out = {"supported": supported, "trials": results, "ch": ch, "name": name,
           "tau_break_mean": (float(np.mean([r["tau_break"] for r in moved_any]))
                              if moved_any else None)}

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
