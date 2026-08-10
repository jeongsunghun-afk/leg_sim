"""control/jog.py — per-axis 안전 저속 위치 jog. 첫 딜리버러블(각축 검증)의 핵심.

GUI 가 축별 목표각(deg, 다리 8관절)을 주면 max_speed 로 램프 → 급격한 명령 방지.
jog 안전한계(JointMap.jog_min/max) 로 클램프. 출력 = 채널 배열(hw.write_jog).
"""
from __future__ import annotations
import numpy as np
from joint_map import JointMap


class Jogger:
    DT_CAP = 0.05          # 한 틱에 반영할 경과시간 상한[s] — 긴 정지 후 급이동 방지

    def __init__(self, jm: JointMap, dt: float, max_speed_dps: float):
        self.jm = jm
        self.max_speed = float(max_speed_dps)      # [deg/s] — 실경과시간 기준 제한용
        self.max_step = max_speed_dps * dt         # 1스텝 최대 이동[deg] (dt 미제공 시 폴백)
        self.q_leg = np.zeros(jm.n_leg)            # 램프 중인 명령(다리 8, deg)

    def reset(self, q_leg_deg):
        """현재 측정각에서 시작(명령 점프·튀는 동작 방지)."""
        self.q_leg = np.clip(np.asarray(q_leg_deg, float), self.jm.jog_min, self.jm.jog_max)

    def step(self, goal_leg_deg, dt: float | None = None) -> np.ndarray:
        """1틱 전진. dt 를 주면 **실제 경과시간** 기준으로 속도를 제한한다.

        ★왜 실경과시간인가 (2026-08-07 실측): 종전엔 호출 1회당 max_speed·dt_nominal 만큼
          움직였다. 그런데 Pi 가 과부하면(load 5.14/4코어) 루프가 밀렸다가 **여러 틱을
          한꺼번에** 돈다. 그러면 호출 횟수 기준 제한은 그대로 통과하면서 실제 속도는
          20 dps → 500 dps 로 뚫린다. 즉 **안전한계가 조용히 무력화**된다.
          경과시간 기준이면 몰아 돌든 말든 실제 각속도가 max_speed 를 넘지 않는다.
        ⚠dt 는 상한을 둔다 — 프로세스가 몇 초 멈췄다 깨어나면 그만큼 한 번에 움직이려 든다.
        """
        step = self.max_step if dt is None else self.max_speed * min(dt, self.DT_CAP)
        goal = np.clip(np.asarray(goal_leg_deg, float), self.jm.jog_min, self.jm.jog_max)
        err = goal - self.q_leg
        self.q_leg += np.clip(err, -step, step)
        out = np.zeros(self.jm.n_channel)
        out[self.jm.ch] = self.q_leg
        return out

    def at_goal(self, goal_leg_deg, tol_deg) -> bool:
        goal = np.clip(np.asarray(goal_leg_deg, float), self.jm.jog_min, self.jm.jog_max)
        return bool(np.all(np.abs(goal - self.q_leg) < tol_deg))
