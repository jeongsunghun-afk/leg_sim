"""control/jog.py — per-axis 안전 저속 위치 jog. 첫 딜리버러블(각축 검증)의 핵심.

GUI 가 축별 목표각(deg, 다리 8관절)을 주면 max_speed 로 램프 → 급격한 명령 방지.
jog 안전한계(JointMap.jog_min/max) 로 클램프. 출력 = 채널 배열(hw.write_jog).
"""
from __future__ import annotations
import numpy as np
from joint_map import JointMap


class Jogger:
    def __init__(self, jm: JointMap, dt: float, max_speed_dps: float):
        self.jm = jm
        self.max_step = max_speed_dps * dt         # 1스텝 최대 이동[deg]
        self.q_leg = np.zeros(jm.n_leg)            # 램프 중인 명령(다리 8, deg)

    def reset(self, q_leg_deg):
        """현재 측정각에서 시작(명령 점프·튀는 동작 방지)."""
        self.q_leg = np.clip(np.asarray(q_leg_deg, float), self.jm.jog_min, self.jm.jog_max)

    def step(self, goal_leg_deg) -> np.ndarray:
        goal = np.clip(np.asarray(goal_leg_deg, float), self.jm.jog_min, self.jm.jog_max)
        err = goal - self.q_leg
        self.q_leg += np.clip(err, -self.max_step, self.max_step)
        out = np.zeros(self.jm.n_channel)
        out[self.jm.ch] = self.q_leg
        return out

    def at_goal(self, goal_leg_deg, tol_deg) -> bool:
        goal = np.clip(np.asarray(goal_leg_deg, float), self.jm.jog_min, self.jm.jog_max)
        return bool(np.all(np.abs(goal - self.q_leg) < tol_deg))
