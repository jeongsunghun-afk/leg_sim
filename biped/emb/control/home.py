"""control/home.py — 정해진 홈 자세로 스무스(S-curve)하게 복귀하는 궤적.

레퍼런스: motion-controller/motorcortex_bridge/motion_controller.py 의
  move_to_home / move_j / trapezoid_profile / _quintic_segment.

레퍼런스와 다르게 만든 곳 세 군데. 전부 **의도적**이고, 이유가 있다.

  1. **비차단(stepper)으로 바꿨다.**
     레퍼런스 `_send_waypoints` 는 자기 안에서 sleep 루프를 돌려 궤적을 끝까지
     전송한다(blocking). 그 구조를 여기 그대로 가져오면 궤적이 도는 몇 초 동안
     app/biped_emb.py 의 500Hz 메인루프가 통째로 멈춘다 — 워치독·기울기/토크/속도
     E-stop·상태발행이 전부 그 시간만큼 죽는다는 뜻이다. **복귀 동작은 사람이 로봇
     옆에 있을 때 누르는 버튼이라 하필 보호장치가 가장 필요한 구간이다.**
     → 진입 시 궤적을 만들어 두고, 매 틱 한 점씩 꺼내 쓴다(Jogger 와 동일한 계약).

  2. **구간시간 T 를 5차식 자체의 제약에서 계산한다.**
     레퍼런스는 사다리꼴 프로파일로 T 를 정한 뒤(`trapezoid_profile` 의 스텝 수)
     그 T 위에 다시 5차 다항식을 얹는다(`move_j`). 그런데 경계 속도·가속도가 0 인
     5차식의 최대속도는 1.875·d/T 다. 사다리꼴 기준으로 T 를 잡으면 그 T 에서
     사다리꼴은 정확히 v_max 를 찍으므로, 같은 T 의 5차식은 **v_max 를 87.5% 초과**한다
     (가속도도 마찬가지로 초과). 즉 레퍼런스의 max_vel/max_acc 는 실제 상한이 아니다.
     여기서는 5차식의 정규화 극값을 직접 써서
         T = max( S_VMAX·d/v_max ,  sqrt(S_AMAX·d/a_max) ,  min_time )
     로 잡는다. 두 한계를 실제로 지킨다.

  3. **전 축 동시 시작·동시 도착.** 가장 오래 걸리는 축의 T 를 전 축이 공유하므로
     관절공간에서 직선으로 움직인다(레퍼런스 move_to_home 의 "비례 스케일" 과 같은 의도).

궤적: q(t) = q0 + (q_home − q0)·s(τ),  τ = t/T
      s(τ) = 10τ³ − 15τ⁴ + 6τ⁵      (s(0)=0, s(1)=1, s'=s''=0 at 양끝)
경계 가속도까지 0 이라 토크 지령이 계단으로 튀지 않는다 — Jogger 의 등속 램프(가속도가
시작/끝에서 불연속)와 이 모드의 차이가 바로 이것이다.
"""
from __future__ import annotations

import math

import numpy as np

from joint_map import JointMap


class HomeTrajectory:
    """홈 복귀 S-curve 궤적 생성기. 상태를 들고 매 틱 step() 으로 한 점씩 뱉는다."""

    # 5차식 s(τ)=10τ³−15τ⁴+6τ⁵ 의 정규화 극값(위 주석 참조).
    #   s'(τ)  = 30τ²(1−τ)²          → 최대 1.875   (τ=0.5)
    #   s''(τ) = 60τ−180τ²+120τ³     → 최대 10/√3   (τ=(1−1/√3)/2)
    S_VMAX = 1.875
    S_AMAX = 10.0 / math.sqrt(3.0)          # ≈ 5.7735

    def __init__(self, jm: JointMap, dt: float, q_home_leg_deg,
                 max_speed_dps: float, max_acc_dps2: float,
                 min_time_s: float = 0.5):
        self.jm = jm
        self.dt = float(dt)
        self.vmax = max(float(max_speed_dps), 1e-6)
        self.amax = max(float(max_acc_dps2), 1e-6)
        self.min_time = max(float(min_time_s), 0.0)

        q_home = np.asarray(q_home_leg_deg, float)[: jm.n_leg]
        if q_home.size != jm.n_leg:
            raise ValueError(f"home.q_deg 길이 {q_home.size} ≠ 관절수 {jm.n_leg}")
        # ★홈 목표도 jog 안전한계 안으로 클램프한다. 조용히 자르면 "홈에 갔다" 는
        #   보고와 실제 자세가 어긋나므로, 잘렸는지는 clamped 로 밖에 알린다.
        self.q_home = np.clip(q_home, jm.jog_min, jm.jog_max)
        self.clamped = [(jm.names[i], float(q_home[i]), float(self.q_home[i]))
                        for i in range(jm.n_leg)
                        if abs(q_home[i] - self.q_home[i]) > 1e-9]

        self.q0 = self.q_home.copy()
        self.d = np.zeros(jm.n_leg)
        self.T = 0.0
        self.t = 0.0
        self._q_leg = self.q_home.copy()

    # ── 궤적 생성 ───────────────────────────────────────────────────────────
    def start(self, q_leg_now_deg) -> float:
        """현재 측정각에서 홈까지의 궤적을 만든다. 소요시간[s] 반환.

        ★반드시 **측정각**으로 시작할 것(명령각이 아니라). 부하로 처져 있는 상태에서
          명령각을 기점으로 잡으면 첫 틱에 그 편차만큼 계단 명령이 나간다.

        ★q0 는 **클램프하지 않는다**(2026-08-10). 시작점을 jog 한계로 자르면 현재 자세가
          범위 밖일 때 첫 틱에 그 차이만큼 계단 명령이 나간다 — 막으려던 바로 그 현상이다.
          클램프는 **목표(q_home)** 에만 건다(__init__). 그러면 범위 밖에서 시작해도
          S-curve 가 범위 안 목표까지 매끄럽게 데려온다.
          (calf 영점을 −55° 로 잡으면 jog 한계 −27.5° 와 27.5° 차이라 실제로 문제가 된다)
        """
        q0 = np.asarray(q_leg_now_deg, float)[: self.jm.n_leg].copy()
        self.q0 = q0
        self.d = self.q_home - q0
        dmax = float(np.max(np.abs(self.d))) if self.d.size else 0.0
        if dmax < 1e-9:
            self.T = 0.0                                  # 이미 홈 — 즉시 완료
        else:
            t_v = self.S_VMAX * dmax / self.vmax          # 속도한계에서 나오는 최소시간
            t_a = math.sqrt(self.S_AMAX * dmax / self.amax)   # 가속도한계에서 나오는 최소시간
            self.T = max(t_v, t_a, self.min_time)
        self.t = 0.0
        self._q_leg = q0.copy()
        return self.T

    # ── 1틱 ────────────────────────────────────────────────────────────────
    DT_CAP = 0.05          # 한 틱에 반영할 경과시간 상한[s]

    def step(self, dt: float | None = None) -> np.ndarray:
        """다음 목표각을 **모델각 배열[deg]**(n_leg)로 반환하고 시간을 진행.

        dt 를 주면 **실제 경과시간**만큼 궤적을 진행한다(미제공 시 공칭 dt).
        ★왜: 루프가 밀렸다 몰아 돌면 self.t 가 실제보다 빨리 흘러 궤적이 **빨리감기** 된다
          → v/a 한계가 무의미해진다. 경과시간 기준이면 몰아 돌아도 궤적 속도가 유지된다.
          (2026-08-07 Pi 과부하에서 loop_hz 500→4452 스파이크로 발견)

        T 를 지난 뒤에도 계속 호출해도 된다 — τ 가 1 로 클램프되어 홈 자세를 유지한다.
        """
        tau = 1.0 if self.T <= 0.0 else min(self.t / self.T, 1.0)
        s = tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)
        self._q_leg = self.q0 + self.d * s
        self.t += self.dt if dt is None else min(dt, self.DT_CAP)
        return self._q_leg.copy()      # ★모델각(n_leg). 채널 변환은 hw_interface 담당

    # ── 상태 ────────────────────────────────────────────────────────────────
    @property
    def progress(self) -> float:
        """0→1. 시간 기준(명령 진행률)이지 도달 보증이 아니다 — 도달은 at_goal 로 볼 것."""
        if self.T <= 0.0:
            return 1.0
        return float(min(self.t / self.T, 1.0))

    @property
    def done(self) -> bool:
        return self.T <= 0.0 or self.t >= self.T

    @property
    def q_cmd_leg(self) -> np.ndarray:
        return self._q_leg.copy()

    def at_goal(self, q_leg_now_deg, tol_deg: float) -> bool:
        """**측정각**이 홈에 들어왔는지. progress 와 달리 실제 도달을 본다."""
        q = np.asarray(q_leg_now_deg, float)[: self.jm.n_leg]
        return bool(np.all(np.abs(q - self.q_home) < tol_deg))


# ── 자체검증(하드웨어 불필요): python3 control/home.py ────────────────────────
def _selftest() -> int:
    """생성된 궤적이 실제로 v_max·a_max 를 지키는지 수치미분으로 확인한다.
    (레퍼런스 방식대로 사다리꼴 T 에 5차식을 얹으면 여기서 87.5% 초과로 실패한다.)"""
    class _JM:                                   # JointMap 최소 스텁
        n_leg, n_channel = 4, 10
        ch = np.array([0, 1, 2, 3])
        names = ["a", "b", "c", "d"]
        jog_min = np.array([-90.0] * 4)
        jog_max = np.array([90.0] * 4)

    dt, vmax, amax = 1.0 / 500, 15.0, 30.0
    jm = _JM()
    ok = True
    for q0 in ([30.0, -20.0, 5.0, 0.0], [0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]):
        tr = HomeTrajectory(jm, dt, [0.0] * 4, vmax, amax, min_time_s=0.5)
        T = tr.start(q0)
        q = []
        while not tr.done:
            q.append(tr.step()[jm.ch])
        q.append(tr.step()[jm.ch])
        q = np.array(q)
        v = np.diff(q, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        v_pk = float(np.max(np.abs(v))) if v.size else 0.0
        a_pk = float(np.max(np.abs(a))) if a.size else 0.0
        end_err = float(np.max(np.abs(q[-1])))
        v0 = float(np.max(np.abs(v[0]))) if v.size else 0.0
        good = (v_pk <= vmax * 1.02 and a_pk <= amax * 1.05
                and end_err < 1e-6 and v0 < vmax * 0.02)
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] q0={q0}  T={T:.3f}s  "
              f"v_pk={v_pk:6.2f}/{vmax} dps  a_pk={a_pk:6.2f}/{amax} dps²  "
              f"시작속도={v0:.3f}  종점오차={end_err:.2e}")
    print(f"=== home 궤적 자체검증 {'통과' if ok else '실패'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
