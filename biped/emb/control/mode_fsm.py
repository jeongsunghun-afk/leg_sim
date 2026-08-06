"""control/mode_fsm.py — 운용 모드 상태기계.

  off   : 모터 전원 차단(limp). 안전 기본값.
  jog   : per-axis 저속 위치 검증(각축 확인). ★첫 딜리버러블.
  home  : 정해진 홈 자세로 S-curve 복귀(control/home.py). 도달 후 그 자세 유지.
  hold  : 현재자세 임피던스 홀드.
  stand : 모델기반 균형 서기(WBIC+MPC).      ← 배선만, jog 검증 후 실행
  walk  : 모델기반 보행.                       ← 배선만
전이 진입/이탈을 app 이 감지해 부작용(모터 enable, jogger reset, 컨트롤러 reset) 처리.

★home 과 hold 의 차이: hold 는 **지금 있는 자리**를 잡고, home 은 **정해진 자리로 간다**.
  home 은 궤적이 끝나도 mode 가 바뀌지 않는다 — 홈 자세를 계속 명령하므로 사실상
  "홈에서의 hold" 다. 완료 여부는 상태의 home_done/home_at_goal 로 본다.
"""
from __future__ import annotations

OFF, JOG, HOME, HOLD, STAND, WALK = "off", "jog", "home", "hold", "stand", "walk"
MODES = (OFF, JOG, HOME, HOLD, STAND, WALK)
MODEL_BASED = (STAND, WALK)                # 컨트롤러(MuJoCo 동역학) 필요


class ModeFSM:
    def __init__(self, mode: str = OFF):
        self.mode = mode if mode in MODES else OFF
        self.prev = self.mode

    def set(self, mode: str) -> bool:
        """모드 갱신. 바뀌었으면 True(진입 처리 트리거)."""
        if mode not in MODES:
            return False
        self.prev = self.mode
        self.mode = mode
        return self.mode != self.prev

    def entered(self, mode: str) -> bool:
        return self.mode == mode and self.prev != mode

    def is_model_based(self) -> bool:
        return self.mode in MODEL_BASED
