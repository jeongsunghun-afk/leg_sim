"""state_pub.py — 뷰어/GUI 가 읽는 상태 파일 발행. **단일 구현**.

★왜 분리했나 (2026-08-11)
  종전에는 app/biped_emb.py 안에만 있었다. 그래서 PACE 하니스로 시험할 때는
  (writer 가 하나여야 하므로 biped_emb 를 끄니까) **뷰어가 아무것도 못 봤다**.
  사람이 로봇 옆에서 토크시험을 돌리는데 화면에 자세가 안 뜨는 상태였다.
  ⇒ 발행을 여기로 빼서 biped_emb 와 PACE 가 **같은 파일·같은 스키마**로 쓴다.
    복사본을 만들면 스키마가 갈라져서 뷰어가 한쪽만 이해하게 된다.

경로는 QUAD_STATE 환경변수로 바꿀 수 있다(기본 /tmp/biped_state.json).
쓰기는 tmp→os.replace 로 **원자적**이다 — 뷰어가 반쯤 쓰인 JSON 을 읽지 않는다.
"""
from __future__ import annotations

import json
import os

import numpy as np

STATE_PATH = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")


def publish_state(mode, q_leg_deg, rpy_deg, loop_hz, motors_on, backend, extra=None,
                  path: str | None = None):
    """상태 1건 발행. 실패는 조용히 삼킨다 — 발행 실패가 제어를 멈추면 안 된다.

    q_leg_deg 는 **모델각**이다(채널각 아님). 뷰어의 MJCF qpos 와 같은 좌표계여야 한다.
    """
    st = {"mode": mode, "q_leg_deg": [round(float(x), 2) for x in q_leg_deg],
          "rpy_deg": [round(float(x), 2) for x in rpy_deg],
          "tilt_deg": round(float(np.hypot(rpy_deg[0], rpy_deg[1])), 2),
          "loop_hz": round(float(loop_hz), 1), "motors_on": bool(motors_on),
          "backend": backend}
    if extra:
        st.update(extra)
    p = path or STATE_PATH
    try:
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, p)
    except Exception:
        pass
