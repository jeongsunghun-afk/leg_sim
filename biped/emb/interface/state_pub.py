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


def leg_extra(jm, dq_ch=None, tau_ch=None, q_cmd_ch=None, tau_cmd_ch=None,
              dq_cmd_ch=None, stt=None, kp=None, kd=None):
    """채널 배열 → 모니터가 쓰는 확장 키(**모델각·관절토크**). 없는 항목은 빼고 낸다.

    ★왜 여기 두나 (2026-08-13): PACE 하니스도 상태를 발행하는데(state_pub 를 분리한 이유가
      그것이다) **위치만** 넘기고 있었다. 그래서 PACE 로 시험하는 동안 값 모니터의
      속도·토크가 통째로 비었다. 변환식을 하니스마다 복붙하면 반드시 갈라지므로
      (같은 실수를 GUI 한계·gen_emb_init_pose·calib_zero 에서 이미 세 번 했다)
      **JointMap 에 위임하는 한 곳**을 만든다.
    ⚠단위 규약: 각도·속도는 모델각[deg·deg/s], 토크는 관절[Nm]. 채널값을 그대로 내면 안 된다.
    """
    import numpy as _np
    e = {}
    if dq_ch is not None:
        e["dq_leg_dps"] = [round(float(v), 2) for v in jm.ch_to_dq_ctrl(dq_ch) * (180.0 / _np.pi)]
    if tau_ch is not None:
        e["tau_leg_nm"] = [round(float(v), 3) for v in jm.ch_to_tau_joint(tau_ch)]
    if q_cmd_ch is not None:
        e["q_cmd_deg"] = [round(float(v), 2) for v in jm.ch_to_q_joint(q_cmd_ch)]
    if dq_cmd_ch is not None:
        # ★0 이면 **0 으로 낸다.** 키를 빼면 모니터가 곡선을 안 그려서 "명령 0" 과
        #   "명령 모름" 이 화면에서 같아진다 — 전혀 다른 상태인데.
        e["dq_cmd_dps"] = [round(float(v), 2) for v in jm.ch_to_dq_ctrl(dq_cmd_ch) * (180.0 / _np.pi)]
    if stt is not None:
        # ucStatus 원값 = MD80 ERROR VECTOR 하위 8bit. 래치오프 원인의 단서.
        e["stt_raw"] = [int(v) for v in _np.asarray(stt, int)[jm.ch]]
    if tau_cmd_ch is not None:
        # ★τ_ff 명령. PACE 는 이게 가진 본체다 — 위치만 보면 시험의 절반이 안 보인다.
        #   측정토크와 **같은 변환**(gear_k·커플링 전치)을 거쳐야 나란히 놓고 뺄 수 있다.
        e["tau_cmd_nm"] = [round(float(v), 3) for v in jm.ch_to_tau_joint(tau_cmd_ch)]
    # ★게인은 **raw 좌표**로 낸다: kp_raw = kp_ch·gear_k²  (2026-08-21 정정)
    #   ⚠종전엔 채널값을 그대로 `kp_leg` 로 실었는데, C++ biped_deploy 는 같은 키에
    #     ×gear_k² 를 실었다 — 같은 로봇·같은 모니터인데 calf 가 **80 vs 180** 으로 찍혔다.
    #     좌표를 키 이름에 박아 두 발행자를 맞춘다.
    #   ⚠모델각(관절) 게인으로 내지 않는 이유: 발목 커플링 때문에 모델각 강성은
    #     Aᵀ·diag(kp_raw)·A 라 **비대각이 생겨** 축별 스칼라로 쓸 수 없다.
    _k2 = jm.k ** 2
    if kp is not None:
        e["kp_raw"] = [round(float(v), 1) for v in _np.asarray(kp, float)[jm.ch] * _k2]
    if kd is not None:
        e["kd_raw"] = [round(float(v), 2) for v in _np.asarray(kd, float)[jm.ch] * _k2]
    return e


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
