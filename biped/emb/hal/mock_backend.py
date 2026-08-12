"""hal/mock_backend.py — SHM 없는 데스크톱/CI 백엔드. 명령을 1차 지연으로 추종해 에코.

실제 모터 대신 각 채널이 q_des 로 속도제한 램프. jog 루프·GUI·매핑을 SHM/Pi 없이 검증.
IMU는 평평(rpy=0, acc=[0,0,g], gyro=0). 실기 검증 전 인터페이스 회귀용.
★연결 시뮬: 환경변수 MOCK_CONNECTED="0,4"(채널) → 그 채널만 connected(LED 초록). 기본=전체 연결.
★enable: kp[i]==0 인 채널(=배선 off)은 움직이지 않음 → 미배선 모터 미구동 재현.
"""
from __future__ import annotations
import os
import numpy as np
from backend import Backend, RawState, UPT_POS, UPT_VEL, UPT_TOR, UPT_IMU


class MockBackend(Backend):
    def __init__(self, n_channel: int, dt: float = 0.002, max_dps: float = 60.0):
        super().__init__(n_channel)
        self.dt = dt
        self.max_step = max_dps * dt          # 채널당 1스텝 최대 이동[deg] (모터 응답 모사)
        self.q = np.zeros(n_channel)          # 현재 위치[deg]
        self.dq = np.zeros(n_channel)
        self.q_des = np.zeros(n_channel)
        self.kp = np.zeros(n_channel)         # 마지막 명령 kp(=enable 여부: >0 이면 구동)
        self.on = False
        # 연결 시뮬: 기본 전체, MOCK_CONNECTED 로 부분연결. MOCK_FAULT 로 그 채널 ucStatus=1(에러).
        self.connected = self._chan_set(os.environ.get("MOCK_CONNECTED"), default_all=True)
        fault = self._chan_set(os.environ.get("MOCK_FAULT"), default_all=False)
        self.status = fault.astype(int)                # ucStatus 시뮬(0=정상, 1=에러)

    def _chan_set(self, env, default_all):
        out = np.zeros(self.n_channel, bool)
        if env is None or env.strip() == "":
            out[:] = default_all
        else:
            for tok in env.split(","):
                tok = tok.strip()
                if tok.isdigit() and int(tok) < self.n_channel:
                    out[int(tok)] = True
        return out

    def init(self) -> None:
        print(f"[MockBackend] n_channel={self.n_channel} dt={self.dt} "
              f"connected={list(np.where(self.connected)[0])} "
              f"fault={list(np.where(self.status)[0])} (SHM 없음)")

    def read(self) -> RawState:
        if self.on:                            # 연결·kp>0·에러없음 채널만 명령 위치로 램프
            drive = (self.kp > 0) & self.connected & (self.status == 0)
            err = np.where(drive, self.q_des - self.q, 0.0)
            step = np.clip(err, -self.max_step, self.max_step)
            self.q += step
            self.dq = step / self.dt
        else:
            self.dq[:] = 0.0
        acc = np.array([0.0, 0.0, 9.81])
        return RawState(q_deg=self.q.copy(), dq_dps=self.dq.copy(),
                        tau_nm=np.zeros(self.n_channel), imu_acc=acc,
                        connected=self.connected.copy(), status=self.status.copy(),
                        updated=UPT_POS | UPT_VEL | UPT_TOR | UPT_IMU)

    def write_pos(self, q_des_deg, kp, kd) -> int:
        if self.on:
            self.q_des = np.asarray(q_des_deg, float)[: self.n_channel].copy()
            self.kp = np.asarray(kp, float)[: self.n_channel].copy()

    def write_mit(self, q_des_deg, dq_des_dps, tau_ff_nm, kp, kd) -> None:
        if self.on:
            self.q_des = np.asarray(q_des_deg, float)[: self.n_channel].copy()
            self.kp = np.asarray(kp, float)[: self.n_channel].copy()

    def enable(self, on: bool) -> None:
        self.on = bool(on)
