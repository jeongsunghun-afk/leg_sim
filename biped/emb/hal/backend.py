"""hal/backend.py — SHM 채널 단위(degree) 백엔드 경계. quad_ctrl RobotInterface 대응.

컨트롤러/관절매핑 위쪽은 이 인터페이스만 보고 동작 → 실HW(ShmBackend)인지 데스크톱(MockBackend)인지 모른다.
단위: 위치 deg · 속도 deg/s · 토크 Nm · IMU RPY deg · gyro deg/s(또는 config imu_deg 로 rad 처리).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

UPT_POS, UPT_VEL, UPT_TOR, UPT_CUR, UPT_IMU = 0x01, 0x02, 0x04, 0x08, 0x10


@dataclass
class RawState:
    """Gait 채널 단위 센서 스냅샷(매핑 전)."""
    q_deg:   np.ndarray                                # (n_channel,)
    dq_dps:  np.ndarray                                # (n_channel,)
    tau_nm:  np.ndarray                                # (n_channel,)
    imu_rpy_deg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_acc:     np.ndarray = field(default_factory=lambda: np.zeros(3))
    imu_gyro:    np.ndarray = field(default_factory=lambda: np.zeros(3))
    connected:   np.ndarray = field(default_factory=lambda: np.zeros(0, bool))  # 채널별 통신연결
    status:      np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # 채널별 ucStatus(임베디드 보고)
    updated: int = 0                                   # 갱신 비트마스크(UPT_*)


class Backend(ABC):
    """SHM 플랜트 경계. 모든 배열은 길이 n_channel(Gait 전체 채널)."""

    def __init__(self, n_channel: int):
        self.n_channel = int(n_channel)

    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def read(self) -> RawState: ...

    @abstractmethod
    def write_pos(self, q_des_deg: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        """위치+임피던스(jog/hold)."""

    @abstractmethod
    def write_mit(self, q_des_deg, dq_des_dps, tau_ff_nm, kp, kd) -> None:
        """풀 MIT(모델기반 tau_ff)."""

    @abstractmethod
    def enable(self, on: bool) -> None:
        """모터 전원. off=limp(토크 0)."""

    def close(self) -> None:
        pass
