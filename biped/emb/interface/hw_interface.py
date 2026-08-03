"""interface/hw_interface.py — 백엔드(채널·deg) + JointMap + IMU 변환을 묶은 실기 인터페이스.

역할 분리:
  · jog/hold  : 채널 위치 명령(deg) — 각축 검증용. 컨트롤러 불필요.
  · model기반 : 컨트롤러 상태(rad·quat) 공급 + 토크(Nm) 수신 → 채널 MIT. (stand/walk, 이후 단계)
IMU RPY(deg) → quat(wxyz), gyro(deg/s) → rad/s. 접촉은 실기 힘센서 부재 시 추정(TODO).
"""
from __future__ import annotations
import numpy as np
from joint_map import JointMap, D2R


def rpy_to_quat(roll, pitch, yaw) -> np.ndarray:
    """ZYX(yaw-pitch-roll) → quat wxyz."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return np.array([cr*cp*cy + sr*sp*sy,
                     sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy,
                     cr*cp*sy - sr*sp*cy])


class HwInterface:
    def __init__(self, backend, jmap: JointMap, imu_deg: bool = True):
        self.be = backend
        self.jm = jmap
        self.imu_deg = imu_deg
        self._raw = None

    def init(self):
        self.be.init()

    # ── 센서 ────────────────────────────────────────────────────────────────
    def read(self):
        self._raw = self.be.read()
        return self._raw

    def q_leg_deg(self) -> np.ndarray:
        """다리 8관절 현재각[deg] (GUI 피드백·jog 기준)."""
        return self._raw.q_deg[self.jm.ch].copy()

    def ctrl_state(self):
        """모델기반용 상태: (q_rad[8], dq_rad[8], quat_wxyz[4], gyro_rad[3], acc[3], contact[2])."""
        r = self._raw
        q = self.jm.ch_to_q_ctrl(r.q_deg)
        dq = self.jm.ch_to_dq_ctrl(r.dq_dps)
        rpy = r.imu_rpy_deg * (D2R if self.imu_deg else 1.0)
        quat = rpy_to_quat(rpy[0], rpy[1], rpy[2])
        gyro = r.imu_gyro * (D2R if self.imu_deg else 1.0)
        contact = self._estimate_contact()
        return q, dq, quat, gyro, r.imu_acc.copy(), contact

    def _estimate_contact(self) -> np.ndarray:
        # ★실기 발 힘센서 부재 → 발목 토크 임계 기반 추정(TODO: 실측 임계 캘리브레이션).
        #   jog 단계선 미사용. stand 도입 시 확정.
        return np.array([True, True])

    # ── 명령 ────────────────────────────────────────────────────────────────
    #   미배선 모터는 통신이 없어 명령이 무효 → 별도 enable 게이팅 없이 전 채널 명령(임베디드가 흡수).
    def write_jog(self, q_ch_deg):
        """각축 검증: jog 안전한계 클램프 + 저게인 위치."""
        q = self.jm.clamp_jog(q_ch_deg)
        self.be.write_pos(q, self.jm.kp_ch(), self.jm.kd_ch())

    def write_hold(self, q_ch_deg):
        """현재자세 홀드(관절한계 클램프)."""
        q = self.jm.clamp_ch(q_ch_deg)
        self.be.write_pos(q, self.jm.kp_ch(), self.jm.kd_ch())

    def write_torque(self, q_ctrl_rad, dq_ctrl_rad, tau_ctrl_nm, kp_leg=0.0, kd_leg=0.0):
        """모델기반: 컨트롤러 토크(Nm) → 채널 MIT. kp/kd=0 = 순수 토크."""
        q = self.jm.q_ctrl_to_ch(q_ctrl_rad)
        dq = self.jm.dq_ctrl_to_ch(dq_ctrl_rad)
        tau = self.jm.tau_ctrl_to_ch(tau_ctrl_nm)
        self.be.write_mit(q, dq, tau, self.jm.kp_ch(kp_leg), self.jm.kd_ch(kd_leg))

    def enable(self, on: bool):
        self.be.enable(on)

    def close(self):
        self.be.close()
