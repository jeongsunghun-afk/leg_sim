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
        """다리 8관절 현재각 [**모델각 deg**] — GUI 표시·jog·home·hold 의 기준 단위.

        ★2026-08-10: 종전엔 채널각(드라이버 채널각)을 그대로 반환했다. 그래서
          sign 이 안 먹어 GUI/뷰어가 모델과 반대로 보였다.
          ⇒ 여기서 모델각으로 바꿔 **위 계층 전체를 한 단위로** 통일한다.
        ⚠calf·foot 은 드라이버 감속비 오설정으로 채널각이 실제의 1.5/1.2 배다.
          소프트 보정을 넣지 않기로 했으므로(config 사유 참조) 그 두 축의 **크기**는 부정확하다.
        """
        return self.jm.ch_to_q_joint(self._raw.q_deg)

    def dq_leg_dps(self) -> np.ndarray:
        """다리 8관절 각속도 [모델각 deg/s]. offset 은 상수라 미적용."""
        return np.asarray(self._raw.dq_dps, float)[self.jm.ch] / self.jm.sign

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
    def write_jog(self, q_leg_joint_deg):
        """각축 검증: **모델각**을 받아 jog 안전한계로 클램프 후 채널각으로 변환해 기록.

        ★한계를 **모델각에서** 건다. 종전엔 채널각에 걸어서 sign=−1 축의 허용범위가
          거울처럼 뒤집혔다(HR_thigh 물리한계 2.5° 초과).
        """
        qj = self.jm.clamp_jog_joint(q_leg_joint_deg)
        self.be.write_pos(self.jm.q_joint_to_ch(qj), self.jm.kp_ch(), self.jm.kd_ch())

    def write_hold(self, q_leg_joint_deg):
        """현재자세 홀드: **모델각** 입력, 관절한계 클램프 후 채널각으로 변환."""
        qj = self.jm.clamp_joint(q_leg_joint_deg)
        self.be.write_pos(self.jm.q_joint_to_ch(qj), self.jm.kp_ch(), self.jm.kd_ch())

    def write_limp(self):
        """무여자 기록 — 위치는 **현재 측정각**을 그대로 되돌려 준다(계단 방지).
        ★enable(False) 상태에서 브리지가 kp=kd=0 으로 쓰므로 위치값 자체는 무의미하지만,
          0 을 쓰면 재무장 순간 0 으로 튀는 명령이 남는다. 측정각을 유지하는 편이 안전하다."""
        z = np.zeros(self.jm.n_channel)
        self.be.write_pos(self._raw.q_deg.copy(), z, z)

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
