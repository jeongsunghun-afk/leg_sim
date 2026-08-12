"""interface/hw_interface.py — 백엔드(채널·deg) + JointMap + IMU 변환을 묶은 실기 인터페이스.

역할 분리:
  · jog/hold  : 채널 위치 명령(deg) — 각축 검증용. 컨트롤러 불필요.
  · model기반 : 컨트롤러 상태(rad·quat) 공급 + 토크(Nm) 수신 → 채널 MIT. (stand/walk, 이후 단계)
IMU RPY(deg) → quat(wxyz), gyro(deg/s) → rad/s. 접촉은 실기 힘센서 부재 시 추정(TODO).
"""
from __future__ import annotations
import numpy as np

R2D = 180.0 / np.pi
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
        self.n_write_fail = 0   # ★SHM 쓰기 실패 누적(부분실패는 위험한 방향으로 조용하다)

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
        """다리 8관절 각속도 [모델각 deg/s]. offset 은 상수라 미적용.

        ★2026-08-10 수정 — 여기만 `sign` 으로만 나눠 **gear_k 와 커플링이 빠져 있었다.**
          위치 경로(ch_to_q_joint)·컨트롤러 경로(ch_to_dq_ctrl)와 규약이 달라, calf 는
          속도가 1.5배, foot 은 커플링 항만큼 틀린 값이 나왔다.
          ⇒ JointMap.ch_to_dq_ctrl 에 위임한다(rad/s 로 주므로 deg/s 로 되돌린다).
            수식을 여기 복사하지 않는다 — 같은 실수를 GUI 한계·gen_emb_init_pose·
            calib_zero 에서 이미 세 번 했다.
        ⚠지금은 이 함수를 아무도 안 쓴다(전수 grep 0건). 그래서 드러나지 않았다.
          RL·모델기반이 붙으면 바로 쓰게 되는 자리라 미리 맞춰 둔다.
        """
        return self.jm.ch_to_dq_ctrl(self._raw.dq_dps) * R2D

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
    def write_ramped(self, q_leg_joint_deg, q_meas_joint_deg):
        """궤적 기록 — **현재 위치보다 더 바깥으로는 안 보내되, 계단은 안 만든다.** — **현재 위치보다 더 바깥으로는 안 보내되, 계단은 안 만든다.**

        ★왜 단순 클램프를 못 쓰나 (2026-08-12 실기 사고)
          늘어진 자세에서 HL_foot 모델각이 **+60°** 였다(커플링: q_foot = raw − q_calf,
          calf 가 −61° 로 처지면 raw 가 그대로여도 모델각이 +61° 로 읽힌다).
          jog 한계는 +25.2° 라 write_jog 가 명령을 거기서 잘랐고,
          **첫 틱에 34.8° 계단**이 나갔다 → kp 30 × 34.8° = 18 Nm → 발이 튕겨
          426dps 로 폭주 → E-stop.
          ⚠HomeTrajectory 도 Jogger.reset 도 바로 이걸 막으려고 시작점을 **일부러
            클램프 안 한다**(home.py·jog.py 주석). 그 방어를 하류 클램프가 두 경로 모두에서
            무효화하고 있었다 — HOME 을 고친 뒤 사용자가 "JOG 는 off 에서 바로 안 되고
            HOME 뒤에야 된다" 고 보고해 같은 뿌리임이 드러났다.

        ⇒ 한계를 **현재 측정각까지 늘려서** 적용한다:
              lo_eff = min(lo, q_meas)   ·   hi_eff = max(hi, q_meas)
          · 이미 범위 밖이면 그 자리는 허용한다(계단 없음)
          · 더 바깥으로는 못 간다(보호 유지)
          · 목표(홈)는 범위 안이므로 궤적이 **범위 쪽으로만** 데려간다 — 안전하다
        """
        qj = np.asarray(q_leg_joint_deg, float).copy()
        qm = np.asarray(q_meas_joint_deg, float)
        lo = np.minimum(self.jm.jog_min, qm)
        hi = np.maximum(self.jm.jog_max, qm)
        qj = np.clip(qj, lo, hi)
        rc = self.be.write_pos(self.jm.q_joint_to_ch(qj), self.jm.kp_ch(), self.jm.kd_ch())
        if rc not in (0, None):
            self.n_write_fail += 1
        return rc

    def write_jog(self, q_leg_joint_deg, q_meas_joint_deg=None):
        """각축 검증: **모델각**을 받아 jog 안전한계로 클램프 후 채널각으로 변환해 기록.

        ★한계를 **모델각에서** 건다. 종전엔 채널각에 걸어서 sign=−1 축의 허용범위가
          거울처럼 뒤집혔다(HR_thigh 물리한계 2.5° 초과).
        ★q_meas 를 주면 **계단 없는 클램프**를 쓴다(write_ramped). 안 주면 종전 동작.
          늘어진 자세에서 JOG 로 바로 들어가면 foot 모델각이 +60° 라 jog 한계 +25.2° 에서
          잘려 34.8° 계단이 나갔다 — HOME 과 같은 사고다(2026-08-12).
        """
        if q_meas_joint_deg is not None:
            return self.write_ramped(q_leg_joint_deg, q_meas_joint_deg)
        qj = self.jm.clamp_jog_joint(q_leg_joint_deg)
        rc = self.be.write_pos(self.jm.q_joint_to_ch(qj), self.jm.kp_ch(), self.jm.kd_ch())
        if rc not in (0, None):
            self.n_write_fail += 1
        return rc

    def write_hold(self, q_leg_joint_deg):
        """현재자세 홀드: **모델각** 입력, 관절한계 클램프 후 채널각으로 변환."""
        qj = self.jm.clamp_joint(q_leg_joint_deg)
        rc = self.be.write_pos(self.jm.q_joint_to_ch(qj), self.jm.kp_ch(), self.jm.kd_ch())
        if rc not in (0, None):
            self.n_write_fail += 1
        return rc

    def write_limp(self):
        """무여자 기록 — 위치는 **현재 측정각**을 그대로 되돌려 준다(계단 방지).
        ★enable(False) 상태에서 브리지가 kp=kd=0 으로 쓰므로 위치값 자체는 무의미하지만,
          0 을 쓰면 재무장 순간 0 으로 튀는 명령이 남는다. 측정각을 유지하는 편이 안전하다."""
        z = np.zeros(self.jm.n_channel)
        rc = self.be.write_pos(self._raw.q_deg.copy(), z, z)
        # ★limp 실패는 **가장 위험한 방향으로** 조용하다 — 남은 채널이 직전 명령을
        #   그대로 유지해 계속 힘을 낸다(shm_backend.write_pos 주석 참조).
        if rc not in (0, None):
            self.n_write_fail += 1
            if self.n_write_fail in (1, 10, 100) or self.n_write_fail % 500 == 0:
                print(f"[hw] ⚠⚠ limp 기록 실패 {self.n_write_fail}회 — **일부 축이 안 풀렸을 수 있다**. "
                      f"드라이버가 명령을 거부하는 상태다. 모터 전원 재투입을 검토할 것.",
                      flush=True)
        return rc

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
