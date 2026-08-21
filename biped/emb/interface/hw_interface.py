"""interface/hw_interface.py — 백엔드(채널·deg) + JointMap + IMU 변환을 묶은 실기 인터페이스.

역할 분리:
  · jog/hold  : 채널 위치 명령(deg) — 각축 검증용. 컨트롤러 불필요.
  · model기반 : 컨트롤러 상태(rad·quat) 공급 + 토크(Nm) 수신 → 채널 MIT. (stand/walk, 이후 단계)
IMU RPY(deg) → quat(wxyz), gyro(deg/s) → rad/s. 접촉은 실기 힘센서 부재 시 추정(TODO).
"""
from __future__ import annotations
import os
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
        # ★★위치모드 게인 배율 (2026-08-20) — C++ biped_deploy 와 **같은 env 이름**.
        #   두 제어기가 같은 옵션을 갖게 한다. 한쪽에만 있으면 "올렸는데 안 변한다" 가 된다
        #   (실제로 POS_KP_SCALE=5 를 이 앱에 줬는데 무시돼 그 혼동이 있었다).
        #   ⚠kp 를 올리면 둘이 같이 움직인다:
        #     ① 토크트립이 예민해진다 — kp_ch 1 당 0.0175 Nm/deg. 500 이면 1.71°에서 트립.
        #     ② 감쇠비 ζ ∝ kd/√kp — kp 만 올리면 떨림이 커진다.
        #   ⇒ kd 는 **√배율**을 기본으로 같이 올린다(ζ 보존). POS_KD_SCALE 로 따로 줄 수 있다.
        #   적용 대상은 **위치모드 쓰기 3곳**(ramped·jog·hold)뿐이다.
        self.POS_KP = float(os.environ.get("POS_KP_SCALE", "1.0"))
        self.POS_KD = float(os.environ.get("POS_KD_SCALE", str(max(1e-9, self.POS_KP) ** 0.5)))
        if self.POS_KP != 1.0 or self.POS_KD != 1.0:
            print(f"[hw] 위치게인 배율 kp×{self.POS_KP:.2f} · kd×{self.POS_KD:.2f} "
                  f"— hip kp {self.jm.kp_leg[0]:.0f}→{self.jm.kp_leg[0]*self.POS_KP:.0f} "
                  f"· hip {self.jm.kp_leg[0]*self.POS_KP*0.0175:.1f} Nm/deg "
                  f"(τ_trip ÷ 이 값 = 트립까지 각도)", flush=True)
        self.imu_deg = imu_deg
        self._raw = None
        self.n_write_fail = 0   # ★SHM 쓰기 실패 누적(부분실패는 위험한 방향으로 조용하다)
        # ★마지막으로 **실제 나간** 명령을 모델각/관절토크로 기록한다 (2026-08-13).
        #   왜 여기냐: 상위(app)가 만든 목표는 클램프·램프를 거치기 **전** 값이라,
        #   그걸 그대로 표시하면 "명령대로 안 따라온다" 는 오진을 부른다. 실제로 나간 것과
        #   측정을 나란히 놔야 추종오차가 의미를 가진다.
        #   ⚠채널각이 아니라 **모델각**으로 둔다 — 측정(q_leg_deg)과 같은 단위여야 뺄 수 있다.
        n = self.jm.n_leg
        self.cmd_q_deg   = np.zeros(n)     # 위치명령[모델각 deg]
        self.cmd_dq_dps  = np.zeros(n)     # 속도명령[모델각 deg/s] — 위치모드에선 0
        self.cmd_tau_nm  = np.zeros(n)     # 토크 피드포워드[관절 Nm] — 위치모드에선 0
        self.cmd_kp      = np.zeros(n)     # 그때 쓴 게인(모드에 따라 0=limp)
        self.cmd_kd      = np.zeros(n)

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

    def tau_leg_nm(self) -> np.ndarray:
        """다리 8관절 측정토크 [**관절 Nm**]. 드라이버 보고토크를 관절축으로 되돌린 값.

        ★`ch_to_tau_joint` 에 위임한다 — gear_k 로 곱하고 커플링을 **전치로** 푼다.
          여기 수식을 복사하면 안 된다(같은 실수를 GUI 한계·gen_emb_init_pose 에서 이미 했다).
        ⚠보고토크의 **절대 스케일 α 는 미검증**이다(fCurrent 가 fTorque 복제라 독립 검증
          수단이 없다). 추세·좌우대조·명령대비 편차를 보는 용도로 쓸 것.
        """
        return self.jm.ch_to_tau_joint(self._raw.tau_nm)

    # ── 명령 기록 (모니터링용) ────────────────────────────────────────────
    def _log_cmd(self, q_deg=None, dq_dps=None, tau_nm=None, kp=None, kd=None):
        """실제로 나간 명령을 모델각/관절토크로 남긴다. 안 준 항목은 0 으로 둔다."""
        n = self.jm.n_leg
        self.cmd_q_deg  = np.asarray(q_deg,  float).copy() if q_deg  is not None else np.zeros(n)
        self.cmd_dq_dps = np.asarray(dq_dps, float).copy() if dq_dps is not None else np.zeros(n)
        self.cmd_tau_nm = np.asarray(tau_nm, float).copy() if tau_nm is not None else np.zeros(n)
        self.cmd_kp = (np.full(n, kp, float) if np.isscalar(kp) else
                       (np.asarray(kp, float).copy() if kp is not None else self.jm.kp_leg.copy()))
        self.cmd_kd = (np.full(n, kd, float) if np.isscalar(kd) else
                       (np.asarray(kd, float).copy() if kd is not None else self.jm.kd_leg.copy()))

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
        self._log_cmd(q_deg=qj)                      # ★클램프 **후** = 실제 나간 값
        rc = self.be.write_pos(self.jm.q_joint_to_ch(qj),
                               self.jm.kp_ch(self.POS_KP), self.jm.kd_ch(self.POS_KD))
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
        self._log_cmd(q_deg=qj)
        rc = self.be.write_pos(self.jm.q_joint_to_ch(qj),
                               self.jm.kp_ch(self.POS_KP), self.jm.kd_ch(self.POS_KD))
        if rc not in (0, None):
            self.n_write_fail += 1
        return rc

    def write_hold(self, q_leg_joint_deg):
        """현재자세 홀드: **모델각** 입력, 관절한계 클램프 후 채널각으로 변환."""
        qj = self.jm.clamp_joint(q_leg_joint_deg)
        self._log_cmd(q_deg=qj)
        rc = self.be.write_pos(self.jm.q_joint_to_ch(qj),
                               self.jm.kp_ch(self.POS_KP), self.jm.kd_ch(self.POS_KD))
        if rc not in (0, None):
            self.n_write_fail += 1
        return rc

    def write_limp(self):
        """무여자 기록 — 위치는 **현재 측정각**을 그대로 되돌려 준다(계단 방지).
        ★enable(False) 상태에서 브리지가 kp=kd=0 으로 쓰므로 위치값 자체는 무의미하지만,
          0 을 쓰면 재무장 순간 0 으로 튀는 명령이 남는다. 측정각을 유지하는 편이 안전하다."""
        z = np.zeros(self.jm.n_channel)
        self._log_cmd(q_deg=self.jm.ch_to_q_joint(self._raw.q_deg), kp=0.0, kd=0.0)
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
        self._log_cmd(q_deg=np.asarray(q_ctrl_rad, float) * R2D,
                      dq_dps=np.asarray(dq_ctrl_rad, float) * R2D,
                      tau_nm=tau_ctrl_nm, kp=kp_leg, kd=kd_leg)
        q = self.jm.q_ctrl_to_ch(q_ctrl_rad)
        dq = self.jm.dq_ctrl_to_ch(dq_ctrl_rad)
        tau = self.jm.tau_ctrl_to_ch(tau_ctrl_nm)
        self.be.write_mit(q, dq, tau, self.jm.kp_ch(kp_leg), self.jm.kd_ch(kd_leg))

    def enable(self, on: bool):
        self.be.enable(on)

    def close(self):
        self.be.close()
