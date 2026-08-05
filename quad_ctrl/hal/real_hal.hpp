#pragma once
// real HAL(실기, Pi 전용) — RGA RobotSharedMem(Gait) 백엔드. `RobotInterface` 구현.
//   ★참조: ~/문서/jsh/RobotTestGait (SHM/RT골격) + ★biped/emb/pace/RESULTS.md (실측으로 제어법칙·단위·필드 확정).
//   read():  SHM Pos/Vel/Tor/IMU(관절축 deg) → LowState(rad).   write(): LowCmd(rad) → MotGeneral_t(deg·float16) → SHM.
//   ★컨트롤러·estimator 불변 — sim(MujocoHal) 대비 read/write만 실센서/모터로 교체(이관 3단계).
//   ※RobotSharedMem.h는 Pi에만 존재 → 데스크톱 빌드는 자동 skip(아래 가드). robot_main도 이 가드로 Pi에서만.
//
// ★★PACE 실측 확정(2026-08-05, biped/emb/pace — 같은 RobotSharedMem/Gait 인터페이스):
//   · 드라이버 제어법칙 = τ = fGainKp·(fPosition−q)[rad] + fGainKd·(fVelocity−q̇)[rad/s] + fTorque  (MIT 임피던스, 예측 vs 실측 2%)
//   · torque_frame = **joint(관절/출력축)** — fPosition/fTorque 전부 감속 후 관절축(36° 명령→36° 출력 회전 확인)
//   · 단위 = 명령/피드백 **deg**, fGainKp 1단위 = 0.0175 Nm/deg ≈ 1.0027 Nm/rad (★biped 실측, quad는 재확인 필요)
//   · MotGeneral_t: fAccelrationOrTemperture=1.0(미사용 상수)·fCurrent=fTorque(중복)·float16 전필드
//
// ★★★배포 결정 = 해결(2026-08-05, ★실기 검증): **순수토크 가능** — 드라이버가 Kp=Kd=0·fTorque 명령 수용(0.45Nm서 ~1% 오차).
//   → 컨트롤러 A(WBIC 순수토크 tau_ff)를 real_hal **직결**: 아래 write의 fGainKp=fGainKd=0·fTorque=tau_ff 가 곧 배포 경로. 재정식화 불요.
//   ⚠2차 확인 권장: 0.45Nm=저토크(마찰 floor 근처). 배포 고토크(수십 Nm)·동적 토크추종 정확도도 실기 확인.
#include "hal/robot_interface.hpp"
#include <vector>

#if defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h")
#include "/usr/include/RobotSharedMem.h"    // MotGeneral_t·float16·WRITE_SHMMotorCommand·UPT_* (RGA SDK)
#include <cmath>
#include <algorithm>

namespace qc {

// 관절맵/한계 — RobotTestGait `m_fGaitMotorRange` 형식(deg). config(yaml)에서 로드.
//   chan = Gait SHM 채널, sign/zero = MJCF 관절 ↔ 실모터 부호·오프셋. ★quad 17-DOF는 실기서 축별 JOG로 실측 확정(TODO).
struct GaitJointCfg { int chan; int sign; double zero_deg, min_deg, max_deg, vel_max_dps; };

class RealHal : public RobotInterface {
  int nu_; double dt_;
  std::vector<GaitJointCfg> jc_;                 // nu_개 (config)
  std::vector<MotGeneral_t> cmd_;                // 채널별 명령 버퍼(WRITE_SHMMotorCommand 대상)
  static constexpr double D2R = M_PI / 180.0, R2D = 180.0 / M_PI;
 public:
  RealHal(int nu, double dt, std::vector<GaitJointCfg> jc)
      : nu_(nu), dt_(dt), jc_(std::move(jc)), cmd_(nu) {}
  int    nu() const override { return nu_; }
  double dt() const override { return dt_; }

  bool read(LowState& s) override {
    s.q.resize(nu_); s.dq.resize(nu_); s.tau_est.resize(nu_);
    for (int i = 0; i < nu_; ++i) {
      const auto& j = jc_[i];
      // TODO(Pi): SHM 게터로 관절축 값 취득(update flag UPT_POS/VEL/TOR=0x01/02/04 확인 후). RobotTestGait 참조.
      //   double pos_deg = SHM_GetMotorPos(j.chan);  double vel_dps = SHM_GetMotorVel(j.chan);  double tau = SHM_GetMotorTor(j.chan);
      // ★PACE 확정 변환(관절축 deg → MJCF rad, 부호·오프셋):
      //   s.q[i]       = j.sign * (pos_deg - j.zero_deg) * D2R;
      //   s.dq[i]      = j.sign *  vel_dps               * D2R;
      //   s.tau_est[i] = j.sign *  tau;   // ★PACE: fTorque=드라이버 산출토크(독립 힘측정 아님)=명령/추정치
      (void)j;
    }
    // TODO(Pi): IMU m_fIMUStt_Body(RPY deg 또는 quat) → s.imu_quat(wxyz)·s.imu_gyro(rad/s)·s.imu_acc(world=R·f_body+g).
    //   접촉 s.foot_force = 발 힘센서/추정(임계 >0.5). RobotTestGait IMU 변환 참조.
    return true;   // 통신 두절/update flag 실패 시 false(안전정지)
  }

  bool write(const LowCmd& c) override {
    for (int i = 0; i < nu_; ++i) {
      const auto& j = jc_[i];
      double q_des_deg = j.zero_deg + j.sign * c.q_des[i] * R2D;               // MJCF rad → 관절축 deg
      q_des_deg = std::min(j.max_deg, std::max(j.min_deg, q_des_deg));         // ★한계 clip(안전)
      MotGeneral_t& m = cmd_[i];                                              // PACE 확정 필드 매핑
      m.ucDevID  = (unsigned char)j.chan; m.ucMode = 1; m.ucCommand = 0;
      m.fPosition = (float16) q_des_deg;                                       // 관절축 deg
      m.fVelocity = (float16)( j.sign * c.dq_des[i] * R2D );                   // deg/s
      m.fTorque   = (float16)( j.sign * c.tau_ff[i] );                         // 관절 토크(WBIC)
      m.fGainKp   = (float16) c.kp[i];   // ★A=순수토크라 0. 임피던스 시 driver단위 환산 필요(biped 1≈1.0027 Nm/rad, quad TBD)
      m.fGainKd   = (float16) c.kd[i];   // ★A=0
      m.fGainKi   = (float16) 0.0f;
      m.fAccelrationOrTemperture = (float16) 1.0f;                             // PACE: 미사용 상수 1.0
    }
    // TODO(Pi): WRITE_SHMMotorCommand(cmd_.data(), nu_);   // 실제 SHM 전송(transmit enable 시). 정확한 API=RobotTestGait.
    return true;
  }

  void enable(bool /*on*/) { /* TODO(Pi): 모터 전원/enable. off=limp(fTorque=0·Kp=0). RobotTestGait m_ucGaitTransmitEna_MotorCommand */ }
};

}  // namespace qc
#endif   // RobotSharedMem.h 존재(Pi)
