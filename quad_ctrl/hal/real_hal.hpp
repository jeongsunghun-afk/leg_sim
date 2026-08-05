#pragma once
// real HAL(실기, Pi 전용) — RGA RobotSharedMem(Gait) 백엔드. `RobotInterface` 구현.
//   ★참조: ~/문서/jsh/RobotTestGait (src/main.cpp: MotGeneral_t·SHM read/write·RT timer·관절 range).
//   read():  SHM Pos/Vel/Tor/IMU(deg) → LowState(rad).   write(): LowCmd(rad) → MotGeneral_t(deg·float16) → SHM.
//   ★컨트롤러·estimator 불변 — sim(MujocoHal) 대비 read/write만 실센서/모터로 교체(이관 3단계).
//   ※RobotSharedMem.h는 Pi에만 존재 → 데스크톱 빌드는 자동 skip(아래 가드). robot_main도 이 가드로 Pi에서만.
#include "hal/robot_interface.hpp"
#include <vector>

#if defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h")
#include "/usr/include/RobotSharedMem.h"    // MotGeneral_t·float16·WRITE_SHMMotorCommand·UPT_* (RGA SDK)
#include <cmath>

namespace qc {

// 관절맵/한계 — RobotTestGait `m_fGaitMotorRange` 형식(deg). config(yaml)에서 로드.
//   chan = Gait SHM 채널, sign/zero = MJCF 관절 ↔ 실모터 부호·오프셋(★실측 확정, biped/emb 절차).
struct GaitJointCfg { int chan; int sign; double zero_deg, min_deg, max_deg, vel_max_dps; };

class RealHal : public RobotInterface {
  int nu_; double dt_;
  std::vector<GaitJointCfg> jc_;                 // nu_개 (config)
  static constexpr double D2R = M_PI / 180.0, R2D = 180.0 / M_PI;
 public:
  RealHal(int nu, double dt, std::vector<GaitJointCfg> jc) : nu_(nu), dt_(dt), jc_(std::move(jc)) {}
  int    nu() const override { return nu_; }
  double dt() const override { return dt_; }

  bool read(LowState& s) override {
    // TODO(Pi): SHM에서 Pos/Vel/Tor/IMU 읽기(update flag UPT_POS/VEL/TOR/IMU=0x01~0x10 확인). RobotTestGait 참조.
    s.q.resize(nu_); s.dq.resize(nu_); s.tau_est.resize(nu_);
    for (int i = 0; i < nu_; ++i) {
      const auto& j = jc_[i];
      // double pos_deg = SHM_GetMotorPos(j.chan);  double vel_dps = SHM_GetMotorVel(j.chan);  double tau = SHM_GetMotorTor(j.chan);
      // s.q[i]      = j.sign * (pos_deg - j.zero_deg) * D2R;   // 실모터 deg → MJCF rad(부호·오프셋)
      // s.dq[i]     = j.sign *  vel_dps               * D2R;
      // s.tau_est[i]= j.sign *  tau;
    }
    // IMU: m_fIMUStt_Body(RPY deg 또는 quat) → s.imu_quat(wxyz)·s.imu_gyro(rad/s). RobotTestGait IMU 변환 참조.
    return true;   // 통신 두절/update flag 실패 시 false(안전정지)
  }

  bool write(const LowCmd& c) override {
    // TODO(Pi): LowCmd(rad) → MotGeneral_t(deg·float16) 채널별 → WRITE_SHMMotorCommand(transmit enable 시).
    for (int i = 0; i < nu_; ++i) {
      const auto& j = jc_[i];
      double q_des_deg = j.zero_deg + j.sign * c.q_des[i] * R2D;             // MJCF rad → 실모터 deg
      q_des_deg = std::min(j.max_deg, std::max(j.min_deg, q_des_deg));       // ★한계 clip(안전)
      // MotGeneral_t m{}; m.ucDevID = j.chan; m.ucMode = 1; m.ucCommand = 0;
      // m.fPosition = (float16) q_des_deg;
      // m.fVelocity = (float16)( j.sign * c.dq_des[i] * R2D );
      // m.fTorque   = (float16)( j.sign * c.tau_ff[i] );                    // WBIC tau_ff
      // m.fGainKp   = (float16) c.kp[i];  m.fGainKd = (float16) c.kd[i];  m.fGainKi = (float16)0.0f;
      // cmd[j.chan] = m;
      (void)q_des_deg;
    }
    // WRITE_SHMMotorCommand(...);
    return true;
  }

  void enable(bool /*on*/) { /* TODO(Pi): 모터 전원/enable. off=limp(토크0). RobotTestGait m_ucGaitTransmitEna_MotorCommand */ }
};

}  // namespace qc
#endif   // RobotSharedMem.h 존재(Pi)
