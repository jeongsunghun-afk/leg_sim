#pragma once
// real HAL(실기, Pi 전용) — RGA RobotSharedMem(Gait) 백엔드. `RobotInterface` 구현.
//   ★참조: ~/문서/jsh/RobotTestGait (실제 SHM API·RT골격) + ★biped/emb/pace/RESULTS.md (제어법칙·단위·필드 실측확정).
//   read():  RobotMemGait_GetMotorStatus16 + GetIMU (관절축 deg) → LowState(rad).  write(): LowCmd → MotGeneral_t → SetMotorCommand16.
//   ★컨트롤러·estimator 불변 — sim(MujocoHal) 대비 read/write만 실센서/모터로 교체(이관 3단계).
//   ※RobotSharedMem.h는 Pi에만 존재 → 데스크톱 빌드는 자동 skip(아래 가드). robot_main도 이 가드로 Pi에서만.
//
// ★★PACE 실측 확정(2026-08-05): τ=fGainKp·(fPos−q)[rad]+fGainKd·(fVel−q̇)[rad/s]+fTorque(MIT)·torque_frame=joint(관절축)·단위 deg.
//   MotGeneral_t: fAccel=1.0(미사용)·fCurrent=fTorque(중복)·float16 전필드.
// ★★★배포 결정 = 해결(2026-08-05 실기): **순수토크 가능**(드라이버 Kp=Kd=0·fTorque 수용, 0.45Nm서 ~1%). → A 직결(fGainKp=fGainKd=0·fTorque=tau_ff).
//   ⚠2차 확인: 고토크·동적 토크추종(0.45Nm=저토크·마찰floor 근처).
// ★확정 SHM API(RobotTestGait): read=RobotMemGait_{IsUpdatedMotorStatus16,GetMotorStatus16,IsUpdatedIMU,GetIMU} · write=RobotMemGait_SetMotorCommand16(ch별).
//   IMU=m_fIMUStt_Body[LEN_OF_IMU_DATA]: [IDX_OF_IMU_ARPY]=자세 RPY(deg)·[IDX_OF_IMU_ACCL]=가속. ★gyro 인덱스·RPY 순서/부호=실기 확정 TODO.
#include "hal/robot_interface.hpp"
#include <vector>

#if defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h")
#include "/usr/include/RobotSharedMem.h"    // MotGeneral_t·MotorParam16_t·float16·RobotMemGait_*·IDX_OF_IMU_*·LEN_OF_IMU_DATA
#include <cmath>
#include <algorithm>

namespace qc {

// 관절맵/한계 — config(config/joint_map_17dof.yaml). chan=Gait SHM 채널, sign/zero=MJCF관절↔실모터 부호·오프셋(★축별 JOG 실측 TODO).
struct GaitJointCfg { int chan; int sign; double zero_deg, min_deg, max_deg, vel_max_dps; };

class RealHal : public RobotInterface {
  int nu_; double dt_;
  std::vector<GaitJointCfg> jc_;                 // nu_개 (config)
  std::vector<MotGeneral_t> cmd_;                // 채널별 명령 버퍼
  bool tx_enabled_ = false;                      // ★안전: 상태 수신 확인 전엔 명령 전송 금지(robot_main이 enable)
  static constexpr double D2R = M_PI / 180.0, R2D = 180.0 / M_PI;

  // RPY(rad) → quat wxyz. ★convention(ZYX/XYZ·부호)=실 IMU 확정 TODO. 여기선 ZYX(yaw-pitch-roll) 표준.
  static void rpy2quat(double r, double p, double y, Vector4d& q) {
    double cr=std::cos(r*0.5), sr=std::sin(r*0.5), cp=std::cos(p*0.5), sp=std::sin(p*0.5), cy=std::cos(y*0.5), sy=std::sin(y*0.5);
    q[0]=cr*cp*cy+sr*sp*sy; q[1]=sr*cp*cy-cr*sp*sy; q[2]=cr*sp*cy+sr*cp*sy; q[3]=cr*cp*sy-sr*sp*cy;
  }
 public:
  RealHal(int nu, double dt, std::vector<GaitJointCfg> jc)
      : nu_(nu), dt_(dt), jc_(std::move(jc)), cmd_(nu) {}
  int    nu() const override { return nu_; }
  double dt() const override { return dt_; }
  void   enable(bool on) { tx_enabled_ = on; }   // robot_main: 상태 N프레임 수신 후 true(=RobotTestGait 안전패턴). off=limp

  bool read(LowState& s) override {
    if (RobotMemGait_IsUpdatedMotorStatus16() != 1) return false;   // ★fresh 상태 없음 → 안전정지(호출부가 hold/estop)
    s.q.resize(nu_); s.dq.resize(nu_); s.tau_est.resize(nu_);
    MotorParam16_t st;
    for (int i = 0; i < nu_; ++i) {
      const auto& j = jc_[i];
      if (RobotMemGait_GetMotorStatus16(&st, j.chan) != ENUM_RESULT_SUCCESS) return false;
      s.q[i]       = j.sign * ((double)st.fPosition - j.zero_deg) * D2R;   // 관절축 deg → MJCF rad(부호·오프셋)
      s.dq[i]      = j.sign *  (double)st.fVelocity * D2R;
      s.tau_est[i] = j.sign *  (double)st.fTorque;                          // ★PACE: 드라이버 산출토크(독립 힘측정 아님)
    }
    // IMU: RPY(deg)+accel. ★gyro 인덱스·RPY convention 실기 확정 TODO.
    if (RobotMemGait_IsUpdatedIMU() == 1) {
      float imu[LEN_OF_IMU_DATA] = {0};
      if (RobotMemGait_GetIMU(&imu[0], IDX_OF_IMU_ForeC_START, LEN_OF_IMU_DATA) == ENUM_RESULT_SUCCESS) {
        rpy2quat(imu[IDX_OF_IMU_ARPY+0]*D2R, imu[IDX_OF_IMU_ARPY+1]*D2R, imu[IDX_OF_IMU_ARPY+2]*D2R, s.imu_quat);
        for (int a = 0; a < 3; ++a) s.imu_acc[a] = imu[IDX_OF_IMU_ACCL+a];   // ★world/body 프레임·중력보정=실기 확정 TODO(KF는 world 기대)
        // TODO(Pi): s.imu_gyro = imu[IDX_OF_IMU_GYRO+0..2]*D2R (gyro 인덱스 SDK서 확인). 없으면 RPY 수치미분.
      }
    }
    // TODO(Pi): s.foot_force = 발 힘센서/추정(임계 >0.5). 없으면 관절토크·기구학 기반 접촉추정.
    return true;
  }

  bool write(const LowCmd& c) override {
    for (int i = 0; i < nu_; ++i) {
      const auto& j = jc_[i];
      double q_des_deg = j.zero_deg + j.sign * c.q_des[i] * R2D;               // MJCF rad → 관절축 deg
      q_des_deg = std::min(j.max_deg, std::max(j.min_deg, q_des_deg));         // ★한계 clip(안전)
      MotGeneral_t& m = cmd_[i];                                              // PACE 확정 필드 매핑(순수토크: Kp=Kd=0)
      m.ucDevID  = (unsigned char)j.chan; m.ucMode = 1; m.ucCommand = 0;
      m.fPosition = (float16) q_des_deg;                                       // 관절축 deg
      m.fVelocity = (float16)( j.sign * c.dq_des[i] * R2D );                   // deg/s
      m.fTorque   = (float16)( j.sign * c.tau_ff[i] );                         // 관절 토크(WBIC) — 순수토크 배포경로
      m.fGainKp   = (float16) c.kp[i];   // ★A=0(순수토크). 임피던스 시 driver단위 환산(biped 1≈1.0027 Nm/rad, quad TBD)
      m.fGainKd   = (float16) c.kd[i];   // ★A=0
      m.fGainKi   = (float16) 0.0f;
      m.fAccelrationOrTemperture = (float16) 1.0f;                             // PACE: 미사용 상수 1.0
      if (tx_enabled_) RobotMemGait_SetMotorCommand16((MotorParam16_t*)&m, j.chan);   // ★안전게이트 후 전송
    }
    return true;
  }
};

}  // namespace qc
#endif   // RobotSharedMem.h 존재(Pi)
