#pragma once
// ★sim↔real 경계. 컨트롤러는 이 인터페이스 뒤(실제 HW인지 MuJoCo인지) 모른다.
//   Unitree식: LowState(센서) 읽고 LowCmd(임피던스 명령) 쓴다.
#include "common/types.hpp"

namespace qc {

// 센서 원값 — HAL(sim/real)이 채운다.
struct LowState {
  double time = 0;
  VectorXd q, dq, tau_est;          // 관절 위치/속도/추정토크 (nu)
  Vector4d imu_quat  {1, 0, 0, 0};  // 몸통 자세 wxyz
  Vector3d imu_gyro  {0, 0, 0};     // body 각속도
  Vector3d imu_acc   {0, 0, 0};     // world 선가속(KF 예측; sim=qacc, real_hal=R·f_body+g)
  Vector4d foot_force{0, 0, 0, 0};  // 발 접촉 지시자(sim=0/1 via dist<0.002 · real=수직력). 임계 >0.5
};

// 액추에이터 명령 — 컨트롤러가 채운다. 실행: tau = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff.
struct LowCmd {
  VectorXd q_des, dq_des, kp, kd, tau_ff;   // (nu)
};

// MuJoCo(sim) 또는 모터드라이버+IMU(real)가 구현.
class RobotInterface {
 public:
  virtual ~RobotInterface() = default;
  virtual int    nu() const = 0;             // 액추에이터 수
  virtual double dt() const = 0;             // 제어주기[s]
  virtual bool   read(LowState&) = 0;        // 센서 읽기
  virtual bool   write(const LowCmd&) = 0;   // 명령 쓰기(+한 스텝 진행: sim)
};

}  // namespace qc
