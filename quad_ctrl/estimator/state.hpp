#pragma once
// ★컨트롤러의 유일한 입력. sim/real 무관하게 동일 → WBIC/MPC는 State만 보고 동작.
//   Estimator가 LowState(센서) → State(추정) 변환. sim=ground truth, real=EKF.
#include "common/types.hpp"
#include "hal/robot_interface.hpp"

namespace qc {

struct State {
  double time = 0;
  // 몸통(floating base)
  Vector3d base_pos    {0, 0, 0};   // world
  Vector4d base_quat   {1, 0, 0, 0};// wxyz
  Vector3d base_lin_vel{0, 0, 0};   // world
  Vector3d base_ang_vel{0, 0, 0};   // body
  // 관절
  VectorXd q, dq;                   // (nu)
  // 접촉·질량중심
  Vector4d contact{0, 0, 0, 0};     // 발 접촉(0/1)
  Vector3d com{0, 0, 0}, com_vel{0, 0, 0};
};

class Estimator {
 public:
  virtual ~Estimator() = default;
  virtual void update(const LowState&) = 0;   // 센서 → State
  virtual const State& state() const = 0;
};

}  // namespace qc
