#pragma once
// 데모/구조검증용 목 HAL(외부 의존 없음, Eigen만). 실제는 mujoco_hal / real_hal.
#include "hal/robot_interface.hpp"

namespace qc {

class MockHal : public RobotInterface {
  int nu_; double dt_, t_ = 0;
  LowState ls_;
 public:
  MockHal(int nu, double dt) : nu_(nu), dt_(dt) {
    ls_.q.setZero(nu); ls_.dq.setZero(nu); ls_.tau_est.setZero(nu);
  }
  int nu() const override { return nu_; }
  double dt() const override { return dt_; }
  bool read(LowState& s) override { ls_.time = t_; s = ls_; return true; }
  bool write(const LowCmd&) override { t_ += dt_; return true; }   // 물리 없음(데모)
};

}  // namespace qc
