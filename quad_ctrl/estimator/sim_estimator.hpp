#pragma once
// sim용 추정기: MuJoCo가 넘긴 값이 이미 ground truth라 거의 복사.
//   real 배포 시 ekf_estimator(IMU + leg-odometry + contact)로 교체 — 인터페이스 동일.
#include "estimator/state.hpp"

namespace qc {

class SimEstimator : public Estimator {
  State s_;
 public:
  explicit SimEstimator(int nu) { s_.q.setZero(nu); s_.dq.setZero(nu); }

  void update(const LowState& ls) override {
    s_.time = ls.time;
    s_.q = ls.q; s_.dq = ls.dq;
    s_.base_quat = ls.imu_quat;
    s_.base_ang_vel = ls.imu_gyro;
    for (int i = 0; i < NLEG; i++) s_.contact[i] = ls.foot_force[i] > 0.5 ? 1.0 : 0.0;  // 접촉 지시자(sim 0/1)
    // TODO(sim): base_pos/lin_vel/com은 MuJoCo ground truth를 HAL이 함께 넘기거나(확장 LowState)
    //            여기서 FK로 계산. real에서는 EKF가 채운다.
  }

  const State& state() const override { return s_; }
};

}  // namespace qc
