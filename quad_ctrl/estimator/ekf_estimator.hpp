#pragma once
// EKF 추정기(실기형) — quad/cpp `::StateEstimator`(18-state 접촉KF, IMU 가속도 융합, Cheetah3식)를 qc 경계로 정식화.
//   update(LowState) → d_est(mjData): base=KF추정(p,v) · 자세/gyro/관절=측정 · mj_forward.
//   ★컨트롤러는 d_est만 보고 계산(TrotBridge.step(cmd, ekf.d_est())). GT(d_phys) 대비 base pose만 추정으로 교체.
//   이것이 목표 아키텍처 원형: HAL.read(LowState) → Estimator.update(→d_est) → Controller.step(d_est) → HAL.write.
//   trot_sim.cpp EST_CTRL 경로를 모듈로 옮긴 것(재작성 아님). real_hal은 LowState만 실센서로 바뀌고 이 KF 그대로.
#include "hal/robot_interface.hpp"    // qc::LowState
#include "state_estimator.hpp"        // quad/cpp/src ::StateEstimator
#include "quad_control.hpp"           // ::QuadControl (m, fgid, fr)
#include <mujoco/mujoco.h>
#include <cstdlib>
#include <vector>

namespace qc {

class EkfEstimator {
  ::QuadControl& q_;
  ::StateEstimator est_;
  mjData* d_est_ = nullptr;                // 추정상태 mjData(컨트롤러가 여기서 Jacobian/MPC/WBIC 계산)
  std::vector<int>    fgid_;               // 발 geom id(접촉/기구학)
  std::vector<double> fr_;                 // 발 sphere 반경
 public:
  explicit EkfEstimator(::QuadControl& q) : q_(q) {
    d_est_ = mj_makeData(q_.m);
    fgid_ = {q_.fgid[0], q_.fgid[1], q_.fgid[2], q_.fgid[3]};
    fr_   = {q_.fr[0],   q_.fr[1],   q_.fr[2],   q_.fr[3]};
    // KF 튜닝(trot_sim EST_CTRL와 동일 env 패리티; 미설정=struct 기본, clean 검증서 bit-동등)
    if (getenv("KF_QV"))     est_.KF_QV     = atof(getenv("KF_QV"));
    if (getenv("KF_QP"))     est_.KF_QP     = atof(getenv("KF_QP"));
    if (getenv("KF_RV"))     est_.KF_RV     = atof(getenv("KF_RV"));
    if (getenv("KF_ACC_LP")) est_.KF_ACC_LP = atof(getenv("KF_ACC_LP"));
    est_.reset(Eigen::Vector3d(q_.d->qpos[0], q_.d->qpos[1], q_.d->qpos[2]));  // 초기 base(crouch_home)로 리셋
  }
  ~EkfEstimator() { if (d_est_) mj_deleteData(d_est_); }

  mjData* d_est() { return d_est_; }       // 컨트롤러가 계산할 추정상태

  void update(const LowState& s) {
    const int NJ = q_.m->nq - 7;
    std::vector<bool> cts(4);
    for (int i = 0; i < 4; ++i) cts[i] = s.foot_force[i] > 0.5;   // HAL이 dist<0.002로 채운 접촉 지시자
    double aw[3]   = {s.imu_acc[0],  s.imu_acc[1],  s.imu_acc[2]};   // world 선가속
    double quat[4] = {s.imu_quat[0], s.imu_quat[1], s.imu_quat[2], s.imu_quat[3]};
    double gyro[3] = {s.imu_gyro[0], s.imu_gyro[1], s.imu_gyro[2]};
    est_.estimate_kf(q_.m, s.q.data(), s.dq.data(), quat, gyro, aw, fgid_, fr_, cts, q_.m->opt.timestep);
    // d_est: base=KF추정 · 자세/gyro=측정 · 관절=측정 (trot_sim과 동일 조립)
    for (int c = 0; c < 3; ++c) { d_est_->qpos[c]     = est_.p[c]; d_est_->qvel[c]     = est_.v[c]; }
    for (int c = 0; c < 4; ++c)   d_est_->qpos[3 + c] = quat[c];
    for (int c = 0; c < 3; ++c)   d_est_->qvel[3 + c] = gyro[c];
    for (int j = 0; j < NJ; ++j) { d_est_->qpos[7 + j] = s.q[j]; d_est_->qvel[6 + j] = s.dq[j]; }
    d_est_->time = s.time;
    mj_forward(q_.m, d_est_);
  }
};

}  // namespace qc
