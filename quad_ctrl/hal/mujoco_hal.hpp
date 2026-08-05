#pragma once
// sim HAL(실물 배선) — MuJoCo 백엔드 + quad/cpp `QuadControl` 코어 소유(재기어/GEARBOX/솔버 셋업 재사용).
//   read():  q.d → LowState(관절·IMU).   write(): LowCmd.tau_ff → q.d->ctrl + mj_step(1스텝).
//   ★이관 1단계(GT): 검증된 컨트롤러를 재작성 없이 HAL 경계 뒤로. real_hal은 read/write만 실센서/모터로 교체.
#include "hal/robot_interface.hpp"
#include <mujoco/mujoco.h>
#include "quad_control.hpp"        // quad/cpp/src (CMake include_dir) — ::QuadControl
#include "trot_controller.hpp"     // apply_env_gains

namespace qc {

class MujocoHal : public RobotInterface {
  ::QuadControl q_;                 // m,d 소유 + 재기어/게인/q_home LUT/MPC 셋업(quad/cpp 그대로)
 public:
  explicit MujocoHal(const char* mjcf) {
    q_.load(mjcf); apply_env_gains(q_); q_.crouch_home(); q_.build_qhome_lut(); q_.setup_mpc();
  }
  ::QuadControl& core() { return q_; }            // 컨트롤러 브리지가 같은 코어 공유(TrotCtrl(q_))

  int    nu() const override { return q_.nu; }
  double dt() const override { return q_.m->opt.timestep; }

  bool read(LowState& s) override {
    mjData* d = q_.d; const int nu = q_.nu;
    s.time = d->time;
    s.q.resize(nu); s.dq.resize(nu); s.tau_est.resize(nu);
    for (int i = 0; i < nu; ++i) { s.q[i] = d->qpos[7 + i]; s.dq[i] = d->qvel[6 + i]; s.tau_est[i] = d->actuator_force[i]; }
    for (int a = 0; a < 4; ++a) s.imu_quat[a] = d->qpos[3 + a];   // wxyz
    for (int a = 0; a < 3; ++a) s.imu_gyro[a] = d->qvel[3 + a];   // body 각속도
    for (int a = 0; a < 3; ++a) s.imu_acc[a]  = d->qacc[a];       // world 선가속(KF 예측 입력; real_hal=R·f_body+g)
    // 발 접촉(trot_sim EST와 동일 판정 dist<0.002) → foot_force에 0/1 지시자(KF stance 게이팅)
    for (int i = 0; i < 4; ++i) {
      bool con = false;
      for (int ci = 0; ci < d->ncon; ++ci) { const auto& c = d->contact[ci];
        if ((c.geom1 == q_.fgid[i] || c.geom2 == q_.fgid[i]) && c.dist < 0.002) { con = true; break; } }
      s.foot_force[i] = con ? 1.0 : 0.0;
    }
    return true;
  }

  bool write(const LowCmd& c) override {
    mjData* d = q_.d;
    for (int i = 0; i < q_.nu; ++i) d->ctrl[i] = c.tau_ff[i];     // TrotCtrl이 tau 계산(kp/kd=0 규약)
    mj_step(q_.m, d);
    return true;
  }
};

}  // namespace qc
