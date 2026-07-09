#pragma once
// sim HAL: MuJoCo 백엔드. 현 trot_sim/trot_view의 mj_step·d->ctrl·센서읽기를 이 뒤로 격리.
// ★이관 원본: simulation/quad/cpp/src/trot_sim.cpp / trot_view.cpp (mj 루프)
//   read(): d->qpos/qvel/sensordata → LowState (+ ground truth base pose는 확장필드로).
//   write(): LowCmd(kp/kd/tau_ff) → d->ctrl = tau_ff + kp·(q_des−q) + kd·(dq_des−dq), 한 스텝 mj_step.
// #include <mujoco/mujoco.h>
#include "hal/robot_interface.hpp"

namespace qc {

class MujocoHal : public RobotInterface {
  // mjModel* m_; mjData* d_; int nu_;
 public:
  // explicit MujocoHal(const std::string& mjcf);   // TODO: mj_loadXML, 재기어/GEARBOX/솔버 설정 이관
  // int nu() const override; double dt() const override;
  // bool read(LowState&) override; bool write(const LowCmd&) override;
};

}  // namespace qc
