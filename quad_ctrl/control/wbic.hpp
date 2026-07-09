#pragma once
// 단일 QP 전신제어(WBIC): CoM + 몸통자세 + 접촉정합 → 관절토크(GRF로 base 닫음).
// ★이관 원본: simulation/quad/cpp/src/quad_control.hpp
//     - wbic_stance()  : 4발 지지 균형(서기/앉기 홀드)
//     - wbic_track()   : 스윙 WBIC(보행 중)
//     - wbic_jump()    : 궤적추종 WBIC(점프/기립)
//   부동베이스 동역학 M·q̈ = −h + Jᵀλ, 마찰원뿔·최소수직력, eiquadprog.
#include "estimator/state.hpp"
#include "hal/robot_interface.hpp"

namespace qc {

class Wbic {
 public:
  // Wbic(const RobotModel&);   // TODO: 모델(M,h,Jc) 주입
  // 접촉발 지지 균형 → out.tau_ff. contacts=발 인덱스 집합.
  // bool stance(const State&, const Vector3d& com_ref, const Vector4d& quat_ref,
  //             const std::vector<int>& contacts, LowCmd& out);   // TODO 이관
};

}  // namespace qc
