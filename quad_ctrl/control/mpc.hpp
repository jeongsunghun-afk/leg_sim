#pragma once
// Convex SRBD MPC(Di Carlo): 단강체 가정으로 지평선 GRF 계획 → 몸통 속도/자세 추종.
// ★이관 원본: simulation/quad/cpp/src/mpc.hpp (mpc_qp_plan, MpcCfg)
//   ★주의: euler-rate는 world/body 프레임 정합 필수 — Rz(yaw)ᵀ 표준수정(커밋 1c56d2f) 유지.
#include "estimator/state.hpp"

namespace qc {

class Mpc {
 public:
  // Mpc(const MpcCfg&);
  // 상태 + 참조(속도/yaw) → 발별 GRF(다음 스텝). WBIC/보행이 소비.
  // Eigen::Matrix<double,4,3> plan(const State&, const VectorXd& x_ref,
  //                                const std::vector<std::array<int,4>>& contact_sched);   // TODO 이관
};

}  // namespace qc
