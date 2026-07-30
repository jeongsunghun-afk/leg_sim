// D1 Phase 3b: 스윙발 발판을 유효 지형(convex 영역)에 배치하는 제약(legged_perceptive FootPlacementConstraint 포팅).
//   value = A·p_foot + b ≥ 0 (발 XY가 convex 폴리곤 내부). A,b = 발판 nominal 근처 유효지형 로컬 박스(브리지서 heightmap로 생성, CGAL 회피).
//   스윙 국면 + 발판선택됨일 때만 활성. A footScore식(스윙시작 1회 선택·홀드)을 MPC 제약으로.
#pragma once
#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <ocs2_legged_robot/reference_manager/SwitchedModelReferenceManager.h>
#include <Eigen/Dense>
#include <array>
#include <memory>

// 발별 convex 발판영역: A·p + b ≥ 0 (nRow 행, p=발 3D위치 world). 브리지가 매 스윙 갱신.
struct FootRegions {
  static constexpr int nRow = 4;  // 박스=4 반평면
  std::array<Eigen::Matrix<double, nRow, 3>, 4> A;
  std::array<Eigen::Matrix<double, nRow, 1>, 4> b;
  std::array<bool, 4> active{false, false, false, false};  // 이 발 발판제약 활성
  FootRegions() { for (int i = 0; i < 4; ++i) { A[i].setZero(); b[i].setOnes(); } }  // 비활성기본=항상만족(b=1)
  // XY 박스: 중심(cx,cy)·반폭(hx,hy) → A·p+b≥0 4행. (±x: ±(cx±hx − p.x)…)
  void setBox(int i, double cx, double cy, double hx, double hy) {
    A[i].setZero();
    A[i](0, 0) = 1;  b[i](0) = hx - cx;   // p.x ≤ cx+hx  →  −p.x + (cx+hx) ≥ 0  → row: (1,0,0)·p? 부호 통일 아래
    A[i](1, 0) = -1; b[i](1) = hx + cx;   // p.x ≥ cx−hx  →  p.x − (cx−hx) ≥ 0
    A[i](2, 1) = 1;  b[i](2) = hy - cy;   // p.y ≤ cy+hy
    A[i](3, 1) = -1; b[i](3) = hy + cy;   // p.y ≥ cy−hy
    active[i] = true;
  }
};

class FootTerrainPlacementConstraint final : public ocs2::StateConstraint {
 public:
  using scalar_t = ocs2::scalar_t;
  using vector_t = ocs2::vector_t;

  FootTerrainPlacementConstraint(const ocs2::legged_robot::SwitchedModelReferenceManager& refMgr,
                                 const ocs2::EndEffectorKinematics<scalar_t>& eeKin,
                                 std::shared_ptr<FootRegions> regions, size_t contactIdx)
      : StateConstraint(ocs2::ConstraintOrder::Linear),
        refMgr_(&refMgr), eeKin_(eeKin.clone()), regions_(std::move(regions)), idx_(contactIdx) {}

  FootTerrainPlacementConstraint* clone() const override { return new FootTerrainPlacementConstraint(*this); }
  bool isActive(scalar_t time) const override {  // 스윙(비접촉) ∧ 발판선택됨
    return !refMgr_->getContactFlags(time)[idx_] && regions_->active[idx_];
  }
  size_t getNumConstraints(scalar_t) const override { return FootRegions::nRow; }

  vector_t getValue(scalar_t, const vector_t& state, const ocs2::PreComputation&) const override {
    return regions_->A[idx_] * eeKin_->getPosition(state).front() + regions_->b[idx_];
  }
  ocs2::VectorFunctionLinearApproximation getLinearApproximation(scalar_t, const vector_t& state,
                                                                 const ocs2::PreComputation&) const override {
    auto approx = ocs2::VectorFunctionLinearApproximation::Zero(FootRegions::nRow, state.size(), 0);
    const auto posApprox = eeKin_->getPositionLinearApproximation(state).front();
    approx.f = regions_->A[idx_] * posApprox.f + regions_->b[idx_];
    approx.dfdx = regions_->A[idx_] * posApprox.dfdx;
    return approx;
  }

 private:
  FootTerrainPlacementConstraint(const FootTerrainPlacementConstraint& o)
      : StateConstraint(ocs2::ConstraintOrder::Linear),
        refMgr_(o.refMgr_), eeKin_(o.eeKin_->clone()), regions_(o.regions_), idx_(o.idx_) {}

  const ocs2::legged_robot::SwitchedModelReferenceManager* refMgr_;
  std::unique_ptr<ocs2::EndEffectorKinematics<scalar_t>> eeKin_;
  std::shared_ptr<FootRegions> regions_;
  size_t idx_;
};
