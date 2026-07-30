// D1 Phase 3b: 발판을 유효 지형(convex 박스)에 배치하는 제약(legged_perceptive FootPlacementConstraint 포팅).
//   value = A·p_foot + b ≥ 0 (발 XY가 박스 내부). A,b = LocalConvexRegion(heightmap로 생성, CGAL 회피).
//   ★활성 = 접촉(stance) ∧ 발판선택됨(valid) ∧ time≥stanceEnd = 현재 stance 이후의 '미래 착지'만 성형
//   (상류 getFootPlacementFlags=getContactFlags && time≥initStandFinalTime과 동일; 현재 딛은 발은 안 되당김).
#pragma once
#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <ocs2_legged_robot/reference_manager/SwitchedModelReferenceManager.h>
#include "local_convex_region.hpp"
#include <memory>

class FootTerrainPlacementConstraint final : public ocs2::StateConstraint {
 public:
  using scalar_t = ocs2::scalar_t;
  using vector_t = ocs2::vector_t;

  FootTerrainPlacementConstraint(const ocs2::legged_robot::SwitchedModelReferenceManager& refMgr,
                                 const ocs2::EndEffectorKinematics<scalar_t>& eeKin,
                                 std::shared_ptr<LocalConvexRegion> region, size_t contactIdx)
      : StateConstraint(ocs2::ConstraintOrder::Linear),
        refMgr_(&refMgr), eeKin_(eeKin.clone()), region_(std::move(region)), idx_(contactIdx) {}

  FootTerrainPlacementConstraint* clone() const override { return new FootTerrainPlacementConstraint(*this); }
  bool isActive(scalar_t time) const override {  // stance ∧ 발판선택됨 ∧ 현재 stance 이후(미래 착지)
    return refMgr_->getContactFlags(time)[idx_] && region_->valid(idx_) && time >= region_->stanceEnd(idx_);
  }
  size_t getNumConstraints(scalar_t) const override { return FootRegionAB::nRow; }

  vector_t getValue(scalar_t, const vector_t& state, const ocs2::PreComputation&) const override {
    const auto& ab = region_->params(idx_);
    return ab.a * eeKin_->getPosition(state).front() + ab.b;
  }
  ocs2::VectorFunctionLinearApproximation getLinearApproximation(scalar_t, const vector_t& state,
                                                                 const ocs2::PreComputation&) const override {
    const auto& ab = region_->params(idx_);
    auto approx = ocs2::VectorFunctionLinearApproximation::Zero(FootRegionAB::nRow, state.size(), 0);
    const auto posApprox = eeKin_->getPositionLinearApproximation(state).front();
    approx.f = ab.a * posApprox.f + ab.b;
    approx.dfdx = ab.a * posApprox.dfdx;
    return approx;
  }

 private:
  FootTerrainPlacementConstraint(const FootTerrainPlacementConstraint& o)
      : StateConstraint(ocs2::ConstraintOrder::Linear),
        refMgr_(o.refMgr_), eeKin_(o.eeKin_->clone()), region_(o.region_), idx_(o.idx_) {}

  const ocs2::legged_robot::SwitchedModelReferenceManager* refMgr_;
  std::unique_ptr<ocs2::EndEffectorKinematics<scalar_t>> eeKin_;
  std::shared_ptr<LocalConvexRegion> region_;
  size_t idx_;
};
