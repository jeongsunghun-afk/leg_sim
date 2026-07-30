// D1 Phase 3a: 스윙발 지형 클리어런스 제약(legged_perceptive FootCollisionConstraint 포팅).
//   스윙 국면 발만 활성(stance발은 지형 위=SDF 0이라 제외). value = SDF(발) − clearance ≥ 0.
//   SDF=MjTerrainSdf(DistanceTransformInterface), gating=표준 SwitchedModelReferenceManager 접촉플래그.
#pragma once
#include <ocs2_core/constraint/StateConstraint.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <ocs2_legged_robot/reference_manager/SwitchedModelReferenceManager.h>
#include <ocs2_perceptive/distance_transform/DistanceTransformInterface.h>
#include <memory>

class FootTerrainClearanceConstraint final : public ocs2::StateConstraint {
 public:
  using scalar_t = ocs2::scalar_t;
  using vector_t = ocs2::vector_t;

  FootTerrainClearanceConstraint(const ocs2::legged_robot::SwitchedModelReferenceManager& refMgr,
                                 const ocs2::EndEffectorKinematics<scalar_t>& eeKin,
                                 std::shared_ptr<ocs2::DistanceTransformInterface> sdf, size_t contactIdx, scalar_t clearance)
      : StateConstraint(ocs2::ConstraintOrder::Linear),
        refMgr_(&refMgr), eeKin_(eeKin.clone()), sdf_(std::move(sdf)), idx_(contactIdx), clearance_(clearance) {}

  FootTerrainClearanceConstraint* clone() const override { return new FootTerrainClearanceConstraint(*this); }
  bool isActive(scalar_t time) const override {  // 스윙(비접촉) 국면만 활성(± offset으로 전환 여유)
    scalar_t off = 0.05;
    return !refMgr_->getContactFlags(time)[idx_] &&
           !refMgr_->getContactFlags(time + 0.5 * off)[idx_] && !refMgr_->getContactFlags(time - off)[idx_];
  }
  size_t getNumConstraints(scalar_t) const override { return 1; }

  vector_t getValue(scalar_t, const vector_t& state, const ocs2::PreComputation&) const override {
    vector_t v(1);
    v(0) = sdf_->getValue(eeKin_->getPosition(state).front()) - clearance_;
    return v;
  }
  ocs2::VectorFunctionLinearApproximation getLinearApproximation(scalar_t, const vector_t& state,
                                                                 const ocs2::PreComputation&) const override {
    auto approx = ocs2::VectorFunctionLinearApproximation::Zero(1, state.size(), 0);
    auto vg = sdf_->getLinearApproximation(eeKin_->getPosition(state).front());  // (value, gradient)
    approx.f(0) = vg.first - clearance_;
    approx.dfdx = vg.second.transpose() * eeKin_->getPositionLinearApproximation(state).front().dfdx;
    return approx;
  }

 private:
  FootTerrainClearanceConstraint(const FootTerrainClearanceConstraint& o)
      : StateConstraint(ocs2::ConstraintOrder::Linear),
        refMgr_(o.refMgr_), eeKin_(o.eeKin_->clone()), sdf_(o.sdf_), idx_(o.idx_), clearance_(o.clearance_) {}

  const ocs2::legged_robot::SwitchedModelReferenceManager* refMgr_;
  std::unique_ptr<ocs2::EndEffectorKinematics<scalar_t>> eeKin_;
  std::shared_ptr<ocs2::DistanceTransformInterface> sdf_;
  size_t idx_;
  scalar_t clearance_;
};
