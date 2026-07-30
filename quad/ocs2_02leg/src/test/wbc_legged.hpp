// 02_Leg WBC — legged_control(qiayuanl) WbcBase+WeightedWbc 충실 이식.
//   결정변수 x = [q̈(nv), f(3*4), τ(nJ)]. hard: full EOM·torque limits·friction pyramid·no-contact-motion.
//   soft: swing(Cartesian)·base accel(MPC 모멘텀률 feedforward)·contact force. eiquadprog.
//   원본=legged_control/legged_wbc/{WbcBase,WeightedWbc}. Go1→02_Leg·qpOASES→eiquadprog 적응.
#pragma once
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cstdio>
#include <cstdlib>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_centroidal_model/CentroidalModelInfo.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_legged_robot/gait/MotionPhaseDefinition.h>
#include <eiquadprog/eiquadprog-fast.hpp>
#ifdef WBC_USE_QPOASES
#include <qpOASES.hpp>
#endif

class WbcLegged {
 public:
  using vector_t = Eigen::VectorXd;
  using matrix_t = Eigen::MatrixXd;
  using vector3_t = Eigen::Vector3d;

  WbcLegged(ocs2::PinocchioInterface pin, ocs2::CentroidalModelInfo info, const std::array<int, 4>& eeFrames)
      : pinM_(pin), pinD_(std::move(pin)), info_(std::move(info)), mapping_(info_) {
    nv_ = info_.generalizedCoordinatesNum;   // 18
    nJ_ = info_.actuatedDofNum;              // 12
    nc_ = info_.numThreeDofContacts;         // 4
    nx_ = nv_ + 3 * nc_ + nJ_;               // q̈ + f + τ = 42
    for (int i = 0; i < 4; ++i) eeId_[i] = eeFrames[i];
    inputLast_ = vector_t::Zero(info_.inputDim);
    torqueLimits_ = vector_t(3); torqueLimits_ << 84.0, 84.0, 126.0;  // 02_Leg (hip,thigh,calf) peak Nm
  }

  // stateDesired/inputDesired=OCS2 centroidal, rbdMeasured=[eulZYX,pos,jointPos,angVel_w,linVel_w,jointVel], mode, period=dt.
  vector_t update(const vector_t& stateDesired, const vector_t& inputDesired, const vector_t& rbdMeasured, size_t mode,
                  double period) {
    auto cf = ocs2::legged_robot::modeNumber2StanceLeg(mode);
    numContacts_ = 0;
    for (int i = 0; i < 4; ++i) { contact_[i] = cf[i]; if (cf[i]) numContacts_++; }
    updateMeasured(rbdMeasured);
    updateDesired(stateDesired, inputDesired, period);

    // ── hard 제약: EOM(eq) + no-contact-motion(eq) + swing zero-force(eq) + torque limit(ineq) + friction(ineq) ──
    std::vector<matrix_t> eqA; std::vector<vector_t> eqb;
    std::vector<matrix_t> ineqD; std::vector<vector_t> ineqf;
    { auto t = eomTask(); eqA.push_back(t.first); eqb.push_back(t.second); }
    { auto t = noContactMotionTask(); if (t.first.rows()) { eqA.push_back(t.first); eqb.push_back(t.second); } }
    { auto t = frictionEqTask(); if (t.first.rows()) { eqA.push_back(t.first); eqb.push_back(t.second); } }  // swing f=0
    { auto t = torqueLimitTask(); ineqD.push_back(t.first); ineqf.push_back(t.second); }
    { auto t = frictionIneqTask(); if (t.first.rows()) { ineqD.push_back(t.first); ineqf.push_back(t.second); } }

    int neq = 0; for (auto& a : eqA) neq += a.rows();
    int nineq = 0; for (auto& d : ineqD) nineq += d.rows();
    neq_ = neq; nineq_ = nineq;

    // ── soft 비용 태스크: swing*wSw + baseAccel*wBase(FF) + force*wF ──
    std::vector<std::pair<matrix_t, vector_t>> costs;
    auto addCost = [&](std::pair<matrix_t, vector_t> t, double w) { if (t.first.rows()) costs.push_back({w * t.first, w * t.second}); };
    addCost(swingTask(), wSwing_);
    addCost(baseAccelTask(), wBase_);
    addCost(contactForceTask(inputDesired), wForce_);

    vector_t x(nx_);
#ifdef WBC_USE_QPOASES
    // legged WeightedWbc 정식: H=AᵀA(weighted 스택), min ½xᵀHx+gᵀx s.t. lbA≤A_c·x≤ubA.
    //   eq: lbA=ubA=b / ineq(d·x≤f): lbA=-INF,ubA=f. setToMPC=semi-def H·active-set robust.
    int nrCost = 0; for (auto& c : costs) nrCost += c.first.rows();
    matrix_t Aw(nrCost, nx_); vector_t bw(nrCost); { int r = 0; for (auto& c : costs) { Aw.middleRows(r, c.first.rows()) = c.first; bw.segment(r, c.second.size()) = c.second; r += c.first.rows(); } }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H = Aw.transpose() * Aw;
    vector_t g = -Aw.transpose() * bw;
    int nCon = neq + nineq;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ac(nCon, nx_);
    vector_t lbA(nCon), ubA(nCon);
    { int r = 0;
      for (size_t k = 0; k < eqA.size(); ++k) { Ac.middleRows(r, eqA[k].rows()) = eqA[k]; lbA.segment(r, eqb[k].size()) = eqb[k]; ubA.segment(r, eqb[k].size()) = eqb[k]; r += eqA[k].rows(); }
      for (size_t k = 0; k < ineqD.size(); ++k) { Ac.middleRows(r, ineqD[k].rows()) = ineqD[k]; lbA.segment(r, ineqf[k].size()).setConstant(-qpOASES::INFTY); ubA.segment(r, ineqf[k].size()) = ineqf[k]; r += ineqD[k].rows(); } }
    qpOASES::QProblem qp(nx_, nCon);
    qpOASES::Options opt; opt.setToMPC(); opt.printLevel = qpOASES::PL_NONE; opt.enableEqualities = qpOASES::BT_TRUE; qp.setOptions(opt);
    int nWsr = 20;
    qpOASES::returnValue rv = qp.init(H.data(), g.data(), Ac.data(), nullptr, nullptr, lbA.data(), ubA.data(), nWsr);
    lastStatus_ = (int)rv;
    if (rv == qpOASES::SUCCESSFUL_RETURN && qp.getPrimalSolution(x.data()) == qpOASES::SUCCESSFUL_RETURN && x.allFinite()) last_ = x;
    else { qpFail_++; if (last_.size() == nx_) x = last_; else x.setZero(); }
#else
    // eiquadprog: min .5xᵀGx+gᵀx s.t. CE x+ce0=0, CI x+ci0≥0.  (G positive-definite 요구 → reg_)
    matrix_t CE(neq, nx_); vector_t ce0(neq); { int r = 0; for (size_t k = 0; k < eqA.size(); ++k) { CE.middleRows(r, eqA[k].rows()) = eqA[k]; ce0.segment(r, eqb[k].size()) = -eqb[k]; r += eqA[k].rows(); } }
    matrix_t CI(nineq, nx_); vector_t ci0(nineq); { int r = 0; for (size_t k = 0; k < ineqD.size(); ++k) { CI.middleRows(r, ineqD[k].rows()) = -ineqD[k]; ci0.segment(r, ineqf[k].size()) = ineqf[k]; r += ineqD[k].rows(); } }  // d·x≤f → -d·x+f≥0
    matrix_t G = matrix_t::Zero(nx_, nx_); vector_t g = vector_t::Zero(nx_);
    for (auto& c : costs) { G.noalias() += c.first.transpose() * c.first; g.noalias() -= c.first.transpose() * c.second; }
    G.diagonal().array() += reg_;  // τ·joint-q̈ 블록 near-singular → 최소노름 정규화
    eiquadprog::solvers::EiquadprogFast qp; qp.reset(nx_, neq, nineq);
    auto st = qp.solve_quadprog(G, g, CE, ce0, CI, ci0, x);
    lastStatus_ = st;
    if (st != eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) { qpFail_++; if (last_.size() == nx_) x = last_; else x.setZero(); }
    else last_ = x;
#endif
    // ── 진단 덤프(LEGDBG): 첫 몇 콜서 내부값 객관 계측 ──
    if (std::getenv("LEGDBG") && dbgN_ < 4) {
      dbgN_++;
      vector_t qdd = x.head(nv_), f = x.segment(nv_, 3 * nc_), tau = x.tail(nJ_);
      matrix_t S = matrix_t::Zero(nJ_, nv_); S.rightCols(nJ_).setIdentity();
      vector_t eomRes = M_ * qdd - j_.transpose() * f - S.transpose() * tau + nle_;
      fprintf(stderr, "\n[LEGDBG#%d] status=%d qpFail=%d numContacts=%d\n", dbgN_, lastStatus_, qpFail_, numContacts_);
      fprintf(stderr, "  aBaseFF(des base q̈) = [%.2f %.2f %.2f | %.2f %.2f %.2f]\n", aBaseFF_(0), aBaseFF_(1), aBaseFF_(2), aBaseFF_(3), aBaseFF_(4), aBaseFF_(5));
      fprintf(stderr, "  qddBase(solved)     = [%.2f %.2f %.2f | %.2f %.2f %.2f]\n", qdd(0), qdd(1), qdd(2), qdd(3), qdd(4), qdd(5));
      fprintf(stderr, "  Fz per foot: solved=[%.1f %.1f %.1f %.1f]  fDes=[%.1f %.1f %.1f %.1f] (sum solved=%.1f, mg=%.1f)\n",
              f(2), f(5), f(8), f(11), inputDesired(2), inputDesired(5), inputDesired(8), inputDesired(11),
              f(2) + f(5) + f(8) + f(11), info_.robotMass * 9.81);
      fprintf(stderr, "  Fx per foot: solved=[%.1f %.1f %.1f %.1f]  Fy=[%.1f %.1f %.1f %.1f]\n",
              f(0), f(3), f(6), f(9), f(1), f(4), f(7), f(10));
      fprintf(stderr, "  tau(12) = [%.1f %.1f %.1f | %.1f %.1f %.1f | %.1f %.1f %.1f | %.1f %.1f %.1f]\n",
              tau(0), tau(1), tau(2), tau(3), tau(4), tau(5), tau(6), tau(7), tau(8), tau(9), tau(10), tau(11));
      fprintf(stderr, "  EOM residual |r|=%.3e  (base6 |r|=%.3e)  nle base=[%.1f %.1f %.1f|%.1f %.1f %.1f]\n",
              eomRes.norm(), eomRes.head(6).norm(), nle_(0), nle_(1), nle_(2), nle_(3), nle_(4), nle_(5));
    }
    return x.tail(nJ_);  // τ (관절 토크)
  }

  double swingKp_ = 350, swingKd_ = 30;
  double wSwing_ = 100, wBase_ = 1, wForce_ = 0.01;
  bool basePd_ = false, baseNoFF_ = false; double kpBase_ = 100, kdBase_ = 10;  // BASE_PD/BASE_NOFF 진단
  double reg_ = 1e-4;  // eiquadprog G positive-definite 정규화(qpOASES 불요, eiquadprog 필수)
  vector_t torqueLimits_;
  int qpFail_ = 0, lastStatus_ = 0, neq_ = 0, nineq_ = 0, dbgN_ = 0;

 private:
  // ── measured 준비 (legged updateMeasured) ──
  void updateMeasured(const vector_t& rbd) {
    qM_ = vector_t(nv_); vM_ = vector_t(nv_);
    qM_.head<3>() = rbd.segment<3>(3);                        // pos
    qM_.segment<3>(3) = rbd.head<3>();                        // eulerZYX
    qM_.tail(nJ_) = rbd.segment(6, nJ_);                      // joints
    vM_.head<3>() = rbd.segment<3>(nv_ + 3);                  // linVel world
    vM_.segment<3>(3) = ocs2::getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<double>(qM_.segment<3>(3), rbd.segment<3>(nv_));
    vM_.tail(nJ_) = rbd.segment(nv_ + 6, nJ_);
    auto& model = pinM_.getModel(); auto& data = pinM_.getData();
    pinocchio::forwardKinematics(model, data, qM_, vM_);
    pinocchio::computeJointJacobians(model, data);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::crba(model, data, qM_);
    data.M.triangularView<Eigen::StrictlyLower>() = data.M.transpose().triangularView<Eigen::StrictlyLower>();
    M_ = data.M; nle_ = pinocchio::nonLinearEffects(model, data, qM_, vM_);
    j_ = matrix_t::Zero(3 * nc_, nv_);
    for (int i = 0; i < nc_; ++i) { matrix_t jac = matrix_t::Zero(6, nv_); pinocchio::getFrameJacobian(model, data, eeId_[i], pinocchio::LOCAL_WORLD_ALIGNED, jac); j_.middleRows(3 * i, 3) = jac.topRows<3>(); }
    pinocchio::computeJointJacobiansTimeVariation(model, data, qM_, vM_);
    dj_ = matrix_t::Zero(3 * nc_, nv_);
    for (int i = 0; i < nc_; ++i) { matrix_t jac = matrix_t::Zero(6, nv_); pinocchio::getFrameJacobianTimeVariation(model, data, eeId_[i], pinocchio::LOCAL_WORLD_ALIGNED, jac); dj_.middleRows(3 * i, 3) = jac.topRows<3>(); }
  }
  // ── desired 준비 (base accel FF + swing 목표) ──
  void updateDesired(const vector_t& sD, const vector_t& iD, double period) {
    auto& model = pinD_.getModel(); auto& data = pinD_.getData();
    mapping_.setPinocchioInterface(pinD_);
    qD_ = mapping_.getPinocchioJointPosition(sD);
    pinocchio::forwardKinematics(model, data, qD_);
    pinocchio::computeJointJacobians(model, data, qD_);
    pinocchio::updateFramePlacements(model, data);
    ocs2::updateCentroidalDynamics(pinD_, info_, qD_);
    vD_ = mapping_.getPinocchioJointVelocity(sD, iD);
    pinocchio::forwardKinematics(model, data, qD_, vD_);
    // base accel FF
    vector_t jAccel = ocs2::centroidal_model::getJointVelocities(vector_t(iD - inputLast_), info_) / period;
    inputLast_ = iD;
    const auto& A = ocs2::getCentroidalMomentumMatrix(pinD_);
    Eigen::Matrix<double, 6, 6> Ab = A.leftCols<6>();
    Eigen::Matrix<double, 6, 6> AbInv = ocs2::computeFloatingBaseCentroidalMomentumMatrixInverse(Ab);
    matrix_t Aj = A.rightCols(nJ_);
    matrix_t ADot = pinocchio::dccrba(model, data, qD_, vD_);
    Eigen::Matrix<double, 6, 1> momRate = info_.robotMass * ocs2::getNormalizedCentroidalMomentumRate(pinD_, info_, iD);
    momRate.noalias() -= ADot * vD_;
    momRate.noalias() -= Aj * jAccel;
    aBaseFF_ = AbInv * momRate;
    // 접촉/스윙 발 위치·속도(측정·목표)
    for (int i = 0; i < 4; ++i) {
      posD_[i] = data.oMf[eeId_[i]].translation();
      matrix_t jac = matrix_t::Zero(6, nv_); pinocchio::getFrameJacobian(model, data, eeId_[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
      velD_[i] = jac.topRows<3>() * vD_;
    }
    auto& mM = pinM_.getModel(); auto& dM = pinM_.getData();
    for (int i = 0; i < 4; ++i) { posM_[i] = dM.oMf[eeId_[i]].translation(); velM_[i] = j_.middleRows(3 * i, 3) * vM_; }
  }
  // ── tasks ──
  std::pair<matrix_t, vector_t> eomTask() {  // [M, -jᵀ, -Sᵀ] x = -nle  (전체 18행)
    matrix_t S = matrix_t::Zero(nJ_, nv_); S.rightCols(nJ_).setIdentity();
    matrix_t a(nv_, nx_); a << M_, -j_.transpose(), -S.transpose();
    return {a, -nle_};
  }
  std::pair<matrix_t, vector_t> noContactMotionTask() {  // stance: J q̈ = -dJ v
    matrix_t a = matrix_t::Zero(3 * numContacts_, nx_); vector_t b(3 * numContacts_);
    int jc = 0; for (int i = 0; i < nc_; ++i) if (contact_[i]) { a.block(3 * jc, 0, 3, nv_) = j_.middleRows(3 * i, 3); b.segment(3 * jc, 3) = -dj_.middleRows(3 * i, 3) * vM_; jc++; }
    return {a, b};
  }
  std::pair<matrix_t, vector_t> frictionEqTask() {  // swing: f=0
    int ns = nc_ - numContacts_;
    matrix_t a = matrix_t::Zero(3 * ns, nx_); int jc = 0;
    for (int i = 0; i < nc_; ++i) if (!contact_[i]) { a.block(3 * jc++, nv_ + 3 * i, 3, 3).setIdentity(); }
    return {a, vector_t::Zero(3 * ns)};
  }
  std::pair<matrix_t, vector_t> torqueLimitTask() {  // -τlim ≤ τ ≤ τlim  → [I;-I]τ ≤ [lim;lim]
    matrix_t d = matrix_t::Zero(2 * nJ_, nx_); int off = nv_ + 3 * nc_;
    d.block(0, off, nJ_, nJ_).setIdentity(); d.block(nJ_, off, nJ_, nJ_) = -matrix_t::Identity(nJ_, nJ_);
    vector_t f(2 * nJ_); for (int l = 0; l < 2 * nJ_ / 3; ++l) f.segment<3>(3 * l) = torqueLimits_;
    return {d, f};
  }
  std::pair<matrix_t, vector_t> frictionIneqTask() {  // stance: pyramid·f ≤ 0
    Eigen::Matrix<double, 5, 3> P; P << 0, 0, -1, 1, 0, -mu_, -1, 0, -mu_, 0, 1, -mu_, 0, -1, -mu_;
    matrix_t d = matrix_t::Zero(5 * numContacts_, nx_); int jc = 0;
    for (int i = 0; i < nc_; ++i) if (contact_[i]) { d.block(5 * jc++, nv_ + 3 * i, 5, 3) = P; }
    return {d, vector_t::Zero(5 * numContacts_)};
  }
  std::pair<matrix_t, vector_t> swingTask() {
    int ns = nc_ - numContacts_;
    matrix_t a = matrix_t::Zero(3 * ns, nx_); vector_t b(3 * ns); int jc = 0;
    for (int i = 0; i < nc_; ++i) if (!contact_[i]) {
      vector3_t accel = swingKp_ * (posD_[i] - posM_[i]) + swingKd_ * (velD_[i] - velM_[i]);
      a.block(3 * jc, 0, 3, nv_) = j_.middleRows(3 * i, 3);
      b.segment(3 * jc, 3) = accel - dj_.middleRows(3 * i, 3) * vM_; jc++;
    }
    return {a, b};
  }
  std::pair<matrix_t, vector_t> baseAccelTask() {  // q̈[0:6] = aBaseFF (+옵션 PD 되먹임)
    matrix_t a = matrix_t::Zero(6, nx_); a.block(0, 0, 6, 6).setIdentity();
    Eigen::Matrix<double, 6, 1> b = baseNoFF_ ? Eigen::Matrix<double, 6, 1>::Zero() : aBaseFF_;  // BASE_NOFF=순수PD(내 기본WBC식)
    if (basePd_) {  // ★진단/수정: 순수FF에 base 위치·속도 PD 추가(A WBIC·내 기본WBC와 동일 원리)
      b.head<3>() += kpBase_ * (qD_.head<3>() - qM_.head<3>()) + kdBase_ * (vD_.head<3>() - vM_.head<3>());       // pos
      b.tail<3>() += kpBase_ * (qD_.segment<3>(3) - qM_.segment<3>(3)) + kdBase_ * (vD_.segment<3>(3) - vM_.segment<3>(3));  // eulerZYX
    }
    return {a, b};
  }
  std::pair<matrix_t, vector_t> contactForceTask(const vector_t& iD) {  // f = fDes
    matrix_t a = matrix_t::Zero(3 * nc_, nx_); a.block(0, nv_, 3 * nc_, 3 * nc_).setIdentity();
    return {a, iD.head(3 * nc_)};
  }

  ocs2::PinocchioInterface pinM_, pinD_;
  ocs2::CentroidalModelInfo info_;
  ocs2::CentroidalModelPinocchioMapping mapping_;
  std::array<int, 4> eeId_;
  int nv_, nJ_, nc_, nx_, numContacts_ = 4;
  std::array<bool, 4> contact_{true, true, true, true};
  double mu_ = 0.5;
  vector_t qM_, vM_, qD_, vD_, nle_, inputLast_, last_;
  matrix_t M_, j_, dj_;
  Eigen::Matrix<double, 6, 1> aBaseFF_ = Eigen::Matrix<double, 6, 1>::Zero();
  std::array<vector3_t, 4> posD_, velD_, posM_, velM_;
};
