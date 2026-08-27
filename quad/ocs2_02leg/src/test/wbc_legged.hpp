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
    perLeg_ = nJ_ / nc_;  // 다리당 관절수: 3(point-foot) or 4(발목포함)
    torqueLimits_ = vector_t(perLeg_);
    if (perLeg_ == 4) torqueLimits_ << 84.0, 84.0, 126.0, 100.8;  // hip,thigh,calf,ankle peak Nm (★ankle=재기어 8.4:1 실값 100.8, 구 168=14:1 stale)
    else torqueLimits_ << 84.0, 84.0, 126.0;                       // hip,thigh,calf
  }

  // stateDesired/inputDesired=OCS2 centroidal, rbdMeasured=[eulZYX,pos,jointPos,angVel_w,linVel_w,jointVel], mode, period=dt.
  vector_t update(const vector_t& stateDesired, const vector_t& inputDesired, const vector_t& rbdMeasured, size_t mode,
                  double period) {
    auto cf = ocs2::legged_robot::modeNumber2StanceLeg(mode);
    // ★접촉-일관 집합(전환 과지지 방지): motionC=planted(스케줄stance∧실접촉), forceC=힘가능(실접촉),
    //   swingC=스윙task(스케줄swing). useActual_ 아니면 실접촉=스케줄이라 기존과 동일(하위호환).
    nMotion_ = nForce_ = 0;
    for (int i = 0; i < 4; ++i) {
      bool sched = cf[i];
      bool touch = useActual_ ? actualContact_[i] : sched;
      motionC_[i] = sched && touch;  if (motionC_[i]) nMotion_++;
      forceC_[i] = touch;            if (forceC_[i]) nForce_++;
      swingC_[i] = !sched;
      contact_[i] = sched;  // 참조 유지
    }
    numContacts_ = nForce_;
    updateMeasured(rbdMeasured);
    updateDesired(stateDesired, inputDesired, period);

    // ── hard 제약: EOM(eq) + no-contact-motion(eq) + swing zero-force(eq) + torque limit(ineq) + friction(ineq) ──
    std::vector<matrix_t> eqA; std::vector<vector_t> eqb;
    std::vector<matrix_t> ineqD; std::vector<vector_t> ineqf;
    { auto t = eomTask(); eqA.push_back(t.first); eqb.push_back(t.second); }
    { auto t = noContactMotionTask(); if (t.first.rows()) { eqA.push_back(t.first); eqb.push_back(t.second); } }
    if (ankleHard_ && perLeg_ == 4) { auto t = ankleTask(); eqA.push_back(t.first); eqb.push_back(t.second); }  // ★발목 hard PD
    { auto t = frictionEqTask(); if (t.first.rows()) { eqA.push_back(t.first); eqb.push_back(t.second); } }  // swing f=0
    { auto t = torqueLimitTask(); ineqD.push_back(t.first); ineqf.push_back(t.second); }
    if (qLo_.size() == nJ_) { auto t = jointLimitTask(); ineqD.push_back(t.first); ineqf.push_back(t.second); }  // ★조인트 각도한계(MJCF range)
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
    if (wPosture_ > 0) addCost(postureTask(), wPosture_);  // ★널스페이스 posture(재중복 발목 표류 제어)

    vector_t x(nx_);
#ifdef WBC_USE_QPOASES
    // legged WeightedWbc 정식: H=AᵀA(weighted 스택), min ½xᵀHx+gᵀx s.t. lbA≤A_c·x≤ubA.
    //   eq: lbA=ubA=b / ineq(d·x≤f): lbA=-INF,ubA=f. setToMPC=semi-def H·active-set robust.
    int nrCost = 0; for (auto& c : costs) nrCost += c.first.rows();
    matrix_t Aw(nrCost, nx_); vector_t bw(nrCost); { int r = 0; for (auto& c : costs) { Aw.middleRows(r, c.first.rows()) = c.first; bw.segment(r, c.second.size()) = c.second; r += c.first.rows(); } }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> H = Aw.transpose() * Aw;
    vector_t g = -Aw.transpose() * bw;
    // ★null-space 평활(앞다리 채터 억제): 비용 미지정 방향(관절q̈·τ)이 min-norm으로 튀는 것 방지.
    //   Bellicoso 식(20) 토크최소화류. reg_ 소량 대각 정규화 → 스텝간 해 연속성↑.
    H.diagonal().array() += reg_;
    int nCon = neq + nineq;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ac(nCon, nx_);
    vector_t lbA(nCon), ubA(nCon);
    { int r = 0;
      for (size_t k = 0; k < eqA.size(); ++k) { Ac.middleRows(r, eqA[k].rows()) = eqA[k]; lbA.segment(r, eqb[k].size()) = eqb[k]; ubA.segment(r, eqb[k].size()) = eqb[k]; r += eqA[k].rows(); }
      for (size_t k = 0; k < ineqD.size(); ++k) { Ac.middleRows(r, ineqD[k].rows()) = ineqD[k]; lbA.segment(r, ineqf[k].size()).setConstant(-qpOASES::INFTY); ubA.segment(r, ineqf[k].size()) = ineqf[k]; r += ineqD[k].rows(); } }
    qpOASES::QProblem qp(nx_, nCon);
    qpOASES::Options opt; opt.setToMPC(); opt.printLevel = qpOASES::PL_NONE; opt.enableEqualities = qpOASES::BT_TRUE; qp.setOptions(opt);
    int nWsr = nWsr_;  // 기본 20(legged), NWSR env로 상향(스윙국면 active-set 반복 부족 대응)
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
      double fdx = inputDesired(0) + inputDesired(3) + inputDesired(6) + inputDesired(9);
      fprintf(stderr, "  fDes(MPC) Fx per foot=[%.1f %.1f %.1f %.1f] sumFx=%.1f (→aBaseFF_x=sumFx/m=%.2f)\n",
              inputDesired(0), inputDesired(3), inputDesired(6), inputDesired(9), fdx, fdx / info_.robotMass);
      fprintf(stderr, "  base pos err(des-meas)=[%.4f %.4f %.4f]  euler err=[%.4f %.4f %.4f]\n",
              qD_(0) - qM_(0), qD_(1) - qM_(1), qD_(2) - qM_(2), qD_(3) - qM_(3), qD_(4) - qM_(4), qD_(5) - qM_(5));
    }
    return x.tail(nJ_);  // τ (관절 토크)
  }

  double swingKp_ = 350, swingKd_ = 30;
  double swingErrCur_ = 0;   // ★진단: 현 스텝 스윙발 추종오차 max||posD-posM|| (SWING_DBG)
  bool swingFF_ = false;  // ★스윙 가속도 ff(eq19). SWING_FF=1로 켬(accD=J·u̇_des, jAccel 노이즈로 12-DOF선 버징 미개선)
  double wSwing_ = 100, wBase_ = 1, wForce_ = 0.01;
  double stanceKd_ = 0.0;   // ★stance 접촉 속도댐핑(baumgarte): J q̈ = -J̇v - KD·Jv. 착지 잔류속도→슬립 저감. 0=기존
  double wPosture_ = 0.5, kpPost_ = 20, kdPost_ = 2;  // ★널스페이스 posture task(weighted, 16-DOF 발목표류 제어). ★튜닝결론(2026-08, OMP=1 결정론하): 발목 게인으로 고속을 못 잡음 — POST=0.5는 VX≤0.4 강건(3/3)이나 VX=0.5 0/6, POST=8은 VX=0.5 6/6이나 VX=0.4 0/3(강건엔벨로프 깸). 어떤 값도 0.4·0.5 동시 불가=VX0.4~0.5는 본질적 marginal(발목 아닌 base 동역학). ∴ 기본=0.5 유지(VX≤0.4 강건). 고속 실험은 POST env로. weighted>발목hard(ANKLE_HARD 4/6<6/6)·>strict. POST=0으로 끔
  bool ankleHard_ = false; double kpAnkle_ = 150, kdAnkle_ = 12;  // ★발목 hard-constraint(ANKLE_HARD, 기본off). 발목4 q̈=PD를 hard eq로=TSID 내 발목만 strict("poor-man's HQP", 진짜 HQP계층 아님). ⚠검증(2026-08): weighted posture보다 나쁨(VX=0.5 4/6 vs weighted 6/6)+강성 낮춰야만 안정(kpAnk30 OK·80+ 붕괴) — 강체 발목이 터치다운 컴플라이언스·QP trade-off 자유를 뺏음. 기록용 유지
  bool basePd_ = false, baseNoFF_ = false; double kpBase_ = 100, kdBase_ = 10;  // BASE_PD/BASE_NOFF 진단
  void setActualContact(const bool a[4]) { for (int i = 0; i < 4; ++i) actualContact_[i] = a[i]; useActual_ = true; }  // 실제접촉 오버라이드(1프레임)
  vector_t rotorArm_;  // ★반사관성(Irot·N², pinocchio 관절순). MuJoCo plant의 dof_armature와 정합=컨트롤러가 실제 관절관성 반영(고속 저-토크 붕괴 근본수정). 빈 벡터=미적용(하위호환).
  void setRotorArmature(const vector_t& a) { rotorArm_ = a; }
  vector_t qLo_, qHi_; double kpLim_ = 200, kdLim_ = 20, margLim_ = 0.15;  // ★조인트 각도한계(MJCF range). 컨트롤러가 관절을 한계 밖으로 몰면 MuJoCo 클램프와 싸워 붕괴 → q̈에 PD-wall 부등식(한계 margLim_ rad 이내서만 걸림=중앙/정상자세 무간섭). 빈 벡터=미적용.
  void setJointLimits(const vector_t& lo, const vector_t& hi) { qLo_ = lo; qHi_ = hi; }
  double reg_ = 1e-4;  // eiquadprog G positive-definite 정규화(qpOASES 불요, eiquadprog 필수)
  vector_t torqueLimits_;
  int qpFail_ = 0, lastStatus_ = 0, neq_ = 0, nineq_ = 0, dbgN_ = 0, nWsr_ = 20;

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
    M_ = data.M;
    if (rotorArm_.size() == nJ_) for (int j = 0; j < nJ_; ++j) M_(6 + j, 6 + j) += rotorArm_(j);  // ★반사관성=plant dof_armature 정합(순수 관성항, nle 불변)
    nle_ = pinocchio::nonLinearEffects(model, data, qM_, vM_);
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
    // 접촉/스윙 발 위치·속도·★가속도 목표(모델정합 r̈_des = J·u̇_des + J̇·u_des, Bellicoso 식19)
    vector_t uDotDes(nv_); uDotDes.head<6>() = aBaseFF_; uDotDes.tail(nJ_) = jAccel;  // 목표 일반화가속도
    pinocchio::computeJointJacobiansTimeVariation(model, data, qD_, vD_);  // 목표 config의 J̇
    for (int i = 0; i < 4; ++i) {
      posD_[i] = data.oMf[eeId_[i]].translation();
      matrix_t jac = matrix_t::Zero(6, nv_); pinocchio::getFrameJacobian(model, data, eeId_[i], pinocchio::LOCAL_WORLD_ALIGNED, jac);
      velD_[i] = jac.topRows<3>() * vD_;
      matrix_t djac = matrix_t::Zero(6, nv_); pinocchio::getFrameJacobianTimeVariation(model, data, eeId_[i], pinocchio::LOCAL_WORLD_ALIGNED, djac);
      accD_[i] = jac.topRows<3>() * uDotDes + djac.topRows<3>() * vD_;  // 스윙 발 목표 가속도 ff
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
  std::pair<matrix_t, vector_t> noContactMotionTask() {  // planted(스케줄stance∧실접촉): J q̈ = -dJ v
    matrix_t a = matrix_t::Zero(3 * nMotion_, nx_); vector_t b(3 * nMotion_);
    int jc = 0; for (int i = 0; i < nc_; ++i) if (motionC_[i]) { a.block(3 * jc, 0, 3, nv_) = j_.middleRows(3 * i, 3); b.segment(3 * jc, 3) = -dj_.middleRows(3 * i, 3) * vM_ - stanceKd_ * (j_.middleRows(3 * i, 3) * vM_); jc++; }  // ★-KD·Jv=속도댐핑
    return {a, b};
  }
  std::pair<matrix_t, vector_t> ankleTask() {  // ★발목 hard eq: 각 다리 마지막 관절(발목) q̈ = kp(qD-qM)+kd(vD-vM)
    matrix_t a = matrix_t::Zero(nc_, nx_); vector_t b(nc_);
    for (int leg = 0; leg < nc_; ++leg) {
      int jIdx = leg * perLeg_ + (perLeg_ - 1);   // 관절공간 발목 인덱스 (perLeg_==4 → 3,7,11,15)
      a(leg, 6 + jIdx) = 1.0;                       // 결정변수 q̈ 관절부 = base6 + jIdx
      b(leg) = kpAnkle_ * (qD_(6 + jIdx) - qM_(6 + jIdx)) + kdAnkle_ * (vD_(6 + jIdx) - vM_(6 + jIdx));
    }
    return {a, b};
  }
  std::pair<matrix_t, vector_t> frictionEqTask() {  // 비접촉발: f=0
    int ns = nc_ - nForce_;
    matrix_t a = matrix_t::Zero(3 * ns, nx_); int jc = 0;
    for (int i = 0; i < nc_; ++i) if (!forceC_[i]) { a.block(3 * jc++, nv_ + 3 * i, 3, 3).setIdentity(); }
    return {a, vector_t::Zero(3 * ns)};
  }
  std::pair<matrix_t, vector_t> torqueLimitTask() {  // -τlim ≤ τ ≤ τlim  → [I;-I]τ ≤ [lim;lim]
    // ★calf-foot 기구커플(실기 PACE c=1: 발목모터가 q_foot+q_calf 좌표 구동 → τ_calf=τ_km+τ_am·τ_foot=τ_am):
    //   16-DOF(perLeg=4)면 calf 행을 모터공간으로 교체 — |τ_calf−τ_foot|≤126(무릎모터)·|τ_foot|≤100.8(발목모터).
    //   독립박스는 실기 불가능한 코너(τ_calf=126,τ_foot=−100.8→무릎모터226.8) 허용=sim2real 갭.
    matrix_t d = matrix_t::Zero(2 * nJ_, nx_); int off = nv_ + 3 * nc_;
    d.block(0, off, nJ_, nJ_).setIdentity(); d.block(nJ_, off, nJ_, nJ_) = -matrix_t::Identity(nJ_, nJ_);
    vector_t f(2 * nJ_); for (int l = 0; l < 2 * nJ_ / perLeg_; ++l) f.segment(perLeg_ * l, perLeg_) = torqueLimits_;
    if (perLeg_ == 4) for (int lg = 0; lg < nJ_ / perLeg_; ++lg) {
      int c = lg * perLeg_ + 2, ft = lg * perLeg_ + 3;                 // calf·foot 관절 인덱스
      d(c, off + ft) = -1.0; d(nJ_ + c, off + ft) = 1.0;              // ±(τ_calf − τ_foot) ≤ 126
    }
    return {d, f};
  }
  std::pair<matrix_t, vector_t> jointLimitTask() {  // ★조인트 각도한계(MJCF range): q̈에 PD-wall 부등식(d·x≤f). 중앙=완화·한계근접=강함
    matrix_t d = matrix_t::Zero(2 * nJ_, nx_); vector_t f(2 * nJ_);
    for (int j = 0; j < nJ_; ++j) {
      d(2 * j, 6 + j) = 1.0; d(2 * j + 1, 6 + j) = -1.0;
      if (perLeg_ == 4 && j % perLeg_ == perLeg_ - 1) { f(2 * j) = 1e9; f(2 * j + 1) = 1e9; continue; }  // ★발목 제외(정상 운용서 한계 근처=null-space 소프트규제, MuJoCo 클램프 benign)
      double q = qM_(6 + j), qd = vM_(6 + j), du = qHi_(j) - q, dl = q - qLo_(j);
      f(2 * j) = (du > margLim_) ? 1e9 : kpLim_ * du - kdLim_ * qd;      // 상한벽(한계 margLim_ 이내서만)
      f(2 * j + 1) = (dl > margLim_) ? 1e9 : kpLim_ * dl + kdLim_ * qd;  // 하한벽
    }
    return {d, f};
  }
  std::pair<matrix_t, vector_t> frictionIneqTask() {  // 접촉발: pyramid·f ≤ 0
    Eigen::Matrix<double, 5, 3> P; P << 0, 0, -1, 1, 0, -mu_, -1, 0, -mu_, 0, 1, -mu_, 0, -1, -mu_;
    matrix_t d = matrix_t::Zero(5 * nForce_, nx_); int jc = 0;
    for (int i = 0; i < nc_; ++i) if (forceC_[i]) { d.block(5 * jc++, nv_ + 3 * i, 5, 3) = P; }
    return {d, vector_t::Zero(5 * nForce_)};
  }
  std::pair<matrix_t, vector_t> swingTask() {  // 스케줄 스윙발: Cartesian PD (닿아 있어도 들어올림)
    int ns = 0; for (int i = 0; i < nc_; ++i) if (swingC_[i]) ns++;
    matrix_t a = matrix_t::Zero(3 * ns, nx_); vector_t b(3 * ns); int jc = 0;
    swingErrCur_ = 0;
    for (int i = 0; i < nc_; ++i) if (swingC_[i]) {
      { double e = (posD_[i] - posM_[i]).norm(); if (e > swingErrCur_) swingErrCur_ = e; }
      // ★eq19: r̈_des(ff) + Kp·posErr + Kd·velErr  (ff가 스윙 운동 담당 → 피드백 부담↓ → 버징↓)
      vector3_t accel = (swingFF_ ? accD_[i] : vector3_t::Zero()) + swingKp_ * (posD_[i] - posM_[i]) + swingKd_ * (velD_[i] - velM_[i]);
      a.block(3 * jc, 0, 3, nv_) = j_.middleRows(3 * i, 3);
      b.segment(3 * jc, 3) = accel - dj_.middleRows(3 * i, 3) * vM_; jc++;
    }
    return {a, b};
  }
  std::pair<matrix_t, vector_t> postureTask() {  // ★관절 자세 정규화(널스페이스 제어): q̈_joint → PD(관절→desired)
    matrix_t a = matrix_t::Zero(nJ_, nx_);
    a.block(0, 6, nJ_, nJ_).setIdentity();  // 결정변수 q̈의 관절부(인덱스 6..6+nJ) 선택
    vector_t b = kpPost_ * (qD_.tail(nJ_) - qM_.tail(nJ_)) + kdPost_ * (vD_.tail(nJ_) - vM_.tail(nJ_));
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
  int nv_, nJ_, nc_, nx_, numContacts_ = 4, nMotion_ = 4, nForce_ = 4, perLeg_ = 3;
  std::array<bool, 4> contact_{true, true, true, true};
  std::array<bool, 4> motionC_{true, true, true, true}, forceC_{true, true, true, true}, swingC_{false, false, false, false};
  std::array<bool, 4> actualContact_{true, true, true, true}; bool useActual_ = false;
  double mu_ = 0.5;
  vector_t qM_, vM_, qD_, vD_, nle_, inputLast_, last_;
  matrix_t M_, j_, dj_;
  Eigen::Matrix<double, 6, 1> aBaseFF_ = Eigen::Matrix<double, 6, 1>::Zero();
  std::array<vector3_t, 4> posD_, velD_, posM_, velM_, accD_;
};
