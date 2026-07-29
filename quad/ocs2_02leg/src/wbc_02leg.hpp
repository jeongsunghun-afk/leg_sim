// 02_Leg WBC — weighted QP whole-body controller (legged_control 스타일).
//  결정변수 x = [q̈(nv), f(3*4)].
//  hard: floating-base 동역학(6) · stance no-slip(3/발) · swing force=0(3/발) · friction cone.
//  soft: base 가속 추종 · swing 발 가속 추종 · 접촉력(MPC f_des) 추종 · q̈ 정규화.
//  τ = [M q̈ + h − Jcᵀ f]_actuated.
#pragma once
#include <Eigen/Dense>
#include <array>

#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <eiquadprog/eiquadprog-fast.hpp>

class Wbc02Leg {
 public:
  Wbc02Leg(ocs2::PinocchioInterface pin, const std::array<int, 4>& footFrameId, int baseFrameId, double mu = 0.5)
      : pin_(std::move(pin)), footId_(footFrameId), baseId_(baseFrameId), mu_(mu) {
    nv_ = pin_.getModel().nv;              // 18
    nJ_ = nv_ - 6;                         // 12
    nx_ = nv_ + 12;                        // q̈ + 4*3 force
  }

  // q_pin(nq), v_pin(nv): pinocchio 규약(base quat xyzw, base vel local).
  // stance[4]: 접촉 플래그. footPosDes/footVelDes[4]: swing 목표(world). basePosDes/eulDes/linDes/angDes: base 목표(world).
  // fDes(12): MPC 접촉력(발0..3 각 xyz, world). 반환=관절토크(12, pinocchio 관절순).
  Eigen::VectorXd compute(const Eigen::VectorXd& q, const Eigen::VectorXd& v, const std::array<bool, 4>& stance,
                          const std::array<Eigen::Vector3d, 4>& footPosDes, const std::array<Eigen::Vector3d, 4>& footVelDes,
                          const Eigen::Vector3d& basePosDes, const Eigen::Vector3d& baseLinDes,
                          const Eigen::Matrix3d& baseRotDes, const Eigen::Vector3d& baseAngDes,
                          const Eigen::VectorXd& fDes, const Eigen::VectorXd& jointPosDes, const Eigen::VectorXd& jointVelDes) {
    using namespace Eigen;
    auto& model = pin_.getModel();
    auto& data = pin_.getData();
    const int nv = nv_, nJ = nJ_, nx = nx_;

    if (getenv("WBC_DBG")) {
      // 깨끗한 FK로 발 위치 확인
      pinocchio::forwardKinematics(model, data, q);
      pinocchio::updateFramePlacements(model, data);
      for (int i = 0; i < 4; ++i)
        std::cerr << "  [WBC] frame[" << footId_[i] << "]=" << model.frames[footId_[i]].name
                  << " pos=" << data.oMf[footId_[i]].translation().transpose() << "\n";
    }
    // --- 동역학 ---
    pinocchio::crba(model, data, q);
    data.M.triangularView<StrictlyLower>() = data.M.transpose().triangularView<StrictlyLower>();
    MatrixXd M = data.M;
    VectorXd h = pinocchio::nonLinearEffects(model, data, q, v);  // C v + g
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);
    // 발 Jacobian(3) + drift J̇v (q̈=0 classical accel)
    pinocchio::forwardKinematics(model, data, q, v, VectorXd::Zero(nv));
    std::array<MatrixXd, 4> Jf;
    std::array<Vector3d, 4> Jdv, footPos, footVel;
    for (int i = 0; i < 4; ++i) {
      MatrixXd J6 = MatrixXd::Zero(6, nv);
      pinocchio::getFrameJacobian(model, data, footId_[i], pinocchio::LOCAL_WORLD_ALIGNED, J6);
      Jf[i] = J6.topRows<3>();
      Jdv[i] = pinocchio::getFrameClassicalAcceleration(model, data, footId_[i], pinocchio::LOCAL_WORLD_ALIGNED).linear();
      footPos[i] = data.oMf[footId_[i]].translation();
      footVel[i] = (Jf[i] * v);
    }
    // 접촉 Jacobian 스택 Jc(12×nv), Jcᵀ
    MatrixXd Jc(12, nv);
    for (int i = 0; i < 4; ++i) Jc.middleRows<3>(3 * i) = Jf[i];
    // base 링크 6D Jacobian + drift (발 task와 동일한 LOCAL_WORLD_ALIGNED 프레임 방식)
    MatrixXd Jbase = MatrixXd::Zero(6, nv);
    pinocchio::getFrameJacobian(model, data, baseId_, pinocchio::LOCAL_WORLD_ALIGNED, Jbase);
    Matrix<double, 6, 1> JdvBase;
    JdvBase.head<3>() = pinocchio::getFrameClassicalAcceleration(model, data, baseId_, pinocchio::LOCAL_WORLD_ALIGNED).linear();
    JdvBase.tail<3>() = pinocchio::getFrameClassicalAcceleration(model, data, baseId_, pinocchio::LOCAL_WORLD_ALIGNED).angular();
    Vector3d basePos = data.oMf[baseId_].translation();
    Matrix3d baseRot = data.oMf[baseId_].rotation();
    Matrix<double, 6, 1> baseVel = Jbase * v;  // world (linear, angular)

    // --- soft 비용: G, g0 ---
    MatrixXd G = MatrixXd::Zero(nx, nx);
    VectorXd g0 = VectorXd::Zero(nx);
    auto addTask = [&](const MatrixXd& A, const VectorXd& b, double w) {
      G.noalias() += w * A.transpose() * A;
      g0.noalias() -= w * A.transpose() * b;
    };
    // base 6D 목표 가속(world). baseHard_면 hard 제약, 아니면 soft task.
    Vector3d oriErr = 0.5 * unskew(baseRotDes * baseRot.transpose() - baseRot * baseRotDes.transpose());
    Matrix<double, 6, 1> aBaseDes;
    aBaseDes.head<3>() = kpB_ * (basePosDes - basePos) + kdB_ * (baseLinDes - baseVel.head<3>());
    aBaseDes.tail<3>() = kpO_ * oriErr + kdO_ * (baseAngDes - baseVel.tail<3>());
    if (!baseHard_) {
      MatrixXd A = MatrixXd::Zero(6, nx); A.block(0, 0, 6, nv) = Jbase;
      addTask(A, VectorXd(aBaseDes - JdvBase), wBase_);
    }
    // swing 발 가속(3/발): J_sw q̈ = a_sw_des − J̇v
    for (int i = 0; i < 4; ++i) {
      if (stance[i]) continue;
      MatrixXd A = MatrixXd::Zero(3, nx); A.block(0, 0, 3, nv) = Jf[i];
      Vector3d aDes = kpF_ * (footPosDes[i] - footPos[i]) + kdF_ * (footVelDes[i] - footVel[i]);
      addTask(A, VectorXd(aDes - Jdv[i]), wSwing_);
    }
    // 접촉력 추종(12): f = f_des
    {
      MatrixXd A = MatrixXd::Zero(12, nx); A.block(0, nv, 12, 12) = MatrixXd::Identity(12, 12);
      addTask(A, fDes, wForce_);
    }
    // 관절 posture task(12): q̈_joint = kpJ(qDes−q) + kdJ(vDes−v) — ff+PD의 관절 PD 대응
    {
      MatrixXd A = MatrixXd::Zero(nJ, nx); A.block(0, 6, nJ, nJ) = MatrixXd::Identity(nJ, nJ);
      VectorXd b = kpJ_ * (jointPosDes - q.tail(nJ)) + kdJ_ * (jointVelDes - v.tail(nJ));
      addTask(A, b, wPost_);
    }
    // q̈ 정규화
    { MatrixXd A = MatrixXd::Zero(nv, nx); A.block(0, 0, nv, nv) = MatrixXd::Identity(nv, nv); addTask(A, VectorXd::Zero(nv), wReg_); }
    G.diagonal().array() += 1e-8;

    // --- hard equality: CE x + ce0 = 0 ---
    std::vector<Triplet> ceRows; VectorXd ce0v;
    // (1) floating-base 동역학(6): M[0:6] q̈ − Jcᵀ[0:6] f + h[0:6] = 0
    MatrixXd JcT = Jc.transpose();  // nv×12
    MatrixXd CE_base(6, nx); CE_base << M.topRows<6>(), -JcT.topRows<6>();
    // (2) base 6D(옵션 hard), (3) stance no-slip, (4) swing zero-force
    int nEq = 6 + (baseHard_ ? 6 : 0) + 4 * 3;
    MatrixXd CE = MatrixXd::Zero(nEq, nx); VectorXd ce0 = VectorXd::Zero(nEq);
    CE.topRows<6>() = CE_base; ce0.head<6>() = h.head<6>();
    int r = 6;
    if (baseHard_) {  // Jbase q̈ + (JdvBase − aBaseDes) = 0
      CE.block(r, 0, 6, nv) = Jbase; ce0.segment<6>(r) = JdvBase - aBaseDes; r += 6;
    }
    for (int i = 0; i < 4; ++i) {
      if (stance[i]) {  // J_st q̈ + J̇v = 0
        CE.block(r, 0, 3, nv) = Jf[i]; ce0.segment<3>(r) = Jdv[i];
      } else {          // f_sw = 0
        CE.block(r, nv + 3 * i, 3, 3) = Matrix3d::Identity();
      }
      r += 3;
    }

    // --- inequality: CI x + ci0 ≥ 0 (friction, stance만) ---
    int nStance = 0; for (bool s : stance) if (s) nStance++;
    MatrixXd CI = MatrixXd::Zero(5 * nStance, nx); VectorXd ci0 = VectorXd::Zero(5 * nStance);
    int c = 0;
    for (int i = 0; i < 4; ++i) {
      if (!stance[i]) continue;
      int fx = nv + 3 * i, fy = fx + 1, fz = fx + 2;
      CI(c, fz) = 1; ci0(c) = -fzMin_; c++;                 // fz ≥ fzMin
      CI(c, fz) = mu_; CI(c, fx) = -1; c++;                 // μfz − fx ≥ 0
      CI(c, fz) = mu_; CI(c, fx) = 1; c++;                  // μfz + fx ≥ 0
      CI(c, fz) = mu_; CI(c, fy) = -1; c++;                 // μfz − fy ≥ 0
      CI(c, fz) = mu_; CI(c, fy) = 1; c++;                  // μfz + fy ≥ 0
    }

    // --- solve ---
    VectorXd x(nx);
    eiquadprog::solvers::EiquadprogFast qp;
    qp.reset(nx, nEq, 5 * nStance);
    auto st = qp.solve_quadprog(G, g0, CE, ce0, CI, ci0, x);
    if (st != eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) { qpFail_++; x = lastX_.size() == nx ? lastX_ : VectorXd::Zero(nx); }
    else lastX_ = x;
    lastStatus_ = st;

    // --- 토크: τ = [M q̈ + h − Jcᵀ f]_actuated ---
    VectorXd qacc = x.head(nv), f = x.tail(12);
    VectorXd full = M * qacc + h - JcT * f;
    if (getenv("WBC_DBG")) {
      VectorXd g = pinocchio::computeGeneralizedGravity(model, data, q);
      std::cerr << "  [WBC] g.tail   = " << g.tail(nJ).transpose() << "\n";
      std::cerr << "  [WBC] h.tail   = " << h.tail(nJ).transpose() << "  |v|=" << v.norm() << "\n";
      std::cerr << "  [WBC] (JcTf).tl= " << (JcT * fDes).tail(nJ).transpose() << "\n";
      std::cerr << "  [WBC] tauGrav  = " << (h - JcT * fDes).tail(nJ).transpose() << "\n";
      std::cerr << "  [WBC] |qacc|=" << qacc.norm() << " |f-fDes|=" << (f - fDes).norm()
                << " |Jf0 q̈+Jdv0|=" << (Jf[0] * qacc + Jdv[0]).norm() << "\n";
      std::cerr << "  [WBC] footPos0=" << footPos[0].transpose() << "\n";
    }
    return full.tail(nJ);
  }

  // 목표 발 pos/vel(world) = 목표 관절배치서 FK. compute() 전에 호출(내부 data는 compute서 재계산).
  void footFK(const Eigen::VectorXd& q, const Eigen::VectorXd& v, std::array<Eigen::Vector3d, 4>& pos,
              std::array<Eigen::Vector3d, 4>& vel) {
    auto& model = pin_.getModel(); auto& data = pin_.getData();
    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::updateFramePlacements(model, data);
    pinocchio::computeJointJacobians(model, data, q);
    for (int i = 0; i < 4; ++i) {
      pos[i] = data.oMf[footId_[i]].translation();
      Eigen::MatrixXd J6 = Eigen::MatrixXd::Zero(6, nv_);
      pinocchio::getFrameJacobian(model, data, footId_[i], pinocchio::LOCAL_WORLD_ALIGNED, J6);
      vel[i] = J6.topRows<3>() * v;
    }
  }

  double kpB_ = 100, kdB_ = 20, kpO_ = 100, kdO_ = 20, kpF_ = 400, kdF_ = 40, kpJ_ = 100, kdJ_ = 10;
  double wBase_ = 10, wSwing_ = 20, wForce_ = 1, wReg_ = 1e-3, wPost_ = 0.0, fzMin_ = 1.0;  // wPost>0=악화 확인
  bool baseHard_ = false;  // base 6D task를 hard 제약으로(강한 자세 안정화)
  int qpFail_ = 0, lastStatus_ = 0;

 private:
  using Triplet = Eigen::Triplet<double>;
  static Eigen::Matrix3d q2R(const Eigen::VectorXd& q) {
    Eigen::Quaterniond quat(q(6), q(3), q(4), q(5));  // pinocchio xyzw → Quaterniond(w,x,y,z)
    return quat.toRotationMatrix();
  }
  static Eigen::Vector3d unskew(const Eigen::Matrix3d& S) { return Eigen::Vector3d(S(2, 1), S(0, 2), S(1, 0)); }

  ocs2::PinocchioInterface pin_;
  std::array<int, 4> footId_;
  int baseId_;
  double mu_;
  int nv_, nJ_, nx_;
  Eigen::VectorXd lastX_;
};
