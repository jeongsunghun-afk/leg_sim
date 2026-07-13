#pragma once
// S3-b — 점프 OCP (aligator/proxddp). B/C(simple_mpc)와 동일 라이브러리로 통일.
//   JumpSolver: 무거운 셋업(URDF·MJCF·모델·crouch IK·접촉모델)은 init() 1회, 점프마다 solve()만.
//   3상: push(4접촉,vz·vx↑) → flight(무접촉,ballistic) → land(4접촉,서기회귀). 허리 lock=16 leg DOF.
//   ★토크한계 = 하드 BoxConstraint(AL 처리, 실기 실현성). 결과 궤적 = MuJoCo 17-DOF(qpos순, 허리=0).
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <memory>
#include <Eigen/Dense>

#include <mujoco/mujoco.h>
#include <pinocchio/multibody/joint.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/proximal.hpp>

#include <aligator/modelling/spaces/multibody.hpp>
#include <aligator/modelling/dynamics/multibody-constraint-fwd.hpp>
#include <aligator/modelling/dynamics/integrator-euler.hpp>
#include <aligator/modelling/costs/quad-state-cost.hpp>
#include <aligator/modelling/costs/quad-residual-cost.hpp>
#include <aligator/modelling/costs/sum-of-costs.hpp>
#include <aligator/modelling/state-error.hpp>
#include <aligator/modelling/constraints/box-constraint.hpp>
#include <aligator/core/stage-model.hpp>
#include <aligator/core/traj-opt-problem.hpp>
#include <aligator/solvers/proxddp/solver-proxddp.hpp>

// 점프 궤적 결과(MuJoCo 17-DOF qpos순, 허리=0). load_jump 포맷과 동일 필드.
struct JumpTraj {
  int N = 0; double dt = 0.01; bool ok = false;
  std::vector<Eigen::VectorXd> q, dq, tau;
  std::vector<int> ph;
  std::vector<Eigen::Vector3d> com, comv, acom;   // ★WBIC-추종용 CoM 위치/속도/가속(MuJoCo subtree_com 프레임)
  double apex = 0, tilt_land = 0;
};

struct JumpSolver {
  using MBSpace   = aligator::MultibodyPhaseSpace<double>;
  using ConFwd    = aligator::dynamics::MultibodyConstraintFwdDynamicsTpl<double>;
  using IntEuler  = aligator::dynamics::IntegratorEulerTpl<double>;
  using CostStack = aligator::CostStackTpl<double>;
  using QStateCost= aligator::QuadraticStateCostTpl<double>;
  using QCtrlCost = aligator::QuadraticControlCostTpl<double>;
  using QResCost  = aligator::QuadraticResidualCostTpl<double>;
  using StateErr  = aligator::StateErrorResidualTpl<double>;
  using CtrlErr   = aligator::ControlErrorResidualTpl<double>;
  using BoxCstr   = aligator::BoxConstraintTpl<double>;
  using StageModel= aligator::StageModelTpl<double>;
  using TrajProblem = aligator::TrajOptProblemTpl<double>;
  using Solver    = aligator::SolverProxDDPTpl<double>;
  using ProxSettings = pinocchio::ProximalSettingsTpl<double>;
  using RCModelVec = PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::RigidConstraintModel);

  // ── vx-독립 셋업(init 1회) ──
  std::shared_ptr<pinocchio::Model> model;
  mjModel* mjm = nullptr; mjData* mjd = nullptr;
  std::vector<std::string> FEET = {"FL_foot_contact_link", "FR_foot_contact_link",
                                   "HL_foot_contact_link", "HR_foot_contact_link"};
  int nq = 0, nv = 0; std::size_t nu = 0; int NJ = 0, ndx = 0;
  Eigen::VectorXd q_crouch, q_stand, umin, umax;
  Eigen::MatrixXd actu_mat;
  RCModelVec all_contacts;
  ProxSettings prox = ProxSettings(1e-9, 1e-10, 10);
  std::shared_ptr<MBSpace> space;
  std::vector<int> qi2jid;
  bool ready = false;
  // ── vx-의존 문제 캐시 ──
  double cached_vx = -1e9, dt = 0.01, vz_tk = 0;
  int n_push = 0, n_fly = 0, n_land = 0;
  std::shared_ptr<TrajProblem> problem;
  std::shared_ptr<Solver> solver;

  ~JumpSolver() { if (mjd) mj_deleteData(mjd); if (mjm) mj_deleteModel(mjm); }

  bool init(const std::string& URDF, const std::string& MJCF) {
    using Eigen::VectorXd;
    pinocchio::Model full;
    pinocchio::urdf::buildModel(URDF, pinocchio::JointModelFreeFlyer(), full);
    const pinocchio::JointIndex wj = full.getJointId("FB_waist_joint");
    VectorXd q0full = pinocchio::neutral(full);
    model = std::make_shared<pinocchio::Model>();
    pinocchio::buildReducedModel(full, std::vector<pinocchio::JointIndex>{wj}, q0full, *model);
    nq = model->nq; nv = model->nv; ndx = 2 * nv;

    char merr[1000];
    mjm = mj_loadXML(MJCF.c_str(), nullptr, merr, 1000);
    if (!mjm) { std::cerr << "[JumpSolver] mjcf 로드실패: " << merr << "\n"; return false; }
    mjd = mj_makeData(mjm);
    const char* LEGS[4] = {"HL", "HR", "FL", "FR"};
    const char* JN[4] = {"hip", "thigh", "calf", "foot"};
    int mfgid[4]; double mfr[4];
    for (int i = 0; i < 4; i++) {
      mfgid[i] = mj_name2id(mjm, mjOBJ_GEOM, (std::string(LEGS[i]) + "_sphere").c_str());
      mfr[i] = mjm->geom_size[3 * mfgid[i]];
    }
    auto mj_crouch = [&](double base_z) -> VectorXd {
      if (mjm->nkey > 0) mj_resetDataKeyframe(mjm, mjd, 0);
      else { for (int i = 0; i < mjm->nq; i++) mjd->qpos[i] = 0; mjd->qpos[3] = 1; }
      mjd->qpos[2] = 0.55; mj_forward(mjm, mjd);
      double fxy[4][2];
      for (int i = 0; i < 4; i++) { fxy[i][0] = mjd->geom_xpos[3 * mfgid[i]]; fxy[i][1] = mjd->geom_xpos[3 * mfgid[i] + 1]; }
      mjd->qpos[2] = base_z;
      std::vector<double> jp(3 * mjm->nv);
      for (int it = 0; it < 300; it++) {
        mj_kinematics(mjm, mjd);
        for (int i = 0; i < 4; i++) {
          Eigen::Vector3d e(fxy[i][0] - mjd->geom_xpos[3 * mfgid[i]],
                            fxy[i][1] - mjd->geom_xpos[3 * mfgid[i] + 1],
                            0.0 - (mjd->geom_xpos[3 * mfgid[i] + 2] - mfr[i]));
          mjtNum pnt[3] = {mjd->geom_xpos[3 * mfgid[i]], mjd->geom_xpos[3 * mfgid[i] + 1], mjd->geom_xpos[3 * mfgid[i] + 2]};
          mj_jac(mjm, mjd, jp.data(), nullptr, pnt, mjm->geom_bodyid[mfgid[i]]);
          int cols[4], qa[4];
          for (int k = 0; k < 4; k++) {
            int jid = mj_name2id(mjm, mjOBJ_JOINT, (std::string(LEGS[i]) + "_" + JN[k] + "_joint").c_str());
            cols[k] = mjm->jnt_dofadr[jid]; qa[k] = mjm->jnt_qposadr[jid];
          }
          Eigen::Matrix<double, 3, 4> Jl;
          for (int r = 0; r < 3; r++) for (int k = 0; k < 4; k++) Jl(r, k) = jp[r * mjm->nv + cols[k]];
          Eigen::Vector4d dq = 0.5 * Jl.transpose() *
              (Jl * Jl.transpose() + 1e-4 * Eigen::Matrix3d::Identity()).ldlt().solve(e);
          for (int k = 0; k < 4; k++) mjd->qpos[qa[k]] += dq[k];
        }
      }
      mj_forward(mjm, mjd);
      VectorXd out(mjm->nq); for (int i = 0; i < mjm->nq; i++) out[i] = mjd->qpos[i];
      return out;
    };
    auto mj2pin = [&](const VectorXd& qpos) -> VectorXd {
      VectorXd q = pinocchio::neutral(*model);
      q[0] = qpos[0]; q[1] = qpos[1]; q[2] = qpos[2];
      q[3] = qpos[4]; q[4] = qpos[5]; q[5] = qpos[6]; q[6] = qpos[3];  // wxyz→xyzw
      for (pinocchio::JointIndex jid = 1; jid < (pinocchio::JointIndex)model->njoints; ++jid) {
        int mjid = mj_name2id(mjm, mjOBJ_JOINT, model->names[jid].c_str());
        if (mjid >= 0) q[model->idx_qs[jid]] = qpos[mjm->jnt_qposadr[mjid]];
      }
      return q;
    };
    q_crouch = mj2pin(mj_crouch(0.30));
    q_stand  = mj2pin(mj_crouch(0.50));

    space = std::make_shared<MBSpace>(*model);
    nu = nv - 6;
    // 액추에이션 행렬(nv×nu, 하단 nu행 단위=관절만 구동)
    actu_mat = Eigen::MatrixXd::Zero(nv, nu);
    actu_mat.bottomRows(nu).setIdentity();
    // 토크한계 box(다리 peak, foot 8:1=96)
    umin = Eigen::VectorXd::Constant(nu, -100.0); umax = Eigen::VectorXd::Constant(nu, 100.0);
    { std::size_t idx = 0;
      for (pinocchio::JointIndex jid = 1; jid < (pinocchio::JointIndex)model->njoints; ++jid) {
        const std::string& nm = model->names[jid];
        double lim = -1;
        if (nm.find("hip") != std::string::npos) lim = 84;
        else if (nm.find("thigh") != std::string::npos) lim = 84;
        else if (nm.find("calf") != std::string::npos) lim = 126;
        else if (nm.find("foot") != std::string::npos) lim = 96;
        if (lim > 0) for (int k = 0; k < model->joints[jid].nv(); ++k) { if (idx < nu) { umin[idx] = -lim; umax[idx] = lim; idx++; } }
      }
    }
    // 접촉모델 4발(pinocchio RigidConstraintModel, CONTACT_3D)
    all_contacts.clear();
    for (auto& fn : FEET) {
      pinocchio::FrameIndex fid = model->getFrameId(fn);
      pinocchio::JointIndex jid = model->frames[fid].parentJoint;
      pinocchio::SE3 pl1 = model->frames[fid].placement;
      pinocchio::RigidConstraintModel cm(pinocchio::ContactType::CONTACT_3D, *model, jid, pl1,
                                         0, pinocchio::SE3::Identity(), pinocchio::LOCAL);
      cm.name = fn;   // Baumgarte corrector=기본값(짧은 점프엔 충분)
      all_contacts.push_back(cm);
    }
    // pin→mj 변환 사전계산
    NJ = mjm->nq - 7;
    qi2jid.assign(NJ, -1);
    for (int jid = 0; jid < mjm->njnt; jid++) {
      int qa = mjm->jnt_qposadr[jid];
      if (qa >= 7 && qa < 7 + NJ) qi2jid[qa - 7] = jid;
    }
    ready = true;
    return true;
  }

  // 상태 정규화 가중행렬(ndx×ndx 대각). base 자세 강조.
  Eigen::MatrixXd Wx_mat() const {
    Eigen::VectorXd wx(ndx); wx.setOnes();
    wx.segment(0, 3).setZero(); wx.segment(3, 3).setConstant(300);
    wx.segment(nv, 3).setConstant(1); wx.segment(nv + 3, 3).setConstant(10);
    wx.segment(nv + 6, nv - 6).setConstant(0.1);
    return Eigen::MatrixXd(wx.array().square().matrix().asDiagonal());
  }

  StageModel make_stage(bool contact, bool has_push, double vz_tar, double vx_tar,
                        const Eigen::VectorXd& xreg_to, double wpush) {
    using Eigen::VectorXd; using Eigen::MatrixXd;
    RCModelVec cms; if (contact) cms = all_contacts;   // flight=빈 벡터=free dynamics
    ConFwd ode(*space, actu_mat, cms, prox);
    IntEuler dyn(ode, dt);
    CostStack cost(*space, nu);
    VectorXd xref(nq + nv); xref << xreg_to, VectorXd::Zero(nv);
    cost.addCost("xreg", QStateCost(*space, nu, xref, Wx_mat()), 0.2);
    cost.addCost("ureg", QCtrlCost(*space, VectorXd::Zero(nu), MatrixXd::Identity(nu, nu)), 1e-3);
    if (has_push) {
      VectorXd vtar = VectorXd::Zero(nv); vtar[2] = vz_tar; vtar[0] = vx_tar;
      VectorXd pref(nq + nv); pref << q_stand, vtar;
      VectorXd wv = VectorXd::Zero(ndx); wv[nv + 2] = 1.0; if (vx_tar != 0.0) wv[nv + 0] = 1.0;
      MatrixXd Wv(wv.array().square().matrix().asDiagonal());
      cost.addCost("pushv", QResCost(*space, StateErr(*space, (int)nu, pref), Wv), wpush);
    }
    StageModel stm(cost, dyn);
    stm.addConstraint(CtrlErr(ndx, Eigen::VectorXd::Zero(nu)), BoxCstr(umin, umax));  // ★하드 토크한계
    return stm;
  }

  void build_problem(double VX) {
    using Eigen::VectorXd; using Eigen::MatrixXd;
    const double G = 9.81;
    dt = 0.01; vz_tk = std::sqrt(2 * G * 0.15);
    n_push = 22; n_land = 40;
    n_fly = std::max(6, (int)(2 * vz_tk / G / dt));
    std::vector<xyz::polymorphic<StageModel>> stages;
    for (int i = 0; i < n_push; i++) stages.push_back(make_stage(true,  true,  vz_tk, VX, q_stand, 4.0));
    for (int i = 0; i < n_fly;  i++) stages.push_back(make_stage(false, false, 0, 0, q_crouch, 0.0));
    for (int i = 0; i < n_land; i++) stages.push_back(make_stage(true,  false, 0, 0, q_stand, 0.0));
    // 종단 비용(state, coef 2.0)
    CostStack term(*space, nu);
    VectorXd xref(nq + nv); xref << q_stand, VectorXd::Zero(nv);
    term.addCost("xreg", QStateCost(*space, nu, xref, Wx_mat()), 2.0);
    VectorXd x0(nq + nv); x0 << q_crouch, VectorXd::Zero(nv);
    problem = std::make_shared<TrajProblem>(x0, stages, term);
    solver = std::make_shared<Solver>(1e-4, 1e-2, 200, aligator::QUIET);   // (tol, mu_init, max_iters, verbose)
    solver->rollout_type_ = aligator::RolloutType::LINEAR;
    solver->force_initial_condition_ = true;
    solver->setup(*problem);
    cached_vx = VX;
  }

  JumpTraj solve(double VX, int MAXIT, bool verbose = false) {
    using Eigen::VectorXd;
    const double G = 9.81;
    JumpTraj R;
    if (!ready) { std::cerr << "[JumpSolver] init 안됨\n"; return R; }
    if (VX != cached_vx || !problem) build_problem(VX);
    solver->max_iters = MAXIT;

    const int T = n_push + n_fly + n_land;
    const double D = VX * (2 * vz_tk / G);
    // ballistic warm-start
    std::vector<VectorXd> xs; xs.reserve(T + 1);
    VectorXd x0(nq + nv); x0 << q_crouch, VectorXd::Zero(nv);
    xs.push_back(x0);
    for (int k = 0; k < n_push; k++) {
      double a = (k + 1.0) / n_push;
      VectorXd q = (1 - a) * q_crouch + a * q_stand;
      VectorXd v = VectorXd::Zero(nv); v[2] = a * vz_tk; v[0] = a * VX;
      VectorXd x(nq + nv); x << q, v; xs.push_back(x);
    }
    for (int k = 0; k < n_fly; k++) {
      double tt = k * dt; VectorXd q = q_crouch;
      q[2] = q_stand[2] + vz_tk * tt - 0.5 * G * tt * tt; q[0] = VX * tt;
      VectorXd v = VectorXd::Zero(nv); v[2] = vz_tk - G * tt; v[0] = VX;
      VectorXd x(nq + nv); x << q, v; xs.push_back(x);
    }
    VectorXd qsf = q_stand; qsf[0] = D;
    VectorXd xsf(nq + nv); xsf << qsf, VectorXd::Zero(nv);
    while ((int)xs.size() < T + 1) xs.push_back(xsf);
    xs.resize(T + 1);
    std::vector<VectorXd> us(T, VectorXd::Zero(nu));

    R.ok = solver->run(*problem, xs, us);
    const auto& X = solver->results_.xs;
    const auto& U = solver->results_.us;
    double zpk = X[0][2]; for (auto& x : X) zpk = std::max(zpk, x[2]);
    R.apex = zpk - q_crouch[2];
    if (verbose)
      std::cout << "[JumpSolver] 수렴=" << R.ok << " iter=" << solver->results_.num_iters
                << " cost=" << solver->results_.traj_cost_ << " apex=" << R.apex << "m\n";

    // pin 16-DOF → MuJoCo 17-DOF(qpos순, 허리=0)
    R.N = T; R.dt = dt;
    R.q.clear(); R.dq.clear(); R.tau.clear(); R.ph.clear();
    for (int k = 0; k < T; k++) {
      int ph = (k < n_push) ? 0 : (k < n_push + n_fly ? 1 : 2);
      const VectorXd& xk = X[k]; const VectorXd& uk = U[k];
      VectorXd qj = VectorXd::Zero(NJ), dj = VectorXd::Zero(NJ), tj = VectorXd::Zero(NJ);
      for (int j = 0; j < NJ; j++) {
        int jid = qi2jid[j]; if (jid < 0) continue;
        const char* nmc = mj_id2name(mjm, mjOBJ_JOINT, jid);
        std::string nm = nmc ? nmc : "";
        if (nm == "FB_waist_joint") continue;
        pinocchio::JointIndex pj = model->getJointId(nm);
        if (pj == 0 || pj >= (pinocchio::JointIndex)model->njoints) continue;
        int qs = model->joints[pj].idx_q(), vs = model->joints[pj].idx_v();
        qj[j] = xk[qs]; dj[j] = xk[nq + vs];
        int ui = vs - 6; if (ui >= 0 && ui < (int)nu) tj[j] = uk[ui];
      }
      R.q.push_back(qj); R.dq.push_back(dj); R.tau.push_back(tj); R.ph.push_back(ph);
    }
    // ── ★WBIC-추종용 CoM 참조: 각 노드 full qpos→mj_forward→subtree_com, gradient로 vel/accel ──
    R.com.assign(T, Eigen::Vector3d::Zero());
    for (int k = 0; k < T; k++) {
      const VectorXd& xk = X[k];
      for (int i = 0; i < mjm->nq; i++) mjd->qpos[i] = 0;
      mjd->qpos[0] = xk[0]; mjd->qpos[1] = xk[1]; mjd->qpos[2] = xk[2];
      mjd->qpos[3] = xk[6]; mjd->qpos[4] = xk[3]; mjd->qpos[5] = xk[4]; mjd->qpos[6] = xk[5];  // pin xyzw → mj wxyz
      for (int j = 0; j < NJ; j++) mjd->qpos[7 + j] = R.q[k][j];   // 관절(허리=0)
      for (int i = 0; i < mjm->nv; i++) mjd->qvel[i] = 0;
      mj_forward(mjm, mjd);
      R.com[k] = Eigen::Vector3d(mjd->subtree_com[0], mjd->subtree_com[1], mjd->subtree_com[2]);
    }
    R.comv.assign(T, Eigen::Vector3d::Zero());
    R.acom.assign(T, Eigen::Vector3d::Zero());
    for (int k = 0; k < T; k++) { int kp = std::min(k + 1, T - 1), km = std::max(k - 1, 0);
      double hh = (kp - km) * dt; if (hh > 0) R.comv[k] = (R.com[kp] - R.com[km]) / hh; }
    for (int k = 0; k < T; k++) { int kp = std::min(k + 1, T - 1), km = std::max(k - 1, 0);
      double hh = (kp - km) * dt; if (hh > 0) R.acom[k] = (R.comv[kp] - R.comv[km]) / hh; }
    auto tilt = [&](const VectorXd& x) {
      Eigen::Quaterniond quat(x[6], x[3], x[4], x[5]);
      double r22 = quat.toRotationMatrix()(2, 2);
      return std::acos(std::min(1.0, std::max(-1.0, r22))) * 180.0 / M_PI;
    };
    R.tilt_land = tilt(X[std::min((int)X.size() - 1, n_push + n_fly)]);
    return R;
  }
};

// jump_ocp.cpp(파일생성)용 원샷 wrapper — init+solve 1회.
inline JumpTraj jump_solve(const std::string& URDF, const std::string& MJCF,
                           double VX, int MAXIT, bool verbose = false) {
  JumpSolver s;
  if (!s.init(URDF, MJCF)) return JumpTraj{};
  return s.solve(VX, MAXIT, verbose);
}
