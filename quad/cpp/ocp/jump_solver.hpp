#pragma once
// S3-b Step2+캐싱 — 점프 OCP. JumpSolver: 무거운 셋업(URDF·MJCF·모델·crouch IK)은 init() 1회,
//   점프마다 solve()만(vx 바뀔 때만 crocoddyl 문제 재구축) → crouch중 stall ~464ms→~solve만.
//   crocoddyl FDDP 다상: push(4접촉,vz↑) → flight(무접촉) → land(4접촉). 허리 lock=16 leg DOF.
//   결과 궤적 = MuJoCo 17-DOF(qpos순, 허리=0) 배포 replay 포맷과 동일 인메모리.
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

#include <crocoddyl/multibody/states/multibody.hpp>
#include <crocoddyl/multibody/actuations/floating-base.hpp>
#include <crocoddyl/multibody/contacts/contact-3d.hpp>
#include <crocoddyl/multibody/contacts/multiple-contacts.hpp>
#include <crocoddyl/multibody/actions/contact-fwddyn.hpp>
#include <crocoddyl/multibody/residuals/state.hpp>
#include <crocoddyl/core/residuals/control.hpp>
#include <crocoddyl/core/costs/cost-sum.hpp>
#include <crocoddyl/core/costs/residual.hpp>
#include <crocoddyl/core/activations/weighted-quadratic.hpp>
#include <crocoddyl/core/activations/quadratic-barrier.hpp>
#include <crocoddyl/core/integrator/euler.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>

// 점프 궤적 결과(MuJoCo 17-DOF qpos순, 허리=0). load_jump 포맷과 동일 필드.
struct JumpTraj {
  int N = 0; double dt = 0.01; bool ok = false;
  std::vector<Eigen::VectorXd> q, dq, tau;   // 각 원소 dim = 관절수(17)
  std::vector<int> ph;                        // push=0 / flight=1 / land=2
  double apex = 0, tilt_land = 0;
};

// 점프 OCP 솔버 — 셋업 1회(init) + 점프별 solve. 뷰어 live-solve용(재사용).
struct JumpSolver {
  // ── vx-독립 셋업(init 1회) ──
  std::shared_ptr<pinocchio::Model> model;
  mjModel* mjm = nullptr; mjData* mjd = nullptr;
  std::vector<pinocchio::FrameIndex> fids;
  std::vector<std::string> FEET = {"FL_foot_contact_link", "FR_foot_contact_link",
                                   "HL_foot_contact_link", "HR_foot_contact_link"};
  int nq = 0, nv = 0; std::size_t nu = 0; int NJ = 0;
  Eigen::VectorXd q_crouch, q_stand, tau_lim;
  std::shared_ptr<crocoddyl::StateMultibody> state;
  std::shared_ptr<crocoddyl::ActuationModelFloatingBase> actu;
  std::vector<int> qi2jid;
  bool ready = false;
  // ── vx-의존 문제 캐시 ──
  double cached_vx = -1e9, dt = 0.01, vz_tk = 0;
  int n_push = 0, n_fly = 0, n_land = 0;
  std::shared_ptr<crocoddyl::ShootingProblem> problem;
  std::shared_ptr<crocoddyl::SolverFDDP> solver;

  ~JumpSolver() { if (mjd) mj_deleteData(mjd); if (mjm) mj_deleteModel(mjm); }

  // ── 무거운 셋업 1회: 모델 빌드 + MJCF 로드 + crouch/stand IK + state/actuation/tau_lim ──
  bool init(const std::string& URDF, const std::string& MJCF) {
    using Eigen::VectorXd; namespace cr = crocoddyl;
    pinocchio::Model full;
    pinocchio::urdf::buildModel(URDF, pinocchio::JointModelFreeFlyer(), full);
    const pinocchio::JointIndex wj = full.getJointId("FB_waist_joint");
    VectorXd q0full = pinocchio::neutral(full);
    model = std::make_shared<pinocchio::Model>();
    pinocchio::buildReducedModel(full, std::vector<pinocchio::JointIndex>{wj}, q0full, *model);
    nq = model->nq; nv = model->nv;
    fids.clear(); for (auto& f : FEET) fids.push_back(model->getFrameId(f));

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

    state = std::make_shared<cr::StateMultibody>(model);
    actu  = std::make_shared<cr::ActuationModelFloatingBase>(state);
    nu = actu->get_nu();
    tau_lim = VectorXd::Constant(nu, 100.0);
    { std::size_t idx = 0;
      for (pinocchio::JointIndex jid = 1; jid < (pinocchio::JointIndex)model->njoints; ++jid) {
        const std::string& nm = model->names[jid];
        double lim = -1;
        if (nm.find("hip") != std::string::npos) lim = 84;
        else if (nm.find("thigh") != std::string::npos) lim = 84;
        else if (nm.find("calf") != std::string::npos) lim = 126;
        else if (nm.find("foot") != std::string::npos) lim = 96;
        if (lim > 0) for (int k = 0; k < model->joints[jid].nv(); ++k) { if (idx < nu) tau_lim[idx++] = lim; }
      }
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

  // ── vx별 crocoddyl 문제 구축(vx 바뀔 때만 호출) ──
  void build_problem(double VX) {
    using Eigen::VectorXd; using Eigen::Vector3d; using Eigen::Vector2d; namespace cr = crocoddyl;
    using SP = std::shared_ptr<cr::ActionModelAbstract>;
    const double G = 9.81;
    auto make_contacts = [&](bool active) {
      auto cm = std::make_shared<cr::ContactModelMultiple>(state, nu);
      if (active)
        for (size_t i = 0; i < FEET.size(); ++i) {
          auto c = std::make_shared<cr::ContactModel3D>(state, fids[i], Vector3d::Zero(),
                                                        pinocchio::LOCAL_WORLD_ALIGNED, nu, Vector2d(0., 50.));
          cm->addContact(FEET[i], c);
        }
      return cm;
    };
    auto costs = [&](bool has_push, double vz_tar, double vx_tar, const VectorXd& xreg_to, double wpush, bool term) {
      auto cs = std::make_shared<cr::CostModelSum>(state, nu);
      VectorXd xref(nq + nv); xref << xreg_to, VectorXd::Zero(nv);
      VectorXd wx(2 * nv);
      wx.setOnes();
      wx.segment(0, 3).setZero(); wx.segment(3, 3).setConstant(300);
      wx.segment(nv, 3).setConstant(1); wx.segment(nv + 3, 3).setConstant(10);
      wx.segment(nv + 6, nv - 6).setConstant(0.1);
      auto act_x = std::make_shared<cr::ActivationModelWeightedQuad>(wx.array().square().matrix());
      auto res_x = std::make_shared<cr::ResidualModelState>(state, xref, nu);
      cs->addCost("xreg", std::make_shared<cr::CostModelResidual>(state, act_x, res_x), term ? 2.0 : 0.2);
      cs->addCost("ureg", std::make_shared<cr::CostModelResidual>(
                              state, std::make_shared<cr::ResidualModelControl>(state, nu)), 1e-3);
      auto bounds = cr::ActivationBounds(-tau_lim, tau_lim);
      auto act_b = std::make_shared<cr::ActivationModelQuadraticBarrier>(bounds);
      cs->addCost("taulim", std::make_shared<cr::CostModelResidual>(
                                state, act_b, std::make_shared<cr::ResidualModelControl>(state, nu)), 1.0);
      if (has_push) {
        VectorXd vtar = VectorXd::Zero(nv); vtar[2] = vz_tar; vtar[0] = vx_tar;
        VectorXd wv = VectorXd::Zero(2 * nv); wv[nv + 2] = 1.0; if (vx_tar != 0.0) wv[nv + 0] = 1.0;
        VectorXd pref(nq + nv); pref << q_stand, vtar;
        auto act_v = std::make_shared<cr::ActivationModelWeightedQuad>(wv.array().square().matrix());
        auto res_v = std::make_shared<cr::ResidualModelState>(state, pref, nu);
        cs->addCost("pushv", std::make_shared<cr::CostModelResidual>(state, act_v, res_v), wpush);
      }
      return cs;
    };
    auto run_model = [&](double dt_, bool contact, bool has_push, double vz_tar, double vx_tar,
                         const VectorXd& xreg_to, double wpush, bool term) -> SP {
      auto dam = std::make_shared<cr::DifferentialActionModelContactFwdDynamics>(
          state, actu, make_contacts(contact), costs(has_push, vz_tar, vx_tar, xreg_to, wpush, term), 0., true);
      return std::make_shared<cr::IntegratedActionModelEuler>(dam, dt_);
    };
    dt = 0.01;
    vz_tk = std::sqrt(2 * G * 0.15);
    n_push = 22; n_land = 40;
    n_fly = std::max(6, (int)(2 * vz_tk / G / dt));
    std::vector<SP> models;
    for (int i = 0; i < n_push; i++) models.push_back(run_model(dt, true,  true,  vz_tk, VX, q_stand, 4.0, false));
    for (int i = 0; i < n_fly;  i++) models.push_back(run_model(dt, false, false, 0, 0, q_crouch, 0.0, false));
    for (int i = 0; i < n_land; i++) models.push_back(run_model(dt, true,  false, 0, 0, q_stand, 0.0, false));
    SP terminal = run_model(dt, true, false, 0, 0, q_stand, 0.0, true);
    VectorXd x0(nq + nv); x0 << q_crouch, VectorXd::Zero(nv);
    problem = std::make_shared<cr::ShootingProblem>(x0, models, terminal);
    solver = std::make_shared<cr::SolverFDDP>(problem);
    cached_vx = VX;
  }

  // ── 점프 궤적 풀이(vx 바뀌면 재구축) ──
  JumpTraj solve(double VX, int MAXIT, bool verbose = false) {
    using Eigen::VectorXd; namespace cr = crocoddyl;
    const double G = 9.81;
    JumpTraj R;
    if (!ready) { std::cerr << "[JumpSolver] init 안됨\n"; return R; }
    if (VX != cached_vx || !problem) build_problem(VX);

    const int T = (int)problem->get_runningModels().size();
    const double D = VX * (2 * vz_tk / G);
    // ── 점프형 ballistic warm-start ──
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
    problem->quasiStatic(us, std::vector<VectorXd>(xs.begin(), xs.end() - 1));

    R.ok = solver->solve(xs, us, MAXIT, false, 1e-4);
    const auto& X = solver->get_xs();
    const auto& U = solver->get_us();
    double zpk = X[0][2]; for (auto& x : X) zpk = std::max(zpk, x[2]);
    R.apex = zpk - q_crouch[2];
    if (verbose)
      std::cout << "[JumpSolver] 수렴=" << R.ok << " iter=" << solver->get_iter()
                << " cost=" << solver->get_cost() << " apex=" << R.apex << "m\n";

    // ── pin 16-DOF → MuJoCo 17-DOF(qpos순, 허리=0) 변환 ──
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
