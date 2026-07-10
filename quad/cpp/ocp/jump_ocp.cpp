// S1 — 점프 OCP C++ 포팅 (offline/jump/jump_ocp.py 대응). 실시간 OCP(§9)의 backbone.
//   crocoddyl FDDP 다상: push(4접촉,vz↑) → flight(무접촉) → land(4접촉). 허리 lock=16 leg DOF.
//   q_crouch·q_stand = MuJoCo IK 산물이라 /tmp/ocp_q0.txt(Python DUMP_Q0)에서 로드(IK 포팅은 후속).
//   사용: jump_ocp [URDF] [VX]   VX=전방 이륙속도(기본 0.6). Python parity 검증용.
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

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

using Eigen::VectorXd; using Eigen::Vector3d; using Eigen::Vector2d;
namespace cr = crocoddyl;
using Model = cr::ActionModelAbstract;
using SP = std::shared_ptr<Model>;

int main(int argc, char** argv) {
  const double G = 9.81;
  const std::string URDF = argc > 1 ? argv[1]
      : "/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf";
  const double VX = argc > 2 ? std::atof(argv[2]) : 0.6;

  // ── 모델(허리 lock → reduced 16 leg DOF) ──
  pinocchio::Model full;
  pinocchio::urdf::buildModel(URDF, pinocchio::JointModelFreeFlyer(), full);
  const pinocchio::JointIndex wj = full.getJointId("FB_waist_joint");
  VectorXd q0full = pinocchio::neutral(full);
  auto model = std::make_shared<pinocchio::Model>();
  pinocchio::buildReducedModel(full, std::vector<pinocchio::JointIndex>{wj}, q0full, *model);
  pinocchio::Data data(*model);
  const int nq = model->nq, nv = model->nv;
  const std::vector<std::string> FEET = {"FL_foot_contact_link", "FR_foot_contact_link",
                                         "HL_foot_contact_link", "HR_foot_contact_link"};
  std::vector<pinocchio::FrameIndex> fids;
  for (auto& f : FEET) fids.push_back(model->getFrameId(f));

  // ── q_crouch·q_stand 로드(/tmp/ocp_q0.txt) ──
  std::ifstream qf("/tmp/ocp_q0.txt");
  if (!qf) { std::cerr << "q0 파일 없음 → DUMP_Q0=/tmp/ocp_q0.txt python jump_ocp.py 먼저 실행\n"; return 2; }
  int fnq, fnv; qf >> fnq >> fnv;
  VectorXd q_crouch(nq), q_stand(nq);
  for (int i = 0; i < nq; i++) qf >> q_crouch[i];
  for (int i = 0; i < nq; i++) qf >> q_stand[i];
  VectorXd x0(nq + nv); x0 << q_crouch, VectorXd::Zero(nv);

  // ── state·actuation ──
  auto state = std::make_shared<cr::StateMultibody>(model);
  auto actu  = std::make_shared<cr::ActuationModelFloatingBase>(state);
  const std::size_t nu = actu->get_nu();

  // 토크한계(다리 peak, foot 8:1=96). 매칭 관절만 순서대로(freeflyer 제외).
  VectorXd tau_lim = VectorXd::Constant(nu, 100.0);
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

  // ── 헬퍼: 접촉 모델 ──
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
  // ── 헬퍼: 비용 ──
  auto costs = [&](bool has_push, double vz_tar, double vx_tar, const VectorXd& xreg_to,
                   double wpush, bool term) {
    auto cs = std::make_shared<cr::CostModelSum>(state, nu);
    VectorXd xref(nq + nv); xref << xreg_to, VectorXd::Zero(nv);
    VectorXd wx(2 * nv);            // base 자세 강조: [0,0,0,300,300,300, 1×(nv-6) | 1,1,1,10,10,10, 0.1×(nv-6)]
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
  auto run_model = [&](double dt, bool contact, bool has_push, double vz_tar, double vx_tar,
                       const VectorXd& xreg_to, double wpush, bool term) -> SP {
    auto dam = std::make_shared<cr::DifferentialActionModelContactFwdDynamics>(
        state, actu, make_contacts(contact), costs(has_push, vz_tar, vx_tar, xreg_to, wpush, term), 0., true);
    return std::make_shared<cr::IntegratedActionModelEuler>(dam, dt);
  };

  // ── 스케줄 ──
  const double dt = 0.01;
  const double vz_tk = std::sqrt(2 * G * 0.15);
  const double T_fly = 2 * vz_tk / G, D = VX * T_fly;
  const int n_push = 22, n_land = 40;
  const int n_fly = std::max(6, (int)(2 * vz_tk / G / dt));
  std::vector<SP> models;
  for (int i = 0; i < n_push; i++) models.push_back(run_model(dt, true,  true,  vz_tk, VX, q_stand, 4.0, false));
  for (int i = 0; i < n_fly;  i++) models.push_back(run_model(dt, false, false, 0, 0, q_crouch, 0.0, false));
  for (int i = 0; i < n_land; i++) models.push_back(run_model(dt, true,  false, 0, 0, q_stand, 0.0, false));
  SP terminal = run_model(dt, true, false, 0, 0, q_stand, 0.0, true);
  std::cout << "[ocp] 스케줄 push=" << n_push << " flight=" << n_fly << " land=" << n_land
            << " (T=" << (n_push + n_fly + n_land) * dt << "s)\n";

  auto problem = std::make_shared<cr::ShootingProblem>(x0, models, terminal);
  cr::SolverFDDP solver(problem);

  // ── 점프형 warm-start ──
  std::vector<VectorXd> xs; xs.reserve(models.size() + 1);
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
  while ((int)xs.size() < (int)models.size() + 1) xs.push_back(xsf);
  xs.resize(models.size() + 1);

  std::vector<VectorXd> us(models.size(), VectorXd::Zero(nu));   // ★void quasiStatic=us 사전할당(T개) 규약
  problem->quasiStatic(us, std::vector<VectorXd>(xs.begin(), xs.end() - 1));

  std::cout << "[ocp] FDDP 풀이…\n";
  bool ok = solver.solve(xs, us, 200, false, 1e-4);
  const auto& X = solver.get_xs();
  double z0 = X[0][2], zpk = z0, zend = X.back()[2];
  for (auto& x : X) zpk = std::max(zpk, x[2]);
  std::cout << "[ocp] 수렴=" << ok << " iter=" << solver.get_iter() << " cost=" << solver.get_cost() << "\n";
  std::cout << "[ocp] base_z: 시작 " << z0 << " → peak " << zpk << " (apex " << (zpk - q_crouch[2])
            << "m) → 끝 " << zend << "\n";
  // 착지 tilt(quat→R[2,2])
  auto tilt = [&](const VectorXd& x) {
    Eigen::Quaterniond quat(x[6], x[3], x[4], x[5]);  // (w,x,y,z)
    double r22 = quat.toRotationMatrix()(2, 2);
    return std::acos(std::min(1.0, std::max(-1.0, r22))) * 180.0 / M_PI;
  };
  std::cout << "[ocp] tilt: peak중 " << tilt(X[n_push + n_fly / 2]) << "° 착지 " << tilt(X[n_push + n_fly]) << "°\n";
  return ok ? 0 : 1;
}
