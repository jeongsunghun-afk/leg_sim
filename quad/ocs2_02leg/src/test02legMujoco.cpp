// 02_Leg OCS2↔MuJoCo 폐루프 브리지 (Phase 2).
//  루프: MuJoCo 상태 → OCS2 centroidal state → SqpMpc 재계획(MRT) →
//        ff토크(RBD 역동역학) + 관절 PD → MuJoCo ctrl → mj_step.
//  발목(foot)·허리(waist)는 0 홀드(OCS2 point-foot 모델이 발목잠금이므로 정합).
#include <iostream>
#include <iomanip>
#include <array>
#include <cmath>

#include <mujoco/mujoco.h>

#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_legged_robot/gait/MotionPhaseDefinition.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_legged_robot/reference_manager/SwitchedModelReferenceManager.h>
#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_sqp/SqpMpc.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include "wbc_02leg.hpp"

using namespace ocs2;
using namespace legged_robot;

// OCS2 관절순 [FL,FR,HL,HR] x (hip,thigh,calf)
static const char* kJoint[12] = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "HL_hip_joint", "HL_thigh_joint", "HL_calf_joint", "HR_hip_joint", "HR_thigh_joint", "HR_calf_joint"};
static const char* kAnkle[4] = {"FL_foot_joint", "FR_foot_joint", "HL_foot_joint", "HR_foot_joint"};

// 나머지-order euler ZYX(yaw,pitch,roll) from quaternion(wxyz)
static void quat2zyx(const double q[4], double& z, double& y, double& x) {
  double R[9];
  mju_quat2Mat(R, q);  // row-major 3x3
  z = std::atan2(R[3], R[0]);
  y = std::atan2(-R[6], std::sqrt(R[7] * R[7] + R[8] * R[8]));
  x = std::atan2(R[7], R[8]);
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: test02legMujoco <task.info> <urdf> <reference.info> <mjcf> [gait] [simTime]\n";
    return 1;
  }
  const std::string taskFile = argv[1], urdfFile = argv[2], refFile = argv[3], mjcfFile = argv[4];
  const std::string gait = (argc > 5) ? argv[5] : "trot";
  const double simTime = (argc > 6) ? std::atof(argv[6]) : 4.0;

  // ---- OCS2 ----
  std::cerr << "[OCS2] 인터페이스 로드...\n";
  LeggedRobotInterface interface(taskFile, urdfFile, refFile, false);
  const auto& info = interface.getCentroidalModelInfo();
  const int nJ = info.actuatedDofNum;  // 12
  CentroidalModelRbdConversions rbd(interface.getPinocchioInterface(), info);

  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(),
             interface.getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  auto refMgr = std::dynamic_pointer_cast<SwitchedModelReferenceManager>(interface.getReferenceManagerPtr());
  if (gait != "stance") {
    auto tmpl = loadModeSequenceTemplate(std::string(argv[3]).substr(0, refFile.find_last_of('/') + 1) + "gait.info",
                                         gait, false);
    refMgr->getGaitSchedule()->insertModeSequenceTemplate(tmpl, 0.0, simTime + 5.0);
  }
  MPC_MRT_Interface mrt(mpc);
  mrt.initRollout(&interface.getRollout());

  // WBC (weighted QP). 발 프레임 id = contactNames3DoF 순 [FL,FR,HL,HR].
  const bool useWbc = getenv("WBC");
  const char* footFrame[4] = {"FL_foot_contact_link", "FR_foot_contact_link", "HL_foot_contact_link", "HR_foot_contact_link"};
  std::array<int, 4> footId;
  for (int i = 0; i < 4; ++i) footId[i] = interface.getPinocchioInterface().getModel().getFrameId(footFrame[i]);
  Wbc02Leg wbc(interface.getPinocchioInterface(), footId, 0.5);
  if (getenv("W_BASE")) wbc.wBase_ = atof(getenv("W_BASE"));
  if (getenv("W_SW")) wbc.wSwing_ = atof(getenv("W_SW"));
  if (getenv("W_F")) wbc.wForce_ = atof(getenv("W_F"));
  if (getenv("W_REG")) wbc.wReg_ = atof(getenv("W_REG"));
  const int nqPin = interface.getPinocchioInterface().getModel().nq;
  const int nvPin = interface.getPinocchioInterface().getModel().nv;

  // ---- MuJoCo ----
  char err[1000] = "";
  mjModel* m = mj_loadXML(mjcfFile.c_str(), nullptr, err, sizeof(err));
  if (!m) { std::cerr << "MJCF 로드 실패: " << err << "\n"; return 2; }
  mjData* d = mj_makeData(m);

  // 관절/액추에이터 주소 해석
  auto actName = [](const char* jn) { std::string s(jn); return s.substr(0, s.size() - 6); };  // "..._joint" 제거
  int qadr[12], vadr[12], act[12];
  for (int i = 0; i < 12; ++i) {
    int j = mj_name2id(m, mjOBJ_JOINT, kJoint[i]);
    qadr[i] = m->jnt_qposadr[j]; vadr[i] = m->jnt_dofadr[j];
    act[i] = mj_name2id(m, mjOBJ_ACTUATOR, actName(kJoint[i]).c_str());  // FL_hip/FL_thigh/FL_calf
    if (act[i] < 0) { std::cerr << "액추에이터 미발견: " << actName(kJoint[i]) << "\n"; return 4; }
  }
  int aqadr[4], avadr[4], aact[4];
  for (int i = 0; i < 4; ++i) {
    int j = mj_name2id(m, mjOBJ_JOINT, kAnkle[i]);
    aqadr[i] = m->jnt_qposadr[j]; avadr[i] = m->jnt_dofadr[j];
    aact[i] = mj_name2id(m, mjOBJ_ACTUATOR, actName(kAnkle[i]).c_str());
  }
  int wj = mj_name2id(m, mjOBJ_JOINT, "FB_waist_joint");
  int wq = m->jnt_qposadr[wj], wv = m->jnt_dofadr[wj];
  int wact = mj_name2id(m, mjOBJ_ACTUATOR, "FB_waist");

  // 초기 자세 = OCS2 nominal(발목·허리 0)
  const vector_t x0 = interface.getInitialState();
  vector_t jNom = x0.tail(nJ);
  d->qpos[2] = centroidal_model::getBasePose(const_cast<vector_t&>(x0), info)(2);  // base z
  d->qpos[3] = 1; d->qpos[4] = d->qpos[5] = d->qpos[6] = 0;                        // quat identity
  for (int i = 0; i < 12; ++i) d->qpos[qadr[i]] = jNom(i);
  for (int i = 0; i < 4; ++i) d->qpos[aqadr[i]] = 0.0;
  d->qpos[wq] = 0.0;
  mj_forward(m, d);

  // 전진 목표 (VX env, 기본 0.3; stance 격리시 VX=0)
  const double vx = getenv("VX") ? std::atof(getenv("VX")) : 0.3;
  vector_t xGoal = x0;
  centroidal_model::getBasePose(xGoal, info)(0) += vx * simTime;
  interface.getReferenceManagerPtr()->setTargetTrajectories(
      TargetTrajectories({0.0, simTime}, {x0, xGoal}, {vector_t::Zero(info.inputDim), vector_t::Zero(info.inputDim)}));

  // 초기 MPC
  SystemObservation obs;
  obs.time = 0.0; obs.state = x0; obs.input = vector_t::Zero(info.inputDim); obs.mode = ModeNumber::STANCE;
  mrt.setCurrentObservation(obs);
  mrt.advanceMpc();
  mrt.updatePolicy();

  const double dt = m->opt.timestep;
  const double mpcHz = getenv("MPC_HZ") ? std::atof(getenv("MPC_HZ")) : 50.0;
  const int mpcDecim = std::max(1, int((1.0 / mpcHz) / dt));  // 재계획 주기
  const double Kp = getenv("KP") ? std::atof(getenv("KP")) : 60.0;
  const double Kd = getenv("KD") ? std::atof(getenv("KD")) : 2.0;
  const double KpA = 40.0, KdA = 1.5;   // 발목/허리 홀드 PD
  vector_t jAcc = vector_t::Zero(nJ);

  std::cerr << "[SIM] dt=" << dt << " mpcDecim=" << mpcDecim << " gait=" << gait << "\n";
  std::cerr << "  t[s]   base_z   tilt°   base_x   재계획\n";
  int falls = 0; double t = 0;
  for (int step = 0; t < simTime; ++step, t += dt) {
    // --- MuJoCo → rbdState(36) ---
    // rbdState = [eulerZYX(3), position(3), jointPos(nJ), angVel_world(3), linVel_world(3), jointVel(nJ)]
    vector_t rbd_s(6 + nJ + 6 + nJ);
    double z, py, rx; quat2zyx(&d->qpos[3], z, py, rx);
    rbd_s.segment<3>(0) << z, py, rx;                             // euler ZYX
    rbd_s.segment<3>(3) << d->qpos[0], d->qpos[1], d->qpos[2];    // position
    for (int i = 0; i < 12; ++i) rbd_s(6 + i) = d->qpos[qadr[i]];
    double R[9]; mju_quat2Mat(R, &d->qpos[3]);
    double wl[3] = {d->qvel[3], d->qvel[4], d->qvel[5]}, ww[3];
    mju_mulMatVec(ww, R, wl, 3, 3);                              // 각속도 local→world
    rbd_s.segment<3>(6 + nJ) << ww[0], ww[1], ww[2];             // angular vel (world) FIRST
    rbd_s.segment<3>(6 + nJ + 3) << d->qvel[0], d->qvel[1], d->qvel[2];  // linear vel (world)
    for (int i = 0; i < 12; ++i) rbd_s(6 + nJ + 6 + i) = d->qvel[vadr[i]];

    vector_t xMeas = rbd.computeCentroidalStateFromRbdModel(rbd_s);
    if (step == 0 && getenv("DBG")) {
      std::cerr << "  [DBG] x0    = " << x0.transpose() << "\n";
      std::cerr << "  [DBG] xMeas = " << xMeas.transpose() << "\n";
      std::cerr << "  [DBG] diff  = " << (xMeas - x0).transpose() << "\n";
    }

    // --- MPC 재계획(50Hz) ---
    bool replan = (step % mpcDecim == 0);
    if (replan) {
      obs.time = t; obs.state = xMeas;
      mrt.setCurrentObservation(obs);
      try { mrt.advanceMpc(); mrt.updatePolicy(); } catch (const std::exception& e) {}
    }

    // --- 정책 평가 ---
    vector_t xDes, uDes; size_t md;
    mrt.evaluatePolicy(t, xMeas, xDes, uDes, md);

    if (useWbc) {
      // pinocchio 측정 q/v (base quat xyzw, base vel local)
      Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Rm(R);
      vector_t qPin(nqPin), vPin(nvPin);
      qPin.head<3>() << d->qpos[0], d->qpos[1], d->qpos[2];
      qPin.segment<4>(3) << d->qpos[4], d->qpos[5], d->qpos[6], d->qpos[3];  // wxyz→xyzw
      for (int i = 0; i < 12; ++i) qPin(7 + i) = d->qpos[qadr[i]];
      vPin.head<3>() = Rm.transpose() * Eigen::Vector3d(d->qvel[0], d->qvel[1], d->qvel[2]);  // world→local
      vPin.segment<3>(3) << d->qvel[3], d->qvel[4], d->qvel[5];                               // 각속도 local
      for (int i = 0; i < 12; ++i) vPin(6 + i) = d->qvel[vadr[i]];
      // MPC 참조 (RBD 변환: world twist)
      vector_t rbdDes = rbd.computeRbdStateFromCentroidalModel(xDes, uDes);
      Eigen::Vector3d basePosDes = rbdDes.segment<3>(3), eulDes = rbdDes.segment<3>(0);
      Eigen::Matrix3d baseRotDes = (Eigen::AngleAxisd(eulDes(0), Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(eulDes(1), Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(eulDes(2), Eigen::Vector3d::UnitX())).toRotationMatrix();
      Eigen::Vector3d baseAngDes = rbdDes.segment<3>(6 + nJ), baseLinDes = rbdDes.segment<3>(6 + nJ + 3);
      // 목표 발 pos/vel = 목표 배치서 FK
      vector_t qDesPin(nqPin), vDesPin(nvPin);
      qDesPin.head<3>() = basePosDes;
      Eigen::Quaterniond qd(baseRotDes); qDesPin.segment<4>(3) << qd.x(), qd.y(), qd.z(), qd.w();
      qDesPin.tail(12) = rbdDes.segment(6, 12);
      vDesPin.head<3>() = baseRotDes.transpose() * baseLinDes;
      vDesPin.segment<3>(3) = baseRotDes.transpose() * baseAngDes;
      vDesPin.tail(12) = rbdDes.segment(6 + nJ + 6, 12);
      std::array<Eigen::Vector3d, 4> fpDes, fvDes;
      wbc.footFK(qDesPin, vDesPin, fpDes, fvDes);
      // f_des, 접촉플래그
      vector_t fDes(12);
      for (int i = 0; i < 4; ++i) fDes.segment<3>(3 * i) = centroidal_model::getContactForces(uDes, i, info);
      auto cf = modeNumber2StanceLeg(md);
      std::array<bool, 4> stance{cf[0], cf[1], cf[2], cf[3]};
      vector_t tauJ = wbc.compute(qPin, vPin, stance, fpDes, fvDes, basePosDes, baseLinDes, baseRotDes, baseAngDes, fDes);
      if (step == 0 && getenv("DBG")) {
        vector_t tauFF = rbd.computeRbdTorqueFromCentroidalModel(xDes, uDes, jAcc);
        std::cerr << "  [DBG] QP status=" << wbc.lastStatus_ << " fail=" << wbc.qpFail_ << "\n";
        std::cerr << "  [DBG] tau_WBC = " << tauJ.transpose() << "\n";
        std::cerr << "  [DBG] tau_ff  = " << tauFF.tail(nJ).transpose() << "\n";
        std::cerr << "  [DBG] stance  = " << stance[0] << stance[1] << stance[2] << stance[3]
                  << "  fDes= " << fDes.transpose() << "\n";
      }
      for (int i = 0; i < 12; ++i) d->ctrl[act[i]] = tauJ(i);
    } else {
      const bool pdOnly = getenv("PD_ONLY");
      const double ffScale = pdOnly ? 0.0 : 1.0;
      vector_t tau = rbd.computeRbdTorqueFromCentroidalModel(xDes, uDes, jAcc);
      vector_t qDes = pdOnly ? jNom : xDes.tail(nJ);
      vector_t vDes = pdOnly ? vector_t::Zero(nJ) : vector_t(centroidal_model::getJointVelocities(uDes, info));
      const double kp = pdOnly ? 120.0 : Kp, kd = pdOnly ? 4.0 : Kd;
      for (int i = 0; i < 12; ++i)
        d->ctrl[act[i]] = ffScale * tau(6 + i) + kp * (qDes(i) - d->qpos[qadr[i]]) + kd * (vDes(i) - d->qvel[vadr[i]]);
    }
    for (int i = 0; i < 4; ++i)  // 발목 0 홀드
      d->ctrl[aact[i]] = KpA * (0.0 - d->qpos[aqadr[i]]) + KdA * (0.0 - d->qvel[avadr[i]]);
    d->ctrl[wact] = KpA * (0.0 - d->qpos[wq]) + KdA * (0.0 - d->qvel[wv]);  // 허리 0

    mj_step(m, d);

    // --- 진단 ---
    double tilt = std::acos(std::max(-1.0, std::min(1.0, 1 - 2 * (d->qpos[4] * d->qpos[4] + d->qpos[5] * d->qpos[5])))) * 180 / M_PI;
    if (d->qpos[2] < 0.20 || tilt > 60) falls++;
    if (step % int(0.25 / dt) == 0) {
      std::cerr << std::fixed << std::setprecision(3) << "  " << t << "   " << d->qpos[2]
                << "   " << std::setprecision(1) << tilt << "   " << std::setprecision(3) << d->qpos[0]
                << "   " << (replan ? "o" : "") << "\n";
    }
  }

  std::cerr << "\n===== 결과 =====\n";
  std::cerr << "  최종 base_x : " << d->qpos[0] << " m  (목표 " << 0.3 * simTime << ")\n";
  std::cerr << "  최종 base_z : " << d->qpos[2] << " m\n";
  std::cerr << "  낙상 스텝수 : " << falls << (falls == 0 ? "  ✅ falls=0" : "  ✗") << "\n";
  mj_deleteData(d); mj_deleteModel(m);
  return falls == 0 ? 0 : 3;
}
