// 02_Leg OCS2↔MuJoCo 폐루프 브리지 (Phase 2).
//  루프: MuJoCo 상태 → OCS2 centroidal state → SqpMpc 재계획(MRT) →
//        ff토크(RBD 역동역학) + 관절 PD → MuJoCo ctrl → mj_step.
//  발목(foot)·허리(waist)는 0 홀드(OCS2 point-foot 모델이 발목잠금이므로 정합).
#include <iostream>
#include <iomanip>
#include <array>
#include <cmath>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include <cstring>
#include <vector>

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_legged_robot/gait/MotionPhaseDefinition.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_legged_robot/reference_manager/SwitchedModelReferenceManager.h>
#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <ocs2_sqp/SqpMpc.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include "wbc_02leg.hpp"
#include "wbc_legged.hpp"
// ★D1 Phase 3a: perceptive(발-지형 클리어런스 SDF 제약)
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include "mj_terrain_sdf.hpp"
#include "foot_terrain_clearance.hpp"
#include "foot_terrain_placement.hpp"   // ★D1 Phase 3b: 발판배치 제약(A·p+b≥0)
#include "local_convex_region.hpp"      // ★D1 Phase 3b: CGAL 없는 convex 발판영역(박스성장)

using namespace ocs2;
using namespace legged_robot;

// OCS2 관절순 [FL,FR,HL,HR] x (hip,thigh,calf)
static const char* kJoint[12] = {
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "HL_hip_joint", "HL_thigh_joint", "HL_calf_joint", "HR_hip_joint", "HR_thigh_joint", "HR_calf_joint"};
static const char* kAnkle[4] = {"FL_foot_joint", "FR_foot_joint", "HL_foot_joint", "HR_foot_joint"};

// ── GLFW 뷰어(VIEW=1) ──
static mjvCamera cam; static mjvOption vopt; static mjvScene scn; static mjrContext con;
static const mjModel* gM = nullptr; static mjData* gD = nullptr;
static bool bL = false, bR = false, bM = false; static double lx = 0, ly = 0;
static void mbtn(GLFWwindow* w, int, int, int) {
  bL = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
  bR = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
  bM = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
  glfwGetCursorPos(w, &lx, &ly);
}
static void mmove(GLFWwindow* w, double xp, double yp) {
  if (!bL && !bR && !bM) return;
  double dx = xp - lx, dy = yp - ly; lx = xp; ly = yp;
  int W, H; glfwGetWindowSize(w, &W, &H);
  mjtMouse a = bR ? mjMOUSE_MOVE_H : (bL ? mjMOUSE_ROTATE_H : mjMOUSE_ZOOM);
  mjv_moveCamera(gM, a, dx / H, dy / H, &scn, &cam);
}
static void mscroll(GLFWwindow* w, double, double dy) { mjv_moveCamera(gM, mjMOUSE_ZOOM, 0, -0.05 * dy, &scn, &cam); }

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

  // ★D1 Phase 3a: perceptive 발-지형 클리어런스 SDF 제약 주입(PERCEPTIVE=1). MPC 구성 전에 문제에 추가.
  std::shared_ptr<MjTerrainSdf> terrainSdf;
  std::shared_ptr<LocalConvexRegion> region;   // ★Phase 3b: 발판배치 영역(PLACEMENT=1일 때 생성)
  if (getenv("PERCEPTIVE")) {
    terrainSdf = std::make_shared<MjTerrainSdf>();
    auto& prob = interface.getMutableOptimalControlProblem();
    auto pRefMgr = std::dynamic_pointer_cast<SwitchedModelReferenceManager>(interface.getReferenceManagerPtr());
    const auto& ms = interface.modelSettings();
    double clr = getenv("CLEARANCE") ? atof(getenv("CLEARANCE")) : 0.04;
    for (size_t i = 0; i < info.numThreeDofContacts; ++i) {
      const std::string& footName = ms.contactNames3DoF[i];
      const auto infoCppAd = info.toCppAd();
      const CentroidalModelPinocchioMappingCppAd mapCppAd(infoCppAd);
      auto velCb = [infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pAd) {
        const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
        updateCentroidalDynamics(pAd, infoCppAd, q);
      };
      std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKin(new PinocchioEndEffectorKinematicsCppAd(
          interface.getPinocchioInterface(), mapCppAd, {footName}, info.stateDim, info.inputDim,
          velCb, footName + "_perc", ms.modelFolderCppAd, ms.recompileLibrariesCppAd, ms.verboseCppAd));
      std::unique_ptr<PenaltyBase> pen(new RelaxedBarrierPenalty(RelaxedBarrierPenalty::Config(1e-2, 1e-3)));
      std::unique_ptr<StateConstraint> con(new FootTerrainClearanceConstraint(*pRefMgr, *eeKin, terrainSdf, i, clr));
      prob.stateSoftConstraintPtr->add(footName + "_terrainClearance",
          std::unique_ptr<StateCost>(new StateSoftConstraint(std::move(con), std::move(pen))));
    }
    std::cerr << "[PERCEPTIVE] 발-지형 클리어런스 SDF 제약 " << info.numThreeDofContacts << "발 주입(clr=" << clr << ")\n";

    // ★D1 Phase 3b: 발판배치(convex 박스영역) 제약. legged_perceptive FootPlacementConstraint 포팅.
    //   soft·RelaxedBarrier(1e-2,1e-4)(clearance의 1e-3와 구별). A·p+b≥0(박스 4행)=발 XY를 walkable 박스 내로.
    if (getenv("PLACEMENT")) {
      region = std::make_shared<LocalConvexRegion>(terrainSdf);
      for (size_t i = 0; i < info.numThreeDofContacts; ++i) {
        const std::string& footName = ms.contactNames3DoF[i];
        const auto infoCppAd = info.toCppAd();
        const CentroidalModelPinocchioMappingCppAd mapCppAd(infoCppAd);
        auto velCb = [infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pAd) {
          const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
          updateCentroidalDynamics(pAd, infoCppAd, q);
        };
        std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKin(new PinocchioEndEffectorKinematicsCppAd(
            interface.getPinocchioInterface(), mapCppAd, {footName}, info.stateDim, info.inputDim,
            velCb, footName + "_place", ms.modelFolderCppAd, ms.recompileLibrariesCppAd, ms.verboseCppAd));
        std::unique_ptr<PenaltyBase> pen(new RelaxedBarrierPenalty(RelaxedBarrierPenalty::Config(1e-2, 1e-4)));
        std::unique_ptr<StateConstraint> con(new FootTerrainPlacementConstraint(*pRefMgr, *eeKin, region, i));
        prob.stateSoftConstraintPtr->add(footName + "_footPlacement",
            std::unique_ptr<StateCost>(new StateSoftConstraint(std::move(con), std::move(pen))));
      }
      std::cerr << "[PERCEPTIVE] 발판배치 제약 " << info.numThreeDofContacts << "발 주입(soft·RelaxedBarrier 1e-2/1e-4)\n";
    }
  }

  // MPC 백엔드: 기본 SQP / DDP=1 → GaussNewtonDDP(SLQ, Riccati 정규화 내장=trust-region류)
  std::unique_ptr<MPC_BASE> mpcPtr;
  if (getenv("DDP")) {
    std::cerr << "[MPC] GaussNewtonDDP(SLQ) 백엔드\n";
    mpcPtr = std::make_unique<GaussNewtonDDP_MPC>(interface.mpcSettings(), interface.ddpSettings(), interface.getRollout(),
                                                  interface.getOptimalControlProblem(), interface.getInitializer());
  } else {
    mpcPtr = std::make_unique<SqpMpc>(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(),
                                      interface.getInitializer());
  }
  MPC_BASE& mpc = *mpcPtr;
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  auto refMgr = std::dynamic_pointer_cast<SwitchedModelReferenceManager>(interface.getReferenceManagerPtr());
  if (gait != "stance") {
    auto tmpl = loadModeSequenceTemplate(std::string(argv[3]).substr(0, refFile.find_last_of('/') + 1) + "gait.info",
                                         gait, false);
    // ★GAIT_T: gait 주기(총 switching 시간) 스케일 → 고속=짧은 주기(A 속도의존 주기 원리, 고정 0.70s가 ~0.7m/s 상한).
    //   고정 phase가 고속엔 너무 길어 발판 과전방→지지부족→base 침하. 짧은 주기=빠른 cadence로 고속 추종. 미설정=gait.info 그대로.
    if (getenv("GAIT_T") && !tmpl.switchingTimes.empty()) {
      const double want = std::atof(getenv("GAIT_T")), cur = tmpl.switchingTimes.back();
      if (cur > 1e-6 && want > 1e-6) { const double sc = want / cur; for (auto& tt : tmpl.switchingTimes) tt *= sc;
        std::cerr << "[GAIT] 주기 " << cur << "→" << want << "s (고속 스케일)\n"; }
    }
    // SETTLE(env, 기본 1.0s): 시작 STANCE 후 gait 개시 → MPC가 CoM 사전이동 준비.
    const double settle = getenv("SETTLE") ? std::atof(getenv("SETTLE")) : 0.0;
    refMgr->getGaitSchedule()->insertModeSequenceTemplate(tmpl, settle, simTime + 5.0);
  }
  MPC_MRT_Interface mrt(mpc);
  mrt.initRollout(&interface.getRollout());

  // WBC (weighted QP). 발 프레임 id = contactNames3DoF 순 [FL,FR,HL,HR].
  const bool useWbc = getenv("WBC");
  const char* footFrame[4] = {"FL_foot_contact_link", "FR_foot_contact_link", "HL_foot_contact_link", "HR_foot_contact_link"};
  std::array<int, 4> footId;
  for (int i = 0; i < 4; ++i) footId[i] = interface.getPinocchioInterface().getModel().getFrameId(footFrame[i]);
  int baseId = interface.getPinocchioInterface().getModel().getFrameId("Base");
  Wbc02Leg wbc(interface.getPinocchioInterface(), footId, baseId, 0.5);
  // ★legged_control 충실 이식 WBC (WBC_LEGGED=1). [q̈,f,τ]·full EOM·torque limit·FF base.
  const bool wbcLegged = getenv("WBC_LEGGED");
  WbcLegged wbcL(interface.getPinocchioInterface(), info, footId);
  if (getenv("W_SW")) wbcL.wSwing_ = atof(getenv("W_SW"));
  if (getenv("W_BASE")) wbcL.wBase_ = atof(getenv("W_BASE"));
  if (getenv("W_F")) wbcL.wForce_ = atof(getenv("W_F"));
  if (getenv("KP_F")) wbcL.swingKp_ = atof(getenv("KP_F"));
  if (getenv("KD_F")) wbcL.swingKd_ = atof(getenv("KD_F"));
  if (getenv("REG")) wbcL.reg_ = atof(getenv("REG"));
  if (getenv("NWSR")) wbcL.nWsr_ = atoi(getenv("NWSR"));
  if (getenv("SWING_FF")) wbcL.swingFF_ = atoi(getenv("SWING_FF"));
  if (getenv("POST")) wbcL.wPosture_ = atof(getenv("POST"));      // ★널스페이스 posture 가중
  if (getenv("KP_POST")) wbcL.kpPost_ = atof(getenv("KP_POST"));
  if (getenv("KD_POST")) wbcL.kdPost_ = atof(getenv("KD_POST"));
  wbcL.basePd_ = !getenv("NO_BASE_PD");   // ★표준=FF+PD (Bellicoso2016 식17/18): aBaseFF + Kp·posErr + Kd·velErr
  if (getenv("BASE_NOFF")) wbcL.baseNoFF_ = true;   // 순수PD(진단용)
  if (getenv("KP_B")) wbcL.kpBase_ = atof(getenv("KP_B"));
  if (getenv("KD_B")) wbcL.kdBase_ = atof(getenv("KD_B"));
  if (getenv("W_BASE")) wbc.wBase_ = atof(getenv("W_BASE"));
  if (getenv("W_SW")) wbc.wSwing_ = atof(getenv("W_SW"));
  if (getenv("W_F")) wbc.wForce_ = atof(getenv("W_F"));
  if (getenv("W_REG")) wbc.wReg_ = atof(getenv("W_REG"));
  if (getenv("KP_B")) wbc.kpB_ = atof(getenv("KP_B"));
  if (getenv("KD_B")) wbc.kdB_ = atof(getenv("KD_B"));
  if (getenv("KP_O")) wbc.kpO_ = atof(getenv("KP_O"));
  if (getenv("KD_O")) wbc.kdO_ = atof(getenv("KD_O"));
  if (getenv("W_POST")) wbc.wPost_ = atof(getenv("W_POST"));
  if (getenv("BASE_HARD")) wbc.baseHard_ = true;
  if (getenv("SWING_JOINT")) wbc.swingJoint_ = true;
  if (getenv("KP_JS")) wbc.kpJs_ = atof(getenv("KP_JS"));
  if (getenv("KD_JS")) wbc.kdJs_ = atof(getenv("KD_JS"));
  // ★FF_BASE: base task를 MPC 모멘텀률 feedforward로(legged_control식, 전환 강건성 근본fix)
  const bool ffBase = getenv("FF_BASE");
  if (ffBase) {  // FF + moderate PD 피드백(별도 task). 순수FF는 KP_B=0으로.
    wbc.useFF_ = true;
    if (!getenv("KP_B")) wbc.kpB_ = 30; if (!getenv("KD_B")) wbc.kdB_ = 10;
    if (!getenv("KP_O")) wbc.kpO_ = 30; if (!getenv("KD_O")) wbc.kdO_ = 10;
  }
  if (getenv("W_BASE_PD")) wbc.wBasePD_ = atof(getenv("W_BASE_PD"));
  PinocchioInterface pinFF = interface.getPinocchioInterface();  // FF 계산용 별도 사본(오염 방지)
  vector_t uLast = vector_t::Zero(info.inputDim);
  if (getenv("KP_F")) wbc.kpF_ = atof(getenv("KP_F"));
  if (getenv("KD_F")) wbc.kdF_ = atof(getenv("KD_F"));
  const int nqPin = interface.getPinocchioInterface().getModel().nq;
  const int nvPin = interface.getPinocchioInterface().getModel().nv;
  if (getenv("DBG")) {
    const auto& mdl = interface.getPinocchioInterface().getModel();
    std::cerr << "[DBG] OCS2 model 관절순: ";
    for (int j = 2; j < mdl.njoints; ++j) std::cerr << mdl.names[j] << " ";
    std::cerr << "\n";
  }

  // ---- MuJoCo ----
  char err[1000] = "";
  mjModel* m = mj_loadXML(mjcfFile.c_str(), nullptr, err, sizeof(err));
  if (!m) { std::cerr << "MJCF 로드 실패: " << err << "\n"; return 2; }
  mjData* d = mj_makeData(m);

  // ★관절/액추에이터 주소 = OCS2 jointNames 순서로 동적 구성 (12 or 16 DOF 자동 대응)
  auto actName = [](const std::string& jn) { return jn.substr(0, jn.size() - 6); };  // "..._joint" 제거
  const auto& jNames = interface.modelSettings().jointNames;  // nJ개 (12 or 16)
  std::vector<int> qadr(nJ), vadr(nJ), act(nJ);
  for (int i = 0; i < nJ; ++i) {
    int j = mj_name2id(m, mjOBJ_JOINT, jNames[i].c_str());
    if (j < 0) { std::cerr << "관절 미발견: " << jNames[i] << "\n"; return 4; }
    qadr[i] = m->jnt_qposadr[j]; vadr[i] = m->jnt_dofadr[j];
    act[i] = mj_name2id(m, mjOBJ_ACTUATOR, actName(jNames[i]).c_str());
    if (act[i] < 0) { std::cerr << "액추에이터 미발견: " << actName(jNames[i]) << "\n"; return 4; }
  }
  // 제어 안 되는 발목만 0-홀드 대상(12-DOF). 16-DOF는 발목이 jNames에 있어 WBC 제어 → 홀드 없음.
  std::vector<int> holdQ, holdV, holdA;
  for (int i = 0; i < 4; ++i) {
    bool controlled = false; for (int n = 0; n < nJ; ++n) if (jNames[n] == kAnkle[i]) controlled = true;
    if (!controlled) { int j = mj_name2id(m, mjOBJ_JOINT, kAnkle[i]);
      holdQ.push_back(m->jnt_qposadr[j]); holdV.push_back(m->jnt_dofadr[j]); holdA.push_back(mj_name2id(m, mjOBJ_ACTUATOR, actName(kAnkle[i]).c_str())); } }
  std::cerr << "[SIM] nJ=" << nJ << " (제어관절), 홀드발목=" << holdA.size() << "\n";
  // ★발 sphere geom id (실제 접촉 감지용, 순서 [FL,FR,HL,HR]=contactNames3DoF)
  int footGeom[4];
  { const char* sph[4] = {"FL_sphere", "FR_sphere", "HL_sphere", "HR_sphere"};
    for (int i = 0; i < 4; ++i) footGeom[i] = mj_name2id(m, mjOBJ_GEOM, sph[i]); }
  // ★외란 push용 base body id (free joint의 body)
  int baseBody = -1;
  for (int j = 0; j < m->njnt; ++j) if (m->jnt_type[j] == mjJNT_FREE) { baseBody = m->jnt_bodyid[j]; break; }
  int wj = mj_name2id(m, mjOBJ_JOINT, "FB_waist_joint");
  int wq = m->jnt_qposadr[wj], wv = m->jnt_dofadr[wj];
  int wact = mj_name2id(m, mjOBJ_ACTUATOR, "FB_waist");

  // 초기 자세 = OCS2 nominal(발목·허리 0)
  const vector_t x0 = interface.getInitialState();
  vector_t jNom = x0.tail(nJ);
  d->qpos[2] = centroidal_model::getBasePose(const_cast<vector_t&>(x0), info)(2);  // base z
  d->qpos[3] = 1; d->qpos[4] = d->qpos[5] = d->qpos[6] = 0;                        // quat identity
  for (int i = 0; i < nJ; ++i) d->qpos[qadr[i]] = jNom(i);
  for (size_t i = 0; i < holdQ.size(); ++i) d->qpos[holdQ[i]] = 0.0;
  d->qpos[wq] = 0.0;
  mj_forward(m, d);

  // ★Phase 3b: nominal 발-base xy 오프셋(yaw프레임). 초기 nominal stance(yaw=0)서 발 sphere − base.
  //   매틱 발판영역 씨앗 = base_xy + vel·Δt + Rz(yaw)·offset (≈ getNominalFoothold FK, 평지 pitch≈0).
  double footOff[4][2];
  for (int i = 0; i < 4; ++i) {
    footOff[i][0] = d->geom_xpos[3 * footGeom[i] + 0] - d->qpos[0];
    footOff[i][1] = d->geom_xpos[3 * footGeom[i] + 1] - d->qpos[1];
  }
  const double comHeight = 0.50;                                  // reference.info comHeight(지형적응 base높이 기준)
  const double mpcHorizon = interface.mpcSettings().timeHorizon_; // 발판 stanceEnd 탐색·참조 재구성 창

  // 전진 목표 (VX env, 기본 0.3; stance 격리시 VX=0)
  const double vx = getenv("VX") ? std::atof(getenv("VX")) : 0.3;
  vector_t xGoal = x0;
  centroidal_model::getBasePose(xGoal, info)(0) += vx * simTime;
  interface.getReferenceManagerPtr()->setTargetTrajectories(
      TargetTrajectories({0.0, simTime}, {x0, xGoal}, {vector_t::Zero(info.inputDim), vector_t::Zero(info.inputDim)}));

  // ★스레드 연속 MPC (legged_control LeggedController 패턴). MPC_THREAD=1로 활성(기본=동기, 빠른 헤드리스).
  const bool mpcThread = getenv("MPC_THREAD");
  SystemObservation obs;
  obs.time = 0.0; obs.state = x0; obs.input = vector_t::Zero(info.inputDim); obs.mode = ModeNumber::STANCE;
  mrt.setCurrentObservation(obs);
  while (!mrt.initialPolicyReceived()) mrt.advanceMpc();  // 첫 정책 대기
  mrt.updatePolicy();
  std::atomic<bool> mpcAlive{true};
  std::thread mpcWorker;
  if (mpcThread) {
    mpcWorker = std::thread([&]() {
      while (mpcAlive) {
        try { mrt.advanceMpc(); } catch (const std::exception&) {}
      }
    });
  }

  // 뷰어(VIEW=1)
  const bool view = getenv("VIEW");
  GLFWwindow* win = nullptr;
  if (view) {
    if (!glfwInit()) { std::cerr << "glfw init 실패\n"; return 5; }
    win = glfwCreateWindow(1280, 900, "02_Leg OCS2 NMPC + WBC", nullptr, nullptr);
    if (!win) { std::cerr << "창 생성 실패(DISPLAY?)\n"; glfwTerminate(); return 5; }
    glfwMakeContextCurrent(win); glfwSwapInterval(1);
    gM = m; gD = d;
    mjv_defaultCamera(&cam); mjv_defaultOption(&vopt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
    mjv_makeScene(m, &scn, 2000); mjr_makeContext(m, &con, mjFONTSCALE_150);
    glfwSetMouseButtonCallback(win, mbtn); glfwSetCursorPosCallback(win, mmove); glfwSetScrollCallback(win, mscroll);
    cam.distance = 2.5; cam.elevation = -20; cam.azimuth = 120;
  }

  // ★A 제어기와 동일 1kHz(1ms) — MJCF 0.002 오버라이드(동적 접촉 안정 + WBC 균형 대역폭 2배)
  m->opt.timestep = getenv("TIMESTEP") ? std::atof(getenv("TIMESTEP")) : 0.001;
  // ★A와 동일 접촉 강성(solref 시정수 0.02→0.005): 발 침투 35mm→3mm, 동적 접촉 안정(quad_control.hpp:64)
  { double stiff = getenv("STIFF") ? std::atof(getenv("STIFF")) : 0.005;
    for (int g = 0; g < m->ngeom; ++g) { m->geom_solref[g * 2] = stiff; m->geom_solref[g * 2 + 1] = 1.0; } }
  // ★A와 동일 GEARBOX(반사관성+댐핑+마찰): dof_armature=I_rot·N²(MJCF 0→발목 flail 과장 보정). 감속비 hip7·thigh7·calf10.5·foot8.4
  { const char* GN[4] = {"hip", "thigh", "calf", "foot"}; double gear[4] = {7.0, 7.0, 10.5, 8.4};
    bool gbx = !(getenv("GEARBOX") && !std::strcmp(getenv("GEARBOX"), "0"));  // 기본 ON, GEARBOX=0으로만 끔
    double Irot = getenv("ROTOR_I") ? std::atof(getenv("ROTOR_I")) : 1e-4;
    double jdmp = getenv("JDAMP") ? std::atof(getenv("JDAMP")) : 0.1, jfrc = getenv("JFRIC") ? std::atof(getenv("JFRIC")) : 0.5;
    if (gbx) for (int k = 0; k < m->nu; ++k) { int jid = m->actuator_trnid[k * 2]; if (jid < 0) continue;
      const char* jn = mj_id2name(m, mjOBJ_JOINT, jid); if (!jn) continue;
      int gi = 0; for (int g = 0; g < 4; ++g) if (std::strstr(jn, GN[g])) gi = g;  // FB_waist→hip fallback(감속7:1, A와 동일)
      double N = gear[gi]; int dof = m->jnt_dofadr[jid];
      m->dof_armature[dof] = Irot * N * N; m->dof_damping[dof] = jdmp; m->dof_frictionloss[dof] = jfrc; } }
  const double dt = m->opt.timestep;
  const double mpcHz = getenv("MPC_HZ") ? std::atof(getenv("MPC_HZ")) : 50.0;
  const int mpcDecim = std::max(1, int((1.0 / mpcHz) / dt));  // 재계획 주기
  const double Kp = getenv("KP") ? std::atof(getenv("KP")) : 60.0;
  const double Kd = getenv("KD") ? std::atof(getenv("KD")) : 2.0;
  const double KpA = getenv("ANKLE_KP") ? std::atof(getenv("ANKLE_KP")) : 40.0, KdA = getenv("ANKLE_KD") ? std::atof(getenv("ANKLE_KD")) : 1.5;  // 발목 홀드 PD
  const double KpW = getenv("WAIST_KP") ? std::atof(getenv("WAIST_KP")) : 300.0, KdW = getenv("WAIST_KD") ? std::atof(getenv("WAIST_KD")) : 12.0;  // ★허리 단단히 홀드(무거운 몸통분리 DOF)
  vector_t jAcc = vector_t::Zero(nJ);

  std::cerr << "[SIM] dt=" << dt << " mpcDecim=" << mpcDecim << " gait=" << gait << "\n";
  std::cerr << "  t[s]   base_z   tilt°   base_x   재계획\n";
  // 실시간 페이싱: 스레드 MPC가 물리적 재계획 몫을 갖도록 sim을 wall-clock에 맞춤(스레드모드 기본 on).
  const bool pace = mpcThread && !getenv("NO_PACE");
  const auto tWall0 = std::chrono::steady_clock::now();
  int falls = 0; double t = 0;
  double frontJvPk = 0, frontTauPk = 0; vector_t tauPrev = vector_t::Zero(nJ);  // 앞다리 떨림 계측(관절속도·토크변화 피크)
  for (int step = 0; view ? !glfwWindowShouldClose(win) : (t < simTime); ++step, t += dt) {
    // --- MuJoCo → rbdState(36) ---
    // rbdState = [eulerZYX(3), position(3), jointPos(nJ), angVel_world(3), linVel_world(3), jointVel(nJ)]
    vector_t rbd_s(6 + nJ + 6 + nJ);
    double z, py, rx; quat2zyx(&d->qpos[3], z, py, rx);
    rbd_s.segment<3>(0) << z, py, rx;                             // euler ZYX
    rbd_s.segment<3>(3) << d->qpos[0], d->qpos[1], d->qpos[2];    // position
    for (int i = 0; i < nJ; ++i) rbd_s(6 + i) = d->qpos[qadr[i]];
    double R[9]; mju_quat2Mat(R, &d->qpos[3]);
    double wl[3] = {d->qvel[3], d->qvel[4], d->qvel[5]}, ww[3];
    mju_mulMatVec(ww, R, wl, 3, 3);                              // 각속도 local→world
    rbd_s.segment<3>(6 + nJ) << ww[0], ww[1], ww[2];             // angular vel (world) FIRST
    rbd_s.segment<3>(6 + nJ + 3) << d->qvel[0], d->qvel[1], d->qvel[2];  // linear vel (world)
    for (int i = 0; i < nJ; ++i) rbd_s(6 + nJ + 6 + i) = d->qvel[vadr[i]];

    vector_t xMeas = rbd.computeCentroidalStateFromRbdModel(rbd_s);
    if (step == 0 && getenv("DBG")) {
      std::cerr << "  [DBG] x0    = " << x0.transpose() << "\n";
      std::cerr << "  [DBG] xMeas = " << xMeas.transpose() << "\n";
      std::cerr << "  [DBG] diff  = " << (xMeas - x0).transpose() << "\n";
    }

    // ★perceptive: MPC 솔브 전 지형 SDF를 로봇중심 heightmap으로 갱신(in-place=제약이 즉시 봄)
    if (terrainSdf) terrainSdf->update(m, d, d->qpos[0], d->qpos[1]);
    // ★D1 Phase 3b: 발판영역 갱신(PLACEMENT) + 지형적응 base높이 참조(TERRAIN_Z). 재계획 주기(동기모드=레이스 없음).
    if (terrainSdf && step % mpcDecim == 0 && (region || getenv("TERRAIN_Z"))) {
      const double H = mpcHorizon, baseVx = d->qvel[0], baseVy = d->qvel[1], yaw = z;  // z=euler yaw(위 quat2zyx)
      const double cy = std::cos(yaw), sy = std::sin(yaw);
      // (A) 발판영역: 발별 stanceEnd(=initStandFinalTime, ConvexRegionSelector 복제) + nominal 씨앗 → updateFoot
      if (region) {
        auto msched = refMgr->getGaitSchedule()->getModeSchedule(t - H, t + H);
        const auto& ev = msched.eventTimes; const auto& seq = msched.modeSequence;
        const int nP = (int)seq.size();
        for (int i = 0; i < (int)info.numThreeDofContacts; ++i) {
          double stanceEnd_i = 0.0;  // t를 감싸는 stance의 liftoff시각, 스윙이면 0(현재 딛은 발은 안 되당김)
          for (int p = 0; p < nP; ++p) {
            if (!modeNumber2StanceLeg(seq[p])[i]) continue;
            int s = 0;      for (int ip = p - 1; ip >= 0; --ip) if (!modeNumber2StanceLeg(seq[ip])[i]) { s = ip; break; }
            int f = nP - 2; for (int ip = p + 1; ip < nP; ++ip) if (!modeNumber2StanceLeg(seq[ip])[i]) { f = ip - 1; break; }
            if (s < (int)ev.size() && f >= 0 && f < (int)ev.size() && ev[s] < t && t < ev[f]) stanceEnd_i = ev[f];
          }
          double dtm = ((stanceEnd_i > t) ? 0.5 * (t + stanceEnd_i) : t + 0.5 * H) - t;  // stance 중간≈착지 부근
          double rx = cy * footOff[i][0] - sy * footOff[i][1], ry = sy * footOff[i][0] + cy * footOff[i][1];
          region->updateFoot(i, d->qpos[0] + baseVx * dtm + rx, d->qpos[1] + baseVy * dtm + ry, stanceEnd_i);
        }
      }
      // (B) 지형적응 base높이: [t,t+H] 11노드 참조(base z=h+comH/cos pitch·pitch=지형법선). modifyReferences 포팅.
      //   ★base x,y는 원본 절대 forward 램프(x0_x+vx·tn) 유지(modifyReferences가 desired x,y 보존하듯) — z/pitch만 덮어씀.
      if (getenv("TERRAIN_Z")) {
        vector_t x0m = x0; const double x0x = centroidal_model::getBasePose(x0m, info)(0),
                                        x0y = centroidal_model::getBasePose(x0m, info)(1);
        // ★smooth heightmap(box-avg ±SW): legged_perceptive "smooth_planar" 대응. 원 mj_ray 계단 날카로움 완화
        //   → base z가 계단 앞서 점진 상승(급점프 전복 방지). pitch도 step=0.3 넓은 차분(legged_perceptive 동일).
        const double SW = getenv("SMOOTH_W") ? std::atof(getenv("SMOOTH_W")) : 0.14;
        auto hS = [&](double x, double y) { double s = 0; int c = 0;
          for (double dx = -SW; dx <= SW + 1e-9; dx += 0.04) for (double dy = -SW; dy <= SW + 1e-9; dy += 0.04) { s += terrainSdf->height(x + dx, y + dy); ++c; }
          return s / c; };
        const double STEP = 0.3;
        const int N = 11; std::vector<scalar_t> tt(N); std::vector<vector_t> xs(N), us(N);
        for (int n = 0; n < N; ++n) {
          double tn = t + (double)n * H / (N - 1);
          vector_t xn = x0;                                                     // nominal 자세·momentum, base만 지형적응
          double bx = x0x + vx * tn, by = x0y;                                  // 절대 forward 램프(원본 xGoal와 동일)
          double refYaw = centroidal_model::getBasePose(xn, info)(3);            // 직진=0
          double nX = (hS(bx - STEP, by) - hS(bx + STEP, by)) / (2 * STEP);      // n=[-∂h/∂x,-∂h/∂y,1]=법선(smooth·넓은차분)
          double nY = (hS(bx, by - STEP) - hS(bx, by + STEP)) / (2 * STEP);
          double vx_ = std::cos(refYaw) * nX + std::sin(refYaw) * nY;           // (Rz(yaw)ᵀ·n).x
          double pitch = std::atan2(vx_, 1.0);
          centroidal_model::getBasePose(xn, info)(0) = bx;
          centroidal_model::getBasePose(xn, info)(1) = by;
          centroidal_model::getBasePose(xn, info)(4) = pitch;                    // pitch 먼저(z가 읽음)
          centroidal_model::getBasePose(xn, info)(2) = hS(bx, by) + comHeight / std::cos(pitch);
          tt[n] = tn; xs[n] = xn; us[n] = vector_t::Zero(info.inputDim);
        }
        interface.getReferenceManagerPtr()->setTargetTrajectories(TargetTrajectories(std::move(tt), std::move(xs), std::move(us)));
      }
    }
    // --- MPC 관측 공급 + 정책 스왑 ---
    obs.time = t; obs.state = xMeas;
    mrt.setCurrentObservation(obs);       // 최신 상태를 MPC에(스레드 안전)
    bool replan = false;
    if (mpcThread) {
      replan = mrt.updatePolicy();        // 스레드가 푼 최신 정책 스왑(있으면 true)
    } else if (step % mpcDecim == 0) {     // 동기모드(MPC_SYNC)
      try { mrt.advanceMpc(); mrt.updatePolicy(); replan = true; }
      catch (const std::exception& e) { std::cerr << "[MPC FAIL] t=" << t << "  " << e.what() << "\n"; }
    }

    // --- 정책 평가 ---
    vector_t xDes, uDes; size_t md;
    mrt.evaluatePolicy(t, xMeas, xDes, uDes, md);

    // ★실제 MuJoCo 접촉 감지(발별) — 스케줄 모드와 비교/오버라이드용
    bool actC[4] = {false, false, false, false};
    for (int c = 0; c < d->ncon; ++c) {
      int g1 = d->contact[c].geom1, g2 = d->contact[c].geom2;
      for (int i = 0; i < 4; ++i) if (g1 == footGeom[i] || g2 == footGeom[i]) actC[i] = true;
    }
    if (useWbc && wbcLegged) {
      // ★legged_control 충실 이식 WBC: (xDes,uDes,rbd_s,mode,dt)→관절토크(내부서 전부 처리).
      // CONTACT_ACTUAL=1: 접촉력 분배를 스케줄 대신 실제접촉 기준(접촉전환 과지지 방지).
      if (getenv("CONTACT_ACTUAL")) wbcL.setActualContact(actC);
      vector_t tauJ = wbcL.update(xDes, uDes, rbd_s, md, dt);
      // ★표준 저수준(Bellicoso2016·legged_control): τ = τ_ff + τ_pd
      //   τ_ff = τ_wbc(WBC 전신 역동역학 피드포워드), τ_pd = kp(q_des−q)+kd(q̇_des−q̇) 관절추종.
      //   legged: LeggedController.cpp:135(kp=0,kd=3) → LeggedHWSim.cpp:163. velDes=MPC 관절속도(uDes 후반 nJ).
      double jkd = getenv("JKD") ? atof(getenv("JKD")) : 0.0;   // 이상토크 MuJoCo=0(kd=3는 WBC 정확토크와 충돌)
      double jkp = getenv("JKP") ? atof(getenv("JKP")) : 0.0;   // kp(옵션, legged 기본 0)
      for (int i = 0; i < nJ; ++i) {
        double qMeas = rbd_s(6 + i), qdMeas = rbd_s(12 + nJ + i);  // rbd: jointPos@6.., jointVel@(12+nJ)..
        double qDes = xDes(12 + i);                               // centroidal state: momentum(6)+basePose(6)+jointPos(nJ)
        double qdDes = uDes(3 * 4 + i);                           // centroidal input: contactForce(3*4)+jointVel(nJ)
        d->ctrl[act[i]] = tauJ(i) + jkp * (qDes - qMeas) + jkd * (qdDes - qdMeas);  // τ_ff + τ_pd
      }
      // 앞다리(FL,FR = 첫 2다리 = 관절 0~2*perLeg) 떨림 계측: 관절속도 피크 + 토크 변화(채터) 피크
      int nFront = 2 * (nJ / 4);
      for (int i = 0; i < nFront; ++i) { double v = std::abs(d->qvel[vadr[i]]); if (v > frontJvPk) frontJvPk = v;
        double dtau = std::abs(tauJ(i) - tauPrev(i)); if (dtau > frontTauPk) frontTauPk = dtau; }
      for (int i = 0; i < nJ; ++i) tauPrev(i) = tauJ(i);
      if (getenv("FINE") && t > 2.0 && t < 2.4)  // 앞다리(FL_thigh=1) 매스텝: vel·tau·접촉 → 진동/충격 분류
        std::cerr << "  [FINE] t=" << t << " FLthigh_v=" << d->qvel[vadr[1]] << " FLthigh_tau=" << tauJ(1)
                  << " FLcalf_v=" << d->qvel[vadr[2]] << " FLcontact=" << actC[0] << "\n";
      if (getenv("TROT_DBG") && step % int(0.1 / dt) == 0) {
        double tilt2 = std::acos(std::max(-1.0, std::min(1.0, 1 - 2 * (d->qpos[4] * d->qpos[4] + d->qpos[5] * d->qpos[5])))) * 180 / M_PI;
        auto sc = modeNumber2StanceLeg(md);  // 스케줄 stance(0/1) [FL,FR,HL,HR]
        std::cerr << "  [LEGGED] t=" << t << " base_x=" << d->qpos[0] << " base_z=" << d->qpos[2] << " tilt=" << tilt2 << " qpFail=" << wbcL.qpFail_
                  << " nineq=" << wbcL.nineq_ << " frontJvPk=" << frontJvPk << " frontdTauPk=" << frontTauPk
                  << " sched[" << sc[0] << sc[1] << sc[2] << sc[3] << "]\n";
        frontJvPk = 0; frontTauPk = 0;
      }
    } else if (useWbc) {
      // ★OCS2 pinocchio 모델 base = Composite(Translation + SphericalZYX) = euler base(nq=nv=18):
      //   q = [pos(3), eulerZYX(3), joints(12)], v = [linVel_world(3), eulerZYX_rate(3), jointVel(12)].
      // 측정 q/v (MuJoCo → euler base). euler(z,py,rx)·world 각속도 ww는 위 rbdState 구간서 계산됨.
      Eigen::Vector3d eul(z, py, rx);
      Eigen::Vector3d wWorld(ww[0], ww[1], ww[2]);
      vector_t qPin(nqPin), vPin(nvPin);
      qPin.head<3>() << d->qpos[0], d->qpos[1], d->qpos[2];
      qPin.segment<3>(3) = eul;
      for (int i = 0; i < 12; ++i) qPin(6 + i) = d->qpos[qadr[i]];
      vPin.head<3>() << d->qvel[0], d->qvel[1], d->qvel[2];  // world linear
      vPin.segment<3>(3) = getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(eul, wWorld);
      for (int i = 0; i < 12; ++i) vPin(6 + i) = d->qvel[vadr[i]];
      // MPC 참조 (RBD 변환: world twist)
      vector_t rbdDes = rbd.computeRbdStateFromCentroidalModel(xDes, uDes);
      Eigen::Vector3d basePosDes = rbdDes.segment<3>(3), eulDes = rbdDes.segment<3>(0);
      Eigen::Matrix3d baseRotDes = (Eigen::AngleAxisd(eulDes(0), Eigen::Vector3d::UnitZ()) *
                                    Eigen::AngleAxisd(eulDes(1), Eigen::Vector3d::UnitY()) *
                                    Eigen::AngleAxisd(eulDes(2), Eigen::Vector3d::UnitX())).toRotationMatrix();
      Eigen::Vector3d baseAngDes = rbdDes.segment<3>(6 + nJ), baseLinDes = rbdDes.segment<3>(6 + nJ + 3);
      vector_t jointPosDes = rbdDes.segment(6, 12), jointVelDes = rbdDes.segment(6 + nJ + 6, 12);
      // ★진단: HOLD_NOM=고정 nominal 참조(MPC 분리) — WBC 제어 vs MPC 참조 드리프트 격리
      if (getenv("HOLD_NOM")) {
        basePosDes = centroidal_model::getBasePose(const_cast<vector_t&>(x0), info).head<3>();
        eulDes.setZero(); baseRotDes.setIdentity(); baseAngDes.setZero(); baseLinDes.setZero();
        jointPosDes = jNom; jointVelDes.setZero();
      }
      // 목표 발 pos/vel = 목표 배치서 FK (euler base)
      vector_t qDesPin(nqPin), vDesPin(nvPin);
      qDesPin.head<3>() = basePosDes;
      qDesPin.segment<3>(3) = eulDes;
      qDesPin.segment(6, 12) = jointPosDes;
      vDesPin.head<3>() = baseLinDes;  // world linear
      vDesPin.segment<3>(3) = getEulerAnglesZyxDerivativesFromGlobalAngularVelocity<scalar_t>(eulDes, baseAngDes);
      vDesPin.segment(6, 12) = jointVelDes;
      std::array<Eigen::Vector3d, 4> fpDes, fvDes;
      wbc.footFK(qDesPin, vDesPin, fpDes, fvDes);
      // f_des, 접촉플래그
      vector_t fDes(12);
      for (int i = 0; i < 4; ++i) fDes.segment<3>(3 * i) = centroidal_model::getContactForces(uDes, i, info);
      if (getenv("HOLD_NOM")) {  // 균등 중력분배
        double fz = info.robotMass * 9.81 / 4.0;
        for (int i = 0; i < 4; ++i) fDes.segment<3>(3 * i) << 0, 0, fz;
      }
      auto cf = modeNumber2StanceLeg(md);
      std::array<bool, 4> stance{cf[0], cf[1], cf[2], cf[3]};
      if (getenv("TROT_DBG") && step % int(0.1 / dt) == 0) {
        std::array<Eigen::Vector3d, 4> apos, avel; wbc.footFK(qPin, vPin, apos, avel);
        char b[200];
        snprintf(b, sizeof(b), "  t=%.2f st=%d%d%d%d |w|act=%.2f |w|des(MPC)=%.2f baseZdes=%.3f fDes_z[FL]=%.1f",
                 t, stance[0], stance[1], stance[2], stance[3], Eigen::Vector3d(ww[0], ww[1], ww[2]).norm(),
                 baseAngDes.norm(), basePosDes(2), fDes(2));
        std::cerr << b << "\n";
      }
      // ★base 6D 가속 feedforward (legged_control WbcBase::formulateBaseAccelTask):
      //   b = Ab⁻¹·(m·정규화모멘텀률(uDes) − Ȧ·vDes − Aj·q̈_joint). pinFF(별도사본)서 계산해 오염 방지.
      Eigen::Matrix<double, 6, 1> aBaseFF = Eigen::Matrix<double, 6, 1>::Zero();
      if (ffBase) {
        auto& mFF = pinFF.getModel(); auto& dFF = pinFF.getData();
        // ★legged_control updateDesired 순서: FK+프레임배치 먼저(getPositionComToContactPoint용), 그다음 centroidal, 마지막 vel-FK(dccrba용).
        pinocchio::forwardKinematics(mFF, dFF, qDesPin);
        pinocchio::updateFramePlacements(mFF, dFF);
        updateCentroidalDynamics(pinFF, info, qDesPin);
        pinocchio::forwardKinematics(mFF, dFF, qDesPin, vDesPin);
        const auto& Amat = getCentroidalMomentumMatrix(pinFF);         // 6×nv
        Eigen::Matrix<double, 6, 6> Ab = Amat.leftCols<6>();
        Eigen::Matrix<double, 6, 6> AbInv = computeFloatingBaseCentroidalMomentumMatrixInverse(Ab);
        Eigen::MatrixXd Aj = Amat.rightCols(nJ);
        Eigen::MatrixXd ADot = pinocchio::dccrba(mFF, dFF, qDesPin, vDesPin);  // 6×nv
        vector_t jAccel = centroidal_model::getJointVelocities(vector_t(uDes - uLast), info) / dt;
        Eigen::Matrix<double, 6, 1> t1 = info.robotMass * getNormalizedCentroidalMomentumRate(pinFF, info, uDes);
        Eigen::Matrix<double, 6, 1> t2 = ADot * vDesPin;
        Eigen::Matrix<double, 6, 1> t3 = Aj * jAccel;
        Eigen::Matrix<double, 6, 1> momRate = t1 - t2 - t3;
        aBaseFF = AbInv * momRate;
        uLast = uDes;
        if (getenv("FF_DBG") && step % int(0.1 / dt) == 0) {
          std::cerr << "  [FF] t=" << t << " aBaseFF=" << aBaseFF.transpose() << "\n";
          std::cerr << "       t1(mom)=" << t1.transpose() << "\n";
          std::cerr << "       t2(Adot*v)=" << t2.transpose() << "  t3(Aj*qddot)=" << t3.transpose() << "\n";
        }
      }
      vector_t tauJ = wbc.compute(qPin, vPin, stance, fpDes, fvDes, basePosDes, baseLinDes, baseRotDes, baseAngDes, fDes,
                                  jointPosDes, jointVelDes, aBaseFF);
      // WBC 토크 위에 직접 관절 impedance PD(ff+PD의 안정화 요소). stance 발만(swing은 WBC가 담당).
      const double jpdKp = getenv("JPD_KP") ? atof(getenv("JPD_KP")) : 0.0;
      const double jpdKd = getenv("JPD_KD") ? atof(getenv("JPD_KD")) : 0.0;
      for (int i = 0; i < 12; ++i)
        tauJ(i) += jpdKp * (jointPosDes(i) - qPin(6 + i)) + jpdKd * (jointVelDes(i) - vPin(6 + i));
      if (step == 0 && getenv("DBG")) {
        vector_t tauFF = rbd.computeRbdTorqueFromCentroidalModel(xDes, uDes, jAcc);
        std::cerr << "  [DBG] footId=" << footId[0] << "," << footId[1] << "," << footId[2] << "," << footId[3]
                  << " baseId=" << baseId << " nframes=" << interface.getPinocchioInterface().getModel().nframes << "\n";
        std::cerr << "  [DBG] qPin  = " << qPin.transpose() << "\n";
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
    for (size_t i = 0; i < holdA.size(); ++i)  // 제어 안 되는 발목만 0 홀드(12-DOF; 16-DOF는 WBC 제어)
      d->ctrl[holdA[i]] = KpA * (0.0 - d->qpos[holdQ[i]]) + KdA * (0.0 - d->qvel[holdV[i]]);
    d->ctrl[wact] = KpW * (0.0 - d->qpos[wq]) + KdW * (0.0 - d->qvel[wv]);  // 허리 0 (단단히)
    // ★외란 push: PUSH(N) 크기·PUSH_T(s) 시각·PUSH_DUR(s) 지속·PUSH_AX(0=x,1=y,2=z) 방향
    if (getenv("PUSH") && baseBody >= 0) {
      double pf = std::atof(getenv("PUSH")), pt = getenv("PUSH_T") ? std::atof(getenv("PUSH_T")) : 3.0;
      double pdur = getenv("PUSH_DUR") ? std::atof(getenv("PUSH_DUR")) : 0.1;
      int pax = getenv("PUSH_AX") ? std::atoi(getenv("PUSH_AX")) : 1;
      for (int k = 0; k < 3; ++k) d->xfrc_applied[6 * baseBody + k] = 0.0;
      if (t >= pt && t < pt + pdur) { d->xfrc_applied[6 * baseBody + pax] = pf; if (getenv("TROT_DBG") && step % int(0.1 / dt) == 0) std::cerr << "  [PUSH] t=" << t << " F=" << pf << "N ax=" << pax << "\n"; }
    }

    mj_step(m, d);
    // ★진단(WAIST_PIN): 허리를 강체로 고정(OCS2 fixed 모델과 정합) — 앞다리 버징 원인 규명용
    if (getenv("WAIST_PIN")) { d->qpos[wq] = 0; d->qvel[wv] = 0; }
    if (pace) std::this_thread::sleep_until(tWall0 + std::chrono::duration<double>((step + 1) * dt));

    // --- 뷰어 렌더(vsync 페이싱) ---
    if (view && step % 8 == 0) {
      cam.lookat[0] = d->qpos[0]; cam.lookat[1] = d->qpos[1];  // 로봇 추종
      mjrRect vp{0, 0, 0, 0}; glfwGetFramebufferSize(win, &vp.width, &vp.height);
      mjv_updateScene(m, d, &vopt, nullptr, &cam, mjCAT_ALL, &scn);
      mjr_render(vp, &scn, &con);
      char hud[128]; snprintf(hud, sizeof(hud), "t=%.1fs  base_z=%.3f  gait=%s  %s", t, d->qpos[2], gait.c_str(),
                              useWbc ? "WBC" : "ff+PD");
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, vp, "02_Leg OCS2 NMPC+WBC", hud, &con);
      glfwSwapBuffers(win); glfwPollEvents();
    }

    // --- 진단 ---
    double tilt = std::acos(std::max(-1.0, std::min(1.0, 1 - 2 * (d->qpos[4] * d->qpos[4] + d->qpos[5] * d->qpos[5])))) * 180 / M_PI;
    if (d->qpos[2] < 0.20 || tilt > 60) falls++;
    if (step % int(0.25 / dt) == 0) {
      std::cerr << std::fixed << std::setprecision(3) << "  " << t << "   " << d->qpos[2]
                << "   " << std::setprecision(1) << tilt << "   " << std::setprecision(3) << d->qpos[0]
                << "   " << (replan ? "o" : "") << "\n";
    }
  }

  mpcAlive = false;
  if (mpcWorker.joinable()) mpcWorker.join();
  std::cerr << "\n===== 결과 =====\n";
  std::cerr << "  최종 base_x : " << d->qpos[0] << " m  (목표 " << 0.3 * simTime << ")\n";
  std::cerr << "  최종 base_z : " << d->qpos[2] << " m\n";
  std::cerr << "  낙상 스텝수 : " << falls << (falls == 0 ? "  ✅ falls=0" : "  ✗") << "\n";
  if (useWbc) std::cerr << "  WBC QP 실패수 : " << wbc.qpFail_ << "\n";
  if (view) { mjv_freeScene(&scn); mjr_freeContext(&con); glfwTerminate(); }
  mj_deleteData(d); mj_deleteModel(m);
  return falls == 0 ? 0 : 3;
}
