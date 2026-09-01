// 02_Leg OCS2↔MuJoCo 폐루프 브리지 (Phase 2).
//  루프: MuJoCo 상태 → OCS2 centroidal state → SqpMpc 재계획(MRT) →
//        ff토크(RBD 역동역학) + 관절 PD → MuJoCo ctrl → mj_step.
//  발목(foot)·허리(waist)는 0 홀드(OCS2 point-foot 모델이 발목잠금이므로 정합).
#include <iostream>
#include <fstream>
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
#include <ocs2_core/misc/LinearInterpolation.h>
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
  // ★결정론: 라이브러리 OpenMP 병렬(pinocchio/Eigen 리덕션 순서)이 run-to-run 편차의 주 원인 —
  //   OMP_NUM_THREADS=1로 제거(검증: default falls 편차 대폭↓, 마진 튜닝 신뢰 회복). 벤치용 override 가능.
  if (!getenv("OMP_NUM_THREADS")) setenv("OMP_NUM_THREADS", "1", 1);
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
  // ★발판 마커=MPC 실제 계획 발 착지점 시각화용(제어 pinocchio 비오염 복사 + 매핑)
  PinocchioInterface vizPin = interface.getPinocchioInterface();
  CentroidalModelPinocchioMapping vizMap(info); vizMap.setPinocchioInterface(vizPin);
  // ★게이트별 base task 가중 기본값(2026-08-10): trot=150(base 회복 authority↑). ⚠trot 전용 — bound/static_walk는
  //   150에서 붕괴(falls>2000)하므로 50 유지. env W_BASE로 override. (엔벨로프 절대치=반사관성 정합 후 실측물리 기준: trot/bound≤0.8.)
  if (gait == "trot") wbcL.wBase_ = 150; else if (gait != "stance") wbcL.wBase_ = 50;
  if (getenv("W_SW")) wbcL.wSwing_ = atof(getenv("W_SW"));
  if (getenv("W_BASE")) wbcL.wBase_ = atof(getenv("W_BASE"));
  if (getenv("W_F")) wbcL.wForce_ = atof(getenv("W_F"));
  if (getenv("KP_F")) wbcL.swingKp_ = atof(getenv("KP_F"));
  if (getenv("KD_F")) wbcL.swingKd_ = atof(getenv("KD_F"));
  if (getenv("REG")) wbcL.reg_ = atof(getenv("REG"));
  if (getenv("NWSR")) wbcL.nWsr_ = atoi(getenv("NWSR"));
  if (getenv("SWING_FF")) wbcL.swingFF_ = atoi(getenv("SWING_FF"));
  if (getenv("POST")) wbcL.wPosture_ = atof(getenv("POST"));      // ★널스페이스 posture 가중
  if (getenv("STANCE_KD")) wbcL.stanceKd_ = atof(getenv("STANCE_KD"));  // ★stance 접촉 속도댐핑(착지 슬립 저감)
  if (getenv("KP_POST")) wbcL.kpPost_ = atof(getenv("KP_POST"));
  if (getenv("KD_POST")) wbcL.kdPost_ = atof(getenv("KD_POST"));
  if (getenv("ANKLE_HARD")) wbcL.ankleHard_ = true;              // ★발목 hard-constraint(poor-man's HQP): weighted posture 대신 발목 strict eq
  if (getenv("KP_ANK")) wbcL.kpAnkle_ = atof(getenv("KP_ANK"));
  if (getenv("KD_ANK")) wbcL.kdAnkle_ = atof(getenv("KD_ANK"));
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
  vector_t qLo(nJ), qHi(nJ);   // ★조인트 각도한계(MJCF range) → WBC 제약
  for (int i = 0; i < nJ; ++i) {
    int j = mj_name2id(m, mjOBJ_JOINT, jNames[i].c_str());
    if (j < 0) { std::cerr << "관절 미발견: " << jNames[i] << "\n"; return 4; }
    qadr[i] = m->jnt_qposadr[j]; vadr[i] = m->jnt_dofadr[j];
    act[i] = mj_name2id(m, mjOBJ_ACTUATOR, actName(jNames[i]).c_str());
    if (act[i] < 0) { std::cerr << "액추에이터 미발견: " << actName(jNames[i]) << "\n"; return 4; }
    if (m->jnt_limited[j]) { qLo(i) = m->jnt_range[2 * j]; qHi(i) = m->jnt_range[2 * j + 1]; }
    else { qLo(i) = -1e9; qHi(i) = 1e9; }
  }
  // ★조인트 각도한계를 WBC에 전달(opt-in: JLIM=1). 컨트롤러가 관절을 range 밖으로 몰아 MuJoCo 클램프와 싸우던 붕괴 방지용.
  //   ⚠기본OFF: 이득이 D1 marginal 변동에 묻혀 미검증 + 슬로프 등반(한계근처 자세) 과잉제한 위험. 극단명령 시나리오 진단/보호용.
  if (wbcLegged && getenv("JLIM") && std::strcmp(getenv("JLIM"), "0") != 0) {
    if (getenv("KP_LIM")) wbcL.kpLim_ = atof(getenv("KP_LIM"));
    if (getenv("KD_LIM")) wbcL.kdLim_ = atof(getenv("KD_LIM"));
    if (getenv("MARG_LIM")) wbcL.margLim_ = atof(getenv("MARG_LIM"));
    wbcL.setJointLimits(qLo, qHi);
    std::cerr << "[JLIM] 조인트 각도한계 ON (kp=" << wbcL.kpLim_ << " kd=" << wbcL.kdLim_ << ")\n";
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
  // ★GUI: 초기 qpos 캡처(Reset/Ready 버튼서 복원) + 발배치 시각화 버퍼
  std::vector<double> qpos0(d->qpos, d->qpos + m->nq);
  double resetSeq = 0, homeSeq = 0; int lastResetSeq = 0, lastHomeSeq = 0;
  double vizSeed[4][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  double vizFoot[4][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};  // ★MPC 계획 발 착지점(마커, liftoff에 1회 계산·고정)
  double footSeed[4][2] = {{0,0},{0,0},{0,0},{0,0}}; bool footLocked[4] = {false,false,false,false};  // ★발판 liftoff 고정용

  // ★Phase 3b: nominal 발-base xy 오프셋(yaw프레임). 초기 nominal stance(yaw=0)서 발 sphere − base.
  //   매틱 발판영역 씨앗 = base_xy + vel·Δt + Rz(yaw)·offset (≈ getNominalFoothold FK, 평지 pitch≈0).
  double footOff[4][2];
  for (int i = 0; i < 4; ++i) {
    footOff[i][0] = d->geom_xpos[3 * footGeom[i] + 0] - d->qpos[0];
    footOff[i][1] = d->geom_xpos[3 * footGeom[i] + 1] - d->qpos[1];
    footSeed[i][0] = d->geom_xpos[3 * footGeom[i] + 0]; footSeed[i][1] = d->geom_xpos[3 * footGeom[i] + 1];  // ★발판 초기=현재 발 위치
  }
  const double comHeight = 0.50;                                  // reference.info comHeight(지형적응 base높이 기준)
  const double mpcHorizon = interface.mpcSettings().timeHorizon_; // 발판 stanceEnd 탐색·참조 재구성 창

  // 전진 목표 (VX env, 기본 0.3; stance 격리시 VX=0). ★GUI 배선: CMDFILE(teleop_gui JSON) 라이브 구동
  double vx = getenv("VX") ? std::atof(getenv("VX")) : 0.3;
  double vyCmd = 0.0, wCmd = 0.0;
  double vxTgt = vx, vyTgt = 0.0, wTgt = 0.0;   // ★GUI 원시 명령 타깃(슬루 전). 실제 vx/vyCmd/wCmd는 저역통과된 값
  std::string cmdGait, cmdMode;                          // ★GUI gait/mode 필드(라이브 게이트/모드 전환)
  std::string activeSeq = gait;                          // 현재 스케줄에 들어간 시퀀스(런치 게이트로 시작)
  const std::string gaitInfoPath = std::string(argv[3]).substr(0, refFile.find_last_of('/') + 1) + "gait.info";
  auto mapGait = [](const std::string& g) -> std::string { if (g == "walk") return "static_walk"; if (g == "run") return "trot"; return g; };  // GUI→D1 게이트명
  const char* cmdfile = getenv("CMDFILE");   // teleop_gui_17dof가 발행하는 /tmp/quad_cmd.json
  const double cmdTau = getenv("CMD_TAU") ? std::atof(getenv("CMD_TAU")) : 0.30;  // ★GUI 명령 슬루 시정수[s](0=즉시). 실기 가속제한 동형
  const double cmdJitter = getenv("CMD_JITTER") ? std::atof(getenv("CMD_JITTER")) : 0.0;  // ★검증용 결정론적 명령지터 주기[s](sim-time 구형파, 파일무관)
  const bool liveCmd = cmdfile || cmdJitter > 0.0;
  auto readCmd = [&]() {                      // JSON 최소파싱(v/vy/w). 없거나 실패=값 유지
    if (!cmdfile) return;
    std::ifstream cf(cmdfile); if (!cf.good()) return;
    std::string s((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
    auto num = [&](const char* k, double& o) {
      std::string key = std::string("\"") + k + "\"";
      auto p = s.find(key); if (p == std::string::npos) return;
      p = s.find(':', p + key.size()); if (p == std::string::npos) return;
      o = std::atof(s.c_str() + p + 1); };
    auto str = [&](const char* k, std::string& o) {          // ★문자열 값 파싱(gait/mode)
      std::string key = std::string("\"") + k + "\"";
      auto p = s.find(key); if (p == std::string::npos) return;
      p = s.find(':', p + key.size()); if (p == std::string::npos) return;
      p = s.find('"', p); if (p == std::string::npos) return;
      auto q = s.find('"', p + 1); if (q == std::string::npos) return;
      o = s.substr(p + 1, q - p - 1); };
    num("v", vxTgt); num("vy", vyTgt); num("w", wTgt);   // ★슬루 타깃에만 반영(급변 완화는 아래 저역통과)
    num("reset_seq", resetSeq); num("home_seq", homeSeq);
    str("gait", cmdGait); str("mode", cmdMode); };   // ★Reset/Ready 버튼 + GUI gait/mode(라이브 전환)
  readCmd();
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
    double Irot = getenv("ROTOR_I") ? std::atof(getenv("ROTOR_I")) : 7.327e-4;  // ★PACE 최종(2026-08-14, biped RESULTS.md=단일출처)
    const double dmpK[4] = {0.090, 0.0, 0.0, 0.110}, frcK[4] = {0.724, 0.604, 0.871, 0.639};  // 축별(hip,thigh,calf,foot=tendon값 foot-dof 근사)
    if (gbx) for (int k = 0; k < m->nu; ++k) { int jid = m->actuator_trnid[k * 2]; if (jid < 0) continue;
      const char* jn = mj_id2name(m, mjOBJ_JOINT, jid); if (!jn) continue;
      int gi = 0; for (int g = 0; g < 4; ++g) if (std::strstr(jn, GN[g])) gi = g;  // FB_waist→hip fallback(감속7:1, A와 동일)
      double N = gear[gi]; int dof = m->jnt_dofadr[jid];
      double jd = getenv("JDAMP") ? std::atof(getenv("JDAMP")) : dmpK[gi], jf = getenv("JFRIC") ? std::atof(getenv("JFRIC")) : frcK[gi];
      m->dof_armature[dof] = Irot * N * N; m->dof_damping[dof] = jd; m->dof_frictionloss[dof] = jf; }
    // ★★컨트롤러 모델 정합(2026-08-10): WBC pinocchio M에도 같은 반사관성(Irot·N²)을 넣어야 실제 관절관성 반영.
    //   안 하면 plant는 무거운데 컨트롤러는 가벼운 줄 알아 저-토크 명령→고속서 다리 지연→붕괴(사용자 "댐핑 부족" 지적의 근본).
    //   WBC 관절순=jointNames(FL,FR,HL,HR × hip,thigh,calf,foot) → type=j%perLeg. GEARBOX=0이면 plant도 armature 0이라 미적용.
    if (gbx && wbcLegged) { const int perLeg = nJ / 4; vector_t rotorArm(nJ);
      for (int j = 0; j < nJ; ++j) { double N = gear[j % perLeg]; rotorArm(j) = Irot * N * N; }
      wbcL.setRotorArmature(rotorArm); }
    fprintf(stderr, "[GBX] GEARBOX=%d ROTOR_I=%.2e(%s) JFRIC=%.3f JDAMP=%.3f | armature hip=%.4f calf=%.4f foot=%.4f\n",
            gbx, Irot, (Irot > 5e-4 ? "PACE실측" : "placeholder"), frcK[0], dmpK[0], Irot * 49, Irot * 10.5 * 10.5, Irot * 8.4 * 8.4); }
  const double dt = m->opt.timestep;
  // ★기본 100Hz(2026-08-10): 재계획률↑=base 회복 authority↑=고속 엔벨로프 확장(범용 레버). 저속/stance 무회귀.
  //   연산 2배지만 solve~7ms<10ms budget 여유(D1=연구·비실시간). MPC_HZ env로 override.
  //   ※엔벨로프 절대치는 반사관성 정합(GEARBOX plant + WBC rotorArm) 후 = 실측 PACE 물리 기준 trot/bound≤0.8·walk≤0.3.
  const double mpcHz = getenv("MPC_HZ") ? std::atof(getenv("MPC_HZ")) : 100.0;
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
  double swingErrPk = 0;   // ★스윙 추종오차 피크(SWING_DBG)
  double tdErrMax = 0, tdErrSum = 0; int tdErrN = 0;   // ★착지오차(TD_DBG): 실제 발 vs 커밋 발판
  double kmPk = 0, amPk = 0; long kmViol = 0, coupleN = 0;   // ★COUPLE_DBG: 실기 커플토크(무릎모터=τ_calf−τ_foot·발목모터=τ_foot) 피크/위반
  double frontJvPk = 0, frontTauPk = 0; vector_t tauPrev = vector_t::Zero(nJ);  // 앞다리 떨림 계측(관절속도·토크변화 피크)
  for (int step = 0; view ? !glfwWindowShouldClose(win) : (t < simTime); ++step, t += dt) {
    if (cmdfile && step % 20 == 0) {            // ★GUI: 50Hz로 명령 갱신(v/vy/w)
      readCmd();
      if ((int)resetSeq != lastResetSeq || (int)homeSeq != lastHomeSeq) {   // ★Reset/Ready 버튼=초기자세 복원
        lastResetSeq = (int)resetSeq; lastHomeSeq = (int)homeSeq;
        for (int q = 0; q < m->nq; ++q) d->qpos[q] = qpos0[q];
        for (int v2 = 0; v2 < m->nv; ++v2) d->qvel[v2] = 0.0;
        mj_forward(m, d);
        for (int i = 0; i < 4; ++i) {  // ★발판도 리셋: 리셋 pose의 실제 발 위치로·lock 해제(다음 liftoff에 새로 커밋)
          footSeed[i][0] = d->geom_xpos[3 * footGeom[i] + 0]; footSeed[i][1] = d->geom_xpos[3 * footGeom[i] + 1]; footLocked[i] = false;
        }
        std::cerr << "  [RESET] 로봇 초기자세 복원(t=" << t << ")\n";
      }
      // ★GUI 라이브 게이트/모드 전환(Walk/Ready/Ground + 게이트 버튼): OCS2 런타임 mode-seq 삽입.
      //   mode=move→보행게이트(gait필드) · stand_up/stand_down→제자리 stance · mode없고 gait만→보행게이트.
      {
        std::string want = activeSeq;
        if (cmdMode == "move" || cmdMode == "tamols") { if (!cmdGait.empty()) want = mapGait(cmdGait); }
        else if (!cmdMode.empty())                    want = "stance";
        else if (!cmdGait.empty())                    want = mapGait(cmdGait);
        if (want != activeSeq) {
          try {
            auto tmpl = loadModeSequenceTemplate(gaitInfoPath, want, false);
            refMgr->getGaitSchedule()->insertModeSequenceTemplate(tmpl, t + 0.1, t + 20.0);
            if (!getenv("W_BASE")) wbcL.wBase_ = (want == "trot") ? 150.0 : 50.0;
            activeSeq = want;
            std::cerr << "  [GAIT] \xe2\x86\x92 " << want << " (mode=" << cmdMode << ", t=" << t << ")\n";
          } catch (const std::exception& e) { std::cerr << "  [GAIT] '" << want << "' \xeb\xa1\x9c\xeb\x93\x9c \xec\x8b\xa4\xed\x8c\xa8, \xec\x9c\xa0\xec\xa7\x80\n"; }
        }
      }
    }
    // ★검증용 결정론적 명령 지터(CMD_JITTER=주기[s]): sim-time 구형파로 vxTgt/wTgt 급변 생성(파일 무관, SYNC 결정론 A/B용).
    if (cmdJitter > 0.0) { bool hi = std::fmod(t, 2 * cmdJitter) < cmdJitter;
      vxTgt = hi ? 0.7 : 0.0; vyTgt = hi ? 0.0 : 0.15; wTgt = hi ? 0.6 : -0.6; }
    // ★GUI 명령 슬루(1차 저역통과, τ=cmdTau): 조이스틱 급변/떨림을 부드럽게 → NMPC 속도참조 불연속 완화(보행붕괴 방지).
    //   명령 없으면(헤드리스) 미적용=env VX 그대로(기존 결과 불변). 상태클램프 아님=명령 필터라 물리적(실기 teleop 가속제한과 동형).
    if (liveCmd && cmdTau > 1e-6) {
      const double aCmd = dt / (cmdTau + dt);
      vx    += aCmd * (vxTgt - vx);
      vyCmd += aCmd * (vyTgt - vyCmd);
      wCmd  += aCmd * (wTgt  - wCmd);
    } else if (liveCmd) { vx = vxTgt; vyCmd = vyTgt; wCmd = wTgt; }

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
          // ★발판=계획된 착지점을 liftoff에 1회 커밋 후 다음 liftoff까지 고정(swing+stance 내내 불변).
          //   발이 그 발판으로 가서 머무름 — 발판을 발끝에 맞춰 재생성하지 않음(계획이 실제를 끌어야지 반대 아님).
          const bool planted = (stanceEnd_i > t);
          if (!planted && !footLocked[i]) {                          // liftoff: 착지목표 1회 계산·커밋
            // ★리드=실제 착지시각(스케줄의 다음 swing→stance 전환)까지. 0.5·H 고정은 과예측(마커 70mm 앞섬)→착지점 정합.
            double dtm = 0.5 * H;                                     // fallback(전환 못 찾으면)
            for (int ip = 1; ip < nP; ++ip)
              if (ip - 1 < (int)ev.size() && ev[ip - 1] > t && modeNumber2StanceLeg(seq[ip])[i] && !modeNumber2StanceLeg(seq[ip - 1])[i]) { dtm = ev[ip - 1] - t; break; }
            dtm = std::max(0.02, dtm);
            double rx = cy * footOff[i][0] - sy * footOff[i][1], ry = sy * footOff[i][0] + cy * footOff[i][1];
            footSeed[i][0] = d->qpos[0] + baseVx * dtm + rx; footSeed[i][1] = d->qpos[1] + baseVy * dtm + ry;
            footLocked[i] = true;
          } else if (planted) {                                   // 착지 → 다음 liftoff 재커밋 준비(footSeed는 유지)
            if (footLocked[i] && getenv("TD_DBG")) {                    // ★방금 착지: 실제 발 vs 커밋 발판 gap
              double dx = d->geom_xpos[3 * footGeom[i] + 0] - vizFoot[i][0], dy = d->geom_xpos[3 * footGeom[i] + 1] - vizFoot[i][1];
              double e = std::hypot(dx, dy); tdErrSum += e; ++tdErrN; if (e > tdErrMax) tdErrMax = e; }
            footLocked[i] = false;
          }
          const double seedX = footSeed[i][0], seedY = footSeed[i][1];  // 커밋된 발판(고정)
          region->updateFoot(i, seedX, seedY, stanceEnd_i);
          // ★마커=MPC 정책 발 위치: stance=현재시각·swing=다음 착지시각. 매 사이클 갱신→발과 정합(드리프트 추종).
          { double evalT = t;
            if (!planted) { for (int ip = 1; ip < nP; ++ip)
              if (ip - 1 < (int)ev.size() && ev[ip - 1] > t && modeNumber2StanceLeg(seq[ip])[i] && !modeNumber2StanceLeg(seq[ip - 1])[i]) { evalT = ev[ip - 1]; break; } }
            const auto& sol = mrt.getPolicy();
            if (!sol.timeTrajectory_.empty()) {
              scalar_t eT = std::max(sol.timeTrajectory_.front(), std::min((scalar_t)evalT, sol.timeTrajectory_.back()));
              vector_t xE = ocs2::LinearInterpolation::interpolate(eT, sol.timeTrajectory_, sol.stateTrajectory_);
              vector_t qPin = vizMap.getPinocchioJointPosition(xE);
              pinocchio::forwardKinematics(vizPin.getModel(), vizPin.getData(), qPin);
              pinocchio::updateFramePlacements(vizPin.getModel(), vizPin.getData());
              auto fp = vizPin.getData().oMf[footId[i]].translation();
              vizFoot[i][0] = fp(0); vizFoot[i][1] = fp(1); vizFoot[i][2] = fp(2);
            } }
          vizSeed[i][0] = vizFoot[i][0]; vizSeed[i][1] = vizFoot[i][1];   // ★마커=MPC 계획 발 착지점(Raibert seed 아님)
          vizSeed[i][2] = terrainSdf ? terrainSdf->height(vizFoot[i][0], vizFoot[i][1]) : vizFoot[i][2];
        }
      }
      // (B) 지형적응 base높이: [t,t+H] 11노드 참조(base z=h+comH/cos pitch·pitch=지형법선). modifyReferences 포팅.
      //   ★base x,y는 원본 절대 forward 램프(x0_x+vx·tn) 유지(modifyReferences가 desired x,y 보존하듯) — z/pitch만 덮어씀.
      if (getenv("TERRAIN_Z")) {
        // ★GUI: 참조 원점=현재 base pose(로봇-상대 속도명령)·cmdfile 없으면 x0(원래 절대 forward 램프=하위호환)
        const double oX = cmdfile ? d->qpos[0] : centroidal_model::getBasePose(x0, info)(0);
        const double oY = cmdfile ? d->qpos[1] : centroidal_model::getBasePose(x0, info)(1);
        const double oYaw = cmdfile ? z : 0.0;   // z=현재 yaw(quat2zyx)
        const double tRef = cmdfile ? t : 0.0;   // ★로봇-상대(cmdfile)=현재시점 오프셋 기준·절대(고정VX)=0. 참조 이중계산(폭주) 방지
        // ★smooth heightmap(box-avg ±SW): legged_perceptive "smooth_planar" 대응. 원 mj_ray 계단 날카로움 완화
        //   → base z가 계단 앞서 점진 상승(급점프 전복 방지). pitch도 step=0.3 넓은 차분(legged_perceptive 동일).
        const double SW = getenv("SMOOTH_W") ? std::atof(getenv("SMOOTH_W")) : 0.14;
        auto hS = [&](double x, double y) { double s = 0; int c = 0;
          for (double dx = -SW; dx <= SW + 1e-9; dx += 0.04) for (double dy = -SW; dy <= SW + 1e-9; dy += 0.04) { s += terrainSdf->height(x + dx, y + dy); ++c; }
          return s / c; };
        const double STEP = 0.3;
        // ★지형 속도정책(TERRAIN_CAP, 기본ON): 현재+전방 국소 경사·거칠기로 전진속도 자동캡 — 과속 낙상 방지.
        //   경사≥임계→VXCAP_SLOPE(0.2)·거칠기≥임계→VXCAP_ROUGH(0.15). 평지/완만=무캡. (검증: VX0.3 지형 낙상↔VX0.2 완주)
        double vxEff = vx;
        if (!(getenv("TERRAIN_CAP") && !std::strcmp(getenv("TERRAIN_CAP"), "0"))) {
          const double bxN = d->qpos[0], byN = d->qpos[1];   // ★현재 실제 base 위치(oX는 헤드리스서 nominal 0이라 못 씀)
          const double cy = std::cos(z), sy = std::sin(z), h0 = hS(bxN, byN);  // z=현재 yaw
          double slope = 0, rough = 0;
          for (double f = 0.0; f <= 1.0 + 1e-9; f += 0.25) {   // 현재+전방 1.0m까지(bump/램프 선행 감속=가속 전 캡)
            double px = bxN + f * cy, py = byN + f * sy;
            double gX = (hS(px - STEP, py) - hS(px + STEP, py)) / (2 * STEP);
            double gY = (hS(px, py - STEP) - hS(px, py + STEP)) / (2 * STEP);
            slope = std::max(slope, std::atan2(std::hypot(gX, gY), 1.0));
            rough = std::max(rough, std::abs(hS(px, py) - h0));
          }
          double cap = 1e9;
          if (slope > 0.06) cap = std::min(cap, getenv("VXCAP_SLOPE") ? std::atof(getenv("VXCAP_SLOPE")) : 0.2);  // >3.4°
          if (rough > 0.02) cap = std::min(cap, getenv("VXCAP_ROUGH") ? std::atof(getenv("VXCAP_ROUGH")) : 0.15); // >2cm
          vxEff = std::max(-cap, std::min(vx, cap));
          if (getenv("TC_DBG") && step % 500 == 0)
            fprintf(stderr, "[TCAP] x=%.2f slope=%.3f(%.1f°) rough=%.3f vx=%.2f→vxEff=%.2f\n", bxN, slope, slope * 57.3, rough, vx, vxEff);
        }
        const int N = 11; std::vector<scalar_t> tt(N); std::vector<vector_t> xs(N), us(N);
        for (int n = 0; n < N; ++n) {
          double tn = t + (double)n * H / (N - 1);
          const double tau = tn - tRef;   // ★참조 적분은 시작시점 기준 경과시간(절대 tn 아님)
          vector_t xn = x0;                                                     // nominal 자세·momentum, base만 지형적응
          double yawN = oYaw + wCmd * tau;                                       // ★GUI 선회(yaw-rate 적분)
          double bx = oX + vxEff * tau * std::cos(oYaw + 0.5 * wCmd * tau) - vyCmd * tau * std::sin(oYaw);
          double by = oY + vxEff * tau * std::sin(oYaw + 0.5 * wCmd * tau) + vyCmd * tau * std::cos(oYaw);
          double nX = (hS(bx - STEP, by) - hS(bx + STEP, by)) / (2 * STEP);      // n=[-∂h/∂x,-∂h/∂y,1]=법선(smooth·넓은차분)
          double nY = (hS(bx, by - STEP) - hS(bx, by + STEP)) / (2 * STEP);
          double vx_ = std::cos(yawN) * nX + std::sin(yawN) * nY;               // (Rz(yaw)ᵀ·n).x
          double pitch = std::atan2(vx_, 1.0);
          centroidal_model::getBasePose(xn, info)(0) = bx;
          centroidal_model::getBasePose(xn, info)(1) = by;
          centroidal_model::getBasePose(xn, info)(3) = yawN;                     // ★yaw 참조(GUI 선회)
          centroidal_model::getBasePose(xn, info)(4) = pitch;                    // pitch 먼저(z가 읽음)
          centroidal_model::getBasePose(xn, info)(2) = hS(bx, by) + comHeight / std::cos(pitch);
          tt[n] = tn; xs[n] = xn; us[n] = vector_t::Zero(info.inputDim);
        }
        interface.getReferenceManagerPtr()->setTargetTrajectories(TargetTrajectories(std::move(tt), std::move(xs), std::move(us)));
      }
    }
    else if (cmdfile && step % mpcDecim == 0) {   // ★GUI 평지(perceptive off): 라이브 속도 목표(로봇-상대·선회)
      const double Hh = mpcHorizon;
      vector_t xa = x0, xb = x0;
      centroidal_model::getBasePose(xa, info)(0) = d->qpos[0];
      centroidal_model::getBasePose(xa, info)(1) = d->qpos[1];
      centroidal_model::getBasePose(xa, info)(3) = z;
      centroidal_model::getBasePose(xb, info)(0) = d->qpos[0] + vx * Hh * std::cos(z + 0.5 * wCmd * Hh) - vyCmd * Hh * std::sin(z);
      centroidal_model::getBasePose(xb, info)(1) = d->qpos[1] + vx * Hh * std::sin(z + 0.5 * wCmd * Hh) + vyCmd * Hh * std::cos(z);
      centroidal_model::getBasePose(xb, info)(3) = z + wCmd * Hh;
      interface.getReferenceManagerPtr()->setTargetTrajectories(TargetTrajectories({t, t + Hh}, {xa, xb},
          {vector_t::Zero(info.inputDim), vector_t::Zero(info.inputDim)}));
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
      static double wbcTsum = 0; static long wbcN = 0;
      auto tw0 = std::chrono::steady_clock::now();
      vector_t tauJ = wbcL.update(xDes, uDes, rbd_s, md, dt);
      if (getenv("SWING_DBG") && wbcL.swingErrCur_ > swingErrPk) swingErrPk = wbcL.swingErrCur_;
      // ★COUPLE_DBG: calf-foot 기구커플(실기 q_raw=q_foot+q_calf, PACE c=1) → 무릎모터 필요토크=τ_calf−τ_foot(한계126)·발목모터=τ_foot(100.8)
      if (getenv("COUPLE_DBG")) { const int pl = nJ / 4;
        for (int lg = 0; lg < 4 && pl == 4; ++lg) {
          double tc = tauJ(lg * pl + 2), tf = tauJ(lg * pl + 3), km = std::fabs(tc - tf), am = std::fabs(tf);
          if (km > kmPk) kmPk = km; if (am > amPk) amPk = am; if (km > 126.0) ++kmViol; ++coupleN; } }
      wbcTsum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw0).count(); wbcN++;
      if (getenv("WBC_TIME") && wbcN % 3000 == 0) fprintf(stderr, "[WBCT] WBC avg=%.3f ms (n=%ld, 1kHz budget=1ms)\n", wbcTsum / wbcN, wbcN);
      // ★표준 저수준(Bellicoso2016·legged_control): τ = τ_ff + τ_pd
      //   τ_ff = τ_wbc(WBC 전신 역동역학 피드포워드), τ_pd = kp(q_des−q)+kd(q̇_des−q̇) 관절추종.
      //   legged: LeggedController.cpp:135(kp=0,kd=3) → LeggedHWSim.cpp:163. velDes=MPC 관절속도(uDes 후반 nJ).
      double jkd = getenv("JKD") ? atof(getenv("JKD")) : 0.0;   // 이상토크 MuJoCo=0(kd=3는 WBC 정확토크와 충돌)
      double jkp = getenv("JKP") ? atof(getenv("JKP")) : 0.0;   // kp(옵션, legged 기본 0)
      // ★위상 스케줄 블렌드(2026-09-01, leg-heavy 설계 반영): swing=PD 지배(모델오차 큰 곳=고게인 피드백)·
      //   stance=tau_ff 지배(GRF 최적화 신뢰=저게인). JKP_SW/JKD_SW 미설정=균일 JKP/JKD(하위호환).
      //   ⚠순수 PD 아님 — tau_ff는 유지(무거운 다리의 중력·관성을 FF가 지고 PD는 잔차만).
      double jkpSw = getenv("JKP_SW") ? atof(getenv("JKP_SW")) : jkp;
      double jkdSw = getenv("JKD_SW") ? atof(getenv("JKD_SW")) : jkd;
      const auto cfBlend = modeNumber2StanceLeg(md);   // 관절순 [FL,FR,HL,HR]=contact flag 순
      for (int i = 0; i < nJ; ++i) {
        double qMeas = rbd_s(6 + i), qdMeas = rbd_s(12 + nJ + i);  // rbd: jointPos@6.., jointVel@(12+nJ)..
        double qDes = xDes(12 + i);                               // centroidal state: momentum(6)+basePose(6)+jointPos(nJ)
        double qdDes = uDes(3 * 4 + i);                           // centroidal input: contactForce(3*4)+jointVel(nJ)
        const bool stanceLeg = cfBlend[i / (nJ / 4)];
        double kpU = stanceLeg ? jkp : jkpSw, kdU = stanceLeg ? jkd : jkdSw;
        d->ctrl[act[i]] = tauJ(i) + kpU * (qDes - qMeas) + kdU * (qdDes - qdMeas);  // τ_ff + τ_pd(위상 스케줄)
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
    // ★외란 push: PUSH(N) 크기·PUSH_T(s) 첫시각·PUSH_DUR(s) 지속·PUSH_AX(0=x,1=y,2=z)·PUSH_REP(반복수)·PUSH_EVERY(간격s, 교대방향)
    if (getenv("PUSH") && baseBody >= 0) {
      double pf = std::atof(getenv("PUSH")), pt = getenv("PUSH_T") ? std::atof(getenv("PUSH_T")) : 3.0;
      double pdur = getenv("PUSH_DUR") ? std::atof(getenv("PUSH_DUR")) : 0.1;
      int pax = getenv("PUSH_AX") ? std::atoi(getenv("PUSH_AX")) : 1;
      int prep = getenv("PUSH_REP") ? std::atoi(getenv("PUSH_REP")) : 1;
      double pevery = getenv("PUSH_EVERY") ? std::atof(getenv("PUSH_EVERY")) : 2.0;
      for (int k = 0; k < 3; ++k) d->xfrc_applied[6 * baseBody + k] = 0.0;
      for (int r = 0; r < prep; ++r) {
        double trp = pt + r * pevery;
        if (t >= trp && t < trp + pdur) {  // 반복마다 교대 방향(±)으로 밀어 반복 외란 강건성 시연
          d->xfrc_applied[6 * baseBody + pax] = (r % 2 == 0 ? pf : -pf);
          if (getenv("TROT_DBG") && step % int(0.1 / dt) == 0) std::cerr << "  [PUSH] t=" << t << " F=" << (r % 2 == 0 ? pf : -pf) << "N ax=" << pax << "\n";
        }
      }
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
      // ★발배치 시각화(기본 on, VIZ_OFF로 끔): walkable 영역 박스(녹색 반투명) + 발판 seed(노랑 구)
      if (!getenv("VIZ_OFF") && region) {
        for (int i = 0; i < 4 && scn.ngeom + 2 <= scn.maxgeom; ++i) {
          if (region->valid(i)) {
            const auto& ab = region->params(i);
            double hx = (ab.b(0) + ab.b(1)) / 2, cx = (ab.b(1) - ab.b(0)) / 2;
            double hy = (ab.b(2) + ab.b(3)) / 2, cy = (ab.b(3) - ab.b(2)) / 2;
            double hz = terrainSdf ? terrainSdf->height(cx, cy) : 0.0;
            mjtNum szb[3] = {hx > 0.02 ? hx : 0.02, hy > 0.02 ? hy : 0.02, 0.004}, posb[3] = {cx, cy, hz + 0.008};
            float grn[4] = {0.15f, 0.85f, 0.25f, 0.35f};
            mjv_initGeom(&scn.geoms[scn.ngeom++], mjGEOM_BOX, szb, posb, nullptr, grn);
          }
          mjtNum szs[3] = {0.028, 0.028, 0.028}, poss[3] = {vizSeed[i][0], vizSeed[i][1], vizSeed[i][2] + 0.02};
          float yel[4] = {1.0f, 0.85f, 0.1f, 0.95f};
          mjv_initGeom(&scn.geoms[scn.ngeom++], mjGEOM_SPHERE, szs, poss, nullptr, yel);
        }
      }
      // ★heightmap 시각화(HMAP=1): 컨트롤러가 mj_ray로 인지하는 지형격자. 높이→색(파랑낮음·초록중간·빨강높음).
      if (getenv("HMAP") && terrainSdf) {
        const int hnx = terrainSdf->nx(), hny = terrainSdf->ny(); const double hx0 = terrainSdf->x0(), hy0 = terrainSdf->y0(), hcl = terrainSdf->cell();
        const int hstep = 3;
        for (int iy = 0; iy < hny && scn.ngeom < scn.maxgeom; iy += hstep)
          for (int ix = 0; ix < hnx && scn.ngeom < scn.maxgeom; ix += hstep) {
            double hx = hx0 + ix * hcl, hy = hy0 + iy * hcl, hh = terrainSdf->height(hx, hy);
            double tt = std::max(0.0, std::min(1.0, hh / 0.3));
            float hcol[4] = {(float)tt, (float)(1.0 - std::fabs(tt - 0.5) * 2.0), (float)(1.0 - tt), 0.5f};
            mjtNum szc[3] = {hcl * hstep * 0.45, hcl * hstep * 0.45, 0.003}, posc[3] = {hx, hy, hh + 0.003};
            mjv_initGeom(&scn.geoms[scn.ngeom++], mjGEOM_BOX, szc, posc, nullptr, hcol);
          }
      }
      // ★push 외란 시각화: 활성 push(xfrc≠0) 시 base에 힘 방향 빨간 화살표(민다는 걸 명확히)
      double fx = baseBody >= 0 ? d->xfrc_applied[6 * baseBody + 0] : 0, fy = baseBody >= 0 ? d->xfrc_applied[6 * baseBody + 1] : 0,
             fz = baseBody >= 0 ? d->xfrc_applied[6 * baseBody + 2] : 0;
      double fmag = std::sqrt(fx * fx + fy * fy + fz * fz);
      if (fmag > 1.0 && scn.ngeom < scn.maxgeom) {
        double L = 0.4 + fmag / 1000.0 * 0.5;
        mjtNum from[3] = {d->qpos[0] - fx / fmag * L, d->qpos[1] - fy / fmag * L, d->qpos[2] + 0.15};
        mjtNum to[3] = {d->qpos[0] - fx / fmag * 0.12, d->qpos[1] - fy / fmag * 0.12, d->qpos[2] + 0.15};
        float red[4] = {1.0f, 0.15f, 0.1f, 1.0f};
        mjv_initGeom(&scn.geoms[scn.ngeom], mjGEOM_ARROW, nullptr, nullptr, nullptr, red);
        mjv_connector(&scn.geoms[scn.ngeom], mjGEOM_ARROW, 0.025, from, to);
        scn.ngeom++;
      }
      mjr_render(vp, &scn, &con);
      char pmsg[48] = "";
      if (fmag > 1.0) snprintf(pmsg, sizeof(pmsg), "   <<<< PUSH %.0fN", fmag);
      char hud[160]; snprintf(hud, sizeof(hud), "t=%.1fs  base_z=%.3f  gait=%s  %s%s", t, d->qpos[2], gait.c_str(),
                              useWbc ? "WBC" : "ff+PD", pmsg);
      mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, vp, "02_Leg OCS2 NMPC+WBC", hud, &con);
      // ★VIDEO 캡처: 렌더 프레임을 GL framebuffer서 직접 readPixels → raw RGB 파일(가림 무관). ffmpeg로 인코딩.
      if (getenv("VIDEO") && step % 32 == 0) {  // 32ms sim = ~31fps
        static FILE* vf = fopen(getenv("VIDEO"), "wb");
        static std::vector<unsigned char> vbuf(vp.width * vp.height * 3);
        static int vframes = 0; const int VMAX = getenv("VIDEO_N") ? atoi(getenv("VIDEO_N")) : 300;
        if (vf && vframes < VMAX) {
          mjr_readPixels(vbuf.data(), nullptr, vp, &con);
          fwrite(vbuf.data(), 1, vbuf.size(), vf); vframes++;
          if (vframes == 1) fprintf(stderr, "[VIDEO] %dx%d capture start (VMAX=%d)\n", vp.width, vp.height, VMAX);
          if (vframes >= VMAX) { fclose(vf); vf = nullptr; fprintf(stderr, "[VIDEO] done %d frames\n", vframes); }
        }
      }
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
    if (getenv("JL_DBG") && step % int(0.25 / dt) == 0) {  // ★조인트 한계 근접도(정상 운용서 관절이 한계에 얼마나 붙나 = JLIM 마진 튜닝용)
      double minMarg = 1e9, minLeg = 1e9; int mjid = -1, mleg = -1; const int pl = nJ / 4;
      for (int i = 0; i < nJ; ++i) { double q = d->qpos[qadr[i]], mm = std::min(q - qLo(i), qHi(i) - q);
        if (mm < minMarg) { minMarg = mm; mjid = i; }
        if (i % pl != pl - 1 && mm < minLeg) { minLeg = mm; mleg = i; } }  // 발목(j%pl==pl-1) 제외 = 다리관절만
      fprintf(stderr, "  [JLM] t=%.2f allMin=%.3f(%s) legMin=%.3f(%s)\n", t, minMarg, mjid >= 0 ? jNames[mjid].c_str() : "?",
              minLeg, mleg >= 0 ? jNames[mleg].c_str() : "?");
    }
  }

  mpcAlive = false;
  if (mpcWorker.joinable()) mpcWorker.join();
  std::cerr << "\n===== 결과 =====\n";
  std::cerr << "  최종 base_x : " << d->qpos[0] << " m  (목표 " << 0.3 * simTime << ")\n";
  std::cerr << "  최종 base_z : " << d->qpos[2] << " m\n";
  if (getenv("SWING_DBG")) std::cerr << "  스윙 추종오차 피크 : " << swingErrPk * 1000 << " mm\n";
  if (getenv("SNAP_DBG") && region) std::cerr << "  SDF 발판스냅 : " << region->snapCount_ << "회, 최대이동 " << region->snapDistMax_ * 1000 << " mm\n";
  if (getenv("COUPLE_DBG") && coupleN) std::cerr << "  커플토크(실기환산) : 무릎모터 피크 " << kmPk << " Nm(한계126, 위반 " << kmViol << "/" << coupleN << ") · 발목모터 피크 " << amPk << " Nm(한계100.8)\n";
  if (getenv("TD_DBG")) std::cerr << "  착지오차(실제발-발판) : 평균 " << (tdErrN ? tdErrSum / tdErrN * 1000 : 0) << " mm · 최대 " << tdErrMax * 1000 << " mm (" << tdErrN << "착지)\n";
  std::cerr << "  낙상 스텝수 : " << falls << (falls == 0 ? "  ✅ falls=0" : "  ✗") << "\n";
  if (useWbc) std::cerr << "  WBC QP 실패수 : " << wbc.qpFail_ << "\n";
  if (view) { mjv_freeScene(&scn); mjr_freeContext(&con); glfwTerminate(); }
  mj_deleteData(d); mj_deleteModel(m);
  return falls == 0 ? 0 : 3;
}
