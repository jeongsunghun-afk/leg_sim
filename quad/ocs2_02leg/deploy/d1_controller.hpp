#pragma once
// ================================================================================================
// D1 제어 코어 (OCS2 통합 perceptive NMPC + WeightedWBC) — test02legMujoco.cpp에서 추출.
//   입력: qc::State(base pose/twist·q/dq·contact, 컨트롤러 유일입력) + TerrainProvider(지형) + Command(cmd_vel).
//   출력: 관절토크 τ(nJ, OCS2 관절순 [FL,FR,HL,HR]×[hip,thigh,calf]).
//   ★MuJoCo 완전 비의존 — 상태는 State, 지형은 TerrainProvider.fill()로만 받는다.
//   env 파라미터(PERCEPTIVE/PLACEMENT/TERRAIN_Z/W_*/KP_F/... /MPC_HZ/CMD_TAU)는 test02legMujoco와 동일.
// ================================================================================================
#include <array>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>
#include <cstdlib>
#include <Eigen/Dense>

#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_legged_robot/gait/MotionPhaseDefinition.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_legged_robot/reference_manager/SwitchedModelReferenceManager.h>
#include <ocs2_centroidal_model/CentroidalModelRbdConversions.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_robotic_tools/common/RotationDerivativesTransforms.h>
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include <ocs2_sqp/SqpMpc.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>

#include "wbc_legged.hpp"
#include "mj_terrain_sdf.hpp"
#include "foot_terrain_clearance.hpp"
#include "foot_terrain_placement.hpp"
#include "local_convex_region.hpp"
#include "terrain_provider.hpp"
#include "estimator/state.hpp"   // quad_ctrl: qc::State

namespace d1 {

using namespace ocs2;
using namespace ocs2::legged_robot;

// GUI/네비 명령(cmd_vel + gait/mode). mode: "move"=보행, "stand_up"/"stand_down"=제자리 stance.
struct Command {
  double vx = 0, vy = 0, w = 0;
  std::string gait = "", mode = "move";
};

class D1Controller {
 public:
  // 셋업: OCS2 문제 구성 + perceptive(env) + MPC/MRT + WBC(env 게인) + 게이트 스케줄 + 초기 참조/워밍업.
  //   gaitInfoDir = reference.info가 있는 config 디렉토리(끝에 '/'). initGait = 시작 게이트.
  void setup(const std::string& taskFile, const std::string& urdfFile, const std::string& refFile,
             const std::string& gaitInfoDir, const std::string& initGait, double dt) {
    dt_ = dt;
    gait_ = initGait;
    gaitInfoPath_ = gaitInfoDir + "gait.info";
    interface_ = std::make_unique<LeggedRobotInterface>(taskFile, urdfFile, refFile, false);
    const auto& info = interface_->getCentroidalModelInfo();
    nJ_ = info.actuatedDofNum;
    rbd_ = std::make_unique<ocs2::CentroidalModelRbdConversions>(interface_->getPinocchioInterface(), info);

    // ── perceptive: 지형 SDF + 발-지형 클리어런스(제약이 shared_ptr로 SDF 소유 → 매틱 fill이 즉시 반영) ──
    if (std::getenv("PERCEPTIVE")) {
      terrainSdf_ = std::make_shared<MjTerrainSdf>();
      auto& prob = interface_->getMutableOptimalControlProblem();
      auto pRefMgr = std::dynamic_pointer_cast<SwitchedModelReferenceManager>(interface_->getReferenceManagerPtr());
      const auto& ms = interface_->modelSettings();
      double clr = std::getenv("CLEARANCE") ? atof(std::getenv("CLEARANCE")) : 0.04;
      for (size_t i = 0; i < info.numThreeDofContacts; ++i) {
        const std::string& footName = ms.contactNames3DoF[i];
        const auto infoCppAd = info.toCppAd();
        const CentroidalModelPinocchioMappingCppAd mapCppAd(infoCppAd);
        auto velCb = [infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pAd) {
          const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
          updateCentroidalDynamics(pAd, infoCppAd, q);
        };
        std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKin(new PinocchioEndEffectorKinematicsCppAd(
            interface_->getPinocchioInterface(), mapCppAd, {footName}, info.stateDim, info.inputDim,
            velCb, footName + "_perc", ms.modelFolderCppAd, ms.recompileLibrariesCppAd, ms.verboseCppAd));
        std::unique_ptr<PenaltyBase> pen(new RelaxedBarrierPenalty(RelaxedBarrierPenalty::Config(1e-2, 1e-3)));
        std::unique_ptr<StateConstraint> con(new FootTerrainClearanceConstraint(*pRefMgr, *eeKin, terrainSdf_, i, clr));
        prob.stateSoftConstraintPtr->add(footName + "_terrainClearance",
            std::unique_ptr<StateCost>(new StateSoftConstraint(std::move(con), std::move(pen))));
      }
      // ── 발판배치(convex 박스영역) 제약 ──
      if (std::getenv("PLACEMENT")) {
        region_ = std::make_shared<LocalConvexRegion>(terrainSdf_);
        for (size_t i = 0; i < info.numThreeDofContacts; ++i) {
          const std::string& footName = ms.contactNames3DoF[i];
          const auto infoCppAd = info.toCppAd();
          const CentroidalModelPinocchioMappingCppAd mapCppAd(infoCppAd);
          auto velCb = [infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pAd) {
            const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
            updateCentroidalDynamics(pAd, infoCppAd, q);
          };
          std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKin(new PinocchioEndEffectorKinematicsCppAd(
              interface_->getPinocchioInterface(), mapCppAd, {footName}, info.stateDim, info.inputDim,
              velCb, footName + "_place", ms.modelFolderCppAd, ms.recompileLibrariesCppAd, ms.verboseCppAd));
          std::unique_ptr<PenaltyBase> pen(new RelaxedBarrierPenalty(RelaxedBarrierPenalty::Config(1e-2, 1e-4)));
          std::unique_ptr<StateConstraint> con(new FootTerrainPlacementConstraint(*pRefMgr, *eeKin, region_, i));
          prob.stateSoftConstraintPtr->add(footName + "_footPlacement",
              std::unique_ptr<StateCost>(new StateSoftConstraint(std::move(con), std::move(pen))));
        }
      }
    }

    // ── MPC (기본 SQP) ──
    if (std::getenv("DDP")) {
      mpc_ = std::make_unique<ocs2::GaussNewtonDDP_MPC>(interface_->mpcSettings(), interface_->ddpSettings(),
          interface_->getRollout(), interface_->getOptimalControlProblem(), interface_->getInitializer());
    } else {
      mpc_ = std::make_unique<ocs2::SqpMpc>(interface_->mpcSettings(), interface_->sqpSettings(),
          interface_->getOptimalControlProblem(), interface_->getInitializer());
    }
    mpc_->getSolverPtr()->setReferenceManager(interface_->getReferenceManagerPtr());
    refMgr_ = std::dynamic_pointer_cast<SwitchedModelReferenceManager>(interface_->getReferenceManagerPtr());

    // ── 게이트 스케줄 ──
    activeSeq_ = gait_;
    if (gait_ != "stance") {
      auto tmpl = loadModeSequenceTemplate(gaitInfoPath_, gait_, false);
      const double settle = std::getenv("SETTLE") ? atof(std::getenv("SETTLE")) : 0.0;
      refMgr_->getGaitSchedule()->insertModeSequenceTemplate(tmpl, settle, 1e4);
    }

    // ── MRT + WBC ──
    mrt_ = std::make_unique<ocs2::MPC_MRT_Interface>(*mpc_);
    mrt_->initRollout(&interface_->getRollout());
    const char* footFrame[4] = {"FL_foot_contact_link", "FR_foot_contact_link", "HL_foot_contact_link", "HR_foot_contact_link"};
    for (int i = 0; i < 4; ++i) footId_[i] = interface_->getPinocchioInterface().getModel().getFrameId(footFrame[i]);
    wbc_ = std::make_unique<WbcLegged>(interface_->getPinocchioInterface(), info, footId_);
    applyWbcParams();

    // ── GEARBOX(반사관성): WBC M을 플랜트 armature와 정합. 기본 ON(GEARBOX=0으로만 끔). ──
    //   rotorArm(j)=Irot·N². perLeg=nJ/4(12-DOF point-foot=3: hip/thigh/calf). 백엔드가 같은 값으로 플랜트 dof_armature 설정.
    rotorArm_ = vector_t::Zero(nJ_);
    if (!(std::getenv("GEARBOX") && !std::strcmp(std::getenv("GEARBOX"), "0"))) {
      const double Irot = std::getenv("ROTOR_I") ? atof(std::getenv("ROTOR_I")) : 7.4e-4;
      const double gear[4] = {7.0, 7.0, 10.5, 8.4};
      const int perLeg = nJ_ / 4;
      for (int j = 0; j < nJ_; ++j) { double N = gear[j % perLeg]; rotorArm_(j) = Irot * N * N; }
      wbc_->setRotorArmature(rotorArm_);
    }

    // ── 초기 상태/참조 + MRT 워밍업 ──
    x0_ = interface_->getInitialState();
    comHeight_ = 0.50;
    mpcHorizon_ = interface_->mpcSettings().timeHorizon_;
    const double mpcHz = std::getenv("MPC_HZ") ? atof(std::getenv("MPC_HZ")) : 100.0;
    mpcDecim_ = std::max(1, int((1.0 / mpcHz) / dt_));
    cmdTau_ = std::getenv("CMD_TAU") ? atof(std::getenv("CMD_TAU")) : 0.30;
    vx_ = vy_ = w_ = 0;

    vector_t xGoal = x0_;
    ocs2::centroidal_model::getBasePose(xGoal, info)(0) += 0.0;
    interface_->getReferenceManagerPtr()->setTargetTrajectories(ocs2::TargetTrajectories(
        {0.0, 1e4}, {x0_, xGoal}, {vector_t::Zero(info.inputDim), vector_t::Zero(info.inputDim)}));

    ocs2::SystemObservation obs;
    obs.time = 0.0; obs.state = x0_; obs.input = vector_t::Zero(info.inputDim); obs.mode = ModeNumber::STANCE;
    mrt_->setCurrentObservation(obs);
    while (!mrt_->initialPolicyReceived()) mrt_->advanceMpc();
    mrt_->updatePolicy();
  }

  int nJ() const { return nJ_; }
  // 백엔드가 관절매핑/초기포즈에 쓰는 값들(OCS2 관절순 [FL,FR,HL,HR]×[hip,thigh,calf]).
  std::vector<std::string> jointNames() const { return interface_->modelSettings().jointNames; }
  vector_t initialJointConfig() const { return x0_.tail(nJ_); }
  double initialBaseZ() const {
    vector_t x = x0_;
    return ocs2::centroidal_model::getBasePose(x, interface_->getCentroidalModelInfo())(2);
  }
  const vector_t& rotorArmature() const { return rotorArm_; }

  // 발-베이스 nominal xy 오프셋(yaw프레임). 백엔드(FK 아는 쪽)가 셋업 시 1회 주입. 순서=OCS2 [FL,FR,HL,HR].
  void setFootOffsets(const double off[4][2]) { for (int i = 0; i < 4; ++i) { footOff_[i][0] = off[i][0]; footOff_[i][1] = off[i][1]; } }

  // 리셋(GUI Reset/Ready): 발판을 현재 발 위치로·lock 해제. footPos = 현재 발 xy(4×2, OCS2순).
  void resetFootholds(const double footPos[4][2]) {
    for (int i = 0; i < 4; ++i) { footSeed_[i][0] = footPos[i][0]; footSeed_[i][1] = footPos[i][1]; footLocked_[i] = false; }
  }

  // ── 매 제어틱 ── State + 지형 + 명령 → 관절토크 τ(nJ)
  vector_t update(const qc::State& st, TerrainProvider& terrain, const Command& cmd) {
    const auto& info = interface_->getCentroidalModelInfo();
    const double t = st.time;

    // (1) 명령 슬루(1차 저역통과)
    if (cmdTau_ > 1e-6) {
      const double a = dt_ / (cmdTau_ + dt_);
      vx_ += a * (cmd.vx - vx_); vy_ += a * (cmd.vy - vy_); w_ += a * (cmd.w - w_);
    } else { vx_ = cmd.vx; vy_ = cmd.vy; w_ = cmd.w; }

    // (2) State → rbd_s(36) [eulZYX(3),pos(3),jPos(nJ),angVel_world(3),linVel_world(3),jVel(nJ)]
    double z, py, rx; quatZYX(st.base_quat, z, py, rx);
    vector_t rbd_s(6 + nJ_ + 6 + nJ_);
    rbd_s.segment<3>(0) << z, py, rx;
    rbd_s.segment<3>(3) = st.base_pos;
    for (int i = 0; i < nJ_; ++i) rbd_s(6 + i) = st.q(i);
    Eigen::Quaterniond q(st.base_quat(0), st.base_quat(1), st.base_quat(2), st.base_quat(3));  // wxyz
    Eigen::Vector3d angW = q.toRotationMatrix() * st.base_ang_vel;   // body → world
    rbd_s.segment<3>(6 + nJ_) = angW;
    rbd_s.segment<3>(6 + nJ_ + 3) = st.base_lin_vel;                 // world
    for (int i = 0; i < nJ_; ++i) rbd_s(6 + nJ_ + 6 + i) = st.dq(i);
    vector_t xMeas = rbd_->computeCentroidalStateFromRbdModel(rbd_s);

    const bool replanTick = (stepCount_ % mpcDecim_ == 0);

    // (3) 라이브 게이트/모드 전환
    if (replanTick) switchGait(cmd, t);

    // (4) 지형 SDF 갱신
    if (terrainSdf_) terrain.fill(*terrainSdf_, st.base_pos.x(), st.base_pos.y());

    // (5) 발판영역 + 지형적응 base높이 참조
    if (terrainSdf_ && replanTick && (region_ || std::getenv("TERRAIN_Z"))) {
      updateFootRegionAndRef(st, t);
    } else if (replanTick) {
      // perceptive off: 평지 로봇-상대 속도목표(2노드)
      const double Hh = mpcHorizon_;
      vector_t xa = x0_, xb = x0_;
      ocs2::centroidal_model::getBasePose(xa, info)(0) = st.base_pos.x();
      ocs2::centroidal_model::getBasePose(xa, info)(1) = st.base_pos.y();
      ocs2::centroidal_model::getBasePose(xa, info)(3) = z;
      ocs2::centroidal_model::getBasePose(xb, info)(0) = st.base_pos.x() + vx_ * Hh * std::cos(z + 0.5 * w_ * Hh) - vy_ * Hh * std::sin(z);
      ocs2::centroidal_model::getBasePose(xb, info)(1) = st.base_pos.y() + vx_ * Hh * std::sin(z + 0.5 * w_ * Hh) + vy_ * Hh * std::cos(z);
      ocs2::centroidal_model::getBasePose(xb, info)(3) = z + w_ * Hh;
      interface_->getReferenceManagerPtr()->setTargetTrajectories(ocs2::TargetTrajectories(
          {t, t + Hh}, {xa, xb}, {vector_t::Zero(info.inputDim), vector_t::Zero(info.inputDim)}));
    }

    // (6) MPC 관측 + 정책(동기: replanTick마다 advance)
    ocs2::SystemObservation obs;
    obs.time = t; obs.state = xMeas; obs.input = vector_t::Zero(info.inputDim);
    mrt_->setCurrentObservation(obs);
    if (replanTick) { try { mrt_->advanceMpc(); mrt_->updatePolicy(); } catch (const std::exception&) {} }
    vector_t xDes, uDes; size_t md;
    mrt_->evaluatePolicy(t, xMeas, xDes, uDes, md);

    // (7) 실접촉 오버라이드(옵션) — st.contact 는 OCS2순 [FL,FR,HL,HR]
    if (std::getenv("CONTACT_ACTUAL")) {
      bool actC[4]; for (int i = 0; i < 4; ++i) actC[i] = st.contact(i) > 0.5;
      wbc_->setActualContact(actC);
    }

    // (8) WBC → 관절토크
    ++stepCount_;
    return wbc_->update(xDes, uDes, rbd_s, md, dt_);
  }

 private:
  void applyWbcParams() {
    if (gait_ == "trot") wbc_->wBase_ = 150; else if (gait_ != "stance") wbc_->wBase_ = 50;
    auto ef = [](const char* k, double& v) { if (std::getenv(k)) v = atof(std::getenv(k)); };
    ef("W_SW", wbc_->wSwing_); ef("W_BASE", wbc_->wBase_); ef("W_F", wbc_->wForce_);
    ef("KP_F", wbc_->swingKp_); ef("KD_F", wbc_->swingKd_); ef("REG", wbc_->reg_);
    if (std::getenv("NWSR")) wbc_->nWsr_ = atoi(std::getenv("NWSR"));
    if (std::getenv("SWING_FF")) wbc_->swingFF_ = atoi(std::getenv("SWING_FF"));
    ef("POST", wbc_->wPosture_); ef("STANCE_KD", wbc_->stanceKd_);
    ef("KP_POST", wbc_->kpPost_); ef("KD_POST", wbc_->kdPost_);
    if (std::getenv("ANKLE_HARD")) wbc_->ankleHard_ = true;
    ef("KP_ANK", wbc_->kpAnkle_); ef("KD_ANK", wbc_->kdAnkle_);
    wbc_->basePd_ = !std::getenv("NO_BASE_PD");
    if (std::getenv("BASE_NOFF")) wbc_->baseNoFF_ = true;
    ef("KP_B", wbc_->kpBase_); ef("KD_B", wbc_->kdBase_);
  }

  static std::string mapGait(const std::string& g) {
    if (g == "walk") return "static_walk"; if (g == "run") return "trot"; return g;
  }

  void switchGait(const Command& cmd, double t) {
    std::string want = activeSeq_;
    if (cmd.mode == "move" || cmd.mode == "tamols") { if (!cmd.gait.empty()) want = mapGait(cmd.gait); }
    else if (!cmd.mode.empty()) want = "stance";
    else if (!cmd.gait.empty()) want = mapGait(cmd.gait);
    if (want != activeSeq_) {
      try {
        auto tmpl = loadModeSequenceTemplate(gaitInfoPath_, want, false);
        refMgr_->getGaitSchedule()->insertModeSequenceTemplate(tmpl, t + 0.1, t + 1e4);
        if (!std::getenv("W_BASE")) wbc_->wBase_ = (want == "trot") ? 150.0 : 50.0;
        activeSeq_ = want;
      } catch (const std::exception&) {}
    }
  }

  // TERRAIN_Z: 발판 커밋-고정 + 지형적응 base높이 참조(N=11노드). test02legMujoco B.1/B.3 이식.
  void updateFootRegionAndRef(const qc::State& st, double t) {
    const auto& info = interface_->getCentroidalModelInfo();
    const double H = mpcHorizon_, baseVx = st.base_lin_vel.x(), baseVy = st.base_lin_vel.y();
    double z, py, rx; quatZYX(st.base_quat, z, py, rx);
    const double yaw = z, cy = std::cos(yaw), sy = std::sin(yaw);

    // (A) 발판영역: 발별 stanceEnd + 커밋-고정 seed → updateFoot
    if (region_) {
      auto msched = refMgr_->getGaitSchedule()->getModeSchedule(t - H, t + H);
      const auto& ev = msched.eventTimes; const auto& seq = msched.modeSequence;
      const int nP = (int)seq.size();
      for (int i = 0; i < (int)info.numThreeDofContacts; ++i) {
        double stanceEnd_i = 0.0;
        for (int p = 0; p < nP; ++p) {
          if (!modeNumber2StanceLeg(seq[p])[i]) continue;
          int s = 0;      for (int ip = p - 1; ip >= 0; --ip) if (!modeNumber2StanceLeg(seq[ip])[i]) { s = ip; break; }
          int f = nP - 2; for (int ip = p + 1; ip < nP; ++ip) if (!modeNumber2StanceLeg(seq[ip])[i]) { f = ip - 1; break; }
          if (s < (int)ev.size() && f >= 0 && f < (int)ev.size() && ev[s] < t && t < ev[f]) stanceEnd_i = ev[f];
        }
        const bool planted = (stanceEnd_i > t);
        if (!planted && !footLocked_[i]) {                 // liftoff: 착지목표 1회 커밋
          const double dtm = 0.5 * H;
          double rxo = cy * footOff_[i][0] - sy * footOff_[i][1], ryo = sy * footOff_[i][0] + cy * footOff_[i][1];
          footSeed_[i][0] = st.base_pos.x() + baseVx * dtm + rxo;
          footSeed_[i][1] = st.base_pos.y() + baseVy * dtm + ryo;
          footLocked_[i] = true;
        } else if (planted) footLocked_[i] = false;
        region_->updateFoot(i, footSeed_[i][0], footSeed_[i][1], stanceEnd_i);
      }
    }

    // (B) 지형적응 base높이 참조 (TERRAIN_Z). 로봇-상대: oX=현재 base, tau=경과시간.
    if (std::getenv("TERRAIN_Z")) {
      const double oX = st.base_pos.x(), oY = st.base_pos.y(), oYaw = z, tRef = t;
      const double SW = std::getenv("SMOOTH_W") ? atof(std::getenv("SMOOTH_W")) : 0.14;
      auto hS = [&](double x, double y) { double s = 0; int c = 0;
        for (double dx = -SW; dx <= SW + 1e-9; dx += 0.04) for (double dy = -SW; dy <= SW + 1e-9; dy += 0.04) { s += terrainSdf_->height(x + dx, y + dy); ++c; }
        return s / c; };
      const double STEP = 0.3;
      double vxEff = vx_;
      if (!(std::getenv("TERRAIN_CAP") && !std::strcmp(std::getenv("TERRAIN_CAP"), "0"))) {
        const double h0 = hS(oX, oY);
        double slope = 0, rough = 0;
        for (double ff = 0.0; ff <= 1.0 + 1e-9; ff += 0.25) {
          double px = oX + ff * cy, pyv = oY + ff * sy;
          double gX = (hS(px - STEP, pyv) - hS(px + STEP, pyv)) / (2 * STEP);
          double gY = (hS(px, pyv - STEP) - hS(px, pyv + STEP)) / (2 * STEP);
          slope = std::max(slope, std::atan2(std::hypot(gX, gY), 1.0));
          rough = std::max(rough, std::abs(hS(px, pyv) - h0));
        }
        double cap = 1e9;
        if (slope > 0.06) cap = std::min(cap, std::getenv("VXCAP_SLOPE") ? atof(std::getenv("VXCAP_SLOPE")) : 0.2);
        if (rough > 0.02) cap = std::min(cap, std::getenv("VXCAP_ROUGH") ? atof(std::getenv("VXCAP_ROUGH")) : 0.15);
        vxEff = std::max(-cap, std::min(vx_, cap));
      }
      const int N = 11; std::vector<scalar_t> tt(N); std::vector<vector_t> xs(N), us(N);
      for (int n = 0; n < N; ++n) {
        double tn = t + (double)n * H / (N - 1), tau = tn - tRef;
        vector_t xn = x0_;
        double yawN = oYaw + w_ * tau;
        double bx = oX + vxEff * tau * std::cos(oYaw + 0.5 * w_ * tau) - vy_ * tau * std::sin(oYaw);
        double by = oY + vxEff * tau * std::sin(oYaw + 0.5 * w_ * tau) + vy_ * tau * std::cos(oYaw);
        double nX = (hS(bx - STEP, by) - hS(bx + STEP, by)) / (2 * STEP);
        double nY = (hS(bx, by - STEP) - hS(bx, by + STEP)) / (2 * STEP);
        double vx_n = std::cos(yawN) * nX + std::sin(yawN) * nY;
        double pitch = std::atan2(vx_n, 1.0);
        ocs2::centroidal_model::getBasePose(xn, info)(0) = bx;
        ocs2::centroidal_model::getBasePose(xn, info)(1) = by;
        ocs2::centroidal_model::getBasePose(xn, info)(3) = yawN;
        ocs2::centroidal_model::getBasePose(xn, info)(4) = pitch;
        ocs2::centroidal_model::getBasePose(xn, info)(2) = hS(bx, by) + comHeight_ / std::cos(pitch);
        tt[n] = tn; xs[n] = xn; us[n] = vector_t::Zero(info.inputDim);
      }
      interface_->getReferenceManagerPtr()->setTargetTrajectories(ocs2::TargetTrajectories(std::move(tt), std::move(xs), std::move(us)));
    }
  }

  // quat(wxyz) → eulerZYX (z=yaw,y=pitch,x=roll). test02legMujoco quat2zyx와 동일(mju 대신 Eigen).
  static void quatZYX(const Eigen::Vector4d& quat, double& z, double& y, double& x) {
    Eigen::Quaterniond q(quat(0), quat(1), quat(2), quat(3));
    Eigen::Matrix3d R = q.toRotationMatrix();
    z = std::atan2(R(1, 0), R(0, 0));
    y = std::atan2(-R(2, 0), std::sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
    x = std::atan2(R(2, 1), R(2, 2));
  }

  // OCS2/WBC 상태
  std::unique_ptr<LeggedRobotInterface> interface_;
  std::unique_ptr<ocs2::CentroidalModelRbdConversions> rbd_;
  std::unique_ptr<ocs2::MPC_BASE> mpc_;
  std::unique_ptr<ocs2::MPC_MRT_Interface> mrt_;
  std::shared_ptr<SwitchedModelReferenceManager> refMgr_;
  std::unique_ptr<WbcLegged> wbc_;
  std::shared_ptr<MjTerrainSdf> terrainSdf_;
  std::shared_ptr<LocalConvexRegion> region_;
  std::array<int, 4> footId_{};
  vector_t x0_, rotorArm_;
  int nJ_ = 12;
  double dt_ = 0.001, comHeight_ = 0.50, mpcHorizon_ = 1.0, cmdTau_ = 0.30;
  int mpcDecim_ = 10;
  std::string gait_ = "trot", activeSeq_ = "trot", gaitInfoPath_;
  // 명령/발판 상태
  double vx_ = 0, vy_ = 0, w_ = 0;
  double footOff_[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
  double footSeed_[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
  bool footLocked_[4] = {false, false, false, false};
  long stepCount_ = 0;
};

}  // namespace d1
