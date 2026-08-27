// ================================================================================================
// D1(OCS2 perceptive NMPC) 기반 오프라인 발판/base 캐시 생성기 — DTC teacher용.
//   cpp/tamols/cache_gen_stairs.cpp 의 드롭인 대체: 동일 셀그리드(vx×step_h)·동일 4파일 포맷
//   (footholds.bin [cells,4,3] · base.bin [cells,n_samp,12] · contacts.bin [cells,n_samp,4] · meta.json)
//   동일 로컬프레임(base 시작=원점·z 상대화). 차이=계획기가 TAMOLS QP(GIAC proxy) 대신
//   D1 센트로이드 OCP(실제 접촉력+마찰뿔+전신운동학 = 동역학-정합 teacher).
//   ★가치 전제(실측 P2.7/P2.9): teacher = 샘플효율 레버(점근 능력 아님).
//   planner-only: MuJoCo 없음 — 지형=MjTerrainSdf::setGrid(절차적 계단), 솔브=SqpMpc.run 직접.
// 사용: d1_cache_gen <task.info> <urdf> <reference.info> [outdir=stair_cache_d1]
//   env: D1CACHE_GAIT(기본 static_walk)·CLEARANCE 등 D1 env 그대로.
// ================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <Eigen/Dense>

#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_legged_robot/gait/MotionPhaseDefinition.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_legged_robot/reference_manager/SwitchedModelReferenceManager.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>
#include <ocs2_core/soft_constraint/StateSoftConstraint.h>
#include <ocs2_core/penalties/penalties/RelaxedBarrierPenalty.h>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_sqp/SqpMpc.h>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include "wbc_legged.hpp"              // (include 체인용; 미사용)
#include "mj_terrain_sdf.hpp"
#include "foot_terrain_clearance.hpp"
#include "foot_terrain_placement.hpp"
#include "local_convex_region.hpp"

using namespace ocs2;
using namespace ocs2::legged_robot;

int main(int argc, char** argv) {
  setenv("OMP_NUM_THREADS", "1", 0);
  if (argc < 4) { std::fprintf(stderr, "사용: d1_cache_gen <task.info> <urdf> <reference.info> [outdir]\n"); return 1; }
  const std::string taskFile = argv[1], urdfFile = argv[2], refFile = argv[3];
  const std::string outdir = (argc > 4) ? argv[4] : "stair_cache_d1";
  const std::string gaitDir = refFile.substr(0, refFile.find_last_of('/') + 1);
  const std::string gaitName = std::getenv("D1CACHE_GAIT") ? std::getenv("D1CACHE_GAIT") : "static_walk";

  // ── 셀 그리드: cache_gen_stairs.cpp 동일 ──
  const std::vector<double> vx_vals = {0.2, 0.3, 0.4};
  const std::vector<double> sh_vals = {0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15};
  const double step_d = 0.35, dt_s = 0.02, comH = 0.50;
  const int n_vx = (int)vx_vals.size(), n_sh = (int)sh_vals.size();
  auto cidx = [&](int i, int s) { return (long)(i * n_sh + s); };

  // ── OCS2 셋업 (D1Controller::setup의 planner-only 부분) ──
  LeggedRobotInterface interface(taskFile, urdfFile, refFile, false);
  const auto& info = interface.getCentroidalModelInfo();
  auto terrainSdf = std::make_shared<MjTerrainSdf>();
  auto region = std::make_shared<LocalConvexRegion>(terrainSdf);
  {  // perceptive 제약(클리어런스+발판배치) — test02legMujoco와 동일 주입
    auto& prob = interface.getMutableOptimalControlProblem();
    auto pRefMgr = std::dynamic_pointer_cast<SwitchedModelReferenceManager>(interface.getReferenceManagerPtr());
    const auto& ms = interface.modelSettings();
    double clr = std::getenv("CLEARANCE") ? atof(std::getenv("CLEARANCE")) : 0.04;
    for (size_t i = 0; i < info.numThreeDofContacts; ++i) {
      const std::string& footName = ms.contactNames3DoF[i];
      const auto infoCppAd = info.toCppAd();
      const CentroidalModelPinocchioMappingCppAd mapCppAd(infoCppAd);
      auto velCb = [infoCppAd](const ad_vector_t& state, PinocchioInterfaceCppAd& pAd) {
        const ad_vector_t q = centroidal_model::getGeneralizedCoordinates(state, infoCppAd);
        updateCentroidalDynamics(pAd, infoCppAd, q);
      };
      std::unique_ptr<EndEffectorKinematics<scalar_t>> eeK1(new PinocchioEndEffectorKinematicsCppAd(
          interface.getPinocchioInterface(), mapCppAd, {footName}, info.stateDim, info.inputDim,
          velCb, footName + "_perc", ms.modelFolderCppAd, ms.recompileLibrariesCppAd, ms.verboseCppAd));
      std::unique_ptr<PenaltyBase> pen1(new RelaxedBarrierPenalty(RelaxedBarrierPenalty::Config(1e-2, 1e-3)));
      std::unique_ptr<StateConstraint> con1(new FootTerrainClearanceConstraint(*pRefMgr, *eeK1, terrainSdf, i, clr));
      prob.stateSoftConstraintPtr->add(footName + "_terrainClearance",
          std::unique_ptr<StateCost>(new StateSoftConstraint(std::move(con1), std::move(pen1))));
      std::unique_ptr<EndEffectorKinematics<scalar_t>> eeK2(new PinocchioEndEffectorKinematicsCppAd(
          interface.getPinocchioInterface(), mapCppAd, {footName}, info.stateDim, info.inputDim,
          velCb, footName + "_place", ms.modelFolderCppAd, ms.recompileLibrariesCppAd, ms.verboseCppAd));
      std::unique_ptr<PenaltyBase> pen2(new RelaxedBarrierPenalty(RelaxedBarrierPenalty::Config(1e-2, 1e-4)));
      std::unique_ptr<StateConstraint> con2(new FootTerrainPlacementConstraint(*pRefMgr, *eeK2, region, i));
      prob.stateSoftConstraintPtr->add(footName + "_footPlacement",
          std::unique_ptr<StateCost>(new StateSoftConstraint(std::move(con2), std::move(pen2))));
    }
  }
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(),
             interface.getOptimalControlProblem(), interface.getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  auto refMgr = std::dynamic_pointer_cast<SwitchedModelReferenceManager>(interface.getReferenceManagerPtr());
  auto tmpl = loadModeSequenceTemplate(gaitDir + "gait.info", gaitName, false);
  refMgr->getGaitSchedule()->insertModeSequenceTemplate(tmpl, 0.0, 1e4);

  // FK용 pinocchio 복사 + 매핑
  PinocchioInterface pin = interface.getPinocchioInterface();
  CentroidalModelPinocchioMapping map(info); map.setPinocchioInterface(pin);
  const char* footFrame[4] = {"FL_foot_contact_link", "FR_foot_contact_link", "HL_foot_contact_link", "HR_foot_contact_link"};
  std::array<int, 4> footId;
  for (int i = 0; i < 4; ++i) footId[i] = pin.getModel().getFrameId(footFrame[i]);
  auto footFK = [&](const vector_t& x, int leg) -> Eigen::Vector3d {
    vector_t q = map.getPinocchioJointPosition(x);
    pinocchio::forwardKinematics(pin.getModel(), pin.getData(), q);
    pinocchio::updateFramePlacements(pin.getModel(), pin.getData());
    return pin.getData().oMf[footId[leg]].translation();
  };

  const double T = interface.mpcSettings().timeHorizon_;   // 호라이즌(=캐시 구간)
  const int n_samp = (int)std::floor(T / dt_s) + 1;
  const vector_t xInit = interface.getInitialState();

  // nominal 발-base 오프셋(FK at x0, yaw=0)
  double footOff[4][2];
  { vector_t x0 = xInit;
    centroidal_model::getBasePose(x0, info).setZero();
    centroidal_model::getBasePose(x0, info)(2) = comH;
    for (int l = 0; l < 4; ++l) { auto f = footFK(x0, l); footOff[l][0] = f(0); footOff[l][1] = f(1); } }

  std::vector<float> fh((long)n_vx * n_sh * 4 * 3, 0.f);
  std::vector<float> bs((long)n_vx * n_sh * n_samp * 12, 0.f);
  std::vector<float> ct((long)n_vx * n_sh * n_samp * 4, 0.f);
  int nfail = 0;

  for (int i = 0; i < n_vx; ++i) for (int s = 0; s < n_sh; ++s) {
    const double vx = vx_vals[i], sh = sh_vals[s];
    // 절차적 계단(cache_gen_stairs 동일): base가 등반 중간서 시작
    auto hAt = [&](double x, double) { return (sh > 0 && x > -0.5) ? sh * std::floor((x + 0.5) / step_d) : 0.0; };
    terrainSdf->setGrid(0.0, 0.0, hAt);
    const double z_here = hAt(0, 0), z0 = comH + z_here;

    vector_t x0 = xInit;
    centroidal_model::getBasePose(x0, info).setZero();
    centroidal_model::getBasePose(x0, info)(2) = z0;
    centroidal_model::getMomentum(x0, info).setZero();

    // 발판영역 seed(Raibert) — 각 발 첫 스윙의 착지 게이팅은 stanceEnd=첫 liftoff 시각
    auto msched = refMgr->getGaitSchedule()->getModeSchedule(0.0, T);
    for (int l = 0; l < 4; ++l) {
      double liftoff = 0.0;
      for (size_t k = 0; k + 1 < msched.modeSequence.size() && k < msched.eventTimes.size(); ++k)
        if (modeNumber2StanceLeg(msched.modeSequence[k])[l] && !modeNumber2StanceLeg(msched.modeSequence[k + 1])[l]) { liftoff = msched.eventTimes[k]; break; }
      double seedX = footOff[l][0] + vx * 0.5 * T, seedY = footOff[l][1];
      region->updateFoot(l, seedX, seedY, liftoff);
    }

    // 지형적응 참조(TERRAIN_Z식 N=11): bx=vx·t, z=h+comH/cosθ, pitch=경사
    { const int N = 11; std::vector<scalar_t> tt(N); std::vector<vector_t> xs(N), us(N);
      const double STEP = 0.3;
      for (int n = 0; n < N; ++n) {
        double tn = (double)n * T / (N - 1), bx = vx * tn;
        double nX = (hAt(bx - STEP, 0) - hAt(bx + STEP, 0)) / (2 * STEP);
        double pitch = std::atan2(nX, 1.0);
        vector_t xn = x0;
        centroidal_model::getBasePose(xn, info)(0) = bx;
        centroidal_model::getBasePose(xn, info)(4) = pitch;
        centroidal_model::getBasePose(xn, info)(2) = hAt(bx, 0) + comH / std::cos(pitch);
        tt[n] = tn; xs[n] = xn; us[n] = vector_t::Zero(info.inputDim);
      }
      interface.getReferenceManagerPtr()->setTargetTrajectories(TargetTrajectories(std::move(tt), std::move(xs), std::move(us))); }

    // 솔브: cold+웜 체인(3회) — cache_gen_stairs 패턴
    bool ok = true;
    mpc.reset();
    for (int w = 0; w < 3; ++w) { try { mpc.run(0.0, x0); } catch (const std::exception& e) { ok = false; } }
    PrimalSolution sol;
    if (ok) { mpc.getSolverPtr()->getPrimalSolution(T, &sol); ok = !sol.timeTrajectory_.empty(); }

    // 추출: 발판=발별 첫 터치다운 FK(없으면 t=0 stance FK) · base=0.02s 샘플 · contacts=모드
    if (ok) {
      const auto& mss = sol.modeSchedule_;
      double fhW[4][3];
      for (int l = 0; l < 4; ++l) {
        double td = -1;
        for (size_t k = 0; k + 1 < mss.modeSequence.size() && k < mss.eventTimes.size(); ++k)
          if (!modeNumber2StanceLeg(mss.modeSequence[k])[l] && modeNumber2StanceLeg(mss.modeSequence[k + 1])[l]) { td = mss.eventTimes[k]; break; }
        vector_t xE = (td >= 0) ? LinearInterpolation::interpolate(std::min(td, sol.timeTrajectory_.back()), sol.timeTrajectory_, sol.stateTrajectory_)
                                : sol.stateTrajectory_.front();
        auto f = footFK(xE, l);
        fhW[l][0] = f(0); fhW[l][1] = f(1); fhW[l][2] = f(2);
        // 게이트: 발판이 지형 위인가(±4cm)
        if (std::fabs(f(2) - hAt(f(0), f(1))) > 0.04) ok = false;
      }
      if (ok) {
        long c = cidx(i, s);
        for (int l = 0; l < 4; ++l) { fh[c * 12 + l * 3 + 0] = (float)fhW[l][0]; fh[c * 12 + l * 3 + 1] = (float)fhW[l][1];
                                      fh[c * 12 + l * 3 + 2] = (float)(fhW[l][2] - z_here); }   // z 상대화
        // base 샘플(pose=[x,y,z,roll,pitch,yaw] TAMOLS 순서, z-z0) + vel=유한차분(euler-rate, TAMOLS 스플라인 미분 등가)
        std::vector<std::array<double, 6>> pose(n_samp);
        for (int n = 0; n < n_samp; ++n) {
          double tn = std::min((double)n * dt_s, sol.timeTrajectory_.back());
          vector_t xN = LinearInterpolation::interpolate(tn, sol.timeTrajectory_, sol.stateTrajectory_);
          vector_t bp = centroidal_model::getBasePose(xN, info);   // [x,y,z,yaw,pitch,roll]
          pose[n] = {bp(0), bp(1), bp(2) - z0, bp(5), bp(4), bp(3)};
          size_t mode = mss.modeAtTime(tn);
          auto cf = modeNumber2StanceLeg(mode);
          for (int l = 0; l < 4; ++l) ct[(c * n_samp + n) * 4 + l] = cf[l] ? 1.f : 0.f;
        }
        for (int n = 0; n < n_samp; ++n) {
          int n2 = std::min(n + 1, n_samp - 1), n1 = std::max(n - 1, 0);
          for (int d = 0; d < 6; ++d) {
            bs[(c * n_samp + n) * 12 + d] = (float)pose[n][d];
            bs[(c * n_samp + n) * 12 + 6 + d] = (float)((pose[n2][d] - pose[n1][d]) / ((n2 - n1) * dt_s));
          }
        }
      }
    }
    if (!ok) ++nfail;
    std::fprintf(stderr, "  [%d/%d] vx=%.1f sh=%.2f %s\n", (int)cidx(i, s) + 1, n_vx * n_sh, vx, sh, ok ? "ok" : "FAIL");
  }

  // ── 쓰기: cache_gen_stairs 동일 4파일 ──
  auto wbin = [&](const std::string& nm, const std::vector<float>& v) {
    std::ofstream f(outdir + "/" + nm, std::ios::binary); f.write((const char*)v.data(), v.size() * sizeof(float)); };
  if (system(("mkdir -p " + outdir).c_str())) {}
  wbin("footholds.bin", fh); wbin("base.bin", bs); wbin("contacts.bin", ct);
  { std::ofstream mj(outdir + "/meta.json");
    mj << "{\n  \"planner\": \"D1-OCS2 perceptive NMPC (centroidal OCP: forces+friction+kinematics exact)\",\n";
    mj << "  \"gait\": \"" << gaitName << " (D1 gait.info)\",\n  \"vx\": ["; for (int i = 0; i < n_vx; ++i) mj << vx_vals[i] << (i + 1 < n_vx ? ", " : "");
    mj << "],\n  \"step_h\": ["; for (int s = 0; s < n_sh; ++s) mj << sh_vals[s] << (s + 1 < n_sh ? ", " : "");
    mj << "],\n  \"step_d\": " << step_d << ",\n  \"n_samp\": " << n_samp << ",\n  \"dt\": " << dt_s << ",\n  \"horizon\": " << T << ",\n  \"base_h\": " << comH << ",\n";
    mj << "  \"shapes\": {\"footholds\": [" << n_vx << ", " << n_sh << ", 4, 3], \"base\": [" << n_vx << ", " << n_sh << ", " << n_samp << ", 12], \"contacts\": [" << n_vx << ", " << n_sh << ", " << n_samp << ", 4]},\n";
    mj << "  \"foot_order\": \"FL,FR,RL,RR\",\n";
    mj << "  \"frame\": \"local: base start origin, +x fwd, foot/base z RELATIVE to base tread z; world = base_pos + Rz(yaw)*local (z incl)\"\n}\n"; }
  std::fprintf(stderr, "[d1_cache_gen] %s: %d셀 (fail %d) n_samp=%d gait=%s\n", outdir.c_str(), n_vx * n_sh, nfail, n_samp, gaitName.c_str());
  return 0;
}
