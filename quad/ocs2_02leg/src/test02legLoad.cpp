// 02_Leg OCS2 포팅 검증: 인터페이스 로드 → 모델정보 → SqpMpc 1스텝(평지 trot 계획).
#include <iostream>
#include <ocs2_legged_robot/LeggedRobotInterface.h>
#include <ocs2_centroidal_model/AccessHelperFunctions.h>
#include <ocs2_centroidal_model/ModelHelperFunctions.h>
#include <ocs2_sqp/SqpMpc.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>
#include <ocs2_legged_robot/gait/MotionPhaseDefinition.h>
#include <ocs2_legged_robot/gait/ModeSequenceTemplate.h>
#include <ocs2_legged_robot/reference_manager/SwitchedModelReferenceManager.h>

using namespace ocs2;
using namespace legged_robot;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: test02legLoad <task.info> <urdf> <reference.info> [gait.info]\n";
    return 1;
  }
  const std::string taskFile = argv[1], urdfFile = argv[2], referenceFile = argv[3];
  const std::string gaitFile = (argc > 4) ? argv[4] : "";

  std::cerr << "\n===== [1] LeggedRobotInterface 로드 (CppAD codegen 발생) =====\n";
  LeggedRobotInterface interface(taskFile, urdfFile, referenceFile, /*hardFriction*/ false);
  const auto& info = interface.getCentroidalModelInfo();

  std::cerr << "\n===== [2] 모델 정보 =====\n";
  std::cerr << "  centroidalModelType : " << static_cast<int>(info.centroidalModelType) << " (1=SRBD)\n";
  std::cerr << "  robotMass           : " << info.robotMass << " kg\n";
  std::cerr << "  stateDim            : " << info.stateDim << "\n";
  std::cerr << "  inputDim            : " << info.inputDim << "\n";
  std::cerr << "  numThreeDofContacts : " << info.numThreeDofContacts << "\n";
  std::cerr << "  generalizedCoordDim : " << info.generalizedCoordinatesNum << "\n";

  const vector_t x0 = interface.getInitialState();
  std::cerr << "  initial base pose   : "
            << centroidal_model::getBasePose(x0, info).transpose() << "\n";

  std::cerr << "\n===== [3] SqpMpc 구성 + 평지 전진 목표 =====\n";
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(),
             interface.getOptimalControlProblem(), interface.getInitializer());
  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());

  // 옵션: trot gait를 호라이즌 전체에 주입
  bool trot = false;
  if (!gaitFile.empty()) {
    auto tmpl = loadModeSequenceTemplate(gaitFile, "trot", true);
    auto refMgr = std::dynamic_pointer_cast<SwitchedModelReferenceManager>(interface.getReferenceManagerPtr());
    if (refMgr) {
      refMgr->getGaitSchedule()->insertModeSequenceTemplate(tmpl, 0.0, 10.0);
      trot = true;
      std::cerr << "  [gait] trot 주입 (0~10s)\n";
    }
  }

  // 평지 전진 목표: 1s 후 x=+0.3m (0.3 m/s), 자세·높이 유지.
  const scalar_t t0 = 0.0, T = 1.0, vx = 0.3;
  vector_t xGoal = x0;
  centroidal_model::getBasePose(xGoal, info)(0) += vx * T;
  TargetTrajectories tt({t0, t0 + T}, {x0, xGoal},
                        {vector_t::Zero(info.inputDim), vector_t::Zero(info.inputDim)});
  interface.getReferenceManagerPtr()->setTargetTrajectories(std::move(tt));

  std::cerr << "\n===== [4] MPC 1스텝 solve =====\n";
  MPC_MRT_Interface mrt(mpc);
  mrt.initRollout(&interface.getRollout());
  SystemObservation obs;
  obs.time = t0;
  obs.state = x0;
  obs.input = vector_t::Zero(info.inputDim);
  obs.mode = ModeNumber::STANCE;
  mrt.setCurrentObservation(obs);

  bool ok = false;
  for (int i = 0; i < 3; ++i) {  // 몇 번 반복해 워밍업
    try {
      mrt.advanceMpc();
      ok = true;
    } catch (const std::exception& e) {
      std::cerr << "  [iter " << i << "] " << e.what() << "\n";
    }
  }
  if (!ok) { std::cerr << "  MPC solve 실패\n"; return 2; }

  mrt.updatePolicy();

  std::cerr << "\n===== [5] 계획된 발별 수직력 Fz [FL FR HL HR] (N) =====\n";
  std::cerr << "   t[s]      FL      FR      HL      HR    합계   base_x\n";
  for (scalar_t tt2 = 0.02; tt2 <= 0.70; tt2 += 0.07) {
    vector_t xOpt, uOpt;
    size_t md;
    mrt.evaluatePolicy(t0 + tt2, x0, xOpt, uOpt, md);
    scalar_t Fz = 0;
    char buf[160];
    scalar_t f[4];
    for (size_t c = 0; c < 4; ++c) { f[c] = centroidal_model::getContactForces(uOpt, c, info)(2); Fz += f[c]; }
    snprintf(buf, sizeof(buf), "  %4.2f  %6.1f  %6.1f  %6.1f  %6.1f  %6.1f  %+.4f",
             tt2, f[0], f[1], f[2], f[3], Fz, centroidal_model::getBasePose(xOpt, info)(0));
    std::cerr << buf << "\n";
  }
  std::cerr << "  (기대 총합 mg=" << info.robotMass * 9.81 << " N";
  if (trot) std::cerr << " · trot=대각쌍 FL+HR / FR+HL 교대로 Fz→0(유각)";
  std::cerr << ")\n";
  std::cerr << "\n===== OK: 02_Leg OCS2 로드+MPC 계획 생성 성공 =====\n";
  return 0;
}
