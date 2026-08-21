// ================================================================================================
// D1 배포 진입점 — HAL(MuJoCo 백엔드) → State → D1Controller → τ → HAL 루프.
//   ★핵심: 컨트롤러는 State/Terrain/Command만 보고 τ만 낸다. real 이식 시 이 루프의 HAL/Terrain만 교체.
//   sim 재현으로 HAL 경계 검증(경사 등반 falls=0 = test02legMujoco와 동일 확인).
// 사용: d1_deploy <task.info> <urdf> <reference.info> <mjcf> [gait] [simTime]
//   env: VX(고정 전진, cmdfile 없을 때)·CMDFILE(GUI JSON)·PERCEPTIVE/PLACEMENT/TERRAIN_Z/W_BASE/MPC_HZ 등
//        (컨트롤러가 test02legMujoco와 동일 env 소비)
// ================================================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <fstream>
#include <iterator>
#include "d1_controller.hpp"
#include "mujoco_backend.hpp"
#include "estimator/state.hpp"

using namespace d1;

int main(int argc, char** argv) {
  setenv("OMP_NUM_THREADS", "1", 0);   // 단일스레드 결정성(권장)
  if (argc < 5) { std::fprintf(stderr, "사용: d1_deploy <task.info> <urdf> <reference.info> <mjcf> [gait] [simTime]\n"); return 1; }
  const std::string taskFile = argv[1], urdfFile = argv[2], refFile = argv[3], mjcfFile = argv[4];
  const std::string gait = (argc > 5) ? argv[5] : "trot";
  const double simTime = (argc > 6) ? atof(argv[6]) : 10.0;
  const std::string gaitDir = refFile.substr(0, refFile.find_last_of('/') + 1);

  // ── HAL(MuJoCo 백엔드) ──
  D1MujocoHal hal;
  if (!hal.load(mjcfFile)) return 1;
  hal.setupPhysics();
  const double dt = hal.model()->opt.timestep;

  // ── 컨트롤러(OCS2) ──
  D1Controller ctrl;
  ctrl.setup(taskFile, urdfFile, refFile, gaitDir, gait, dt);

  // ── 매핑/초기포즈/발오프셋 ──
  hal.buildMapping(ctrl.jointNames());
  { auto jc = ctrl.initialJointConfig(); std::vector<double> jNom(jc.data(), jc.data() + jc.size());
    hal.setInitialPose(ctrl.initialBaseZ(), jNom); }
  { double off[4][2]; hal.footOffsets(off); ctrl.setFootOffsets(off); }
  MujocoTerrain mjTerrain(hal.model(), hal.data());
  // ★검증/실기 데모: SYNTH_FLAT=1이면 인지 heightmap(HeightmapTerrainProvider, 평지 콜백)으로 — real 인지 플러그 배관 확인.
  HeightmapTerrainProvider flatTerrain([](double, double) { return 0.0; });
  TerrainProvider& terrain = std::getenv("SYNTH_FLAT") ? static_cast<TerrainProvider&>(flatTerrain) : static_cast<TerrainProvider&>(mjTerrain);

  // ── 명령 소스: CMDFILE(GUI) 또는 VX(고정) ──
  const char* cmdfile = std::getenv("CMDFILE");
  const double vxEnv = std::getenv("VX") ? atof(std::getenv("VX")) : 0.3;
  double resetSeqPrev = 0;
  auto readCmd = [&](Command& c) {
    if (!cmdfile) { c.vx = vxEnv; c.mode = "move"; c.gait = ""; return; }
    std::ifstream f(cmdfile); if (!f.good()) return;
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    auto num = [&](const char* k, double& o) { std::string key = std::string("\"") + k + "\"";
      auto p = s.find(key); if (p == std::string::npos) return; p = s.find(':', p + key.size());
      if (p == std::string::npos) return; o = atof(s.c_str() + p + 1); };
    auto str = [&](const char* k, std::string& o) { std::string key = std::string("\"") + k + "\"";
      auto p = s.find(key); if (p == std::string::npos) return; p = s.find(':', p + key.size());
      if (p == std::string::npos) return; p = s.find('"', p); if (p == std::string::npos) return;
      auto q = s.find('"', p + 1); if (q == std::string::npos) return; o = s.substr(p + 1, q - p - 1); };
    num("v", c.vx); num("vy", c.vy); num("w", c.w); str("gait", c.gait); str("mode", c.mode);
    double rs = resetSeqPrev; num("reset_seq", rs); num("home_seq", rs);
    if ((int)rs != (int)resetSeqPrev) { resetSeqPrev = rs; hal.reset();
      double fp[4][2]; hal.footPositions(fp); ctrl.resetFootholds(fp);
      std::fprintf(stderr, "  [RESET] t=%.2f\n", hal.data()->time); }
  };

  std::fprintf(stderr, "[D1-DEPLOY] dt=%.4f nJ=%d gait=%s | HAL 경계 뒤 OCS2 제어\n", dt, ctrl.nJ(), gait.c_str());
  std::fprintf(stderr, "  t[s]   base_z   tilt°   base_x\n");

  // ── 메인 루프 ──
  const int nSteps = (int)(simTime / dt);
  int falls = 0; Command cmd;
  qc::State st; st.q.resize(ctrl.nJ()); st.dq.resize(ctrl.nJ());
  for (int step = 0; step < nSteps; ++step) {
    if (step % 20 == 0) readCmd(cmd);        // 50Hz 명령 갱신
    hal.readState(st);                       // 센서 → State(sim GT)
    auto tau = ctrl.update(st, terrain, cmd);// 제어 → τ
    hal.applyTorque(tau);                    // 명령
    hal.step();                              // 1스텝
    // 진단
    mjData* d = hal.data();
    double tilt = std::acos(std::max(-1.0, std::min(1.0, 1 - 2 * (d->qpos[4] * d->qpos[4] + d->qpos[5] * d->qpos[5])))) * 180 / M_PI;
    if (d->qpos[2] < 0.20 || tilt > 60) falls++;
    if (step % 250 == 0) std::fprintf(stderr, "  %6.3f  %6.3f  %5.1f  %6.3f\n", d->time, d->qpos[2], tilt, d->qpos[0]);
  }
  mjData* d = hal.data();
  std::fprintf(stderr, "\n===== 결과 =====\n");
  std::fprintf(stderr, "  최종 base_x : %.3f m\n", d->qpos[0]);
  std::fprintf(stderr, "  최종 base_z : %.3f m\n", d->qpos[2]);
  std::fprintf(stderr, "  낙상 스텝수 : %d  %s\n", falls, falls ? "✗" : "✅ falls=0");
  return 0;
}
