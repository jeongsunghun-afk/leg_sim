// sim_bridge — 이관 1단계 검증 진입점: MujocoHal + TrotBridge 로 HAL 경계 루프.
//   목표: `sim_bridge` 결과(x·z·tilt·falls)가 `quad/cpp` `trot_sim`(GT, NO_JUMP_WARMUP)과 동등.
//   ★컨트롤러는 재작성 없이 HAL 뒤에서 동작 → real 이식 시 이 루프의 HAL(read/write)만 real로 교체.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "hal/mujoco_hal.hpp"
#include "control/trot_bridge.hpp"
#include "command/sport_client.hpp"

using namespace qc;

int main(int argc, char** argv) {
  const char* mjcf = argc > 1 ? argv[1]
                     : "/home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf";
  int STEPS = getenv("STEPS") ? atoi(getenv("STEPS")) : 3000;

  MujocoHal hal(mjcf);
  TrotBridge ctrl(hal.core());

  HighCmd hc; hc.mode = Mode::Walk;                        // 기본=보행
  if (getenv("MODE")) { const char* m = getenv("MODE");
    if      (!strcmp(m, "off"))        hc.mode = Mode::Off;
    else if (!strcmp(m, "sit"))        hc.mode = Mode::Sit;
    else if (!strcmp(m, "stand_up"))   hc.mode = Mode::StandUp;
    else if (!strcmp(m, "stand_down")) hc.mode = Mode::StandDown;
    else if (!strcmp(m, "stand"))      hc.mode = Mode::Stand;
    else                                hc.mode = Mode::Walk; }
  if (getenv("GAIT"))   hc.gait = getenv("GAIT");
  if (getenv("TROT_V")) hc.vx   = atof(getenv("TROT_V")); else hc.vx = 0.30;
  hc.body_h = getenv("BODY_H") ? atof(getenv("BODY_H")) : 0.0;   // 0=모델 기본(base_z0)=trot_sim 정합
  ctrl.set_command(hc);

  mjData* d = hal.core().d;
  LowState ls; LowCmd cmd;
  int falls = 0; double max_tilt = 0;
  for (int k = 0; k < STEPS; ++k) {
    hal.read(ls);                                          // 센서 → LowState
    ctrl.step(cmd);                                        // 제어 → LowCmd(tau)
    hal.write(cmd);                                        // 명령 적용 + 1스텝
    // 진단(trot_sim 정합): tilt = body-z 대 world-z 각
    double qx = d->qpos[4], qy = d->qpos[5];
    double td = std::acos(std::max(-1.0, std::min(1.0, 1.0 - 2.0 * (qx*qx + qy*qy)))) * 57.29578;
    max_tilt = std::max(max_tilt, td);
    if (td > 50 || d->qpos[2] < 0.2) falls++;
  }
  std::printf("[sim_bridge] STEPS=%d(%.1fs) x=%+.3f z=%.3f max_tilt=%.1f falls=%d "
              "| HAL+TrotCtrl 배선 == trot_sim 검증\n",
              STEPS, STEPS * hal.dt(), d->qpos[0], d->qpos[2], max_tilt, falls);
  return 0;
}
