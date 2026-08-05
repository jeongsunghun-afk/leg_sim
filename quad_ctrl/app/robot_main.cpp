// robot_main — 실기(Pi) 진입점: RealHal + Estimator + Controller 를 RT 루프로.
//   ★sim(sim_bridge)과 동일 구조 — HAL만 RealHal(실모터/센서)로 교체. 컨트롤러·estimator 불변(=이관 핵심).
//     루프: RealHal.read → EkfEstimator.update(→d_est) → TrotBridge.step(d_est) → RealHal.write(tau_ff, kp=kd=0 순수토크).
//   ※Pi 전용(RobotSharedMem). 데스크톱은 자동 skip(#else 스텁).
#include <cstdio>
#include <ctime>
#include <csignal>
#include "hal/real_hal.hpp"

#if defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h")
#include "quad_control.hpp"                 // ::QuadControl (모델=M/h/Jac 계산기)
#include "trot_controller.hpp"              // apply_env_gains
#include "estimator/ekf_estimator.hpp"
#include "control/trot_bridge.hpp"
#include "command/sport_client.hpp"
#include "common/config.hpp"
#include "config/joint_map_17dof.hpp"

using namespace qc;
static volatile sig_atomic_t g_run = 1;
static void on_sigint(int) { g_run = 0; }

int main(int argc, char** argv) {
  load_config_env();                          // QC_CONFIG(게인·물리·관절범위)
  const char* mjcf = argc > 1 ? argv[1]
      : "/home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf";

  // ★모델(동역학 계산기) — sim과 동일 셋업. 실기서도 MuJoCo를 M/h/Jac 계산에 씀(물리엔진 아님, mj_step 안함).
  ::QuadControl q;
  q.load(mjcf); apply_env_gains(q); q.crouch_home(); q.build_qhome_lut(); q.setup_mpc();

  EkfEstimator ekf(q);                        // KF 추정 → d_est
  TrotBridge   ctrl(q);                        // 검증된 MPC+WBIC(순수토크)
  RealHal      hal(q.nu, q.m->opt.timestep, joint_map_17dof());   // ★관절맵=축별 JOG 실측 확정 필요

  HighCmd hc; hc.mode = Mode::Stand;           // TODO(nav/teleop): HighCmd(cmd_vel)를 SportClient 경계서 수신
  ctrl.set_command(hc);

  std::signal(SIGINT, on_sigint);
  // TODO(Pi): SCHED_FIFO + mlockall 로 RT 우선순위(지연 예산 <12ms 위해). 여기선 CLOCK_MONOTONIC 절대주기.
  const long period_ns = (long)(q.m->opt.timestep * 1e9);       // 제어주기(=모델 timestep, 통상 1ms)
  struct timespec next; clock_gettime(CLOCK_MONOTONIC, &next);

  LowState ls; LowCmd cmd; long status_cnt = 0, miss = 0;
  while (g_run) {
    if (hal.read(ls)) {                        // 센서(신선한 상태)
      if (++status_cnt > 100) hal.enable(true);// ★안전: 상태 100프레임 수신 후 명령 전송 허용(RobotTestGait 패턴)
      ekf.update(ls);                          // 추정 → d_est
      ctrl.step(cmd, ekf.d_est());             // 제어 → LowCmd(순수토크 tau_ff, kp=kd=0)
      hal.write(cmd);                          // 명령(전송 enable 시)
      miss = 0;
    } else if (++miss > 10) {
      hal.enable(false);                       // ★통신 두절 → limp(안전정지)
    }
    next.tv_nsec += period_ns;                 // 다음 주기 절대시각
    while (next.tv_nsec >= 1000000000L) { next.tv_nsec -= 1000000000L; next.tv_sec++; }
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
  }
  hal.enable(false);                           // 종료 시 limp
  std::printf("[robot_main] 종료 (status_cnt=%ld)\n", status_cnt);
  return 0;
}
#else
int main() { std::printf("[robot_main] Pi 전용(RobotSharedMem.h 없음) — 데스크톱 빌드 skip\n"); return 0; }
#endif
