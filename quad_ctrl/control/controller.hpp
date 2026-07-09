#pragma once
// 조율층: State + HighCmd → LowCmd. 모드FSM이 WBIC/MPC/gait/getup을 선택·호출.
//   여기엔 "무엇을 언제 부를지"만. 실제 수식은 각 모듈(wbic/mpc/gait/getup)에.
#include "estimator/state.hpp"
#include "hal/robot_interface.hpp"
#include "command/sport_client.hpp"
#include "control/wbic.hpp"
#include "control/mpc.hpp"
#include "control/gait.hpp"
#include "control/mode_fsm.hpp"
#include "planner/getup_traj.hpp"

namespace qc {

class QuadController {
  ModeFsm  fsm_;
  Gait     gait_;
  Mpc      mpc_;
  Wbic     wbic_;
  GetupTraj getup_;
  HighCmd  cmd_;
  int nu_;
 public:
  explicit QuadController(int nu /*, const Config&*/) : nu_(nu) {
    // TODO: 모델 로드, gait 프리셋, getup_.load(cfg.getup.traj)
  }

  void set_command(const HighCmd& c) {
    cmd_ = c;
    fsm_.request(c.mode);
  }

  // 1틱 제어. 안전기본값=현재자세 홀드(kp/kd만).
  void step(const State& st, LowCmd& out) {
    // TODO: switch(fsm_.current())
    //   Off        → damp (kp=0, kd=REST)
    //   Sit        → haunch fold-in + PD홀드
    //   StandUp    → getup_.active()? getup_.step(...) : wbic 상승
    //   Stand      → wbic_.stance(...)
    //   Walk       → gait_.schedule → mpc_.plan → wbic_.track
    (void)st; (void)out;
  }
};

}  // namespace qc
