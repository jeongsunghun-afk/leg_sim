#pragma once
// 모드 상태기계: off / stand_down(눕기) / sit(앉기) / stand_up(기립) / stand / walk.
// ★이관 원본: simulation/quad/cpp/src/trot_controller.hpp (control()의 mode dispatch)
//   전환·홀드·저-PD fold·from_sit·haunch fold-in/getup 로직을 여기로 집약.
//   각 모드가 wbic/mpc/gait/getup 중 무엇을 호출할지 결정하는 얇은 조율층.
#include "command/sport_client.hpp"
#include "estimator/state.hpp"

namespace qc {

class ModeFsm {
  Mode cur_ = Mode::Off;
 public:
  Mode current() const { return cur_; }
  void request(Mode m) { cur_ = m; }   // TODO: 전환 가드(안전조건)·램프
  // 모드별 서브컨트롤러 선택은 QuadController::step에서. FSM은 상태·전환만.
};

}  // namespace qc
