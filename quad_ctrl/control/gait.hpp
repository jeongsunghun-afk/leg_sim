#pragma once
// 게이트 스케줄 + 반응형 발배치(Raibert). 노드별 접촉집합, 스윙궤적, 착지점.
// ★이관 원본: simulation/quad/cpp/src/trot_controller.hpp
//     - set_gait()/gait()      : trot/walk/run 프리셋, 위상→접촉
//     - Raibert 발배치          : foot = 0.5·v·T_st(중립) + k·(v−v_des)(capture). raibert_k=0.5 표준중립
//     - 자동whip(paw-flick)     : 고속 trot 수동 whip
#include "estimator/state.hpp"

namespace qc {

struct GaitParam { double T = 0.5, swf = 0.5, raibert = 0.5, step_h = 0.10; double off[NLEG] = {0, .5, .5, 0}; };

class Gait {
  GaitParam p_;
 public:
  void set_preset(const char* name);   // TODO: "trot"/"walk"/"run" → p_
  // 위상 tg → 다리별 stance/swing + 스윙목표(Raibert). MPC 접촉스케줄·WBIC 스윙에 공급.
  // void schedule(const State&, double tg, ...);   // TODO 이관
};

}  // namespace qc
