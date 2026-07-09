#pragma once
// offline 기립 궤적(gather) 로드·추종. sit→gather(CoM 전진)→일어서기.
// ★이관 원본: trot_controller.hpp의 getup 블록(load_getup/phaseA PD추종+속도FF/phaseB wbic 인계)
//   궤적 생성: getup_kinematic.py(gather) 또는 getup_mppi.py(contact-implicit) → getup_traj.txt.
//   ★속도 피드포워드(KD·(dq−qvel)) 필수(없으면 전복). phaseA=PD, phaseB=WBIC 상승.
#include <vector>
#include <string>
#include "estimator/state.hpp"
#include "hal/robot_interface.hpp"

namespace qc {

class GetupTraj {
  std::vector<VectorXd> q_, dq_;
  std::vector<int> phase_;          // 0=A1,1=A2,2=B
  double dt_ = 0.01;
  int N_ = 0, k_ = -1;
 public:
  bool load(const std::string& path);   // TODO: "N dt \n phase q[nu] dq[nu]" 파싱
  bool active() const { return k_ >= 0; }
  void reset() { k_ = -1; }
  // phaseA면 out에 PD추종토크 채우고 true, phaseB 진입/종료면 false(→ wbic 인계).
  // bool step(const State&, double kp, double kd, LowCmd& out);   // TODO 이관
};

}  // namespace qc
