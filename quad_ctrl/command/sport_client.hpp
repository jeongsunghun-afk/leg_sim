#pragma once
// 고수준 명령(GUI/조이스틱/원격 → 컨트롤러). 현 teleop_gui_17dof + CMDFILE(/tmp/quad_cmd.json) 대체.
//   Unitree SportClient(Move/StandUp/...) 유사. 컨트롤러는 HighCmd만 받는다.
namespace qc {

enum class Mode { Off, StandDown, Sit, StandUp, Stand, Walk };

struct HighCmd {
  Mode mode = Mode::Off;
  double vx = 0, vy = 0, wz = 0;   // Move(전진/횡/선회)
  double steer = 0;                // 자동차식 조향각[rad] (허리 핸들)
  double body_h = 0.52;            // 서기 높이
  double step_h = 0.10;            // 발 들림
  const char* gait = "trot";       // trot/walk/run
};

}  // namespace qc
