// S1/S3-b — 점프 OCP C++ (jump_solver.hpp 사용). crocoddyl FDDP 다상: push→flight→land, 허리 lock=16 leg DOF.
//   이 실행파일 = OCP solve 후 궤적을 배포 replay 포맷(/tmp/jump_traj.txt)으로 파일 저장(gen_jump.sh용).
//   solve 로직은 jump_solver.hpp의 jump_solve()에 있고 trot_view live-solve와 공용.
//   사용: jump_ocp [URDF] [VX] [maxit]   VX=전방 이륙속도(기본 0.6). JUMP_OUT=경로(기본 /tmp/jump_traj.txt).
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <chrono>
#include "jump_solver.hpp"

int main(int argc, char** argv) {
  const std::string URDF = argc > 1 ? argv[1]
      : "/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf";
  const double VX = argc > 2 ? std::atof(argv[2]) : 0.6;
  const int MAXIT = argc > 3 ? std::atoi(argv[3]) : 200;
  const std::string MJCF = "/home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf";

  auto t0 = std::chrono::steady_clock::now();
  JumpTraj J = jump_solve(URDF, MJCF, VX, MAXIT, true);
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (J.N == 0) { std::cerr << "[jump_ocp] solve 실패\n"; return 2; }
  std::cout << "[jump_ocp] solve+변환 " << ms << " ms · 수렴=" << J.ok
            << " apex=" << J.apex << "m 착지tilt=" << J.tilt_land << "°\n";

  // ── 배포 replay 포맷으로 파일 저장 ──
  const char* outp = std::getenv("JUMP_OUT");
  std::string OUT = outp ? outp : "/tmp/jump_traj.txt";
  std::ofstream of(OUT);
  of << J.N << " " << J.dt << "\n";
  const int NJ = (int)J.q[0].size();
  for (int k = 0; k < J.N; k++) {
    of << J.ph[k];
    for (int j = 0; j < NJ; j++) of << " " << J.q[k][j];
    for (int j = 0; j < NJ; j++) of << " " << J.dq[k][j];
    for (int j = 0; j < NJ; j++) of << " " << J.tau[k][j];
    for (int c = 0; c < 3; c++) of << " " << J.com[k][c];    // ★WBIC-추종용 CoM 위치
    for (int c = 0; c < 3; c++) of << " " << J.comv[k][c];   //   속도
    for (int c = 0; c < 3; c++) of << " " << J.acom[k][c];   //   가속
    of << "\n";
  }
  of.close();
  std::cout << "[jump_ocp] 궤적 저장 → " << OUT << " (N=" << J.N << " 노드 · " << NJ << "-DOF MuJoCo순 · 허리=0)\n";
  return J.ok ? 0 : 1;
}
