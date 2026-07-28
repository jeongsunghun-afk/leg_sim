// export_traj.cpp — TAMOLS 해(/tmp/tamols_sol.txt) → TrotCtrl tamols 재생 포맷(/tmp/tamols_traj.txt)
//   포맷(load_tamols): "N dt" / "FOOTHOLDS" 4×(x y z) / "SAMPLES" N×(t pose6 vel6 c0..c3)
//   pose6=(x y z roll pitch yaw) · vel6=(vx vy vz wr wp wy) · c=접촉(1/0).
#include "constraints.hpp"     // load_solution
#include "tamols_track.hpp"    // locate, total_duration
#include <cstdio>
#include <fstream>

using namespace tamols;

int main(int argc, char** argv) {
  const char* in  = (argc > 1) ? argv[1] : "/tmp/tamols_sol.txt";
  const char* out = (argc > 2) ? argv[2] : "/tmp/tamols_traj.txt";
  double dt = (argc > 3) ? atof(argv[3]) : 0.01;

  TamolsState st;
  if (!load_solution(in, st)) { std::printf("★ %s 로드 실패\n", in); return 1; }
  double T = total_duration(st);
  int N = (int)(T / dt);

  bool symy = getenv("SYMY") != nullptr;               // 발판 y 대칭화(±0.14)로 측방안정
  std::ofstream f(out);
  f << N << " " << dt << "\nFOOTHOLDS\n";
  for (int L = 0; L < 4; ++L) {
    double y = st.p(L, 1);
    if (symy) y = std::copysign(0.14, st.p(L, 1));     // 좌(+)/우(-) 대칭 nominal
    f << st.p(L, 0) << " " << y << " " << st.p(L, 2) << "\n";
  }
  f << "SAMPLES\n";
  for (int k = 0; k < N; ++k) {
    double t = k * dt; int ph; double tau; locate(st, t, ph, tau);
    Vector6d pose = st.pos_at(ph, tau), vel = st.vel_at(ph, tau);
    f << t;
    for (int d = 0; d < 6; ++d) f << " " << pose(d);   // x y z roll pitch yaw
    for (int d = 0; d < 6; ++d) f << " " << vel(d);     // vx vy vz wr wp wy(euler율≈각속도)
    for (int i = 0; i < 4; ++i) f << " " << st.gait[ph].contact[i];
    f << "\n";
  }
  std::printf("[export] %s → %s  N=%d dt=%.3f T=%.2f  발판FL(%.2f,%.2f,%.2f)\n",
              in, out, N, dt, T, st.p(0, 0), st.p(0, 1), st.p(0, 2));
  return 0;
}
