// tamols_solve.cpp — C++ solve_fast로 갭 크로싱 플랜을 직접 풀기(Drake 없이 standalone 검증)
//   나이브 초기추정(정지 스플라인·gap-avoid 발판·트롯 게이트)에서 solve_fast → /tmp/tamols_sol.txt 기록.
//   성공 시 export_traj + trot_sim 재생으로 크로싱 검증 가능(= C++ 솔버 end-to-end).
#include "tamols_jac.hpp"
#include <cstdio>
#include <fstream>
#include <chrono>

using namespace tamols;

static bool load_cost_map(const char* p, Grid& h, double& cell, int& ms) {
  std::ifstream f(p); if (!f) return false; f >> cell >> ms; h.resize(ms, ms);
  for (int i = 0; i < ms; ++i) for (int j = 0; j < ms; ++j) f >> h(i, j); return true;
}

// 나이브 초기추정: 트롯 게이트·정지 스플라인·gap-avoid 발판
TamolsState make_init() {
  TamolsState st; int P = 5;
  st.gait.resize(P);
  double dur = 0.4;
  int cs[5][4] = {{1,1,1,1},{0,1,1,0},{1,1,1,1},{1,0,0,1},{1,1,1,1}};
  int ad[5][4] = {{0,0,0,0},{0,0,0,0},{1,0,0,1},{1,0,0,1},{1,1,1,1}};
  for (int k = 0; k < P; ++k) { st.gait[k].duration = dur; for (int i = 0; i < 4; ++i) { st.gait[k].contact[i] = cs[k][i]; st.gait[k].at_des[i] = ad[k][i]; } }
  st.base_pose << 0, 0, 0.52, 0, 0, 0; st.base_vel.setZero();
  st.p_meas << 0.225, 0.14, 0,  0.225, -0.14, 0,  -0.225, 0.14, 0,  -0.225, -0.14, 0;   // nominal stance
  st.a.assign(P, MatrixXd::Zero(6, 4));
  // ★동역학 일관 nominal init: 정지서 출발→smootherstep 전진(초기속도0·위상연속=등식 자동충족)
  double T = P * dur, Xf = 0.5;                              // 총 2.0s, 전진 0.5m
  auto xg = [&](double t){ double s = t / T; return Xf * (10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s); };
  auto vg = [&](double t){ double s = t / T; return Xf * (30*s*s - 60*s*s*s + 30*s*s*s*s) / T; };   // v(0)=v(T)=0
  for (int k = 0; k < P; ++k) {
    st.a[k].col(0) = st.base_pose; st.a[k](2, 0) = 0.52;      // 상수 DOF(y,z,rpy)=정지·연속
    double t0 = k * dur, x0 = xg(t0), x1 = xg(t0 + dur), v0 = vg(t0), v1 = vg(t0 + dur);  // Hermite 큐빅(x-DOF)
    st.a[k](0, 0) = x0; st.a[k](0, 1) = v0;
    st.a[k](0, 2) = (3*(x1 - x0)/dur - 2*v0 - v1) / dur;
    st.a[k](0, 3) = (2*(x0 - x1)/dur + v0 + v1) / (dur*dur);
  }
  st.p << 0.75, 0.14, 0,  0.75, -0.14, 0,  0.40, 0.14, 0,  0.40, -0.14, 0;                // gap-avoid 발판(앞 넘어·뒤 앞)
  st.epsilon = VectorXd::Zero(P);
  return st;
}

int main() {
  Grid h; double cell; int ms;
  if (!load_cost_map("/tmp/tamols_cost.txt", h, cell, ms)) { std::printf("★ cost map 없음(먼저 Drake export로 heightmap 생성)\n"); return 1; }
  TamolsState st = make_init();
  QpOptions o; o.max_iter = 200;   // cold-start=많은 iter 허용
  o.rp_max = 0.06; o.zlo = 0.48; o.zhi = 0.55; o.yaw_max = 0.08;   // ★자세 타이트닝(추종성=Drake 해 성공조건과 동일)
  o.x_target = 0.68;                                               // ★덜 공격적 전진(발판 과전방 방지, Drake≈0.68)
  auto t0 = std::chrono::high_resolution_clock::now();
  QpResult r = solve_fast(st, h, cell, ms, o);
  auto t1 = std::chrono::high_resolution_clock::now();
  double m = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("[C++ solve_fast cold-start]  ok=%d iters=%d cost=%.5f eq위반=%.2e ineq위반=%.2e → %.0f ms\n",
              r.ok, r.iters, r.cost, r.eq_viol, r.ineq_viol, m);
  std::printf("  base x: 시작 %.3f → 끝(마지막 phase 끝) ", st.pos_at(0, 0)(0));
  { int k = st.num_phases() - 1; std::printf("%.3f\n", st.pos_at(k, st.gait[k].duration)(0)); }
  std::printf("  발판: FL(%.2f,%.2f) FR(%.2f,%.2f) RL(%.2f,%.2f) RR(%.2f,%.2f)\n",
              st.p(0,0),st.p(0,1),st.p(1,0),st.p(1,1),st.p(2,0),st.p(2,1),st.p(3,0),st.p(3,1));

  // 성공 시 sol 기록(export_traj·trot_sim 재생용)
  if (r.eq_viol < 1e-4 && r.ineq_viol < 1e-3) {
    int P = st.num_phases();
    std::ofstream f("/tmp/tamols_sol_cpp.txt");
    f << P << " 6 4 4\n";
    for (int k = 0; k < P; ++k) f << st.gait[k].duration << (k+1<P?" ":"\n");
    for (int k = 0; k < P; ++k) { for (int i=0;i<4;++i) f<<st.gait[k].contact[i]<<(i<3?" ":"\n"); }
    for (int k = 0; k < P; ++k) { for (int i=0;i<4;++i) f<<st.gait[k].at_des[i]<<(i<3?" ":"\n"); }
    for (int d=0;d<6;++d) f<<st.base_pose(d)<<(d<5?" ":"\n");
    for (int d=0;d<6;++d) f<<st.base_vel(d)<<(d<5?" ":"\n");
    for (int L=0;L<4;++L){ for(int c=0;c<3;++c) f<<st.p_meas(L,c)<<(c<2?" ":"\n"); }
    for (int k=0;k<P;++k) for(int d=0;d<6;++d){ for(int i=0;i<4;++i) f<<st.a[k](d,i)<<(i<3?" ":"\n"); }
    for (int L=0;L<4;++L){ for(int c=0;c<3;++c) f<<st.p(L,c)<<(c<2?" ":"\n"); }
    for (int L=0;L<4;++L){ for(int c=0;c<3;++c) f<<st.prm.hip_offsets(L,c)<<(c<2?" ":"\n"); }
    for (int k=0;k<P;++k) f<<st.epsilon(k)<<(k+1<P?" ":"\n");
    for (int c=0;c<3;++c) f<<st.prm.inertia_diag(c)<<(c<2?" ":"\n");
    std::printf("  → /tmp/tamols_sol_cpp.txt 기록(C++ 자체 해) · export_traj로 재생 가능\n");
  } else std::printf("  ★ 미수렴 — 초기추정 개선/iter↑ 필요\n");
  return 0;
}
