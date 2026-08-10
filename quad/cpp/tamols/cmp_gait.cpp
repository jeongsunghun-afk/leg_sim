// cmp_gait.cpp — walk vs trot 계단 TAMOLS 레퍼런스 품질 비교 (DTC 캐시 정식화 선택; §P2.8 검증)
//   같은 계단 시나리오를 3정식화로 풀어 "DTC 레퍼런스로 뭐가 나은지" 정량 비교:
//     trot(현 캐시 기본) / walk(버그 그대로) / walk+COM_W+WALK_QLEG(§P2.8 권장)
//   지표: feasibility · 발판 tread정확도(낮을수록 정밀) · base 횡드리프트(낮을수록 안정) · base z상승 · cost
#include "tamols_online.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace tamols;

static void run(const char* name, bool walk, bool qleg, bool com, double step_h) {
  if (qleg) setenv("WALK_QLEG", "1", 1); else unsetenv("WALK_QLEG");
  if (com)  setenv("COM_W", "400", 1);   else unsetenv("COM_W");

  int N = 41; double cell = 0.05, off = cell * N / 2.0; double step_d = 0.35;
  Grid h = Grid::Zero(N, N);
  for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
    double x = i * cell - off; h(i, j) = x > 0 ? step_h * std::floor(x / step_d) : 0.0; }
  int ms = N; Params prm;
  OnlineCfg cfg; cfg.vadv = 0.3; cfg.phase_dur = 0.2; cfg.z0_terrain = true;
  cfg.straddle_init = false; cfg.walk = walk;
  double z0 = 0.52, base_x = 0, vx0 = 0;
  TamolsState st;
  int nrep = 8, feas = 0, nfeet = 0;
  double sc = 0, si = 0, symax = 0, sferr = 0, szrise = 0;
  for (int rp = 0; rp < nrep; ++rp) {
    Eigen::Matrix<double,4,3> foot;
    for (int i = 0; i < 4; ++i) { foot(i,0)=prm.hip_offsets(i,0); foot(i,1)=prm.hip_offsets(i,1);
      foot(i,2)=bilinear_height(h, cell, ms, base_x+prm.hip_offsets(i,0), prm.hip_offsets(i,1)); }
    cfg.warm = rp > 0;
    QpResult r = online_replan(st, h, cell, ms, z0, 0, vx0, 0, foot, cfg);
    if (r.eq_viol < 1e-2 && r.ineq_viol < 1e-2) feas++;
    sc += r.cost; si += r.iters;
    for (int k = 0; k < st.num_phases(); ++k) { double tau = st.gait[k].duration * 0.5;
      auto p = st.pos_at(k, tau); double y = std::abs(p(1)); if (y > symax) symax = y; }
    double zs = st.pos_at(0, 0)(2);
    double ze = st.pos_at(st.num_phases()-1, st.gait.back().duration)(2);
    szrise += (ze - zs);
    for (int i = 0; i < 4; ++i) { double fs = bilinear_height(h, cell, ms, st.p(i,0), st.p(i,1));
      sferr += std::abs(st.p(i,2) - fs); nfeet++; }
    vx0 = std::min(cfg.vadv, vx0 + 0.1); base_x += 0.12;
    z0 = 0.52 + step_h * std::floor(std::max(0.0, base_x) / step_d);
  }
  std::printf("  [%-14s] feas=%3.0f%%  cost=%7.2f  iters=%4.1f  | max|y_drift|=%.3fm  foot-on-tread_err=%.4fm  z_rise/rep=%+.3fm\n",
    name, 100.0*feas/nrep, sc/nrep, si/nrep, symax, sferr/nfeet, szrise/nrep);
}

int main() {
  std::printf("== walk vs trot 계단 TAMOLS 레퍼런스 품질 비교 (8 replan chain, vadv 0.3) ==\n");
  std::printf("   지표: feas 높을수록·cost 낮을수록 / foot-on-tread_err 낮을수록 발판 정밀 / max|y_drift| 낮을수록 CoM 안정\n");
  for (double sh : {0.05, 0.067, 0.08}) {
    std::printf("\n----- 계단 step_h=%.3fm (level≈%.0f) -----\n", sh, sh/0.15*9);
    run("trot(현캐시)",   false, false, false, sh);
    run("walk(plain)",    true,  false, false, sh);
    run("walk+QLEG",      true,  true,  false, sh);
    run("walk+QLEG+COM",  true,  true,  true,  sh);
  }
  return 0;
}
