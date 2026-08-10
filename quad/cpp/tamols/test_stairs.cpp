// test_stairs.cpp — TAMOLS가 계단 heightmap을 푸는지 feasibility 확인 (3D DTC de-risk)
//   계단(+x로 step_h씩 상승) heightmap 주입 → online_replan(z0_terrain) → 수렴·base 상승·발판 surface 확인.
#include "tamols_online.hpp"
#include <cstdio>
#include <cmath>
using namespace tamols;

int main() {
  int N = 41; double cell = 0.05, off = cell * N / 2.0;
  double step_h = 0.08, step_d = 0.35;   // 계단: 높이 8cm·깊이 35cm
  // heightmap: base 로컬 원점(x=0) 앞(+x)으로 계단 상승
  Grid h = Grid::Zero(N, N);
  for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) {
    double x = i * cell - off;
    h(i, j) = x > 0 ? step_h * std::floor(x / step_d) : 0.0;
  }
  int ms = N;
  Params prm;

  OnlineCfg cfg; cfg.vadv = 0.3; cfg.phase_dur = 0.2; cfg.z0_terrain = true; cfg.straddle_init = false;
  double z0 = 0.52, yaw0 = 0, vx0 = 0, vy0 = 0, base_x = 0;
  std::printf("[계단 TAMOLS: step_h=%.2f step_d=%.2f vadv=%.1f z0_terrain=on]\n", step_h, step_d, cfg.vadv);
  bool ok_all = true;
  TamolsState st;
  for (int rp = 0; rp < 8; ++rp) {
    // 발판=현재 base 앞 지형 surface에
    Eigen::Matrix<double,4,3> foot;
    for (int i = 0; i < 4; ++i) { foot(i,0)=prm.hip_offsets(i,0); foot(i,1)=prm.hip_offsets(i,1);
      foot(i,2)=bilinear_height(h, cell, ms, base_x+prm.hip_offsets(i,0), prm.hip_offsets(i,1)); }
    cfg.warm = rp > 0;
    QpResult r = online_replan(st, h, cell, ms, z0, yaw0, vx0, vy0, foot, cfg);
    double z_s = st.pos_at(0, 0)(2);
    double z_e = st.pos_at(st.num_phases()-1, st.gait.back().duration)(2);
    double x_e = st.pos_at(st.num_phases()-1, st.gait.back().duration)(0);
    bool ok = (r.eq_viol < 1e-2 && r.ineq_viol < 1e-2); ok_all &= ok;
    std::printf("  replan%d %s iters=%d cost=%.2f eq=%.1e ineq=%.1e | base z %.3f->%.3f(Δ%+.3f) x_end=%.2f | 발판z FL=%.3f FR=%.3f RL=%.3f RR=%.3f %s\n",
      rp, cfg.warm?"warm":"cold", r.iters, r.cost, r.eq_viol, r.ineq_viol, z_s, z_e, z_e-z_s, x_e,
      st.p(0,2), st.p(1,2), st.p(2,2), st.p(3,2), ok?"":"★viol");
    // 로봇 전진 가정: base_x 진행 → z0를 지형 따라 상승
    vx0 = std::min(cfg.vadv, vx0 + 0.1);
    base_x += 0.12;   // ~replan당 전진
    z0 = 0.52 + step_h * std::floor(std::max(0.0, base_x) / step_d);
  }
  std::printf("  → 계단 TAMOLS %s (수렴=%s). base z가 계단 따라 오르고 발판이 step surface(0.08배수)면 3D DTC 가능.\n",
              ok_all?"OK":"확인필요", ok_all?"전부 feasible":"일부 위반");
  return 0;
}
