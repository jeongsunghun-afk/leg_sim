// test_online.cpp — 온라인 replan(receding-horizon) 검증: 현재상태서 재계획·warm-start·타이밍
#include "tamols_online.hpp"
#include <cstdio>
#include <chrono>
using namespace tamols;

int main() {
  Grid h; double cell; int ms; flat_costmap(h, cell, ms);
  TamolsState st;
  // 명목 stance 발위치(로컬, base 기준)
  Eigen::Matrix<double,4,3> foot; Params prm;
  for (int i = 0; i < 4; ++i) { foot(i,0)=prm.hip_offsets(i,0); foot(i,1)=prm.hip_offsets(i,1); foot(i,2)=0; }
  OnlineCfg cfg; cfg.vadv = 0.4; cfg.phase_dur = 0.2;

  std::printf("[온라인 replan: 평지 전진 walk · receding-horizon]\n");
  double z0=0.52, yaw0=0, vx0=0, vy0=0, maxms=0; bool ok_all=true, rt_all=true;
  for (int rp = 0; rp < 10; ++rp) {
    cfg.warm = (rp > 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    QpResult r = online_replan(st, h, cell, ms, z0, yaw0, vx0, vy0, foot, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    double m = std::chrono::duration<double,std::milli>(t1-t0).count(); maxms=std::max(maxms,m);
    if (m>=20) rt_all=false; if (r.eq_viol>1e-3||r.ineq_viol>1e-3) ok_all=false;
    double xend = st.pos_at(st.num_phases()-1, st.gait.back().duration)(0);
    std::printf("  replan%2d %s iters=%d cost=%.3f eq=%.1e ineq=%.1e xend=%.3f → %.1f ms %s\n",
                rp, cfg.warm?"warm":"cold", r.iters, r.cost, r.eq_viol, r.ineq_viol, xend, m, m<20?"✓":"★>20ms");
    // 다음 스텝: 로봇이 vadv로 0.2s 전진했다 가정(상태 진행) → 재앵커(로컬이라 x는 리셋, 속도만 유지)
    vx0 = std::min(cfg.vadv, vx0 + 0.1);   // 가속 램프
  }
  std::printf("  → 온라인 replan %s (최대 %.1f ms, %s)\n",
              (ok_all&&rt_all)?"성공: feasible·실시간 유지 ✓":"확인필요", maxms, rt_all?"전부 <20ms":"일부 초과");
  return (ok_all&&rt_all)?0:1;
}
