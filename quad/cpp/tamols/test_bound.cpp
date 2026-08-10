// test_bound.cpp — bound 게이트 검증: online_replan(cfg.bound) 후 per-phase contact 스케줄을 찍어
//   앞쌍(FL,FR=idx0,1)·뒷쌍(RL,RR=idx2,3)이 교대로 swing 하는지 확인 + feasibility.
//   플래너 라벨 [FL,FR,RL,RR]=idx[0,1,2,3] (hip_offsets·straddle-init과 동일).
#include "tamols_online.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
using namespace tamols;

static const char* LEG[4] = {"FL", "FR", "RL", "RR"};  // 플래너 index 라벨

static void dump_schedule(const char* name, TamolsState& st) {
  std::printf("  [%s] %d phases  (C=contact, .=swing)\n", name, st.num_phases());
  std::printf("        phase:");
  for (int k = 0; k < st.num_phases(); ++k) std::printf("  P%d", k);
  std::printf("\n");
  for (int i = 0; i < 4; ++i) {
    std::printf("     %s(idx%d):", LEG[i], i);
    for (int k = 0; k < st.num_phases(); ++k)
      std::printf("   %c", st.gait[k].contact[i] ? 'C' : '.');
    std::printf("\n");
  }
  // per-phase swing 그룹 요약
  for (int k = 0; k < st.num_phases(); ++k) {
    int sw = 0; char grp[32] = "";
    for (int i = 0; i < 4; ++i) if (!st.gait[k].contact[i]) { std::strcat(grp, LEG[i]); std::strcat(grp, " "); sw++; }
    if (sw) std::printf("        P%d swing = %s\n", k, grp);
  }
}

static void run(const char* name, bool bound, bool walk) {
  int N = 41; double cell; int ms; Grid h;
  flat_costmap(h, cell, ms, N, 0.05);
  Params prm;
  OnlineCfg cfg; cfg.vadv = 0.3; cfg.phase_dur = 0.2; cfg.rti_iter = 60;   // cold solve: 넉넉한 iter로 수렴 확인(test_stepping 첫solve와 동일)
  cfg.bound = bound; cfg.walk = walk; cfg.straddle_init = false;
  double z0 = 0.52, vx0 = 0; TamolsState st;
  Eigen::Matrix<double,4,3> foot;
  for (int i = 0; i < 4; ++i) { foot(i,0)=prm.hip_offsets(i,0); foot(i,1)=prm.hip_offsets(i,1); foot(i,2)=0; }
  QpResult r = online_replan(st, h, cell, ms, z0, 0, vx0, 0, foot, cfg);
  std::printf("\n----- %s -----\n", name);
  dump_schedule(name, st);
  bool feas = (r.eq_viol < 1e-2 && r.ineq_viol < 1e-2);
  std::printf("        solve: cost=%.3f iters=%d eq_viol=%.2e ineq_viol=%.2e  feasible=%s\n",
              r.cost, r.iters, r.eq_viol, r.ineq_viol, feas ? "YES" : "NO");
}

int main() {
  std::printf("== bound 게이트 검증 (flat, vadv 0.3, 5-phase) ==\n");
  std::printf("   기대: bound=앞쌍(FL,FR) swing → 뒷쌍(RL,RR) swing 교대. trot=대각쌍(FL,RR)/(FR,RL).\n");
  run("trot",  false, false);
  run("bound", true,  false);

  // 자동 판정: bound에서 앞쌍이 한 phase에 함께 swing하고 뒷쌍이 다른 phase에 함께 swing하는지.
  {
    int N = 41; double cell; int ms; Grid h; flat_costmap(h, cell, ms, N, 0.05);
    Params prm; OnlineCfg cfg; cfg.vadv=0.3; cfg.phase_dur=0.2; cfg.bound=true; cfg.straddle_init=false;
    TamolsState st; Eigen::Matrix<double,4,3> foot;
    for (int i=0;i<4;++i){ foot(i,0)=prm.hip_offsets(i,0); foot(i,1)=prm.hip_offsets(i,1); foot(i,2)=0; }
    online_replan(st, h, cell, ms, 0.52, 0, 0, 0, foot, cfg);
    bool front_together = false, rear_together = false, ok = true;
    for (int k = 0; k < st.num_phases(); ++k) {
      bool f0 = !st.gait[k].contact[0], f1 = !st.gait[k].contact[1];   // FL,FR swing?
      bool r2 = !st.gait[k].contact[2], r3 = !st.gait[k].contact[3];   // RL,RR swing?
      if (f0 || f1 || r2 || r3) {                                       // some foot swinging this phase
        if (f0 && f1 && !r2 && !r3) front_together = true;              // exactly the front pair
        else if (r2 && r3 && !f0 && !f1) rear_together = true;          // exactly the rear pair
        else ok = false;                                               // mixed group => not a clean bound
      }
    }
    std::printf("\n== 자동 판정 ==\n");
    std::printf("   앞쌍(FL,FR) 동시 swing phase 존재: %s\n", front_together ? "YES" : "NO");
    std::printf("   뒷쌍(RL,RR) 동시 swing phase 존재: %s\n", rear_together  ? "YES" : "NO");
    std::printf("   혼합(대각/단발) swing 없음: %s\n", ok ? "YES" : "NO");
    std::printf("   >>> BOUND 스케줄 %s\n", (front_together && rear_together && ok) ? "PASS" : "FAIL");
  }
  return 0;
}
