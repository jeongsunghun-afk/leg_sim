// test_rti.cpp — 온라인 RTI(warm-start receding-horizon) 검증
//   매 리플랜: 초기조건 소폭 드리프트 → warm-start solve_fast(적은 iter) → 재수렴·타이밍 측정.
//   실제 온라인 TO(DTC): 이전 해를 warm-start로 매 제어스텝 재풀이 → 수렴 누적.
#include "tamols_jac.hpp"
#include <cstdio>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace tamols;

static bool load_cost_map(const char* p, Grid& h, double& cell, int& ms) {
  std::ifstream f(p); if (!f) return false; f >> cell >> ms; h.resize(ms, ms);
  for (int i = 0; i < ms; ++i) for (int j = 0; j < ms; ++j) f >> h(i, j); return true;
}

int main() {
  TamolsState st; Grid h; double cell; int ms;
  if (!load_solution("/tmp/tamols_sol.txt", st) || !load_cost_map("/tmp/tamols_cost.txt", h, cell, ms)) { std::printf("★ 로드 실패\n"); return 1; }
  QpOptions o; o.max_iter = 3;   // ★RTI: 리플랜당 3 iter만(warm-start 누적)

  std::printf("[온라인 RTI: warm-start receding-horizon · 리플랜당 max 3 iter]\n");
  double maxms = 0; bool all_rt = true, all_feas = true;
  for (int rp = 0; rp < 12; ++rp) {
    // 초기조건 소폭 드리프트(로봇 실제상태가 플랜서 조금 벗어남): base z ±3mm·앞발 x ±2mm
    double dz = 0.003 * std::sin(0.7 * rp), dfx = -0.002 * std::cos(0.5 * rp);
    st.base_pose(2) += dz;                                   // 초기 base z 갱신 → 초기제약 목표 변화
    st.p_meas(0, 0) += dfx; st.p_meas(1, 0) += dfx;          // 측정 발위치 변화
    auto t0 = std::chrono::high_resolution_clock::now();
    QpResult r = solve_fast(st, h, cell, ms, o);             // st(이전 해)=warm-start, in-place 갱신
    auto t1 = std::chrono::high_resolution_clock::now();
    double m = std::chrono::duration<double, std::milli>(t1 - t0).count();
    maxms = std::max(maxms, m); if (m >= 20.0) all_rt = false; if (r.ineq_viol > 1e-3 || r.eq_viol > 1e-3) all_feas = false;
    std::printf("  리플랜%2d  drift(z%+.0fmm,fx%+.0fmm)  iters=%d  cost=%.4f  eq=%.1e ineq=%.1e  → %.1f ms %s\n",
                rp, dz * 1000, dfx * 1000, r.iters, r.cost, r.eq_viol, r.ineq_viol, m, m < 20 ? "✓" : "★>20ms");
  }
  std::printf("  → 온라인 RTI %s (최대 %.1f ms, %s)\n",
              (all_rt && all_feas) ? "성공: 실시간·feasible 유지 ✓" : "확인필요",
              maxms, all_rt ? "전부 <20ms" : "일부 초과");
  return (all_rt && all_feas) ? 0 : 1;
}
