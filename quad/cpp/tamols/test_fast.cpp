// test_fast.cpp — 해석 Jacobian 솔버(solve_fast) vs FD 솔버(solve) 수렴·타이밍
#include "tamols_jac.hpp"
#include <cstdio>
#include <fstream>
#include <chrono>

using namespace tamols;

static bool load_cost_map(const char* p, Grid& h, double& cell, int& ms) {
  std::ifstream f(p); if (!f) return false; f >> cell >> ms; h.resize(ms, ms);
  for (int i = 0; i < ms; ++i) for (int j = 0; j < ms; ++j) f >> h(i, j); return true;
}

int main() {
  TamolsState st0; Grid h; double cell; int ms;
  if (!load_solution("/tmp/tamols_sol.txt", st0) || !load_cost_map("/tmp/tamols_cost.txt", h, cell, ms)) { std::printf("★ 로드 실패\n"); return 1; }
  QpOptions o;
  auto perturb = [&](TamolsState s){ s.p(0,0)-=0.01; s.p(1,0)-=0.01; for(int k=0;k<s.num_phases();++k) s.a[k](2,0)+=0.02; return s; };

  // FD 솔버
  { TamolsState st = perturb(st0);
    auto t0=std::chrono::high_resolution_clock::now(); QpResult r=solve(st,h,cell,ms,o);
    auto t1=std::chrono::high_resolution_clock::now(); double m=std::chrono::duration<double,std::milli>(t1-t0).count();
    std::printf("[FD 솔버]      ok=%d iters=%d cost=%.5f ineq위반=%.2e → %.1f ms (%.2f ms/iter)\n",
                r.ok,r.iters,r.cost,r.ineq_viol,m,m/std::max(1,r.iters)); }
  // 해석 솔버
  { TamolsState st = perturb(st0);
    auto t0=std::chrono::high_resolution_clock::now(); QpResult r=solve_fast(st,h,cell,ms,o);
    auto t1=std::chrono::high_resolution_clock::now(); double m=std::chrono::duration<double,std::milli>(t1-t0).count();
    std::printf("[해석 솔버]    ok=%d iters=%d cost=%.5f ineq위반=%.2e → %.1f ms (%.2f ms/iter)\n",
                r.ok,r.iters,r.cost,r.ineq_viol,m,m/std::max(1,r.iters)); }
  // 해석 솔버 5-iter(RTI)
  { QpOptions o5=o; o5.max_iter=5; TamolsState st = perturb(st0);
    auto t0=std::chrono::high_resolution_clock::now(); QpResult r=solve_fast(st,h,cell,ms,o5);
    auto t1=std::chrono::high_resolution_clock::now(); double m=std::chrono::duration<double,std::milli>(t1-t0).count();
    std::printf("[해석 5-iter RTI] iters=%d cost=%.5f ineq위반=%.2e → %.1f ms  %s\n",
                r.iters,r.cost,r.ineq_viol,m, m<20.0?"★<20ms 실시간!":"20ms 초과"); }
  return 0;
}
