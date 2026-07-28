// test_qp.cpp — TAMOLS SQP 솔버 검증
//   ① Drake 해(feasible·optimal)에서 시작 → 거의 안 움직여야(수렴·저비용·위반0) 정식화 정합 확인.
//   ② 발판·스플라인 섭동 후 시작 → feasible 저비용으로 복귀하는지(솔버 실동작) 확인.
#include "tamols_qp.hpp"
#include <cstdio>
#include <fstream>
#include <chrono>

using namespace tamols;

static bool load_cost_map(const char* path, Grid& h, double& cell, int& map_size) {
  std::ifstream f(path); if (!f) return false;
  f >> cell >> map_size; h.resize(map_size, map_size);
  for (int i = 0; i < map_size; ++i) for (int j = 0; j < map_size; ++j) f >> h(i, j);
  return true;
}

int main() {
  TamolsState st0;
  if (!load_solution("/tmp/tamols_sol.txt", st0)) { std::printf("★ sol 로드 실패\n"); return 1; }
  Grid h; double cell; int map_size;
  if (!load_cost_map("/tmp/tamols_cost.txt", h, cell, map_size)) { std::printf("★ cost map 로드 실패\n"); return 1; }
  QpOptions o;  // Drake와 동일 gap [0.45,0.65], x_target 0.73

  // 초기(Drake 해)에서의 비용·위반
  VectorXd R0 = cost_residuals(st0, h, cell, map_size), E0 = eq_constraints(st0), G0 = ineq_constraints(st0, o);
  double iv0 = 0; for (int i = 0; i < G0.size(); ++i) iv0 = std::max(iv0, std::max(0.0, -G0(i)));
  std::printf("[Drake 해 상태]  cost=%.6f  eq위반=%.2e  ineq위반=%.2e  (nz=%d neq=%d nineq=%d)\n",
              R0.squaredNorm(), E0.cwiseAbs().maxCoeff(), iv0,
              Packer(st0.num_phases()).nz, (int)E0.size(), (int)G0.size());

  // ① Drake 해에서 SQP 시작
  TamolsState st = st0;
  QpResult r = solve(st, h, cell, map_size, o);
  std::printf("[① Drake해서 시작]  ok=%d iters=%d  cost=%.6f  eq위반=%.2e  ineq위반=%.2e\n",
              r.ok, r.iters, r.cost, r.eq_viol, r.ineq_viol);

  // ② 완만한 섭동(warm-start 실사용 규모): 앞발 x −1cm(갭 엣지 안 넘음, 솔리드 지면 유지) + base z 스플라인 +2cm
  TamolsState st2 = st0;
  st2.p(0, 0) -= 0.01; st2.p(1, 0) -= 0.01;
  for (int k = 0; k < st2.num_phases(); ++k) st2.a[k](2, 0) += 0.02;
  VectorXd Rp = cost_residuals(st2, h, cell, map_size), Gp = ineq_constraints(st2, o);
  double ivp = 0; for (int i = 0; i < Gp.size(); ++i) ivp = std::max(ivp, std::max(0.0, -Gp(i)));
  std::printf("[② 섭동 초기]  cost=%.6f  ineq위반=%.2e\n", Rp.squaredNorm(), ivp);
  auto t0 = std::chrono::high_resolution_clock::now();
  QpResult r2 = solve(st2, h, cell, map_size, o);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("[② 섭동서 시작]  ok=%d iters=%d  cost=%.6f  eq위반=%.2e  ineq위반=%.2e\n",
              r2.ok, r2.iters, r2.cost, r2.eq_viol, r2.ineq_viol);
  std::printf("[타이밍]  총 %.1f ms  (%d iters → %.2f ms/iter · FD Jacobian nz=%d)\n",
              ms, r2.iters, ms / std::max(1, r2.iters), Packer(st2.num_phases()).nz);

  // ③ warm-start 실측: 5 iters 상한(실시간 SQP-RTI 모사) — 섭동서 몇 iter로 feasible 되나
  { QpOptions o3 = o; o3.max_iter = 5; TamolsState st3 = st0;
    st3.p(0,0) -= 0.01; st3.p(1,0) -= 0.01; for (int k=0;k<st3.num_phases();++k) st3.a[k](2,0) += 0.02;
    auto ta = std::chrono::high_resolution_clock::now();
    QpResult r3 = solve(st3, h, cell, map_size, o3);
    auto tb = std::chrono::high_resolution_clock::now();
    double ms3 = std::chrono::duration<double,std::milli>(tb-ta).count();
    std::printf("[③ 5-iter 상한(RTI)]  iters=%d  cost=%.5f  ineq위반=%.2e  → %.1f ms\n",
                r3.iters, r3.cost, r3.ineq_viol, ms3);
  }

  bool pass = r.eq_viol < 1e-5 && r.ineq_viol < 1e-5 && r2.eq_viol < 1e-4 && r2.ineq_viol < 1e-4;
  std::printf("  → SQP 솔버 %s\n", pass ? "동작 ✓ (feasible 수렴)" : "확인필요");
  return pass ? 0 : 1;
}
