// test_costs.cpp — TAMOLS 활성 비용 3종 C++ vs Drake(EvalBinding) 대조
//   /tmp/tamols_sol.txt(상태) + /tmp/tamols_cost.txt(cell·map·heightmap·기대비용) 읽어 재계산.
#include "costs.hpp"
#include <cstdio>
#include <fstream>

using namespace tamols;

int main() {
  TamolsState st;
  if (!load_solution("/tmp/tamols_sol.txt", st)) { std::printf("★ tamols_sol.txt 로드 실패\n"); return 1; }

  std::ifstream f("/tmp/tamols_cost.txt");
  if (!f) { std::printf("★ tamols_cost.txt 없음\n"); return 1; }
  double cell; int map_size; f >> cell >> map_size;
  Grid h(map_size, map_size);
  for (int i = 0; i < map_size; ++i) for (int j = 0; j < map_size; ++j) f >> h(i, j);
  double exp_track, exp_foot, exp_nom; f >> exp_track >> exp_foot >> exp_nom;

  double ct = tracking_cost(st);
  double cf = foothold_on_ground_cost(st, h, cell, map_size);
  double cn = nominal_kinematic_cost(st);

  auto rel = [](double a, double b) { double d = std::fabs(a - b); double s = std::max(1.0, std::fabs(b)); return d / s; };
  double et = rel(ct, exp_track), ef = rel(cf, exp_foot), en = rel(cn, exp_nom);

  std::printf("[TAMOLS 비용 vs Drake]  cell=%.3f map=%d\n", cell, map_size);
  std::printf("  tracking          C++=%.6f Drake=%.6f  rel=%.2e %s\n", ct, exp_track, et, et < 1e-5 ? "OK" : "★");
  std::printf("  foothold_on_ground C++=%.6f Drake=%.6f  rel=%.2e %s\n", cf, exp_foot, ef, ef < 1e-5 ? "OK" : "★");
  std::printf("  nominal_kinematic C++=%.6f Drake=%.6f  rel=%.2e %s\n", cn, exp_nom, en, en < 1e-5 ? "OK" : "★");
  bool ok = et < 1e-5 && ef < 1e-5 && en < 1e-5;
  std::printf("  → 비용 3종 %s\n", ok ? "Drake 정합 ✓" : "불일치 — 확인필요");
  return ok ? 0 : 1;
}
