// test_terrain.cpp — TAMOLS 지형 처리 C++ vs Python(Drake) 대조
//   /tmp/terrain_ref.txt(Python process_height_maps 출력) 읽어 h_s1·h_s2·∇h 내부점 비교.
#include "terrain_proc.hpp"
#include <cstdio>
#include <fstream>

using namespace tamols;

static Grid read_grid(std::ifstream& f, int R, int C) {
  Grid M(R, C);
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) f >> M(i, j);
  return M;
}

int main() {
  std::ifstream f("/tmp/terrain_ref.txt");
  if (!f) { std::printf("★ /tmp/terrain_ref.txt 없음\n"); return 1; }
  int R, C; double res; f >> R >> C >> res;
  Grid h    = read_grid(f, R, C);
  Grid h_s1 = read_grid(f, R, C);   // Python 기준
  Grid h_s2 = read_grid(f, R, C);
  Grid gx   = read_grid(f, R, C);
  Grid gy   = read_grid(f, R, C);

  TerrainLayers L = process_height_maps(h, res, 1.0, 2.0);

  // 내부점(경계 2셀 제외)서 최대오차 — 경계 reflect 미세차 배제
  auto maxerr = [&](const Grid& A, const Grid& B) {
    double e = 0;
    for (int i = 2; i < R - 2; ++i) for (int j = 2; j < C - 2; ++j)
      e = std::max(e, std::fabs(A(i, j) - B(i, j)));
    return e;
  };
  double e1 = maxerr(L.h_s1, h_s1), e2 = maxerr(L.h_s2, h_s2);
  double ex = maxerr(L.gh_x, gx), ey = maxerr(L.gh_y, gy);

  std::printf("[TAMOLS terrain_proc vs Python(Drake)]  내부점 최대오차\n");
  std::printf("  h_s1 (gaussian σ1) = %.2e  %s\n", e1, e1 < 1e-5 ? "OK" : "★");
  std::printf("  h_s2 (virtual floor)= %.2e  %s\n", e2, e2 < 1e-5 ? "OK" : "★");
  std::printf("  ∇h_x               = %.2e  %s\n", ex, ex < 1e-5 ? "OK" : "★");
  std::printf("  ∇h_y               = %.2e  %s\n", ey, ey < 1e-5 ? "OK" : "★");
  bool ok = e1 < 1e-5 && e2 < 1e-5 && ex < 1e-5 && ey < 1e-5;
  std::printf("  → 지형 처리 %s\n", ok ? "Python 정합 ✓" : "불일치 — 확인필요");
  return ok ? 0 : 1;
}
