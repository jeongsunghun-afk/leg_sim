// terrain_proc.hpp — TAMOLS 지형 처리 (Drake map_processing.py C++ 포팅)
//   heightmap h → h_s1(gaussian) · h_s2(virtual floor) · ∇h/∇h_s1/∇h_s2(5점 FD)
//   TAMOLS 엣지회피=그래디언트, 발판정합=높이. SDF 불요(D1식 NMPC만 SDF 필요).
//   경계=reflect(scipy 기본). 격자 M(i,j): i=x, j=y. ∇x=axis1(j변화 아님, 원본 axis=1=열)…
//   ※ Drake와 동일 축규약 유지: grad_x=FD along axis=1, grad_y=FD along axis=0.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>

namespace tamols {

using Grid = Eigen::MatrixXd;   // (rows × cols) = h[i][j]

// reflect 인덱스(scipy 'reflect': d c b a | a b c d | d c b a)
inline int reflect_idx(int i, int n) {
  if (n == 1) return 0;
  while (i < 0 || i >= n) { if (i < 0) i = -i - 1; if (i >= n) i = 2 * n - i - 1; }
  return i;
}

// ── 5점 중앙차분 그래디언트 ([1,-8,0,8,-1]/(12·res)) ──
//   grad_x = axis=1(열 방향), grad_y = axis=0(행 방향). Drake compute_gradients 정합.
inline void compute_gradients(const Grid& h, double res, Grid& gx, Grid& gy) {
  const int R = (int)h.rows(), C = (int)h.cols();
  gx.setZero(R, C); gy.setZero(R, C);
  // scipy ndimage.convolve1d = convolution(커널 뒤집음). 반대칭 미분커널이라 correlation 대비 부호반전
  //   → Drake([1,-8,0,8,-1] via convolve1d) 정합 위해 뒤집은 [-1,8,0,-8,1] 사용.
  const double k[5] = {-1, 8, 0, -8, 1};
  const double sc = 1.0 / (12.0 * res);
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) {
      double sx = 0, sy = 0;
      for (int m = -2; m <= 2; ++m) {
        sx += k[m + 2] * h(i, reflect_idx(j + m, C));   // axis=1
        sy += k[m + 2] * h(reflect_idx(i + m, R), j);   // axis=0
      }
      gx(i, j) = sx * sc; gy(i, j) = sy * sc;
    }
}

// ── 분리형 gaussian filter (reflect) ──
inline Grid gaussian_filter(const Grid& h, double sigma) {
  const int R = (int)h.rows(), C = (int)h.cols();
  int rad = std::max(1, (int)(4.0 * sigma + 0.5));       // scipy gaussian_filter truncate=4.0 정합
  std::vector<double> ker(2 * rad + 1); double sum = 0;
  for (int t = -rad; t <= rad; ++t) { double v = std::exp(-0.5 * (t * t) / (sigma * sigma)); ker[t + rad] = v; sum += v; }
  for (double& v : ker) v /= sum;
  Grid tmp(R, C), out(R, C);
  for (int i = 0; i < R; ++i)                            // 열방향(axis=1)
    for (int j = 0; j < C; ++j) { double s = 0;
      for (int t = -rad; t <= rad; ++t) s += ker[t + rad] * h(i, reflect_idx(j + t, C));
      tmp(i, j) = s; }
  for (int i = 0; i < R; ++i)                            // 행방향(axis=0)
    for (int j = 0; j < C; ++j) { double s = 0;
      for (int t = -rad; t <= rad; ++t) s += ker[t + rad] * tmp(reflect_idx(i + t, R), j);
      out(i, j) = s; }
  return out;
}

// ── 3×3 median filter (reflect) ──
inline Grid median3(const Grid& h) {
  const int R = (int)h.rows(), C = (int)h.cols();
  Grid out(R, C); double w[9];
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j) { int n = 0;
      for (int di = -1; di <= 1; ++di) for (int dj = -1; dj <= 1; ++dj)
        w[n++] = h(reflect_idx(i + di, R), reflect_idx(j + dj, C));
      std::nth_element(w, w + 4, w + 9); out(i, j) = w[4]; }
  return out;
}

// ── TAMOLS 지형 처리: h_s1, h_s2(virtual floor), 그래디언트 ──
struct TerrainLayers {
  Grid h_s1, h_s2;
  Grid gh_x, gh_y, gs1_x, gs1_y, gs2_x, gs2_y;    // ∇h, ∇h_s1, ∇h_s2
};

inline TerrainLayers process_height_maps(const Grid& h_raw, double res,
                                         double sigma1 = 1.0, double sigma2 = 2.0) {
  TerrainLayers L;
  const int R = (int)h_raw.rows(), C = (int)h_raw.cols();
  // h_s1 = gaussian(h, σ1)
  L.h_s1 = gaussian_filter(h_raw, sigma1);
  // virtual floor: median → delta≠0 mask → dilate → 3×3 local max → gaussian(σ2)
  Grid hmed = median3(h_raw);
  Eigen::MatrixXi mask(R, C);                     // delta≠0 = edge/transition
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j)
    mask(i, j) = (std::fabs(h_raw(i, j) - hmed(i, j)) > 1e-12) ? 1 : 0;
  Eigen::MatrixXi dil(R, C); dil.setZero();       // 3×3 binary dilation
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) if (mask(i, j)) {
    for (int di = -1; di <= 1; ++di) for (int dj = -1; dj <= 1; ++dj) {
      int ii = i + di, jj = j + dj;
      if (ii >= 0 && ii < R && jj >= 0 && jj < C) dil(ii, jj) = 1; } }
  Grid h_dil = h_raw;                             // dilated 셀=3×3 local max(갭 위 virtual floor)
  for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) if (dil(i, j)) {
    double mx = -1e18;
    for (int di = -1; di <= 1; ++di) for (int dj = -1; dj <= 1; ++dj) {
      int ii = i + di, jj = j + dj;
      if (ii >= 0 && ii < R && jj >= 0 && jj < C) mx = std::max(mx, h_raw(ii, jj)); }
    h_dil(i, j) = mx; }
  L.h_s2 = gaussian_filter(h_dil, sigma2);
  // 그래디언트
  compute_gradients(h_raw, res, L.gh_x,  L.gh_y);
  compute_gradients(L.h_s1, res, L.gs1_x, L.gs1_y);
  compute_gradients(L.h_s2, res, L.gs2_x, L.gs2_y);
  return L;
}

} // namespace tamols
