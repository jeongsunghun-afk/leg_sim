// costs.hpp — TAMOLS 비용 (Drake costs.py 활성 3종 C++ 포팅)
//   활성(setup_costs_and_constraints): tracking · foothold_on_ground · nominal_kinematic.
//   (base_pose_alignment·edge_avoidance·previous_solution·smoothness 는 Drake서 주석처리=비활성.)
//   검증: Drake 해에서 각 비용 EvalBinding 값과 C++ 재계산 대조(≈0 오차).
#pragma once
#include "constraints.hpp"   // R_B, TamolsState
#include <cmath>

namespace tamols {

#ifndef TAMOLS_GRID_ALIAS
#define TAMOLS_GRID_ALIAS
using Grid = Eigen::MatrixXd;   // terrain_proc.hpp와 동일 별칭(중복 include 안전)
#endif

// ── 양선형 높이 보간 (Drake evaluate_height_at_symbolic_xy 정합) ──
//   i=(x+off)/cell, j=(y+off)/cell, off=cell·map_size/2. Σ h[k,l](1−|i−k|)(1−|j−l|), |i−k|<1 & |j−l|<1.
//   grid[k,l]: k=x축 인덱스(행), l=y축 인덱스(열).
inline double bilinear_height(const Grid& h, double cell, int map_size, double x, double y) {
  double off = cell * map_size / 2.0;
  double i = (x + off) / cell, j = (y + off) / cell;
  const int m = (int)h.rows(), n = (int)h.cols();
  double tot = 0;
  int k0 = (int)std::floor(i), l0 = (int)std::floor(j);
  for (int k = k0; k <= k0 + 1; ++k) {
    if (k < 0 || k >= m) continue;
    double wx = 1.0 - std::fabs(i - k); if (wx <= 0) continue;
    for (int l = l0; l <= l0 + 1; ++l) {
      if (l < 0 || l >= n) continue;
      double wy = 1.0 - std::fabs(j - l); if (wy <= 0) continue;
      tot += h(k, l) * wx * wy;
    }
  }
  return tot;
}

// ── tracking 비용 (Drake add_tracking_cost): x속도만 추종 ──
//   Σ_phase Σ_{s=0}^{S-1} (2 T_k/S)(vel_x(τ_s) − ref_vel_x)²,  τ_s = linspace(0,T_k,S+1)[:S]
inline double tracking_cost(const TamolsState& st) {
  const int S = 6;                       // tau_sampling_rate
  double c = 0;
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int s = 0; s < S; ++s) {
      double tau = Tk * (double)s / (double)S;
      double vx = st.vel_at(k, tau)(0);
      double w = 2.0 * Tk / (double)S;
      double d = vx - st.ref_vel(0);
      c += w * d * d;
    }
  }
  return c;
}

// ── foothold_on_ground 비용 (Drake add_foothold_on_ground_cost) ──
//   Σ_i 100 (h(p_i.x,p_i.y) − p_i.z)²
inline double foothold_on_ground_cost(const TamolsState& st, const Grid& h, double cell, int map_size) {
  double c = 0;
  for (int i = 0; i < 4; ++i) {
    double hpi = bilinear_height(h, cell, map_size, st.p(i, 0), st.p(i, 1));
    double d = hpi - st.p(i, 2);
    c += 100.0 * d * d;
  }
  return c;
}

// ── nominal_kinematic 비용 (Drake add_nominal_kinematic_cost) ──
//   마지막 phase, τ=T/2. Σ_i 20 |p_B + R_B·hip_i − l_des − p_i|²,  l_des=[0,0,h_des], p_i=at_des?p:p_meas
inline double nominal_kinematic_cost(const TamolsState& st) {
  int k = st.num_phases() - 1;
  double Tk = st.gait[k].duration, tau = Tk / 2.0;
  Vector6d pose = st.pos_at(k, tau);
  Vector3d pB = pose.head<3>();
  Eigen::Matrix3d R = R_B(pose.tail<3>());
  Vector3d l_des(0, 0, st.prm.h_des);
  double c = 0;
  for (int i = 0; i < 4; ++i) {
    Vector3d bml = pB + R * st.prm.hip_offsets.row(i).transpose() - l_des;
    Vector3d pi = st.gait[k].at_des[i] ? Vector3d(st.p.row(i).transpose())
                                       : Vector3d(st.p_meas.row(i).transpose());
    Vector3d e = bml - pi;
    c += 20.0 * e.dot(e);
  }
  return c;
}

} // namespace tamols
