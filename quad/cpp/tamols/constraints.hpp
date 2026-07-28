// constraints.hpp — TAMOLS 제약 (Drake constraints.py C++ 포팅) + 해 로더
//   검증: Drake feasible 해를 로드해 각 제약 residual≈0(feasibility) 확인.
//   현재: 로더 + 초기(add_initial)·위상연속(junction) [선형]. 다음: kinematic·friction·GIAC.
#pragma once
#include "tamols.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>

namespace tamols {

// ── Drake export(/tmp/tamols_sol.txt) 로드 → TamolsState ──
inline bool load_solution(const std::string& path, TamolsState& st) {
  std::ifstream f(path);
  if (!f) return false;
  int P, bd, ord, nl; f >> P >> bd >> ord >> nl;
  st.prm.base_dims = bd; st.prm.spline_order = ord; st.prm.num_legs = nl;
  st.gait.assign(P, GaitPhase{});
  for (int k = 0; k < P; ++k) f >> st.gait[k].duration;
  for (int k = 0; k < P; ++k) for (int i = 0; i < 4; ++i) f >> st.gait[k].contact[i];
  for (int k = 0; k < P; ++k) for (int i = 0; i < 4; ++i) f >> st.gait[k].at_des[i];   // at_des_position
  for (int d = 0; d < 6; ++d) f >> st.base_pose(d);
  for (int d = 0; d < 6; ++d) f >> st.base_vel(d);
  for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) f >> st.p_meas(L, c);
  st.a.assign(P, MatrixXd(6, ord));
  for (int k = 0; k < P; ++k) for (int d = 0; d < 6; ++d) for (int i = 0; i < ord; ++i) f >> st.a[k](d, i);
  for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) f >> st.p(L, c);
  for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) f >> st.prm.hip_offsets(L, c);
  st.epsilon.resize(P); for (int k = 0; k < P; ++k) f >> st.epsilon(k);         // GIAC slack
  for (int c = 0; c < 3; ++c) f >> st.prm.inertia_diag(c);                       // 관성 diag
  return (bool)f;
}

// ── 초기 제약 (Drake add_initial_constraints 앞부분): a0(d,0)=base_pose(d), a0(d,1)=base_vel(d) ──
inline double initial_residual(const TamolsState& st) {
  double e = 0;
  for (int d = 0; d < 6; ++d) {
    e = std::max(e, std::fabs(st.a[0](d, 0) - st.base_pose(d)));   // pos(0)=base_pose
    e = std::max(e, std::fabs(st.a[0](d, 1) - st.base_vel(d)));    // vel(0)=base_vel
  }
  return e;
}

// ── 위상 연속 제약 (junction): pos_k(Tk)=pos_{k+1}(0), vel_k(Tk)=vel_{k+1}(0) ──
inline double junction_residual(const TamolsState& st) {
  double e = 0;
  for (int k = 0; k + 1 < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    Vector6d pos_end = st.pos_at(k, Tk);   // 현 위상 끝
    Vector6d vel_end = st.vel_at(k, Tk);
    for (int d = 0; d < 6; ++d) {
      e = std::max(e, std::fabs(pos_end(d) - st.a[k + 1](d, 0)));   // 다음 위상 시작 pos = a(d,0)
      e = std::max(e, std::fabs(vel_end(d) - st.a[k + 1](d, 1)));   // 다음 위상 시작 vel = a(d,1)
    }
  }
  return e;
}

// ── ZYX 회전 (Drake get_R_B): phi_B=[psi,theta,phi]=[yaw,pitch,roll], R=Rz(psi)Ry(theta)Rx(phi) ──
inline Eigen::Matrix3d R_B(const Vector3d& phi_B) {
  double psi = phi_B(0), theta = phi_B(1), phi = phi_B(2);
  double cz = std::cos(psi), sz = std::sin(psi), cy = std::cos(theta), sy = std::sin(theta), cx = std::cos(phi), sx = std::sin(phi);
  Eigen::Matrix3d Rz, Ry, Rx;
  Rz << cz, -sz, 0, sz, cz, 0, 0, 0, 1;
  Ry << cy, 0, sy, 0, 1, 0, -sy, 0, cy;
  Rx << 1, 0, 0, 0, cx, -sx, 0, sx, cx;
  return Rz * Ry * Rx;
}

// ── friction cone (Drake 간이): 각 phase·τ서 base z-가속 ≥ -9.81 (자유낙하보다 안 빠름). 선형 ──
inline double friction_residual(const TamolsState& st) {
  double e = 0;
  const int S = 6;   // tau_sampling_rate
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int s = 0; s < S; ++s) {
      double tau = Tk * (double)s / (double)S;             // linspace(0,Tk,S+1)[:S]
      double az = st.acc_at(k, tau)(2);                    // base z-가속
      e = std::max(e, std::max(0.0, -9.81 - az));          // az ≥ -9.81 위반량
    }
  }
  return e;
}

// ── kinematic reach (Drake): at_des_position 다리서 l_min²≤|base+R_B·hip−foot|²≤l_max² ──
inline double kinematic_residual(const TamolsState& st) {
  double e = 0;
  const int S = 6;
  const double lo = st.prm.l_min * st.prm.l_min, hi = st.prm.l_max * st.prm.l_max;
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int L = 0; L < 4; ++L) {
      if (!st.gait[k].at_des[L]) continue;                 // at_des_position 다리만
      for (int s = 0; s < S; ++s) {
        double tau = Tk * (double)s / (double)S;
        Vector6d pose = st.pos_at(k, tau);
        Vector3d base = pose.head<3>();
        Eigen::Matrix3d R = R_B(pose.tail<3>());
        Vector3d diff = base + R * st.prm.hip_offsets.row(L).transpose() - st.p.row(L).transpose();
        double total = diff.dot(diff);
        e = std::max(e, std::max(0.0, lo - total));        // ≥ l_min² 위반
        e = std::max(e, std::max(0.0, total - hi));        // ≤ l_max² 위반
      }
    }
  }
  return e;
}

// ── 스칼라 삼중곱 det([a b c]) = a·(b×c) (Drake determinant) ──
inline double det3(const Vector3d& a, const Vector3d& b, const Vector3d& c) { return a.dot(b.cross(c)); }

// ── GIAC 안정성 (Drake add_dynamics_constraints, Eq17) ──
//   17a(N>0 마찰콘): (μ·a_z)²−a_x²−a_y² ≥ 0
//   17b(N≥3): 접촉쌍마다 m·det(p_ij,p_B−p_i,a_B) − p_ij·L̇ ≤ eps
//   17c(N==2): |위 식| ≤ eps ,  17d: −det(e_z,p_ij,M_i) ≤ eps  (M_i=(p_B−p_i)×(g−a_B)−L̇/m)
//   eps=epsilon[phase](≥0). residual=최대 위반량(feasible 해서 ≈0).
inline double giac_residual(const TamolsState& st) {
  double e = 0;
  const int S = 6;
  const double mu = st.prm.mu, m = st.prm.mass;
  const Vector3d ez(0, 0, 1), gvec(0, 0, -9.81);
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    std::vector<int> stance;
    for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) stance.push_back(i);
    int N = (int)stance.size();
    double eps = (k < st.epsilon.size()) ? st.epsilon(k) : 0.0;
    e = std::max(e, std::max(0.0, -eps));                       // eps ≥ 0
    auto foot = [&](int i) -> Vector3d {
      return st.gait[k].at_des[i] ? Vector3d(st.p.row(i).transpose())
                                  : Vector3d(st.p_meas.row(i).transpose());
    };
    for (int s = 0; s < S; ++s) {
      double tau = Tk * (double)s / (double)S;
      Vector3d pB = st.pos_at(k, tau).head<3>();
      Vector3d aB = st.acc_at(k, tau).head<3>();
      Vector3d Ld = st.Ldot_at(k, tau);
      if (N > 0) {                                              // 17a: (μ a_z)²−a_x²−a_y² ≥ 0
        double lhs = (mu * aB(2)) * (mu * aB(2)) - aB(0) * aB(0) - aB(1) * aB(1);
        e = std::max(e, std::max(0.0, -lhs));
      }
      if (N >= 3) {                                             // 17b: 접촉쌍
        for (size_t x = 0; x < stance.size(); ++x)
          for (size_t y = x + 1; y < stance.size(); ++y) {
            Vector3d pi = foot(stance[x]), pj = foot(stance[y]), pij = pj - pi;
            double lhs = m * det3(pij, pB - pi, aB) - pij.dot(Ld);
            e = std::max(e, std::max(0.0, lhs - eps));
          }
      }
      if (N == 2) {                                             // 17c,d: 이중지지
        Vector3d pi = foot(stance[0]), pj = foot(stance[1]), pij = pj - pi;
        double val = m * det3(pij, pB - pi, aB) - pij.dot(Ld);
        e = std::max(e, std::max(0.0, val - eps));              // 17c: |val| ≤ eps
        e = std::max(e, std::max(0.0, -val - eps));
        Vector3d Mi = (pB - pi).cross(gvec - aB) - Ld / m;      // 17d
        double cost = det3(ez, pij, Mi);
        e = std::max(e, std::max(0.0, -cost - eps));
      }
    }
  }
  return e;
}

} // namespace tamols
