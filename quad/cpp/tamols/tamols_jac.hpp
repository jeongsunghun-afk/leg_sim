// tamols_jac.hpp — TAMOLS 해석 Jacobian (실시간화: FD 대체)
//   비용 잔차 R(z)의 해석 Jacobian(∂R/∂z). FD 대비 정확·고속. FD와 대조 검증(<1e-5).
//   z 색인(Packer 정합): a[k](d,i)=k*24+i*6+d · p(L,c)=24P+L*3+c · eps(k)=24P+12+k.
#pragma once
#include "costs.hpp"       // bilinear_height, R_B, TamolsState
#include "tamols_qp.hpp"   // cost_residuals(순서 정합용)

namespace tamols {

// z 색인 헬퍼
inline int ia(int k, int d, int i) { return k * 24 + i * 6 + d; }       // a[k](d,i)
inline int ip(int P, int L, int c) { return 24 * P + L * 3 + c; }        // p(L,c)

// 양선형 높이 gradient (∂h/∂x, ∂h/∂y) — bilinear_height 정합
inline void bilinear_grad(const Grid& h, double cell, int map_size, double x, double y, double& gx, double& gy) {
  double off = cell * map_size / 2.0;
  double i = (x + off) / cell, j = (y + off) / cell;
  const int m = (int)h.rows(), n = (int)h.cols();
  int k0 = (int)std::floor(i), l0 = (int)std::floor(j);
  gx = 0; gy = 0;
  for (int k = k0; k <= k0 + 1; ++k) {
    if (k < 0 || k >= m) continue;
    double di = i - k, wx = 1.0 - std::fabs(di); if (wx <= 0) continue;
    double dwx = (di > 0 ? -1.0 : 1.0) / cell;                         // d(1-|i-k|)/dx = -sign(di)/cell
    for (int l = l0; l <= l0 + 1; ++l) {
      if (l < 0 || l >= n) continue;
      double dj = j - l, wy = 1.0 - std::fabs(dj); if (wy <= 0) continue;
      double dwy = (dj > 0 ? -1.0 : 1.0) / cell;
      gx += h(k, l) * dwx * wy;
      gy += h(k, l) * wx * dwy;
    }
  }
}

// 회전 미분: Rz(ψ)Ry(θ)Rx(φ), ∂R/∂각. (constraints.hpp R_B 정합: phi_B=(psi,theta,phi))
inline void dR_B(const Vector3d& phi_B, Eigen::Matrix3d& dR0, Eigen::Matrix3d& dR1, Eigen::Matrix3d& dR2) {
  double psi = phi_B(0), theta = phi_B(1), phi = phi_B(2);
  double cz = std::cos(psi), sz = std::sin(psi), cy = std::cos(theta), sy = std::sin(theta), cx = std::cos(phi), sx = std::sin(phi);
  Eigen::Matrix3d Rz, Ry, Rx, dRz, dRy, dRx;
  Rz << cz, -sz, 0, sz, cz, 0, 0, 0, 1;
  Ry << cy, 0, sy, 0, 1, 0, -sy, 0, cy;
  Rx << 1, 0, 0, 0, cx, -sx, 0, sx, cx;
  dRz << -sz, -cz, 0, cz, -sz, 0, 0, 0, 0;
  dRy << -sy, 0, cy, 0, 0, 0, -cy, 0, -sy;
  dRx << 0, 0, 0, 0, -sx, -cx, 0, cx, -sx;
  dR0 = dRz * Ry * Rx;   // ∂/∂psi
  dR1 = Rz * dRy * Rx;   // ∂/∂theta
  dR2 = Rz * Ry * dRx;   // ∂/∂phi
}

// 비용 잔차 R(z)의 해석 Jacobian (행 순서 = cost_residuals와 동일)
inline MatrixXd cost_jacobian(const TamolsState& st, const Grid& h, double cell, int map_size) {
  const int P = st.num_phases(), S = 6, nz = 24 * P + 12 + P;
  int nR = 6 * P + 4 + 12;
  MatrixXd J = MatrixXd::Zero(nR, nz);
  int r = 0;
  // ── tracking: √(2Tk/S)·(vel_x−ref). vel_x=Σ_{i≥1} i·a[k](0,i)·τ^{i-1} ──
  for (int k = 0; k < P; ++k) {
    double Tk = st.gait[k].duration;
    for (int s = 0; s < S; ++s, ++r) {
      double tau = Tk * s / (double)S, w = 2.0 * Tk / S, sw = std::sqrt(w);
      double t = 1.0;   // τ^{i-1}
      for (int i = 1; i < 4; ++i) { J(r, ia(k, 0, i)) = sw * i * t; t *= tau; }
    }
  }
  // ── foothold: 10·(h(p_i.xy)−p_i.z) ──
  for (int i = 0; i < 4; ++i, ++r) {
    double gx, gy; bilinear_grad(h, cell, map_size, st.p(i, 0), st.p(i, 1), gx, gy);
    J(r, ip(P, i, 0)) = 10.0 * gx;
    J(r, ip(P, i, 1)) = 10.0 * gy;
    J(r, ip(P, i, 2)) = -10.0;
  }
  // ── nominal: √20·(pB + R_B·hip − l_des − p_i)[c], 마지막 phase τ=T/2 ──
  {
    int k = P - 1; double Tk = st.gait[k].duration, tau = Tk / 2.0;
    Vector6d pose = st.pos_at(k, tau);
    Vector3d phi_B = pose.tail<3>();
    Eigen::Matrix3d dR0, dR1, dR2; dR_B(phi_B, dR0, dR1, dR2);
    const double sw = std::sqrt(20.0);
    for (int i = 0; i < 4; ++i) {
      Vector3d hip = st.prm.hip_offsets.row(i).transpose();
      Vector3d dRh0 = dR0 * hip, dRh1 = dR1 * hip, dRh2 = dR2 * hip;    // ∂(R_B·hip)/∂각
      bool at = st.gait[k].at_des[i];
      for (int c = 0; c < 3; ++c, ++r) {
        double t = 1.0;
        for (int ii = 0; ii < 4; ++ii) {                               // τ^ii
          J(r, ia(k, c, ii))     += sw * t;                            // ∂pB(c)/∂a(c,ii)
          J(r, ia(k, 3, ii))     += sw * dRh0(c) * t;                  // ∂(R_B·hip)(c)/∂psi(=a(3,ii))
          J(r, ia(k, 4, ii))     += sw * dRh1(c) * t;                  // ∂/∂theta(=a(4,ii))
          J(r, ia(k, 5, ii))     += sw * dRh2(c) * t;                  // ∂/∂phi(=a(5,ii))
          t *= tau;
        }
        if (at) J(r, ip(P, i, c)) += -sw;                              // ∂/∂p_i(c) (at_des 다리만)
      }
    }
  }
  return J;
}

// ── 등식 제약 해석 Jacobian (초기+위상연속, 전부 선형=상수) ──
inline MatrixXd eq_jacobian(const TamolsState& st) {
  const int P = st.num_phases(), nz = 24 * P + 12 + P, ord = st.prm.spline_order;
  int nE = 12 + 12 * (P - 1);
  MatrixXd J = MatrixXd::Zero(nE, nz);
  int r = 0;
  // 초기: a[0](d,0)=base_pose, a[0](d,1)=base_vel
  for (int d = 0; d < 6; ++d) {
    J(r++, ia(0, d, 0)) = 1.0;   // pos(0)
    J(r++, ia(0, d, 1)) = 1.0;   // vel(0)
  }
  // 위상연속: pos_k(Tk)(d) − a[k+1](d,0)=0, vel_k(Tk)(d) − a[k+1](d,1)=0
  for (int k = 0; k + 1 < P; ++k) {
    double Tk = st.gait[k].duration;
    for (int d = 0; d < 6; ++d) {
      // pos_k(Tk) = Σ_i a[k](d,i)·Tk^i
      double t = 1.0;
      for (int i = 0; i < ord; ++i) { J(r, ia(k, d, i)) = t; t *= Tk; }
      J(r, ia(k + 1, d, 0)) = -1.0; ++r;
      // vel_k(Tk) = Σ_{i≥1} i·a[k](d,i)·Tk^{i-1}
      t = 1.0;
      for (int i = 1; i < ord; ++i) { J(r, ia(k, d, i)) = (double)i * t; t *= Tk; }
      J(r, ia(k + 1, d, 1)) = -1.0; ++r;
    }
  }
  return J;
}

// ── 부등식 제약 해석 Jacobian (비GIAC 행: 선형+kinematic 해석 · GIAC 행=mask로 표시) ──
//   행 순서 = ineq_constraints 정합. GIAC 행은 J=0으로 두고 giac_row[r]=true (FD 대상).
inline MatrixXd ineq_jacobian_partial(const TamolsState& st, const QpOptions& o, std::vector<char>& giac_row) {
  const int P = st.num_phases(), S = 6, nz = 24 * P + 12 + P;
  const double lo = st.prm.l_min * st.prm.l_min, hi = st.prm.l_max * st.prm.l_max;
  VectorXd g = ineq_constraints(st, o);   // 행 수 = ineq_constraints 크기
  int nG = (int)g.size();
  MatrixXd J = MatrixXd::Zero(nG, nz);
  giac_row.assign(nG, 0);
  int r = 0;
  for (int k = 0; k < P; ++k) {
    double Tk = st.gait[k].duration;
    // friction: az(τ)+9.81 ≥0.  ∂az/∂a[k](2,i)=i(i-1)τ^{i-2}
    for (int s = 0; s < S; ++s, ++r) {
      double tau = Tk * s / (double)S, t = 1.0;   // τ^{i-2}
      for (int i = 2; i < 4; ++i) { J(r, ia(k, 2, i)) = (double)(i * (i - 1)) * t; t *= tau; }
    }
    // kinematic (at_des): |diff|²−lo≥0, hi−|diff|²≥0.  diff=pB+R_B·hip−foot
    for (int L = 0; L < 4; ++L) if (st.gait[k].at_des[L])
      for (int s = 0; s < S; ++s) {
        double tau = Tk * s / (double)S;
        Vector6d pose = st.pos_at(k, tau); Vector3d pB = pose.head<3>();
        Vector3d phi_B = pose.tail<3>(); Eigen::Matrix3d R = R_B(phi_B);
        Vector3d hip = st.prm.hip_offsets.row(L).transpose();
        Vector3d foot = st.p.row(L).transpose();
        Vector3d diff = pB + R * hip - foot;
        Eigen::Matrix3d dR0, dR1, dR2; dR_B(phi_B, dR0, dR1, dR2);
        Vector3d dRh0 = dR0 * hip, dRh1 = dR1 * hip, dRh2 = dR2 * hip;
        // ∂|diff|²/∂z = 2 diff·∂diff/∂z.  두 행(t−lo, hi−t)은 부호 반대
        VectorXd dt = VectorXd::Zero(nz);   // ∂(|diff|²)/∂z
        double tt = 1.0;
        for (int i = 0; i < 4; ++i) {
          for (int c = 0; c < 3; ++c) dt(ia(k, c, i)) += 2.0 * diff(c) * tt;        // ∂pB(c)
          dt(ia(k, 3, i)) += 2.0 * diff.dot(dRh0) * tt;                             // ∂(R_B·hip)/∂psi
          dt(ia(k, 4, i)) += 2.0 * diff.dot(dRh1) * tt;
          dt(ia(k, 5, i)) += 2.0 * diff.dot(dRh2) * tt;
          tt *= tau;
        }
        for (int c = 0; c < 3; ++c) dt(ip(P, L, c)) += -2.0 * diff(c);              // ∂(−foot)
        J.row(r) = dt.transpose(); ++r;          // t−lo
        J.row(r) = -dt.transpose(); ++r;         // hi−t
      }
    // GIAC: eps + (N>0)17a + (N≥3)17b쌍 + (N==2)17c×2+17d.  ← FD 대상(mask)
    std::vector<int> stance; for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) stance.push_back(i);
    int N = (int)stance.size();
    giac_row[r] = 1; ++r;                                                            // eps≥0
    for (int s = 0; s < S; ++s) {
      if (N > 0) { giac_row[r] = 1; ++r; }                                           // 17a
      if (N >= 3) for (size_t x = 0; x < stance.size(); ++x) for (size_t y = x + 1; y < stance.size(); ++y) { giac_row[r] = 1; ++r; }
      if (N == 2) { giac_row[r] = 1; ++r; giac_row[r] = 1; ++r; giac_row[r] = 1; ++r; }
    }
    // base bounds: z·roll·pitch·yaw (nsamp, τ>0). ∂pos(d)/∂a[k](d,i)=τ^i
    for (int j = 1; j <= o.base_nsamp; ++j) {
      double tau = Tk * j / (double)o.base_nsamp;
      auto lin = [&](int d, double sgn) { double t = 1.0; for (int i = 0; i < 4; ++i) { J(r, ia(k, d, i)) = sgn * t; t *= tau; } ++r; };
      lin(2, 1); lin(2, -1); lin(3, 1); lin(3, -1); lin(4, 1); lin(4, -1); lin(5, 1); lin(5, -1);
    }
  }
  // foot_y: 좌(0,2) p.y−ymin, ymax−p.y ; 우(1,3) −ymin−p.y, p.y+ymax
  for (int L : {0, 2}) { J(r++, ip(P, L, 1)) = 1.0; J(r++, ip(P, L, 1)) = -1.0; }
  for (int Rr : {1, 3}) { J(r++, ip(P, Rr, 1)) = -1.0; J(r++, ip(P, Rr, 1)) = 1.0; }
  // gap: 앞(0,1) p.x−(gap_hi+m) ; 뒤(2,3) (gap_lo−m)−p.x
  if (o.gap) { for (int F : {0, 1}) J(r++, ip(P, F, 0)) = 1.0; for (int H : {2, 3}) J(r++, ip(P, H, 0)) = -1.0; }
  // terminal: 마지막 phase 끝 x−x_target. ∂pos(0)/∂a[last](0,i)=Tk^i
  { int k = P - 1; double Tk = st.gait[k].duration, t = 1.0; for (int i = 0; i < 4; ++i) { J(r, ia(k, 0, i)) = t; t *= Tk; } ++r; }
  return J;
}

// ── GIAC 행만 반환(sparse FD 대상) — ineq_constraints의 GIAC 부분 정합 ──
inline VectorXd giac_residual_only(const TamolsState& st, const QpOptions& o) {
  std::vector<double> g;
  const int S = 6; const double mu = st.prm.mu, m = st.prm.mass;
  const Vector3d ez(0, 0, 1), gvec(0, 0, -9.81);
  (void)o;
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    std::vector<int> stance; for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) stance.push_back(i);
    int N = (int)stance.size(); double eps = (k < st.epsilon.size()) ? st.epsilon(k) : 0.0;
    g.push_back(eps);
    auto foot = [&](int i) { return st.gait[k].at_des[i] ? Vector3d(st.p.row(i).transpose()) : Vector3d(st.p_meas.row(i).transpose()); };
    for (int s = 0; s < S; ++s) {
      double tau = Tk * s / (double)S;
      Vector3d pB = st.pos_at(k, tau).head<3>(), aB = st.acc_at(k, tau).head<3>(), Ld = st.Ldot_at(k, tau);
      if (N > 0) g.push_back((mu * aB(2)) * (mu * aB(2)) - aB(0) * aB(0) - aB(1) * aB(1));
      if (N >= 3) for (size_t x = 0; x < stance.size(); ++x) for (size_t y = x + 1; y < stance.size(); ++y) {
        Vector3d pi = foot(stance[x]), pj = foot(stance[y]), pij = pj - pi;
        g.push_back(eps - (m * det3(pij, pB - pi, aB) - pij.dot(Ld)));
      }
      if (N == 2) {
        Vector3d pi = foot(stance[0]), pj = foot(stance[1]), pij = pj - pi;
        double val = m * det3(pij, pB - pi, aB) - pij.dot(Ld);
        g.push_back(eps - val); g.push_back(eps + val);
        Vector3d Mi = (pB - pi).cross(gvec - aB) - Ld / m; g.push_back(eps + det3(ez, pij, Mi));
      }
    }
  }
  return Eigen::Map<VectorXd>(g.data(), g.size());
}

// ── 단일 phase k의 GIAC 행만 ──
inline VectorXd giac_phase(const TamolsState& st, int k) {
  std::vector<double> g;
  const int S = 6; const double mu = st.prm.mu, m = st.prm.mass;
  const Vector3d ez(0, 0, 1), gvec(0, 0, -9.81);
  double Tk = st.gait[k].duration;
  std::vector<int> stance; for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) stance.push_back(i);
  int N = (int)stance.size(); double eps = (k < st.epsilon.size()) ? st.epsilon(k) : 0.0;
  g.push_back(eps);
  auto foot = [&](int i) { return st.gait[k].at_des[i] ? Vector3d(st.p.row(i).transpose()) : Vector3d(st.p_meas.row(i).transpose()); };
  for (int s = 0; s < S; ++s) {
    double tau = Tk * s / (double)S;
    Vector3d pB = st.pos_at(k, tau).head<3>(), aB = st.acc_at(k, tau).head<3>(), Ld = st.Ldot_at(k, tau);
    if (N > 0) g.push_back((mu * aB(2)) * (mu * aB(2)) - aB(0) * aB(0) - aB(1) * aB(1));
    if (N >= 3) for (size_t x = 0; x < stance.size(); ++x) for (size_t y = x + 1; y < stance.size(); ++y) {
      Vector3d pi = foot(stance[x]), pj = foot(stance[y]), pij = pj - pi;
      g.push_back(eps - (m * det3(pij, pB - pi, aB) - pij.dot(Ld)));
    }
    if (N == 2) {
      Vector3d pi = foot(stance[0]), pj = foot(stance[1]), pij = pj - pi;
      double val = m * det3(pij, pB - pi, aB) - pij.dot(Ld);
      g.push_back(eps - val); g.push_back(eps + val);
      Vector3d Mi = (pB - pi).cross(gvec - aB) - Ld / m; g.push_back(eps + det3(ez, pij, Mi));
    }
  }
  return Eigen::Map<VectorXd>(g.data(), g.size());
}

// ── GIAC Jacobian: 블록-sparse FD (a[k]는 phase-k만 평가, p는 전체, eps[k]는 phase-k) ──
inline MatrixXd giac_jacobian_sparse(const TamolsState& st, const QpOptions& o) {
  const int P = st.num_phases(), nz = 24 * P + 12 + P;
  std::vector<int> off(P + 1, 0);
  for (int k = 0; k < P; ++k) off[k + 1] = off[k] + (int)giac_phase(st, k).size();
  MatrixXd Jg = MatrixXd::Zero(off[P], nz);
  const double e = 1e-6;
  // a[k] 열: phase-k만 평가
  for (int k = 0; k < P; ++k)
    for (int i = 0; i < 4; ++i) for (int d = 0; d < 6; ++d) {
      TamolsState sp = st, sm = st; double b = st.a[k](d, i), h = e * std::max(1.0, std::fabs(b));
      sp.a[k](d, i) = b + h; sm.a[k](d, i) = b - h;
      VectorXd gp = giac_phase(sp, k), gm = giac_phase(sm, k);
      for (int r = 0; r < gp.size(); ++r) Jg(off[k] + r, ia(k, d, i)) = (gp(r) - gm(r)) / (2 * h);
    }
  // p 열: 전체 phase(발판 공유)
  for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) {
    TamolsState sp = st, sm = st; double b = st.p(L, c), h = e * std::max(1.0, std::fabs(b));
    sp.p(L, c) = b + h; sm.p(L, c) = b - h;
    VectorXd gp = giac_residual_only(sp, o), gm = giac_residual_only(sm, o);
    for (int r = 0; r < gp.size(); ++r) Jg(r, ip(P, L, c)) = (gp(r) - gm(r)) / (2 * h);
  }
  // eps[k] 열: phase-k만
  for (int k = 0; k < P; ++k) {
    TamolsState sp = st, sm = st; double b = st.epsilon(k), h = e * std::max(1.0, std::fabs(b));
    sp.epsilon(k) = b + h; sm.epsilon(k) = b - h;
    VectorXd gp = giac_phase(sp, k), gm = giac_phase(sm, k);
    for (int r = 0; r < gp.size(); ++r) Jg(off[k] + r, 24 * P + 12 + k) = (gp(r) - gm(r)) / (2 * h);
  }
  return Jg;
}

// ── 전체 부등식 Jacobian: 비GIAC 해석 + GIAC 블록-sparse FD ──
inline MatrixXd ineq_jacobian_full(const TamolsState& st, const QpOptions& o, const Packer& pk) {
  (void)pk;
  std::vector<char> gmask;
  MatrixXd J = ineq_jacobian_partial(st, o, gmask);
  MatrixXd Jg = giac_jacobian_sparse(st, o);
  int gi = 0;
  for (int r = 0; r < (int)gmask.size(); ++r) if (gmask[r]) J.row(r) = Jg.row(gi++);
  return J;
}

// ── elastic-mode QP 스텝 (ℓ1-SQP / SNOPT elastic mode) ──
//   하드 QP( CE·dz+ce0=0, CI·dz+ci0≥0 )가 국소 infeasible일 때 폴백.
//   슬랙 p,n(등식 ±)·t(부등식)로 완화 → QP가 항상 feasible → 위반-감소 방향 보장.
//   min ½dzᵀH dz + gᵀdz + μ(Σp+Σn+Σt)  s.t.  CE·dz+ce0 = p−n,  CI·dz+ci0 ≥ −t,  p,n,t≥0
inline bool elastic_step(const MatrixXd& H, const VectorXd& gg,
                         const MatrixXd& CE, const VectorXd& ce0,
                         const MatrixXd& CI, const VectorXd& ci0,
                         double mu, VectorXd& dz_out) {
  const int nz = (int)gg.size(), neq = (int)ce0.size(), nineq = (int)ci0.size();
  const int ny = nz + 2 * neq + nineq;                 // [dz | p(neq) | n(neq) | t(nineq)]
  const double eps = 1e-6;
  MatrixXd He = MatrixXd::Zero(ny, ny);
  He.topLeftCorner(nz, nz) = H;
  for (int i = nz; i < ny; ++i) He(i, i) += eps;       // 슬랙 PD 유지
  VectorXd ge(ny); ge.setZero(); ge.head(nz) = gg;
  ge.segment(nz, 2 * neq + nineq).setConstant(mu);     // ℓ1 페널티
  // 등식: CE·dz − p + n + ce0 = 0
  MatrixXd CEe = MatrixXd::Zero(neq, ny); VectorXd ce0e = ce0;
  if (neq) { CEe.leftCols(nz) = CE;
    CEe.block(0, nz, neq, neq) = -MatrixXd::Identity(neq, neq);
    CEe.block(0, nz + neq, neq, neq) = MatrixXd::Identity(neq, neq); }
  // 부등식: (CI·dz + t + ci0 ≥ 0) + (p≥0)(n≥0)(t≥0)
  const int mi = nineq + 2 * neq + nineq;
  MatrixXd CIe = MatrixXd::Zero(mi, ny); VectorXd ci0e = VectorXd::Zero(mi);
  if (nineq) { CIe.block(0, 0, nineq, nz) = CI;
    CIe.block(0, nz + 2 * neq, nineq, nineq) = MatrixXd::Identity(nineq, nineq);
    ci0e.head(nineq) = ci0; }
  for (int i = 0; i < 2 * neq + nineq; ++i) CIe(nineq + i, nz + i) = 1.0;   // 슬랙 ≥ 0
  eiquadprog::solvers::EiquadprogFast qp; qp.reset(ny, neq, mi);
  VectorXd y(ny);
  auto stt = qp.solve_quadprog(He, ge, CEe, ce0e, CIe, ci0e, y);
  if (stt != eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) return false;
  dz_out = y.head(nz); return true;
}

// ── 해석 Jacobian 솔버 (solve()의 FD → 해석 배선) — 실시간 측정용 ──
inline QpResult solve_fast(TamolsState& st, const Grid& h, double cell, int map_size, const QpOptions& o = QpOptions()) {
  Packer pk(st.num_phases());
  VectorXd z = pk.pack(st);
  auto Rf = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return cost_residuals(s, h, cell, map_size); };
  auto Ef = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return eq_constraints(s); };
  auto Gf = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return ineq_constraints(s, o); };
  const double mu = 1e3;
  auto merit = [&](const VectorXd& zz){ return Rf(zz).squaredNorm() + mu * l1_viol(Ef(zz), Gf(zz)); };
  eiquadprog::solvers::EiquadprogFast qp;
  QpResult res{false, 0, 0, 0, 0};
  double reg = o.reg; const int nz = pk.nz;
  for (int it = 0; it < o.max_iter; ++it) {
    TamolsState s = st; pk.unpack(z, s);
    VectorXd R = cost_residuals(s, h, cell, map_size), E = eq_constraints(s), G = ineq_constraints(s, o);
    MatrixXd JR = cost_jacobian(s, h, cell, map_size), JE = eq_jacobian(s), JG = ineq_jacobian_full(s, o, pk);   // ★해석
    int neq = (int)E.size(), nineq = (int)G.size();
    MatrixXd JRtJR = 2.0 * JR.transpose() * JR; VectorXd gg = 2.0 * JR.transpose() * R;
    MatrixXd CE = JE, CI = JG; VectorXd ce0 = E, ci0 = G;
    double m0 = merit(z), stepnorm = 0; bool stepped = false; VectorXd best_dz; double best_alpha = 0;
    for (int tr = 0; tr < 20; ++tr) {
      MatrixXd H = JRtJR; H.diagonal().array() += reg;
      VectorXd dz(nz); qp.reset(nz, neq, nineq);
      auto stt = qp.solve_quadprog(H, gg, CE, ce0, CI, ci0, dz);
      bool have_dir = (stt == eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL);
      if (!have_dir) have_dir = elastic_step(H, gg, CE, ce0, CI, ci0, mu, dz);   // ★폴백: 선형화제약 infeasible→슬랙완화(항상 feasible)
      if (have_dir) {
        double alpha = 1.0; int ls;
        for (ls = 0; ls < 25; ++ls) { if (merit(z + alpha * dz) < m0 - 1e-10) break; alpha *= 0.5; }
        if (ls < 25) {
          if (alpha > best_alpha) { best_alpha = alpha; best_dz = alpha * dz; }
          if (alpha >= 0.25) { z += best_dz; stepnorm = best_dz.norm(); stepped = true; if (alpha >= 0.5) reg = std::max(1e-4, reg * 0.5); break; }
        }
      }
      reg = std::min(1e6, reg * 4.0);
    }
    if (!stepped && best_alpha > 0) { z += best_dz; stepnorm = best_dz.norm(); stepped = true; }
    res.iters = it + 1;
    if (!stepped) break;
    if (stepnorm < o.tol) { res.ok = true; break; }
  }
  pk.unpack(z, st);
  VectorXd R = cost_residuals(st, h, cell, map_size), E = eq_constraints(st), G = ineq_constraints(st, o);
  res.cost = R.squaredNorm(); res.eq_viol = E.size() ? E.cwiseAbs().maxCoeff() : 0;
  res.ineq_viol = 0; for (int i = 0; i < G.size(); ++i) res.ineq_viol = std::max(res.ineq_viol, std::max(0.0, -G(i)));
  if (res.eq_viol < 1e-5 && res.ineq_viol < 1e-5) res.ok = true;
  return res;
}

} // namespace tamols
