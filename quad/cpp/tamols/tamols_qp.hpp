// tamols_qp.hpp — TAMOLS SQP-RTI 솔버 (eiquadprog QP)
//   정식화(제약5+비용3, Drake 검증)를 결정벡터 z로 묶어 SQP로 풂.
//   비용=가중 비선형최소제곱 → Gauss-Newton(H=2JᵀJ, g=2JᵀR). 제약=선형화(FD Jacobian).
//   QP: min ½Δzᵀ H Δz + gᵀΔz  s.t.  CE·Δz+ce0=0, CI·Δz+ci0≥0  (eiquadprog 규약).
//   ★1차 목표=정확성(FD Jacobian). 실시간(해석 Jacobian)은 수렴검증 후.
#pragma once
#include "costs.hpp"          // bilinear_height, 비용 항, constraints.hpp(R_B·det3) 포함
#include <eiquadprog/eiquadprog-fast.hpp>
#include <functional>

namespace tamols {

// ── 문제 옵션(Drake tamols_02leg 제약과 정합) ──
struct QpOptions {
  double zlo = 0.45, zhi = 0.60, rp_max = 0.20, yaw_max = 0.15;  // base bounds
  int    base_nsamp = 4;
  double x_target = 0.73;                                        // terminal 전진
  double y_min = 0.10, y_max = 0.22;                             // foot_y 대칭
  bool   gap = true; double gap_lo = 0.45, gap_hi = 0.65, gap_margin = 0.03;
  int    max_iter = 50; double tol = 1e-7; double reg = 1.0;     // SQP 시작 reg(적응형 LM, trust-region)
  bool   verbose = false;
};

// ── 결정벡터 packing: z = [a[0..P-1] (각 6×4, 열우선) | p(4×3) | epsilon(P)] ──
struct Packer {
  int P, nz, na;
  Packer(int P_) : P(P_) { na = 24 * P; nz = na + 12 + P; }
  VectorXd pack(const TamolsState& st) const {
    VectorXd z(nz); int o = 0;
    for (int k = 0; k < P; ++k) for (int i = 0; i < 4; ++i) for (int d = 0; d < 6; ++d) z(o++) = st.a[k](d, i);
    for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) z(o++) = st.p(L, c);
    for (int k = 0; k < P; ++k) z(o++) = (k < st.epsilon.size()) ? st.epsilon(k) : 0.0;
    return z;
  }
  void unpack(const VectorXd& z, TamolsState& st) const {
    int o = 0;
    if ((int)st.a.size() != P) st.a.assign(P, MatrixXd(6, 4));
    for (int k = 0; k < P; ++k) for (int i = 0; i < 4; ++i) for (int d = 0; d < 6; ++d) st.a[k](d, i) = z(o++);
    for (int L = 0; L < 4; ++L) for (int c = 0; c < 3; ++c) st.p(L, c) = z(o++);
    st.epsilon.resize(P); for (int k = 0; k < P; ++k) st.epsilon(k) = z(o++);
  }
};

// ── 비용 최소제곱 잔차 R(z): cost = |R|² = Σ w(...)²,  r_j = √w·(...) ──
inline VectorXd cost_residuals(const TamolsState& st, const Grid& h, double cell, int map_size) {
  std::vector<double> r;
  const int S = 6;
  // tracking: √(2Tk/S)·(vel_x−ref_x)
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int s = 0; s < S; ++s) {
      double tau = Tk * s / (double)S, w = 2.0 * Tk / S;
      r.push_back(std::sqrt(w) * (st.vel_at(k, tau)(0) - st.ref_vel(0)));
    }
  }
  // foothold_on_ground: √100·(h(p_i)−p_i.z)
  for (int i = 0; i < 4; ++i)
    r.push_back(10.0 * (bilinear_height(h, cell, map_size, st.p(i, 0), st.p(i, 1)) - st.p(i, 2)));
  // nominal_kinematic: √20·(p_B+R_B·hip−l_des−p_i)[c], 마지막 phase τ=T/2
  {
    int k = st.num_phases() - 1; double tau = st.gait[k].duration / 2.0;
    Vector6d pose = st.pos_at(k, tau); Vector3d pB = pose.head<3>();
    Eigen::Matrix3d R = R_B(pose.tail<3>()); Vector3d l_des(0, 0, st.prm.h_des);
    for (int i = 0; i < 4; ++i) {
      Vector3d bml = pB + R * st.prm.hip_offsets.row(i).transpose() - l_des;
      Vector3d pi = st.gait[k].at_des[i] ? Vector3d(st.p.row(i).transpose()) : Vector3d(st.p_meas.row(i).transpose());
      Vector3d e = bml - pi;
      for (int c = 0; c < 3; ++c) r.push_back(std::sqrt(20.0) * e(c));
    }
  }
  return Eigen::Map<VectorXd>(r.data(), r.size());
}

// ── 등식 제약 c(z)=0: 초기(12) + 위상연속(12·(P−1)) ──
inline VectorXd eq_constraints(const TamolsState& st) {
  std::vector<double> c;
  for (int d = 0; d < 6; ++d) { c.push_back(st.a[0](d, 0) - st.base_pose(d)); c.push_back(st.a[0](d, 1) - st.base_vel(d)); }
  for (int k = 0; k + 1 < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration; Vector6d pe = st.pos_at(k, Tk), ve = st.vel_at(k, Tk);
    for (int d = 0; d < 6; ++d) { c.push_back(pe(d) - st.a[k + 1](d, 0)); c.push_back(ve(d) - st.a[k + 1](d, 1)); }
  }
  return Eigen::Map<VectorXd>(c.data(), c.size());
}

// ── 부등식 제약 g(z)≥0: friction·kinematic·GIAC·base bounds·foot_y·gap·terminal·eps ──
inline VectorXd ineq_constraints(const TamolsState& st, const QpOptions& o) {
  std::vector<double> g;
  const int S = 6;
  const double lo = st.prm.l_min * st.prm.l_min, hi = st.prm.l_max * st.prm.l_max;
  const double mu = st.prm.mu, m = st.prm.mass;
  const Vector3d ez(0, 0, 1), gvec(0, 0, -9.81);
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    // friction: az(τ)+9.81 ≥ 0
    for (int s = 0; s < S; ++s) { double tau = Tk * s / (double)S; g.push_back(st.acc_at(k, tau)(2) + 9.81); }
    // kinematic reach (at_des): |diff|²−lo ≥ 0, hi−|diff|² ≥ 0
    for (int L = 0; L < 4; ++L) if (st.gait[k].at_des[L])
      for (int s = 0; s < S; ++s) {
        double tau = Tk * s / (double)S; Vector6d pose = st.pos_at(k, tau);
        Vector3d diff = pose.head<3>() + R_B(pose.tail<3>()) * st.prm.hip_offsets.row(L).transpose() - st.p.row(L).transpose();
        double t = diff.dot(diff); g.push_back(t - lo); g.push_back(hi - t);
      }
    // GIAC(Eq17)
    std::vector<int> stance; for (int i = 0; i < 4; ++i) if (st.gait[k].contact[i]) stance.push_back(i);
    int N = (int)stance.size(); double eps = (k < st.epsilon.size()) ? st.epsilon(k) : 0.0;
    g.push_back(eps);                                                   // eps ≥ 0
    auto foot = [&](int i) { return st.gait[k].at_des[i] ? Vector3d(st.p.row(i).transpose()) : Vector3d(st.p_meas.row(i).transpose()); };
    for (int s = 0; s < S; ++s) {
      double tau = Tk * s / (double)S;
      Vector3d pB = st.pos_at(k, tau).head<3>(), aB = st.acc_at(k, tau).head<3>(), Ld = st.Ldot_at(k, tau);
      if (N > 0) g.push_back((mu * aB(2)) * (mu * aB(2)) - aB(0) * aB(0) - aB(1) * aB(1));   // 17a ≥0
      if (N >= 3) for (size_t x = 0; x < stance.size(); ++x) for (size_t y = x + 1; y < stance.size(); ++y) {
        Vector3d pi = foot(stance[x]), pj = foot(stance[y]), pij = pj - pi;
        g.push_back(eps - (m * det3(pij, pB - pi, aB) - pij.dot(Ld)));                        // 17b: lhs≤eps
      }
      if (N == 2) {
        Vector3d pi = foot(stance[0]), pj = foot(stance[1]), pij = pj - pi;
        double val = m * det3(pij, pB - pi, aB) - pij.dot(Ld);
        g.push_back(eps - val); g.push_back(eps + val);                                       // 17c |val|≤eps
        Vector3d Mi = (pB - pi).cross(gvec - aB) - Ld / m; g.push_back(eps + det3(ez, pij, Mi)); // 17d
      }
    }
    // base bounds (nsamp, τ>0): z·roll·pitch·yaw
    for (int j = 1; j <= o.base_nsamp; ++j) {
      double tau = Tk * j / (double)o.base_nsamp; Vector6d s = st.pos_at(k, tau);
      g.push_back(s(2) - o.zlo); g.push_back(o.zhi - s(2));
      g.push_back(s(3) + o.rp_max); g.push_back(o.rp_max - s(3));
      g.push_back(s(4) + o.rp_max); g.push_back(o.rp_max - s(4));
      g.push_back(s(5) + o.yaw_max); g.push_back(o.yaw_max - s(5));
    }
  }
  // foot_y 대칭: 좌(0,2) y∈[ymin,ymax], 우(1,3) y∈[−ymax,−ymin]
  for (int L : {0, 2}) { g.push_back(st.p(L, 1) - o.y_min); g.push_back(o.y_max - st.p(L, 1)); }
  for (int R : {1, 3}) { g.push_back(-o.y_min - st.p(R, 1)); g.push_back(st.p(R, 1) + o.y_max); }
  // gap 회피: 앞(0,1) x≥gap_hi+m, 뒤(2,3) x≤gap_lo−m
  if (o.gap) {
    for (int F : {0, 1}) g.push_back(st.p(F, 0) - (o.gap_hi + o.gap_margin));
    for (int H : {2, 3}) g.push_back((o.gap_lo - o.gap_margin) - st.p(H, 0));
  }
  // terminal 전진: 마지막 phase 끝 x ≥ x_target
  { int k = st.num_phases() - 1; g.push_back(st.pos_at(k, st.gait[k].duration)(0) - o.x_target); }
  return Eigen::Map<VectorXd>(g.data(), g.size());
}

// ── 벡터함수 f(z)의 FD Jacobian (중앙차분) ──
inline MatrixXd fd_jacobian(const std::function<VectorXd(const VectorXd&)>& f, const VectorXd& z, double eps = 1e-6) {
  VectorXd f0 = f(z); int m = (int)f0.size(), n = (int)z.size();
  MatrixXd J(m, n); VectorXd zp = z;
  for (int j = 0; j < n; ++j) {
    double h = eps * std::max(1.0, std::fabs(z(j)));
    zp(j) = z(j) + h; VectorXd fp = f(zp);
    zp(j) = z(j) - h; VectorXd fm = f(zp);
    zp(j) = z(j); J.col(j) = (fp - fm) / (2 * h);
  }
  return J;
}

// ── SQP 결과 ──
struct QpResult { bool ok; int iters; double cost; double eq_viol; double ineq_viol; };

// ── ℓ1 위반량 : Σ|eq| + Σ max(0,−ineq) ──
inline double l1_viol(const VectorXd& E, const VectorXd& G) {
  double v = E.cwiseAbs().sum();
  for (int i = 0; i < G.size(); ++i) v += std::max(0.0, -G(i));
  return v;
}

// ── SQP-RTI 솔버: st(초기추정)→수렴. ℓ1 merit 백트래킹 라인서치(비선형 제약서 발산 방지) ──
inline QpResult solve(TamolsState& st, const Grid& h, double cell, int map_size, const QpOptions& o = QpOptions()) {
  Packer pk(st.num_phases());
  VectorXd z = pk.pack(st);
  auto Rf = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return cost_residuals(s, h, cell, map_size); };
  auto Ef = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return eq_constraints(s); };
  auto Gf = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return ineq_constraints(s, o); };
  const double mu = 1e3;   // ℓ1 페널티 계수(위반 지배 → feasibility 우선). merit = cost + μ·viol
  auto merit = [&](const VectorXd& zz){ return Rf(zz).squaredNorm() + mu * l1_viol(Ef(zz), Gf(zz)); };

  eiquadprog::solvers::EiquadprogFast qp;
  QpResult res{false, 0, 0, 0, 0};
  double reg = o.reg;                       // 적응형 trust-region 정칙화(LM)
  const int nz = pk.nz;
  for (int it = 0; it < o.max_iter; ++it) {
    VectorXd R = Rf(z), E = Ef(z), G = Gf(z);
    MatrixXd JR = fd_jacobian(Rf, z), JE = fd_jacobian(Ef, z), JG = fd_jacobian(Gf, z);
    int neq = (int)E.size(), nineq = (int)G.size();
    MatrixXd JRtJR = 2.0 * JR.transpose() * JR; VectorXd gg = 2.0 * JR.transpose() * R;
    MatrixXd CE = JE, CI = JG; VectorXd ce0 = E, ci0 = G;                    // JE·Δz+E=0, JG·Δz+G≥0
    double m0 = merit(z), stepnorm = 0, alpha_used = 0;
    // LM 내부: 큰 α의 merit 감소 스텝 찾을 때까지 reg↑(trust region 축소). α작음=스텝불량 → reg↑
    bool stepped = false; VectorXd best_dz; double best_alpha = 0;
    for (int tr = 0; tr < 20; ++tr) {
      MatrixXd H = JRtJR; H.diagonal().array() += reg;
      VectorXd dz(nz); qp.reset(nz, neq, nineq);
      auto stt = qp.solve_quadprog(H, gg, CE, ce0, CI, ci0, dz);
      if (stt == eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) {
        double alpha = 1.0; int ls;
        for (ls = 0; ls < 25; ++ls) { if (merit(z + alpha * dz) < m0 - 1e-10) break; alpha *= 0.5; }
        if (ls < 25) {                                                       // 감소 스텝 존재 → 폴백 기록
          if (alpha > best_alpha) { best_alpha = alpha; best_dz = alpha * dz; }
          if (alpha >= 0.25) {                                              // 충분히 큰 스텝 → 채택
            z += best_dz; stepnorm = best_dz.norm(); alpha_used = alpha; stepped = true;
            if (alpha >= 0.5) reg = std::max(1e-4, reg * 0.5);              // 좋은 스텝 → trust region 확대
            break;
          }
        }
      }
      reg = std::min(1e6, reg * 4.0);                                        // α작음/QP실패 → trust region 축소
    }
    if (!stepped && best_alpha > 0) { z += best_dz; stepnorm = best_dz.norm(); alpha_used = best_alpha; stepped = true; }
    res.iters = it + 1;
    if (o.verbose) {
      VectorXd E2 = Ef(z), G2 = Gf(z); double iv = 0; for (int i = 0; i < G2.size(); ++i) iv = std::max(iv, std::max(0.0, -G2(i)));
      std::printf("   it%2d  α=%.3f reg=%.1e  cost=%.5f  eq=%.2e  ineq=%.2e  |dz|=%.2e\n",
                  it, alpha_used, reg, Rf(z).squaredNorm(), E2.cwiseAbs().maxCoeff(), iv, stepnorm);
    }
    if (!stepped) break;                                                     // 진척 불가
    if (stepnorm < o.tol) { res.ok = true; break; }
  }
  pk.unpack(z, st);
  VectorXd R = cost_residuals(st, h, cell, map_size), E = eq_constraints(st), G = ineq_constraints(st, o);
  res.cost = R.squaredNorm();
  res.eq_viol = E.size() ? E.cwiseAbs().maxCoeff() : 0;
  res.ineq_viol = 0; for (int i = 0; i < G.size(); ++i) res.ineq_viol = std::max(res.ineq_viol, std::max(0.0, -G(i)));
  if (res.eq_viol < 1e-5 && res.ineq_viol < 1e-5) res.ok = true;
  return res;
}

} // namespace tamols
