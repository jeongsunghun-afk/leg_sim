// test_spline.cpp — TAMOLS 스플라인 평가 결정적 검증
//   ① vel(τ) ≈ d/dτ pos(τ) (유한차분)  ② acc(τ) ≈ d/dτ vel(τ)  ③ 초기조건 매핑(pos(0)=a0, vel(0)=a1)
//   빌드: g++ -O2 -std=c++17 test_spline.cpp -I<eigen> -o test_spline && ./test_spline
#include "tamols.hpp"
#include <cstdio>
#include <random>

using namespace tamols;

int main() {
  TamolsState st;
  st.gait = { {0.4, {1,1,1,1}}, {0.4, {1,0,1,0}} };   // 2-phase 더미
  // 임의 스플라인 계수 (6×4) — 결정적 시드
  std::mt19937 rng(42); std::uniform_real_distribution<double> U(-1.0, 1.0);
  st.a.clear();
  for (int ph = 0; ph < st.num_phases(); ++ph) {
    MatrixXd a(6, 4);
    for (int r = 0; r < 6; ++r) for (int c = 0; c < 4; ++c) a(r, c) = U(rng);
    st.a.push_back(a);
  }

  const double eps = 1e-6;
  double max_vel_err = 0, max_acc_err = 0;
  for (int ph = 0; ph < st.num_phases(); ++ph) {
    for (double tau = 0.05; tau <= 0.4; tau += 0.05) {
      // ① vel vs FD(pos)
      Vector6d v_ana = st.vel_at(ph, tau);
      Vector6d v_fd  = (st.pos_at(ph, tau + eps) - st.pos_at(ph, tau - eps)) / (2 * eps);
      max_vel_err = std::max(max_vel_err, (v_ana - v_fd).cwiseAbs().maxCoeff());
      // ② acc vs FD(vel)
      Vector6d a_ana = st.acc_at(ph, tau);
      Vector6d a_fd  = (st.vel_at(ph, tau + eps) - st.vel_at(ph, tau - eps)) / (2 * eps);
      max_acc_err = std::max(max_acc_err, (a_ana - a_fd).cwiseAbs().maxCoeff());
    }
  }

  // ③ 초기조건: pos(0) = a.col(0), vel(0) = a.col(1) (Drake add_initial_constraints 정합)
  Vector6d p0 = st.pos_at(0, 0.0), v0 = st.vel_at(0, 0.0);
  double ic_pos = (p0 - st.a[0].col(0)).cwiseAbs().maxCoeff();
  double ic_vel = (v0 - st.a[0].col(1)).cwiseAbs().maxCoeff();

  std::printf("[TAMOLS spline test]\n");
  std::printf("  ① vel vs FD(pos)  최대오차 = %.2e  %s\n", max_vel_err, max_vel_err < 1e-4 ? "OK" : "★FAIL");
  std::printf("  ② acc vs FD(vel)  최대오차 = %.2e  %s\n", max_acc_err, max_acc_err < 1e-4 ? "OK" : "★FAIL");
  std::printf("  ③ 초기조건 pos(0)=a0 (%.2e) · vel(0)=a1 (%.2e)  %s\n",
              ic_pos, ic_vel, (ic_pos < 1e-12 && ic_vel < 1e-12) ? "OK" : "★FAIL");
  bool ok = max_vel_err < 1e-4 && max_acc_err < 1e-4 && ic_pos < 1e-12 && ic_vel < 1e-12;
  std::printf("  → 스플라인 평가 %s\n", ok ? "정합 ✓ (Drake helpers.py 대응)" : "불일치 — 확인필요");
  return ok ? 0 : 1;
}
