// test_jac.cpp — 비용 잔차 해석 Jacobian vs FD 대조 + 타이밍
#include "tamols_jac.hpp"
#include <cstdio>
#include <fstream>
#include <chrono>

using namespace tamols;

int main() {
  TamolsState st;
  if (!load_solution("/tmp/tamols_sol.txt", st)) { std::printf("★ sol 로드 실패\n"); return 1; }
  std::ifstream f("/tmp/tamols_cost.txt");
  if (!f) { std::printf("★ cost map 없음\n"); return 1; }
  double cell; int map_size; f >> cell >> map_size;
  Grid h(map_size, map_size);
  for (int i = 0; i < map_size; ++i) for (int j = 0; j < map_size; ++j) f >> h(i, j);

  Packer pk(st.num_phases());
  auto Rf = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return cost_residuals(s, h, cell, map_size); };
  VectorXd z = pk.pack(st);

  // 해석 vs FD
  auto t0 = std::chrono::high_resolution_clock::now();
  MatrixXd Ja = cost_jacobian(st, h, cell, map_size);
  auto t1 = std::chrono::high_resolution_clock::now();
  MatrixXd Jf = fd_jacobian(Rf, z);
  auto t2 = std::chrono::high_resolution_clock::now();

  double err = (Ja - Jf).cwiseAbs().maxCoeff();
  double ta = std::chrono::duration<double, std::micro>(t1 - t0).count();
  double tf = std::chrono::duration<double, std::micro>(t2 - t1).count();
  std::printf("[비용 Jacobian 해석 vs FD]  (%dx%d)\n", (int)Ja.rows(), (int)Ja.cols());
  std::printf("  최대오차 = %.2e  %s\n", err, err < 1e-5 ? "OK" : "★불일치");
  std::printf("  타이밍: 해석 %.1f µs · FD %.1f µs · 가속 %.0f×\n", ta, tf, tf / std::max(1.0, ta));

  // 등식 제약 Jacobian
  auto Ef = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return eq_constraints(s); };
  MatrixXd Ea = eq_jacobian(st), Ef2 = fd_jacobian(Ef, z);
  double eerr = (Ea - Ef2).cwiseAbs().maxCoeff();
  std::printf("[등식 Jacobian 해석 vs FD]  (%dx%d)  최대오차 = %.2e  %s\n",
              (int)Ea.rows(), (int)Ea.cols(), eerr, eerr < 1e-5 ? "OK" : "★불일치");

  // 부등식: 비GIAC 행(선형+kinematic) 해석 vs FD (GIAC 행은 mask=FD 대상이라 제외)
  QpOptions o;
  auto Gf = [&](const VectorXd& zz){ TamolsState s = st; pk.unpack(zz, s); return ineq_constraints(s, o); };
  std::vector<char> gmask;
  MatrixXd Ia = ineq_jacobian_partial(st, o, gmask);
  MatrixXd If = fd_jacobian(Gf, z);
  double ierr = 0; int n_ana = 0, n_giac = 0;
  for (int rr = 0; rr < Ia.rows(); ++rr) {
    if (gmask[rr]) { ++n_giac; continue; }               // GIAC=FD대상, 비교 제외
    ++n_ana; ierr = std::max(ierr, (Ia.row(rr) - If.row(rr)).cwiseAbs().maxCoeff());
  }
  std::printf("[부등식 Jacobian 비GIAC(선형+kinematic) 해석 vs FD]  행 %d(해석)+%d(GIAC=FD)  최대오차 = %.2e  %s\n",
              n_ana, n_giac, ierr, ierr < 1e-5 ? "OK" : "★불일치");
  // 전체 부등식(비GIAC 해석 + GIAC sparse FD) vs 전체 FD — 타이밍 비교
  auto t3 = std::chrono::high_resolution_clock::now();
  MatrixXd Ifull = ineq_jacobian_full(st, o, pk);
  auto t4 = std::chrono::high_resolution_clock::now();
  MatrixXd Iref = fd_jacobian(Gf, z);
  auto t5 = std::chrono::high_resolution_clock::now();
  double ferr = (Ifull - Iref).cwiseAbs().maxCoeff();
  double tfull = std::chrono::duration<double, std::micro>(t4 - t3).count();
  double tref = std::chrono::duration<double, std::micro>(t5 - t4).count();
  std::printf("[전체 부등식 Jacobian: 해석+GIAC-sparseFD vs 전체FD]  최대오차 = %.2e  %s\n", ferr, ferr < 1e-5 ? "OK" : "★");
  std::printf("  타이밍: 하이브리드 %.0f µs · 전체FD %.0f µs · 가속 %.1f×\n", tfull, tref, tref / std::max(1.0, tfull));

  bool pass = err < 1e-5 && eerr < 1e-5 && ierr < 1e-5 && ferr < 1e-5;
  std::printf("  → 해석 Jacobian 전체 %s\n", pass ? "검증 ✓" : "확인필요");
  return pass ? 0 : 1;
}
