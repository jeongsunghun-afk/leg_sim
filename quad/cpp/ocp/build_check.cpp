// crocoddyl C++ 빌드 통합 확인(S0). 실시간 OCP 개발(§9)의 빌드 de-risk.
//   crocoddyl 3.2.1 = std::shared_ptr(boost 아님). 링크: -lcrocoddyl -lpinocchio_default + conda libstdc++.
#include <iostream>
#include <crocoddyl/core/states/euclidean.hpp>
#include <crocoddyl/core/actions/lqr.hpp>
#include <crocoddyl/core/optctrl/shooting.hpp>
#include <crocoddyl/core/solvers/fddp.hpp>

int main() {
  std::size_t nx = 4, nu = 2, N = 10;
  auto model = std::make_shared<crocoddyl::ActionModelLQR>(nx, nu);
  std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> models(N, model);
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(nx);
  auto problem = std::make_shared<crocoddyl::ShootingProblem>(x0, models, model);
  crocoddyl::SolverFDDP solver(problem);
  bool ok = solver.solve({}, {}, 20);
  std::cout << "[ocp_check] crocoddyl C++ OK: solved=" << ok
            << " iters=" << solver.get_iter()
            << " cost=" << solver.get_cost() << std::endl;
  return ok ? 0 : 1;
}
