#pragma once
// 공통 타입/상수. 모든 모듈이 이것만 공유(순환의존 방지).
#include <Eigen/Dense>

namespace qc {
using Eigen::VectorXd;
using Eigen::Vector3d;
using Eigen::Vector4d;

constexpr int NLEG = 4;                 // 다리 수
enum Leg { HL = 0, HR = 1, FL = 2, FR = 3 };   // MuJoCo legqp 순서

// 관절수(nu)는 모델 의존(17-DOF: 4다리×4관절 + 허리1 = 17). 런타임 값 사용.
}  // namespace qc
