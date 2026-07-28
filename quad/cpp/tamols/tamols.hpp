// tamols.hpp — C++ 실시간 TAMOLS (모델기반 지형 joint MPC) · 상태·파라미터·스플라인 평가
//
//   목표: D1(OCS2 Perceptive NMPC) 청사진을 우리 MuJoCo/C++ 스택에 native로 구현.
//         base 스플라인(phase별 6-DOF×order) + 발판 + GIAC 안정성 + 지형(footScore/edge)을
//         공동최적화하는 실시간 지형 MPC. SQP-RTI + 빠른 QP(eiquadprog).
//   근거: Drake 프로토타입 tamols_02leg.py(우리가 검증한 정식화)를 C++로 포팅.
//         변수/제약/비용 = tamols.py·constraints.py·costs.py·helpers.py 대응.
//   이 파일: 결정적 수치 핵심 = 상태 구조 + 스플라인 평가(pos/vel/acc). Drake helpers.py 정합.
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>

namespace tamols {

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;
using Vector6d = Eigen::Matrix<double, 6, 1>;

// ── 로봇/문제 파라미터 (02_Leg, Drake setup_02leg_state 정합) ──
struct Params {
  double mass          = 37.9;    // 02_Leg 17dof
  double mu            = 0.6;     // 배포 MU
  double foot_radius   = 0.018;   // sphere 발
  double nominal_height= 0.52;    // base_z0
  double h_des         = 0.52;    // base 정렬 목표높이
  int    base_dims     = 6;       // x,y,z,roll,pitch,yaw
  int    spline_order  = 4;       // cubic (a0+a1 τ+a2 τ²+a3 τ³)
  int    num_legs      = 4;       // FL,FR,RL,RR
  double l_min         = 0.12;    // 다리 reach 최소/최대
  double l_max         = 0.80;
  double min_foot_dist = 0.10;
  double g             = 9.81;
  Vector3d inertia_diag = Vector3d(0.02448, 0.098077, 0.107);  // ★Drake 기본(Go2)·02_Leg 미override → 정합 위해 그대로
  // hip 오프셋(base 기준, FL/FR/RL/RR) — Drake와 동일
  Eigen::Matrix<double, 4, 3> hip_offsets = (Eigen::Matrix<double, 4, 3>() <<
       0.225,  0.14, 0.0,
       0.225, -0.14, 0.0,
      -0.225,  0.14, 0.0,
      -0.225, -0.14, 0.0).finished();
};

// ── 게이트 위상 (Drake gait_pattern) ──
struct GaitPhase {
  double duration;                 // phase 길이 [s]
  std::array<int, 4> contact;      // 1=stance, 0=swing (FL,FR,RL,RR)
  std::array<int, 4> at_des{0,0,0,0}; // 1=발이 목표위치 도달(kinematic reach 제약 적용 대상)
};

// ── TAMOLS 상태 (결정변수 + 초기조건 + 지형) ──
struct TamolsState {
  Params prm;
  std::vector<GaitPhase> gait;                 // 위상들

  // 결정변수 (Drake: spline_coeffs, p, epsilon)
  std::vector<MatrixXd> a;   // a[phase] = (base_dims × spline_order) 스플라인 계수. col(i)=τ^i 계수(6벡터)
  Eigen::Matrix<double, 4, 3> p;               // 발판 (world, 로봇중심 프레임)
  Eigen::VectorXd epsilon;                     // phase별 GIAC slack

  // 초기조건 (Drake: base_pose, base_vel, p_meas)
  Vector6d base_pose = Vector6d::Zero();       // [x,y,z,r,p,yw]
  Vector6d base_vel  = Vector6d::Zero();
  Eigen::Matrix<double, 4, 3> p_meas;          // 측정 발위치
  Vector3d ref_vel   = Vector3d(0.4, 0, 0);    // 명령 속도

  int num_phases() const { return (int)gait.size(); }

  // ── 스플라인 평가 (Drake helpers.py 정합) ──
  //   pos(τ) = Σ_{i=0}^{order-1} a.col(i) · τ^i
  Vector6d pos(const MatrixXd& a_k, double tau) const {
    Vector6d s = Vector6d::Zero();
    double t = 1.0;                              // τ^0
    for (int i = 0; i < prm.spline_order; ++i) { s += a_k.col(i) * t; t *= tau; }
    return s;
  }
  //   vel(τ) = Σ_{i=1}^{order-1} i · a.col(i) · τ^{i-1}
  Vector6d vel(const MatrixXd& a_k, double tau) const {
    Vector6d s = Vector6d::Zero();
    double t = 1.0;                              // τ^{i-1}, i=1 → τ^0
    for (int i = 1; i < prm.spline_order; ++i) { s += (double)i * a_k.col(i) * t; t *= tau; }
    return s;
  }
  //   acc(τ) = Σ_{i=2}^{order-1} i(i-1) · a.col(i) · τ^{i-2}
  Vector6d acc(const MatrixXd& a_k, double tau) const {
    Vector6d s = Vector6d::Zero();
    double t = 1.0;                              // τ^{i-2}, i=2 → τ^0
    for (int i = 2; i < prm.spline_order; ++i) { s += (double)(i * (i - 1)) * a_k.col(i) * t; t *= tau; }
    return s;
  }

  // 위상 τ에서의 base 위치/속도/가속 (a[phase] 사용)
  Vector6d pos_at(int phase, double tau) const { return pos(a[phase], tau); }
  Vector6d vel_at(int phase, double tau) const { return vel(a[phase], tau); }
  Vector6d acc_at(int phase, double tau) const { return acc(a[phase], tau); }

  // ── 각운동량 도함수 L_dot (Drake evaluate_angular_momentum_derivative 정합) ──
  //   euler=[phi,theta,psi]=pos[3,4,5](=roll,pitch,yaw). ω·dω(오일러율) → L_dot=I·dω+ω×(I·ω)
  Vector3d Ldot_at(int phase, double tau) const {
    Vector6d p6 = pos_at(phase, tau), v6 = vel_at(phase, tau), a6 = acc_at(phase, tau);
    double phi = p6(3), theta = p6(4);                                  // psi(=p6(5)) 미사용(공식이 psi_dot만)
    double pd = v6(3), td = v6(4), sd = v6(5);                          // phi_dot, theta_dot, psi_dot
    double pdd = a6(3), tdd = a6(4), sdd = a6(5);
    double sphi = std::sin(phi), cphi = std::cos(phi), sth = std::sin(theta), cth = std::cos(theta);
    Vector3d w, dw;
    w(0) = pd - sd * sth;
    w(1) = td * cphi + sd * sphi * cth;
    w(2) = sd * cphi * cth - td * sphi;
    dw(0) = pdd - (sdd * sth + sd * cth * td);
    dw(1) = tdd * cphi - td * pd * sphi + sdd * sphi * cth + sd * (cphi * cth * pd - sphi * sth * td);
    dw(2) = sdd * cphi * cth - sd * (sphi * cth * pd + cphi * sth * td) - tdd * sphi - td * pd * cphi;
    const Vector3d& I = prm.inertia_diag;
    Vector3d Iw(I(0) * w(0), I(1) * w(1), I(2) * w(2));
    Vector3d Idw(I(0) * dw(0), I(1) * dw(1), I(2) * dw(2));
    return Idw + w.cross(Iw);
  }
};

} // namespace tamols
