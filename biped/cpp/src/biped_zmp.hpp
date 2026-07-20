// biped ZMP 프리뷰 제어 (cart-table, Katayama 최적 프리뷰 서보) — 평발 보행 CoM 궤적 생성기.
// footstep 기반 ZMP 레퍼런스 → CoM x,y 궤적(ZMP를 밑창 안에 유지). WBIC가 이 CoM ref를 추종.
// 게인(F·Gp)은 Riccati 반복으로 init서 산출(DARE, dt·zc 고정 상수).
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>
using namespace Eigen;

struct ZmpPreview {
  double dt, zc, g;
  Matrix3d A; Vector3d B; RowVector3d C;
  RowVector3d F;                 // 상태 피드백
  std::vector<double> Gp;        // 프리뷰 피드포워드
  int N=0;
  // CoM ref 상태 (축별: pos,vel,acc)
  Vector3d sx, sy;

  void init(double dt_, double zc_, double g_=9.81, double Qe=1.0, double R=1e-6, double Tprev=1.6){
    dt=dt_; zc=zc_; g=g_;
    A << 1,dt,dt*dt/2,  0,1,dt,  0,0,1;
    B << dt*dt*dt/6, dt*dt/2, dt;
    C << 1, 0, -zc/g;
    // DARE(A,B,C'Qe C,R) — Riccati 반복
    Matrix3d Q = C.transpose()*Qe*C;
    Matrix3d K = Q;
    for(int it=0; it<20000; ++it){
      double denom = R + (B.transpose()*K*B)(0,0);
      Matrix3d Kn = A.transpose()*K*A
                    - (A.transpose()*K*B)*(B.transpose()*K*A)/denom + Q;
      if((Kn-K).cwiseAbs().maxCoeff() < 1e-11){ K=Kn; break; }
      K=Kn;
    }
    double denom = R + (B.transpose()*K*B)(0,0);
    F = (B.transpose()*K*A)/denom;                 // 1x3
    Matrix3d Ac = A - B*F;
    N = (int)std::round(Tprev/dt);
    Gp.assign(N,0.0);
    Vector3d tmp = C.transpose()*Qe;               // (Ac')^0 C' Qe
    for(int j=0;j<N;++j){ Gp[j] = (B.transpose()*tmp)(0,0)/denom; tmp = Ac.transpose()*tmp; }
    sx.setZero(); sy.setZero();
  }
  void reset(double cx,double cy){ sx<<cx,0,0; sy<<cy,0,0; }
  void set_state(double cx,double vx,double cy,double vy){ sx<<cx,vx,sx(2); sy<<cy,vy,sy(2); }  // 실제 CoM 재동기(closed-loop)

  // 미래 ZMP 레퍼런스(px[N], py[N])로 CoM ref 한 스텝 전진. c/v 반환.
  void step(const double* px,const double* py, double&cx,double&vx,double&cy,double&vy){
    double ffx=0, ffy=0; for(int j=0;j<N;++j){ ffx+=Gp[j]*px[j]; ffy+=Gp[j]*py[j]; }
    double ux = -(F*sx)(0,0) + ffx;
    double uy = -(F*sy)(0,0) + ffy;
    cx=sx(0); vx=sx(1); cy=sy(0); vy=sy(1);
    sx = A*sx + B*ux;  sy = A*sy + B*uy;
  }
  double zmp_x(){ return (C*sx)(0,0); }
  double zmp_y(){ return (C*sy)(0,0); }
};
