// CI-MPC C++ 3단계: multiple-shooting FDDP 서기 OCP (ci_ocp_ms 포트). relaxed 그래디언트 사용.
//   gap 주입 backward + feasibility-driven forward + merit. gap 폐쇄(feasible) 수렴 검증.
#include "ci_dyn.hpp"
#include <cstdio>
#include <vector>
#include <cmath>
using namespace cimpc;
using Eigen::VectorXd; using Eigen::MatrixXd;

int main(){
  CiDyn ci("/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf");
  int nv=ci.nv, nu=ci.nu; ci.eps=1e-3;
  const double DT=0.005, GAP_W=100.0, REG=1e-1; const int N=25, NSUB=10, ITERS=25;
  VectorXd qstar=ci.stance_q(), vstar=VectorXd::Zero(nv);

  VectorXd q=qstar, v=VectorXd::Zero(nv), tau_hold=VectorXd::Zero(nu);   // settle
  for(int i=0;i<200;i++){ tau_hold=150.0*(qstar.tail(nu)-q.tail(nu))-8.0*v.tail(nu);
    VectorXd qn,vn; ci.step_relaxed(q,v,tau_hold,DT*0.05,qn,vn); q=qn; v=vn; }
  VectorXd q0=q, v0=v;

  VectorXd Qxd(2*nv); Qxd.head(nv).setConstant(20.0); Qxd.tail(nv).setConstant(1.0);
  MatrixXd Qx=Qxd.asDiagonal(); MatrixXd Qf=Qx*10.0; MatrixXd Ru=MatrixXd::Identity(nu,nu)*1e-3;

  std::vector<VectorXd> Xq(N+1),Xv(N+1),U(N);
  Xq[0]=q0; Xv[0]=v0; for(int k=1;k<=N;k++){ Xq[k]=qstar; Xv[k]=vstar; }
  for(int k=0;k<N;k++) U[k]=tau_hold;

  auto evalt=[&](std::vector<VectorXd>&Xq,std::vector<VectorXd>&Xv,std::vector<VectorXd>&U,
                 std::vector<VectorXd>&gaps,double&J,double&M){
    gaps.assign(N+1,VectorXd::Zero(2*nv)); J=0; double gs=0;
    for(int k=0;k<N;k++){ VectorXd qn,vn; ci.step_relaxed(Xq[k],Xv[k],U[k],DT,qn,vn);
      gaps[k+1]=ci.sdiff(Xq[k+1],Xv[k+1],qn,vn); gs+=gaps[k+1].norm();
      VectorXd e=ci.sdiff(qstar,vstar,Xq[k],Xv[k]); J+=0.5*e.dot(Qx*e)+0.5*U[k].dot(Ru*U[k]); }
    VectorXd e=ci.sdiff(qstar,vstar,Xq[N],Xv[N]); J+=0.5*e.dot(Qf*e); M=J+GAP_W*gs;
  };
  std::vector<VectorXd> gaps; double J,M; evalt(Xq,Xv,U,gaps,J,M);
  double g0=0; for(int k=1;k<=N;k++) g0=std::max(g0,gaps[k].norm());
  std::printf("[C++ FDDP 서기] iter 0  J=%.3f merit=%.3f |gap|max=%.3f\n",J,M,g0);

  for(int it=0;it<ITERS;it++){
    std::vector<MatrixXd> As(N),Bs(N);
    for(int k=0;k<N;k++){ MatrixXd A,B; ci.lin_AB(Xq[k],Xv[k],U[k],DT,A,B); As[k]=A; Bs[k]=B; }
    VectorXd e=ci.sdiff(qstar,vstar,Xq[N],Xv[N]); VectorXd Vx=Qf*e; MatrixXd Vxx=Qf;
    std::vector<MatrixXd> Ks(N); std::vector<VectorXd> ks(N);
    for(int k=N-1;k>=0;k--){
      VectorXd Vxp=Vx+Vxx*gaps[k+1];
      VectorXd ek=ci.sdiff(qstar,vstar,Xq[k],Xv[k]); VectorXd lx=Qx*ek, lu=Ru*U[k];
      MatrixXd&A=As[k]; MatrixXd&B=Bs[k];
      VectorXd Qx_=lx+A.transpose()*Vxp, Qu_=lu+B.transpose()*Vxp;
      MatrixXd Qxx=Qx+A.transpose()*Vxx*A, Quu=Ru+B.transpose()*Vxx*B, Qux=B.transpose()*Vxx*A;
      MatrixXd Qinv=(Quu+REG*MatrixXd::Identity(nu,nu)).inverse();
      MatrixXd K=-Qinv*Qux; VectorXd kk=-Qinv*Qu_; Ks[k]=K; ks[k]=kk;
      Vx=Qx_+K.transpose()*Quu*kk+K.transpose()*Qu_+Qux.transpose()*kk;
      Vxx=Qxx+K.transpose()*Quu*K+K.transpose()*Qux+Qux.transpose()*K; Vxx=0.5*(Vxx+Vxx.transpose());
    }
    double bestM=1e18; std::vector<VectorXd> bXq,bXv,bU; bool found=false;
    for(double alpha : {1.0,0.5,0.25,0.1,0.05}){
      std::vector<VectorXd> Xqn(N+1),Xvn(N+1),Un(N); Xqn[0]=Xq[0]; Xvn[0]=Xv[0]; bool ok=true;
      for(int k=0;k<N;k++){
        VectorXd dx=ci.sdiff(Xq[k],Xv[k],Xqn[k],Xvn[k]);
        VectorXd u=U[k]+alpha*ks[k]+Ks[k]*dx; Un[k]=u;
        VectorXd qn,vn; ci.step_relaxed(Xqn[k],Xvn[k],u,DT,qn,vn);
        if(!qn.allFinite()){ ok=false; break; }
        VectorXd xq,xv; ci.sint(qn,vn,VectorXd(-(1.0-alpha)*gaps[k+1]),xq,xv); Xqn[k+1]=xq; Xvn[k+1]=xv;
      }
      std::vector<VectorXd> gn; double Jn,Mn; evalt(Xqn,Xvn,Un,gn,Jn,Mn);
      if(std::isfinite(Mn)&&Mn<bestM){ bestM=Mn; bXq=Xqn; bXv=Xvn; bU=Un; found=true; }
    }
    if(!found||bestM>=M*0.999999) break;
    Xq=bXq; Xv=bXv; U=bU; evalt(Xq,Xv,U,gaps,J,M);
    double gmax=0; for(int k=1;k<=N;k++) gmax=std::max(gmax,gaps[k].norm());
    std::printf("  iter %d  J=%.3f merit=%.3f |gap|max=%.4f\n",it+1,J,M,gmax);
  }
  double gmax=0; for(int k=1;k<=N;k++) gmax=std::max(gmax,gaps[k].norm());
  std::printf("  최종: |gap|max=%.4f(초기 %.1f) base_z=%.3f  %s\n",gmax,g0,Xq[N][2],
              gmax<0.05?"✅ FDDP 수렴(gap 폐쇄=feasible)":"△");
  return 0;
}
