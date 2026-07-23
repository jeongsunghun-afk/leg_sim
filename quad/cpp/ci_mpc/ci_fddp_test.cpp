// CI-MPC C++ 4단계: multiple-shooting FDDP (ci_ocp_ms.py 포트). 서기 검증(gap 폐쇄).
//   각 노드 상태=결정변수 + gap(dynamics defect)이 물리 불일치 흡수 → nominal=전 노드 서기
//   (open-loop 발산 원천 제거). gap 주입 backward + feasibility-driven forward + merit=J+GAP_W·Σ|gap|.
//   ★setZero 수정 후 안정. forward=step_relaxed(FWD_REL=1 기본, backward와 일관) · backward=lin_AB(relaxed ρD).
//   ★검증: (a)dt0.001·nsub1·N100(0.1s) gap폐쇄·종단오차0.338. (b)★multi-rate(lin_AB_multi·step_relaxed nsub)
//     dt0.01·nsub10·N50(0.5s=5배 horizon) gap 완전폐쇄(0.0000)·전 iter α=1.0·종단오차 0.184(더 우수).
//     ★consistency 필수: soft forward(FWD_REL=0)+relaxed backward는 gap 미폐쇄(정체). relaxed 일관=수렴.
//   후속: VX 전진 reference + foot-slip cost(HOUND eq22) = 보행 창발.
#include "ci_dyn.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
using namespace cimpc;
using Eigen::VectorXd; using Eigen::MatrixXd;
static double envd(const char*k,double d){ const char*v=std::getenv(k); return v?std::atof(v):d; }
static int    envi(const char*k,int d){ const char*v=std::getenv(k); return v?std::atoi(v):d; }

int main(){
  setvbuf(stdout,nullptr,_IONBF,0);
  CiDyn ci("/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf");
  int nv=ci.nv, nu=ci.nu;
  double DT=envd("DT",0.01), REG=envd("REG",0.1), GAP_W=envd("GAP_W",50.0);
  int N=envi("N",50), NSUB=envi("NSUB",10), ITERS=envi("ITERS",30);   // dt0.01·N50·nsub10=0.5s(multi-rate)
  std::printf("[C++ FDDP 서기] nq=%d nv=%d nu=%d · dt=%.4f N=%d nsub=%d relax=%s\n",
              ci.nq,nv,nu,DT,N,NSUB,ci.relax_mode.c_str());
  bool FWD_REL=envi("FWD_REL",1);   // 1=relaxed forward(backward와 일관, 기본)·0=soft
  std::printf("  [cfg] forward=%s\n", FWD_REL?"relaxed(backward 일관)":"soft");
  auto fwd=[&](const VectorXd&q,const VectorXd&v,const VectorXd&u,VectorXd&qn,VectorXd&vn){
    if(FWD_REL) ci.step_relaxed(q,v,u,DT,qn,vn,NSUB); else ci.step_soft(q,v,u,DT,NSUB,qn,vn); };
  VectorXd qstar=ci.stance_q(), vstar=VectorXd::Zero(nv);
  VectorXd q=qstar, v=VectorXd::Zero(nv), tau_hold=VectorXd::Zero(nu);
  for(int i=0;i<200;i++){ tau_hold=150.0*(qstar.tail(nu)-q.tail(nu))-8.0*v.tail(nu);
    VectorXd qn,vn; ci.step_soft(q,v,tau_hold,DT*0.1,1,qn,vn); q=qn; v=vn; }
  VectorXd q0=q, v0=v;

  VectorXd Qxd(2*nv); Qxd.head(nv).setConstant(20.0); Qxd.tail(nv).setConstant(1.0);
  MatrixXd Qx=Qxd.asDiagonal(); MatrixXd Qf=Qx*20.0; MatrixXd Ru=MatrixXd::Identity(nu,nu)*1e-3;

  // ★nominal: 노드0=settle, 나머지=서기(qstar). gap이 불일치 흡수 → 발산 안함
  std::vector<VectorXd> Xq(N+1),Xv(N+1),U(N);
  Xq[0]=q0; Xv[0]=v0; for(int k=1;k<=N;k++){ Xq[k]=qstar; Xv[k]=vstar; }
  for(int k=0;k<N;k++) U[k]=tau_hold;

  auto eval=[&](std::vector<VectorXd>&Xq,std::vector<VectorXd>&Xv,std::vector<VectorXd>&U,
                std::vector<VectorXd>&gaps,double&J,double&M){
    gaps.assign(N+1,VectorXd::Zero(2*nv)); J=0; double gs=0;
    for(int k=0;k<N;k++){ VectorXd qn,vn; fwd(Xq[k],Xv[k],U[k],qn,vn);
      gaps[k+1]=ci.sdiff(Xq[k+1],Xv[k+1],qn,vn); gs+=gaps[k+1].norm();
      VectorXd e=ci.sdiff(qstar,vstar,Xq[k],Xv[k]); J+=0.5*e.dot(Qx*e)+0.5*U[k].dot(Ru*U[k]); }
    VectorXd e=ci.sdiff(qstar,vstar,Xq[N],Xv[N]); J+=0.5*e.dot(Qf*e); M=J+GAP_W*gs; };
  auto gmaxf=[&](std::vector<VectorXd>&g){ double x=0; for(int k=1;k<=N;k++) x=std::max(x,g[k].norm()); return x; };

  std::vector<VectorXd> gaps; double J,M; eval(Xq,Xv,U,gaps,J,M);
  double g0=gmaxf(gaps), J_init=J;
  std::printf("  iter 0  J=%.3f merit=%.3f |gap|max=%.3f\n",J,M,g0);

  for(int it=0;it<ITERS;it++){
    std::vector<MatrixXd> As(N),Bs(N);
    for(int k=0;k<N;k++){ MatrixXd A,B; ci.lin_AB_multi(Xq[k],Xv[k],U[k],DT,NSUB,A,B); As[k]=A; Bs[k]=B; }
    VectorXd e=ci.sdiff(qstar,vstar,Xq[N],Xv[N]); VectorXd Vx=Qf*e; MatrixXd Vxx=Qf;
    std::vector<MatrixXd> Ks(N); std::vector<VectorXd> ks(N);
    for(int k=N-1;k>=0;k--){
      VectorXd Vxp=Vx+Vxx*gaps[k+1];                       // ★gap 주입
      VectorXd ek=ci.sdiff(qstar,vstar,Xq[k],Xv[k]); VectorXd lx=Qx*ek, lu=Ru*U[k];
      MatrixXd&A=As[k]; MatrixXd&B=Bs[k];
      VectorXd Qx_=lx+A.transpose()*Vxp, Qu_=lu+B.transpose()*Vxp;
      MatrixXd Qxx=Qx+A.transpose()*Vxx*A, Quu=Ru+B.transpose()*Vxx*B, Qux=B.transpose()*Vxx*A;
      MatrixXd Qinv=(Quu+REG*MatrixXd::Identity(nu,nu)).inverse();
      MatrixXd K=-Qinv*Qux; VectorXd kk=-Qinv*Qu_; Ks[k]=K; ks[k]=kk;
      Vx=Qx_+K.transpose()*Quu*kk+K.transpose()*Qu_+Qux.transpose()*kk;
      Vxx=Qxx+K.transpose()*Quu*K+K.transpose()*Qux+Qux.transpose()*K; Vxx=0.5*(Vxx+Vxx.transpose());
    }
    double bestM=1e18; std::vector<VectorXd> bXq,bXv,bU; double ba=0; bool found=false;
    for(double alpha:{1.0,0.5,0.25,0.1,0.05,0.02,0.01}){     // ★feasibility-driven forward(gap 수축)
      std::vector<VectorXd> Xqn(N+1),Xvn(N+1),Un(N); Xqn[0]=Xq[0]; Xvn[0]=Xv[0]; bool ok=true;
      for(int k=0;k<N;k++){
        VectorXd dx=ci.sdiff(Xq[k],Xv[k],Xqn[k],Xvn[k]);
        VectorXd u=U[k]+alpha*ks[k]+Ks[k]*dx; Un[k]=u;
        VectorXd qn,vn; fwd(Xqn[k],Xvn[k],u,qn,vn);
        if(!qn.allFinite()){ ok=false; break; }
        VectorXd xq,xv; ci.sint(qn,vn,VectorXd(-(1.0-alpha)*gaps[k+1]),xq,xv); Xqn[k+1]=xq; Xvn[k+1]=xv;
      }
      if(!ok||!Xqn[N].allFinite()) continue;
      std::vector<VectorXd> gn; double Jn,Mn; eval(Xqn,Xvn,Un,gn,Jn,Mn);
      if(std::isfinite(Mn)&&Mn<bestM){ bestM=Mn; bXq=Xqn; bXv=Xvn; bU=Un; ba=alpha; found=true; }
    }
    if(!found||bestM>=M*0.999999) break;
    Xq=bXq; Xv=bXv; U=bU; eval(Xq,Xv,U,gaps,J,M);
    std::printf("  iter %d  J=%.3f merit=%.3f (α=%.2f) |gap|max=%.4f\n",it+1,J,M,ba,gmaxf(gaps));
  }
  double gmax=gmaxf(gaps);
  std::printf("  최종: J=%.1f(초기%.1f) base_z=%.3f 종단오차=%.3f |gap|max=%.4f(초기%.2f)  %s\n",
              J,J_init,Xq[N][2],ci.sdiff(qstar,vstar,Xq[N],Xv[N]).norm(),gmax,g0,
              gmax<0.05?"✅ FDDP 수렴(gap 폐쇄=feasible)":gmax<g0*0.9?"△ 부분폐쇄":"✗ 미폐쇄");
  return 0;
}
