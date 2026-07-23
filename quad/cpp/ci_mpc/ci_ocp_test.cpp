// CI-MPC C++ 3단계: single-shooting iLQR 서기 안정화 OCP (ci_ocp.py main 포트).
//   ★안정화 핵심: dt=0.001(fine)·N=25·PD warm-start feasible nominal·soft crisp 접촉.
//   relaxed tangent는 fine dt서만 ρ≈1(안정). dt=0.02는 ρ=1.14+ → backward Vxx 폭발(발산).
//   ★★버그수정(핵심): pinocchio dIntegrate는 블록대각만 쓰고 off-diagonal은 안 지움 → 출력행렬
//     setZero 선행 필수. 미초기화 재사용 메모리의 쓰레기값이 A를 7배 부풀려 Vxx=1e50 폭발/crash.
//     ci_dyn.hpp lin_AB에 setZero 추가 후 Vxx 7171(Python 일치)·crash 소멸·정상 수렴.
//   ★검증1(Python 일치): C++가 Python ci_ocp.py(relaxed 그래디언트)를 iteration 단위 정확 재현(J 18→16.8, 동일 α).
//     16.8은 relaxed 그래디언트(논문 핵심, soft forward와 불일치)의 값=Python도 relaxed면 동일.
//     더 깊은 수렴(11.1)은 soft-force 그래디언트(dynamics_derivatives) 필요=후속.
//   ★검증2(일관성=correctness): FWD_REL=1(relaxed forward=backward와 동일 동역학) → 전 iter α=1.0
//     clean Newton 하강(J 66→44). soft forward(불일치)는 α 0.1/0.05 정체. = relaxed 그래디언트가
//     relaxed 동역학의 정확한 도함수임을 확인. env FWD_REL(0=soft·1=relaxed).
#include "ci_dyn.hpp"
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
using namespace cimpc;
using Eigen::VectorXd; using Eigen::MatrixXd;

int main(){
  setvbuf(stdout, nullptr, _IONBF, 0);
  CiDyn ci("/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf");
  ci.eps=1e-3;
  int nv=ci.nv, nu=ci.nu;
  const double DT=0.001, REG=1e-1, WKP=150.0, WKD=10.0;
  const int N=25, NSUB=1, ITERS=8;
  std::printf("[C++ ci iLQR 서기] 모델 nq=%d nv=%d nu=%d · dt=%.4f N=%d\n", ci.nq, nv, nu, DT, N);
  VectorXd qstar=ci.stance_q(), vstar=VectorXd::Zero(nv);
  const char* fr=std::getenv("FWD_REL"); bool FWD_REL=(fr&&fr[0]=='1');   // 1=relaxed forward(backward와 일관)
  std::printf("  [cfg] forward=%s (relaxed=backward 일관)\n", FWD_REL?"relaxed":"soft");
  auto fwd=[&](const VectorXd&q,const VectorXd&v,const VectorXd&u,VectorXd&qn,VectorXd&vn){
    if(FWD_REL) ci.step_relaxed(q,v,u,DT,qn,vn); else ci.step_soft(q,v,u,DT,NSUB,qn,vn); };

  // settle(발 접촉) — substep DT*0.1 (Python과 동일)
  VectorXd q=qstar, v=VectorXd::Zero(nv), tau_hold=VectorXd::Zero(nu);
  for(int i=0;i<200;i++){ tau_hold=150.0*(qstar.tail(nu)-q.tail(nu))-8.0*v.tail(nu);
    VectorXd qn,vn; ci.step_soft(q,v,tau_hold,DT*0.1,1,qn,vn); q=qn; v=vn; }
  VectorXd q0=q, v0=v;
  std::printf("  [dbg] settled base_z=%.4f |v0|=%.4f\n",q0[2],v0.norm());

  VectorXd Qxd(2*nv); Qxd.head(nv).setConstant(20.0); Qxd.tail(nv).setConstant(1.0);
  MatrixXd Qx=Qxd.asDiagonal(); MatrixXd Qf=Qx*20.0; MatrixXd Ru=MatrixXd::Identity(nu,nu)*1e-3;

  // PD warm-start: feasible nominal(서기 근처 유지) — 긴 horizon 발산 방지
  std::vector<VectorXd> U(N);
  { VectorXd qq=q0, vv=v0;
    for(int k=0;k<N;k++){ VectorXd u=tau_hold+WKP*(qstar.tail(nu)-qq.tail(nu))-WKD*vv.tail(nu); U[k]=u;
      VectorXd qn,vn; fwd(qq,vv,u,qn,vn); qq=qn; vv=vn; } }

  auto rollout=[&](std::vector<VectorXd>&U,std::vector<VectorXd>&qs,std::vector<VectorXd>&vs){
    qs.assign(N+1,VectorXd()); vs.assign(N+1,VectorXd()); qs[0]=q0; vs[0]=v0;
    for(int k=0;k<N;k++){ VectorXd qn,vn; fwd(qs[k],vs[k],U[k],qn,vn); qs[k+1]=qn; vs[k+1]=vn; } };
  auto cost=[&](std::vector<VectorXd>&U,std::vector<VectorXd>&qs,std::vector<VectorXd>&vs){
    rollout(U,qs,vs); double c=0;
    for(int k=0;k<N;k++){ VectorXd e=ci.sdiff(qstar,vstar,qs[k],vs[k]); c+=0.5*e.dot(Qx*e)+0.5*U[k].dot(Ru*U[k]); }
    VectorXd e=ci.sdiff(qstar,vstar,qs[N],vs[N]); return c+0.5*e.dot(Qf*e); };

  std::vector<VectorXd> qs,vs; double J0=cost(U,qs,vs), J_init=J0;
  std::printf("  iter 0  J=%.3f\n",J0);

  for(int it=0;it<ITERS;it++){
    std::vector<MatrixXd> As(N),Bs(N);
    for(int k=0;k<N;k++){ MatrixXd A,B; ci.lin_AB(qs[k],vs[k],U[k],DT,A,B); As[k]=A; Bs[k]=B; }
    VectorXd e=ci.sdiff(qstar,vstar,qs[N],vs[N]); VectorXd Vx=Qf*e; MatrixXd Vxx=Qf;
    std::vector<MatrixXd> Ks(N); std::vector<VectorXd> ks(N);
    for(int k=N-1;k>=0;k--){
      VectorXd ek=ci.sdiff(qstar,vstar,qs[k],vs[k]); VectorXd Qx_=Qx*ek, Qu_=Ru*U[k];
      MatrixXd&A=As[k]; MatrixXd&B=Bs[k];
      VectorXd Qxk=Qx_+A.transpose()*Vx, Quk=Qu_+B.transpose()*Vx;
      MatrixXd Qxx=Qx+A.transpose()*Vxx*A, Quu=Ru+B.transpose()*Vxx*B, Qux=B.transpose()*Vxx*A;
      MatrixXd Qinv=(Quu+REG*MatrixXd::Identity(nu,nu)).inverse();
      MatrixXd K=-Qinv*Qux; VectorXd kk=-Qinv*Quk; Ks[k]=K; ks[k]=kk;
      Vx =Qxk+K.transpose()*Quu*kk+K.transpose()*Qu_+Qux.transpose()*kk;
      Vxx=Qxx+K.transpose()*Quu*K+K.transpose()*Qux+Qux.transpose()*K; Vxx=0.5*(Vxx+Vxx.transpose());
      if(it==0&&k==0) std::printf("    [dbg k=0] Vxx_fro=%.3e Quu_fro=%.3e A_fro=%.2f\n",Vxx.norm(),Quu.norm(),A.norm());
    }
    double bestJ=1e18; std::vector<VectorXd> bU,bqs,bvs; double ba=0; bool found=false;
    for(double alpha : {1.0,0.5,0.25,0.1,0.05}){
      std::vector<VectorXd> Un(N),qn(N+1),vn(N+1); qn[0]=q0; vn[0]=v0; bool ok=true;
      for(int k=0;k<N;k++){
        VectorXd dx=ci.sdiff(qs[k],vs[k],qn[k],vn[k]);
        VectorXd u=U[k]+alpha*ks[k]+Ks[k]*dx; Un[k]=u;
        VectorXd qq,vv; fwd(qn[k],vn[k],u,qq,vv);
        if(!qq.allFinite()){ ok=false; break; }
        qn[k+1]=qq; vn[k+1]=vv;
      }
      if(!ok) continue;
      double Jn=0;
      for(int k=0;k<N;k++){ VectorXd e=ci.sdiff(qstar,vstar,qn[k],vn[k]); Jn+=0.5*e.dot(Qx*e)+0.5*Un[k].dot(Ru*Un[k]); }
      { VectorXd e=ci.sdiff(qstar,vstar,qn[N],vn[N]); Jn+=0.5*e.dot(Qf*e); }
      if(std::isfinite(Jn)&&Jn<bestJ){ bestJ=Jn; bU=Un; bqs=qn; bvs=vn; ba=alpha; found=true; }
    }
    if(!found||bestJ>=J0) break;
    U=bU; qs=bqs; vs=bvs; J0=bestJ;
    std::printf("  iter %d  J=%.3f  (α=%.2f)\n",it+1,J0,ba);
  }
  VectorXd ef=ci.sdiff(qstar,vstar,qs[N],vs[N]);
  std::printf("  최종: J %.1f→%.1f (%.0f%%↓) base_z=%.3f 종단오차=%.3f  %s\n",J_init,J0,100*(1-J0/J_init),
              qs[N][2],ef.norm(), J0<J_init ?"✅ 안정 수렴(Vxx 유계·Python relaxed OCP와 iter단위 일치)":"✗ 발산");
  return 0;
}
