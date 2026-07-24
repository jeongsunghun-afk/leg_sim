// CI-MPC C++ 5단계: receding-horizon MPC (ci_mpc_walk.py 포트). FDDP를 폐루프로 감싸 걸음 창발.
//   매 제어스텝: FDDP 재풀이(warm-start·iters 적게) → apply first control(u0+K0·δx) → sim 1노드 전진 → shift.
//   planner·sim 모두 relaxed(step_relaxed, 논문판 ρD 기본). 후속=step_kkt hard sim + PD+FF fine추종.
//   실행: env VX N DT NSUB ITERS MPC_STEPS CF AIR_W W_BASE VXVEL GAP_W. QPOS_OUT은 후속(뷰어).
//   ★검증(N40·ITERS10·CF1000·VXVEL120): 폐루프 안정(base_z 0.39·낙상0)·전진 0.035m(0.07m/s)=Python
//     ci_mpc_walk "약한 CF=안정하나 스텝안함(~0.027m)" 영역과 동일. 발lift≈0=슬라이딩/lean(planted 발론
//     lean까지만, 전진엔 스텝 필요). ★깨끗한 스텝=관절 gait 참조 필요(Python C-2도 foot-slip만으론 슬라이딩,
//     트롯 관절참조+강가중으로 스텝 달성). receding MPC 폐루프 머신러리는 완성·검증.
//   ★관절 gait 참조 배선(GAIT=1·W_JOINT·GAIT_T·STEP_H·ik_feet 트롯테이블): 발스케줄 IK로 관절참조가
//     발 lift를 요구·강가중 추종. BUT 실측 발lift≈0(footmin<0)=여전히 안 뜸. ★근본원인=relaxed forward가
//     4발 항상 접촉(active={0,1,2,3} bilateral clamping·ρD도 λⁿ>0 floor)이라 발이 지면서 못 떨어짐.
//     gait 참조 인프라는 준비됐으나 **진짜 스텝=step_kkt(hard active-set: 발이 접촉 떠남) 필수**(메모리 재확인).
//   후속: ①step_kkt hard active-set forward 포팅(발 lift/land) → gait참조로 스텝 ②40Hz 실시간.
#include "ci_dyn.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
using namespace cimpc;
using Eigen::VectorXd; using Eigen::MatrixXd;
static double envd(const char*k,double d){ const char*v=std::getenv(k); return v?std::atof(v):d; }
static int    envi(const char*k,int d){ const char*v=std::getenv(k); return v?std::atoi(v):d; }

// ── FDDP 1회 풀이(warm-start). Xq,Xv,U 갱신·K[0] 반환 ──
static MatrixXd solve_fddp(CiDyn&ci,std::vector<VectorXd>&Xq,std::vector<VectorXd>&Xv,std::vector<VectorXd>&U,
    std::vector<VectorXd>&Rq,std::vector<VectorXd>&Rv,int N,double DT,int NSUB,
    double GAP_W,double REG,int ITERS,const MatrixXd&Qx,const MatrixXd&Qf,const MatrixXd&Ru){
  int nv=ci.nv,nu=ci.nu; MatrixXd K0=MatrixXd::Zero(nu,2*nv);
  auto eval=[&](std::vector<VectorXd>&Xq,std::vector<VectorXd>&Xv,std::vector<VectorXd>&U,std::vector<VectorXd>&gaps,double&J,double&M){
    gaps.assign(N+1,VectorXd::Zero(2*nv)); J=0; double gs=0;
    for(int k=0;k<N;k++){ VectorXd qn,vn; ci.step_relaxed(Xq[k],Xv[k],U[k],DT,qn,vn,NSUB);
      gaps[k+1]=ci.sdiff(Xq[k+1],Xv[k+1],qn,vn); gs+=gaps[k+1].norm();
      VectorXd e=ci.sdiff(Rq[k],Rv[k],Xq[k],Xv[k]); J+=0.5*e.dot(Qx*e)+0.5*U[k].dot(Ru*U[k])+ci.foot_val(Xq[k],Xv[k]); }
    VectorXd e=ci.sdiff(Rq[N],Rv[N],Xq[N],Xv[N]); J+=0.5*e.dot(Qf*e); M=J+GAP_W*gs; };
  std::vector<VectorXd> gaps; double J,M; eval(Xq,Xv,U,gaps,J,M);
  for(int it=0;it<ITERS;it++){
    std::vector<MatrixXd> As(N),Bs(N);
    for(int k=0;k<N;k++){ MatrixXd A,B; ci.lin_AB_multi(Xq[k],Xv[k],U[k],DT,NSUB,A,B); As[k]=A; Bs[k]=B; }
    VectorXd e=ci.sdiff(Rq[N],Rv[N],Xq[N],Xv[N]); VectorXd Vx=Qf*e; MatrixXd Vxx=Qf;
    std::vector<MatrixXd> Ks(N); std::vector<VectorXd> ks(N);
    for(int k=N-1;k>=0;k--){
      VectorXd Vxp=Vx+Vxx*gaps[k+1];
      VectorXd ek=ci.sdiff(Rq[k],Rv[k],Xq[k],Xv[k]); VectorXd lx=Qx*ek, lu=Ru*U[k];
      double fc; VectorXd fg; MatrixXd fH; ci.foot_slip_cost(Xq[k],Xv[k],fc,fg,fH);
      lx+=fg; MatrixXd Qxx0=Qx+fH;
      MatrixXd&A=As[k]; MatrixXd&B=Bs[k];
      VectorXd Qx_=lx+A.transpose()*Vxp, Qu_=lu+B.transpose()*Vxp;
      MatrixXd Qxx=Qxx0+A.transpose()*Vxx*A, Quu=Ru+B.transpose()*Vxx*B, Qux=B.transpose()*Vxx*A;
      MatrixXd Qinv=(Quu+REG*MatrixXd::Identity(nu,nu)).inverse();
      MatrixXd K=-Qinv*Qux; VectorXd kk=-Qinv*Qu_; Ks[k]=K; ks[k]=kk;
      Vx=Qx_+K.transpose()*Quu*kk+K.transpose()*Qu_+Qux.transpose()*kk;
      Vxx=Qxx+K.transpose()*Quu*K+K.transpose()*Qux+Qux.transpose()*K; Vxx=0.5*(Vxx+Vxx.transpose());
    }
    K0=Ks[0];
    double bestM=1e18; std::vector<VectorXd> bXq,bXv,bU; bool found=false;
    for(double alpha:{1.0,0.5,0.25,0.1,0.05,0.02,0.01}){
      std::vector<VectorXd> Xqn(N+1),Xvn(N+1),Un(N); Xqn[0]=Xq[0]; Xvn[0]=Xv[0]; bool ok=true;
      for(int k=0;k<N;k++){
        VectorXd dx=ci.sdiff(Xq[k],Xv[k],Xqn[k],Xvn[k]);
        VectorXd u=U[k]+alpha*ks[k]+Ks[k]*dx; Un[k]=u;
        VectorXd qn,vn; ci.step_relaxed(Xqn[k],Xvn[k],u,DT,qn,vn,NSUB);
        if(!qn.allFinite()){ ok=false; break; }
        VectorXd xq,xv; ci.sint(qn,vn,VectorXd(-(1.0-alpha)*gaps[k+1]),xq,xv); Xqn[k+1]=xq; Xvn[k+1]=xv;
      }
      if(!ok||!Xqn[N].allFinite()) continue;
      std::vector<VectorXd> gn; double Jn,Mn; eval(Xqn,Xvn,Un,gn,Jn,Mn);
      if(std::isfinite(Mn)&&Mn<bestM){ bestM=Mn; bXq=Xqn; bXv=Xvn; bU=Un; found=true; }
    }
    if(!found||bestM>=M*0.999999) break;
    Xq=bXq; Xv=bXv; U=bU; eval(Xq,Xv,U,gaps,J,M);
  }
  return K0;
}

int main(){
  setvbuf(stdout,nullptr,_IONBF,0);
  CiDyn ci("/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf");
  int nv=ci.nv, nu=ci.nu;
  double DT=envd("DT",0.01), REG=envd("REG",0.1), GAP_W=envd("GAP_W",50.0);
  int N=envi("N",30), NSUB=envi("NSUB",10), ITERS=envi("ITERS",5), STEPS=envi("MPC_STEPS",120);
  double VX=envd("VX",0.3);
  ci.CF=envd("CF",2000.0); ci.C1S=envd("C1S",-30.0); ci.AIR_W=envd("AIR_W",100.0);
  VectorXd Qxd(2*nv); Qxd.head(nv).setConstant(20.0); Qxd.tail(nv).setConstant(1.0);
  MatrixXd Qx=Qxd.asDiagonal();
  if(VX>0){ Qx(0,0)*=envd("VXPOS",0.1); Qx(nv,nv)*=envd("VXVEL",60.0);
    double WB=envd("W_BASE",5.0); Qx(2,2)*=WB; Qx(3,3)*=WB; Qx(4,4)*=WB; }
  double WJ=envd("W_JOINT",20.0); Qx.diagonal().segment(6,16).setConstant(WJ);   // ★다리 관절 gait 추종(스텝 강제)
  MatrixXd Qf=Qx*20.0, Ru=MatrixXd::Identity(nu,nu)*1e-3;

  VectorXd qstar=ci.stance_q(), vstar=VectorXd::Zero(nv);
  // ★트롯 gait 참조: 발스케줄(대각쌍 FL/HR·FR/HL 교대, stance 후방sweep·swing 전방arc) IK → phase→관절 테이블
  double GAIT=envd("GAIT",VX>0?1.0:0.0), GT=envd("GAIT_T",0.4), STEP_H=envd("STEP_H",0.05), BZ=envd("BASE_Z",0.40);
  double stride=VX*GT*0.5, goff[4]={0.0,0.5,0.5,0.0};
  std::vector<Vector3d> gnom={{0.30,0.16,0.0},{0.30,-0.16,0.0},{-0.30,0.16,0.0},{-0.30,-0.16,0.0}};
  const int MT=40; std::vector<VectorXd> gtab(MT);
  for(int m=0;m<MT;m++){ double ph=(double)m/MT; std::vector<Vector3d> t(4);
    for(int i=0;i<4;i++){ double pi=std::fmod(ph+goff[i],1.0); Vector3d n=gnom[i];
      if(pi<0.5) n[0]+=stride*(0.5-pi/0.5);
      else{ double sw=(pi-0.5)/0.5; n[0]+=stride*(sw-0.5); n[2]=STEP_H*std::sin(M_PI*sw); }
      t[i]=n; }
    gtab[m]=ci.ik_feet(t,BZ); }
  auto gref=[&](double ph)->VectorXd{ ph=std::fmod(ph,1.0); if(ph<0)ph+=1; return gtab[((int)(ph*MT))%MT]; };
  VectorXd q=qstar, v=VectorXd::Zero(nv), tau_hold=VectorXd::Zero(nu);
  for(int i=0;i<200;i++){ tau_hold=150.0*(qstar.tail(nu)-q.tail(nu))-8.0*v.tail(nu);
    VectorXd qn,vn; ci.step_soft(q,v,tau_hold,DT*0.1,1,qn,vn); q=qn; v=vn; }
  double x0=q[0];
  std::vector<VectorXd> U(N,tau_hold);
  std::printf("[C++ receding MPC] N=%d dt=%.3f %dHz재풀이 VX=%.2f CF=%.0f relax=%s\n",N,DT,(int)(1.0/DT),VX,ci.CF,ci.relax_mode.c_str());
  auto footmin=[&](const VectorXd&q,const VectorXd&v){ std::vector<double>phi;std::vector<MatrixXd>J;std::vector<Vector3d>vf;
    ci.foot_kin(q,v,phi,J,vf); double mn=1e9; for(int i=0;i<4;i++)mn=std::min(mn,phi[i]); return mn; };

  double liftmax=0;
  for(int s=0;s<STEPS;s++){
    std::vector<VectorXd> Rq(N+1),Rv(N+1); double phase=s*DT/GT;
    for(int k=0;k<=N;k++){ Rq[k]= GAIT>0.5 ? gref(phase+k*DT/GT) : qstar;   // ★gait 관절참조 or 서기
      Rq[k][0]=q[0]+VX*k*DT; Rv[k]=vstar; Rv[k][0]=VX; }
    std::vector<VectorXd> Xq(N+1),Xv(N+1); Xq[0]=q; Xv[0]=v;
    for(int k=1;k<=N;k++){ Xq[k]=Rq[k]; Xv[k]=Rv[k]; }
    MatrixXd K0=solve_fddp(ci,Xq,Xv,U,Rq,Rv,N,DT,NSUB,GAP_W,REG,ITERS,Qx,Qf,Ru);
    VectorXd u0=U[0];                                   // δx=0(x_actual=Xq[0]) → K0항 생략
    VectorXd qn,vn; ci.step_relaxed(q,v,u0,DT,qn,vn,NSUB); q=qn; v=vn;   // apply first control, 1노드 전진
    if(!q.allFinite()){ std::printf("  ✗ 발산 step %d\n",s+1); break; }
    liftmax=std::max(liftmax,footmin(q,v));
    for(int k=0;k<N-1;k++) U[k]=U[k+1]; U[N-1]=tau_hold;   // warm-start shift
    if((s+1)%15==0) std::printf("  step %3d t=%.2fs 전진=%.3fm base_z=%.3f vx=%.2f footmin=%.3f\n",
                                s+1,(s+1)*DT,q[0]-x0,q[2],v[0],footmin(q,v));
  }
  std::printf("  최종: %.2fs 전진 %.3fm(평균 %.2f m/s·목표 %.2f) base_z=%.3f 발lift최대=%.3f  %s\n",
              STEPS*DT,q[0]-x0,(q[0]-x0)/(STEPS*DT),VX,q[2],liftmax,
              q[2]>0.30?"✅ 전진(균형유지)":"△ 균형약함/붕괴");
  return 0;
}
