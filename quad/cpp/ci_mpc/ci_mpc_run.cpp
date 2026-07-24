// CI-MPC C++ 5단계: receding-horizon MPC (ci_mpc_walk.py 포트). FDDP를 폐루프로 감싸 걸음 창발.
//   매 제어스텝: FDDP 재풀이(warm-start·iters 적게) → apply first control(u0+K0·δx) → sim 1노드 전진 → shift.
//   planner·sim 모두 relaxed(step_relaxed, 논문판 ρD 기본). 후속=step_kkt hard sim + PD+FF fine추종.
//   실행: env VX N DT NSUB ITERS MPC_STEPS CF AIR_W W_BASE VXVEL GAP_W. QPOS_OUT은 후속(뷰어).
//   ★검증(N40·ITERS10·CF1000·VXVEL120): 폐루프 안정(base_z 0.39·낙상0)·전진 0.035m(0.07m/s)=Python
//     ci_mpc_walk "약한 CF=안정하나 스텝안함(~0.027m)" 영역과 동일. 발lift≈0=슬라이딩/lean(planted 발론
//     lean까지만, 전진엔 스텝 필요). ★깨끗한 스텝=관절 gait 참조 필요(Python C-2도 foot-slip만으론 슬라이딩,
//     트롯 관절참조+강가중으로 스텝 달성). receding MPC 폐루프 머신러리는 완성·검증.
//   ★★스텝 달성(SIM_KKT=1·PLAN_SOFT=1·GAIT=1): step_kkt(hard active-set) sim + soft planner(발 lift 계획)
//     + gait 관절참조 → **발 실제로 뜸(발lift 5.3cm)=진짜 스텝**(relaxed는 4발 접착이라 불가였음).
//     BUT 스텝 중 base 붕괴(0.40→0.21)=균형 상실("강한 CF=스텝하나 발산" Python 영역). 스텝은 됨, 남은=균형.
//   조합: planner=soft(발 lift 계획, PLAN_SOFT=1)·sim=step_kkt(발 lift 실행, SIM_KKT=1)·backward=relaxed ρ그래디언트
//     ·gait 관절참조(W_JOINT 강가중). =HOUND식 soft/fast planner + hard sim.
//   ★★균형 개선(HOUND §6.3 fine-rate PD+FF 추종, KP_T/KD_T): u0 held(0.01s) 대신 계획(Xq[1])을 h=DT/NSUB
//     ≤0.001s로 PD+FF 추종 → hard 접촉 안정. 결과: base_z 붕괴(0.21)→**유지(0.33~0.40)+발 5.3cm 스텝**
//     =안정 스텝 보행 근접(0.6s 균형유지). "제어 유지간격 짧아야 hard 안정"(메모리) 확정.
//   ★★실시간 프로파일/최적화: 병목=FDDP solve(288ms, sim step_kkt는 0.4ms만). 100% 선형화
//     (lin_AB_multi가 노드×ITERS×NSUB×44 FD기하). knob LIN_NSUB(선형화 substep)·PLAN_NSUB(planner
//     forward substep)·line-search 4알파로 축소 → **N15·ITERS3·LIN_NSUB1·PLAN_NSUB3=20.4ms/step
//     =40Hz(25ms) 실시간 달성**. 단 품질저하(전진~0·안정만). 속도-품질 트레이드오프:
//     실시간+풀품질은 FD기하→해석 그래디언트(HOUND ~70μs) 필요. 풀품질(3s보행)=96~288ms offline.
//   후속: 해석 기하 도함수(kinematic hessian)로 실시간+품질 동시 · capture-point 지속성.
//   ★험지(gap) 크로싱(GAP_X0/GAP_X1): CiDyn.in_gap로 step_kkt가 틈 위 발=지지없음(active제외)+gref_gap이
//     발판을 solid ground로 shift(재IK). baseline=걷다 gap서 앞발 빠져 붕괴(gap 물리 작동 확인). 발판회피시
//     발이 gap 너머로 과신전→vx lunge runaway→붕괴(Python perceptive-nav "gap-edge lunge"와 동일 프론티어).
//     인프라(gap접촉·발판회피)는 작동, 안정 크로싱=capture-point 동적안정화/RL 필요=연구급.
#include "ci_dyn.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <chrono>
using namespace cimpc;
static double now_ms(){ return std::chrono::duration<double,std::milli>(
  std::chrono::steady_clock::now().time_since_epoch()).count(); }
using Eigen::VectorXd; using Eigen::MatrixXd;
static double envd(const char*k,double d){ const char*v=std::getenv(k); return v?std::atof(v):d; }
static int    envi(const char*k,int d){ const char*v=std::getenv(k); return v?std::atoi(v):d; }

// ── FDDP 1회 풀이(warm-start). Xq,Xv,U 갱신·K[0] 반환 ──
static MatrixXd solve_fddp(CiDyn&ci,std::vector<VectorXd>&Xq,std::vector<VectorXd>&Xv,std::vector<VectorXd>&U,
    std::vector<VectorXd>&Rq,std::vector<VectorXd>&Rv,int N,double DT,int NSUB,
    double GAP_W,double REG,int ITERS,const MatrixXd&Qx,const MatrixXd&Qf,const MatrixXd&Ru){
  int nv=ci.nv,nu=ci.nu; MatrixXd K0=MatrixXd::Zero(nu,2*nv);
  // ★DCM(capture point) 항: ξ=x+v/ω, ω=√(g/z). 발산 모드 d=e_x+e_vx/ω를 페널티(속도폭주 근절).
  //   대각 Qf는 발산/수렴 모드에 가중 분산 → DCM은 발산 성분 집중. 값·gradient·rank1 hessian.
  double W_DCM=envd("W_DCM",0.0), GRV=9.81, DCM_TF=envd("DCM_TF",20.0);   // TF=터미널배율(Qf=Qx*20 대응)
  auto dcm_val=[&](const VectorXd&Xqk,const VectorXd&Xvk,const VectorXd&Rqk,const VectorXd&Rvk,double w)->double{
    if(w<=0)return 0; double om=std::sqrt(GRV/std::max(0.15,Rqk[2])); VectorXd e=ci.sdiff(Rqk,Rvk,Xqk,Xvk);
    double c=0; for(int ax=0;ax<2;ax++){ double d=e[ax]+e[nv+ax]/om; c+=0.5*w*d*d; } return c; };
  auto dcm_grad=[&](const VectorXd&Xqk,const VectorXd&Xvk,const VectorXd&Rqk,const VectorXd&Rvk,double w,VectorXd&gx,MatrixXd&Hxx){
    if(w<=0)return; double om=std::sqrt(GRV/std::max(0.15,Rqk[2])); VectorXd e=ci.sdiff(Rqk,Rvk,Xqk,Xvk);
    for(int ax=0;ax<2;ax++){ double d=e[ax]+e[nv+ax]/om;
      gx[ax]+=w*d; gx[nv+ax]+=w*d/om;
      Hxx(ax,ax)+=w; Hxx(ax,nv+ax)+=w/om; Hxx(nv+ax,ax)+=w/om; Hxx(nv+ax,nv+ax)+=w/(om*om); } };
  int PS=envi("PLAN_SOFT",0);   // 1=planner forward=step_soft(발 lift 가능, gait 참조가 발 듦)·0=relaxed
  int HF=envi("HARD_FWD",0);    // ★1=planner forward=step_kkt(hard, sim과 접촉모델 일치=HOUND식 hard forward)
  auto pf=[&](const VectorXd&q,const VectorXd&v,const VectorXd&u,VectorXd&qn,VectorXd&vn){
    int PN=envi("PLAN_NSUB",NSUB);   // planner forward substep(sim NSUB보다 작게=속도↑)
    if(HF) ci.step_kkt(q,v,u,DT,PN,qn,vn);        // hard forward(불일치 제거) — backward 그래디언트는 relaxed 유지
    else if(PS) ci.step_soft(q,v,u,DT,PN,qn,vn); else ci.step_relaxed(q,v,u,DT,qn,vn,PN); };
  auto eval=[&](std::vector<VectorXd>&Xq,std::vector<VectorXd>&Xv,std::vector<VectorXd>&U,std::vector<VectorXd>&gaps,double&J,double&M){
    gaps.assign(N+1,VectorXd::Zero(2*nv)); J=0; double gs=0;
    for(int k=0;k<N;k++){ VectorXd qn,vn; pf(Xq[k],Xv[k],U[k],qn,vn);
      gaps[k+1]=ci.sdiff(Xq[k+1],Xv[k+1],qn,vn); gs+=gaps[k+1].norm();
      VectorXd e=ci.sdiff(Rq[k],Rv[k],Xq[k],Xv[k]); J+=0.5*e.dot(Qx*e)+0.5*U[k].dot(Ru*U[k])+ci.foot_val(Xq[k],Xv[k])
        +dcm_val(Xq[k],Xv[k],Rq[k],Rv[k],W_DCM); }
    VectorXd e=ci.sdiff(Rq[N],Rv[N],Xq[N],Xv[N]); J+=0.5*e.dot(Qf*e)+dcm_val(Xq[N],Xv[N],Rq[N],Rv[N],DCM_TF*W_DCM); M=J+GAP_W*gs; };
  std::vector<VectorXd> gaps; double J,M; eval(Xq,Xv,U,gaps,J,M);
  for(int it=0;it<ITERS;it++){
    std::vector<MatrixXd> As(N),Bs(N);
    int LN=envi("LIN_NSUB",NSUB);   // ★선형화 substep(forward NSUB보다 작게=속도↑, FDDP gap이 mismatch 흡수)
    for(int k=0;k<N;k++){ MatrixXd A,B; ci.lin_AB_multi(Xq[k],Xv[k],U[k],DT,LN,A,B); As[k]=A; Bs[k]=B; }
    VectorXd e=ci.sdiff(Rq[N],Rv[N],Xq[N],Xv[N]); VectorXd Vx=Qf*e; MatrixXd Vxx=Qf;
    dcm_grad(Xq[N],Xv[N],Rq[N],Rv[N],DCM_TF*W_DCM,Vx,Vxx);   // ★DCM 터미널(발산 모드 cost-to-go)
    std::vector<MatrixXd> Ks(N); std::vector<VectorXd> ks(N);
    for(int k=N-1;k>=0;k--){
      VectorXd Vxp=Vx+Vxx*gaps[k+1];
      VectorXd ek=ci.sdiff(Rq[k],Rv[k],Xq[k],Xv[k]); VectorXd lx=Qx*ek, lu=Ru*U[k];
      double fc; VectorXd fg; MatrixXd fH; ci.foot_slip_cost(Xq[k],Xv[k],fc,fg,fH);
      lx+=fg; MatrixXd Qxx0=Qx+fH;
      dcm_grad(Xq[k],Xv[k],Rq[k],Rv[k],W_DCM,lx,Qxx0);   // ★DCM 러닝(발산 모드 억제)
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
    for(double alpha:{1.0,0.5,0.25,0.1,0.05,0.02,0.01}){   // line-search(품질). 속도는 N/ITERS/LIN_NSUB로
      std::vector<VectorXd> Xqn(N+1),Xvn(N+1),Un(N); Xqn[0]=Xq[0]; Xvn[0]=Xv[0]; bool ok=true;
      for(int k=0;k<N;k++){
        VectorXd dx=ci.sdiff(Xq[k],Xv[k],Xqn[k],Xvn[k]);
        VectorXd u=U[k]+alpha*ks[k]+Ks[k]*dx; Un[k]=u;
        VectorXd qn,vn; pf(Xqn[k],Xvn[k],u,qn,vn);
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
  int SIM_KKT=envi("SIM_KKT",0);   // 1=sim을 step_kkt(hard active-set, 발 lift 가능)로 → 스텝 보행
  ci.analytic_grad=envi("ANALYTIC",0);   // 1=해석 그래디언트(FD 44geom 제거, 실시간)
  ci.rho_relax=envd("RHO",1e-4);   // ★relaxed 완화계수(작을수록 hard에 근접=planner-sim 불일치↓)
  ci.CF=envd("CF",2000.0); ci.C1S=envd("C1S",-30.0); ci.AIR_W=envd("AIR_W",100.0); ci.SYM=envd("SYM",0.0);   // eq22-24
  ci.gap_x0=envd("GAP_X0",1e9); ci.gap_x1=envd("GAP_X1",-1e9);   // ★험지: 틈 [x0,x1]
  VectorXd Qxd(2*nv); Qxd.head(nv).setConstant(20.0); Qxd.tail(nv).setConstant(1.0);
  MatrixXd Qx=Qxd.asDiagonal();
  { double WB=envd("W_BASE",5.0); Qx(2,2)*=WB; Qx(3,3)*=WB; Qx(4,4)*=WB; }   // base z·roll·pitch 추종(서기·보행 공통)
  if(VX>0){ Qx(0,0)*=envd("VXPOS",0.1); Qx(nv,nv)*=envd("VXVEL",60.0); }      // 전진 위치완화·속도추종(보행 전용)
  double WJ=envd("W_JOINT",20.0); Qx.diagonal().segment(6,16).setConstant(WJ);   // ★다리 관절 gait 추종(스텝 강제)
  Qx.diagonal()[nv+2]*=envd("W_BVZ",1.0);   // ★base 수직속도 감쇠(진동/드리프트 억제)
  Qx.diagonal()[nv+0]*=envd("W_BVX",1.0);   // base 전방속도 추가감쇠(뒷걸음 드리프트 억제)
  MatrixXd Qf=Qx*20.0, Ru=MatrixXd::Identity(nu,nu)*envd("RU",1e-3);   // ★control-reg(작을수록 지지토크 편향↓)

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
  // ★gap-인지 발판: 발 world착지 x가 틈 안이면 solid ground(틈 너머/앞)로 shift + 재IK. 험지 크로싱
  bool has_gap = ci.gap_x1 > ci.gap_x0;
  auto gref_gap=[&](double ph,double base_x)->VectorXd{
    ph=std::fmod(ph,1.0); if(ph<0)ph+=1; std::vector<Vector3d> t(4);
    for(int i=0;i<4;i++){ double pi=std::fmod(ph+goff[i],1.0); if(pi<0)pi+=1; Vector3d n=gnom[i];
      if(pi<0.5) n[0]+=stride*(0.5-pi/0.5);
      else{ double sw=(pi-0.5)/0.5; n[0]+=stride*(sw-0.5); n[2]=STEP_H*std::sin(M_PI*sw); }
      double wx=base_x+n[0];
      if(ci.in_gap(wx)){ double mid=(ci.gap_x0+ci.gap_x1)/2;   // 틈 위 발 → 가까운 solid로(앞:틈전·뒤:틈후)
        double twx = (wx<mid)? ci.gap_x0-0.03 : ci.gap_x1+0.03; n[0]=twx-base_x; }
      t[i]=n; }
    return ci.ik_feet(t,BZ);
  };
  VectorXd q=qstar, v=VectorXd::Zero(nv), tau_hold=VectorXd::Zero(nu);
  for(int i=0;i<200;i++){ tau_hold=150.0*(qstar.tail(nu)-q.tail(nu))-8.0*v.tail(nu);
    VectorXd qn,vn; ci.step_soft(q,v,tau_hold,DT*0.1,1,qn,vn); q=qn; v=vn; }
  double x0=q[0];
  std::vector<VectorXd> U(N,tau_hold);
  std::printf("[C++ receding MPC] N=%d dt=%.3f %dHz재풀이 VX=%.2f CF=%.0f relax=%s\n",N,DT,(int)(1.0/DT),VX,ci.CF,ci.relax_mode.c_str());
  auto footmin=[&](const VectorXd&q,const VectorXd&v){ std::vector<double>phi;std::vector<MatrixXd>J;std::vector<Vector3d>vf;
    ci.foot_kin(q,v,phi,J,vf); double mn=1e9; for(int i=0;i<4;i++)mn=std::min(mn,phi[i]); return mn; };
  auto footmax=[&](const VectorXd&q,const VectorXd&v){ std::vector<double>phi;std::vector<MatrixXd>J;std::vector<Vector3d>vf;
    ci.foot_kin(q,v,phi,J,vf); double mx=-1e9; for(int i=0;i<4;i++)mx=std::max(mx,phi[i]); return mx; };  // ★스텝=어느 발이든 뜨는가

  const char* qout=std::getenv("QPOS_OUT");   // ★뷰어용 pin q 궤적 덤프(텍스트, Python이 mj변환+replay)
  std::vector<VectorXd> hist; if(qout) hist.push_back(q);
  auto tilt_deg=[&](const VectorXd&q){ double x=q[3],y=q[4],z=q[5],w=q[6];   // ★base 기울기(자세 품질)
    double rzz=1.0-2.0*(x*x+y*y); return std::acos(std::max(-1.0,std::min(1.0,rzz)))*180.0/M_PI; };
  double liftmax=0, solve_ms=0, sim_ms=0, tiltmax=0, bzmin=1e9, bzmax=-1e9;   // ★실시간+품질 지표
  for(int s=0;s<STEPS;s++){
    std::vector<VectorXd> Rq(N+1),Rv(N+1); double phase=s*DT/GT;
    for(int k=0;k<=N;k++){ Rq[k]= GAIT<0.5 ? qstar : (has_gap ? gref_gap(phase+k*DT/GT, q[0]+VX*k*DT) : gref(phase+k*DT/GT));   // ★gap 있으면 발판 회피
      Rq[k][0]=q[0]+VX*k*DT; Rv[k]=vstar; Rv[k][0]=VX; }
    std::vector<VectorXd> Xq(N+1),Xv(N+1); Xq[0]=q; Xv[0]=v;
    for(int k=1;k<=N;k++){ Xq[k]=Rq[k]; Xv[k]=Rv[k]; }
    double _t0=now_ms();
    MatrixXd K0=solve_fddp(ci,Xq,Xv,U,Rq,Rv,N,DT,NSUB,GAP_W,REG,ITERS,Qx,Qf,Ru);
    double _t1=now_ms(); solve_ms+=(_t1-_t0);
    // ★HOUND §6.3: 계획(u0)을 fine-rate PD+FF로 추종(제어 유지간격 h=DT/NSUB≤0.001=hard 접촉 안정).
    //   u = u_ff + Kp(q_plan−q) + Kd(v_plan−v). q_plan/v_plan=계획 다음노드(Xq[1]).
    VectorXd u0=U[0], qtgt=Xq[1], vtgt=Xv[1]; double hc=DT/NSUB;
    double KPT=envd("KP_T",150.0), KDT=envd("KD_T",12.0);
    for(int sub=0; sub<NSUB; sub++){
      VectorXd u = u0 + KPT*(qtgt.tail(nu)-q.tail(nu)) + KDT*(vtgt.tail(nu)-v.tail(nu));   // PD+FF
      VectorXd qn,vn;
      if(SIM_KKT) ci.step_kkt(q,v,u,hc,1,qn,vn); else ci.step_relaxed(q,v,u,hc,qn,vn,1);
      q=qn; v=vn; if(!q.allFinite()) break;
    }
    sim_ms+=(now_ms()-_t1);
    if(!q.allFinite()){ std::printf("  ✗ 발산 step %d\n",s+1); break; }
    if(qout) hist.push_back(q);
    liftmax=std::max(liftmax,footmax(q,v));
    tiltmax=std::max(tiltmax,tilt_deg(q)); bzmin=std::min(bzmin,q[2]); bzmax=std::max(bzmax,q[2]);
    for(int k=0;k<N-1;k++) U[k]=U[k+1]; U[N-1]=tau_hold;   // warm-start shift
    if((s+1)%15==0) std::printf("  step %3d t=%.2fs 전진=%.3fm base_z=%.3f vx=%.2f 발높이[min %.3f max %.3f]\n",
                                s+1,(s+1)*DT,q[0]-x0,q[2],v[0],footmin(q,v),footmax(q,v));
  }
  bool clean = q[2]>0.30 && tiltmax<12.0 && (bzmax-bzmin)<0.10;   // ★깨끗=미붕괴+저기울기+저상하요동
  std::printf("  최종: %.2fs 전진 %.3fm(평균 %.2f m/s·목표 %.2f) base_z=%.3f 발lift최대=%.3f  %s\n",
              STEPS*DT,q[0]-x0,(q[0]-x0)/(STEPS*DT),VX,q[2],liftmax,
              clean?"✅ 깨끗한 전진 보행":q[2]>0.30?"△ 미붕괴이나 거침(기울기/요동↑)":"✗ 붕괴");
  std::printf("  [품질] base기울기 max=%.1f° · base_z 요동 %.3f~%.3f(폭 %.3f) (기울기<12°·요동<0.10=깨끗)\n",
              tiltmax,bzmin,bzmax,bzmax-bzmin);
  int done=std::max(1,STEPS); double per=(solve_ms+sim_ms)/done;
  std::printf("  [프로파일] 스텝당 solve=%.1fms sim=%.1fms 합=%.1fms  → 실시간(DT=%.0fms) %s (RT계수 %.1fx느림, 40Hz엔 %.1fx)\n",
              solve_ms/done,sim_ms/done,per,DT*1000,per<=DT*1000?"✅달성":"✗미달",per/(DT*1000),per/25.0);
  if(qout){ FILE*f=std::fopen(qout,"w");
    for(auto&qq:hist){ for(int i=0;i<ci.nq;i++) std::fprintf(f,"%.10g ",qq[i]); std::fprintf(f,"\n"); }
    std::fclose(f); std::printf("  qpos 덤프: %s (%zu 프레임, pin q) — Python이 mj변환+replay\n",qout,hist.size()); }
  return 0;
}
