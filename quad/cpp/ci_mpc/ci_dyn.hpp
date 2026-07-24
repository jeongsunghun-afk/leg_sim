// CI-MPC C++ 동역학/그래디언트 라이브러리 (ci_action.py + ci_ocp.py 포트).
//   relaxed 상보성 접촉 그래디언트(검증됨) + soft forward(rollout) + foot-slip cost + stance IK.
#pragma once
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/constrained-dynamics.hpp>   // step_kkt(hard active-set)
#include <pinocchio/algorithm/kinematics-derivatives.hpp>  // ★해석 그래디언트: ∂J/∂q(kinematic Hessian)
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>        // ★∂M/∂q(RNEA 도함수 트릭)
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

namespace cimpc {
using namespace pinocchio;
using Eigen::VectorXd; using Eigen::MatrixXd; using Eigen::Vector3d;

inline double softplus(double x,double rho){ double z=x/rho; return z>30?rho*z:rho*std::log1p(std::exp(z<-30?-30:z)); }
inline double sigmoidf(double x,double rho){ double z=x/rho; z=z>30?30:(z<-30?-30:z); return 1.0/(1.0+std::exp(-z)); }

struct CiDyn {
  Model model; Data data;
  int nv,nq,nu; std::vector<FrameIndex> fids;
  double FOOT_R=0.025, margin=0.05, eps=1e-3;
  double rho=0.004, kn=1.2e4, bn=120.0, bt=80.0, mu=0.8;      // soft force law (rollout)
  double CF=2500.0, C1S=-30.0, AIR_W=100.0, SYM=0.0;           // foot-slip/air-time/symmetry cost(eq22-24)
  std::string relax_mode="D"; double rho_relax=1e-4;           // ★기본=논문판 ρD(vⁿλⁿ=ρ·법선전용). "eps"=Tikhonov
  double bg_kp=10.0, bg_kd=50.0;                                // step_kkt Baumgarte(위치 드리프트 보정)
  bool analytic_grad=false;                                    // ★해석 그래디언트(kinematic Hessian+RNEA, FD 대체)
  double gap_x0=1e9, gap_x1=-1e9;                               // ★험지: 지면 없는 틈 [x0,x1](기본 없음)
  bool in_gap(double x) const { return x>gap_x0 && x<gap_x1; }  // 틈 위=지지 없음(발 빠짐)

  CiDyn(const std::string& urdf_path){
    if(const char*rm=std::getenv("RELAX_MODE")) relax_mode=rm;  // env로도 지정
    if(const char*rr=std::getenv("RELAX_RHO"))  rho_relax=std::atof(rr);
    Model full; pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), full);
    std::vector<JointIndex> lock={ full.getJointId("FB_waist_joint") };
    buildReducedModel(full, lock, neutral(full), model); data=Data(model);
    nv=model.nv; nq=model.nq; nu=nv-6;
    double arm[4]={1e-4*49.0,1e-4*49.0,1e-4*110.25,1e-4*70.56};
    for(int leg=0;leg<4;leg++) for(int j=0;j<4;j++) model.armature[6+leg*4+j]=arm[j];
    const char* feet[4]={"FL","FR","HL","HR"};
    for(auto&L:feet) fids.push_back(model.getFrameId(std::string(L)+"_foot_contact_link"));
  }

  // 발바닥 z(지면위높이) + LWA 선형 Jacobian
  void foot_kin(const VectorXd&q,const VectorXd&v,std::vector<double>&phi,std::vector<MatrixXd>&J,std::vector<Vector3d>&vf){
    forwardKinematics(model,data,q,v); updateFramePlacements(model,data); computeJointJacobians(model,data,q);
    phi.resize(4); J.resize(4); vf.resize(4);
    for(int i=0;i<4;i++){ const auto&oMf=data.oMf[fids[i]];
      Vector3d p=oMf.translation()+oMf.rotation()*Vector3d(0,0,-FOOT_R); phi[i]=p[2];
      MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(model,data,fids[i],LOCAL_WORLD_ALIGNED,J6);
      J[i]=J6.topRows(3); vf[i]=J[i]*v; }
  }

  // ★임의 4발 타겟(base frame) IK: base 고정(0,0,base_z)·발을 tgt[i]로. 관절 gait 참조용
  VectorXd ik_feet(const std::vector<Vector3d>&tgt, double base_z){ return ik_feet(tgt,base_z,0.0); }
  // ★pitch 오버로드: base를 pitch(nose-up, y축) 만큼 기울인 자세(앉기·두발서기용)
  VectorXd ik_feet(const std::vector<Vector3d>&tgt, double base_z, double pitch){
    VectorXd q=neutral(model); q[2]=base_z;
    if(pitch!=0.0){ Eigen::Quaterniond quat(Eigen::AngleAxisd(pitch, Vector3d::UnitY()));
      q[3]=quat.x(); q[4]=quat.y(); q[5]=quat.z(); q[6]=quat.w(); }   // base 자세=pitch
    for(int it=0;it<200;it++){
      forwardKinematics(model,data,q); updateFramePlacements(model,data); computeJointJacobians(model,data,q);
      VectorXd err(12); MatrixXd Js(12,nv);
      for(int i=0;i<4;i++){ const auto&oMf=data.oMf[fids[i]];
        Vector3d p=oMf.translation()+oMf.rotation()*Vector3d(0,0,-FOOT_R);
        err.segment(3*i,3)=tgt[i]-p;
        MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(model,data,fids[i],LOCAL_WORLD_ALIGNED,J6); Js.middleRows(3*i,3)=J6.topRows(3); }
      Js.leftCols(6).setZero();
      if(err.norm()<1e-5) break;
      VectorXd dq=Js.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(err);
      q=integrate(model,q,VectorXd(0.5*dq));
    }
    return q;
  }
  VectorXd stance_q(){
    std::vector<Vector3d> tgt={{0.30,0.16,0.0},{0.30,-0.16,0.0},{-0.30,0.16,0.0},{-0.30,-0.16,0.0}};  // FL,FR,HL,HR
    return ik_feet(tgt, 0.42);
  }

  // soft force law f(phi,vf) → world 3D (rollout용)
  Vector3d force_law(double phi,const Vector3d&vf){
    double w=sigmoidf(-phi,rho), depth=softplus(-phi,rho);
    double fn=softplus(kn*depth-bn*vf[2]*w,1.0);
    Vector3d ft(-bt*vf[0]*w,-bt*vf[1]*w,0); double ftn=ft.head(2).norm()+1e-9, cap=mu*fn;
    if(ftn>cap) ft.head(2)*=cap/ftn;
    return Vector3d(ft[0],ft[1],fn);
  }
  // soft forward: ddq=aba(q,v,tau+ΣJᵀf) → semi-implicit
  void step_soft(const VectorXd&q0,const VectorXd&v0,const VectorXd&u,double dt,int nsub,VectorXd&qn,VectorXd&vn){
    double h=dt/nsub; VectorXd q=q0,v=v0;
    for(int s=0;s<nsub;s++){
      std::vector<double> phi; std::vector<MatrixXd> J; std::vector<Vector3d> vf; foot_kin(q,v,phi,J,vf);
      VectorXd tau(nv); tau.head(6).setZero(); tau.tail(nu)=u;
      for(int i=0;i<4;i++) tau+=J[i].transpose()*force_law(phi[i],vf[i]);
      VectorXd ddq=aba(model,data,q,v,tau); v=v+h*ddq; q=integrate(model,q,VectorXd(h*v));
    }
    qn=q; vn=v;
  }

  // ★relaxed λ solve(모드공유): "D"=논문 ρD(vⁿλⁿ=ρ Newton·법선전용·λⁿ>0), "eps"=Tikhonov
  VectorXd relaxed_lambda(const MatrixXd&Acc,const VectorXd&bcc,int nc){
    if(relax_mode!="D") return -(Acc+eps*MatrixXd::Identity(nc,nc)).inverse()*bcc;
    std::vector<int> nrm; for(int k=0;k<nc/3;k++) nrm.push_back(3*k+2);
    VectorXd lam=-(Acc+1e-6*MatrixXd::Identity(nc,nc)).ldlt().solve(bcc);
    for(int l:nrm) if(lam[l]<1e-4) lam[l]=1e-4;
    for(int it=0;it<12;it++){
      MatrixXd Dm=MatrixXd::Zero(nc,nc); VectorXd r=VectorXd::Zero(nc);
      for(int l:nrm){ Dm(l,l)=1.0/(lam[l]*lam[l]); r[l]=rho_relax/lam[l]; }
      lam-=(Acc+rho_relax*Dm).ldlt().solve(Acc*lam+bcc-r);
      for(int l:nrm) if(lam[l]<1e-9) lam[l]=1e-9;
    }
    return lam;
  }
  // relaxed forward(도함수 없이, rollout용): lin_AB와 일관. v_next=qdot_free+M⁻¹Jᵀλ. nsub=multi-rate 서브스텝
  void step_relaxed(const VectorXd&q0,const VectorXd&v0,const VectorXd&u,double dt,VectorXd&qn,VectorXd&vn,int nsub=1){
    double h=dt/nsub; VectorXd q=q0,v=v0; VectorXd tau(nv); tau.head(6).setZero(); tau.tail(nu)=u;
    for(int s=0;s<nsub;s++){
      VectorXd a_free=aba(model,data,q,v,tau);
      computeJointJacobians(model,data,q); updateFramePlacements(model,data);
      computeMinverse(model,data,q); MatrixXd Mi=data.Minv;
      Mi.triangularView<Eigen::StrictlyLower>()=Mi.transpose().triangularView<Eigen::StrictlyLower>();
      MatrixXd Jcc(12,nv);
      for(int k=0;k<4;k++){ MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(model,data,fids[k],LOCAL_WORLD_ALIGNED,J6); Jcc.middleRows(3*k,3)=J6.topRows(3); }
      MatrixXd W=Mi*Jcc.transpose(), Acc=Jcc*W;
      VectorXd qdot_free=v+h*a_free, lam=relaxed_lambda(Acc,Jcc*qdot_free,12);
      v=qdot_free+W*lam; q=integrate(model,q,VectorXd(h*v));
    }
    qn=q; vn=v;
  }

  // ★hard 접촉 forward: constraintDynamics(proper KKT)+active-set(단방향). 발이 접촉 떠날 수 있음=스텝 가능.
  void step_kkt(const VectorXd&q0,const VectorXd&v0,const VectorXd&u,double dt,int nsub,VectorXd&qn,VectorXd&vn){
    double h=dt/nsub; VectorXd q=q0,v=v0; VectorXd tau(nv); tau.head(6).setZero(); tau.tail(nu)=u;
    ProximalSettings prox(1e-10, rho_relax, 40);
    for(int s=0;s<nsub;s++){
      forwardKinematics(model,data,q); updateFramePlacements(model,data);
      std::vector<int> active;
      for(int i=0;i<4;i++){ const SE3&oMf=data.oMf[fids[i]];
        Vector3d cp=oMf.translation()+oMf.rotation()*Vector3d(0,0,-FOOT_R);
        if(cp[2]<margin && !in_gap(cp[0])) active.push_back(i); }   // ★틈 위 발=지지없음(active 제외→빠짐)
      VectorXd a;
      for(int as=0; as<5; as++){
        if(active.empty()){ a=aba(model,data,q,v,tau); break; }
        PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintModel) cms;
        PINOCCHIO_ALIGNED_STD_VECTOR(RigidConstraintData) cds;
        for(int i:active){ const Frame&fr=model.frames[fids[i]];
          SE3 pl=fr.placement*SE3(Eigen::Matrix3d::Identity(),Vector3d(0,0,-FOOT_R));
          RigidConstraintModel cm(CONTACT_3D, model, fr.parentJoint, pl, LOCAL_WORLD_ALIGNED);
          cms.push_back(cm); }
        for(auto&cm:cms) cds.push_back(RigidConstraintData(cm));
        initConstraintDynamics(model,data,cms,cds);
        a=constraintDynamics(model,data,q,v,tau,cms,cds,prox);
        std::vector<int> keep;                                  // 접촉력 법선<0(당김)=분리
        for(size_t j=0;j<active.size();j++) if(cds[j].contact_force.linear()[2]>=0.0) keep.push_back(active[j]);
        if(keep.size()==active.size()) break;
        active=keep;
      }
      v=v+h*a; q=integrate(model,q,VectorXd(h*v));
    }
    qn=q; vn=v;
  }

  // ★relaxed 상보성 ddq + tangent A,B (검증됨)
  void dyn_relaxed(const VectorXd&q,const VectorXd&v,const VectorXd&u,double dt,
                   VectorXd&ddq,MatrixXd&ddq_dq,MatrixXd&ddq_dv,MatrixXd&ddq_du){
    VectorXd tau(nv); tau.head(6).setZero(); tau.tail(nu)=u;
    std::vector<int> active={0,1,2,3}; int nc=12;
    computeABADerivatives(model,data,q,v,tau);
    VectorXd a_free=data.ddq; MatrixXd aq=data.ddq_dq, av=data.ddq_dv;
    MatrixXd Minv=data.Minv; Minv.triangularView<Eigen::StrictlyLower>()=Minv.transpose().triangularView<Eigen::StrictlyLower>();
    MatrixXd au=Minv.rightCols(nu);
    auto geom=[&](const VectorXd&qq,MatrixXd&Jcc,MatrixXd&W,MatrixXd&Acc){
      computeJointJacobians(model,data,qq); updateFramePlacements(model,data);
      computeMinverse(model,data,qq); MatrixXd Mi=data.Minv;
      Mi.triangularView<Eigen::StrictlyLower>()=Mi.transpose().triangularView<Eigen::StrictlyLower>();
      Jcc.resize(nc,nv);
      for(int k=0;k<4;k++){ MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(model,data,fids[active[k]],LOCAL_WORLD_ALIGNED,J6); Jcc.middleRows(3*k,3)=J6.topRows(3); }
      W=Mi*Jcc.transpose(); Acc=Jcc*W; };
    MatrixXd Jcc,W,Acc; geom(q,Jcc,W,Acc);
    VectorXd qdot_free=v+dt*a_free, bcc=Jcc*qdot_free;
    VectorXd lam=relaxed_lambda(Acc,bcc,nc); MatrixXd Ari;   // ★모드공유 λ solve(D=ρD Newton·eps=Tikhonov)
    if(relax_mode=="D"){                        // gradient용 Ari=(A_cc+ρD)⁻¹, D=diag(1/λ_n²) 법선전용
      MatrixXd Dm=MatrixXd::Zero(nc,nc); for(int k=0;k<nc/3;k++){ int l=3*k+2; Dm(l,l)=1.0/(lam[l]*lam[l]); }
      Ari=(Acc+rho_relax*Dm).inverse();
    } else {                                    // εI Tikhonov(구)
      Ari=(Acc+eps*MatrixXd::Identity(nc,nc)).inverse();
    }
    ddq=a_free+W*lam/dt;
    MatrixXd dbg=MatrixXd::Zero(nc,nv), dWl=MatrixXd::Zero(nv,nv), DAy(nc,nv);  // dbg=∂bcc/∂q·[Jcc부분], dWl=∂(Wλ)/∂q, DAy=∂A_cc/∂q·(−λ)
    VectorXd y=-lam;
    if(analytic_grad){   // ★해석: kinematic Hessian(∂Jcc/∂q)+RNEA트릭(∂M⁻¹/∂q), FD 44geom 제거
      VectorXd Wlam=W*lam;                                          // Wλ (nv)
      computeJointKinematicHessians(model,data,q);
      MatrixXd Term1=MatrixXd::Zero(nc,nv), JTl=MatrixXd::Zero(nv,nv);
      for(int k=0;k<4;k++){ Tensor<double,3> Hk(6,nv,nv); Hk.setZero();
        getFrameKinematicHessian(model,data,fids[active[k]],LOCAL_WORLD_ALIGNED,Hk);
        for(int a=0;a<3;a++) for(int jj=0;jj<nv;jj++){ double d=0,t1=0;
          for(int b=0;b<nv;b++){ double h=Hk(a,b,jj); d+=h*qdot_free[b]; t1+=h*Wlam[b]; }
          dbg(3*k+a,jj)=d; Term1(3*k+a,jj)=t1; }                    // dbg=(∂Jcc/∂q)qdot_free · Term1=(∂Jcc/∂q)Wλ
        for(int b=0;b<nv;b++) for(int jj=0;jj<nv;jj++){ double s=0;
          for(int a=0;a<3;a++) s+=Hk(a,b,jj)*lam[3*k+a]; JTl(b,jj)+=s; } }  // JTλ=∂(Jccᵀλ)/∂q
      VectorXd z=VectorXd::Zero(nv);                                // DMy=∂(M·Wλ)/∂q = RNEAderiv(y)−RNEAderiv(0)
      computeRNEADerivatives(model,data,q,z,Wlam); MatrixXd dq_y=data.dtau_dq;
      computeRNEADerivatives(model,data,q,z,z);    MatrixXd DMy=dq_y-data.dtau_dq;
      dWl=Minv*(JTl-DMy);                                           // ∂(Wλ)/∂q = M⁻¹(JTλ−DMy)
      DAy=-(Term1+Jcc*dWl);                                         // ∂A_cc/∂q·(−λ) = −(Term1+Jcc·dWl)
    } else {             // ── FD 기하 도함수(기존, 검증 기준) ──
      std::vector<MatrixXd> dAcc(nv,MatrixXd::Zero(nc,nc)); double e=1e-6;
      for(int j=0;j<nv;j++){ VectorXd dq=VectorXd::Zero(nv); dq[j]=e; MatrixXd Jp,Wp,Ap,Jm,Wm,Am;
        geom(integrate(model,q,dq),Jp,Wp,Ap); geom(integrate(model,q,VectorXd(-dq)),Jm,Wm,Am);
        dAcc[j]=(Ap-Am)/(2*e); dbg.col(j)=(Jp-Jm)*qdot_free/(2*e); dWl.col(j)=(Wp-Wm)*lam/(2*e); }
      for(int j=0;j<nv;j++) DAy.col(j)=dAcc[j]*y;
    }
    MatrixXd dl_dq(nc,nv);       // δλ/δz=−Ari(δA·λ+δb)
    for(int j=0;j<nv;j++) dl_dq.col(j)=Ari*DAy.col(j);
    dl_dq-=Ari*(dbg+Jcc*(dt*aq));
    MatrixXd dl_dv=-Ari*(Jcc*(MatrixXd::Identity(nv,nv)+dt*av)), dl_du=-Ari*(Jcc*(dt*au));
    ddq_dq=aq+(dWl+W*dl_dq)/dt; ddq_dv=av+(W*dl_dv)/dt; ddq_du=au+(W*dl_du)/dt;
  }
  // 단일 스텝 tangent 선형화 A,B + 다음 상태(qn,vn) 반환(multi-rate 합성용)
  void lin_AB(const VectorXd&q,const VectorXd&v,const VectorXd&u,double dt,MatrixXd&A,MatrixXd&B,VectorXd&qn,VectorXd&vn){
    VectorXd ddq; MatrixXd dq_,dv_,du_; dyn_relaxed(q,v,u,dt,ddq,dq_,dv_,du_);
    VectorXd v_next=v+dt*ddq, w=dt*v_next;
    MatrixXd dvn_dq=dt*dq_, dvn_dv=MatrixXd::Identity(nv,nv)+dt*dv_, dvn_du=dt*du_;
    // ★dIntegrate는 블록대각만 쓰고 off-diagonal은 안 지움 → 반드시 setZero 선행(안 하면 쓰레기값 폭발)
    MatrixXd dInt0=MatrixXd::Zero(nv,nv),dInt1=MatrixXd::Zero(nv,nv); dIntegrate(model,q,w,dInt0,ARG0); dIntegrate(model,q,w,dInt1,ARG1);
    MatrixXd dqn_dq=dInt0+dInt1*(dt*dvn_dq), dqn_dv=dInt1*(dt*dvn_dv), dqn_du=dInt1*(dt*dvn_du);
    A.resize(2*nv,2*nv); A<<dqn_dq,dqn_dv,dvn_dq,dvn_dv; B.resize(2*nv,nu); B<<dqn_du,dvn_du;
    vn=v_next; qn=integrate(model,q,w);
  }
  void lin_AB(const VectorXd&q,const VectorXd&v,const VectorXd&u,double dt,MatrixXd&A,MatrixXd&B){
    VectorXd qn,vn; lin_AB(q,v,u,dt,A,B,qn,vn); }
  // ★multi-rate 노드 Jacobian: A_node=∏Aₖ · B_node=Σ연쇄율(Python lin_AB nsub 합성). u는 노드 내 상수
  void lin_AB_multi(const VectorXd&q0,const VectorXd&v0,const VectorXd&u,double dt,int nsub,MatrixXd&A,MatrixXd&B){
    double h=dt/nsub; VectorXd q=q0,v=v0;
    A=MatrixXd::Identity(2*nv,2*nv); B=MatrixXd::Zero(2*nv,nu); bool first=true;
    for(int s=0;s<nsub;s++){
      MatrixXd Ak,Bk; VectorXd qn,vn; lin_AB(q,v,u,h,Ak,Bk,qn,vn);
      A=Ak*A; B = first ? Bk : (Ak*B+Bk).eval(); first=false;
      q=qn; v=vn;
    }
  }

  // foot-slip/clearance(eq22)+air-time(φ²) 값
  double foot_val(const VectorXd&q,const VectorXd&v){
    std::vector<double> phi; std::vector<MatrixXd> J; std::vector<Vector3d> vf; foot_kin(q,v,phi,J,vf);
    double c=0;
    for(int i=0;i<4;i++){ double w2=vf[i].head(2).squaredNorm(); double S=1.0/(1.0+std::exp(-C1S*phi[i]));
      c+=CF*S*w2; if(AIR_W>0&&phi[i]>0) c+=AIR_W*phi[i]*phi[i]; }
    if(SYM>0){ double d1=phi[0]-phi[3], d2=phi[1]-phi[2]; c+=SYM*(d1*d1+d2*d2); }   // 대각쌍 대칭
    return c;
  }
  // ★foot-slip cost 값+gradient(2nv)+Gauss-Newton hessian(2nv²). ∂c/∂q=sigmoid높이항(exact)·∂c/∂v=∂vt/∂v(exact).
  //   ★analytic_grad=1: ∂vt/∂q 커플링도 kinematic Hessian(∂J/∂q·v)로 exact화(걸음 창발 gradient 완전). 0=제외(부분).
  void foot_slip_cost(const VectorXd&q,const VectorXd&v,double&c,VectorXd&g,MatrixXd&H){
    c=0; g=VectorXd::Zero(2*nv); H=MatrixXd::Zero(2*nv,2*nv);
    if(CF<=0 && AIR_W<=0 && SYM<=0) return;
    std::vector<double> phi; std::vector<MatrixXd> J; std::vector<Vector3d> vf; foot_kin(q,v,phi,J,vf);
    if(analytic_grad) computeJointKinematicHessians(model,data,q);   // ∂J/∂q 준비(foot_kin이 FK·jointJac 선행)
    std::vector<VectorXd> Jzs(4);
    for(int i=0;i<4;i++){
      VectorXd Jz=J[i].row(2).transpose(); Jzs[i]=Jz;        // ∂φ/∂q (nv, 프레임원점 근사)
      if(analytic_grad){   // ★접촉점(오프셋 r=R·(0,0,-FOOT_R)) 정확 z-Jacobian: J_v−skew(r)J_ω 의 z행
        const auto&oMf=data.oMf[fids[i]]; Vector3d r=oMf.rotation()*Vector3d(0,0,-FOOT_R);
        MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(model,data,fids[i],LOCAL_WORLD_ALIGNED,J6);
        Jz=(J6.row(2)+r[1]*J6.row(3)-r[0]*J6.row(4)).transpose(); Jzs[i]=Jz;   // z: +ry·ωx−rx·ωy
      }
      double vx=vf[i][0], vy=vf[i][1], w2=vx*vx+vy*vy;
      double S=1.0/(1.0+std::exp(-C1S*phi[i])), Sp=S*(1.0-S);
      c += CF*S*w2;
      g.head(nv) += CF*Sp*C1S*w2 * Jz;                       // ∂c/∂q (높이 sigmoid, exact)
      g.tail(nv) += CF*S*2.0*(vx*J[i].row(0).transpose() + vy*J[i].row(1).transpose());  // ∂c/∂v
      if(analytic_grad){   // ★∂(CF·S·w2)/∂q 의 접선속도 항: ∂w2/∂q=2(vx·∂vx/∂q+vy·∂vy/∂q), ∂v_f=∂J/∂q·v
        Tensor<double,3> Hk(6,nv,nv); Hk.setZero();
        getFrameKinematicHessian(model,data,fids[i],LOCAL_WORLD_ALIGNED,Hk);
        VectorXd dvx=VectorXd::Zero(nv), dvy=VectorXd::Zero(nv);
        for(int jj=0;jj<nv;jj++){ double ax=0,ay=0;
          for(int b=0;b<nv;b++){ ax+=Hk(0,b,jj)*v[b]; ay+=Hk(1,b,jj)*v[b]; }
          dvx[jj]=ax; dvy[jj]=ay; }
        g.head(nv) += CF*S*2.0*(vx*dvx + vy*dvy);            // ★∂vt/∂q (걸음 창발 gradient exact)
      }
      MatrixXd Jt=J[i].topRows(2);                           // 2×nv (접선)
      H.bottomRightCorner(nv,nv) += CF*2.0*S*(Jt.transpose()*Jt);   // GN hessian(속도)
      if(AIR_W>0 && phi[i]>0){                               // air-time φ²
        c += AIR_W*phi[i]*phi[i];
        g.head(nv) += AIR_W*2.0*phi[i]*Jz;
        H.topLeftCorner(nv,nv) += AIR_W*2.0*(Jz*Jz.transpose());
      }
    }
    if(SYM>0){   // ★eq24 대칭: 대각쌍(FL0-HR3, FR1-HL2) 발높이 동기화 → 트롯 유도(바운싱 억제)
      int pr[2][2]={{0,3},{1,2}};
      for(auto&p:pr){ double d=phi[p[0]]-phi[p[1]]; VectorXd Jd=Jzs[p[0]]-Jzs[p[1]];
        c+=SYM*d*d; g.head(nv)+=SYM*2.0*d*Jd; H.topLeftCorner(nv,nv)+=SYM*2.0*(Jd*Jd.transpose()); }
    }
  }

  // 상태 매니폴드: sdiff(a→b tangent), sint(a⊕t)
  VectorXd sdiff(const VectorXd&qa,const VectorXd&va,const VectorXd&qb,const VectorXd&vb){
    VectorXd e(2*nv); e.head(nv)=difference(model,qa,qb); e.tail(nv)=vb-va; return e; }
  void sint(const VectorXd&qa,const VectorXd&va,const VectorXd&t,VectorXd&qb,VectorXd&vb){
    qb=integrate(model,qa,VectorXd(t.head(nv))); vb=va+t.tail(nv); }
};
} // namespace cimpc
