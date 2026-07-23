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
  double CF=2500.0, C1S=-30.0, AIR_W=100.0;                    // foot-slip/air-time cost
  std::string relax_mode="D"; double rho_relax=1e-4;           // ★기본=논문판 ρD(vⁿλⁿ=ρ·법선전용). "eps"=Tikhonov

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

  VectorXd stance_q(){
    VectorXd q=neutral(model); q[2]=0.42;
    double tx[4]={0.30,0.30,-0.30,-0.30}, ty[4]={0.16,-0.16,0.16,-0.16};
    for(int it=0;it<300;it++){
      forwardKinematics(model,data,q); updateFramePlacements(model,data); computeJointJacobians(model,data,q);
      VectorXd err(12); MatrixXd Js(12,nv);
      for(int i=0;i<4;i++){ const auto&oMf=data.oMf[fids[i]];
        Vector3d p=oMf.translation()+oMf.rotation()*Vector3d(0,0,-FOOT_R);
        err.segment(3*i,3)=Vector3d(tx[i],ty[i],0.0)-p;
        MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(model,data,fids[i],LOCAL_WORLD_ALIGNED,J6); Js.middleRows(3*i,3)=J6.topRows(3); }
      Js.leftCols(6).setZero();
      if(err.norm()<1e-5) break;
      VectorXd dq=Js.bdcSvd(Eigen::ComputeThinU|Eigen::ComputeThinV).solve(err);
      q=integrate(model,q,VectorXd(0.5*dq));
    }
    return q;
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
  // relaxed forward(도함수 없이, rollout용): lin_AB와 일관. v_next=qdot_free+M⁻¹Jᵀλ
  void step_relaxed(const VectorXd&q,const VectorXd&v,const VectorXd&u,double dt,VectorXd&qn,VectorXd&vn){
    VectorXd tau(nv); tau.head(6).setZero(); tau.tail(nu)=u;
    VectorXd a_free=aba(model,data,q,v,tau);
    computeJointJacobians(model,data,q); updateFramePlacements(model,data);
    computeMinverse(model,data,q); MatrixXd Mi=data.Minv;
    Mi.triangularView<Eigen::StrictlyLower>()=Mi.transpose().triangularView<Eigen::StrictlyLower>();
    MatrixXd Jcc(12,nv);
    for(int k=0;k<4;k++){ MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(model,data,fids[k],LOCAL_WORLD_ALIGNED,J6); Jcc.middleRows(3*k,3)=J6.topRows(3); }
    MatrixXd W=Mi*Jcc.transpose(), Acc=Jcc*W;
    VectorXd qdot_free=v+dt*a_free, lam=relaxed_lambda(Acc,Jcc*qdot_free,12);
    vn=qdot_free+W*lam; qn=integrate(model,q,VectorXd(dt*vn));
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
    std::vector<MatrixXd> dAcc(nv,MatrixXd::Zero(nc,nc)); MatrixXd dbg=MatrixXd::Zero(nc,nv), dWl=MatrixXd::Zero(nv,nv);
    double e=1e-6;
    for(int j=0;j<nv;j++){ VectorXd dq=VectorXd::Zero(nv); dq[j]=e; MatrixXd Jp,Wp,Ap,Jm,Wm,Am;
      geom(integrate(model,q,dq),Jp,Wp,Ap); geom(integrate(model,q,VectorXd(-dq)),Jm,Wm,Am);
      dAcc[j]=(Ap-Am)/(2*e); dbg.col(j)=(Jp-Jm)*qdot_free/(2*e); dWl.col(j)=(Wp-Wm)*lam/(2*e); }
    VectorXd y=-lam; MatrixXd dl_dq(nc,nv);       // δλ/δz=−Ari(δA·λ+δb) (εI선 Ari*bcc=−lam과 동치)
    for(int j=0;j<nv;j++) dl_dq.col(j)=Ari*(dAcc[j]*y);
    dl_dq-=Ari*(dbg+Jcc*(dt*aq));
    MatrixXd dl_dv=-Ari*(Jcc*(MatrixXd::Identity(nv,nv)+dt*av)), dl_du=-Ari*(Jcc*(dt*au));
    ddq_dq=aq+(dWl+W*dl_dq)/dt; ddq_dv=av+(W*dl_dv)/dt; ddq_du=au+(W*dl_du)/dt;
  }
  void lin_AB(const VectorXd&q,const VectorXd&v,const VectorXd&u,double dt,MatrixXd&A,MatrixXd&B){
    VectorXd ddq; MatrixXd dq_,dv_,du_; dyn_relaxed(q,v,u,dt,ddq,dq_,dv_,du_);
    VectorXd v_next=v+dt*ddq, w=dt*v_next;
    MatrixXd dvn_dq=dt*dq_, dvn_dv=MatrixXd::Identity(nv,nv)+dt*dv_, dvn_du=dt*du_;
    // ★dIntegrate는 블록대각만 쓰고 off-diagonal은 안 지움 → 반드시 setZero 선행(안 하면 쓰레기값 폭발)
    MatrixXd dInt0=MatrixXd::Zero(nv,nv),dInt1=MatrixXd::Zero(nv,nv); dIntegrate(model,q,w,dInt0,ARG0); dIntegrate(model,q,w,dInt1,ARG1);
    MatrixXd dqn_dq=dInt0+dInt1*(dt*dvn_dq), dqn_dv=dInt1*(dt*dvn_dv), dqn_du=dInt1*(dt*dvn_du);
    A.resize(2*nv,2*nv); A<<dqn_dq,dqn_dv,dvn_dq,dvn_dv; B.resize(2*nv,nu); B<<dqn_du,dvn_du;
  }

  // foot-slip/clearance(eq22)+air-time(φ²) 값
  double foot_val(const VectorXd&q,const VectorXd&v){
    std::vector<double> phi; std::vector<MatrixXd> J; std::vector<Vector3d> vf; foot_kin(q,v,phi,J,vf);
    double c=0;
    for(int i=0;i<4;i++){ double w2=vf[i].head(2).squaredNorm(); double S=1.0/(1.0+std::exp(-C1S*phi[i]));
      c+=CF*S*w2; if(AIR_W>0&&phi[i]>0) c+=AIR_W*phi[i]*phi[i]; }
    return c;
  }

  // 상태 매니폴드: sdiff(a→b tangent), sint(a⊕t)
  VectorXd sdiff(const VectorXd&qa,const VectorXd&va,const VectorXd&qb,const VectorXd&vb){
    VectorXd e(2*nv); e.head(nv)=difference(model,qa,qb); e.tail(nv)=vb-va; return e; }
  void sint(const VectorXd&qa,const VectorXd&va,const VectorXd&t,VectorXd&qb,VectorXd&vb){
    qb=integrate(model,qa,VectorXd(t.head(nv))); vb=va+t.tail(nv); }
};
} // namespace cimpc
