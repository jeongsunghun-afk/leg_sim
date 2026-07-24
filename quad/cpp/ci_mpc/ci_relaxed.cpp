// CI-MPC C++ 포트 — 1단계: relaxed 상보성 접촉 그래디언트(dyn_derivs_relaxed) Pinocchio C++ 포팅.
//   A_cc=J M⁻¹ Jᵀ+εI, λ=-(A_cc+εI)⁻¹ b_cc, ddq_eff=a_free+M⁻¹Jᵀλ/dt. ∂λ 이미지공식(해석 역행렬)
//   +ABA도함수(해석)+기하항(∂A_cc/∂q 등 소행렬 FD). Python dyn_derivs_relaxed와 값 대조 검증.
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>
#include "ci_dyn.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

using namespace pinocchio;
using Eigen::VectorXd; using Eigen::MatrixXd;

static VectorXd readvec(const std::string& path, const std::string& name, int n){
  std::ifstream fs(path); std::string line;
  while(std::getline(fs,line)){ std::istringstream ss(line); std::string tag; ss>>tag;
    if(tag==name){ VectorXd x(n); for(int i=0;i<n;i++) ss>>x[i]; return x; } }
  throw std::runtime_error("state.txt: "+name+" 없음");
}

int main(){
  const std::string urdf_path="/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf";
  Model full; pinocchio::urdf::buildModel(urdf_path, JointModelFreeFlyer(), full);
  std::vector<JointIndex> lock={ full.getJointId("FB_waist_joint") };
  Model model; buildReducedModel(full, lock, neutral(full), model);
  Data data(model);
  const int nv=model.nv, nq=model.nq, nu=nv-6;
  double arm[4]={1e-4*49.0, 1e-4*49.0, 1e-4*110.25, 1e-4*70.56};   // tile [7²,7²,10.5²,8.4²]×1e-4
  for(int leg=0;leg<4;leg++) for(int j=0;j<4;j++) model.armature[6+leg*4+j]=arm[j];
  std::vector<std::string> feet={"FL","FR","HL","HR"};
  std::vector<FrameIndex> fids; for(auto&L:feet) fids.push_back(model.getFrameId(L+"_foot_contact_link"));
  const double eps=1e-3, dt=0.001;
  const std::string sp="/home/jsh/문서/jsh/simulation/quad/cpp/ci_mpc/state.txt";
  VectorXd q=readvec(sp,"q",nq), v=readvec(sp,"v",nv), u=readvec(sp,"u",nu);
  VectorXd tau_full(nv); tau_full.head(6).setZero(); tau_full.tail(nu)=u;
  std::vector<int> active={0,1,2,3}; int nc=3*(int)active.size();

  computeABADerivatives(model,data,q,v,tau_full);
  VectorXd a_free=data.ddq; MatrixXd aq=data.ddq_dq, av=data.ddq_dv;
  MatrixXd Minv=data.Minv; Minv.triangularView<Eigen::StrictlyLower>()=Minv.transpose().triangularView<Eigen::StrictlyLower>();
  MatrixXd au=Minv.rightCols(nu);

  auto geom=[&](const VectorXd& qq, MatrixXd& Jcc, MatrixXd& W, MatrixXd& Acc){
    computeJointJacobians(model,data,qq); updateFramePlacements(model,data);
    computeMinverse(model,data,qq); MatrixXd Mi=data.Minv;
    Mi.triangularView<Eigen::StrictlyLower>()=Mi.transpose().triangularView<Eigen::StrictlyLower>();
    Jcc.resize(nc,nv);
    for(size_t k=0;k<active.size();k++){ MatrixXd J6=MatrixXd::Zero(6,nv);
      getFrameJacobian(model,data,fids[active[k]],LOCAL_WORLD_ALIGNED,J6); Jcc.middleRows(3*k,3)=J6.topRows(3); }
    W=Mi*Jcc.transpose(); Acc=Jcc*W;
  };
  MatrixXd Jcc,W,Acc; geom(q,Jcc,W,Acc);
  VectorXd qdot_free=v+dt*a_free;
  MatrixXd Ari=(Acc+eps*MatrixXd::Identity(nc,nc)).inverse();
  VectorXd bcc=Jcc*qdot_free, lam=-Ari*bcc, ddq=a_free+W*lam/dt;

  std::vector<MatrixXd> dAcc(nv, MatrixXd::Zero(nc,nc));
  MatrixXd dbg=MatrixXd::Zero(nc,nv), dWl=MatrixXd::Zero(nv,nv);
  double e=1e-6;
  for(int j=0;j<nv;j++){ VectorXd dq=VectorXd::Zero(nv); dq[j]=e;
    MatrixXd Jp,Wp,Ap,Jm,Wm,Am;
    geom(integrate(model,q,dq),Jp,Wp,Ap); geom(integrate(model,q,VectorXd(-dq)),Jm,Wm,Am);
    dAcc[j]=(Ap-Am)/(2*e); dbg.col(j)=(Jp-Jm)*qdot_free/(2*e); dWl.col(j)=(Wp-Wm)*lam/(2*e); }
  VectorXd y=Ari*bcc; MatrixXd dl_dq(nc,nv);
  for(int j=0;j<nv;j++) dl_dq.col(j)=Ari*(dAcc[j]*y);
  dl_dq-=Ari*(dbg+Jcc*(dt*aq));
  MatrixXd dl_dv=-Ari*(Jcc*(MatrixXd::Identity(nv,nv)+dt*av));
  MatrixXd dl_du=-Ari*(Jcc*(dt*au));
  MatrixXd ddq_dq=aq+(dWl+W*dl_dq)/dt, ddq_dv=av+(W*dl_dv)/dt, ddq_dtau=au+(W*dl_du)/dt;

  // ── tangent 선형화 A,B (lin_AB_relaxed): dIntegrate 연쇄 ──
  VectorXd v_next=v+dt*ddq, w=dt*v_next;
  MatrixXd dvn_dq=dt*ddq_dq, dvn_dv=MatrixXd::Identity(nv,nv)+dt*ddq_dv, dvn_du=dt*ddq_dtau;
  MatrixXd dInt0=MatrixXd::Zero(nv,nv),dInt1=MatrixXd::Zero(nv,nv);   // ★dIntegrate 전 setZero 필수(블록대각만 씀)
  dIntegrate(model,q,w,dInt0,ARG0); dIntegrate(model,q,w,dInt1,ARG1);
  MatrixXd dqn_dq=dInt0+dInt1*(dt*dvn_dq), dqn_dv=dInt1*(dt*dvn_dv), dqn_du=dInt1*(dt*dvn_du);
  MatrixXd A(2*nv,2*nv); A<<dqn_dq,dqn_dv,dvn_dq,dvn_dv;
  MatrixXd B(2*nv,nu);   B<<dqn_du,dvn_du;

  std::printf("[C++ ci_relaxed] nq=%d nv=%d nu=%d mass=%.4f\n",nq,nv,nu,computeTotalMass(model));
  std::printf("  ddq[:3]= %.4f %.4f %.4f\n",ddq[0],ddq[1],ddq[2]);
  std::printf("  ddq_dq_fro=%.4f ddq_dv_fro=%.4f ddq_dtau_fro=%.4f\n",ddq_dq.norm(),ddq_dv.norm(),ddq_dtau.norm());
  std::printf("  ★tangent A_fro=%.6f B_fro=%.6f\n",A.norm(),B.norm());
  // ── ci_dyn.hpp CiDyn(라이브러리)의 lin_AB/dyn_relaxed와 대조 = 회귀 가드 ──
  //   ★dIntegrate setZero 버그 재발 방지: 미초기화면 CiDyn 쪽 A가 부풀어 불일치로 잡힘.
  { cimpc::CiDyn cd(urdf_path); cd.eps=eps; cd.relax_mode="eps";   // 수동코드=εI라 εI로 대조(setZero 가드)
    MatrixXd Acd,Bcd; cd.lin_AB(q,v,u,dt,Acd,Bcd);
    double dA=(A-Acd).norm();
    std::printf("  ★[대조] 수동 A_fro=%.6f  vs  CiDyn.lin_AB A_fro=%.6f  ‖diff‖=%.2e  %s\n",
                A.norm(),Acd.norm(),dA, dA<1e-9?"✅ 일치":"✗ 불일치(setZero 확인)"); }
  { cimpc::CiDyn cd(urdf_path); cd.relax_mode="D"; cd.rho_relax=1e-4;   // 논문판 ρD → Python ρD와 대조
    MatrixXd A2,B2; cd.lin_AB(q,v,u,dt,A2,B2);
    std::printf("  ★[ρD] CiDyn.lin_AB(D) A_fro=%.6f  (Python ρD 값과 대조)\n", A2.norm()); }
  // ── ★해석 그래디언트 vs FD 대조(같은 relax 모드): kinematic Hessian+RNEA트릭이 FD와 일치해야 ──
  { cimpc::CiDyn cf(urdf_path), ca(urdf_path); ca.analytic_grad=true;
    MatrixXd Af,Bf,Aa,Ba; cf.lin_AB(q,v,u,dt,Af,Bf); ca.lin_AB(q,v,u,dt,Aa,Ba);
    double dA=(Af-Aa).norm(), dB=(Bf-Ba).norm();
    std::printf("  ★[해석] FD A_fro=%.6f vs 해석 A_fro=%.6f  ‖A diff‖=%.2e ‖B diff‖=%.2e  %s\n",
                Af.norm(),Aa.norm(),dA,dB, dA<1e-5?"✅ 해석=FD(그래디언트 정확)":"✗ 불일치"); }
  // ── ★foot-slip cost ∂c/∂q(접선속도 ∂vt/∂q 포함) 해석 vs FD ──
  { cimpc::CiDyn cd(urdf_path); cd.analytic_grad=true; cd.CF=2500; cd.AIR_W=100; cd.SYM=0;
    double c0; VectorXd g; MatrixXd Hh; cd.foot_slip_cost(q,v,c0,g,Hh);
    VectorXd gq_an=g.head(nv), gq_fd(nv); double e2=1e-6;
    for(int j=0;j<nv;j++){ VectorXd dq=VectorXd::Zero(nv); dq[j]=e2;
      gq_fd[j]=(cd.foot_val(integrate(model,q,dq),v)-cd.foot_val(integrate(model,q,VectorXd(-dq)),v))/(2*e2); }
    double dg=(gq_an-gq_fd).norm(), rel=dg/(gq_fd.norm()+1e-12);
    std::printf("  ★[foot-slip ∂c/∂q] 해석‖=%.4f FD‖=%.4f  ‖diff‖=%.2e rel=%.2e  %s\n",
                gq_an.norm(),gq_fd.norm(),dg,rel, rel<1e-4?"✅ ∂vt/∂q exact":"✗ 불일치"); }
  return 0;
}
