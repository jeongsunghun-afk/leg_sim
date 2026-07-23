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
  const double eps=1e-3, dt=0.002;
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

  std::printf("[C++ ci_relaxed] nq=%d nv=%d nu=%d mass=%.4f\n",nq,nv,nu,computeTotalMass(model));
  std::printf("  ddq[:3]= %.4f %.4f %.4f\n",ddq[0],ddq[1],ddq[2]);
  std::printf("  ddq_dq_fro=%.4f ddq_dv_fro=%.4f ddq_dtau_fro=%.4f\n",ddq_dq.norm(),ddq_dv.norm(),ddq_dtau.norm());
  std::printf("  (Python 기준: ddq[:3]=0.3397 -0.1537 -9.1195 · ddq_dq_fro=926.8784 · ddq_dtau_fro=161.1185)\n");
  return 0;
}
