// 해석 그래디언트 de-risk: ∂b_cc/∂q(dbg=∂(Jcc·qdot_free)/∂q)를 kinematic Hessian으로 해석계산 → FD 대조.
//   convention(LWA) 맞으면 FD와 일치 → 전체 ∂A_cc/∂W 해석화 진행. (Python 트랙이 막힌 지점 재검증)
#include "ci_dyn.hpp"
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <cstdio>
using namespace cimpc; using namespace pinocchio;
using Eigen::VectorXd; using Eigen::MatrixXd;
int main(){
  CiDyn ci("/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf");
  auto&m=ci.model; auto&d=ci.data; int nv=ci.nv;
  VectorXd q=ci.stance_q(), v=VectorXd::Zero(nv); v.setConstant(0.05);
  VectorXd tau=VectorXd::Zero(nv);
  double dt=0.001; VectorXd a_free=aba(m,d,q,v,tau); VectorXd qf=v+dt*a_free;   // qdot_free
  FrameIndex fid=ci.fids[0];
  auto Jlin=[&](const VectorXd&qq){ computeJointJacobians(m,d,qq); updateFramePlacements(m,d);
    MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(m,d,fid,LOCAL_WORLD_ALIGNED,J6); return MatrixXd(J6.topRows(3)); };
  // FD dbg (3×nv)
  MatrixXd dbg_fd(3,nv); double e=1e-6;
  for(int c=0;c<nv;c++){ VectorXd dq=VectorXd::Zero(nv); dq[c]=e;
    MatrixXd Jp=Jlin(integrate(m,q,dq)), Jm=Jlin(integrate(m,q,VectorXd(-dq)));
    dbg_fd.col(c)=(Jp-Jm)*qf/(2*e); }
  // 해석 dbg via kinematic Hessian: H(a,b,c)=∂J[a,b]/∂q_c → dbg[a,c]=Σ_b H(a,b,c)·qf[b]
  computeJointKinematicHessians(m,d,q);
  Tensor<double,3> H(6,nv,nv); H.setZero();
  getFrameKinematicHessian(m,d,fid,LOCAL_WORLD_ALIGNED,H);
  MatrixXd dbg_an(3,nv); dbg_an.setZero();
  for(int a=0;a<3;a++) for(int c=0;c<nv;c++){ double s=0;
    for(int b=0;b<nv;b++) s+=H(a,b,c)*qf[b]; dbg_an(a,c)=s; }
  std::printf("[kinhess] dbg ‖FD‖=%.4e ‖해석‖=%.4e ‖diff‖=%.3e rel=%.3e  %s\n",
    dbg_fd.norm(),dbg_an.norm(),(dbg_fd-dbg_an).norm(),(dbg_fd-dbg_an).norm()/(dbg_fd.norm()+1e-12),
    (dbg_fd-dbg_an).norm()/(dbg_fd.norm()+1e-12)<1e-4?"✅ convention 일치":"✗ 불일치(convention)");

  // ── ∂A_cc/∂q 대조: 부분해석(∂Jcc/∂q만, ∂M⁻¹/∂q 무시) vs FD. ∂M⁻¹/∂q 무시가능한지 판정 ──
  auto Acc0=[&](const VectorXd&qq){ computeJointJacobians(m,d,qq); updateFramePlacements(m,d);
    computeMinverse(m,d,qq); MatrixXd Mi=d.Minv; Mi.triangularView<Eigen::StrictlyLower>()=Mi.transpose().triangularView<Eigen::StrictlyLower>();
    MatrixXd J6=MatrixXd::Zero(6,nv); getFrameJacobian(m,d,fid,LOCAL_WORLD_ALIGNED,J6); MatrixXd J=J6.topRows(3);
    return MatrixXd(J*Mi*J.transpose()); };   // 3×3
  MatrixXd J0=Jlin(q); computeMinverse(m,d,q); MatrixXd Mi0=d.Minv;
  Mi0.triangularView<Eigen::StrictlyLower>()=Mi0.transpose().triangularView<Eigen::StrictlyLower>();
  double fdA=0, diffFull=0, diffPart=0;
  for(int c=0;c<nv;c++){ VectorXd dq=VectorXd::Zero(nv); dq[c]=e;
    MatrixXd dA_fd=(Acc0(integrate(m,q,dq))-Acc0(integrate(m,q,VectorXd(-dq))))/(2*e);   // 3×3 FD(전체)
    MatrixXd dJ(3,nv); for(int a=0;a<3;a++) for(int b=0;b<nv;b++) dJ(a,b)=H(a,b,c);       // ∂Jcc0/∂q_c
    MatrixXd dA_part=dJ*Mi0*J0.transpose()+J0*Mi0*dJ.transpose();                          // 부분(∂M⁻¹항 제외)
    fdA+=dA_fd.squaredNorm(); diffPart+=(dA_fd-dA_part).squaredNorm(); }
  std::printf("[kinhess] ∂A_cc/∂q: ‖FD‖=%.3e · 부분해석(∂M⁻¹무시) rel오차=%.3e  %s\n",
    std::sqrt(fdA), std::sqrt(diffPart/(fdA+1e-20)),
    std::sqrt(diffPart/(fdA+1e-20))<0.05?"✅ ∂M⁻¹항 무시가능(부분해석 충분)":"△ ∂M⁻¹/∂q 필요");
  return 0;
}
