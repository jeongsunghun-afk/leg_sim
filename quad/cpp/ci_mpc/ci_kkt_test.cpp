// step_kkt(hard active-set forward) 검증: 서기 안정 + 발 lift 가능(relaxed와 달리 발이 접촉 떠남).
#include "ci_dyn.hpp"
#include <cstdio>
using namespace cimpc;
using Eigen::VectorXd; using Eigen::Vector3d; using Eigen::MatrixXd;
int main(){
  setvbuf(stdout,nullptr,_IONBF,0);
  CiDyn ci("/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf");
  int nv=ci.nv, nu=ci.nu; ci.margin=0.02;
  VectorXd qstar=ci.stance_q(), q=qstar, v=VectorXd::Zero(nv), th=VectorXd::Zero(nu);
  auto footmin=[&](const VectorXd&q,const VectorXd&v){ std::vector<double>phi;std::vector<MatrixXd>J;std::vector<Vector3d>vf;
    ci.foot_kin(q,v,phi,J,vf); return phi; };
  // 서기 settle(step_kkt)
  for(int i=0;i<400;i++){ th=150.0*(qstar.tail(nu)-q.tail(nu))-8.0*v.tail(nu);
    VectorXd qn,vn; ci.step_kkt(q,v,th,0.001,2,qn,vn); q=qn; v=vn; }
  auto phi=footmin(q,v);
  std::printf("[step_kkt] 서기 settle: base_z=%.4f |v|=%.4f 발높이 FL=%.4f FR=%.4f HL=%.4f HR=%.4f\n",
              q[2],v.norm(),phi[0],phi[1],phi[2],phi[3]);
  // 발 lift 테스트: FL 다리 관절에 굽힘 토크(무릎 당겨 발 들기) → relaxed면 안 뜸, kkt면 뜸
  VectorXd q2=q,v2=v,th2=VectorXd::Zero(nu); th2[2]=60.0;   // FL calf 굽힘(발 들어올림 시도)
  for(int i=0;i<200;i++){ VectorXd qn,vn; ci.step_kkt(q2,v2,th2,0.001,2,qn,vn); q2=qn; v2=vn; }
  auto phi2=footmin(q2,v2);
  std::printf("[step_kkt] FL calf 굽힘 200스텝: FL발높이 %.4f→%.4f (>0=발 뜸=스텝 가능!)  base_z=%.3f\n",
              phi[0],phi2[0],q2[2]);
  return 0;
}
