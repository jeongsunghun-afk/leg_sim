// biped WBIC — biped_mpc_wbic.wbic_track 이식(단일 전신 QP). MuJoCo 유래량은 인자로 받음(파리티용).
// 변수 z=[q̈(nv); λ(3·Kc)]. eiquadprog(성숙 quad_control.hpp 변환 규약: CE=A·ce0=−b, CI=−G·ci0=h).
#pragma once
#include <eiquadprog/eiquadprog-fast.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
using namespace Eigen;

namespace bipedwbic {

struct WbicIn {
  int nv, nu, Kc;                 // 14, 8, 접촉수
  MatrixXd M;                     // nv×nv
  VectorXd h, qv;                 // nv
  VectorXd q;                     // nu (관절 qpos)
  Vector4d qc;                    // base quat(wxyz)
  Vector3d com; double zref;      // subtree_com, com_ref[2]
  MatrixXd Jc;                    // 3×nv (CoM jac)
  std::vector<int> contacts;      // stance leg 인덱스
  std::vector<MatrixXd> cjac;     // Kc개 3×nv
  std::vector<Vector3d> lam;      // Kc개 (contacts[k]에 대응 MPC GRF)
  bool has_swing; int swing_leg;
  MatrixXd Jsw;                   // 3×nv (swing jac)
  Vector3d sw_pos, sw_ptgt, sw_vtgt;
  VectorXd Qhome;                 // nu
  VectorXd tau_peak;              // nu
  std::vector<int> ankle_idx;     // 발목 관절
  // 게인
  double SW_KP, SW_KD, W_ORI, W_ANKLE, W_POST, W_LAM, STANCE_KD, MU_EFF, LAMZ_MIN;
};

inline VectorXd wbic_track(const WbicIn& in){
  int nv=in.nv, nu=in.nu, Kc=in.Kc, nz=nv+3*Kc;
  auto sl=[&](int k){ return nv+3*k; };
  MatrixXd P=MatrixXd::Zero(nz,nz); VectorXd g=VectorXd::Zero(nz);
  std::vector<int> sw_vidx;
  // 스윙 추종
  if(in.has_swing){
    const MatrixXd& J=in.Jsw;
    Vector3d accel=in.SW_KP*(in.sw_ptgt-in.sw_pos)+in.SW_KD*(in.sw_vtgt-J*in.qv);
    P.topLeftCorner(nv,nv)+=90.0*(J.transpose()*J); g.head(nv)-=90.0*(J.transpose()*accel);
    for(int t=0;t<4;t++) sw_vidx.push_back(6+in.swing_leg*4+t);
  }
  // 자세 레벨링(현재 yaw)
  const Vector4d& qc=in.qc;
  double yaw=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
  double qlev[4]={std::cos(yaw/2),0,0,std::sin(yaw/2)};
  // subQuat(qc, qlev): oerr = 2*(qlev^-1 * qc)_xyz  (mju_subQuat)
  double ql_conj[4]={qlev[0],-qlev[1],-qlev[2],-qlev[3]};
  double dq[4]={ql_conj[0]*qc[0]-ql_conj[1]*qc[1]-ql_conj[2]*qc[2]-ql_conj[3]*qc[3],
                ql_conj[0]*qc[1]+ql_conj[1]*qc[0]+ql_conj[2]*qc[3]-ql_conj[3]*qc[2],
                ql_conj[0]*qc[2]-ql_conj[1]*qc[3]+ql_conj[2]*qc[0]+ql_conj[3]*qc[1],
                ql_conj[0]*qc[3]+ql_conj[1]*qc[2]-ql_conj[2]*qc[1]+ql_conj[3]*qc[0]};
  Vector3d oerr; { double s=(dq[0]<0?-1:1); Vector3d v(dq[1],dq[2],dq[3]); double n=v.norm();
    oerr = (n<1e-12)? Vector3d(0,0,0) : (2.0*std::atan2(n, std::abs(dq[0])) * s / n) * v; }
  for(int j=0;j<2;j++){ double a=150*(-oerr[j])-20*in.qv[3+j]; P(3+j,3+j)+=in.W_ORI; g[3+j]-=in.W_ORI*a; }
  double a_yaw=150*(-oerr[2])-20*in.qv[5]; P(5,5)+=1.0; g[5]-=1.0*a_yaw;
  // CoM 높이
  Vector3d Jcqv=in.Jc*in.qv; double a_z=300*(in.zref-in.com[2])-30*Jcqv[2];
  P.topLeftCorner(nv,nv)+=400.0*(in.Jc.row(2).transpose()*in.Jc.row(2));
  g.head(nv)-=400.0*a_z*in.Jc.row(2).transpose();
  // posture
  auto is_ankle=[&](int j){ for(int a:in.ankle_idx) if(a==j) return true; return false; };
  auto is_sw=[&](int vi){ for(int v:sw_vidx) if(v==vi) return true; return false; };
  for(int j=0;j<nu;j++){ double a=60*(in.Qhome[j]-in.q[j])-5*in.qv[6+j];
    double w = is_ankle(j)?in.W_ANKLE : (is_sw(6+j)?5.0:in.W_POST);
    P(6+j,6+j)+=w; g[6+j]-=w*a; }
  P.topLeftCorner(nv,nv)+=1e-3*MatrixXd::Identity(nv,nv);
  // MPC GRF 추종
  for(int k=0;k<Kc;k++){ P.block(sl(k),sl(k),3,3)+=in.W_LAM*Matrix3d::Identity();
    g.segment(sl(k),3)-=in.W_LAM*in.lam[k]; }
  // 등식: base6 + 접촉(STANCE_KD)
  int neq=6+3*Kc; MatrixXd A=MatrixXd::Zero(neq,nz); VectorXd bb=VectorXd::Zero(neq);
  A.block(0,0,6,nv)=in.M.topRows(6); bb.head(6)=-in.h.head(6);
  for(int k=0;k<Kc;k++){ A.block(0,sl(k),6,3)=-in.cjac[k].leftCols(6).transpose();
    A.block(6+3*k,0,3,nv)=in.cjac[k]; bb.segment(6+3*k,3)=-in.STANCE_KD*(in.cjac[k]*in.qv); }
  // 부등식 Gx≤h: 마찰추 + λz≥min + 토크한계
  std::vector<VectorXd> Gr; std::vector<double> hv;
  int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
  for(int k=0;k<Kc;k++){ int o=sl(k);
    for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nz); r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-in.MU_EFF; Gr.push_back(r); hv.push_back(0.0); }
    VectorXd r=VectorXd::Zero(nz); r[o+2]=-1; Gr.push_back(r); hv.push_back(-in.LAMZ_MIN); }
  MatrixXd Tm=MatrixXd::Zero(nu,nz); Tm.leftCols(nv)=in.M.block(6,0,nu,nv);
  for(int k=0;k<Kc;k++) Tm.block(0,sl(k),nu,3)=-in.cjac[k].block(0,6,3,nu).transpose();
  for(int i=0;i<nu;i++){ Gr.push_back(Tm.row(i)); hv.push_back(in.tau_peak[i]-in.h[6+i]);
                         Gr.push_back(-Tm.row(i)); hv.push_back(in.tau_peak[i]+in.h[6+i]); }
  // eiquadprog: CE x+ce0=0 · CI x+ci0≥0
  P=(0.5*(P+P.transpose())).eval()+1e-8*MatrixXd::Identity(nz,nz);
  MatrixXd CE=A; VectorXd ce0=-bb;
  int nci=(int)Gr.size(); MatrixXd CI(nci,nz); VectorXd ci0(nci);
  for(int i=0;i<nci;i++){ CI.row(i)=-Gr[i]; ci0[i]=hv[i]; }
  VectorXd x(nz);
  eiquadprog::solvers::EiquadprogFast qp; qp.reset(nz,neq,nci);
  auto st=qp.solve_quadprog(P,g,CE,ce0,CI,ci0,x);
  VectorXd tau=VectorXd::Zero(nu);
  if(st==eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL){
    VectorXd qdd=x.head(nv); tau=in.M.block(6,0,nu,nv)*qdd+in.h.segment(6,nu);
    for(int k=0;k<Kc;k++) tau-=in.cjac[k].block(0,6,3,nu).transpose()*x.segment(sl(k),3);
    for(int i=0;i<nu;i++) tau[i]=std::max(-in.tau_peak[i],std::min(in.tau_peak[i],tau[i]));
  }
  return tau;
}

} // namespace bipedwbic
