// biped 전체 컨트롤러 (C++) — Python biped_mpc_wbic+biped_step+biped_wbic 통합 이식.
// MuJoCo C API로 M·h·jac·com 계산 → event-DCM 게이트 + base-frame 발배치 + MPC(50Hz) + WBIC.
#pragma once
#include <mujoco/mujoco.h>
#include "biped_mpc.hpp"
#include "biped_wbic.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstring>
using namespace Eigen;

struct BipedControl {
  mjModel* m; mjData* d;
  int nv, nu, NF=2;
  int sph[2], fbody[2];               // 발 sphere geom·contact body
  // ── 파라미터 (Python 동일) ──
  double T_STEP=0.24, DS_FRAC=0.10, STEP_H=0.05, K_CAP=1.0, CAP_CLAMP=0.22;
  double SW_KP=800, SW_KD=60, K_RETURN=0.45, K_RET_LAT=0.0, K_LAT=0.5, SPREAD=1.0, GAP_MIN=0.14, GAP_MAX=0.34;
  double SS_NOMINAL=0.16, SS_MIN=0.10, SS_MAX=0.45, TRIG_Y=0.03, GVEC=9.81;
  double STANCE_KD=20, W_ORI=5, W_POST=1, W_ANKLE=20, MU_EFF=0.8*0.707, LAMZ_MIN=1;
  double MPC_DT=0.02, W_LAM=10, head_lead=0.15;
  int MPC_N=14, mpc_decim=10;
  double tau_peak8[8]={84,84,126,96,84,84,126,96};
  double Qhome8[8]={0,0.05,-0.2,0, 0,0.05,-0.2,0};
  int ankle_idx[2]={3,7};
  double GEAR[4]={7,7,10.5,8}, ROTOR_I=1e-4, JDAMP=0.1, JFRIC=0.5;
  // ── 상태 ──
  double vx_cmd=0, vy_cmd=0, wz_cmd=0, yaw_des=0;
  Vector2d com0; Vector2d nominal_off[2]; double com_ref_z;
  int stance=1, swing=0; double t_ss=0; long _k=0;
  Matrix<double,2,3> lam; bool have_liftoff[2]={false,false}; Vector3d liftoff[2];
  Matrix3d I_body; double mass;

  BipedControl(mjModel* m_, mjData* d_):m(m_),d(d_){
    nv=m->nv; nu=m->nu;
    sph[0]=mj_name2id(m,mjOBJ_GEOM,"HL_sphere"); sph[1]=mj_name2id(m,mjOBJ_GEOM,"HR_sphere");
    fbody[0]=mj_name2id(m,mjOBJ_BODY,"HL_foot_contact_link"); fbody[1]=mj_name2id(m,mjOBJ_BODY,"HR_foot_contact_link");
    lam.setZero(); setup_gearbox();
  }
  void setup_gearbox(){ for(int j=0;j<nu;j++){ double N=GEAR[j%4]; int dof=6+j;
    m->dof_armature[dof]=ROTOR_I*N*N; m->dof_damping[dof]=JDAMP; m->dof_frictionloss[dof]=JFRIC; } }

  double footz(int leg){ return d->geom_xpos[sph[leg]*3+2]; }
  Vector3d spos(int leg){ return Vector3d(d->geom_xpos[sph[leg]*3],d->geom_xpos[sph[leg]*3+1],d->geom_xpos[sph[leg]*3+2]); }

  MatrixXd foot_jac(int leg){ std::vector<double> jp(3*nv);
    double pt[3]={d->geom_xpos[sph[leg]*3],d->geom_xpos[sph[leg]*3+1],d->geom_xpos[sph[leg]*3+2]};
    mj_jac(m,d,jp.data(),nullptr,pt,fbody[leg]);
    MatrixXd J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jp[r*nv+c]; return J; }
  MatrixXd jac_com(){ std::vector<double> jc(3*nv); mj_jacSubtreeCom(m,d,jc.data(),0);
    MatrixXd J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jc[r*nv+c]; return J; }
  VectorXd qvel(){ return Map<VectorXd>(d->qvel,nv); }
  Vector3d com(){ return Vector3d(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]); }
  double base_yaw(){ double* q=&d->qpos[3];
    return std::atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3])); }

  void compute_Icom(){ mj_forward(m,d); mass=m->body_subtreemass[0];
    Vector3d c=com(); I_body.setZero();
    for(int b=1;b<m->nbody;b++){ double ms=m->body_mass[b]; if(ms<=0) continue;
      Vector3d r(d->xipos[b*3]-c[0],d->xipos[b*3+1]-c[1],d->xipos[b*3+2]-c[2]);
      Map<Matrix<double,3,3,RowMajor>> Rb(&d->ximat[b*9]);
      Vector3d bi(m->body_inertia[b*3],m->body_inertia[b*3+1],m->body_inertia[b*3+2]);
      Matrix3d Ib=Rb*bi.asDiagonal()*Rb.transpose();
      I_body+=Ib+ms*(r.dot(r)*Matrix3d::Identity()-r*r.transpose()); } }

  void reset(){
    for(int i=0;i<nq();i++) d->qpos[i]=0; d->qpos[3]=1;
    for(int j=0;j<nu;j++) d->qpos[7+j]=Qhome8[j];
    d->qpos[2]=0.7; mj_forward(m,d);
    double zmin=1e9; for(int l=0;l<2;l++) zmin=std::min(zmin,footz(l)-m->geom_size[sph[l]*3]);
    d->qpos[2]-=zmin; mj_forward(m,d);
    Vector3d c=com(); com0=c.head(2);
    for(int l=0;l<2;l++) nominal_off[l]=spos(l).head(2)-c.head(2);
    com_ref_z=c[2];
    stance=1; swing=0; t_ss=0; _k=0; yaw_des=0; have_liftoff[0]=have_liftoff[1]=false;
    for(int i=0;i<nv;i++) d->qvel[i]=0;
    compute_Icom();
  }
  int nq(){ return m->nq; }

  // ── event-DCM 게이트 ──
  void step_gait(double dt,int&st,int&sw,double&s){
    Vector3d c=com(); VectorXd qv=qvel(); MatrixXd Jc=jac_com(); Vector2d vcom=(Jc*qv).head(2);
    double z=std::max(c[2]-std::min(footz(0),footz(1)),0.15), w=std::sqrt(GVEC/z);
    double xi_y=c[1]+vcom[1]/w;
    double mid_y=0.5*(d->geom_xpos[sph[0]*3+1]+d->geom_xpos[sph[1]*3+1]);
    double sy=(swing==0)?1.0:-1.0;
    s=std::min(std::max(t_ss/SS_NOMINAL,0.0),1.0);
    bool committed=sy*(xi_y-mid_y)>TRIG_Y;
    if(t_ss>SS_MIN&&(committed||t_ss>SS_MAX)){ std::swap(stance,swing); t_ss=0;
      liftoff[swing]=spos(swing); have_liftoff[swing]=true; st=stance; sw=swing; s=0; return; }
    t_ss+=dt; st=stance; sw=swing;
  }

  // ── base-frame 발배치 (dcm_target) ──
  Vector2d dcm_target(int sw,double s){
    Vector3d c=com(); MatrixXd Jc=jac_com(); VectorXd qv=qvel(); Vector2d vcom_w=(Jc*qv).head(2);
    double z=std::max(c[2]-std::min(footz(0),footz(1)),0.15), w=std::sqrt(GVEC/z);
    double yaw=yaw_des, cy=std::cos(yaw), sy=std::sin(yaw);
    auto to_b=[&](Vector2d v){ return Vector2d(cy*v[0]+sy*v[1],-sy*v[0]+cy*v[1]); };
    auto to_w=[&](Vector2d v){ return Vector2d(cy*v[0]-sy*v[1], sy*v[0]+cy*v[1]); };
    Vector2d v_b=to_b(vcom_w), err_b=to_b(c.head(2)-com0), off=nominal_off[sw];
    double lat=(off[1]>0)?1.0:-1.0;
    double rel_fwd=off[0]+K_CAP*v_b[0]/w+K_RETURN*err_b[0];
    rel_fwd=std::min(std::max(rel_fwd,off[0]-CAP_CLAMP),off[0]+CAP_CLAMP);
    double rel_lat=SPREAD*off[1]+K_LAT*(v_b[1]/w)+K_RET_LAT*err_b[1];
    Vector2d st_b=to_b(spos(1-sw).head(2)-c.head(2));
    double gap=std::min(std::max(lat*(rel_lat-st_b[1]),GAP_MIN),GAP_MAX);
    rel_lat=st_b[1]+lat*gap;
    return c.head(2)+to_w(Vector2d(rel_fwd,rel_lat));
  }
  // swing 궤적
  void swing_traj(int leg,double s,Vector3d&p,Vector3d&v){
    Vector3d p0=liftoff[leg]; Vector2d tgt=dcm_target(leg,s);
    double clr=m->geom_size[sph[leg]*3];    // sphere r
    double gz=std::min(footz(0),footz(1))+clr;
    Vector3d p1(tgt[0],tgt[1],gz);
    double ss=std::min(std::max(s,0.0),1.0);
    double sm=10*ss*ss*ss-15*ss*ss*ss*ss+6*ss*ss*ss*ss*ss;
    double dsm=(30*ss*ss-60*ss*ss*ss+30*ss*ss*ss*ss)/std::max(1e-6,(1-DS_FRAC)*T_STEP);
    p=p0+(p1-p0)*sm; double zl=4*STEP_H*ss*(1-ss);
    p[2]=p0[2]+(p1[2]-p0[2])*sm+zl;
    v=(p1-p0)*dsm; v[2]=(p1[2]-p0[2])*dsm+4*STEP_H*(1-2*ss)*dsm;
  }

  // ── MPC ──
  Matrix<double,2,3> mpc_grf(int stanceLeg){
    using namespace bipedmpc; MpcCfg c; c.N=MPC_N; c.DT=MPC_DT; c.TOTAL_MASS=mass; c.G_ACC=9.81;
    c.MU=MU_EFF; c.LAMZ_MIN=LAMZ_MIN; c.LAMZ_MAX=2.0*mass*9.81; c.I_BODY=I_body;
    double qd[13]={200,200,100,0,0,200,0,0,1,10,10,1,0}; for(int i=0;i<13;i++) c.Qdiag[i]=qd[i];
    c.Rdiag=Vector3d(1e-6,1e-6,1e-6);
    // body_x0
    double Rm[9]; mju_quat2Mat(Rm,&d->qpos[3]); Map<Matrix<double,3,3,RowMajor>> R(Rm);
    double pitch=std::asin(std::max(-1.0,std::min(1.0,-R(2,0))));
    double roll=std::atan2(R(2,1),R(2,2)), yaw=std::atan2(R(1,0),R(0,0));
    MatrixXd Jc=jac_com(); VectorXd qv=qvel(); Vector3d vcom=Jc*qv;
    Vector3d wb(d->qvel[3],d->qvel[4],d->qvel[5]), ow=R*wb; Vector3d cc=com();
    Matrix<double,13,1> x0; x0<<roll,pitch,yaw, cc[0],cc[1],cc[2], ow[0],ow[1],ow[2], vcom[0],vcom[1],vcom[2], -9.81;
    Vector3d frel[2]; for(int i=0;i<2;i++) frel[i]=spos(i)-cc;
    std::array<int,2> cur={stanceLeg==0?1:0, stanceLeg==1?1:0};
    std::vector<std::array<int,2>> cs(MPC_N,cur);
    std::array<Vector3d,2> fp0={frel[0],frel[1]}; std::vector<std::array<Vector3d,2>> fp(MPC_N,fp0);
    double ya=base_yaw(), cya=std::cos(ya),sya=std::sin(ya);   // ★속도명령=실제 base yaw(base-relative, 17-DOF yaw_m 방식)
    double vxw=cya*vx_cmd-sya*vy_cmd, vyw=sya*vx_cmd+cya*vy_cmd;
    Matrix<double,13,1> xr; xr<<0,0,yaw_des, cc[0],cc[1],com_ref_z, 0,0,wz_cmd, vxw,vyw,0, -9.81;  // 헤딩참조=yaw_des
    return mpc_qp_plan(c,x0,cs,fp,xr);
  }

  // ── WBIC (MPC lam 추종) ──
  void wbic(int stanceLeg,int sw,const Vector3d&ptgt,const Vector3d&vtgt){
    using namespace bipedwbic; WbicIn in; in.nv=nv; in.nu=nu; in.Kc=1;
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    in.M=Map<Matrix<double,Dynamic,Dynamic,RowMajor>>(Mb.data(),nv,nv);
    in.h=Map<VectorXd>(d->qfrc_bias,nv); in.qv=qvel();
    in.q=Map<VectorXd>(&d->qpos[7],nu); for(int i=0;i<4;i++) in.qc[i]=d->qpos[3+i];
    in.com=com(); in.zref=com_ref_z; in.Jc=jac_com();
    in.contacts={stanceLeg}; in.cjac={foot_jac(stanceLeg)}; in.lam={lam.row(stanceLeg).transpose()};
    in.has_swing=true; in.swing_leg=sw; in.Jsw=foot_jac(sw); in.sw_pos=spos(sw);
    in.sw_ptgt=ptgt; in.sw_vtgt=vtgt;
    in.Qhome=Map<VectorXd>(Qhome8,nu); in.tau_peak=Map<VectorXd>(tau_peak8,nu);
    in.ankle_idx={ankle_idx[0],ankle_idx[1]};
    in.SW_KP=SW_KP; in.SW_KD=SW_KD; in.W_ORI=W_ORI; in.W_ANKLE=W_ANKLE; in.W_POST=W_POST;
    in.W_LAM=W_LAM; in.STANCE_KD=STANCE_KD; in.MU_EFF=MU_EFF; in.LAMZ_MIN=LAMZ_MIN;
    VectorXd tau=wbic_track(in); for(int i=0;i<nu;i++) d->ctrl[i]=tau[i];
  }

  void control(double dt){
    yaw_des+=wz_cmd*dt;
    double ya=base_yaw(); double lag=std::atan2(std::sin(yaw_des-ya),std::cos(yaw_des-ya));
    yaw_des=ya+std::min(std::max(lag,-head_lead),head_lead);
    double cya=std::cos(ya),sya=std::sin(ya);   // ★복귀목표 이동=실제 base yaw 기준(base-relative)
    com0[0]+=(cya*vx_cmd-sya*vy_cmd)*dt; com0[1]+=(sya*vx_cmd+cya*vy_cmd)*dt;
    int st,sw; double s; step_gait(dt,st,sw,s);
    if(_k%mpc_decim==0) lam=mpc_grf(st);
    _k++;
    if(!have_liftoff[sw]){ liftoff[sw]=spos(sw); have_liftoff[sw]=true; }
    Vector3d p,v; swing_traj(sw,s,p,v);
    wbic(st,sw,p,v);
  }
};
