// biped 전체 컨트롤러 (C++) — Python biped_mpc_wbic+biped_step+biped_wbic 통합 이식.
#include <cstdlib>
// MuJoCo C API로 M·h·jac·com 계산 → event-DCM 게이트 + base-frame 발배치 + MPC(50Hz) + WBIC.
#pragma once
#include <mujoco/mujoco.h>
#include "biped_mpc.hpp"
#include "biped_wbic.hpp"
#include "biped_zmp.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstring>
using namespace Eigen;

struct BipedControl {
  mjModel* m; mjData* d;
  int nv, nu, NF=2;
  int sph[2], fbody[2];               // 발 sphere geom(tip)·contact body
  int sph2[2]={-1,-1}; bool has_heel=false;   // ★평발 heel 접촉구(foot_link 원점)
  int cmode=0;                        // 접촉모드: 0=1점(점발 stepping)·1=2점(평발 정적). 통합모델 기본 평발.
  double Qflat8[8]={0,0.25,-0.50,-1.14626, 0,0.25,-0.50,-1.14626};   // ★평발 home(발목 눕힘, CoM 밑창중심)
  // ── 파라미터 (Python 동일) ──
  // ★T_STEP 0.24 → 0.32 (2026-08-05). 실측 ROTOR_I(7.4e-4, 구 placeholder 의 7.4배)를
  //   넣으면 반사관성이 7.4배가 되어 0.24s 스텝의 스윙 가속에 필요한 토크가 tau_peak 을
  //   넘어 QP 가 포화 → 2.18s 낙상. 필요가속도 ∝ 1/T² 이므로 스텝을 늦추는 것이 해법이다.
  //   ⚠ 스윙게인을 올리는 것은 역효과였다(SW_KP 800→1600/3200/5920 = 1.16/1.10/0.65s 낙상)
  //     — 대역 부족이 아니라 토크 포화이기 때문. 실측 스윕:
  //       ROTOR_I 1e-4/2e-4/4e-4/5e-4 = 15s 무낙상 · 6e-4 = 9.4s · 7.4e-4 = 2.18s 낙상
  //       7.4e-4 + T_STEP 0.32 = 15s 무낙상 tilt 2.7°(전 설정 중 최량) · 0.40/0.50 = 낙상
  //   ⚠ vx=0.15 단일 조건의 4점 스윕으로 잡은 값이다. 속도대역 전반 재검증 필요.
  // ★2026-08-05 재튜닝: T_STEP 0.32→0.38, K_RETURN 0.45→0.15.
  //   leg-odom 야코비안 편향(구중심 vs 접촉점)을 제거하자 기존 튜닝이 성립하지 않았다 —
  //   기존 값은 그 편향을 전제로 맞춰져 있었다. 실측 센서노이즈까지 넣고 재스윕한 결과다.
  //   상세: cpp/STABILITY_MAP.md
  double T_STEP=0.38, DS_FRAC=0.10, STEP_H=0.06, K_CAP=1.0, CAP_CLAMP=0.22;
  double SW_KP=800, SW_KD=60, K_RETURN=0.15, K_RET_LAT=0.0, K_LAT=0.5, SPREAD=1.0, GAP_MIN=0.14, GAP_MAX=0.34;
  double SS_NOMINAL=0.16, SS_MIN=0.10, SS_MAX=0.45, TRIG_Y=0.03, GVEC=9.81;
  double FLAT_KCAP=0.6;               // ★평발 전후 capture 게인(발목ZMP가 주 균형, 약한 보조)
  double FLAT_WANK=150;               // ★평발 보행 발목 flat 고정 가중(밑창 유지)
  double FLAT_WLAM=2;                 // ★평발 보행 MPC GRF 추종 가중(↓=WBIC 높이/CoM task 지배)
  double czwalk=0;                    // ★평발 보행 CoM 높이(0=reset값). 튜닝용
  double FLAT_WORI=5;                 // ★평발 보행 base pitch/roll 레벨링 가중
  double FLAT_WLEG=0.05;              // ★평발 정적 thigh/calf posture 가중(낮음=CoM 높이 조절 가능)
  double STANCE_KD=20, W_ORI=5, W_POST=1, W_ANKLE=20, MU_EFF=0.8*0.707, LAMZ_MIN=1;
  double MPC_DT=0.02, W_LAM=10, head_lead=0.15;
  int MPC_N=14, mpc_decim=10;
  // ★2026-08-06: 하드코딩 폐기 → init() 에서 **MJCF jnt_actfrcrange 에서 읽는다**
  //   (quad 와 동일 패턴: quad/cpp/src/quad_control.hpp:81, quad_mpc_wbic_17dof.py:209)
  //   종전 {84,84,126,96,...} 은 foot 이 96 = 12Nm×8 로, GEAR 를 8→8.4 로 고칠 때
  //   따라가지 않아 tau_peak÷gear 가 11.43(≠모터 피크 12.0)이 돼 있었다.
  //   ⇒ 감속비를 바꾸면 토크한계도 따라가야 한다. MJCF 를 단일 출처로 삼아 그걸 강제한다.
  double tau_peak8[8]={84,84,126,100.8,84,84,126,100.8};   // init() 이 MJCF 값으로 덮어씀
  // ★2026-08-12 새 CAD(몸통 placeholder→실측)로 재산출. 구값 (0.05,−0.2) 폐기.
  //   고관절 부착점이 26cm 이동해 구 자세는 HOME 에서 CoM 이 지지중심보다 6cm 앞(구 1.6cm)이었고,
  //   그 오차가 nominal_off 에 스폰 시점에 굳어 매 스텝 반복 → 전방 폭주로 1초 내 낙상했다.
  //   기준: nominal_off_x=+0.02 · 다리높이 0.4651(구와 동일). 상세는 biped_wbic.py Q_HOME 주석.
  //   검증: 15s × 8조건(정지·전진 0.05~0.20·후진·측방·선회) 8/8 무낙상, tilt 3.0~4.1°.
  //   ★T_STEP 은 배포값 0.38 그대로다 — 바뀐 것은 자세뿐이다.
  double Qhome8[8]={0,0.203054,-0.671148,0, 0,0.203054,-0.671148,0};
  int ankle_idx[2]={3,7};
  // ── 액추에이터 물리 — ★2026-08-05 실기 실측 (emb/pace/RESULTS.md) ──
  //   HL_hip·HR_hip 을 PACE 처프로 식별. 전 관절이 동일 모터+7:1 이고 관절별 추가
  //   감속단만 붙으므로 ROTOR_I(모터축 관성)는 **전 관절 공통 상수**다.
  //     ROTOR_I 7.652e-4(HL) / 7.121e-4(HR) → 7.4e-4 (양축 7% 일치).
  //             구 placeholder 1e-4 는 7.4배 과소였다.
  //     JDAMP   0.096~0.102(HL) / 0.071(HR) → 0.09. 등속스윕은 속도가 낮아 점성이
  //             신호에 안 잡히므로(HR 은 음수까지 나옴) **처프값**을 쓴다.
  //     JFRIC   처프 0.375(HL) / 0.382(HR) → 0.38. 저속 정지·유지는 0.50~0.52 인데
  //             Stribeck 때문이며, 보행은 동적 영역이라 처프값이 대표값이다.
  //   ⚠ 실측은 hip 2축·다리 미장착 상태. thigh/calf/foot 의 JDAMP/JFRIC 은 감속단이
  //     늘면 마찰도 늘어 달라진다(ROTOR_I 와 달리 공통 상수가 아님) → 장착 후 재측정.
  //   ⚠ GEAR foot 8 → 8.4 (총 감속비 8.4 = 7×1.2 추가단, 사용자 확인 2026-08-05).
  double GEAR[4]={7,7,10.5,8.4}, ROTOR_I=7.4e-4, JDAMP=0.09, JFRIC=0.38;
  // ── 상태 ──
  double vx_cmd=0, vy_cmd=0, wz_cmd=0, yaw_des=0, yaw_hold=0; bool yaw_hold_set=false;   // ★heading-hold latch
  Vector2d com0; Vector2d nominal_off[2]; double com_ref_z; Vector2d com_ref_xy;   // ★2점 정적 CoM xy 목표
  Vector4d foot_home_quat[2];         // ★평발 swing 발 수평 목표(home world quat, yaw=0)
  int stance=1, swing=0; double t_ss=0; long _k=0; bool walk_init=true; double walk_init_t=0;   // ★평발 보행개시 weight-shift
  // ── ZMP 프리뷰 보행(평발) ──
  ZmpPreview pv; long zkk=-1; double zanchor_x=0, zaf_y[2]={0,0}, z_sx=0;   // 발 앵커·스텝전진
  double cxr=0,vxr=0,cyr=0,vyr=0; int prev_ctr=0;                          // preview CoM ref
  double T_SS_Z=0.32; int PREV_DECIM=5; bool in_zmp_walk=false;            // 공칭 SS시간·preview 데시메이션
  long zlead=0;                                                            // ZMP 리드인 DS 잔여 틱
  // ── 오프라인 1점/2점 전환(toe-pivot 굴림 궤적) ──
  bool trans_on=false; double trans_t=0, T_TRANS=1.4; int trans_to=0;      // 전환중·타이머·목표모드
  double q_from[8], q_to[8], q_live[8], cz_from=0, cz_to=0;                // 자세·높이 보간
  Matrix<double,2,3> lam; bool have_liftoff[2]={false,false}; Vector3d liftoff[2];
  Matrix3d I_body; double mass;

  BipedControl(mjModel* m_, mjData* d_):m(m_),d(d_){
    nv=m->nv; nu=m->nu;
    // ★tau_peak 을 MJCF 에서 읽는다(quad_control.hpp:81 과 동일 패턴).
    //   hinge 관절만 골라 dof 순서(=actuator 순서)로 채운다. 값이 없으면(≤0) 무한대.
    { int k=0;
      for(int j=0;j<m->njnt && k<8;j++){
        if(m->jnt_type[j]!=mjJNT_HINGE) continue;
        double frc=m->jnt_actfrcrange[j*2+1];
        tau_peak8[k++] = (frc>0) ? frc : 1e8;
      }
    }
    sph[0]=mj_name2id(m,mjOBJ_GEOM,"HL_sphere"); sph[1]=mj_name2id(m,mjOBJ_GEOM,"HR_sphere");
    fbody[0]=mj_name2id(m,mjOBJ_BODY,"HL_foot_contact_link"); fbody[1]=mj_name2id(m,mjOBJ_BODY,"HR_foot_contact_link");
    sph2[0]=mj_name2id(m,mjOBJ_GEOM,"HL_sphere2"); sph2[1]=mj_name2id(m,mjOBJ_GEOM,"HR_sphere2");
    has_heel=(sph2[0]>=0 && sph2[1]>=0);        // ★heel 구 보유=통합모델. 기본 평발(2점) 정적 rest.
    cmode = has_heel ? 1 : 0;
    if(getenv("FLAT_KCAP")) FLAT_KCAP=atof(getenv("FLAT_KCAP"));   // 튜닝용 env
    // ★스윙 게인·스텝시간 env — 반사관성(ROTOR_I)이 커지면 스윙 추종 대역이 부족해져
    //   착지가 틀어진다. 실측 armature 하에서 재튜닝하기 위한 노브.
    if(getenv("SW_KP")) SW_KP=atof(getenv("SW_KP"));
    if(getenv("SW_KD")) SW_KD=atof(getenv("SW_KD"));
    if(getenv("T_STEP")) T_STEP=atof(getenv("T_STEP"));
    if(getenv("FLAT_STEPH")) STEP_H=atof(getenv("FLAT_STEPH"));
    if(getenv("FLAT_WLAM")) FLAT_WLAM=atof(getenv("FLAT_WLAM"));
    if(getenv("FLAT_CZ")) czwalk=atof(getenv("FLAT_CZ"));
    if(getenv("FLAT_WORI")) FLAT_WORI=atof(getenv("FLAT_WORI"));
    if(getenv("T_TRANS")) T_TRANS=atof(getenv("T_TRANS"));
    // ★발디딤 게인 env — leg-odom 야코비안 편향(구중심 vs 접촉점)을 제거하면
    //   K_RETURN 이 보던 오차의 성격이 바뀐다. 편향 위에 얹혀 튜닝돼 있던 값이므로
    //   추정기 수정과 반드시 짝지어 재튜닝해야 한다.
    if(getenv("K_RETURN")) K_RETURN=atof(getenv("K_RETURN"));
    if(getenv("K_CAP"))    K_CAP   =atof(getenv("K_CAP"));
    if(getenv("SS_NOMINAL")) SS_NOMINAL=atof(getenv("SS_NOMINAL"));
    pv.init(PREV_DECIM*0.002, 0.362);          // ★ZMP 프리뷰 게인(dt=preview간격, zc=평발 CoM높이)
    lam.setZero(); setup_gearbox();
  }
  void setup_gearbox(){
    // ★env 오버라이드(quad_mpc_wbic_17dof.py:259-261 규약과 동일) — 재빌드 없이 스윕/회귀비교용.
    //   미지정이면 위 실측 기본값을 쓴다.
    if(const char* e=getenv("ROTOR_I")) ROTOR_I=atof(e);
    if(const char* e=getenv("JDAMP"))   JDAMP  =atof(e);
    if(const char* e=getenv("JFRIC"))   JFRIC  =atof(e);
    if(const char* e=getenv("GEAR_FOOT")) GEAR[3]=atof(e);
    for(int j=0;j<nu;j++){ double N=GEAR[j%4]; int dof=6+j;
      m->dof_armature[dof]=ROTOR_I*N*N; m->dof_damping[dof]=JDAMP; m->dof_frictionloss[dof]=JFRIC; }
    foot_rotor_to_tendon(); }

  // ★foot 로터 반사관성을 dof_armature 에서 **tendon 으로 옮긴다**(calf→foot 기구 커플링).
  //   foot 로터는 관절각이 아니라 raw 각으로 돈다(실기 coef=+1, biped_emb.yaml):
  //       raw_foot = q_foot + coef*q_calf
  //   ⇒ 로터 KE = ½·I_rot·N²·(q̇_foot + coef·q̇_calf)² 라 반사관성이 (calf,foot) **비대각**이다:
  //       M += a*[[coef², coef],[coef, 1]]
  //   ⚠dof_armature 는 M 의 **대각뿐**이라 표현 불가. fixed tendon 의 armature 가 위 형태를 만든다.
  //   ⚠**옮기는** 것이지 더하는 게 아니다 — dof_armature[foot] 을 0 으로 안 두면 이중 계상.
  //   ⚠축별 측정에선 이 항이 죽어 있었다(타축 고정). 전축 동시 가진에서만 살아난다.
  //   검증(2026-08-12 HOME): M[foot,foot] 불변 · M[calf,calf] +46% · M[calf,foot] 0.0045→0.0567
  void foot_rotor_to_tendon(){
    int t[2]={mj_name2id(m,mjOBJ_TENDON,"HL_foot_rotor"),
              mj_name2id(m,mjOBJ_TENDON,"HR_foot_rotor")};
    if(t[0]<0||t[1]<0){   // 구 MJCF(tendon 없음) 호환 — 커플링 누락 상태로 돈다
      fprintf(stderr,"  ⚠MJCF 에 *_foot_rotor tendon 이 없다 — calf↔foot 커플 반사관성 누락\n");
      return; }
    for(int j=0;j<nu;j++) if(j%4==3) m->dof_armature[6+j]=0.0;   // ★대각에서 뺀다
    for(int k=0;k<2;k++) m->tendon_armature[t[k]]=ROTOR_I*GEAR[3]*GEAR[3]; }

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

  // ── 평발(2점) 헬퍼 ──
  const double* Qcur(){ if(trans_on) return q_live; return (has_heel&&cmode==1)?Qflat8:Qhome8; }   // 전환중=보간자세
  Vector3d gpos(int geom){ return Vector3d(d->geom_xpos[geom*3],d->geom_xpos[geom*3+1],d->geom_xpos[geom*3+2]); }
  Vector3d foot_center(int leg){ if(cmode==1&&has_heel) return 0.5*(gpos(sph[leg])+gpos(sph2[leg])); return gpos(sph[leg]); }
  MatrixXd foot_jac_at(int geom,int body){ std::vector<double> jp(3*nv);
    double pt[3]={d->geom_xpos[geom*3],d->geom_xpos[geom*3+1],d->geom_xpos[geom*3+2]};
    mj_jac(m,d,jp.data(),nullptr,pt,body);
    MatrixXd J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jp[r*nv+c]; return J; }
  // 발 중심 자코비안(swing task) = 모드별 기준구(점발=tip / 평발=heel+toe) 평균
  MatrixXd foot_jac_center(int leg){ if(cmode==1&&has_heel)
      return 0.5*(foot_jac_at(sph[leg],(int)m->geom_bodyid[sph[leg]])+foot_jac_at(sph2[leg],(int)m->geom_bodyid[sph2[leg]]));
    return foot_jac(leg); }
  // swing 발 회전 자코비안(수평 유지용)
  MatrixXd foot_jacr(int leg){ std::vector<double> jr(3*nv);
    mj_jac(m,d,nullptr,jr.data(),&d->xpos[fbody[leg]*3],fbody[leg]);
    MatrixXd J(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) J(r,c)=jr[r*nv+c]; return J; }
  // 접촉점(적응): 지면 근처 구만. (geom,body) 리스트.
  std::vector<std::pair<int,int>> contact_pts(std::vector<int> stance){
    std::vector<std::pair<int,int>> pts;
    for(int f:stance){ std::vector<int> in; int cand[2]={sph[f], has_heel?sph2[f]:-1};
      for(int g:cand){ if(g<0) continue; if(d->geom_xpos[g*3+2] < m->geom_size[g*3]+0.012) in.push_back(g); }
      if(in.empty()) in.push_back(sph[f]);
      for(int g:in) pts.push_back({g, m->geom_bodyid[g]}); }
    return pts; }

  // ── 2점 정적 양발지지 QP (Python wbic_stance 이식) ──
  void wbic_stance(){
    using namespace bipedwbic;
    auto cpts=contact_pts({0,1}); int K=(int)cpts.size(); int nz=nv+3*K;
    std::vector<double> Mb(nv*nv); mj_fullM(m,Mb.data(),d->qM);
    Map<Matrix<double,Dynamic,Dynamic,RowMajor>> M(Mb.data(),nv,nv);
    VectorXd h=Map<VectorXd>(d->qfrc_bias,nv), qv=qvel();
    std::vector<MatrixXd> Js; for(auto&cp:cpts) Js.push_back(foot_jac_at(cp.first,cp.second));
    MatrixXd Jc=jac_com(); Vector3d c=com();
    MatrixXd P=MatrixXd::Zero(nz,nz); VectorXd g=VectorXd::Zero(nz);
    // CoM task (xy+z)
    Vector3d kp(120,120,200), kd(20,20,25), comref(com_ref_xy[0],com_ref_xy[1],com_ref_z);
    Vector3d a_com=kp.cwiseProduct(comref-c)-kd.cwiseProduct(Jc*qv);
    P.topLeftCorner(nv,nv)+=Jc.transpose()*Jc; g.head(nv)-=Jc.transpose()*a_com;
    // 자세 레벨링(현재 yaw 프레임)
    Vector4d qc; for(int i=0;i<4;i++) qc[i]=d->qpos[3+i];
    double yaw=std::atan2(2*(qc[0]*qc[3]+qc[1]*qc[2]),1-2*(qc[2]*qc[2]+qc[3]*qc[3]));
    double qlev[4]={std::cos(yaw/2),0,0,std::sin(yaw/2)}, ql_conj[4]={qlev[0],-qlev[1],-qlev[2],-qlev[3]};
    double dq[4]={ql_conj[0]*qc[0]-ql_conj[1]*qc[1]-ql_conj[2]*qc[2]-ql_conj[3]*qc[3],
                  ql_conj[0]*qc[1]+ql_conj[1]*qc[0]+ql_conj[2]*qc[3]-ql_conj[3]*qc[2],
                  ql_conj[0]*qc[2]-ql_conj[1]*qc[3]+ql_conj[2]*qc[0]+ql_conj[3]*qc[1],
                  ql_conj[0]*qc[3]+ql_conj[1]*qc[2]-ql_conj[2]*qc[1]+ql_conj[3]*qc[0]};
    Vector3d oerr; { double s=(dq[0]<0?-1:1); Vector3d v(dq[1],dq[2],dq[3]); double n=v.norm();
      oerr=(n<1e-12)?Vector3d(0,0,0):(2.0*std::atan2(n,std::abs(dq[0]))*s/n)*v; }
    for(int j=0;j<3;j++){ double a=150*(-oerr[j])-20*qv[3+j]; P(3+j,3+j)+=W_ORI; g[3+j]-=W_ORI*a; }
    // posture — ★thigh/calf는 약하게(CoM 높이 task가 다리 신전으로 높이 조절 가능하게), 발목/hip은 firm
    const double* Qh=Qcur();
    for(int j=0;j<nu;j++){ double a=60*(Qh[j]-d->qpos[7+j])-5*qv[6+j];
      int lj=j%4; double w=(lj==3)?W_ANKLE : (lj==1||lj==2)?FLAT_WLEG : W_POST;
      P(6+j,6+j)+=w; g[6+j]-=w*a; }
    P.topLeftCorner(nv,nv)+=1e-4*MatrixXd::Identity(nv,nv);
    for(int k=0;k<K;k++) P.block(nv+3*k,nv+3*k,3,3)+=1e-2*Matrix3d::Identity();   // ★λ 정칙화↑(rank-deficient 안정)
    // 등식: base6 + 접촉3K
    int neq=6+3*K; MatrixXd A=MatrixXd::Zero(neq,nz); VectorXd bb=VectorXd::Zero(neq);
    A.block(0,0,6,nv)=M.topRows(6); bb.head(6)=-h.head(6);
    for(int k=0;k<K;k++){ A.block(0,nv+3*k,6,3)=-Js[k].leftCols(6).transpose();
      A.block(6+3*k,0,3,nv)=Js[k]; bb.segment(6+3*k,3)=-STANCE_KD*(Js[k]*qv); }
    // 부등식: 마찰추 + λz≥min (토크한계 없음, Python wbic_stance 동일)
    std::vector<VectorXd> Gr; std::vector<double> hv; int sgn[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
    for(int k=0;k<K;k++){ int o=nv+3*k;
      for(int s=0;s<4;s++){ VectorXd r=VectorXd::Zero(nz); r[o]=sgn[s][0]; r[o+1]=sgn[s][1]; r[o+2]=-MU_EFF; Gr.push_back(r); hv.push_back(0.0);}
      VectorXd r=VectorXd::Zero(nz); r[o+2]=-1; Gr.push_back(r); hv.push_back(-LAMZ_MIN); }
    P=(0.5*(P+P.transpose())).eval()+1e-6*MatrixXd::Identity(nz,nz);   // ★정칙화↑(1e-8→1e-6, eiquadprog 안정)
    MatrixXd CE=A; VectorXd ce0=-bb; int nci=(int)Gr.size(); MatrixXd CI(nci,nz); VectorXd ci0(nci);
    for(int i=0;i<nci;i++){ CI.row(i)=-Gr[i]; ci0[i]=hv[i]; }
    VectorXd x(nz); eiquadprog::solvers::EiquadprogFast qp; qp.reset(nz,neq,nci);
    auto st=qp.solve_quadprog(P,g,CE,ce0,CI,ci0,x);
    if(getenv("WBIC_DBG")){ static int nf=0,nt=0; nt++; if(st!=eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL) nf++;
      if(nt%200==0) std::fprintf(stderr,"[wbic_stance] K=%d QP실패 %d/%d · com_err=(%.3f,%.3f,%.3f)\n",K,nf,nt,com_ref_xy[0]-c[0],com_ref_xy[1]-c[1],com_ref_z-c[2]); }
    if(st==eiquadprog::solvers::EIQUADPROG_FAST_OPTIMAL){
      VectorXd qdd=x.head(nv); VectorXd tau=M.block(6,0,nu,nv)*qdd+h.segment(6,nu);
      for(int k=0;k<K;k++) tau-=Js[k].block(0,6,3,nu).transpose()*x.segment(nv+3*k,3);
      for(int i=0;i<nu;i++) d->ctrl[i]=std::max(-tau_peak8[i],std::min(tau_peak8[i],tau[i]));
    } else { for(int i=0;i<nu;i++) d->ctrl[i]=std::max(-tau_peak8[i],std::min(tau_peak8[i],h[6+i])); }  // 실패=중력보상 홀드
  }

  void set_contact_mode(int cm){ if(!has_heel||cm==cmode) return; cmode=cm; reset(); }   // 초기용(스냅)

  // ★런타임 부드러운 전환 시작(toe-pivot 굴림): 발목·다리·높이를 목표자세로 서서히 굴림
  void transition_to(int cm){
    if(!has_heel || cm==cmode || trans_on) return;
    const double* qf=Qcur(); const double* qt=(cm==1)?Qflat8:Qhome8;
    for(int j=0;j<8;j++){ q_from[j]=qf[j]; q_to[j]=qt[j]; q_live[j]=qf[j]; }
    cz_from=com_ref_z; cz_to=(cm==1)?0.362:0.483;
    trans_on=true; trans_t=0; trans_to=cm;
  }
  // 전환 궤적 재생(양발/toe 적응접촉 wbic_stance로 추종)
  void do_transition(double dt){
    double a=trans_t/T_TRANS; a=a<0?0:(a>1?1:a); a=a*a*(3-2*a);   // smoothstep
    for(int j=0;j<8;j++) q_live[j]=q_from[j]*(1-a)+q_to[j]*a;
    com_ref_z=cz_from*(1-a)+cz_to*a;
    auto cpts=contact_pts({0,1}); Vector3d sc(0,0,0);             // 접지 구 중심(적응: 밑창→toe)
    for(auto&cp:cpts) sc+=gpos(cp.first); if(cpts.size()) sc/=(double)cpts.size();
    com_ref_xy<<sc[0],sc[1];
    wbic_stance();
    yaw_hold=base_yaw(); yaw_hold_set=true; yaw_des=base_yaw();
    trans_t+=dt;
    if(trans_t>=T_TRANS){ trans_on=false; cmode=trans_to; }        // 완료→목표모드 확정
  }

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
    const double* Qh=Qcur();                          // ★모드별 home(점발 세움 / 평발 눕힘)
    for(int j=0;j<nu;j++) d->qpos[7+j]=Qh[j];
    d->qpos[2]=0.7; for(int i=0;i<nv;i++) d->qvel[i]=0; mj_forward(m,d);
    double zmin=1e9;
    for(int l=0;l<2;l++){ zmin=std::min(zmin,footz(l)-m->geom_size[sph[l]*3]);
      if(has_heel) zmin=std::min(zmin,d->geom_xpos[sph2[l]*3+2]-m->geom_size[sph2[l]*3]); }  // 평발=heel도 접지
    d->qpos[2]-=zmin; mj_forward(m,d);
    Vector3d c=com(); com0=c.head(2); com_ref_xy=c.head(2);
    for(int l=0;l<2;l++) nominal_off[l]=foot_center(l).head(2)-c.head(2);
    for(int l=0;l<2;l++) for(int i=0;i<4;i++) foot_home_quat[l][i]=d->xquat[fbody[l]*4+i];   // swing 수평 목표
    com_ref_z=c[2];
    stance=1; swing=0; t_ss=0; _k=0; yaw_des=0; yaw_hold_set=false; have_liftoff[0]=have_liftoff[1]=false;
    for(int i=0;i<nv;i++) d->qvel[i]=0;
    compute_Icom();
  }
  int nq(){ return m->nq; }

  // ── event-DCM 게이트 ──
  void step_gait(double dt,int&st,int&sw,double&s){
    Vector3d c=com(); VectorXd qv=qvel(); MatrixXd Jc=jac_com(); Vector2d vcom=(Jc*qv).head(2);
    double z=std::max(c[2]-std::min(footz(0),footz(1)),0.15), w=std::sqrt(GVEC/z);
    // ★측방 DCM 트리거를 body-frame으로(yaw 나도 올바른 측방=보행 강건). 발 중점 기준 DCM벡터를 body-y축에 투영.
    double ya=base_yaw(), cya=std::cos(ya), sya=std::sin(ya);
    Vector3d fc0=foot_center(0), fc1=foot_center(1);   // 발 중점(점발=tip / 평발=밑창중점)
    double midx=0.5*(fc0[0]+fc1[0]);
    double midy=0.5*(fc0[1]+fc1[1]);
    double dcmx=c[0]+vcom[0]/w-midx, dcmy=c[1]+vcom[1]/w-midy;   // world DCM(발중점 기준)
    double dcm_by=-sya*dcmx+cya*dcmy;                            // body-y 성분(직진 yaw=0시 =dcmy)
    double sy=(swing==0)?1.0:-1.0;
    s=std::min(std::max(t_ss/SS_NOMINAL,0.0),1.0);
    bool committed=sy*dcm_by>TRIG_Y;
    if(t_ss>SS_MIN&&(committed||t_ss>SS_MAX)){ std::swap(stance,swing); t_ss=0;
      liftoff[swing]=foot_center(swing); have_liftoff[swing]=true; st=stance; sw=swing; s=0; return; }
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
    if(cmode==1 && has_heel){          // ★평발: 전후=capture+return(과속 브레이킹, 발목ZMP 보조)·측방=capture(밑창 좁음)
      Vector2d st_b=to_b(foot_center(1-sw).head(2)-c.head(2));
      double rel_fwd = off[0] + FLAT_KCAP*v_b[0]/w + K_RETURN*err_b[0];   // capture로 CoM 앞서기 방지
      rel_fwd = std::min(std::max(rel_fwd, off[0]-CAP_CLAMP), off[0]+CAP_CLAMP);
      double rel_lat_cap = SPREAD*off[1] + K_LAT*(v_b[1]/w);
      double gap=std::min(std::max(lat*(rel_lat_cap-st_b[1]),GAP_MIN),GAP_MAX);
      double rel_lat = st_b[1]+lat*gap;
      return c.head(2)+to_w(Vector2d(rel_fwd,rel_lat));
    }
    double rel_fwd=off[0]+K_CAP*v_b[0]/w+K_RETURN*err_b[0];
    rel_fwd=std::min(std::max(rel_fwd,off[0]-CAP_CLAMP),off[0]+CAP_CLAMP);
    double rel_lat=SPREAD*off[1]+K_LAT*(v_b[1]/w)+K_RET_LAT*err_b[1];
    Vector2d st_b=to_b(foot_center(1-sw).head(2)-c.head(2));
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
    Vector3d frel[2]; for(int i=0;i<2;i++) frel[i]=foot_center(i)-cc;
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
    // ★stance 접촉: 점발=tip 1개 / 평발=stance 발 접지 구(heel+toe 다접촉)
    std::vector<std::pair<int,int>> scp;
    if(cmode==1&&has_heel) scp=contact_pts({stanceLeg}); else scp={{sph[stanceLeg],fbody[stanceLeg]}};
    in.Kc=(int)scp.size(); in.contacts.clear(); in.cjac.clear(); in.lam.clear();
    for(auto&cp:scp){ in.contacts.push_back(stanceLeg); in.cjac.push_back(foot_jac_at(cp.first,cp.second));
      in.lam.push_back(lam.row(stanceLeg).transpose()/(double)scp.size()); }   // MPC GRF 발당→접촉점 분배
    in.has_swing=true; in.swing_leg=sw;
    in.Jsw=(cmode==1&&has_heel)?foot_jac_center(sw):foot_jac(sw);
    in.sw_pos=(cmode==1&&has_heel)?foot_center(sw):spos(sw);
    in.sw_ptgt=ptgt; in.sw_vtgt=vtgt;
    if(cmode==1&&has_heel && !getenv("NO_SWORI")){   // ★평발 swing 발 수평 유지
      in.has_sw_ori=true; in.Jsw_rot=foot_jacr(sw);
      double ya=base_yaw(), qy[4]={std::cos(ya/2),0,0,std::sin(ya/2)}, ftgt[4];
      mju_mulQuat(ftgt,qy,foot_home_quat[sw].data());
      double fq[4]; for(int i=0;i<4;i++) fq[i]=d->xquat[fbody[sw]*4+i];
      double oe[3]; mju_subQuat(oe,fq,ftgt); for(int i=0;i<3;i++) in.sw_oerr[i]=oe[i];
    }
    in.Qhome=Map<const VectorXd>(Qcur(),nu); in.tau_peak=Map<VectorXd>(tau_peak8,nu);   // ★모드별 자세(평발=Qflat)
    in.ankle_idx={ankle_idx[0],ankle_idx[1]};
    if(in_zmp_walk){              // ★ZMP 프리뷰: 전후(x)만 CoM 추종(밑창 ZMP). 측방(y)은 capture 발배치(밑창 좁음).
      in.com_x_track=true; in.com_x_ref=cxr; in.com_vx_ref=vxr;
    } else if(cmode==1&&has_heel){ // (구)평발 보행: 전후 CoM을 com0에 규제
      in.com_x_track=true; in.com_x_ref=com0[0]; in.com_vx_ref=std::cos(base_yaw())*vx_cmd;
    }
    double wank=(cmode==1&&has_heel)?FLAT_WANK:W_ANKLE;   // ★평발=발목 강하게 flat 고정(밑창 유지, 안하면 발목 서서 토플)
    double wori=(cmode==1&&has_heel)?FLAT_WORI:W_ORI;     // ★평발=base pitch 레벨링↑(밑창 ZMP로 pitch 유지)
    in.SW_KP=SW_KP; in.SW_KD=SW_KD; in.W_ORI=wori; in.W_ANKLE=wank; in.W_POST=W_POST;
    in.W_LAM=(cmode==1&&has_heel)?FLAT_WLAM:W_LAM; in.STANCE_KD=STANCE_KD; in.MU_EFF=MU_EFF; in.LAMZ_MIN=LAMZ_MIN;   // 평발=MPC추종↓, WBIC task 지배
    VectorXd tau=wbic_track(in); for(int i=0;i<nu;i++) d->ctrl[i]=tau[i];
  }

  // ZMP 레퍼런스(미래 tick fzkk): 초기 DS 리드인(중앙→첫지지발) 후 SS 계단
  void zmp_ref_at(long fzkk,int TICKS_SS,double&zx,double&zy){
    if(fzkk<TICKS_SS){                           // 리드인 DS: 중앙→첫 지지발(leg1=HR)
      double f=(double)fzkk/TICKS_SS, my=0.5*(zaf_y[0]+zaf_y[1]);
      zx=zanchor_x; zy=my*(1-f)+zaf_y[1]*f;
    } else { long fs=(fzkk-TICKS_SS)/TICKS_SS;    // SS: 지지발 위치
      zx=zanchor_x+fs*z_sx; zy=zaf_y[(fs%2==0)?1:0]; }
  }
  // ── ZMP 프리뷰 평발 보행 (clock 기반 footstep + 프리뷰 CoM 궤적) ──
  // ★event-DCM 측방 타이밍 + 프리뷰 전후. 고정clock 대신 측방 sway 동기(timing 충돌 해결).
  void zmp_walk(double dt){
    if(zkk<0){                                   // 보행 시작 초기화
      Vector3d f0=foot_center(0), f1=foot_center(1); Vector3d c=com();
      zanchor_x=f1[0]; zaf_y[0]=f0[1]; zaf_y[1]=f1[1];   // 앵커=현 지지발(HR) x
      pv.reset(c[0],c[1]); cxr=c[0]; vxr=0; zkk=0; prev_ctr=0;
      stance=1; swing=0; t_ss=0; zlead=(long)std::round(T_SS_Z/dt);   // 리드인 DS
      have_liftoff[0]=have_liftoff[1]=false;
    }
    z_sx=vx_cmd*T_SS_Z;
    Vector3d cc=com(); Vector2d vcm=(jac_com()*qvel()).head(2);
    double zz=std::max(cc[2]-std::min(footz(0),footz(1)),0.15), ww=std::sqrt(GVEC/zz);
    // ── 전후 프리뷰: ZMP staircase(현 지지발 x + 미래 공칭T_ss마다 z_sx) → CoM-x 궤적 ──
    if(prev_ctr==0){
      int Np=pv.N; std::vector<double> px(Np),py(Np,0.0);
      for(int j=0;j<Np;j++){ double ta=t_ss + (double)j*PREV_DECIM*dt;
        long sfut = (zlead>0)? 0 : (long)std::floor(ta/T_SS_Z);       // 리드인 중엔 현 지지발 유지
        px[j]=zanchor_x + sfut*z_sx; }
      double cyd,vyd; pv.step(px.data(),py.data(), cxr,vxr,cyd,vyd);
    }
    prev_ctr=(prev_ctr+1)%PREV_DECIM;
    // ── 리드인 DS: CoM을 첫 지지발로 이동(전후=프리뷰) ──
    if(zlead>0){ com_ref_xy<<cxr, 0.7*zaf_y[1]; wbic_stance();
      yaw_hold=base_yaw(); yaw_hold_set=true; yaw_des=base_yaw(); zlead--; zkk++; return; }
    // ── 측방 event-DCM 트리거(sway가 swing측 넘으면 착지) ──
    double midy=0.5*(foot_center(0)[1]+foot_center(1)[1]);
    double xi_y=cc[1]+vcm[1]/ww, sy=(swing==0)?1.0:-1.0;
    bool committed = sy*(xi_y-midy) > TRIG_Y;
    if(t_ss>SS_MIN && (committed || t_ss>SS_MAX)){
      std::swap(stance,swing); t_ss=0; zanchor_x+=z_sx;              // 전후 앵커 전진
      have_liftoff[swing]=false;
      pv.set_state(cc[0],vcm[0],cc[1],vcm[1]);                       // ★프리뷰 전후 실제CoM 재동기(발산방지)
    }
    int support=stance, sw=swing; double s=std::min(t_ss/T_SS_Z,1.0);
    // swing 발: 전후=다음 지지 x(앵커+z_sx)·측방=capture(밑창 좁음)
    double sw_tx=zanchor_x+z_sx;
    double lat=(sw==0)?1.0:-1.0;
    double sw_ty=cc[1]+lat*std::abs(zaf_y[sw])+K_LAT*vcm[1]/ww;
    { double stf=foot_center(support)[1]; double gap=std::min(std::max(lat*(sw_ty-stf),GAP_MIN),GAP_MAX); sw_ty=stf+lat*gap; }
    if(!have_liftoff[sw]){ liftoff[sw]=foot_center(sw); have_liftoff[sw]=true; }
    Vector3d p0=liftoff[sw];
    double gz=std::min(footz(0),footz(1))+m->geom_size[sph[sw]*3];
    double sm=10*s*s*s-15*s*s*s*s+6*s*s*s*s*s, dsm=(30*s*s-60*s*s*s+30*s*s*s*s)/std::max(1e-6,T_SS_Z);
    Vector3d p(p0[0]+(sw_tx-p0[0])*sm, p0[1]+(sw_ty-p0[1])*sm, p0[2]+(gz-p0[2])*sm+4*STEP_H*s*(1-s));
    Vector3d v((sw_tx-p0[0])*dsm,(sw_ty-p0[1])*dsm,(gz-p0[2])*dsm+4*STEP_H*(1-2*s)*dsm);
    if(getenv("ZMP_DBG")&&zkk%25==0) std::fprintf(stderr,
      "  z t%.2f sup%d com=(%.3f,%.3f,%.3f) cxref%.3f t_ss%.2f swT=(%.2f,%.2f)\n",
      zkk*dt,support,cc[0],cc[1],cc[2],cxr,t_ss,sw_tx,sw_ty);
    t_ss+=dt; zkk++;
    if(_k%mpc_decim==0) lam=mpc_grf(support); _k++;
    in_zmp_walk=true; wbic(support,sw,p,v); in_zmp_walk=false;
    yaw_hold=base_yaw(); yaw_hold_set=true; yaw_des=base_yaw();
  }

  void control(double dt){
    double ya=base_yaw();
    if(trans_on){ do_transition(dt); return; }   // ★1점/2점 전환 굴림 재생 중
    // ★2점 평발: 정지=정적 양발지지(밑창 ZMP). 이동명령=평발 동적 보행(아래 게이트, wbic 다접촉).
    if(has_heel && cmode==1){
      bool flat_walk_en = getenv("FLAT_WALK")!=nullptr;   // ★2점=서기 전용(기본). 보행은 adaptive-timing 프리뷰 완성 후 활성
      bool moving = flat_walk_en && (std::abs(vx_cmd)>0.02 || std::abs(vy_cmd)>0.02 || std::abs(wz_cmd)>0.02);
      if(!moving){                                   // 정지(또는 보행 미활성)=정적 양발지지
        com_ref_z=std::min(std::max(com_ref_z,0.36),0.42);   // ★평발 정적 높이 실현범위 클램프(발 flat 유지 기하제약)
        Vector3d fc=0.5*(foot_center(0)+foot_center(1)); com_ref_xy=fc.head(2);
        wbic_stance();
        t_ss=0; com0=com().head(2); have_liftoff[0]=have_liftoff[1]=false; yaw_hold=ya; yaw_hold_set=true;
        walk_init=true; zkk=-1;                      // 다음 보행개시 재무장(reactive weight-shift · ZMP 재초기화)
        return;
      }
      if(getenv("ZMP_WALK")){ zmp_walk(dt); return; }  // ★ZMP 프리뷰 평발 보행(실험)
      if(walk_init){                                 // ★보행개시: 첫 스텝 전 CoM을 첫 stance 발쪽 측방 이동
        Vector3d sf=foot_center(stance);             // stance=1(HR) 쪽으로 체중 이동(지지면끝까지 못가니 75%)
        double tgt_y=0.75*sf[1];
        com_ref_xy[0]=sf[0]; com_ref_xy[1]=tgt_y;
        wbic_stance(); walk_init_t+=dt;
        Vector3d c=com();
        if(std::abs(c[1]-tgt_y)<0.03 || walk_init_t>0.4){   // 지지발쪽 도달 or 시간초과→스텝 시작
          walk_init=false; walk_init_t=0; t_ss=0; com0=c.head(2);
          have_liftoff[0]=have_liftoff[1]=false; yaw_hold=ya; yaw_hold_set=true; }
        return;
      }
      if(czwalk>0) com_ref_z=czwalk;               // 평발 보행 CoM 높이(튜닝)
    }
    if(std::abs(wz_cmd)>0.02){                    // ★선회: 명령 적분 + 리드 클램프(폭주방지)
      yaw_des+=wz_cmd*dt;
      double lag=std::atan2(std::sin(yaw_des-ya),std::cos(yaw_des-ya));
      yaw_des=ya+std::min(std::max(lag,-head_lead),head_lead); yaw_hold_set=false;
    } else {                                      // ★선회 외 전부(정지/전후진/측방): heading latch(base0.50서 측방도 안정)
      if(!yaw_hold_set){ yaw_hold=ya; yaw_hold_set=true; }
      double err=std::atan2(std::sin(yaw_hold-ya),std::cos(yaw_hold-ya));
      yaw_des=ya+std::min(std::max(err,-head_lead),head_lead);
    }
    double cya=std::cos(ya),sya=std::sin(ya);   // ★복귀목표 이동=실제 base yaw 기준(base-relative)
    com0[0]+=(cya*vx_cmd-sya*vy_cmd)*dt; com0[1]+=(sya*vx_cmd+cya*vy_cmd)*dt;
    int st,sw; double s; step_gait(dt,st,sw,s);
    if(_k%mpc_decim==0) lam=mpc_grf(st);
    _k++;
    if(!have_liftoff[sw]){ liftoff[sw]=foot_center(sw); have_liftoff[sw]=true; }
    Vector3d p,v; swing_traj(sw,s,p,v);
    wbic(st,sw,p,v);
  }
};
