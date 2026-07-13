// TrotCtrl — mode_trot 핵심경로 1틱 제어(설정/스윙/MPC/WBIC → d->ctrl). trot_sim(헤드리스)·trot_view(뷰어) 공유.
#pragma once
#include "quad_control.hpp"
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <fstream>
#include <sstream>
#ifdef HAVE_CROCODDYL
#include "jump_solver.hpp"   // ★S3-b Step2: 점프 crouch중 live-solve(trot_view만, crocoddyl 링크 시)
#endif

// gait 상수(config + trot 프리셋)
static const double GP_T=0.50, GP_SWF=0.50;
static const double GP_OFFSET[4]={0.0,0.5,0.5,0.0};   // HL,HR,FL,FR
static const double TC_SW=0.25, TC_ST=0.25, TC_SDELTA=0.005;
static const double TC_WARMUP=0.6, TC_SETTLE=0.5, TC_ACC=0.6;
static const double TC_KCAP=0.16, TC_RAIBERT=0.8, TC_RAICLIP=0.25;

static inline void tc_gait(int i,double tg,bool&stance,double&sprog){
  double ph=std::fmod(tg/GP_T+GP_OFFSET[i],1.0); if(ph<0) ph+=1.0;
  if(ph>=GP_SWF){ stance=true; sprog=0.0; } else { stance=false; sprog=ph/GP_SWF; }
}
static inline void tc_swing_z(double sh,double Th,double Vz,double&c2,double&c4,double&c6){
  Matrix3d A; A<< Th*Th, std::pow(Th,4), std::pow(Th,6),
                  2.0, 12.0*Th*Th, 30.0*std::pow(Th,4),
                  2.0*Th, 4.0*std::pow(Th,3), 6.0*std::pow(Th,5);
  Vector3d b(-sh,0.0,-Vz); Vector3d c=A.colPivHouseholderQr().solve(b); c2=c[0];c4=c[1];c6=c[2];
}
static inline Vector3d tc_swing_foot(double sw_t,const Vector3d&p0,const Vector3d&pe,const Vector3d&bvel,double sh,double Tl,double Tst){
  if(sw_t>=1.0) return pe;
  double tau=sw_t, s5=10*std::pow(tau,3)-15*std::pow(tau,4)+6*std::pow(tau,5);
  double DXx=(pe[0]-p0[0])+bvel[0]*Tl; Vector3d pos;
  pos[0]=p0[0]-bvel[0]*tau*Tl+DXx*s5; pos[1]=(1.0-s5)*p0[1]+s5*pe[1];
  double Th=Tl/2.0, u=tau*Tl-Th, Vz=TC_SDELTA*M_PI/Tst; double c2,c4,c6; tc_swing_z(sh,Th,Vz,c2,c4,c6);
  pos[2]=p0[2]+sh+c2*u*u+c4*std::pow(u,4)+c6*std::pow(u,6); return pos;
}
static inline double tc_clip(double v,double lo,double hi){ return v<lo?lo:(v>hi?hi:v); }

struct TrotCtrl {
  QuadControl& q;
  double V=0.30, VY=0.0, WZ=0.0;   // 명령속도(뷰어 키보드/GUI)
  double step_h=0.10, raibert_k=0.5;   // ★GUI 슬라이더(live): step height·전방 reach. ★0.5=표준 Raibert 중립점(발을 앞으로 과하게 안던짐→GRF 앞/뒤 균형·뒤thigh 절반). 외란복구는 KCAP+MPC 담당
  double gait_base_z=0.50;   // ★gait별 최적 base height(축별 worst-util 스윕): walk/trot 0.50·run 0.48. 보행 중 추종(set_gait서 설정)
  bool ALIP=false, POS_HOLD=true;   // ★ALIP 기본 off: push복구 이득 미미(300N서만 22vs26°)한데 지속선회 붕괴시킴. ALIP=1로 켬
  bool SPIN_HOLD=false;             // ★제자리선회(V=0,WZ≠0)서도 위치홀드 유지 → 허리조향 표류 상쇄(베이스기준 wz선회). 주행선회엔 영향無
  bool perceptive=true;             // ★perceptive: 스윙 착지 XY 아래 지형높이를 mj_ray로 샘플→착지 z를 지형에 맞춤(계단/험지). off=blind(평지 가정). PERCEPTIVE=0으로 끔
  double PCV_CLR=0.04;              // ★perceptive 상향 스텝 시 추가 스윙 클리어런스(up-step 높이×비율만큼 apex↑, 라이저 헛디딤 방지)
  double com_h0=0.52;               // ★평지 위 CoM 명목높이(arming서 캡처). perceptive 몸통높이 목표=지형높이+com_h0
  double _bterr_s=0.0;              // ★슬루된 지형높이(4hip평균을 부드럽게) → MPC x_ref[5]·WBIC z-task 양쪽 일관 공급
  // ── 모드관리(배포용): move/stand_up(서기)/stand_down(눕기)/off ──
  std::string mode="move";
  double body_h=0.5234, ht_cur=0.5234, qhome_h=0.5234;   // 서기높이 슬라이더·보간높이·q_home 계산높이
  VectorXd q_ref; bool have_qref=false;                  // fold 관절목표 slew
  double SIT_Z=0.32, SIT_PITCH=0.70, SIT_REAR_FOOT=-1.35, SIT_REAR_CALF=1.10, SIT_REAR_THIGH=-0.6; bool have_qsit=false; // ★앉기=crouch-sit(SIT_Z 저crouch, 4발 planted→기립가능). SIT_PITCH등은 구 haunch-sit 잔여
  double SIT_CPITCH=1.0, SIT_REACH=0.08;   // ★앉기 nose-up 목표(~25° 앞올림=앉은자세, wbic_stance 능동제어→안정+기립가능). SIT_REACH 미사용(잔여)
  // ★개-앉기(haunch sit): 정착 후 crouch→haunch(뒷다리 접어 발링크 바닥밀착)로 fold, 기립 시 높이스케줄 언폴드. q_home/com_ref 블렌드.
  double HAUNCH_Z=0.30, HAUNCH_FOLD_RATE=0.35, HAUNCH_UNFOLD_Z=0.40, SIT_POSTURE_W=40.0, HAUNCH_PITCH=0.50, SIT_KP=90.0;   // ★엉덩이 주저앉기: base 낮춤·fold 완만(0.60→0.35=~2.9s, 서기→앉기 급강하 완화)·nose-up 크게. 기립불요·자세우선
  double SIT_BELOW_SPEED=2.5;   // ★눕기(from_below)→앉기 fold 배수: 엉덩이 이미 낮아 엉덩방아 위험 없음→빠르게(위→앉기 완만착지는 미적용). ~2.9s→~1.2s
  double FRONT_PULL_SPEED=2.2;  // ★앞다리 끌어당김 배수(앞다리는 뒤 엉덩이 착지 tail과 무관하게 더 빨리 접힘): 앞은 haunch_fold*이 배수로 조기 완료, 뒤(엉덩방아)는 원래 pace 유지
  VectorXd q_crouch, q_haunch; Vector3d com_crouch=Vector3d::Zero(), com_haunch=Vector3d::Zero(); double haunch_fold=0; bool haunch_ready=false;
  // ★기립 궤적 추종(offline gather 궤적 /tmp/getup_traj.txt): sit→gather(CoM 전진)→일어서기. phaseA(0/1)=PD추종, phaseB(2)=wbic 상승 인계.
  std::vector<VectorXd> getup_q, getup_dqv; std::vector<int> getup_ph; double getup_dt=0.01; int getup_N=-1, getup_k=-1; double getup_kt=0, GETUP_TRAJ_KP=80.0, GETUP_TRAJ_KD=6.0;   // ★앉기→서기 튕김 힘 완화(120→80)+감쇠↑(4→6): 기립 중 최대tilt 70°→49°(더 약하고 안전하게). 기립 완료는 유지(z≈0.49)
  void load_getup(const char* path){
    std::ifstream f(path); if(!f){ getup_N=0; return; }
    f>>getup_N>>getup_dt; getup_q.clear(); getup_dqv.clear(); getup_ph.clear();
    for(int k=0;k<getup_N;k++){ int ph; f>>ph; getup_ph.push_back(ph);
      VectorXd qv(q.nu),dv(q.nu); for(int j=0;j<q.nu;j++) f>>qv[j]; for(int j=0;j<q.nu;j++) f>>dv[j];
      getup_q.push_back(qv); getup_dqv.push_back(dv); }
    std::printf("[getup] 궤적 로드 N=%d dt=%.3f\n", getup_N, getup_dt);
  }
  // ★앉기→서기 스크립트 기립(앞다리 굽혀 앞발 들어 폴볼트 차단 + 뒷다리 박차 extend). 앉기에서만 발동.
  bool was_sit=false; double sit_getup_t0=-1;
  bool sit_init=false, sit_from_below=false;   // ★sit 진입 방향 1회 latch: 눕기 등 아래서 진입 시 0.32 crouch 오버슈트 없이 현재 자세서 곧바로 haunch로 morph
  bool from_sit=false;   // ★crouch-sit서 기립: 저crouch(≥0.29)라 저-PD 대신 wbic_stance로 매끈 기립(오버슈트 방지)
  double SGU_KICK_T=0.5, SGU_FB_THIGH=-0.55, SGU_FB_CALF=1.20, SGU_SLEW=1.5, SGU_KP=120, SGU_GATHER_Z=0.24, SGU_DONE_TILT=22;
  double SGU_WALKOUT_V=0.6, SGU_HANDOFF_Z=0.34;   // ★기립 후 전진 트로트로 인계(walk-out)해 균형회복. bz>HANDOFF_Z면 move로 전환
  double SIT_SLEW=0.6; // ★앉기 하강 슬루(rad/s, 작을수록 천천히·충격↓). 기본 JOINT_SLEW(1.5)보다 느리게→충격 완화
  // ★★점프(J0.2 스크립트 pronk, RPET_JUMP_MPC.md): crouch(wbic)→thrust(강PD 신전)→flight(tuck)→touchdown→흡수→wbic회복. 이벤트 트리거(시간 아님).
  int jphase=-1; double jt0=0, jzpk=0; VectorXd jqc, jqs;
  double JUMP_CROUCH_Z=0.30, JUMP_THRUST_T=0.16, JUMP_KP=380, JUMP_KD=6;   // (구 스크립트 스냅용)
  // ★★J2 통합: OCP 궤적(/tmp/jump_traj.txt, MuJoCo순서 phase·q·dq·tau) 재생 → thrust/flight, 착지는 wbic_stance
  std::vector<VectorXd> jump_q, jump_dq, jump_tau; std::vector<int> jump_ph;
  int jump_N=-1, jump_k=-1; double jump_dt=0.01, jump_kt=0, JUMP_TRAJ_KP=120, JUMP_TRAJ_KD=4;
  void load_jump(const std::string& path){
    std::ifstream f(path); if(!f){ jump_N=0; return; }
    f>>jump_N>>jump_dt; jump_q.clear(); jump_dq.clear(); jump_tau.clear(); jump_ph.clear();
    for(int k=0;k<jump_N;k++){ int ph; f>>ph; jump_ph.push_back(ph);
      VectorXd qq(q.nu),dd(q.nu),tt(q.nu);
      for(int j=0;j<q.nu;j++) f>>qq[j]; for(int j=0;j<q.nu;j++) f>>dd[j]; for(int j=0;j<q.nu;j++) f>>tt[j];
      jump_q.push_back(qq); jump_dq.push_back(dd); jump_tau.push_back(tt); } }
  double JUMP_VX=0.6;   // ★점프 전방 이륙속도(0=수직 제자리). live-solve·gen_jump 공통 의미
#ifdef HAVE_CROCODDYL
  std::string JUMP_URDF="/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf";
  std::string JUMP_MJCF="/home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf";
  int JUMP_MAXIT=8;     // ★crouch중 live-solve FDDP 반복(S2: iter~8=~150ms 수렴). crouch 예산 450ms 내
  bool solve_jump_live(double vx){   // crouch 정착 시 1회 호출 → 신선 궤적을 인메모리로 채움(파일 I/O 없음)
    JumpTraj J=jump_solve(JUMP_URDF,JUMP_MJCF,vx,JUMP_MAXIT,false);
    if(J.N<=0) return false;
    jump_N=J.N; jump_dt=J.dt; jump_q=J.q; jump_dq=J.dq; jump_tau=J.tau; jump_ph=J.ph; return true; }
#endif
  // 모드관리 상수(Python 17dof와 동일)
  double GROUND_Z=0.18, GETUP_TRIG=0.32, GETUP_DONE=0.40, GETUP_KP=90, GETUP_KD=3, GETUP_RATE=0.18, REST_KD=3.0, JOINT_SLEW=1.5, HRATE=0.3;
  double GROUND_LIE_Z=0.226, GROUND_REAR_FOOT=-1.15, GROUND_FRONT_FOOT=-0.5, GROUND_FRONT_THIGH=-0.24, GROUND_FRONT_CALF=-0.4;   // ★눕기(ground) 저자세: base 낮춤 + 앞뒤 발목/앞다리 fold(GUI 실시간 슬라이더로 CoM균형·수평·무슬라이드 조각). PD-fold 홀드. ★기본값=뷰어 라이브튜닝서 확정(base_z≈0.166 깊은 belly-lie)
  // ── 게이트 프리셋(trot/walk/gallop) ──
  std::string gait_type="trot";
  double gp_T=0.5, gp_SWF=0.5, gp_off[4]={0,0.5,0.5,0}, gp_Tsw=0.25, gp_Tst=0.25;
  // ── ★속도 트리거 자동 whip(고속 trot=동물형 채찍질) ──
  bool auto_whip=true; double whip_v0=0.8, whip_v1=1.6;    // ★기본 ON: v0~v1서 whip 선형증가(swing_w 2.0→낮게)
  double whip_hi=2.0, whip_lo_f=0.1, whip_lo_r=0.6;        // ★앞발 paw-tuck whip(앞0.1강·뒤0.6). yaw-fight 수정 후 선회 최고(15.8°)+원래 의도. 슬라이더로 조절
  double waist_steer=0.4, waist_cap=0.20;                  // ★허리 lean 보조 게인(선회시 앞몸통 안쪽 굽힘=차동차식). 캡±0.20rad(11°, 구0.75=43°는 낙상). 다리(MPC)가 선회수행·허리는 lean
  double steer=0.0, Ss_steer=0.0, wheelbase=0.5;           // ★자동차식 조향각 δ[rad](GUI 허리핸들). Ackermann Weff+=V·tanδ/축거(전진해야 조향). 축거=앞뒤 힙거리
  // 상태
  bool armed=false; double t0=0, settle_until=TC_SETTLE;
  bool stop_settle=false;   // ★달리다 서기: 정지 감속·CoM 재중심 중(뒤 엉덩방아 방지)
  double stand_ax=0, stand_ay=0; bool stand_set=false;   // ★서기 위치 앵커(현재 위치서 서기 — 홈으로 빨려감 방지)
  bool off_settled=false;   // ★전원차단(off): 바닥까지 완만 하강 완료 여부(선 채로 툭 damp=낙하 방지)
  double Vs=0,Vys=0,Ws=0, yaw_ref=0; bool yaw_hold_set=false; double yaw_hold=0;
  bool pos_hold_set=false; double phx=0,phy=0;
  VectorXd x_ref=VectorXd::Zero(13);
  std::array<Vector3d,4> liftoff, nominal; std::array<Vector2d,4> hip_off; std::array<double,4> gz;
  std::array<bool,4> have_prev={false,false,false,false}; std::array<Vector3d,4> ptgt_prev;
  Vector3d lam_des[4]={Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero()};
  double mpc_t=-1.0, Veff_dbg=0;

  TrotCtrl(QuadControl& q_):q(q_){ q_ref=VectorXd::Zero(q.nu); q_crouch=q_haunch=VectorXd::Zero(q.nu); body_h=ht_cur=qhome_h=q.base_z0;
    double xf=(q.d->xpos[q.hip_bid[2]*3]+q.d->xpos[q.hip_bid[3]*3])/2;   // FL,FR 힙 x(초기 yaw0=body frame)
    double xr=(q.d->xpos[q.hip_bid[0]*3]+q.d->xpos[q.hip_bid[1]*3])/2;   // HL,HR 힙 x
    wheelbase=std::max(0.15, xf-xr); }                                   // ★축거 L(≈0.61m). Ackermann 조향 반경 R=L/tanδ

  void set_gait(const std::string& g){        // trot/walk/gallop 프리셋(GUI 토글·속도트리거)
    if(g==gait_type) return; gait_type=g;
    if(g=="walk"){ gp_T=0.7; gp_SWF=0.25; gp_off[0]=0.25; gp_off[1]=0.75; gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.10; gait_base_z=0.50; PCV_CLR=0.04; }  // ★walk 안정화(T1.0→0.7·RAI0.8→0.5): reach↓ stumble/bounce방지, 상한~0.6m/s. ★발높이 0.10 복원(9fcfe81서 0.05로 반감됐던 것=발 아치 낮아 앞다리 뻣뻣, falls=0 유지)
    else if(g=="run"){ gp_T=0.40; gp_SWF=0.5; gp_off[0]=0.0; gp_off[1]=0.5; gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.08; gait_base_z=0.48; PCV_CLR=0.04; }  // ★고속 trot(빠른 cadence T0.4·낮은 발높이0.08): 최고속 1.8→~2.0m/s, 발목ω↓. ★base_z 0.48(발목ω 여유)
    else if(g=="stairs"){ gp_T=0.7; gp_SWF=0.25; gp_off[0]=0.25; gp_off[1]=0.75; gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.4; step_h=0.18; gait_base_z=0.50; PCV_CLR=0.08; }  // ★계단 등반: walk 시퀀셜(정적안정)+높은 발높이0.18(라이저 클리어)+낮은 reach0.4(정밀 발배치)+2배 up-step 클리어런스. 느린속도(GUI 0.3)와 조합. 완만한 계단(≤0.10) 확실·가파른(0.15+)은 marginal
    else if(g=="gallop"){ gp_T=0.35; gp_SWF=0.55; gp_off[0]=0.0; gp_off[1]=0.05; gp_off[2]=0.55; gp_off[3]=0.5; raibert_k=0.8; step_h=0.10; PCV_CLR=0.04; } // 회전형 갤럽(비행상 有)
    else         { gp_T=0.5; gp_SWF=0.5;  gp_off[0]=0.0;  gp_off[1]=0.5;  gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.10; gait_base_z=0.50; PCV_CLR=0.04; }  // ★trot 표준 중립(GRF균형·뒤thigh절반·falls=0·push복구↑)
    gp_Tsw=gp_T*gp_SWF; gp_Tst=gp_T*(1.0-gp_SWF); armed=false;   // 재arm=위상 재앵커(불연속 방지)
  }
  void gait(int i,double tg,bool&stance,double&sprog){
    double ph=std::fmod(tg/gp_T+gp_off[i],1.0); if(ph<0) ph+=1.0;
    if(ph>=gp_SWF){ stance=true; sprog=0.0; } else { stance=false; sprog=ph/gp_SWF; }
  }
  void reset(){   // ★시뮬 리셋(RESET 버튼): 컨트롤러 상태 초기화(crouch_home 후 호출)
    armed=false; settle_until=q.d->time+TC_SETTLE; have_qref=false;
    Vs=Vys=Ws=Ss_steer=0; yaw_hold_set=false; mpc_t=-1.0;
    ht_cur=qhome_h=body_h=q.base_z0; for(int i=0;i<4;i++) have_prev[i]=false;
    haunch_ready=false; haunch_fold=0; jphase=-1;
  }

  // 1틱 제어: d->ctrl 설정(mj_step은 호출자). q.d->time 기준.
  void control(){
    mjModel*m=q.m; mjData*d=q.d; int nv=q.nv; double dt=m->opt.timestep;
    double t=d->time; int nu=q.nu;
    // ── 모드 dispatch(배포용): move 외 = 서기/눕기/getup/off ──
    if(mode!="sit") q.sit_pitch+=tc_clip(0.0-q.sit_pitch,-1.2*dt,1.2*dt);   // ★nose-up 부드럽게 해제(리셋 아닌 슬루=언폴드 중 뒷다리 펴짐과 함께 nose-up 풀림)
    if(mode!="sit") sit_init=false;                                          // ★sit 이탈=진입방향 latch 리셋(다음 진입서 재판정)
    q.posture_w=1.0; q.sit_hock_contact=false;   // ★자세task 가중·뒤 hock접촉 기본off(서기/보행). 개-앉기 홀드서만 on
    if(mode!="jump" && jphase>=0) jphase=-1;      // 점프 중 다른 모드로 이탈=상태 초기화(재진입 정상화)
    // ★달리다 서기(stand_up): 전진속도 크면 먼저 move(명령0)로 감속→발이 CoM 밑으로 재중심된 뒤 stand. (뒤 엉덩방아 방지)
    if(mode=="stand_up" && armed){ double sp=std::hypot(d->qvel[0],d->qvel[1]);
      if(sp>0.20) stop_settle=true; else if(sp<0.12) stop_settle=false; }
    else stop_settle=false;
    if(mode!="off") off_settled=false;   // ★off 벗어나면 리셋(다음 off서 다시 완만 하강)
    bool run_move=(mode=="move")||stop_settle;
    if(!run_move){
      if(mode=="off"){   // ★전원차단: 선 채로 툭 damp(낙하) 대신 바닥(GROUND_Z)까지 완만 PD 하강 후 damp (실로봇=눕고 전원끔)
        double bzo=d->qpos[2];
        was_sit=false; sit_getup_t0=-1; from_sit=false; haunch_ready=false; haunch_fold=0; jphase=-1; armed=false;
        if(off_settled || bzo<=GROUND_Z+0.04){   // 바닥 근처 → damp(전원차단 등가)
          off_settled=true; have_qref=false;
          for(int j=0;j<nu;j++) d->ctrl[j]=tc_clip(-REST_KD*d->qvel[6+j],-q.tau_peak[j],q.tau_peak[j]); return; }
        // 아직 높음 → GROUND_Z까지 완만 하강(PD fold): 서 있으면 눕히고, 눕는 중이면 계속
        if(ht_cur>bzo+0.06 || ht_cur<0.06) ht_cur=bzo;   // 진입 시 현재높이 동기화
        ht_cur+=tc_clip(GROUND_Z-ht_cur,-GETUP_RATE*dt,GETUP_RATE*dt);
        if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; }
        if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
        for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        return; }
      if(mode=="jump"){   // ★★J2 통합 점프: crouch(wbic)→OCP궤적 재생(thrust/flight τ_ff+PD)→touchdown→wbic_stance 착지.
        double bz=d->qpos[2];
        int ncon=0; for(int ci=0;ci<d->ncon;ci++){ int g1=d->contact[ci].geom1,g2=d->contact[ci].geom2;
          for(int fi=0;fi<4;fi++) if(g1==q.fgid[fi]||g2==q.fgid[fi]){ ncon++; break; } }
        if(jphase<0){ jphase=0; jt0=t; jzpk=bz; ht_cur=std::max(JUMP_CROUCH_Z,bz); qhome_h=-1; jump_k=-1; }
        if(jphase==0){   // crouch: wbic_stance로 JUMP_CROUCH_Z 정착 → OCP 궤적 로드
          ht_cur+=tc_clip(JUMP_CROUCH_Z-ht_cur,-0.4*dt,0.4*dt);
          if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; }
          q.wbic_stance();
          if(bz<=JUMP_CROUCH_Z+0.015 && std::abs(d->qvel[2])<0.06 && t-jt0>0.45){
            // ★S3-b Step2: crouch 정착 → 신선 궤적 확보. crocoddyl 있으면 이 자리서 live-solve(~150ms 1회 stall,
            //   crouch 예산 내) → 명령 vx로 거리 조정. 없거나 실패 시 /tmp/jump_traj.txt replay fallback.
#ifdef HAVE_CROCODDYL
            if(!solve_jump_live(JUMP_VX)){ if(jump_N<=0) load_jump("/tmp/jump_traj.txt"); }
#else
            if(jump_N<=0) load_jump("/tmp/jump_traj.txt");
#endif
            jump_k=0; jump_kt=0; jphase=1; jt0=t; jzpk=bz; }
          armed=false; return; }
        if(jphase==1){   // ★OCP 궤적 재생(push+flight): τ_ff + 관절 PD (없으면 구 스크립트 스냅 fallback)
          if(jump_N>0 && jump_k<jump_N){
            for(int j=0;j<nu;j++){ double tau=jump_tau[jump_k][j]+JUMP_TRAJ_KP*(jump_q[jump_k][j]-d->qpos[7+j])+JUMP_TRAJ_KD*(jump_dq[jump_k][j]-d->qvel[6+j]);
              d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
            jzpk=std::max(jzpk,bz);
            jump_kt+=dt; if(jump_kt>=jump_dt && jump_k<jump_N-1){ jump_kt=0; jump_k++; }
            bool airborne=(jzpk>JUMP_CROUCH_Z+0.10);
            if(jump_ph[jump_k]>=1 && airborne && ncon>=2 && bz<jzpk-0.02){ jphase=3; jt0=t; }   // touchdown → wbic 착지
            return; }
          if(jump_N<=0){   // 궤적 없음: 구 스크립트 스냅(fallback)
            q.update_stand_qhome(0.52);
            for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+JUMP_KP*(q.q_home[j]-d->qpos[7+j])-JUMP_KD*d->qvel[6+j];
              d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
            jzpk=std::max(jzpk,bz);
            if((ncon==0||t-jt0>JUMP_THRUST_T) && jzpk>JUMP_CROUCH_Z+0.10 && ncon>=2 && bz<jzpk-0.02){ jphase=3; jt0=t; }
            return; }
          jphase=3; jt0=t; return; }   // 궤적 끝 → 착지
        // jphase==3: land — 짧은 저-PD 흡수 → wbic_stance 회복 → 서기 인계
        if(t-jt0<0.06){ for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+45*(q.q_home[j]-d->qpos[7+j])-6*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); } return; }
        ht_cur=std::max(0.30,bz); qhome_h=-1; q.wbic_stance();
        if(t-jt0>0.6){ mode="stand_up"; jphase=-1; }
        return; }
      if(mode=="sit"){   // ★개-앉기(haunch sit): 저crouch로 하강 → 정착 후 crouch→haunch(뒷다리 접어 발링크 바닥밀착) fold + nose-up. 기립=from_sit
        was_sit=false; sit_getup_t0=-1; from_sit=true;
        double bz=d->qpos[2];
        if(!sit_init){ sit_from_below=(bz<SIT_Z-0.02); sit_init=true; }     // ★진입방향 1회 판정: 눕기 등 SIT_Z(0.32)보다 낮게 진입=from_below
        if(sit_from_below){   // ★아래(눕기)서 진입: 0.32 crouch로 안 올림. 현재 자세를 blend 시작점으로 캡처 + haunch 즉시 계산 → settled 대기 없이 직행 morph(base는 다리 접힘 기하로 ~0.25까지만 자연 상승, "일어섰다 다시앉기" 오버슈트 제거)
          if(!haunch_ready){
            for(int j=0;j<nu;j++) q_crouch[j]=d->qpos[7+j];                 // 시작점=현재 눕기 자세
            com_crouch=Vector3d(d->subtree_com[0],d->subtree_com[1],d->subtree_com[2]);
            q.haunch_sit_home(HAUNCH_Z,HAUNCH_PITCH); q_haunch=q.q_home; com_haunch=q.com_ref; haunch_ready=true;
            ht_cur=std::max(bz,0.15); qhome_h=-1;                           // ht_cur=현재높이 동기(위로 램프 안 함)
          }
        } else {              // ★위(서기)서 진입: 기존 로직 — 0.32로 하강 후 fold
          if(bz<GETUP_TRIG && ht_cur>GETUP_DONE) ht_cur=std::max(0.12,bz); // 쓰러짐서 낮음→동기화
          ht_cur+=tc_clip(SIT_Z-ht_cur,-GETUP_RATE*dt,GETUP_RATE*dt);      // 천천히 SIT_Z로
          if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; q_crouch=q.q_home; com_crouch=q.com_ref; }  // crouch 원본(blend용)
          bool settled=(std::abs(ht_cur-SIT_Z)<0.03 && bz>0.29);          // SIT_Z 정착 후에만 fold+nose-up
          if(settled && !haunch_ready){ q.haunch_sit_home(HAUNCH_Z,HAUNCH_PITCH); q_haunch=q.q_home; com_haunch=q.com_ref; haunch_ready=true; }  // ★haunch 1회 계산(nose-up 베이킹→PD홀드가 재현)
        }
        double tf=haunch_ready?1.0:0.0;                                    // ★접힘 시작 후 계속 유지(자세 홀드). settled는 최초 트리거만 담당(앉으면 bz<0.29로 settled 풀려 되펴지던 버그 수정)
        double frate=HAUNCH_FOLD_RATE*std::max(0.18,1.0-haunch_fold);       // ★ease-out: fold 끝(→1)에서 느려져 엉덩이 살포시 착지(엉덩방아 완화). 시작은 빠름
        if(sit_from_below) frate=HAUNCH_FOLD_RATE*SIT_BELOW_SPEED*std::max(0.5,1.0-haunch_fold);  // ★눕기서 진입=엉덩이 이미 낮음→빠른 fold + 얕은 ease-out(tail 완만 불필요, 기어가는 꼬리 제거)
        haunch_fold+=tc_clip(tf-haunch_fold,-frate*dt,frate*dt);
        if(haunch_ready){ double ff=std::min(1.0,haunch_fold*FRONT_PULL_SPEED); int jfront=q.legqp[2][0]-7;  // ★앞다리(FL_hip 이후 관절)=더 빠른 fold로 조기 끌어당김. 뒤(엉덩방아)는 haunch_fold 원래 pace
          for(int j=0;j<nu;j++){ double f=(j>=jfront)?ff:haunch_fold; q.q_home[j]=(1-f)*q_crouch[j]+f*q_haunch[j]; }
                          q.com_ref=(1-haunch_fold)*com_crouch+haunch_fold*com_haunch; }
        if(haunch_ready){ double froll=tc_clip((haunch_fold-0.65)/0.35,0.0,1.0);   // ★뒷발 굴려 착지: 착지중(fold<0.65)=LAND각(발 안 부딪힘) → 바닥 닿은 뒤(fold→1)=CONTACT각(발바닥 밀착)
          double footang=(1.0-froll)*q.HAUNCH_FOOT_LAND+froll*q.HAUNCH_FOOT;
          for(int i=0;i<2;i++) q.q_home[q.legqp[i][3]-7]=footang; }
        q.sit_pitch+=tc_clip(0.0-q.sit_pitch,-1.2*dt,1.2*dt);              // nose-up은 q_home 베이킹(강성 PD홀드)
        double kp=(ht_cur<0.31||bz<0.295)?GETUP_KP:SIT_KP;                 // 하강복구=완만 / 개-앉기 정착=강성(발링크 평평 강제)
        if(ht_cur<0.31 || bz<0.295 || haunch_ready){                       // ★개-앉기 정착(또는 저자세 복구)=중력보상 PD 홀드. 사용자=기립불요·자세우선 → 강성 PD로 접힘+발링크 바닥밀착 강제
          if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
          for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
          for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+kp*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
            d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
          armed=false; return; }
        have_qref=false; q.wbic_stance(); armed=false; return; }           // 정착 전(하강 crouch)=wbic 균형
      have_qsit=false;
      double bz=d->qpos[2];
      // ★앉기→서기 스크립트 기립: 앞다리 굽혀 앞발 들기(폴볼트 차단) + 뒷다리 박차 extend → 몸 올라오면 정상 stance로 인계
      if(was_sit && mode=="stand_up"){
        // ★정적 front-retraction 기립: Phase1 앞발 들어 올림(폴볼트 차단) → Phase2 앞발 몸밑 재배치(q_home) + 뒷다리 세움
        //   → 수평 저crouch 도달 후 정상 getup(off→서기 검증됨)으로 인계. 뒷발 planted(얕은sit) 전제.
        if(sit_getup_t0<0){ sit_getup_t0=t; for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; q.update_stand_qhome(SGU_GATHER_Z); qhome_h=SGU_GATHER_Z; ht_cur=SGU_GATHER_Z; }
        double te=t-sit_getup_t0;
        for(int i=0;i<4;i++){ bool fr=(std::string(q.legs[i])=="FL"||std::string(q.legs[i])=="FR");
          double th,cf,ft,hp;
          if(fr && te<SGU_KICK_T){ hp=0; th=SGU_FB_THIGH; cf=SGU_FB_CALF; ft=-0.70; }   // Phase1: 앞다리 굽혀 발 들기(지면서 뗌)
          else { hp=q.q_home[q.legqp[i][0]-7]; th=q.q_home[q.legqp[i][1]-7];             // Phase2: 앞다리→몸밑 착지, 뒷다리→저crouch
                 cf=q.q_home[q.legqp[i][2]-7]; ft=q.leg_dof[i]==4?q.q_home[q.legqp[i][3]-7]:0; }
          double tar[4]={hp,th,cf,ft};
          for(int cc=0;cc<q.leg_dof[i];cc++){ int j=q.legqp[i][cc]-7;
            q_ref[j]+=tc_clip(tar[cc]-q_ref[j],-SGU_SLEW*dt,SGU_SLEW*dt); } }
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+SGU_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        double jerr=0; for(int j=0;j<nu;j++) jerr+=std::abs(q.q_home[j]-d->qpos[7+j]); jerr/=nu;
        if(te>SGU_KICK_T && tiltdeg()<SGU_DONE_TILT && jerr<0.4){ was_sit=false; sit_getup_t0=-1; }  // 수평 저crouch 도달→정상 getup 인계
        armed=false; return;
      }
      // ★개-앉기 기립: offline gather 궤적(/tmp/getup_traj.txt) 추종. phaseA(gather+뒷발착지)=PD로 CoM 전진 → phaseB=정상 wbic 상승 인계.
      if(mode=="stand_up" && haunch_ready){
        if(getup_k<0){ load_getup("/tmp/getup_traj.txt"); getup_k=0; getup_kt=0; }
        if(getup_N>0 && getup_ph[getup_k]<2){                                // phaseA: 궤적 프레임 PD추종(중력보상 + 속도 피드포워드)
          for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_TRAJ_KP*(getup_q[getup_k][j]-d->qpos[7+j])+GETUP_TRAJ_KD*(getup_dqv[getup_k][j]-d->qvel[6+j]);
            d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
          getup_kt+=dt; if(getup_kt>=getup_dt && getup_k<getup_N-1){ getup_kt=0; getup_k++; }
          armed=false; return;
        }
        haunch_ready=false; haunch_fold=0; getup_k=-1;                       // phaseB 진입(또는 궤적 없음)→ 아래 정상 getup(wbic 상승)으로 인계
        ht_cur=std::max(0.20,d->qpos[2]); qhome_h=-1; have_qref=false;
      }
      if(bz<GETUP_TRIG && ht_cur>GETUP_DONE) ht_cur=std::max(0.12,bz);      // 쓰러짐/off로 낮음→동기화
      if(mode=="stand_down" && ht_cur>bz) ht_cur=std::max(GROUND_Z,bz);    // ★눕기=현재높이서 하강(서기 안 거치고 그대로 눕기)
      if(mode=="stand_down"){ from_sit=false; haunch_ready=false; haunch_fold=0; }  // ★눕기=저자세 leg-fold(PD)로 낮게 눕힘. from_sit=false→아래 fold 경로 사용(wbic 0.29 아님)
      if(mode=="stand_up" && bz>0.47) from_sit=false;                        // 서기 완료→해제
      if(!stand_set){ stand_ax=d->subtree_com[0]; stand_ay=d->subtree_com[1]; stand_set=true; }  // ★서기 진입=현재 위치 캡처(홈 x=0으로 안 빨려가게)
      double tgt=(mode=="stand_down")?GROUND_LIE_Z:body_h;                  // ★눕기=저자세(0.22) leg-fold / 서기=슬라이더
      bool low=(ht_cur<GETUP_DONE)||(tgt<GETUP_DONE); double rate=low?GETUP_RATE:HRATE;
      ht_cur+=tc_clip(tgt-ht_cur,-rate*dt,rate*dt);
      if(std::abs(ht_cur-qhome_h)>6e-3 || mode=="stand_down"){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; q_crouch=q.q_home; com_crouch=q.com_ref; }  // ★눕기=매틱 재계산(LUT, 저비용): 아래 앞다리 오프셋을 깨끗한 base q_home에 얹어 += 누적/슬라이더 무반응 방지(height 고정 시 재계산 스킵되던 버그)
      if(mode=="stand_down") for(int i=0;i<4;i++) if(q.leg_dof[i]==4){   // ★앞뒤 발목 접기 + 앞다리 fold(GUI 슬라이더로 CoM균형·수평·무슬라이드 조각)
        q.q_home[q.legqp[i][3]-7]=(i<2)?GROUND_REAR_FOOT:GROUND_FRONT_FOOT;   // 발목: 뒤/앞
        if(i>=2){ q.q_home[q.legqp[i][1]-7]+=GROUND_FRONT_THIGH; q.q_home[q.legqp[i][2]-7]+=GROUND_FRONT_CALF; } }  // 앞다리 thigh/calf 오프셋(접기)
      if(haunch_ready && mode=="stand_up"){   // ★개-앉기서 기립: 높이-스케줄 언폴드(HAUNCH_Z서 fold=1 → UNFOLD_Z서 0). 몸 오르며 뒷다리 펴짐→q_home 점프 없이 인계
        double target_fold=tc_clip((HAUNCH_UNFOLD_Z-ht_cur)/(HAUNCH_UNFOLD_Z-HAUNCH_Z),0.0,1.0);
        haunch_fold+=tc_clip(target_fold-haunch_fold,-HAUNCH_FOLD_RATE*dt,HAUNCH_FOLD_RATE*dt);
        for(int j=0;j<nu;j++) q.q_home[j]=(1-haunch_fold)*q_crouch[j]+haunch_fold*q_haunch[j];
        q.com_ref=(1-haunch_fold)*com_crouch+haunch_fold*com_haunch;
        if(ht_cur>=HAUNCH_UNFOLD_Z-1e-3 && haunch_fold<0.02){ haunch_ready=false; haunch_fold=0; }   // 언폴드 완료→순수 crouch
      }
      double jerr=0; for(int j=0;j<nu;j++) jerr+=std::abs(q.q_home[j]-d->qpos[7+j]); jerr/=nu;
      // ★눕기=wbic_stance로 저크라우치(0.29) 능동홀드(슬라이드·붕괴 없음). 진짜 belly-flat은 다중접촉문제(haunch-getup과 동일)라 보류.
      // (구 damp 붕괴는 저크라우치서 발 미끄러져 ~0.4m 슬라이드 → 제거)
      double foldZ=GETUP_DONE;                                             // ★눕기·getup 모두 저-PD fold(ht_cur<0.40): 눕기=저자세 leg-fold 홀드, getup=상승 fold. (구 눕기=wbic0.29는 발목으로 못 낮춰 leg-fold로 전환)
      if(ht_cur<foldZ && !from_sit){                                        // 낮은자세=수평 PD fold(눕기/getup). ★from_sit(crouch-sit≥0.29)=wbic_stance로 매끈 기립
        if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
        for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        armed=false; return; }
      if(!haunch_ready){ q.com_ref[0]=stand_ax; q.com_ref[1]=stand_ay; }    // ★서기=멈춘 위치서 홀드(홈 x=0으로 빨려감 방지). haunch getup은 블렌드 com_ref 유지
      have_qref=false; q.wbic_stance(); armed=false; return;                // 서기(높이충분)=wbic_stance
    }
    have_qref=false; stand_set=false;   // move → fold·서기앵커 리셋(다음 서기서 새 위치 캡처)
    auto quat_yaw=[&](){ double*qq=&d->qpos[3]; return std::atan2(2*(qq[0]*qq[3]+qq[1]*qq[2]),1-2*(qq[2]*qq[2]+qq[3]*qq[3])); };
    if(t < settle_until){ q.wbic_stance(); return; }
    if(!armed){ armed=true; t0=t; yaw_ref=0;
      for(int i=0;i<4;i++){ nominal[i]=q.foot_point(i); liftoff[i]=q.foot_point(i); hip_off[i]=q.foot_hip_off[i]; gz[i]=q.foot_gz0[i]; }
      x_ref.setZero(); x_ref[5]=d->subtree_com[2]; x_ref[12]=-9.81; com_h0=d->subtree_com[2]; }   // com_h0=평지 위 CoM 명목높이
    double tg=t-t0; bool go=tg>TC_WARMUP;
    double vt=go?V:0.0, vyt=go?VY:0.0, wt=go?WZ:0.0;
    if(stop_settle){ vt=vyt=wt=0.0; }   // ★정지 감속: 명령 0 → MPC/raibert가 속도 죽이며 발을 CoM 밑으로 재중심
    double acc=stop_settle?1.6:TC_ACC;   // ★달리다 서기: 감속을 빠르게(1.6 m/s²)해 코스팅 단축(구 TC_ACC 0.6=3m 코스팅→불안)
    Vs+=tc_clip(vt-Vs,-acc*dt,acc*dt); Vys+=tc_clip(vyt-Vys,-acc*dt,acc*dt); Ws+=tc_clip(wt-Ws,-2.0*dt,2.0*dt);
    double stt=go?steer:0.0; Ss_steer+=tc_clip(stt-Ss_steer,-0.8*dt,0.8*dt);   // 조향각 스무딩[rad/s]
    double Veff=Vs,Vyeff=Vys,Weff=Ws; Veff_dbg=Veff;
    double steer_wz = tc_clip(Veff*std::tan(tc_clip(Ss_steer,-1.2,1.2))/wheelbase, -0.9, 0.9);  // ★자동차식 조향(Ackermann) 성분. yaw rate 캡0.9(understeer). V=0=무효, WZ스핀과 공존
    Weff += steer_wz;
    q.waist_ref=tc_clip(waist_steer*steer_wz,-waist_cap,waist_cap);    // ★허리 lean=조향(핸들)만 구동. 우스틱 WZ 선회는 순수 다리선회(허리 중립홀드). 핸들 0이면 허리 중립
    double spd=std::abs(Veff);   // ★전진속도만 whip 트리거(좌우이동은 whip 유발 안함). 구 hypot은 측방서도 whip 켜짐
    if(gait_type=="walk"){   // ★walk: 체크박스(auto_whip)로 whip on/off. on=슬라이더 강도 직접적용 / off=매끈(whip_hi)
      if(auto_whip){ q.swing_w_f=whip_lo_f; q.swing_w_r=whip_lo_r; } else { q.swing_w_f=whip_hi; q.swing_w_r=whip_hi; } }
    else if(auto_whip){   // ★trot: 속도↑ → whip↑. swing_w 를 whip_hi(저속,매끈)→whip_lo(고속,슬라이더값) 선형보간
      double s=tc_clip((spd-whip_v0)/(whip_v1-whip_v0),0.0,1.0);
      q.swing_w_f=whip_hi+s*(whip_lo_f-whip_hi); q.swing_w_r=whip_hi+s*(whip_lo_r-whip_hi);
    } else { q.swing_w_f=whip_lo_f; q.swing_w_r=whip_lo_r; }   // 수동=슬라이더값 상수
    double lat=tc_clip(std::abs(Vyeff)/0.35,0.0,1.0);   // ★좌우이동 시 whip 억제: 측방 스윙 flail 완화(swing_w→매끈 페이드). walk 측방붕괴는 게이트 고유(whip 무관)
    q.swing_w_f=q.swing_w_f+lat*(whip_hi-q.swing_w_f); q.swing_w_r=q.swing_w_r+lat*(whip_hi-q.swing_w_r);
    double yaw_m=quat_yaw();
    if(std::abs(Weff)>0.02){ yaw_ref=tc_clip(yaw_ref+Weff*dt,yaw_m-0.3,yaw_m+0.3); yaw_hold_set=false; }
    else { if(!yaw_hold_set){ yaw_hold=yaw_m; yaw_hold_set=true; } yaw_ref=yaw_hold; }
    double cy=std::cos(yaw_m), sy=std::sin(yaw_m);
    double vx_w=Veff*cy-Vyeff*sy, vy_w=Veff*sy+Vyeff*cy;
    // ★위치홀드: 전진/측방명령 0이면 base 위치 앵커링. SPIN_HOLD=제자리선회(V=0,WZ≠0)서도 유지 → 허리조향 표류 상쇄(베이스 기준 wz 선회)
    bool ph_turn = SPIN_HOLD ? true : (std::abs(Weff)<0.05);
    if(POS_HOLD && std::abs(Veff)<0.03 && std::abs(Vyeff)<0.03 && ph_turn){
      if(!pos_hold_set){ phx=d->qpos[0]; phy=d->qpos[1]; pos_hold_set=true; }
      vx_w+=tc_clip(-0.6*(d->qpos[0]-phx),-0.15,0.15); vy_w+=tc_clip(-0.6*(d->qpos[1]-phy),-0.15,0.15);
    } else pos_hold_set=false;
    x_ref[2]=yaw_ref; x_ref[8]=Weff; x_ref[9]=vx_w; x_ref[10]=vy_w;
    // ★perceptive 몸통높이(Python 동일): 4-hip 평균 지형높이=_body_terr → MPC(x_ref[5])·WBIC z-task(quad_control:354) 양쪽 일관 적용. 평지=0(무변화)
    // ★perceptive 몸통높이: base 1점 지형높이+슬루 → MPC x_ref[5]만(WBIC z-task엔 미공급=_body_terr 0).
    //   실측(course): WBIC 공급시 tilt 3.7→4.3°·4힙평균 6.1° 로 오히려 나빠짐 → 보행중 몸통z는 MPC가 지배, WBIC 지형공급은 진동만 추가.
    // ★body_h(서기+보행 통합 높이): 보행 중에도 body_h를 부드럽게 추종 → update_stand_qhome로 com_ref/q_home/com_h0 갱신(MPC·WBIC 정합). 서기와 동일 슬라이더.
    //   ★★버그수정: body_h와 현 높이 차이로 게이팅(구 hr-qhome_h는 1틱램프량이라 3e-3 못 넘어 영영 미호출). 램프 중 매틱 update.
    if(std::abs(body_h-qhome_h)>2e-3){ qhome_h+=tc_clip(body_h-qhome_h,-0.3*dt,0.3*dt); q.update_stand_qhome(qhome_h); com_h0=q.com_ref[2]; }
    double bt=0.0; if(perceptive){ double tz=q.terrain_z(d->qpos[0],d->qpos[1]); if(tz>-50.0) bt=tz; }
    _bterr_s+=tc_clip(bt-_bterr_s,-0.5*dt,0.5*dt); q._body_terr=0.0; x_ref[5]=com_h0+_bterr_s;
    std::vector<int> st; std::map<int,std::pair<Vector3d,Vector3d>> swing;
    for(int i=0;i<4;i++){ bool sch; double sp; gait(i,tg,sch,sp);
      if(sch){ st.push_back(i); have_prev[i]=false; } else { if(sp<0.03) liftoff[i]=q.foot_point(i); } }
    std::vector<double> jcb(3*nv); mj_jacSubtreeCom(m,d,jcb.data(),0);
    Matrix<double,3,Dynamic> Jc(3,nv); for(int r=0;r<3;r++)for(int c=0;c<nv;c++) Jc(r,c)=jcb[r*nv+c];
    Map<VectorXd> qv(d->qvel,nv); Vector3d vcom=Jc*qv;
    Vector2d v_des(vx_w,vy_w), v_fb=vcom.head(2);
    if(ALIP){ mj_subtreeVel(m,d); Vector3d L(d->subtree_angmom[0],d->subtree_angmom[1],d->subtree_angmom[2]);
      double H=std::max(0.1,d->subtree_com[2]); v_fb+=Vector2d(L[1],-L[0])/(q.mpc.TOTAL_MASS*H); }
    Vector2d rai; for(int k=0;k<2;k++) rai[k]=tc_clip(raibert_k*gp_Tst*v_des[k]+TC_KCAP*(v_fb[k]-v_des[k]),-TC_RAICLIP,TC_RAICLIP);
    Matrix2d Rw; Rw<<cy,-sy,sy,cy; double sh=step_h*(0.2+0.8*std::min(1.0,tg/TC_WARMUP));
    double wa=(q.waist_idx>=0)?d->qpos[7+q.waist_idx]:0.0;         // ★허리각(앞몸통이 base대비 꺾인 정도)
    double cyf=std::cos(yaw_m+wa), syf=std::sin(yaw_m+wa); Matrix2d Rwf; Rwf<<cyf,-syf,syf,cyf;  // 앞몸통 방향(base+허리 yaw)
    for(int i=0;i<4;i++){ bool sch; double s_; gait(i,tg,sch,s_); if(sch) continue;
      Vector2d hip_xy(d->xpos[q.hip_bid[i]*3],d->xpos[q.hip_bid[i]*3+1]);
      Vector2d r_xy=hip_xy-Vector2d(d->qpos[0],d->qpos[1]);        // 몸중심→hip
      Vector2d tw=Weff*gp_Tst*Vector2d(-r_xy[1],r_xy[0]);          // ★선회 접선 발배치(yaw) — 없으면 회전시 표류·붕괴
      bool frontleg=(std::string(q.legs[i])=="FL"||std::string(q.legs[i])=="FR");  // ★앞다리=앞몸통방향(허리반영)
      Vector2d pe_xy=hip_xy+(frontleg?Rwf:Rw)*hip_off[i]+rai+tw;
      double land_z=gz[i];                                          // ★기본=평지 참조(foot_gz0)
      if(perceptive){ double tz=q.terrain_z(pe_xy[0],pe_xy[1]); if(tz>-50.0) land_z=gz[i]+tz; }  // ★착지 z=평지gz+지형높이(Python 동일). 평지 tz=0=무변화
      Vector3d p_end(pe_xy[0],pe_xy[1],land_z);
      double dzl=p_end[2]-liftoff[i][2]; Vector3d bvel(vcom[0],vcom[1],0.0);
      double sh_i=sh; if(perceptive && dzl>0.005) sh_i=sh+dzl+PCV_CLR;   // ★상향 스텝: apex를 착지높이+여유 위로(라이저 헛디딤 방지)
      Vector3d p_tgt=tc_swing_foot(s_,liftoff[i],p_end,bvel,sh_i,gp_Tsw,gp_Tst);
      p_tgt[2]+=dzl*(10*std::pow(s_,3)-15*std::pow(s_,4)+6*std::pow(s_,5));
      Vector3d v_tgt=Vector3d::Zero();
      if(have_prev[i]) for(int c=0;c<3;c++) v_tgt[c]=tc_clip((p_tgt[c]-ptgt_prev[i][c])/dt,-1.0,1.0);
      ptgt_prev[i]=p_tgt; have_prev[i]=true; swing[i]={p_tgt,v_tgt}; }
    double dmpc=t-mpc_t;
    if(!st.empty() && (mpc_t<0||dmpc<0||dmpc>=q.mpc.DT)){
      std::vector<std::array<int,4>> cs(q.mpc.N);
      for(int k=0;k<q.mpc.N;k++) for(int i=0;i<4;i++){ bool sch; double sp; gait(i,tg+k*q.mpc.DT,sch,sp); cs[k][i]=sch?1:0; }
      Matrix<double,4,3> L=q.mpc_grf(x_ref,cs); for(int i=0;i<4;i++) lam_des[i]=L.row(i).transpose(); mpc_t=t; }
    Vector3d lam_use[4]; for(int i=0;i<4;i++) lam_use[i]= st.empty()?Vector3d::Zero():lam_des[i];
    q.yaw_des=yaw_ref;                                     // ★자세 task가 명령헤딩 추종(선회시 yaw와 안싸움)
    if(!q.wbic_track(st,swing,lam_use)) q.wbic_stance();
  }
  double tiltdeg(){ double R[9]; mju_quat2Mat(R,&q.d->qpos[3]); return std::acos(tc_clip(R[8],-1,1))*180/M_PI; }
};

// 17dof 튜닝 게인 적용(env 우선). 14dof는 기본값 유지.
static inline void apply_env_gains(QuadControl& q){
  if(getenv("BASE_Z0")) q.base_z0=atof(getenv("BASE_Z0"));
  if(getenv("REAR_ANKLE")){ q.REAR_ANKLE=atof(getenv("REAR_ANKLE")); q.FRONT_ANKLE=q.REAR_ANKLE; }
  if(getenv("FRONT_ANKLE")) q.FRONT_ANKLE=atof(getenv("FRONT_ANKLE"));
  if(getenv("W_AM")) q.W_AM=atof(getenv("W_AM"));
  if(getenv("KD_AM")) q.KD_AM=atof(getenv("KD_AM"));
  if(getenv("W_ORI")) q.w_ori=atof(getenv("W_ORI"));
  if(getenv("W_YAW")) q.w_yaw=atof(getenv("W_YAW"));      // yaw 헤딩홀드 가중(17dof=0권장:선회민감, 14dof=5:직진드리프트방지)
  if(getenv("MU")){ q.MU=atof(getenv("MU")); q.mpc.MU=q.MU*q.MU_MARGIN; }  // ★마찰콘 μ(물리 geom=1.3, 기존 0.6은 과보수→선회 GRF 포화)
  if(getenv("MOTOR_CURVE")) q.motor_curve=true;    // ★토크-속도곡선: 고속서 가용토크↓(속도한계 QP 반영)
  if(getenv("WAIST_W")) q.WAIST_W=atof(getenv("WAIST_W"));    // ★허리 홀드가중
  if(getenv("WAIST_KP")) q.WAIST_KP=atof(getenv("WAIST_KP")); if(getenv("WAIST_KD")) q.WAIST_KD=atof(getenv("WAIST_KD"));
  if(getenv("SWING_W")){ double v=atof(getenv("SWING_W")); q.swing_w_r=v; q.swing_w_f=v; }
  if(getenv("SWING_W_R")) q.swing_w_r=atof(getenv("SWING_W_R"));
  if(getenv("SWING_W_F")) q.swing_w_f=atof(getenv("SWING_W_F"));
  if(getenv("PIN_ANKLE")) q.stance_pin_ankle=true;
}
