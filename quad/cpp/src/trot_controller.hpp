// TrotCtrl — mode_trot 핵심경로 1틱 제어(설정/스윙/MPC/WBIC → d->ctrl). trot_sim(헤드리스)·trot_view(뷰어) 공유.
#pragma once
#include "quad_control.hpp"
#include <vector>
#include <array>
#include <map>
#include <cmath>

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
  bool ALIP=false, POS_HOLD=true;   // ★ALIP 기본 off: push복구 이득 미미(300N서만 22vs26°)한데 지속선회 붕괴시킴. ALIP=1로 켬
  bool SPIN_HOLD=false;             // ★제자리선회(V=0,WZ≠0)서도 위치홀드 유지 → 허리조향 표류 상쇄(베이스기준 wz선회). 주행선회엔 영향無
  // ── 모드관리(배포용): move/stand_up(서기)/stand_down(눕기)/off ──
  std::string mode="move";
  double body_h=0.5234, ht_cur=0.5234, qhome_h=0.5234;   // 서기높이 슬라이더·보간높이·q_home 계산높이
  VectorXd q_ref; bool have_qref=false;                  // fold 관절목표 slew
  double SIT_Z=0.32, SIT_PITCH=0.70, SIT_REAR_FOOT=-1.35, SIT_REAR_CALF=1.10, SIT_REAR_THIGH=-0.6; bool have_qsit=false; // ★앉기=crouch-sit(SIT_Z 저crouch, 4발 planted→기립가능). SIT_PITCH등은 구 haunch-sit 잔여
  double SIT_CPITCH=1.0, SIT_REACH=0.08;   // ★앉기 nose-up 목표(~25° 앞올림=앉은자세, wbic_stance 능동제어→안정+기립가능). SIT_REACH 미사용(잔여)
  // ★앉기→서기 스크립트 기립(앞다리 굽혀 앞발 들어 폴볼트 차단 + 뒷다리 박차 extend). 앉기에서만 발동.
  bool was_sit=false; double sit_getup_t0=-1;
  bool from_sit=false;   // ★crouch-sit서 기립: 저crouch(≥0.29)라 저-PD 대신 wbic_stance로 매끈 기립(오버슈트 방지)
  double SGU_KICK_T=0.5, SGU_FB_THIGH=-0.55, SGU_FB_CALF=1.20, SGU_SLEW=1.5, SGU_KP=120, SGU_GATHER_Z=0.24, SGU_DONE_TILT=22;
  double SGU_WALKOUT_V=0.6, SGU_HANDOFF_Z=0.34;   // ★기립 후 전진 트로트로 인계(walk-out)해 균형회복. bz>HANDOFF_Z면 move로 전환
  double SIT_SLEW=0.6; // ★앉기 하강 슬루(rad/s, 작을수록 천천히·충격↓). 기본 JOINT_SLEW(1.5)보다 느리게→충격 완화
  // 모드관리 상수(Python 17dof와 동일)
  double GROUND_Z=0.18, GETUP_TRIG=0.32, GETUP_DONE=0.40, GETUP_KP=90, GETUP_KD=3, GETUP_RATE=0.18, REST_KD=3.0, JOINT_SLEW=1.5, HRATE=0.3;
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
  double Vs=0,Vys=0,Ws=0, yaw_ref=0; bool yaw_hold_set=false; double yaw_hold=0;
  bool pos_hold_set=false; double phx=0,phy=0;
  VectorXd x_ref=VectorXd::Zero(13);
  std::array<Vector3d,4> liftoff, nominal; std::array<Vector2d,4> hip_off; std::array<double,4> gz;
  std::array<bool,4> have_prev={false,false,false,false}; std::array<Vector3d,4> ptgt_prev;
  Vector3d lam_des[4]={Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero(),Vector3d::Zero()};
  double mpc_t=-1.0, Veff_dbg=0;

  TrotCtrl(QuadControl& q_):q(q_){ q_ref=VectorXd::Zero(q.nu); body_h=ht_cur=qhome_h=q.base_z0;
    double xf=(q.d->xpos[q.hip_bid[2]*3]+q.d->xpos[q.hip_bid[3]*3])/2;   // FL,FR 힙 x(초기 yaw0=body frame)
    double xr=(q.d->xpos[q.hip_bid[0]*3]+q.d->xpos[q.hip_bid[1]*3])/2;   // HL,HR 힙 x
    wheelbase=std::max(0.15, xf-xr); }                                   // ★축거 L(≈0.61m). Ackermann 조향 반경 R=L/tanδ

  void set_gait(const std::string& g){        // trot/walk/gallop 프리셋(GUI 토글·속도트리거)
    if(g==gait_type) return; gait_type=g;
    if(g=="walk"){ gp_T=0.7; gp_SWF=0.25; gp_off[0]=0.25; gp_off[1]=0.75; gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.05; }  // ★walk 안정화(T1.0→0.7·RAI0.8→0.5): reach↓ stumble/bounce방지, 상한~0.6m/s
    else if(g=="run"){ gp_T=0.40; gp_SWF=0.5; gp_off[0]=0.0; gp_off[1]=0.5; gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.08; }  // ★고속 trot(빠른 cadence T0.4·낮은 발높이0.08): 최고속 1.8→~2.0m/s, 발목ω↓
    else if(g=="gallop"){ gp_T=0.35; gp_SWF=0.55; gp_off[0]=0.0; gp_off[1]=0.05; gp_off[2]=0.55; gp_off[3]=0.5; raibert_k=0.8; step_h=0.10; } // 회전형 갤럽(비행상 有)
    else         { gp_T=0.5; gp_SWF=0.5;  gp_off[0]=0.0;  gp_off[1]=0.5;  gp_off[2]=0.5; gp_off[3]=0.0; raibert_k=0.5; step_h=0.10; }  // ★trot 표준 중립(GRF균형·뒤thigh절반·falls=0·push복구↑)
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
  }

  // 1틱 제어: d->ctrl 설정(mj_step은 호출자). q.d->time 기준.
  void control(){
    mjModel*m=q.m; mjData*d=q.d; int nv=q.nv; double dt=m->opt.timestep;
    double t=d->time; int nu=q.nu;
    // ── 모드 dispatch(배포용): move 외 = 서기/눕기/getup/off ──
    if(mode!="sit") q.sit_pitch=0.0;   // ★nose-up은 앉기서만(다른 모드=수평)
    if(mode!="move"){
      if(mode=="off"){ for(int j=0;j<nu;j++) d->ctrl[j]=tc_clip(-REST_KD*d->qvel[6+j],-q.tau_peak[j],q.tau_peak[j]); armed=false; have_qref=false; was_sit=false; sit_getup_t0=-1; from_sit=false; return; }
      if(mode=="sit"){   // ★앉기(nose-up sitting): 저자세=저-PD 복구 → 정착 후 wbic_stance+nose-up으로 천천히 기울임(앉은자세). 기립=from_sit
        was_sit=false; sit_getup_t0=-1; from_sit=true;
        double bz=d->qpos[2];
        if(bz<GETUP_TRIG && ht_cur>GETUP_DONE) ht_cur=std::max(0.12,bz);   // 쓰러짐/눕기서 낮음→동기화
        ht_cur+=tc_clip(SIT_Z-ht_cur,-GETUP_RATE*dt,GETUP_RATE*dt);        // 천천히 SIT_Z로
        if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; }
        bool settled=(std::abs(ht_cur-SIT_Z)<0.03 && bz>0.29);            // SIT_Z 정착 후에만 nose-up
        q.sit_pitch+=tc_clip((settled?SIT_CPITCH:0.0)-q.sit_pitch,-1.2*dt,1.2*dt);  // 천천히 lean back(앉은자세)
        if(ht_cur<0.31 || bz<0.295){                                       // 저자세(prone/눕기 복구)=수평 저-PD로 발 몸밑 정렬(★실제 bz도 확인=wbic 유효높이 보장)
          if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
          for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
          for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
            d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
          armed=false; return; }
        have_qref=false; q.wbic_stance(); armed=false; return; }          // 정착=nose-up 홀드(능동균형)
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
      if(bz<GETUP_TRIG && ht_cur>GETUP_DONE) ht_cur=std::max(0.12,bz);      // 쓰러짐/off로 낮음→동기화
      if(mode=="stand_down" && ht_cur>bz) ht_cur=std::max(GROUND_Z,bz);    // ★눕기=현재높이서 하강(서기 안 거치고 그대로 눕기)
      if(mode=="stand_down") from_sit=true;                                  // ★눕기=이제 wbic 균형 저크라우치(0.29)라 기립도 wbic_stance(from_sit)로 매끈(저-PD 아님)
      if(mode=="stand_up" && bz>0.47) from_sit=false;                        // 서기 완료→해제
      double tgt=(mode=="stand_down")?0.29:body_h;                          // ★눕기=wbic 안정 저크라우치(0.29) 능동홀드(저-PD tuck 슬라이드 제거) / 서기=슬라이더
      bool low=(ht_cur<GETUP_DONE)||(tgt<GETUP_DONE); double rate=low?GETUP_RATE:HRATE;
      ht_cur+=tc_clip(tgt-ht_cur,-rate*dt,rate*dt);
      if(std::abs(ht_cur-qhome_h)>6e-3){ q.update_stand_qhome(ht_cur); qhome_h=ht_cur; }
      double jerr=0; for(int j=0;j<nu;j++) jerr+=std::abs(q.q_home[j]-d->qpos[7+j]); jerr/=nu;
      // ★눕기=wbic_stance로 저크라우치(0.29) 능동홀드(슬라이드·붕괴 없음). 진짜 belly-flat은 다중접촉문제(haunch-getup과 동일)라 보류.
      // (구 damp 붕괴는 저크라우치서 발 미끄러져 ~0.4m 슬라이드 → 제거)
      double foldZ=(mode=="stand_down")?0.0:GETUP_DONE;                     // ★눕기=저-PD fold 안 씀(wbic_stance 균형스쿼트→damp). getup(rising)만 저-PD로 발 몸밑정렬
      if(ht_cur<foldZ && !from_sit){                                        // 낮은자세=수평 PD fold(눕기/getup). ★from_sit(crouch-sit≥0.29)=wbic_stance로 매끈 기립
        if(!have_qref){ for(int j=0;j<nu;j++) q_ref[j]=d->qpos[7+j]; have_qref=true; }
        for(int j=0;j<nu;j++) q_ref[j]+=tc_clip(q.q_home[j]-q_ref[j],-JOINT_SLEW*dt,JOINT_SLEW*dt);
        for(int j=0;j<nu;j++){ double tau=d->qfrc_bias[6+j]+GETUP_KP*(q_ref[j]-d->qpos[7+j])-GETUP_KD*d->qvel[6+j];
          d->ctrl[j]=tc_clip(tau,-q.tau_peak[j],q.tau_peak[j]); }
        armed=false; return; }
      have_qref=false; q.wbic_stance(); armed=false; return;                // 서기(높이충분)=wbic_stance
    }
    have_qref=false;   // move → fold 리셋
    auto quat_yaw=[&](){ double*qq=&d->qpos[3]; return std::atan2(2*(qq[0]*qq[3]+qq[1]*qq[2]),1-2*(qq[2]*qq[2]+qq[3]*qq[3])); };
    if(t < settle_until){ q.wbic_stance(); return; }
    if(!armed){ armed=true; t0=t; yaw_ref=0;
      for(int i=0;i<4;i++){ nominal[i]=q.foot_point(i); liftoff[i]=q.foot_point(i); hip_off[i]=q.foot_hip_off[i]; gz[i]=q.foot_gz0[i]; }
      x_ref.setZero(); x_ref[5]=d->subtree_com[2]; x_ref[12]=-9.81; }
    double tg=t-t0; bool go=tg>TC_WARMUP;
    double vt=go?V:0.0, vyt=go?VY:0.0, wt=go?WZ:0.0;
    Vs+=tc_clip(vt-Vs,-TC_ACC*dt,TC_ACC*dt); Vys+=tc_clip(vyt-Vys,-TC_ACC*dt,TC_ACC*dt); Ws+=tc_clip(wt-Ws,-2.0*dt,2.0*dt);
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
    x_ref[2]=yaw_ref; x_ref[8]=Weff; x_ref[9]=vx_w; x_ref[10]=vy_w; q._body_terr=0.0;
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
      Vector2d pe_xy=hip_xy+(frontleg?Rwf:Rw)*hip_off[i]+rai+tw; Vector3d p_end(pe_xy[0],pe_xy[1],gz[i]);
      double dzl=p_end[2]-liftoff[i][2]; Vector3d bvel(vcom[0],vcom[1],0.0);
      Vector3d p_tgt=tc_swing_foot(s_,liftoff[i],p_end,bvel,sh,gp_Tsw,gp_Tst);
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
