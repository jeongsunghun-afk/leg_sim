// trot_sim — quad_mpc_wbic mode_trot 핵심경로 C++ closed-loop (헤드리스). 제어=TrotCtrl(trot_view와 공유).
// 대상: standalone 평지 trot (DETECT=0 순수스케줄). 검증: falls=0 + 전진거리·tilt를 Python과 비교.
#include "trot_controller.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

int main(int argc,char**argv){
  const char* path=argc>1?argv[1]:"../quad_real_sphere.mjcf";
  int STEPS = (argc>2)?atoi(argv[2]) : (getenv("STEPS")?atoi(getenv("STEPS")):3000);
  QuadControl q; q.load(path); apply_env_gains(q); q.crouch_home(); q.setup_mpc();
  if(getenv("QHDBG")) for(int i=0;i<4;i++) std::printf("[qhome] %s hip=%.3f thigh=%.3f calf=%.3f foot=%.3f\n",
      q.legs[i], q.q_home[q.legqp[i][0]-7], q.q_home[q.legqp[i][1]-7], q.q_home[q.legqp[i][2]-7], q.leg_dof[i]==4?q.q_home[q.legqp[i][3]-7]:0.0);
  TrotCtrl ctrl(q);
  if(getenv("TROT_V")) ctrl.V=atof(getenv("TROT_V"));
  if(getenv("BODY_H")) ctrl.body_h=atof(getenv("BODY_H"));   // ★서기 높이 테스트(슬라이더 범위 검증)
  if(getenv("STANCE_KD")) q.STANCE_KD=atof(getenv("STANCE_KD"));   // ★stance 발 속도감쇠(slip↓)
  if(getenv("TROT_VY")) ctrl.VY=atof(getenv("TROT_VY"));   // ★좌우이동(strafe) 테스트
  if(getenv("WAIST_STEER")) ctrl.waist_steer=atof(getenv("WAIST_STEER"));
  if(getenv("SPIN_HOLD")) ctrl.SPIN_HOLD=true;   // ★허리 조향 게인
  if(getenv("PERCEPTIVE")) ctrl.perceptive=atoi(getenv("PERCEPTIVE"))!=0;   // ★지형인지(mj_ray 착지높이) on/off
  if(getenv("PCV_CLR")) ctrl.PCV_CLR=atof(getenv("PCV_CLR"));
  if(getenv("MODE")) ctrl.mode=getenv("MODE");                              // ★모드 테스트(sit/stand_up/stand_down/off)
  if(getenv("SIT_Z")) ctrl.SIT_Z=atof(getenv("SIT_Z"));
  if(getenv("SIT_PITCH")) ctrl.SIT_PITCH=atof(getenv("SIT_PITCH"));
  if(getenv("SIT_REAR_FOOT")) ctrl.SIT_REAR_FOOT=atof(getenv("SIT_REAR_FOOT"));
  if(getenv("SIT_REAR_CALF")) ctrl.SIT_REAR_CALF=atof(getenv("SIT_REAR_CALF"));
  if(getenv("SIT_REAR_THIGH")) ctrl.SIT_REAR_THIGH=atof(getenv("SIT_REAR_THIGH"));
  if(getenv("SIT_SLEW")) ctrl.SIT_SLEW=atof(getenv("SIT_SLEW"));
  if(getenv("SIT_CPITCH")) ctrl.SIT_CPITCH=atof(getenv("SIT_CPITCH"));
  if(getenv("SIT_REACH")) ctrl.SIT_REACH=atof(getenv("SIT_REACH"));
  if(getenv("HAUNCH_Z")) ctrl.HAUNCH_Z=atof(getenv("HAUNCH_Z"));            // ★개-앉기(haunch sit) 튜닝: 접힘높이·fold속도·언폴드완료높이
  if(getenv("HAUNCH_FOLD_RATE")) ctrl.HAUNCH_FOLD_RATE=atof(getenv("HAUNCH_FOLD_RATE"));
  if(getenv("HAUNCH_UNFOLD_Z")) ctrl.HAUNCH_UNFOLD_Z=atof(getenv("HAUNCH_UNFOLD_Z"));
  if(getenv("SIT_POSTURE_W")) ctrl.SIT_POSTURE_W=atof(getenv("SIT_POSTURE_W"));  // 개-앉기 자세task 가중(접힘 홀드)
  if(getenv("HAUNCH_PITCH")) ctrl.HAUNCH_PITCH=atof(getenv("HAUNCH_PITCH"));  // 개-앉기 nose-up(q_home 베이킹)
  if(getenv("HAUNCH_THIGH")) q.HAUNCH_THIGH=atof(getenv("HAUNCH_THIGH"));   // 뒷다리 개-앉기 시드(무릎-위 가지)
  if(getenv("HAUNCH_CALF")) q.HAUNCH_CALF=atof(getenv("HAUNCH_CALF"));
  if(getenv("HAUNCH_FOOT")) q.HAUNCH_FOOT=atof(getenv("HAUNCH_FOOT"));
  if(getenv("HAUNCH_HOCK_Z")) q.HAUNCH_HOCK_Z=atof(getenv("HAUNCH_HOCK_Z"));  // hock 지면 목표(발링크 평평)
  if(getenv("FRONT_REACH")) q.FRONT_REACH=atof(getenv("FRONT_REACH"));       // 앞발 전방 배치(↑=앞다리 더 폄)
  if(getenv("HAUNCH_FOOT_LAND")) q.HAUNCH_FOOT_LAND=atof(getenv("HAUNCH_FOOT_LAND"));  // 착지 중 뒷발 각도(닿은 뒤 HAUNCH_FOOT로 굴림)
  if(getenv("SGU_KICK_T")) ctrl.SGU_KICK_T=atof(getenv("SGU_KICK_T"));      // ★앉기→서기 스크립트 기립 튜닝
  if(getenv("SGU_FB_THIGH")) ctrl.SGU_FB_THIGH=atof(getenv("SGU_FB_THIGH"));
  if(getenv("SGU_FB_CALF")) ctrl.SGU_FB_CALF=atof(getenv("SGU_FB_CALF"));
  if(getenv("SGU_SLEW")) ctrl.SGU_SLEW=atof(getenv("SGU_SLEW"));
  if(getenv("SGU_KP")) ctrl.SGU_KP=atof(getenv("SGU_KP"));
  if(getenv("GETUP_TRAJ_KP")) ctrl.GETUP_TRAJ_KP=atof(getenv("GETUP_TRAJ_KP"));   // ★개-앉기 기립 궤적추종 강성(=튕김 힘). ↓=부드럽게
  if(getenv("GETUP_TRAJ_KD")) ctrl.GETUP_TRAJ_KD=atof(getenv("GETUP_TRAJ_KD"));   // ↑=튕김 감쇠
  if(getenv("JUMP_KP")) ctrl.JUMP_KP=atof(getenv("JUMP_KP"));                     // ★점프 추진 강성(=점프 높이). ↓=낮게
  if(getenv("JUMP_CROUCH_Z")) ctrl.JUMP_CROUCH_Z=atof(getenv("JUMP_CROUCH_Z"));   // 웅크림 깊이(깊을수록 스트로크↑)
  if(getenv("JUMP_THRUST_T")) ctrl.JUMP_THRUST_T=atof(getenv("JUMP_THRUST_T"));   // 추진 최대시간(이벤트 없을 때 타임아웃)
  if(getenv("SGU_GATHER_Z")) ctrl.SGU_GATHER_Z=atof(getenv("SGU_GATHER_Z"));
  if(getenv("SGU_DONE_TILT")) ctrl.SGU_DONE_TILT=atof(getenv("SGU_DONE_TILT"));
  if(getenv("SGU_WALKOUT_V")) ctrl.SGU_WALKOUT_V=atof(getenv("SGU_WALKOUT_V"));
  if(getenv("SGU_HANDOFF_Z")) ctrl.SGU_HANDOFF_Z=atof(getenv("SGU_HANDOFF_Z"));
  if(getenv("TROT_WZ")) ctrl.WZ=atof(getenv("TROT_WZ"));   // ★선회 각속도(직접 yaw, 제자리 스핀)
  if(getenv("TROT_STEER")) ctrl.steer=atof(getenv("TROT_STEER"));  // ★자동차식 조향각δ(Ackermann R=축거/tanδ)
  if(getenv("GAIT")) ctrl.set_gait(getenv("GAIT"));        // ★게이트 테스트(trot/walk/gallop)
  if(getenv("TROT_T")) ctrl.gp_T=atof(getenv("TROT_T"));           // ★게이트 주기 override(set_gait 뒤)
  if(getenv("TROT_SWF")) ctrl.gp_SWF=atof(getenv("TROT_SWF"));     // ★swing 비율 override
  if(getenv("TROT_T")||getenv("TROT_SWF")){ ctrl.gp_Tsw=ctrl.gp_T*ctrl.gp_SWF; ctrl.gp_Tst=ctrl.gp_T*(1.0-ctrl.gp_SWF); }  // T_sw/T_st 재계산
  if(getenv("TROT_STEPH")) ctrl.step_h=atof(getenv("TROT_STEPH")); // ★발 높이 override
  if(getenv("RAIBERT_K")) ctrl.raibert_k=atof(getenv("RAIBERT_K"));  // set_gait 프리셋 위에 강제 override
  ctrl.auto_whip = !(getenv("AUTO_WHIP") && !strcmp(getenv("AUTO_WHIP"),"0"));  // 기본ON, AUTO_WHIP=0로 끔
  if(getenv("SWING_W")){ double v=atof(getenv("SWING_W")); ctrl.whip_lo_f=v; ctrl.whip_lo_r=v; }  // whip 목표(고속/수동)
  if(getenv("SWING_W_F")) ctrl.whip_lo_f=atof(getenv("SWING_W_F"));
  if(getenv("SWING_W_R")) ctrl.whip_lo_r=atof(getenv("SWING_W_R"));
  if(getenv("ALIP") && !strcmp(getenv("ALIP"),"0")) ctrl.ALIP=false;
  if(getenv("POS_HOLD") && !strcmp(getenv("POS_HOLD"),"0")) ctrl.POS_HOLD=false;
  mjModel*m=q.m; mjData*d=q.d; double dt=m->opt.timestep;
  if(getenv("DBG")) std::printf("[dbg] nu=%d leg_dof=[%d %d %d %d] standing_z=%.5f com_ref=[%.5f %.5f %.5f]\n",
      q.nu,q.leg_dof[0],q.leg_dof[1],q.leg_dof[2],q.leg_dof[3],d->qpos[2],q.com_ref[0],q.com_ref[1],q.com_ref[2]);

  int falls=0; double max_tilt=0, penF=0, penR=0, pitchSum=0, tauEff=0, calfTau=0, footWmax=0; int pn=0;
  // ★관절(축)별 τ·ω peak/RMS (정착후 t>1.5s) — 기립자세 비교용. JSTAT=1 이면 종료시 관절별 출력
  bool JSTAT=getenv("JSTAT")!=nullptr; int _NU=q.nu;
  std::vector<double> tpk(_NU,0), tsq(_NU,0), wpk(_NU,0), wsq(_NU,0); long jstat_n=0;
  std::vector<std::string> jname; for(int jj=0;jj<m->njnt;jj++) if(m->jnt_type[jj]!=mjJNT_FREE){
    const char* nm=mj_id2name(m,mjOBJ_JOINT,jj); std::string s=nm?nm:""; auto pp=s.find("_joint"); if(pp!=std::string::npos) s.erase(pp); jname.push_back(s); }
  // ★임시 slip 계측: 발이 접촉(dist<1mm)인 동안 anchor 대비 수평 이동 최대치 = slip. 접촉종료 시 누적.
  double f_ax[4]={0},f_ay[4]={0},slip_sum[4]={0},slip_mx[4]={0}; bool f_con[4]={false}; int slip_n[4]={0};
  bool SLIP=getenv("SLIPLOG")!=nullptr;
  double grf_fz[4]={0},grf_fx[4]={0}; int grf_n=0; bool GRF=getenv("GRFLOG")!=nullptr;   // ★발별 수직GRF(앞/뒤 비율)+부호있는 수평력(앞제동/뒤추진)
  auto t0=std::chrono::high_resolution_clock::now();
  double switchT=getenv("SWITCH_T")?atof(getenv("SWITCH_T")):-1;   // ★모드전환 테스트: t>SWITCH_T면 MODE2로(getup 검증)
  bool switched=false;
  for(int step=0; step<STEPS; step++){
    if(switchT>0 && d->time>switchT && getenv("MODE2") && !switched){ ctrl.mode=getenv("MODE2"); switched=true; }  // ★1회성(내부 walk-out 인계 안 덮게)
    ctrl.control(); mj_step(m,d);
    if(SLIP){ for(int i=0;i<4;i++){ bool con=false;
        for(int ci=0;ci<d->ncon;ci++){ const auto&c=d->contact[ci]; if((c.geom1==q.fgid[i]||c.geom2==q.fgid[i])&&c.dist<0.001){con=true;break;} }
        double fx=d->geom_xpos[q.fgid[i]*3], fy=d->geom_xpos[q.fgid[i]*3+1];
        if(con){ if(!f_con[i]){ f_ax[i]=fx; f_ay[i]=fy; slip_mx[i]=0; } slip_mx[i]=std::max(slip_mx[i],std::hypot(fx-f_ax[i],fy-f_ay[i])); }
        else if(f_con[i]&&d->time>1.5){ slip_sum[i]+=slip_mx[i]; slip_n[i]++; }
        f_con[i]=con; } }
    if(GRF && d->time>1.5){ for(int ci=0;ci<d->ncon;ci++){ const auto&c=d->contact[ci];
        double f6[6]; mj_contactForce(m,d,ci,f6); double R[9]; for(int r=0;r<9;r++) R[r]=c.frame[r];
        double fzw=R[2]*f6[0]+R[5]*f6[1]+R[8]*f6[2];   // 접촉프레임→world z성분(각 축의 z성분·힘 내적)
        double fxw=R[0]*f6[0]+R[3]*f6[1]+R[6]*f6[2];   // world x성분(부호: +전진방향)
        for(int fi=0;fi<4;fi++) if(c.geom1==q.fgid[fi]||c.geom2==q.fgid[fi]){ grf_fz[fi]+=std::abs(fzw); grf_fx[fi]+=fxw; } }
      grf_n++; }
    double td=ctrl.tiltdeg(); max_tilt=std::max(max_tilt,td);
    if(td>50||d->qpos[2]<0.2) falls++;
    if(d->time>1.5){ // 정착후 앞/뒤 발침투 평균(진단): 스텝별 최소침투를 누적
      double pf=0,pr=0;
      for(int ci=0;ci<d->ncon;ci++){ const auto&c=d->contact[ci];
        for(int fi=0;fi<4;fi++) if(c.geom1==q.fgid[fi]||c.geom2==q.fgid[fi]){
          if(fi>=2) pf=std::min(pf,c.dist); else pr=std::min(pr,c.dist); } }
      penF+=pf; penR+=pr;
      double R[9]; mju_quat2Mat(R,&d->qpos[3]); pitchSum+=std::asin(std::max(-1.0,std::min(1.0,-R[6])))*180/M_PI; pn++;
      for(int j=0;j<q.nu;j++) tauEff+=std::abs(d->ctrl[j]);   // 총 토크 effort(에너지 대리)
      for(int i=0;i<4;i++){ int cj=q.legqv[i][2]-6; if(cj>=0&&cj<q.nu) calfTau+=std::abs(d->ctrl[cj]); }   // calf(whip 관절) 토크
      for(int i=0;i<4;i++) if(q.leg_dof[i]==4) footWmax=std::max(footWmax,std::abs(d->qvel[q.legqv[i][3]]));
      if(JSTAT){ for(int j=0;j<_NU;j++){ double t=d->ctrl[j], w=d->qvel[6+j];   // 관절j: 토크=ctrl[j], 각속도=qvel[6+j](1:1)
        tpk[j]=std::max(tpk[j],std::abs(t)); tsq[j]+=t*t; wpk[j]=std::max(wpk[j],std::abs(w)); wsq[j]+=w*w; } jstat_n++; } }  // ★발목 최대각속도(반사관성 효과 확인)
    if(step%250==0){ double*qq=&d->qpos[3];
      double yaw=std::atan2(2*(qq[0]*qq[3]+qq[1]*qq[2]),1-2*(qq[2]*qq[2]+qq[3]*qq[3]))*180/M_PI;
      std::printf("[hl] s=%d t=%.2f z=%.3f x=%+.3f y=%+.3f yaw=%+.0f° tilt=%.1f falls=%d\n",
                  step,d->time,d->qpos[2],d->qpos[0],d->qpos[1],yaw,td,falls); }
  }
  double wall=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t0).count();
  std::printf("\n=== 종료: STEPS=%d(%.1fs) x=%+.3f z=%.3f max_tilt=%.1f° falls=%d | ★침투평균 앞=%.1fmm 뒤=%.1fmm pitch=%.1f° | %.0f steps/s ===\n",
              STEPS,STEPS*dt,d->qpos[0],d->qpos[2],max_tilt,falls,pn?penF/pn*1000:0,pn?penR/pn*1000:0,pn?pitchSum/pn:0,STEPS/wall);
  std::printf("    토크effort 평균Σ|τ|=%.1fNm  calf평균Σ|τ|=%.2fNm (whip 관절)  발목최대ω=%.1f rad/s\n", pn?tauEff/pn:0, pn?calfTau/pn:0, footWmax);
  if(JSTAT && jstat_n>0){ std::printf("  [JSTAT] 관절별 τ·ω (정착후 %ld스텝)\n", jstat_n);
    std::printf("    %-9s %8s %8s %8s %8s\n","joint","τpeak","τrms","ωpeak","ωrms");
    for(int j=0;j<_NU;j++) std::printf("    %-9s %8.2f %8.2f %8.2f %8.2f\n",
      (j<(int)jname.size()?jname[j].c_str():""), tpk[j], std::sqrt(tsq[j]/jstat_n), wpk[j], std::sqrt(wsq[j]/jstat_n)); }
  std::printf("    ★실제 뒷다리(HL) thigh=%.3f calf=%.3f foot=%.3f | 무릎z=%.3f hockZ=%.3f toeZ=%.3f\n",
      d->qpos[q.legqp[0][1]], d->qpos[q.legqp[0][2]], d->qpos[q.legqp[0][3]], d->xpos[mj_name2id(m,mjOBJ_BODY,"HL_calf_link")*3+2],
      d->xpos[q.rear_hock_bid[0]*3+2], q.foot_point(0)[2]);
  { double fz=d->xpos[q.hip_bid[2]*3+2], rz=d->xpos[q.hip_bid[0]*3+2];   // 앞힙 z vs 뒤힙 z
    std::printf("    ★상체방향: 앞힙z=%.3f 뒤힙z=%.3f → %s\n", fz, rz, fz>rz?"앞이 위=nose-up(엉덩이 주저앉기 ✓)":"뒤가 위=nose-down(✗ 반대)");
    std::printf("    ★앞다리(FL) thigh=%.3f calf=%.3f 무릎z=%.3f (calf≈0=곧게 폄)\n",
      d->qpos[q.legqp[2][1]], d->qpos[q.legqp[2][2]], d->xpos[mj_name2id(m,mjOBJ_BODY,"FL_calf_link")*3+2]); }
  if(GRF && grf_n){ double fr=(grf_fz[0]+grf_fz[1])/grf_n, ff=(grf_fz[2]+grf_fz[3])/grf_n; double tot=fr+ff;
    std::printf("    ★수직GRF 평균[N]: HL=%.0f HR=%.0f FL=%.0f FR=%.0f | 뒤=%.0f(%.0f%%) 앞=%.0f(%.0f%%) 뒤:앞=%.2f\n",
      grf_fz[0]/grf_n,grf_fz[1]/grf_n,grf_fz[2]/grf_n,grf_fz[3]/grf_n, fr,tot>0?fr/tot*100:0, ff,tot>0?ff/tot*100:0, ff>0?fr/ff:0);
    double fxr=(grf_fx[0]+grf_fx[1])/grf_n, fxf=(grf_fx[2]+grf_fx[3])/grf_n;
    std::printf("    ★수평GRF 평균[N](+전진): 뒤=%+.1f 앞=%+.1f 합=%+.1f (뒤+=추진 / 앞−=제동)\n", fxr, fxf, fxr+fxf); }
  if(SLIP){ std::printf("    ★발 slip(접촉중 수평이동 평균, mm): ");
    for(int i=0;i<4;i++) std::printf("%s=%.1f ", q.legs[i], slip_n[i]?slip_sum[i]/slip_n[i]*1000:0);
    std::printf(" | 뒤평균=%.1f 앞평균=%.1f mm\n",
      ((slip_n[0]?slip_sum[0]/slip_n[0]:0)+(slip_n[1]?slip_sum[1]/slip_n[1]:0))/2*1000,
      ((slip_n[2]?slip_sum[2]/slip_n[2]:0)+(slip_n[3]?slip_sum[3]/slip_n[3]:0))/2*1000); }
  if(getenv("DUMP_QPOS")){ FILE*f=fopen(getenv("DUMP_QPOS"),"w");   // ★정착 qpos 덤프(trajopt x0/xf용)
    for(int i=0;i<m->nq;i++) fprintf(f,"%.8f ",d->qpos[i]); fclose(f);
    std::printf("[dump] qpos → %s (nq=%d)\n", getenv("DUMP_QPOS"), m->nq); }
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
