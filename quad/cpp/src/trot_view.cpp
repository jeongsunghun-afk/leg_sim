// trot_view — C++ closed-loop trot을 MuJoCo GLFW 뷰어로 렌더. trot_sim과 동일 제어(TrotCtrl).
//   마우스: 좌드래그=회전 우드래그=이동 휠=줌.  키보드: ↑↓=전진속도 ←→=선회 space=정지 backspace=리셋.
#include "trot_controller.hpp"
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// 평면 JSON에서 "key": 숫자 추출(GUI cmd 파일용, 경량)
static double json_get(const std::string& s,const char* key,double def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; return atof(s.c_str()+p+1);
}
// "key": "문자열" 추출
static std::string json_str(const std::string& s,const char* key,const std::string& def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; auto q1=s.find('"',p+1);
  if(q1==std::string::npos) return def; auto q2=s.find('"',q1+1);
  if(q2==std::string::npos) return def; return s.substr(q1+1,q2-q1-1);
}

// ★상태 발행(STATE_PUB) — Python publish_state와 동일 스키마로 GUI 모니터(IMU/Actuator) 채움.
//   비-free 관절 전체(17-DOF: HL4·HR4·waist1·FL4·FR4 — 허리 포함) q/dq/tau + base_z·rpy·gyro.
static void publish_state(mjModel* m, mjData* d, TrotCtrl& ctrl, const std::string& path){
  int nu=m->nu;
  static std::vector<std::string> jn; static bool init=false;
  if(!init){ init=true;
    for(int j=0;j<m->njnt;j++) if(m->jnt_type[j]!=mjJNT_FREE){
      const char* nm=mj_id2name(m,mjOBJ_JOINT,j); std::string s=nm?nm:"";
      auto p=s.find("_joint"); if(p!=std::string::npos) s.erase(p);
      jn.push_back(s);
    } }
  double R[9]; mju_quat2Mat(R,d->qpos+3);   // world←body 회전
  double roll=atan2(R[7],R[8]), pitch=asin(std::max(-1.0,std::min(1.0,-R[6]))), yaw=atan2(R[3],R[0]);
  double vf=d->qvel[0]*cos(yaw)+d->qvel[1]*sin(yaw);
  static double vf_filt=0; vf_filt=0.97*vf_filt+0.03*vf;
  std::ostringstream o; o.precision(6); o<<std::fixed;
  o<<"{\"mode\":\""<<ctrl.mode<<"\",\"base_z\":"<<d->qpos[2]<<",\"t\":"<<d->time;
  o<<",\"rpy\":["<<roll*57.29578<<","<<pitch*57.29578<<","<<yaw*57.29578<<"]";
  o<<",\"gyro\":["<<d->qvel[3]<<","<<d->qvel[4]<<","<<d->qvel[5]<<"]";
  o<<",\"names\":["; for(int i=0;i<nu;i++){ o<<(i?",":"")<<"\""<<jn[i]<<"\""; }
  o<<"],\"q\":["; for(int i=0;i<nu;i++){ o<<(i?",":"")<<d->qpos[7+i]; }
  o<<"],\"dq\":["; for(int i=0;i<nu;i++){ o<<(i?",":"")<<d->qvel[6+i]; }
  o<<"],\"tau\":["; for(int i=0;i<nu;i++){ o<<(i?",":"")<<d->ctrl[i]; }
  o<<"],\"v_cmd\":"<<ctrl.V<<",\"v_act\":"<<vf_filt<<"}";
  std::string tmp=path+".tmp"; std::ofstream f(tmp); f<<o.str(); f.close();
  std::rename(tmp.c_str(),path.c_str());   // 원자적 교체
}

static mjvCamera cam; static mjvOption opt; static mjvScene scn; static mjrContext con;
static bool btnL=false, btnR=false, btnM=false; static double lastx=0, lasty=0;
static TrotCtrl* gC=nullptr;

// (키보드 제어 삭제 — GUI(teleop_gui_17dof)로 제어. 마우스 카메라만 유지)
static void mouse_btn(GLFWwindow* w,int b,int act,int mods){
  btnL=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS;
  btnR=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS;
  btnM=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS;
  glfwGetCursorPos(w,&lastx,&lasty);
}
static void mouse_move(GLFWwindow* w,double xp,double yp){
  if(!btnL&&!btnR&&!btnM){ lastx=xp; lasty=yp; return; }
  double dx=xp-lastx, dy=yp-lasty; lastx=xp; lasty=yp;
  int W,H; glfwGetWindowSize(w,&W,&H);
  mjtMouse act = btnR? mjMOUSE_MOVE_H : (btnL? mjMOUSE_ROTATE_H : mjMOUSE_ZOOM);
  mjv_moveCamera(gC?gC->q.m:nullptr,act,dx/H,dy/H,&scn,&cam);
}
static void scroll(GLFWwindow* w,double dx,double dy){ mjv_moveCamera(gC?gC->q.m:nullptr,mjMOUSE_ZOOM,0,-0.05*dy,&scn,&cam); }

int main(int argc,char**argv){
  const char* path=argc>1?argv[1]:"../mjcf/quad_real_sphere.mjcf";
  QuadControl q; q.load(path); apply_env_gains(q);
  q.crouch_home(); q.build_qhome_lut(); q.setup_mpc();   // ★q_home LUT 시작시 빌드(RT-safe: 높이변경 IK를 루프서 제거)
  TrotCtrl ctrl(q); gC=&ctrl;
  ctrl.load_jump("/tmp/jump_traj.txt");   // ★점프 궤적 시작 시 preload(fallback용). 없으면 jump_N=0
#ifdef HAVE_JUMP_SOLVER
  if(!getenv("NO_JUMP_WARMUP")){ std::printf("[trot_view] 점프 OCP 예열(모델·IK 캐시)…\n"); std::fflush(stdout); ctrl.warmup_jump(); std::printf("[trot_view] 예열 완료 — 점프 live-solve 활성\n"); }
#endif
  ctrl.mode = getenv("MODE") ? getenv("MODE") : "stand_up";   // ★뷰어는 Ready(서기)로 시작 — GUI 초기값(stand_up)과 일치(시작 보행→멈춤 방지). 헤드리스 trot_sim은 기본 move 유지
  if(getenv("TROT_V")) ctrl.V=atof(getenv("TROT_V"));
  if(getenv("WAIST_STEER")) ctrl.waist_steer=atof(getenv("WAIST_STEER"));   // 허리 lean 게인(기본0.4)
  if(getenv("TROT_STEER")) ctrl.steer=atof(getenv("TROT_STEER"));           // ★자동차식 조향각
  if(getenv("PERCEPTIVE")) ctrl.perceptive=atoi(getenv("PERCEPTIVE"))!=0;    // ★지형인지(mj_ray 착지높이) on/off
  if(getenv("PCV_CLR")) ctrl.PCV_CLR=atof(getenv("PCV_CLR"));
  // ★눕기(ground) 자세 조각 — GUI 슬라이더가 안 될 때 env로 직접 확인/튜닝(GUI cmd와 동일 파라미터)
  if(getenv("GROUND_LIE_Z"))     ctrl.GROUND_LIE_Z=atof(getenv("GROUND_LIE_Z"));
  if(getenv("GROUND_REAR_FOOT")) ctrl.GROUND_REAR_FOOT=atof(getenv("GROUND_REAR_FOOT"));
  if(getenv("GROUND_FRONT_THIGH")) ctrl.GROUND_FRONT_THIGH=atof(getenv("GROUND_FRONT_THIGH"));
  if(getenv("GROUND_FRONT_CALF"))  ctrl.GROUND_FRONT_CALF=atof(getenv("GROUND_FRONT_CALF"));
  if(getenv("GETUP_TRAJ_KP")) ctrl.GETUP_TRAJ_KP=atof(getenv("GETUP_TRAJ_KP"));   // ★개-앉기 기립 궤적추종 강성(=앉기→서기 튕김 힘). ↓=부드럽게
  if(getenv("GETUP_TRAJ_KD")) ctrl.GETUP_TRAJ_KD=atof(getenv("GETUP_TRAJ_KD"));   // ↑=튕김 감쇠
  if(getenv("SGU_KP")) ctrl.SGU_KP=atof(getenv("SGU_KP"));                        // ★크라우치-앉기 기립 강성(뷰어에도 추가)
  if(getenv("JUMP_KP")) ctrl.JUMP_KP=atof(getenv("JUMP_KP"));                     // ★점프 추진 강성(=높이). ↓=낮게
  if(getenv("JUMP_CROUCH_Z")) ctrl.JUMP_CROUCH_Z=atof(getenv("JUMP_CROUCH_Z"));
  if(getenv("JUMP_THRUST_T")) ctrl.JUMP_THRUST_T=atof(getenv("JUMP_THRUST_T"));
  if(getenv("HAUNCH_Z")) ctrl.HAUNCH_Z=atof(getenv("HAUNCH_Z"));            // ★개-앉기(haunch sit) 튜닝
  if(getenv("HAUNCH_FOLD_RATE")) ctrl.HAUNCH_FOLD_RATE=atof(getenv("HAUNCH_FOLD_RATE"));
  if(getenv("HAUNCH_UNFOLD_Z")) ctrl.HAUNCH_UNFOLD_Z=atof(getenv("HAUNCH_UNFOLD_Z"));
  if(getenv("SIT_POSTURE_W")) ctrl.SIT_POSTURE_W=atof(getenv("SIT_POSTURE_W"));
  if(getenv("HAUNCH_THIGH")) q.HAUNCH_THIGH=atof(getenv("HAUNCH_THIGH"));
  if(getenv("HAUNCH_CALF")) q.HAUNCH_CALF=atof(getenv("HAUNCH_CALF"));
  if(getenv("HAUNCH_FOOT")) q.HAUNCH_FOOT=atof(getenv("HAUNCH_FOOT"));
  if(getenv("HAUNCH_HOCK_Z")) q.HAUNCH_HOCK_Z=atof(getenv("HAUNCH_HOCK_Z"));
  if(getenv("SIT_CPITCH")) ctrl.SIT_CPITCH=atof(getenv("SIT_CPITCH"));      // 앉기 nose-up 목표(뷰어 튜닝)
  mjModel*m=q.m; mjData*d=q.d;

  if(!glfwInit()){ std::fprintf(stderr,"glfw init 실패\n"); return 1; }
  GLFWwindow* win=glfwCreateWindow(1280,900,"17-DOF C++ trot (quad_mpc_wbic_17dof)",NULL,NULL);
  if(!win){ std::fprintf(stderr,"창 생성 실패(DISPLAY?)\n"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win); glfwSwapInterval(1);
  mjv_defaultCamera(&cam); mjv_defaultOption(&opt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
  mjv_makeScene(m,&scn,2000); mjr_makeContext(m,&con,mjFONTSCALE_150);
  cam.distance=2.2; cam.elevation=-20; cam.azimuth=135; cam.lookat[2]=0.35;
  opt.flags[mjVIS_CONTACTFORCE]=1;
  glfwSetMouseButtonCallback(win,mouse_btn);   // 마우스 카메라만(키보드 제어 삭제, GUI로 조작)
  glfwSetCursorPosCallback(win,mouse_move); glfwSetScrollCallback(win,scroll);

  double RATE = getenv("RATE")?atof(getenv("RATE")):1.0;   // 재생 배속(env, 1=실시간·0.5=슬로모)
  const char* CMDFILE = getenv("CMDFILE");                 // ★GUI 연동: /tmp/quad_cmd.json 폴링(v/vy/w)
  std::string STATE_PUB = getenv("STATE_PUB")?getenv("STATE_PUB"):"/tmp/quad_state.json";  // ★상태 발행(GUI 모니터)
  int falls=0; double max_tilt=0; long frame=0; bool fallen=false; long reset_seen=-1, jump_seen=-1;
  auto wall0=std::chrono::steady_clock::now(); double sim0=d->time;
  while(!glfwWindowShouldClose(win)){
    // ★GUI 명령 폴링(~20Hz): teleop_gui가 쓴 v/vy/w 반영
    if(CMDFILE && (frame++ %3==0)){
      std::ifstream f(CMDFILE);
      if(f){ std::stringstream ss; ss<<f.rdbuf(); std::string c=ss.str();
        ctrl.V=json_get(c,"v",ctrl.V); ctrl.VY=json_get(c,"vy",ctrl.VY); ctrl.WZ=json_get(c,"w",ctrl.WZ);
        { static double sh_seen=-999; double sh=json_get(c,"step_h",sh_seen); if(sh!=sh_seen){ ctrl.step_h=sh; sh_seen=sh; } }  // ★step height 슬라이더=엣지 override(게이트별 프리셋 walk0.05/trot0.10/run0.08과 공존)
        { static double rk_seen=-999; double rk=json_get(c,"raibert_k",rk_seen); if(rk!=rk_seen){ ctrl.raibert_k=rk; rk_seen=rk; } }  // ★reach 슬라이더=엣지 override(게이트별 프리셋 walk0.5/trot0.8과 공존)
        double sw=json_get(c,"swing_w",-1); if(sw>=0){ ctrl.whip_lo_r=sw; ctrl.whip_lo_f=sw; }  // 통합(하위호환)
        double swf=json_get(c,"swing_w_f",-1); if(swf>=0) ctrl.whip_lo_f=swf;   // ★앞다리 whip 목표(고속target·auto off시 상수)
        double swr=json_get(c,"swing_w_r",-1); if(swr>=0) ctrl.whip_lo_r=swr;   // ★뒷다리 whip 목표
        ctrl.auto_whip = json_get(c,"auto_whip",ctrl.auto_whip?1:0) > 0.5;      // ★속도연동 whip 토글(on=속도스케일, off=슬라이더 상수)
        ctrl.POS_HOLD = json_get(c,"pos_hold",ctrl.POS_HOLD?1:0) > 0.5;         // ★정지 위치홀드(드리프트 보정) 토글
        ctrl.steer = json_get(c,"steer",ctrl.steer);            // ★허리 핸들=자동차식 조향각(GUI 슬라이더). Ackermann 반경으로 선회+허리 lean
        ctrl.GROUND_LIE_Z    = json_get(c,"g_lie_z",   ctrl.GROUND_LIE_Z);      // ★눕기 자세 실시간 조각(GUI 슬라이더)
        ctrl.GROUND_REAR_FOOT= json_get(c,"g_rear_foot",ctrl.GROUND_REAR_FOOT);
        ctrl.GROUND_FRONT_THIGH=json_get(c,"g_front_thigh",ctrl.GROUND_FRONT_THIGH);
        ctrl.GROUND_FRONT_CALF =json_get(c,"g_front_calf", ctrl.GROUND_FRONT_CALF);
        { std::string mc=json_str(c,"mode","move");             // move/stand_up(서기)/stand_down(눕기)/off
          if(ctrl.mode!="jump") ctrl.mode=mc; }                 // ★점프 중엔 GUI mode 무시(안 그러면 매 폴링 덮어써 점프 즉시 취소). 컨트롤러가 완료 후 stand_up 자가전환
        ctrl.set_gait(json_str(c,"gait","trot"));               // trot/walk 게이트 토글
        ctrl.body_h = json_get(c,"body_h",ctrl.body_h);         // 서기 높이 슬라이더
        double rt=json_get(c,"rate",RATE); if(rt>0) RATE=rt;
        long rseq=(long)json_get(c,"reset_seq",reset_seen);     // ★RESET 버튼(상승엣지): mj_resetData+crouch_home+상태초기화
        if(reset_seen<0) reset_seen=rseq;                       //   첫폴링=동기화(시작리셋 방지)
        else if(rseq>reset_seen){ reset_seen=rseq; mj_resetData(m,d); q.crouch_home(); ctrl.reset(); falls=0; fallen=false;
          wall0=std::chrono::steady_clock::now(); sim0=d->time; }
        long jseq=(long)json_get(c,"jump_seq",jump_seen);       // ★Jump 버튼(상승엣지): 스크립트 점프 발동(mode=jump)
        if(jump_seen<0) jump_seen=jseq; else if(jseq>jump_seen){ jump_seen=jseq; ctrl.mode="jump"; } } }
    // ★벽시계 기준 실시간 페이싱: sim_time이 wall_time×RATE 따라가도록(모니터 refresh 무관)
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();
    double target=sim0+wall*RATE; int guard=0;
    while(d->time < target && guard++ < 60){    // 따라잡기(최대 60스텝/프레임=버스트 상한↓, 점프 히치 후 몰아치기 완화)
      ctrl.control(); mj_step(m,d);
      double td=ctrl.tiltdeg(); max_tilt=std::max(max_tilt,td);
      // ★낙상 시 자동재시작 안 함(그대로 쓰러진 채 유지 → RESET 버튼으로 복구). 낙상은 엣지로만 카운트
      bool low=(td>50||d->qpos[2]<0.2); if(low && !fallen) falls++; fallen=low;
    }
    // ★페이싱 재동기: 뒤처짐(히치 후 백로그)·앞섬 양쪽 모두 0.2s 초과면 시계 리셋 → 백로그 버림(점프 후 fast-forward 렉 방지)
    if(target-d->time > 0.2 || d->time-sim0 > wall*RATE+0.5){ wall0=std::chrono::steady_clock::now(); sim0=d->time; }
    if(!STATE_PUB.empty() && (frame%3==0)) publish_state(m,d,ctrl,STATE_PUB);   // ★GUI 모니터(IMU/Actuator 17-DOF) 발행

    mjrRect vp={0,0,0,0}; glfwGetFramebufferSize(win,&vp.width,&vp.height);
    cam.lookat[0]=d->qpos[0]; cam.lookat[1]=d->qpos[1];   // 로봇 추적
    mjv_updateScene(m,d,&opt,NULL,&cam,mjCAT_ALL,&scn);
    mjr_render(vp,&scn,&con);
    char hud[256];
    std::snprintf(hud,256,"cmd V=%.2f  WZ=%.2f m/s\nactual z=%.3f  tilt=%.1f deg\nx=%+.2f  falls=%d\nmouse: rotate/zoom  |  control via GUI",
                  ctrl.V,ctrl.WZ,d->qpos[2],ctrl.tiltdeg(),d->qpos[0],falls);
    mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,vp,"17-DOF C++ trot (GUI controlled)",hud,&con);
    glfwSwapBuffers(win); glfwPollEvents();
  }
  mjv_freeScene(&scn); mjr_freeContext(&con); glfwTerminate();
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
