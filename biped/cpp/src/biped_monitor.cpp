// biped_monitor — ★실기 **모니터 전용** 뷰어. 엔코더 실측각을 3D 로 그리기만 한다.
//
//   biped_view(sim)와 역할이 정반대다:
//     biped_view   : GUI 명령을 받아 **제어 + 물리(mj_step)** 를 돌리고 상태를 발행 → 폐루프
//     biped_monitor: **아무것도 제어하지 않는다.** 엔코더 값을 읽어 mj_forward 로 자세만 갱신
//
//   실기 배선:
//     GUI ──cmd.json──> [제어기 app/biped_emb.py] ──SHM──> 모터
//                                  │ state.json (엔코더 실측)
//                                  └──────────────> [biped_monitor]  ← 표시 전용
//
//   ⚠ 이 프로그램은 **모터에 아무것도 쓰지 않는다.** SHM 도 안 건드린다(state.json 만 읽음).
//     그래서 제어기와 동시에 띄워도 "writer 는 하나만" 규칙에 걸리지 않는다.
//   ⚠ mj_step 을 부르지 않는다. 물리를 돌리면 실제 로봇과 무관한 자세로 발산한다.
//     mj_forward 만 불러 **주어진 관절각의 기구학**만 계산한다.
//
//   실행: STATE=/tmp/biped_state.json CONFIG=../emb/config/biped_emb.yaml \
//         ./biped_monitor ../biped_from_quad.mjcf
#include "deploy_hw.hpp"
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>

using namespace bipedhw;

// state.json 의 `"q_leg_deg":[...]` 배열을 뽑는다.
static bool json_array(const std::string& s, const char* key, std::vector<double>& out){
  std::string k = std::string("\"")+key+"\"";
  auto p = s.find(k); if(p==std::string::npos) return false;
  p = s.find('[', p);  if(p==std::string::npos) return false;
  auto e = s.find(']', p); if(e==std::string::npos) return false;
  out.clear();
  std::stringstream ss(s.substr(p+1, e-p-1)); std::string t;
  while(std::getline(ss,t,',')){ try{ out.push_back(std::stod(t)); }catch(...){} }
  return !out.empty();
}
static std::string json_str(const std::string& s,const char* key,const std::string& def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; auto q1=s.find('"',p+1);
  if(q1==std::string::npos) return def; auto q2=s.find('"',q1+1);
  if(q2==std::string::npos) return def; return s.substr(q1+1,q2-q1-1);
}
static double json_num(const std::string& s,const char* key,double def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; return atof(s.c_str()+p+1);
}

static mjvCamera cam; static mjvOption opt; static mjvScene scn; static mjrContext con;
static bool btnL=false,btnR=false,btnM=false; static double lastx=0,lasty=0; static mjModel* gM=nullptr;
static void mouse_btn(GLFWwindow* w,int,int,int){ btnL=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS;
  btnR=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS; btnM=glfwGetMouseButton(w,GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS;
  glfwGetCursorPos(w,&lastx,&lasty); }
static void mouse_move(GLFWwindow* w,double xp,double yp){ if(!btnL&&!btnR&&!btnM){ lastx=xp; lasty=yp; return; }
  double dx=xp-lastx,dy=yp-lasty; lastx=xp; lasty=yp; int W,H; glfwGetWindowSize(w,&W,&H);
  mjtMouse a=btnR?mjMOUSE_MOVE_H:(btnL?mjMOUSE_ROTATE_H:mjMOUSE_ZOOM); mjv_moveCamera(gM,a,dx/H,dy/H,&scn,&cam); }
static void scroll(GLFWwindow* w,double,double dy){ mjv_moveCamera(gM,mjMOUSE_ZOOM,0,-0.05*dy,&scn,&cam); }

static double now_s(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }

int main(int argc,char**argv){
  const char* path = argc>1 ? argv[1] : "../biped_from_quad.mjcf";
  std::string STATE = getenv("STATE")? getenv("STATE") : "/tmp/biped_state.json";
  std::string CFGP  = getenv("CONFIG")? getenv("CONFIG") : "../emb/config/biped_emb.yaml";

  EmbCfg cfg; std::string cerr;
  if(!load_cfg(CFGP, cfg, cerr)){ std::fprintf(stderr,"✗ %s\n", cerr.c_str()); return 1; }
  JointMap jm(cfg);

  char err[1000]={0}; mjModel* m=mj_loadXML(path,nullptr,err,1000);
  if(!m){ std::fprintf(stderr,"모델 로드 실패: %s\n",err); return 1; }
  mjData* d=mj_makeData(m); gM=m;
  const int NJ = m->nq-7;
  if(NJ != jm.n_leg){ std::fprintf(stderr,"✗ 모델 관절수(%d) ≠ config joints(%d)\n",NJ,jm.n_leg);
    mj_deleteData(d); mj_deleteModel(m); return 1; }

  // 베이스는 고정한다 — IMU 가 없어 자세를 모르고, 있어도 위치는 추정 문제다.
  //   여기 목적은 **관절각 확인**이지 위치 추적이 아니다.
  d->qpos[0]=0; d->qpos[1]=0; d->qpos[2]= getenv("BASE_Z")? atof(getenv("BASE_Z")) : 0.55;
  d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;

  if(!glfwInit()){ std::fprintf(stderr,"glfw init 실패\n"); return 1; }
  GLFWwindow* win=glfwCreateWindow(1100,820,"biped HW monitor (encoder display only)",NULL,NULL);
  if(!win){ std::fprintf(stderr,"창 생성 실패(DISPLAY? 모니터 절전?)\n"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win); glfwSwapInterval(1);
  mjv_defaultCamera(&cam); mjv_defaultOption(&opt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
  mjv_makeScene(m,&scn,2000); mjr_makeContext(m,&con,mjFONTSCALE_150);
  cam.distance=1.8; cam.elevation=-15; cam.azimuth=135; cam.lookat[2]=0.35;

  glfwSetMouseButtonCallback(win,mouse_btn); glfwSetCursorPosCallback(win,mouse_move); glfwSetScrollCallback(win,scroll);

  std::vector<double> q_ch(jm.n_leg,0.0);   // ★모델각[deg] (이름은 유지, 의미는 모델각)
  std::string mode="-", backend="-", raw_prev;
  // ★loop_hz 는 **제어기(biped_emb)의 제어루프 주파수**다 — 뷰어 렌더링 속도도,
  //   데이터 수신 주기도 아니다. HUD 라벨이 `loop=` 라 오해를 샀다 → `ctrl=` 로 바꾸고
  //   지터(주기 p95 / 공칭)를 함께 띄운다. 평균만 보면 지터가 안 보인다.
  double stale_since = now_s(), tilt=0, loop_hz=0, dt_p95=0, dt_nom=0;
  int n_ok=0, n_absent=0;

  while(!glfwWindowShouldClose(win)){
    // ── 엔코더 상태 읽기 (읽기 전용. 모터엔 아무것도 쓰지 않는다) ──
    { std::ifstream f(STATE);
      if(f){ std::stringstream ss; ss<<f.rdbuf(); std::string sj=ss.str();
        if(sj != raw_prev){ raw_prev = sj; stale_since = now_s(); }
        std::vector<double> v;
        if(json_array(sj,"q_leg_deg",v) && (int)v.size()>=jm.n_leg) q_ch = v;
        mode    = json_str(sj,"mode",mode);
        backend = json_str(sj,"backend",backend);
        tilt    = json_num(sj,"tilt_deg",tilt);
        loop_hz = json_num(sj,"loop_hz",loop_hz);
        dt_p95  = json_num(sj,"dt_ms_p95",dt_p95);
        dt_nom  = json_num(sj,"dt_ms_nom",dt_nom);
        n_ok     = (int)json_num(sj,"n_ok",n_ok);
        n_absent = (int)json_num(sj,"n_absent",n_absent);
      } }

    // ── 모델각(deg) → rad → 모델 주입 ──
    //   ★★2026-08-10 단위 통일: state.json 의 q_leg_deg 는 이제 **모델각**이다
    //     (제어기 hw_interface.q_leg_deg 가 sign·offset·gear_k 를 이미 적용해 발행).
    //     ★뷰어는 **재변환하지 않는다** — D2R 만 곱해 qpos 에 넣는다. 여기서 또 곱하면 이중적용.
    //     여기서 sign/offset 을 **다시 적용하면 이중 변환**이 되어 sign=−1 축이
    //     원위치로 돌아가고 감속비 보정도 두 번 걸린다. 단위 환산만 한다.
    for(int i=0;i<jm.n_leg;i++)
      d->qpos[7+i] = q_ch[i] * JointMap::D2R;
    d->qvel[0]=0; for(int i=0;i<m->nv;i++) d->qvel[i]=0;
    mj_forward(m,d);          // ★mj_step 아님 — 물리를 돌리지 않는다(기구학만)

    mjrRect vp={0,0,0,0}; glfwGetFramebufferSize(win,&vp.width,&vp.height);
    mjv_updateScene(m,d,&opt,NULL,&cam,mjCAT_ALL,&scn);
    mjr_render(vp,&scn,&con);

    // ★HUD 는 **ASCII 만** 쓴다. MuJoCo mjr_overlay 는 내장 비트맵 폰트라
    //   한글/유니코드를 못 그린다(전부 깨져서 나온다). 2026-08-07 실기에서 확인.
    { double stale_ms = (now_s()-stale_since)*1e3;
      char hud[700];
      int n = std::snprintf(hud,sizeof hud,
        "mode=%s  backend=%s  ctrl=%.0fHz(dt p95 %.2f/%.2fms)  tilt=%.1f deg\n"
        "axes ok=%d absent=%d   state age=%.0f ms%s\n"
        "---- joint angle (model coord) [deg] ----\n", mode.c_str(), backend.c_str(), loop_hz, dt_p95, dt_nom, tilt,
        n_ok, n_absent, stale_ms, stale_ms>1000 ? "  *** STALE ***" : "");
      for(int i=0;i<jm.n_leg && n<(int)sizeof(hud)-40;i++)
        n += std::snprintf(hud+n,sizeof(hud)-n,"%-9s %+7.2f%s",
             cfg.joints[i].name.c_str(), q_ch[i], (i%2==1)?"\n":"   ");
      mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,vp,
                  "biped HW MONITOR - display only (no control, no physics)",hud,&con);
      mjr_overlay(mjFONT_NORMAL,mjGRID_BOTTOMLEFT,vp,"",
        "This window does NOT control the robot. Use the GUI (teleop_gui_biped).\n"
        "Base is fixed (no IMU) - joint angles are the measured values.\n"
        "mouse: L-drag rotate | R-drag pan | wheel zoom", &con);
    }
    glfwSwapBuffers(win); glfwPollEvents();
  }
  mjv_freeScene(&scn); mjr_freeContext(&con); glfwTerminate();
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
