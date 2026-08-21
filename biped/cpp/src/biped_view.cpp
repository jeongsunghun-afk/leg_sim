// biped_view — C++ biped 폐루프를 MuJoCo GLFW 뷰어로 렌더 + GUI(JSON) 연동.
//   마우스: 좌드래그=회전 우드래그=이동 휠=줌.  GUI(teleop_gui_biped)가 v/vy/w/body_h/mode 조종.
//   실행: CMDFILE=/tmp/biped_cmd.json ./biped_view ../biped_from_quad.mjcf
#include "biped_control.hpp"
#include "deploy_loop.hpp"
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <functional>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double json_get(const std::string& s,const char* key,double def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; return atof(s.c_str()+p+1);
}
static std::string json_str(const std::string& s,const char* key,const std::string& def){
  std::string k=std::string("\"")+key+"\""; auto p=s.find(k);
  if(p==std::string::npos) return def; p=s.find(':',p);
  if(p==std::string::npos) return def; auto q1=s.find('"',p+1);
  if(q1==std::string::npos) return def; auto q2=s.find('"',q1+1);
  if(q2==std::string::npos) return def; return s.substr(q1+1,q2-q1-1);
}

// ★★2026-08-21 축별 배열을 같이 낸다 — **`monitor_plot.py` 를 sim 에도 쓰기 위해서**다.
//   종전엔 스칼라 8개(mode·base_z·vx_cmd·vx_act·yaw·tilt·x·y)만 내서, 값 모니터를 붙여도
//   토크 그래프가 **비어 나왔다**. 그래서 sim 수치는 `biped_sim` 의 TAU_DBG 표로 따로
//   뽑아 손으로 대조해야 했다. 필드명을 `biped_deploy` 와 **같게** 맞추면 모니터가
//   시뮬인지 실기인지 구분하지 않아도 되고, 같은 화면 형식으로 나란히 볼 수 있다.
//
//   ⚠토크 두 열의 뜻이 실기와 미묘하게 다르다 — 읽을 때 주의:
//     `tau_leg_nm`  = `qfrc_actuator` = **실제 관절토크**. 실기의 같은 이름과 직접 비교된다.
//     `tau_cmd_nm`  = `d->ctrl` = **액추에이터 지령**. 관절토크와 다른 축은 **calf 뿐이다**
//                     (`u_calf = τ_calf − τ_foot`, biped_wbic.hpp:21 이 `i%4==2` 만 전단한다).
//                     ⚠**foot 은 같다** — `u_foot = τ_foot`. A⁻ᵀ 의 foot 행이 항등이라서다.
//                     2026-08-21 정정: 종전 "calf·foot 은 다르다" 는 틀렸다. 각도가 어긋나는
//                     축(foot)과 토크가 어긋나는 축(calf)이 **서로 다르다** — 여기가 제일 헷갈린다.
//   ⚠sim 에는 드라이버 게인이 없다 ⇒ kp_raw·kd_raw 는 0 으로 낸다(실기 순수토크모드와 동일).
struct TauWin {                                  // 발행창 통계(루프율로 누적 — 발행율이 아니라)
  std::vector<double> sum, sq, mn, mx; long n=0;
  void init(int k){ sum.assign(k,0.0); sq.assign(k,0.0); mn.assign(k,1e300); mx.assign(k,-1e300); n=0; }
  void add(const mjData* d, int k){
    for(int j=0;j<k;j++){ double v=d->qfrc_actuator[6+j];
      sum[j]+=v; sq[j]+=v*v; if(v<mn[j])mn[j]=v; if(v>mx[j])mx[j]=v; }
    n++;
  }
};

static void arr(std::ostringstream& o, const char* key, int n,
                const std::function<double(int)>& f, int prec=3){
  o<<",\""<<key<<"\":[";
  std::streamsize p=o.precision(); o.precision(prec);
  for(int j=0;j<n;j++){ if(j) o<<","; o<<f(j); }
  o.precision(p); o<<"]";
}

static void publish_state(mjModel* m, mjData* d, BipedControl& c, const std::string& mode,
                          const std::string& path, TauWin& tw, double loop_hz){
  static const double R2D=57.29577951308232;
  double* q=&d->qpos[3];
  double roll=atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
  double pitch=asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
  double yaw=atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]));
  double tilt=std::hypot(roll,pitch)*R2D;
  const int NJ=m->nu;                            // 다리 관절 수(=액추에이터 수)
  std::ostringstream o; o.precision(4); o<<std::fixed;
  o<<"{\"mode\":\""<<mode<<"\",\"backend\":\"sim\""
   <<",\"base_z\":"<<d->qpos[2]<<",\"vx_cmd\":"<<c.vx_cmd
   <<",\"vx_act\":"<<d->qvel[0]<<",\"wz_cmd\":"<<c.wz_cmd<<",\"yaw\":"<<yaw*R2D
   <<",\"tilt\":"<<tilt<<",\"tilt_deg\":"<<tilt<<",\"loop_hz\":"<<loop_hz
   <<",\"x\":"<<d->qpos[0]<<",\"y\":"<<d->qpos[1];
  arr(o,"q_leg_deg", NJ,[&](int j){ return d->qpos[7+j]*R2D; },2);
  arr(o,"dq_leg_dps",NJ,[&](int j){ return d->qvel[6+j]*R2D; },2);
  arr(o,"tau_leg_nm",NJ,[&](int j){ return d->qfrc_actuator[6+j]; });
  arr(o,"tau_cmd_nm",NJ,[&](int j){ return d->ctrl[j]; });
  const long n=tw.n;
  arr(o,"tau_std_nm",NJ,[&](int j){ if(!n) return 0.0;
      double mu=tw.sum[j]/n; return std::sqrt(std::max(0.0, tw.sq[j]/n-mu*mu)); });
  arr(o,"tau_min_nm",NJ,[&](int j){ return n? tw.mn[j] : 0.0; });
  arr(o,"tau_max_nm",NJ,[&](int j){ return n? tw.mx[j] : 0.0; });
  o<<",\"tau_win_n\":"<<n;
  arr(o,"kp_raw",NJ,[&](int){ return 0.0; },1);   // sim 은 드라이버 게인이 없다
  arr(o,"kd_raw",NJ,[&](int){ return 0.0; },1);
  o<<"}";
  tw.init(NJ);                                    // 창을 비운다 — 다음 발행까지 다시 쌓는다
  std::string tmp=path+".tmp"; std::ofstream f(tmp); f<<o.str(); f.close();
  std::rename(tmp.c_str(),path.c_str());
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

int main(int argc,char**argv){
  const char* path=argc>1?argv[1]:"../biped_from_quad.mjcf";
  char err[1000]={0}; mjModel* m=mj_loadXML(path,nullptr,err,1000);
  if(!m){ std::fprintf(stderr,"모델 로드 실패: %s\n",err); return 1; }
  mjData* d=mj_makeData(m); gM=m;
  BipedControl c(m,d); c.reset();
  std::string mode = getenv("MODE")?getenv("MODE"):"stand", prev_mode=mode; double body_h=c.com_ref_z;
  // ★배포 폐루프 기본 ON: 추정(leg-odom+접촉높이)+지연보상. 실HW는 항상 이 경로(센서만). GT=1로만 GT제어.
  bool est_ctrl = getenv("GT")==nullptr;
  DeployLoop dl;
  auto est_reset=[&](){ if(est_ctrl) dl.reset(m,d); };
  if(est_ctrl){
    dl.init(m,c); est_reset();
    std::fprintf(stderr,"[biped_view] EST_CTRL=ON — 추정 폐루프(SLAT=%d ALAT=%d LCOMP=%d step)\n",dl.SLAT,dl.ALAT,dl.LCOMP);
  }

  if(!glfwInit()){ std::fprintf(stderr,"glfw init 실패\n"); return 1; }
  GLFWwindow* win=glfwCreateWindow(1100,820,"biped C++ (MPC+WBIC)",NULL,NULL);
  if(!win){ std::fprintf(stderr,"창 생성 실패(DISPLAY?)\n"); glfwTerminate(); return 1; }
  glfwMakeContextCurrent(win); glfwSwapInterval(1);
  mjv_defaultCamera(&cam); mjv_defaultOption(&opt); mjv_defaultScene(&scn); mjr_defaultContext(&con);
  mjv_makeScene(m,&scn,2000); mjr_makeContext(m,&con,mjFONTSCALE_150);
  cam.distance=2.0; cam.elevation=-18; cam.azimuth=135; cam.lookat[2]=0.3;
  opt.flags[mjVIS_CONTACTFORCE]=1;
  glfwSetMouseButtonCallback(win,mouse_btn); glfwSetCursorPosCallback(win,mouse_move); glfwSetScrollCallback(win,scroll);

  const char* CMDFILE=getenv("CMDFILE");
  std::string STATE_PUB=getenv("STATE_PUB")?getenv("STATE_PUB"):"/tmp/biped_state.json";
  double dt=m->opt.timestep; long frame=0; int falls=0;
  TauWin tw; tw.init(m->nu); double loop_hz=1.0/dt;    // 창 통계 누적기(발행마다 비운다)
  double est_perr=0, est_verr=0;                        // 추정오차(HUD·고스트용)
  auto wall0=std::chrono::steady_clock::now(); double sim0=d->time;
  while(!glfwWindowShouldClose(win)){
    if(CMDFILE && (frame++ %3==0)){                     // GUI 명령 폴링
      std::ifstream f(CMDFILE);
      if(f){ std::stringstream ss; ss<<f.rdbuf(); std::string cj=ss.str();
        mode=json_str(cj,"mode",mode); body_h=json_get(cj,"body_h",body_h);
        // ★접촉모드(1pt/2pt) 전환: 같은 모델서 목표자세 재정착
        std::string contact=json_str(cj,"contact", c.cmode==1?"2pt":"1pt");
        int cm=(contact=="2pt")?1:0;
        if(c.has_heel && cm!=c.cmode){ c.set_contact_mode(cm); est_reset(); wall0=std::chrono::steady_clock::now(); sim0=d->time; mode="stand"; prev_mode="stand"; }
        if(mode=="reset"){ c.reset(); if(!c.has_heel) c.com_ref_z=body_h; mode="stand"; est_reset(); wall0=std::chrono::steady_clock::now(); sim0=d->time; }
        if(prev_mode=="off" && mode!="off"){ c.reset(); c.com_ref_z=body_h; est_reset(); wall0=std::chrono::steady_clock::now(); sim0=d->time; }  // ★전원 재투입: 낙상서 리셋 후 기립
        prev_mode=mode;
        bool walk=(mode=="walk");
        c.vx_cmd=walk?json_get(cj,"v",0):0; c.wz_cmd=walk?json_get(cj,"w",0):0; c.vy_cmd=walk?json_get(cj,"vy",0):0;
        c.com_ref_z=body_h; } }
    // 실시간 페이싱
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();
    double target=sim0+wall; int guard=0;
    bool off=(mode=="off");                              // ★모터 전원 off = 토크 0(limp). 실HW=motor disable
    while(d->time<target && guard++<60){
      if(off){ for(int i=0;i<m->nu;i++) d->ctrl[i]=0.0; }
      else if(est_ctrl) dl.step(m,d,c,dt);                 // ★추정+지연보상 폐루프(배포 경로)
      else c.control(dt);
      mj_step(m,d);
      tw.add(d,m->nu);          // 루프율로 토크 통계 누적(발행율이 아니라)
      if(!off && d->qpos[2]<0.2){ c.reset(); c.com_ref_z=body_h; est_reset(); falls++; wall0=std::chrono::steady_clock::now(); sim0=d->time; break; }  // 낙상 자동리셋(off는 제외)
    }
    if(est_ctrl){                                       // GT 대비 추정오차(고스트/HUD)
      est_perr=std::sqrt(std::pow(dl.est.p[0]-d->qpos[0],2)+std::pow(dl.est.p[1]-d->qpos[1],2)+std::pow(dl.est.p[2]-d->qpos[2],2));
      est_verr=std::sqrt(std::pow(dl.est.v[0]-d->qvel[0],2)+std::pow(dl.est.v[1]-d->qvel[1],2)+std::pow(dl.est.v[2]-d->qvel[2],2));
    }
    if(frame%3==0) publish_state(m,d,c,mode,STATE_PUB,tw,loop_hz);
    mjrRect vp={0,0,0,0}; glfwGetFramebufferSize(win,&vp.width,&vp.height);
    mjv_updateScene(m,d,&opt,NULL,&cam,mjCAT_ALL,&scn);
    if(est_ctrl && scn.ngeom+6<=scn.maxgeom){           // ★고스트: GT(초록)·추정(주황) base 마커 + 드리프트 이격선(17-DOF 방식)
      double H=0.35; float grn[4]={0.2f,0.9f,0.3f,0.9f}, org[4]={1.0f,0.55f,0.15f,0.9f};
      double szG[3]={0.03,0.03,0.03}, szL[3]={0,0,0};
      mjtNum gtM[3]={d->qpos[0],d->qpos[1],d->qpos[2]+H}, esM[3]={dl.est.p[0],dl.est.p[1],dl.est.p[2]+H};
      mjtNum gtB[3]={d->qpos[0],d->qpos[1],d->qpos[2]};
      mjvGeom* g0=&scn.geoms[scn.ngeom]; mjv_initGeom(g0,mjGEOM_SPHERE,szG,gtM,NULL,grn); g0->category=mjCAT_DECOR; scn.ngeom++;
      mjvGeom* g1=&scn.geoms[scn.ngeom]; mjv_initGeom(g1,mjGEOM_SPHERE,szG,esM,NULL,org); g1->category=mjCAT_DECOR; scn.ngeom++;
      mjvGeom* g2=&scn.geoms[scn.ngeom]; mjv_initGeom(g2,mjGEOM_LINE,szL,NULL,NULL,org); mjv_connector(g2,mjGEOM_LINE,5,gtM,esM); g2->category=mjCAT_DECOR; scn.ngeom++;
      mjvGeom* g3=&scn.geoms[scn.ngeom]; mjv_initGeom(g3,mjGEOM_LINE,szL,NULL,NULL,grn); mjv_connector(g3,mjGEOM_LINE,2,gtB,gtM); g3->category=mjCAT_DECOR; scn.ngeom++;
    }
    mjr_render(vp,&scn,&con);
    { char hud[400]; double* q=&d->qpos[3];              // ★오차 HUD (17-DOF trot_view 방식)
      double roll=std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
      double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
      if(est_ctrl)
        std::snprintf(hud,400,"mode=%s  z=%.3f  tilt=%.1f deg\nx=%+.2f  falls=%d\nEST_CTRL (leg-odom + contact-height)  [green=GT  orange=EST]\nest err  pos=%.1f cm  vel=%.3f m/s\nlatency sense=%d act=%d comp=%d step",
          mode.c_str(),d->qpos[2],std::hypot(roll,pitch)*180/M_PI,d->qpos[0],falls,est_perr*100,est_verr,dl.SLAT,dl.ALAT,dl.LCOMP);
      else
        std::snprintf(hud,400,"mode=%s  z=%.3f  tilt=%.1f deg\nx=%+.2f  falls=%d\nGT control (ground truth)\nmouse: rotate/zoom | control via GUI",
          mode.c_str(),d->qpos[2],std::hypot(roll,pitch)*180/M_PI,d->qpos[0],falls);
      mjr_overlay(mjFONT_NORMAL,mjGRID_TOPLEFT,vp,est_ctrl?"biped C++ (EST_CTRL: GT vs EST)":"biped C++ (GT control)",hud,&con);
    }
    glfwSwapBuffers(win); glfwPollEvents();
  }
  mjv_freeScene(&scn); mjr_freeContext(&con); glfwTerminate();
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
