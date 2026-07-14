// biped_view — C++ biped 폐루프를 MuJoCo GLFW 뷰어로 렌더 + GUI(JSON) 연동.
//   마우스: 좌드래그=회전 우드래그=이동 휠=줌.  GUI(teleop_gui_biped)가 v/vy/w/body_h/mode 조종.
//   실행: CMDFILE=/tmp/biped_cmd.json ./biped_view ../biped_from_quad.mjcf
#include "biped_control.hpp"
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>
#include <string>

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

static void publish_state(mjModel* m, mjData* d, BipedControl& c, const std::string& mode, const std::string& path){
  double* q=&d->qpos[3];
  double roll=atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
  double pitch=asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
  double yaw=atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]));
  std::ostringstream o; o.precision(4); o<<std::fixed;
  o<<"{\"mode\":\""<<mode<<"\",\"base_z\":"<<d->qpos[2]<<",\"vx_cmd\":"<<c.vx_cmd
   <<",\"vx_act\":"<<d->qvel[0]<<",\"wz_cmd\":"<<c.wz_cmd<<",\"yaw\":"<<yaw*57.29578
   <<",\"tilt\":"<<std::hypot(roll,pitch)*57.29578<<",\"x\":"<<d->qpos[0]<<",\"y\":"<<d->qpos[1]<<"}";
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
  double dt=m->opt.timestep; long frame=0;
  auto wall0=std::chrono::steady_clock::now(); double sim0=d->time;
  while(!glfwWindowShouldClose(win)){
    if(CMDFILE && (frame++ %3==0)){                     // GUI 명령 폴링
      std::ifstream f(CMDFILE);
      if(f){ std::stringstream ss; ss<<f.rdbuf(); std::string cj=ss.str();
        mode=json_str(cj,"mode",mode); body_h=json_get(cj,"body_h",body_h);
        if(mode=="reset"){ c.reset(); c.com_ref_z=body_h; mode="stand"; wall0=std::chrono::steady_clock::now(); sim0=d->time; }
        if(prev_mode=="off" && mode!="off"){ c.reset(); c.com_ref_z=body_h; wall0=std::chrono::steady_clock::now(); sim0=d->time; }  // ★전원 재투입: 낙상서 리셋 후 기립
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
      else c.control(dt);
      mj_step(m,d);
      if(!off && d->qpos[2]<0.2){ c.reset(); c.com_ref_z=body_h; wall0=std::chrono::steady_clock::now(); sim0=d->time; break; }  // 낙상 자동리셋(off는 제외)
    }
    if(frame%3==0) publish_state(m,d,c,mode,STATE_PUB);
    mjrRect vp={0,0,0,0}; glfwGetFramebufferSize(win,&vp.width,&vp.height);
    mjv_updateScene(m,d,&opt,NULL,&cam,mjCAT_ALL,&scn); mjr_render(vp,&scn,&con);
    glfwSwapBuffers(win); glfwPollEvents();
  }
  mjv_freeScene(&scn); mjr_freeContext(&con); glfwTerminate();
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
