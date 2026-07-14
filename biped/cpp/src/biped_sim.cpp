// biped C++ 단독 sim — MJCF 로드 → BipedControl 제어루프 → mj_step. 헤드리스 폐루프 검증.
// 실행: ./biped_sim [mjcf] [vx] [T]   (기본 ../biped_from_quad.mjcf 0.15 15)
#include <mujoco/mujoco.h>
#include "biped_control.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>

int main(int argc,char**argv){
  const char* mjcf = argc>1?argv[1]:"../biped_from_quad.mjcf";
  double vx = argc>2?atof(argv[2]):0.15;
  double T  = argc>3?atof(argv[3]):15.0;
  char err[1000]={0};
  mjModel* m=mj_loadXML(mjcf,nullptr,err,1000);
  if(!m){ printf("모델 로드 실패: %s\n",err); return 1; }
  mjData* d=mj_makeData(m);
  BipedControl c(m,d); c.reset(); c.vx_cmd=vx;
  double dt=m->opt.timestep; int steps=(int)(T/dt); double fell=-1;
  for(int i=0;i<steps;i++){
    c.control(dt); mj_step(m,d);
    if(d->qpos[2]<0.15){ fell=i*dt; break; }
  }
  double* q=&d->qpos[3];
  double roll=std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
  double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
  double tilt=std::hypot(roll,pitch)*180/M_PI;
  printf("vx=%.2f · 생존 %.2fs%s · base=(%.3f,%.3f,%.3f) tilt=%.1f°\n",
         vx, fell<0?T:fell, fell<0?"(무낙상)":"(낙상)",
         d->qpos[0],d->qpos[1],d->qpos[2], tilt);
  mj_deleteData(d); mj_deleteModel(m); return fell<0?0:0;
}
