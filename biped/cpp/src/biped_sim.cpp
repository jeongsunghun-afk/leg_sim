// biped C++ 단독 sim — MJCF 로드 → BipedControl 제어루프 → mj_step. 헤드리스 폐루프 검증.
// 실행: ./biped_sim [mjcf] [vx] [T]   (기본 ../biped_from_quad.mjcf 0.15 15)
// ★EST_CTRL=1 : 추정 상태(leg-odom+접촉높이)로 폐루프 제어(물리는 GT). 배포 경로 검증. falls 카운트.
#include <mujoco/mujoco.h>
#include "biped_control.hpp"
#include "deploy_loop.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

static double tilt_deg(const double* q){
  double roll=std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
  double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
  return std::hypot(roll,pitch)*180/M_PI;
}

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

  bool est_ctrl = getenv("EST_CTRL")!=nullptr;
  DeployLoop dl; int falls=0;
  if(est_ctrl){ dl.init(m,c); dl.reset(m,d); }

  for(int i=0;i<steps;i++){
    if(est_ctrl) dl.step(m,d,c,dt);      // 추정+지연+보상 → d->ctrl (물리 d 불변)
    else c.control(dt);
    mj_step(m,d);
    if(est_ctrl){                                           // 낙상 자동리셋 + 카운트(장시간 통계)
      if(d->qpos[2]<0.2 || tilt_deg(&d->qpos[3])>45){
        c.reset(); c.vx_cmd=vx; dl.reset(m,d); falls++;
      }
    } else if(d->qpos[2]<0.15){ fell=i*dt; break; }
  }

  if(est_ctrl){
    printf("EST_CTRL vx=%.2f T=%.1fs · falls=%d · 추정 base=(%.2f,%.2f,%.3f) GT=(%.2f,%.2f,%.3f) tilt=%.1f°\n",
           vx, T, falls, dl.est.p[0],dl.est.p[1],dl.est.p[2], d->qpos[0],d->qpos[1],d->qpos[2], tilt_deg(&d->qpos[3]));
  } else {
    printf("vx=%.2f · 생존 %.2fs%s · base=(%.3f,%.3f,%.3f) tilt=%.1f°\n",
           vx, fell<0?T:fell, fell<0?"(무낙상)":"(낙상)",
           d->qpos[0],d->qpos[1],d->qpos[2], tilt_deg(&d->qpos[3]));
  }
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
