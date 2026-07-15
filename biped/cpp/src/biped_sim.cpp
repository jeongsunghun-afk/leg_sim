// biped C++ 단독 sim — MJCF 로드 → BipedControl 제어루프 → mj_step. 헤드리스 폐루프 검증.
// 실행: ./biped_sim [mjcf] [vx] [T]   (기본 ../biped_from_quad.mjcf 0.15 15)
// ★EST_CTRL=1 : 추정 상태(leg-odom+접촉높이)로 폐루프 제어(물리는 GT). 배포 경로 검증. falls 카운트.
#include <mujoco/mujoco.h>
#include "biped_control.hpp"
#include "state_estimator.hpp"
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
  BipedEstimator est;
  int sq=mj_name2id(m,mjOBJ_SENSOR,"imu_quat"), sg=mj_name2id(m,mjOBJ_SENSOR,"imu_gyro");
  int falls=0;
  if(est_ctrl){
    std::vector<int> fg={c.sph[0],c.sph[1]};
    std::vector<double> fr={m->geom_size[c.sph[0]*3], m->geom_size[c.sph[1]*3]};
    est.init(m,fg,fr); est.reset(Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2]));
  }

  for(int i=0;i<steps;i++){
    if(est_ctrl){
      std::vector<bool> cts(2,false);                       // 발 접촉 검출
      for(int ci=0;ci<d->ncon;ci++){ int g1=d->contact[ci].geom1,g2=d->contact[ci].geom2;
        for(int k=0;k<2;k++) if(c.sph[k]==g1||c.sph[k]==g2) cts[k]=true; }
      double* quat = sq>=0 ? &d->sensordata[m->sensor_adr[sq]] : &d->qpos[3];  // IMU 센서(없으면 fallback)
      double* gyro = sg>=0 ? &d->sensordata[m->sensor_adr[sg]] : &d->qvel[3];
      est.estimate(m, &d->qpos[7], &d->qvel[6], quat, gyro, cts, dt);
      double gp[3]={d->qpos[0],d->qpos[1],d->qpos[2]}, gv[3]={d->qvel[0],d->qvel[1],d->qvel[2]};
      for(int a=0;a<3;a++){ d->qpos[a]=est.p[a]; d->qvel[a]=est.v[a]; }   // 추정 base 주입
      mj_forward(m,d); c.control(dt);
      for(int a=0;a<3;a++){ d->qpos[a]=gp[a]; d->qvel[a]=gv[a]; }         // 물리는 GT 복원
      mj_forward(m,d);
    } else c.control(dt);
    mj_step(m,d);
    if(est_ctrl){                                           // 낙상 자동리셋 + 카운트(장시간 통계)
      if(d->qpos[2]<0.2 || tilt_deg(&d->qpos[3])>45){
        c.reset(); c.vx_cmd=vx; est.reset(Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2])); falls++;
      }
    } else if(d->qpos[2]<0.15){ fell=i*dt; break; }
  }

  if(est_ctrl){
    printf("EST_CTRL vx=%.2f T=%.1fs · falls=%d · 추정 base=(%.2f,%.2f,%.3f) GT=(%.2f,%.2f,%.3f) tilt=%.1f°\n",
           vx, T, falls, est.p[0],est.p[1],est.p[2], d->qpos[0],d->qpos[1],d->qpos[2], tilt_deg(&d->qpos[3]));
  } else {
    printf("vx=%.2f · 생존 %.2fs%s · base=(%.3f,%.3f,%.3f) tilt=%.1f°\n",
           vx, fell<0?T:fell, fell<0?"(무낙상)":"(낙상)",
           d->qpos[0],d->qpos[1],d->qpos[2], tilt_deg(&d->qpos[3]));
  }
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
