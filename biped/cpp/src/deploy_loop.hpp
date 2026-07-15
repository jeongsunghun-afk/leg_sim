#pragma once
// biped 배포 폐루프 — 상태추정 + (선택)센서/구동 지연 + 지연보상(모델 롤아웃). Python biped_deploy 포팅.
//   step(): 센서→(지연)→추정→제어상태(base=추정·관절/자세=센서)→(지연보상 전진예측)→제어→(구동지연)→d.ctrl.
//   물리 상태(d)는 주입 후 복원(제어만 추정상태로, mj_step은 호출자가 GT로). 실HW=지연은 실측, 보상만 사용.
//   env: SENSE_LAT_MS(센서지연)·ACT_LAT_MS(구동지연)·LAT_COMP_MS(보상 예측). 실HW는 SENSE/ACT=0·LAT_COMP=실측지연.
#include <mujoco/mujoco.h>
#include "biped_control.hpp"
#include "state_estimator.hpp"
#include <Eigen/Dense>
#include <deque>
#include <vector>
#include <cstdlib>
#include <cmath>

struct DeployLoop {
  BipedEstimator est;
  mjData* dpred=nullptr;
  int sq=-1, sg=-1, SLAT=0, ALAT=0, LCOMP=0, NJ=0, nu=0;
  std::deque<std::vector<double>> sbuf, abuf;   // 센서 스냅샷 / 구동 토크 링버퍼

  ~DeployLoop(){ if(dpred) mj_deleteData(dpred); }
  void init(mjModel* m, BipedControl& c){
    NJ=m->nq-7; nu=m->nu;
    std::vector<int> fg={c.sph[0],c.sph[1]};
    std::vector<double> fr={m->geom_size[c.sph[0]*3], m->geom_size[c.sph[1]*3]};
    est.init(m,fg,fr);
    sq=mj_name2id(m,mjOBJ_SENSOR,"imu_quat"); sg=mj_name2id(m,mjOBJ_SENSOR,"imu_gyro");
    double dt=m->opt.timestep;
    auto ms=[&](const char*e){ const char*v=getenv(e); return v?atof(v):0.0; };
    SLAT=(int)std::lround(ms("SENSE_LAT_MS")/1000.0/dt);
    ALAT=(int)std::lround(ms("ACT_LAT_MS")/1000.0/dt);
    LCOMP=(int)std::lround(ms("LAT_COMP_MS")/1000.0/dt);
    if(LCOMP>0) dpred=mj_makeData(m);
  }
  void reset(mjModel* m, mjData* d){
    est.reset(Eigen::Vector3d(d->qpos[0],d->qpos[1],d->qpos[2])); sbuf.clear(); abuf.clear();
  }

  // d->ctrl 를 (지연·추정·보상 반영) 세팅. 호출자가 이어서 mj_step. 물리 d는 불변(주입 후 복원).
  void step(mjModel* m, mjData* d, BipedControl& c, double dt){
    // ① 센서 스냅샷(현재)  [q(NJ)|dq(NJ)|quat(4)|gyro(3)]
    std::vector<double> snap(2*NJ+7);
    for(int j=0;j<NJ;j++){ snap[j]=d->qpos[7+j]; snap[NJ+j]=d->qvel[6+j]; }
    double* qs = sq>=0? &d->sensordata[m->sensor_adr[sq]] : &d->qpos[3];
    double* gs = sg>=0? &d->sensordata[m->sensor_adr[sg]] : &d->qvel[3];
    for(int a=0;a<4;a++) snap[2*NJ+a]=qs[a];
    for(int a=0;a<3;a++) snap[2*NJ+4+a]=gs[a];
    // ② 센서 지연(링버퍼)
    sbuf.push_back(snap);
    while((int)sbuf.size()>SLAT+1) sbuf.pop_front();
    std::vector<double>& s=sbuf.front();
    double *sqj=&s[0], *sdq=&s[NJ], *squat=&s[2*NJ], *sgyro=&s[2*NJ+4];
    // ③ 추정(지연 센서로)
    std::vector<bool> cts(2,false);
    for(int ci=0;ci<d->ncon;ci++){ int g1=d->contact[ci].geom1,g2=d->contact[ci].geom2;
      for(int k=0;k<2;k++) if(c.sph[k]==g1||c.sph[k]==g2) cts[k]=true; }
    est.estimate(m, sqj, sdq, squat, sgyro, cts, dt);
    // ④ 제어상태: base=추정 · 자세/관절=지연센서
    std::vector<double> cqp(m->nq), cqv(m->nv);
    cqp[0]=est.p[0]; cqp[1]=est.p[1]; cqp[2]=est.p[2];
    for(int a=0;a<4;a++) cqp[3+a]=squat[a];
    for(int j=0;j<NJ;j++) cqp[7+j]=sqj[j];
    cqv[0]=est.v[0]; cqv[1]=est.v[1]; cqv[2]=est.v[2];
    for(int a=0;a<3;a++) cqv[3+a]=sgyro[a];
    for(int j=0;j<NJ;j++) cqv[6+j]=sdq[j];
    // ⑤ 지연보상: in-flight 토크 유지하며 LCOMP step 모델 롤아웃(Smith predictor)
    if(LCOMP>0 && dpred){
      std::vector<double> th(nu,0.0); if(!abuf.empty()) th=abuf.back();
      for(int i=0;i<m->nq;i++) dpred->qpos[i]=cqp[i];
      for(int i=0;i<m->nv;i++) dpred->qvel[i]=cqv[i];
      for(int l=0;l<LCOMP;l++){ for(int i=0;i<nu;i++) dpred->ctrl[i]=th[i]; mj_step(m,dpred); }
      for(int i=0;i<m->nq;i++) cqp[i]=dpred->qpos[i];
      for(int i=0;i<m->nv;i++) cqv[i]=dpred->qvel[i];
    }
    // ⑥ 주입 → 제어 → 물리 복원
    std::vector<double> gp(m->nq), gv(m->nv);
    for(int i=0;i<m->nq;i++){ gp[i]=d->qpos[i]; d->qpos[i]=cqp[i]; }
    for(int i=0;i<m->nv;i++){ gv[i]=d->qvel[i]; d->qvel[i]=cqv[i]; }
    mj_forward(m,d); c.control(dt);
    // ⑦ 구동 지연(링버퍼) → 지연된 토크를 d->ctrl 에
    std::vector<double> tau(nu); for(int i=0;i<nu;i++) tau[i]=d->ctrl[i];
    abuf.push_back(tau);
    while((int)abuf.size()>ALAT+1) abuf.pop_front();
    std::vector<double> ta=abuf.front();
    for(int i=0;i<m->nq;i++) d->qpos[i]=gp[i];
    for(int i=0;i<m->nv;i++) d->qvel[i]=gv[i];
    mj_forward(m,d);
    for(int i=0;i<nu;i++) d->ctrl[i]=ta[i];
  }
};
