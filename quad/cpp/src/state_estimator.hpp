#pragma once
// sim2real 준비 — 상태추정기(leg odometry). IMU(자세·각속도) + 관절 엔코더(q,dq) + 접촉 → base pose/vel 추정.
//   ★절대 base 안 씀(실로봇과 동일). stance 발 정지 가정: v_base = −(ω×R·p_foot + R·v_foot) 평균 + 저역통과.
//   base 위치=속도 적분(드리프트=실기와 동일). 자세=IMU 직접, 각속도=gyro 직접.
//   Python StateEstimator(quad_fulldynamics.py) C++ 포팅. sim(quad)서 검증 후 실기(motion-controller) 재사용.
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <vector>
#include <array>

struct StateEstimator {
  mjData* ed = nullptr;              // 상대 운동학용 scratch(base 원점·단위)
  Eigen::Vector3d p = Eigen::Vector3d::Zero();   // 추정 base 위치(world)
  Eigen::Vector3d v = Eigen::Vector3d::Zero();   // 추정 base 선속도(world)
  ~StateEstimator(){ if(ed) mj_deleteData(ed); }
  void reset(const Eigen::Vector3d& p0){ p = p0; v.setZero(); }

  // qj/dqj: 관절 위치·속도(qpos/qvel 순, NJ개) · quat_wxyz: IMU 자세(MuJoCo wxyz) · gyro: 동체 각속도(3)
  // foot_geom/foot_rad: 발 sphere geom id·반경 · contacts: 발별 stance 여부
  void estimate(mjModel* m, const double* qj, const double* dqj,
                const double* quat_wxyz, const double* gyro,
                const std::vector<int>& foot_geom, const std::vector<double>& foot_rad,
                const std::vector<bool>& contacts, double dt, double alpha = 0.4){
    using namespace Eigen;
    if(!ed) ed = mj_makeData(m);
    int NJ = m->nq - 7;
    for(int i=0;i<m->nq;i++) ed->qpos[i]=0; ed->qpos[3]=1;   // base 원점·단위자세(상대 운동학)
    for(int j=0;j<NJ;j++) ed->qpos[7+j]=qj[j];
    for(int i=0;i<m->nv;i++) ed->qvel[i]=0;
    for(int j=0;j<NJ;j++) ed->qvel[6+j]=dqj[j];
    mj_forward(m, ed);
    double Rm[9]; mju_quat2Mat(Rm, quat_wxyz);
    Map<Matrix<double,3,3,RowMajor>> R(Rm);
    Vector3d gyroV(gyro[0],gyro[1],gyro[2]);
    Vector3d omw = R * gyroV;                                 // 동체 각속도 → world
    std::vector<Vector3d> vbs;
    std::vector<double> jb(3 * m->nv);
    for(size_t k=0;k<foot_geom.size();k++){
      if(!contacts[k]) continue;                             // stance 발만
      int g = foot_geom[k];
      Vector3d pfb(ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]-foot_rad[k]);  // 발접촉점(base frame)
      mjtNum pnt[3]={ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]};
      mj_jac(m, ed, jb.data(), nullptr, pnt, m->geom_bodyid[g]);
      Vector3d vfb = Vector3d::Zero();                        // 발 속도(관절 기여, base frame)
      for(int r=0;r<3;r++){ double s=0; for(int c=0;c<m->nv;c++) s+=jb[r*m->nv+c]*ed->qvel[c]; vfb[r]=s; }
      vbs.push_back(-(omw.cross(R*pfb) + R*vfb));             // 발 정지 → base world 속도
    }
    if(!vbs.empty()){
      Vector3d mean = Vector3d::Zero(); for(auto& x:vbs) mean += x; mean /= (double)vbs.size();
      v = (1-alpha)*v + alpha*mean;                          // 접촉 평균 + 저역통과
    }
    p += v * dt;                                             // 위치 적분(드리프트 허용)
  }
};
