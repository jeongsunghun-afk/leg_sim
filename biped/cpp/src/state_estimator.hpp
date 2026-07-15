#pragma once
// biped 상태추정 (leg-odometry + 접촉 기반 높이) — Python deploy/robot_interface.py StateEstimator C++ 포팅.
//   절대 base 안 씀(실로봇 동일). 센서만(관절 q/dq + IMU quat/gyro + 접촉) → base pose/vel.
//   ★핵심: 높이 z는 적분하지 말고 접촉발이 지면(ground_z)에 붙은 사실로 직접 측정(p_z=ground_z−pfw_z)=드리프트 없음.
//         (leg-odom z 적분 드리프트가 WBIC 높이 task를 붕괴시키는 게 폐루프 실패의 원인이었음 — 접촉높이로 해결.)
//   속도: stance 발 정지 가정 v_base=−(ω×R·p_foot+R·v_foot) 평균 + 저역통과(alpha). xy는 적분(드리프트 무해).
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <vector>

struct BipedEstimator {
  mjData* ed = nullptr;                 // 상대 운동학용 scratch(base 원점·단위자세)
  Eigen::Vector3d p = Eigen::Vector3d::Zero();   // 추정 base 위치(world)
  Eigen::Vector3d v = Eigen::Vector3d::Zero();   // 추정 base 선속도(world)
  double alpha = 0.4, ground_z = 0.0, k_anchor = 0.0;   // 앵커 기본 off(접촉높이가 핵심). >0시 xy 앵커
  bool contact_height = true;
  std::vector<int> fgeom; std::vector<double> frad;
  Eigen::Vector3d anchor[2]; bool has_anchor[2]={false,false};   // 접촉 xy 앵커(드리프트 완화)

  ~BipedEstimator(){ if(ed) mj_deleteData(ed); }
  void init(mjModel* m, const std::vector<int>& fg, const std::vector<double>& fr){
    fgeom = fg; frad = fr; if(!ed) ed = mj_makeData(m);
  }
  void reset(const Eigen::Vector3d& p0){ p = p0; v.setZero(); }

  // qj/dqj: 관절 q/dq(NJ) · quat_wxyz: IMU 자세 · gyro: 동체 각속도 · contacts: 발별 stance
  void estimate(mjModel* m, const double* qj, const double* dqj,
                const double* quat_wxyz, const double* gyro,
                const std::vector<bool>& contacts, double dt){
    using namespace Eigen;
    int NJ = m->nq - 7;
    for(int i=0;i<m->nq;i++) ed->qpos[i]=0; ed->qpos[3]=1;   // base 원점·단위 → 순수 다리 운동학
    for(int j=0;j<NJ;j++) ed->qpos[7+j]=qj[j];
    for(int i=0;i<m->nv;i++) ed->qvel[i]=0;
    for(int j=0;j<NJ;j++) ed->qvel[6+j]=dqj[j];
    mj_forward(m, ed);
    double Rm[9]; mju_quat2Mat(Rm, quat_wxyz);
    Map<Matrix<double,3,3,RowMajor>> R(Rm);
    Vector3d omw = R * Vector3d(gyro[0],gyro[1],gyro[2]);    // 동체 각속도 → world
    std::vector<Vector3d> vbs; std::vector<double> zmeas;
    std::vector<std::pair<int,Vector3d>> pfw_c;             // (발 idx, world 오프셋) — 앵커·높이용
    std::vector<double> jb(3 * m->nv);
    for(size_t k=0;k<fgeom.size();k++){
      if(!contacts[k]){ has_anchor[k]=false; continue; }    // 이지=앵커 리셋. stance 발만.
      int g = fgeom[k];
      Vector3d pfb(ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]-frad[k]);  // 접촉점(base frame)
      mjtNum pnt[3]={ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]};
      mj_jac(m, ed, jb.data(), nullptr, pnt, m->geom_bodyid[g]);
      Vector3d vfb = Vector3d::Zero();                      // 발 속도(관절 기여, base frame)
      for(int r=0;r<3;r++){ double s=0; for(int c=0;c<m->nv;c++) s+=jb[r*m->nv+c]*ed->qvel[c]; vfb[r]=s; }
      Vector3d pfw = R * pfb;                               // 발 오프셋(world)
      vbs.push_back(-(omw.cross(pfw) + R*vfb));             // 발 정지 → base world 속도
      zmeas.push_back(pfw[2]); pfw_c.push_back({(int)k, pfw});
    }
    if(!vbs.empty()){
      Vector3d mean = Vector3d::Zero(); for(auto& x:vbs) mean += x; mean /= (double)vbs.size();
      v = (1-alpha)*v + alpha*mean;                         // 접촉 평균 + 저역통과
    }
    p += v * dt;                                            // xy 적분(드리프트 허용)
    for(auto& pc : pfw_c){                                  // xy 접촉 앵커(드리프트 완화)
      int k=pc.first; const Vector3d& pfw=pc.second;
      if(!has_anchor[k]){ anchor[k]=p+pfw; has_anchor[k]=true; }
      else if(k_anchor>0){ Vector3d pm=anchor[k]-pfw; p[0]+=k_anchor*(pm[0]-p[0]); p[1]+=k_anchor*(pm[1]-p[1]); }
    }
    if(contact_height && !zmeas.empty()){                   // ★z: 접촉발 지면 사실로 직접 측정(드리프트 없음)
      double zm=0; for(double z:zmeas) zm+=z; zm/=(double)zmeas.size();
      p[2] = ground_z - zm;
    }
  }
};
