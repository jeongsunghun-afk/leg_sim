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
  // ★stance-anchored odometry: 접촉 발을 월드 앵커로 고정 → base 위치를 매 스텝 기구학 재고정(속도 순수적분 드리프트 제거).
  //   접촉 개시 프레임에 anchor=현재 p+R·pfb 로 등록(점프 없음) → 접촉 동안 p=mean(anchor−R·pfb). 몸이 올라와도 발 고정 제약으로 정확 추종.
  //   앉기/서기(base 상하·CoM gather)서 ZUPT 없이 드리프트 없이 따라감. 비접촉(flight)=속도적분. 재접지 시 현재 p로 재앵커(실기 leg-odom과 동일한 재배치 드리프트).
  std::vector<Eigen::Vector3d> anchor;   // 발별 월드 앵커(접촉 개시 위치, xy만 사용)
  std::vector<char> anch_on;             // 발별 앵커 활성
  double SLIP_RESET = 0.05;              // 발 implied-base가 합의서 이만큼 벗어나면 미끄럼 판정→재앵커(오추정 전파 차단)
  double ground_z = 0.0;                 // ★평지 지면높이(z 기구학 앵커 기준). 실기 지형=terrain map으로 대체 예정
  // ★설계 근거: leg-odom은 "몸 상승"과 "발 미끄러짐"을 자기수용감각만으론 구분 못 함(앉기/서기 fold서 발 대거 재배치→순수 적분 z 1.2m 폭주).
  //   균형에 결정적인 z·자세는 견고하게 분리: z=접촉발 기구학(평지 가정, 드리프트 0)·자세=IMU직접·xy=stance 앵커(유계)·비접촉=유지(적분 폭주 차단).
  ~StateEstimator(){ if(ed) mj_deleteData(ed); }
  void reset(const Eigen::Vector3d& p0){ p = p0; v.setZero(); anch_on.assign(anch_on.size(),0); }

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
    if((int)anchor.size()!=(int)foot_geom.size()){ anchor.assign(foot_geom.size(),Vector3d::Zero()); anch_on.assign(foot_geom.size(),0); }
    std::vector<Vector3d> vbs, pbs, rels;                     // 속도후보 · 앵커기반 implied base · base→발(world)
    std::vector<int> pk;
    std::vector<double> jb(3 * m->nv);
    for(size_t k=0;k<foot_geom.size();k++){
      if(!contacts[k]){ anch_on[k]=0; continue; }             // stance 발만(비접촉=앵커 해제)
      int g = foot_geom[k];
      Vector3d pfb(ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]-foot_rad[k]);  // 발접촉점(base frame)
      Vector3d rel = R*pfb;                                   // base원점→발접촉점(world 회전)
      mjtNum pnt[3]={ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]};
      mj_jac(m, ed, jb.data(), nullptr, pnt, m->geom_bodyid[g]);
      Vector3d vfb = Vector3d::Zero();                        // 발 속도(관절 기여, base frame)
      for(int r=0;r<3;r++){ double s=0; for(int c=0;c<m->nv;c++) s+=jb[r*m->nv+c]*ed->qvel[c]; vfb[r]=s; }
      vbs.push_back(-(omw.cross(rel) + R*vfb));               // 발 정지 → base world 속도
      if(!anch_on[k]){ anchor[k]=p+rel; anch_on[k]=1; }       // ★접촉 개시=현재 p로 앵커(점프 없음)
      pbs.push_back(anchor[k]-rel); rels.push_back(rel); pk.push_back((int)k);   // ★발 고정 제약 → implied base 위치
    }
    if(!vbs.empty()){
      Vector3d mean = Vector3d::Zero(); for(auto& x:vbs) mean += x; mean /= (double)vbs.size();
      v = (1-alpha)*v + alpha*mean;                          // 접촉 평균 + 저역통과
    }
    if(!pbs.empty()){
      // ── xy: stance 앵커 합의(아웃라이어=미끄럼발 제외 후 재앵커). z 성분은 앵커 대신 아래서 기구학으로 덮어씀 ──
      Vector3d mp = Vector3d::Zero(); for(auto& x:pbs) mp += x; mp /= (double)pbs.size();
      if(pbs.size()>=2){
        Vector3d mp2 = Vector3d::Zero(); int n2=0;
        for(auto& x:pbs) if(std::hypot(x[0]-mp[0],x[1]-mp[1])<=SLIP_RESET){ mp2+=x; n2++; }
        if(n2>0) mp = mp2/(double)n2;
      }
      for(size_t i=0;i<pbs.size();i++) if(std::hypot(pbs[i][0]-mp[0],pbs[i][1]-mp[1])>SLIP_RESET) anchor[pk[i]]=mp+rels[i];
      // ── z: 접촉발 기구학(평지 ground_z 가정). base_z = ground_z − mean(발 상대z). 미끄럼/fold서도 드리프트 0 ──
      double relz=0; for(auto& r:rels) relz += r[2]; relz/=(double)rels.size();
      p[0]=mp[0]; p[1]=mp[1]; p[2]=ground_z - relz;
    }
    // 비접촉(flight): 위치 유지(정적/기립 fold의 노이즈 속도적분 폭주 차단). 점프 탄도는 후속(IMU 가속 융합).
  }

  // ════════ 표준 접촉기반 선형 KF (MIT Cheetah, Bledt 2018 "Cheetah 3") — IMU 가속도 융합 + 공분산 ════════
  //   상태 x=[p(3),v(3),p1,p2,p3,p4(각3)] 18차. 예측=IMU 가속도 적분(p,v)·발위치 상수. 보정=접촉발 상대위치·정지속도·지면z.
  //   접촉 여부로 측정 공분산 스케일(swing발=거대노이즈로 자동 무시). 자세=IMU 직접(필터 밖). InEKF=이 필터의 자세-정합 Lie군 일반화.
  Eigen::Matrix<double,18,1> xkf = Eigen::Matrix<double,18,1>::Zero();
  Eigen::Matrix<double,18,18> Pkf = Eigen::Matrix<double,18,18>::Identity();
  bool kf_init = false;
  double KF_QP=1e-4, KF_QV=1e-2, KF_QF=1e-3, KF_QF_SWING=1e3;   // 프로세스: 위치·속도·접촉발·swing발
  double KF_RP=1e-3, KF_RV=1e-2, KF_RZ=1e-4, KF_R_SWING=1e6;    // 측정: 상대위치·정지속도·지면z·swing스케일
  void reset_kf(const Eigen::Vector3d& p0){ xkf.setZero(); xkf.head<3>()=p0; Pkf.setIdentity(); Pkf*=1e-2; kf_init=true; p=p0; v.setZero(); }

  // accel_w: IMU 가속도(중력 보정 후 world 선가속도, a=R·f_body+g). 나머지 인자=estimate와 동일.
  void estimate_kf(mjModel* m, const double* qj, const double* dqj,
                   const double* quat_wxyz, const double* gyro, const double* accel_w,
                   const std::vector<int>& foot_geom, const std::vector<double>& foot_rad,
                   const std::vector<bool>& contacts, double dt){
    using namespace Eigen;
    if(!ed) ed = mj_makeData(m);
    if(!kf_init) reset_kf(p);
    int NJ = m->nq - 7, NF=(int)foot_geom.size();
    for(int i=0;i<m->nq;i++) ed->qpos[i]=0; ed->qpos[3]=1;
    for(int j=0;j<NJ;j++) ed->qpos[7+j]=qj[j];
    for(int i=0;i<m->nv;i++) ed->qvel[i]=0;
    for(int j=0;j<NJ;j++) ed->qvel[6+j]=dqj[j];
    mj_forward(m, ed);
    double Rm[9]; mju_quat2Mat(Rm, quat_wxyz); Map<Matrix<double,3,3,RowMajor>> R(Rm);
    Vector3d omw = R * Vector3d(gyro[0],gyro[1],gyro[2]);
    Vector3d aw(accel_w[0],accel_w[1],accel_w[2]);

    // ── 예측: p+=v dt+½a dt²; v+=a dt; 발위치 상수 ──
    xkf.segment<3>(0) += xkf.segment<3>(3)*dt + 0.5*aw*dt*dt;
    xkf.segment<3>(3) += aw*dt;
    Matrix<double,18,18> A = Matrix<double,18,18>::Identity();
    A.block<3,3>(0,3) = Matrix3d::Identity()*dt;
    Matrix<double,18,18> Q = Matrix<double,18,18>::Zero();
    Q.block<3,3>(0,0) = Matrix3d::Identity()*KF_QP;
    Q.block<3,3>(3,3) = Matrix3d::Identity()*KF_QV;
    for(int k=0;k<NF;k++) Q.block<3,3>(6+3*k,6+3*k) = Matrix3d::Identity()*(contacts[k]?KF_QF:KF_QF_SWING);
    Pkf = A*Pkf*A.transpose() + Q;

    // ── 보정: 발별 상대위치(3)+정지속도(3), 지면z(1). 접촉이면 정밀, swing이면 R 거대 ──
    std::vector<double> jb(3*m->nv);
    int MM = 7*NF;   // 발당 pos3+vel3+z1
    MatrixXd H = MatrixXd::Zero(MM, 18);
    VectorXd yres(MM); VectorXd Rd(MM);
    for(int k=0;k<NF;k++){
      int g = foot_geom[k];
      Vector3d pfb(ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]-foot_rad[k]);
      Vector3d s = R*pfb;                                   // base→발(world)
      mjtNum pnt[3]={ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]};
      mj_jac(m, ed, jb.data(), nullptr, pnt, m->geom_bodyid[g]);
      Vector3d vfb=Vector3d::Zero(); for(int r=0;r<3;r++){ double sm=0; for(int c=0;c<m->nv;c++) sm+=jb[r*m->nv+c]*ed->qvel[c]; vfb[r]=sm; }
      Vector3d pi = xkf.segment<3>(6+3*k), p_=xkf.segment<3>(0), v_=xkf.segment<3>(3);
      bool ct=contacts[k]; double rsc = ct?1.0:KF_R_SWING;
      // (1) 상대위치: 예측 pi−p = s → resid = s−(pi−p). H: ∂/∂p=−I, ∂/∂pi=+I
      int r0=7*k;
      H.block<3,3>(r0,0) = -Matrix3d::Identity(); H.block<3,3>(r0,6+3*k) = Matrix3d::Identity();
      yres.segment<3>(r0) = s - (pi - p_);
      for(int a=0;a<3;a++) Rd[r0+a]=KF_RP*rsc;
      // (2) 정지속도: 접촉발 world속도 0 → v = −(ω×s + R·vfb). resid = 그 값 − v. H: ∂/∂v=+I
      H.block<3,3>(r0+3,3) = Matrix3d::Identity();
      yres.segment<3>(r0+3) = (-(omw.cross(s) + R*vfb)) - v_;
      for(int a=0;a<3;a++) Rd[r0+3+a]=KF_RV*rsc;
      // (3) 지면 z: 접촉발 world z = ground_z → pi_z = ground_z. resid = ground_z − pi_z. H: ∂/∂pi_z=+1
      H(r0+6,6+3*k+2) = 1.0;
      yres[r0+6] = ground_z - pi[2];
      Rd[r0+6] = KF_RZ*rsc;
    }
    MatrixXd Rm2 = Rd.asDiagonal();
    MatrixXd S = H*Pkf*H.transpose() + Rm2;
    MatrixXd K = Pkf*H.transpose()*S.ldlt().solve(MatrixXd::Identity(MM,MM));
    xkf += K*yres;
    Pkf = (Matrix<double,18,18>::Identity() - K*H)*Pkf;
    p = xkf.segment<3>(0); v = xkf.segment<3>(3);
  }
};
