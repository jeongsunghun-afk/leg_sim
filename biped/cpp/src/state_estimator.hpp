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
  // ★야코비안 평가점. true=접촉점(물리적으로 옳음) · false=구중심(2026-08-05 이전 동작).
  //   2026-08-05 기본값을 true 로 전환. 근거는 아래 estimate() 주석과 cpp/STABILITY_MAP.md.
  bool jac_at_contact = true;
  bool contact_height = true;
  std::vector<int> fgeom; std::vector<double> frad;
  Eigen::Vector3d anchor[2]; bool has_anchor[2]={false,false};   // 접촉 xy 앵커(드리프트 완화)
  // ★★접촉 dwell 게이팅 (2026-08-14 신설). **Python 원본에 있는데 C++ 포팅에서 빠져 있었다.**
  //   deploy/robot_interface.py:63 이 이유를 이미 적어놨다 —
  //     "착지/이지 과도(velocity 스파이크 = 폐루프 붕괴 주범) 제거, 안정 접촉발만 사용"
  //   착지 순간 발은 아직 아래·뒤로 움직이는데 접촉은 이미 잡힌다. 그 샘플을 그대로
  //   "발 정지" 가정에 넣으면 base 속도가 **계통적으로 틀린다**. alpha=0.4 저역통과라
  //   과도 몇 샘플이 계속 섞인다.
  //   증상: 제자리 stepping 60초에 **추정 −0.03 vs 실제 +0.43 m** — 추정기가 전진을
  //   아예 못 본다. 컨트롤러는 그 추정을 믿으니 되잡지 않는다(낙상은 0이라 안 보였다).
  //   ⚠2점 stand 는 앵커가 안 바뀌어 무관하다(추정=GT 1mm). **stepping 전용 문제**다.
  //
  // ⚠**기본 0 = 꺼짐.** 켜면 추정은 맞는데 **낙상이 생긴다**(dwell 5 → 60초에 2회).
  //   원인은 dwell 이 아니라 **발디딤 게인이 이 편향 위에서 맞춰져 있다는 것**이다.
  //   2026-08-05 에 똑같은 일이 있었다(STABILITY_MAP "재튜닝" 절):
  //     "편향만 제거하면 오히려 낙상이 늘었다(vx0.15 에서 1→4회). 그래서 추정기 수정과
  //      발디딤 게인 재튜닝을 **짝으로** 스윕했다."
  //   K_RETURN 만 훑어선(0.10~0.25) 낙상이 안 없어진다(1~4회). T_STEP×K_RETURN×K_CAP
  //   짝 스윕(sweep_stability.sh)이 필요하다. **그 전까지는 켜지 않는다** —
  //   드리프트(낙상 0)보다 낙상이 나쁘기 때문이다.
  //
  //   측정된 효과(제자리 stepping 60초, EST_DWELL=5) — ⚠**노이즈·지연 없는 조건**:
  //     드리프트  GT x  0.43 m → 0.02 m
  //     추정오차        0.46 m → 0.01 m
  //     낙상            0회    → 2회
  //
  // ★★2026-08-19 **위 측정은 조건이 달라서 오해를 만든다. 배포조건에서 재면 정반대다.**
  //   배포조건(EST_CTRL · 지연 8.4ms · 실측 센서노이즈 ENCQ_N/ENCDQ_N) · 8시드 · 40초,
  //   생존 런만 집계:
  //
  //     DWELL   낙상    |GT| 실제드리프트   |est−GT| 추정오차   |est| 추정이 본 것
  //       0     0/8         0.243              0.235             0.020
  //       5     6/8         **0.640**          **0.610**         0.030
  //
  //   ⇒ dwell 은 추정을 **개선하지 않는다 — 악화시킨다**(오차 2.6배). 낙상도 6/8.
  //     이유: dwell 은 속도 표본 수를 줄인다. 노이즈가 없으면 손해가 없지만,
  //     실측 노이즈가 있으면 **평균화할 표본이 줄어** 속도추정이 나빠진다.
  //     ⇒ 노이즈 없는 조건에서 잰 숫자로 이 기능을 판단하면 안 된다.
  //   ⚠게인 짝 스윕(T_STEP×K_RETURN, 189런)으로도 못 살렸다 — DWELL 5/10 은 전 셀에서
  //     DWELL 0 보다 나빴다. **가설 기각.**
  //
  // ★그리고 진짜 문제는 따로 있다 — **DWELL=0 에서도 추정기가 실제 전진을 못 본다.**
  //   추정 0.020 m vs 실제 0.243 m 이고, **추정오차가 곧 드리프트 전부**다.
  //   컨트롤러는 그 추정을 믿으므로(발디딤이 K_RETURN·err_b 를 쓴다) 되잡지 않는다.
  //   낙상은 0 이라 종전 지표로는 전혀 안 보인다. dwell 로 고칠 문제가 아니었다.
  int dwell_steps = 0;             // EST_DWELL 로 켠다. Python deploy 기본은 15
  int dwell[2] = {0, 0};           // 발별 연속 접촉 스텝 수

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
      if(!contacts[k]){ has_anchor[k]=false; dwell[k]=0; continue; }   // 이지=앵커·dwell 리셋
      ++dwell[k];
      int g = fgeom[k];
      Vector3d pfb(ed->geom_xpos[3*g], ed->geom_xpos[3*g+1], ed->geom_xpos[3*g+2]-frad[k]);  // 접촉점(base frame)
      // ★위치는 접촉점(pfb=center−r), 속도는 구 **중심**에서 야코비안 → 불일치.
      //   정지접촉 조건은 "접촉점의 world 속도=0" 이므로 야코비안도 접촉점에서 잡아야 한다.
      //   두 점의 속도차 = ω_발 × (−r·ẑ) 이고 r=0.036 m 라 |ω|≈2.6 rad/s 에서 ~0.09 m/s —
      //   vx 지령 0.15 m/s 의 60% 급이며 스탠스 내내 부호가 일정해 **계통 편향**이 된다.
      //   jac_at_contact=false 로 두면 2026-08-05 이전 동작으로 되돌아간다(회귀 비교용).
  //   EST_JAC_CONTACT=0 env 로도 끌 수 있다.
      mjtNum pnt[3];
      if(jac_at_contact){ pnt[0]=pfb[0]; pnt[1]=pfb[1]; pnt[2]=pfb[2]; }
      else { pnt[0]=ed->geom_xpos[3*g]; pnt[1]=ed->geom_xpos[3*g+1]; pnt[2]=ed->geom_xpos[3*g+2]; }
      mj_jac(m, ed, jb.data(), nullptr, pnt, m->geom_bodyid[g]);
      Vector3d vfb = Vector3d::Zero();                      // 발 속도(관절 기여, base frame)
      for(int r=0;r<3;r++){ double s=0; for(int c=0;c<m->nv;c++) s+=jb[r*m->nv+c]*ed->qvel[c]; vfb[r]=s; }
      Vector3d pfw = R * pfb;                               // 발 오프셋(world)
      // ★★게이팅은 **속도에만** 건다. 깨져 있는 건 "발이 정지" 가정뿐이다.
      //   높이·앵커까지 막으면 안 된다 — 접촉이 잡힌 순간 발은 **이미 지면에 있으므로**
      //   z 관측(p_z = ground_z − pfw_z)은 그때부터 유효하다. 실제로 z 까지 막아봤더니
      //   WBIC 높이 task 가 붕괴해 **60초에 14회 낙상**했다(dwell=15). 헤더 첫머리가
      //   경고한 바로 그 경로다("z 적분 드리프트가 폐루프 실패의 원인이었음").
      if(dwell[k] > dwell_steps)
        vbs.push_back(-(omw.cross(pfw) + R*vfb));           // 발 정지 → base world 속도
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
