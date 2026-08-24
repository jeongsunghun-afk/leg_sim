#pragma once
// sim_hw.hpp — **물리 백엔드**. biped_deploy 를 실기와 **완전히 같은 코드 경로**로
//   MuJoCo 위에서 돌린다. 2026-08-24 신설.
//
// ★왜 만들었나 — 두 반쪽이 겹치는 자리가 비어 있었다:
//     biped_sim      물리는 진짜인데 **모드가 없다**(언제나 WBIC stand/walk 경로).
//                    게다가 d->ctrl 에 토크를 직접 넣어 **드라이버 PD 를 안 거친다** —
//                    STAND_KP_FLOOR·kd 유지가 실제로 어떻게 작용하는지 못 본다.
//     deploy --mock  모드·워치독·트립은 검증되는데 **물리가 없다**(명령 위치로 램프만).
//                    중력·관성·접촉이 없어 블렌드가 안정한지, float 이 실제로 뜨는지 못 본다.
//   ⇒ MockHw 자리에 **진짜 플랜트**를 끼운다. 그러면 biped_deploy 의 1494줄 모드 로직이
//     그대로 물리 위에서 돈다.
//
// ★핵심 설계 — **제어기와 플랜트가 서로 다른 mjModel/mjData 를 쓴다.**
//   biped_sim 은 둘이 같은 m/d 를 공유해서 TORSO_ADD_KG 같은 걸로 **모델↔플랜트 불일치를
//   못 만든다**(둘 다 같이 무거워진다). 여기선 완전히 분리해 α·질량 오차를 진짜로 넣는다.
//
// ★드라이버를 흉내낸다 — 이게 biped_sim 과의 결정적 차이다:
//     τ_ch = kp_ch·(q_des−q)[rad] + kd_ch·(q̇_des−q̇)[rad/s] + τ_ff_ch
//     τ_joint = ch_to_tau_joint(τ_ch)        ← gear_k · 커플링 전치
//     d->ctrl = ALPHA · tau_to_drive(τ_joint)
//   ⇒ gear_k·sign·offset·커플링이 **전부 경로에 들어간다.** 어느 하나가 틀리면 여기서 터진다.
//
// env:
//   SIM_MJCF=<path>     플랜트 모델(기본: 제어기와 같은 파일)
//   SIM_ALPHA=0.9       ★토크 스케일. 실기 미지값을 주입해 처짐을 재현한다
//   SIM_HANG=1          ★크레인에 매단 상태(베이스 용접). float 모드 검증에 필수
//   SIM_MASS_SCALE=1.08 플랜트 링크질량만 배율(제어기 모델은 그대로) — 8% 무거움 재현
//   SIM_TAU_TRUE=1      보고토크를 **실제**로(기본은 실기처럼 **명령 되울림**)
//   SIM_Q0="0,3.68,..."  ★초기 관절자세(모델각 deg, 8개). 기본 0.
//     ⚠**반드시 줄 것.** 전축 0° 로 매달면 다리가 한계까지 자유낙하해 튕기고,
//       그 속도(실측 738dps)가 vel_trip 200 에 즉시 걸린다. 실기도 똑같이 된다 —
//       크레인에 매달 때는 이미 어떤 자세를 잡고 있다. 평발: 0,3.68,-23.87,-59.81 ×2
#include <mujoco/mujoco.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include "deploy_hw.hpp"
#include "biped_wbic.hpp"
#include "biped_control.hpp"

namespace bipedhw {

struct SimHw : HwIface {
  mjModel* m = nullptr;
  mjData*  d = nullptr;
  const JointMap* jm;
  const EmbCfg*   cfg;
  int n, nleg, nu;
  int qa = 7, va = 6;                   // 관절 시작 색인(부동=7/6 · 고정=0/0)
  double dt, ALPHA = 1.0, MASS_SCALE = 1.0;
  bool hang = false, tau_true = false, on = false;
  std::string mjcf;
  // 베이스 용접용 스냅샷
  double base_q[7] = {0,0,0.5, 1,0,0,0};
  // 보고 상태(채널)
  std::vector<float> q_ch, dq_ch, tau_rep;
  std::vector<double> qj, dqj, tauj;
  double t = 0;

  SimHw(const std::string& mjcf_, const JointMap* jm_, const EmbCfg* cfg_, int nch, double dt_)
    : jm(jm_), cfg(cfg_), n(nch), dt(dt_), mjcf(mjcf_) {
    auto ev=[](const char* k, double dv){ const char* v=getenv(k); return v? atof(v) : dv; };
    ALPHA      = ev("SIM_ALPHA", 1.0);
    MASS_SCALE = ev("SIM_MASS_SCALE", 1.0);
    hang       = ev("SIM_HANG", 0) != 0;
    tau_true   = ev("SIM_TAU_TRUE", 0) != 0;
    if(const char* p=getenv("SIM_MJCF")) mjcf = p;
  }
  ~SimHw() override { if(d) mj_deleteData(d); if(m) mj_deleteModel(m); }

  bool init(int) override {
    char err[512] = {0};
    std::string path = mjcf;
    if(hang){
      // ★★**freejoint 를 제거**한다 — 이게 크레인의 정확한 재현이다 (2026-08-24).
      //   ⚠두 번 틀렸다. 먼저 mj_step 뒤 qpos/qvel 을 되돌리는 '사후 용접' 을 썼는데,
      //     스텝 도중엔 베이스가 자유라 부동베이스 동역학이 돌아 관절이 밀렸다.
      //     다음엔 동체 질량을 ×1e6 했는데 **더 틀렸다** — 중력가속도는 질량과 무관해서
      //     아무리 무거워도 **똑같이 자유낙하**한다. 그리고 자유낙하 중인 로봇은
      //     **중력보상이 필요 없다**(무중력). 그래서 τ_ff 를 정확히 넣어도 다리가 떨어졌다.
      //     (파이썬 재현: ctrl=qfrc_bias 로 2초 → 이탈 74.5°(부동) · 95.4°(×1e6))
      //   ⇒ tools/gen_grav_table.py 의 load_fixed_base 와 **같은 방식**으로 간다.
      //     floor 도 뺀다 — 안 빼면 매달린 자세에서 발이 지면에 눌린다.
      std::string xml; { std::ifstream f(mjcf); std::stringstream ss; ss<<f.rdbuf(); xml=ss.str(); }
      if(xml.empty()){ std::printf("[SimHw] ✗ MJCF 읽기 실패: %s\n", mjcf.c_str()); return false; }
      auto drop_tag=[&](const std::string& open){
        size_t i;
        while((i=xml.find(open))!=std::string::npos){
          size_t j=xml.find("/>", i); size_t k=xml.find('>', i);
          size_t e=(j!=std::string::npos && j<k+1)? j+2 : k+1;
          xml.erase(i, e-i); }
      };
      drop_tag("<freejoint");
      { size_t i=xml.find("name=\"floor\"");                 // floor geom 한 줄 제거
        if(i!=std::string::npos){ size_t a=xml.rfind('<', i), b=xml.find("/>", i);
          if(a!=std::string::npos && b!=std::string::npos) xml.erase(a, b+2-a); } }
      // ★원본 **옆에** 써야 한다 — meshdir 이 상대경로라 /tmp 에 쓰면 STL 을 못 찾는다.
      const size_t sl = mjcf.find_last_of('/');
      path = (sl==std::string::npos ? std::string(".") : mjcf.substr(0,sl)) + "/_simhw_fixed.xml";
      { std::ofstream o(path); o<<xml; }
      qa = 0; va = 0;
    }
    m = mj_loadXML(path.c_str(), nullptr, err, sizeof err);
    if(!m){ std::printf("[SimHw] ✗ 플랜트 모델 로드 실패: %s\n", err); return false; }
    d = mj_makeData(m);
    nleg = jm->n_leg; nu = m->nu;
    // ★★**플랜트에도 기어박스를 주입한다** — 이게 없으면 물리가 통째로 틀린다.
    //   MJCF 는 armature/damping/frictionloss 를 **의도적으로 비워** 두고(주석 12~14행)
    //   런타임 setup_gearbox 가 넣는다. 그런데 그건 **제어기 모델에만** 걸린다.
    //   ⇒ 플랜트가 raw MJCF 면 반사관성이 0 이라 발목 관성이 **25배 작아지고**
    //     (I_link 0.0021 vs +tendon_armature 0.0517) 같은 게인이 발산한다.
    //     실제로 그랬다: hold 조차 못 버티고 ch7 이 738dps 로 속도트립에 걸렸다.
    //   ⚠**BipedControl 을 하나 더 만들어 그 함수를 그대로 부른다** — 값을 복사하지 않는다.
    //     복사하면 ROTOR_I·JDAMP·JFRIC 이 두 곳에 생겨 반드시 갈라진다.
    { BipedControl gb(m, d); gb.setup_gearbox(); }
    // ★★크레인 = **동체를 무겁게** 만든다 (2026-08-24).
    //   ⚠처음엔 mj_step 뒤에 qpos/qvel 을 되돌리는 '사후 용접' 을 썼는데 **틀렸다**:
    //     스텝 도중에는 베이스가 자유라 **부동베이스 동역학**이 돌고, 다리 반작용이
    //     베이스를 밀어 관절 가속도가 0 이 안 된다. 중력보상을 정확히 넣어도 다리가 떨어졌다.
    //     (실측: τ_ff·τ_joint·d->ctrl 이 전부 정확한데 dq 가 0→2.6→5.2→7.8dps 로 발산)
    //   ⇒ 동체 관성을 크게 해 **물리적으로** 고정한다. 관절 중력토크는 그 관절보다
    //     **아래(distal)** 링크 질량만으로 정해지므로 이 조작이 그 값을 안 바꾼다.

    q_ch.assign(n,0.f); dq_ch.assign(n,0.f); tau_rep.assign(n,0.f);
    qj.assign(nleg,0.0); dqj.assign(nleg,0.0); tauj.assign(nleg,0.0);
    // ★플랜트 질량만 키운다 — 제어기 모델은 안 건드린다. 이게 "모델이 8% 가볍다" 의 재현이다.
    if(MASS_SCALE != 1.0){
      for(int b=1;b<m->nbody;b++){
        m->body_mass[b] *= MASS_SCALE;
        for(int k=0;k<6;k++) m->body_inertia[b*3 + (k%3)] *= MASS_SCALE;
      }
      mj_setConst(m, d);
    }
    mj_resetData(m, d);
    if(!hang) for(int i=0;i<7 && i<m->nq;i++) d->qpos[i] = base_q[i];
    // ★초기 관절자세 — 안 주면 전축 0° 에서 자유낙하해 한계에 튕긴다(속도트립 발화).
    std::string q0s;
    if(const char* z=getenv("SIM_Q0")) q0s = z;
    if(!q0s.empty()){
      std::vector<double> v; std::string t; 
      for(char ch : q0s + ","){ if(ch==','){ if(!t.empty()){ v.push_back(atof(t.c_str())); t.clear(); } } else t+=ch; }
      for(int j=0;j<nleg && j<(int)v.size();j++) d->qpos[qa+j] = v[j] * JointMap::D2R;
      std::printf("[SimHw] 초기 자세 SIM_Q0 = %s (모델각 deg)\n", q0s.c_str());
    } else {
      std::printf("[SimHw] ⚠SIM_Q0 미지정 — 전축 0° 에서 시작한다. 매달림이면 다리가 한계까지\n"
                  "        떨어져 **속도트립이 발화한다**. 평발: SIM_Q0=\"0,3.68,-23.87,-59.81,0,3.68,-23.87,-59.81\"\n");
    }
    mj_forward(m, d);
    std::printf("[SimHw] **물리 백엔드** — %s (nq=%d nv=%d nu=%d)\n"
                "        α=%.3f · 질량×%.3f · %s · 보고토크=%s\n"
                "        ⚠제어기와 **다른 mjModel** 이다 — 모델↔플랜트 불일치를 진짜로 만든다.\n",
                mjcf.c_str(), m->nq, m->nv, m->nu, ALPHA, MASS_SCALE,
                hang ? "**매달림(베이스 용접)**" : "자유 베이스(접지)",
                tau_true ? "실제(SIM_TAU_TRUE)" : "명령 되울림(실기와 동일)");
    return true;
  }

  // 플랜트 상태 → 채널 보고값
  void sync_report(){
    for(int j=0;j<nleg;j++){ qj[j] = d->qpos[qa+j]; dqj[j] = d->qvel[va+j]; }
    jm->q_ctrl_to_ch(qj.data(), q_ch.data());
    jm->dq_ctrl_to_ch(dqj.data(), dq_ch.data());
  }

  int read(HwState& s) override {
    sync_report();
    s.q_deg = q_ch; s.dq_dps = dq_ch; s.tau_nm = tau_rep;
    // ★fCurrent = fTorque 중복을 그대로 재현한다(RESULTS.md:465). 독립 전류는 없다.
    s.cur_a = tau_rep;
    s.connected.assign(n,1); s.status.assign(n,0);
    // IMU — 실기가 죽어 있으므로 기본은 죽은 채로 둔다(SIM_IMU=1 이면 살린다).
    const bool imu = getenv("SIM_IMU") != nullptr;
    s.rpy[0]=s.rpy[1]=s.rpy[2]=0; s.gyr[0]=s.gyr[1]=s.gyr[2]=0;
    s.acc[0]=s.acc[1]=0; s.acc[2]= imu ? 9.81f : 0.f;
    if(imu && !hang){
      // 자유 베이스면 실제 자세를 준다(쿼터니언 → rpy, ZYX)
      const double w=d->qpos[3], x=d->qpos[4], y=d->qpos[5], z=d->qpos[6];
      s.rpy[0]=(float)std::atan2(2*(w*x+y*z), 1-2*(x*x+y*y));
      s.rpy[1]=(float)std::asin(std::max(-1.0,std::min(1.0,2*(w*y-z*x))));
      s.rpy[2]=(float)std::atan2(2*(w*z+x*y), 1-2*(y*y+z*z));
      for(int k=0;k<3;k++) s.gyr[k]=(float)d->qvel[3+k];
    }
    s.mask = 1 | 16;
    return s.mask;
  }

  int write_pos(const float* q, const float* kp, const float* kd, int nn) override {
    return write_mit(q, nullptr, nullptr, kp, kd, nn);
  }

  int write_mit(const float* q_des, const float* dq_des, const float* tau_ff,
                const float* kp, const float* kd, int nn) override {
    if(nn > n) nn = n;
    sync_report();
    // ★★드라이버 MIT 법칙을 **채널 좌표**에서 그대로 실행한다.
    //   kp/kd 는 Nm/rad 인데 각도는 deg 로 오므로 D2R 을 곱한다(shm_bridge 와 같은 규약).
    std::vector<float> tcmd(n, 0.f);
    for(int i=0;i<nn;i++){
      if(!on){ tcmd[i] = 0.f; continue; }
      const double e  = ((double)q_des[i] - (double)q_ch[i]) * JointMap::D2R;
      const double de = (((dq_des ? (double)dq_des[i] : 0.0) - (double)dq_ch[i])) * JointMap::D2R;
      tcmd[i] = (float)( (double)kp[i]*e + (double)kd[i]*de + (tau_ff ? (double)tau_ff[i] : 0.0) );
    }
    tau_rep = tcmd;                       // ★보고는 **명령 되울림**(실기와 동일)
    // 채널토크 → 관절토크 → 드라이브(액추에이터). gear_k·커플링 전치가 여기서 걸린다.
    jm->ch_to_tau_joint(tcmd.data(), tauj.data());
    Eigen::VectorXd tj(nleg);
    for(int j=0;j<nleg;j++) tj[j] = tauj[j];
    Eigen::VectorXd u = bipedwbic::tau_to_drive(tj);
    for(int i=0;i<nu && i<nleg;i++){
      double v = ALPHA * u[i];            // ★토크 스케일 — 실기 미지값 주입
      const double lim = m->actuator_ctrllimited[i] ? m->actuator_ctrlrange[i*2+1] : 1e9;
      d->ctrl[i] = std::max(-lim, std::min(lim, v));
    }
    if(getenv("SIM_DBG")){
      static int np2=0;
      double ffmax=0; if(tau_ff) for(int i=0;i<8;i++) ffmax=std::max(ffmax,(double)std::fabs(tau_ff[i]));
      if(on && ffmax>0.1 && np2<4){ np2++;      // ★τ_ff 가 실린 틱만 = float/stand
        std::printf("[SimHw] on=%d  tau_ff_ch:", (int)on);
        for(int i=0;i<8;i++) std::printf(" %+.2f",(double)(tau_ff?tau_ff[i]:0.f));
        std::printf("\n        tcmd_ch  :");
        for(int i=0;i<8;i++) std::printf(" %+.2f",(double)tcmd[i]);
        std::printf("\n        tau_joint:");
        for(int j=0;j<nleg;j++) std::printf(" %+.2f",tauj[j]);
        std::printf("\n        d->ctrl  :");
        for(int i=0;i<nu;i++) std::printf(" %+.2f",d->ctrl[i]);
        std::printf("\n"); std::fflush(stdout); }
    }
    mj_step(m, d);
    t += m->opt.timestep;

    if(tau_true){                         // 실제 관절토크를 보고(비교용)
      std::vector<double> ta(nleg,0.0);
      for(int j=0;j<nleg;j++) ta[j] = d->qfrc_actuator[va+j];
      jm->tau_ctrl_to_ch(ta.data(), tau_rep.data());
    }
    return 0;
  }

  int enable(int o) override { on = (o!=0); return 0; }
  const char* name() const override { return "sim"; }
};

} // namespace bipedhw
