// biped_deploy.cpp — ★C++ 실기 배포 (§9, 핸드오프 미완료 #4).
//
//   biped_sim.cpp 의 `mj_step` 자리에 **실모터 read/write** 를 넣은 것이다.
//   데이터흐름:  HW.read → 관절매핑(deg→rad) → 추정(leg-odom) → 모델 주입 → mj_forward
//              → BipedControl.control → d->ctrl(토크) → 관절매핑(rad→deg) → HW.write_mit
//
//   모드: off / hold / stand / walk       (jog·home 은 Python 앱 emb/app/biped_emb.py 담당)
//   ⚠ **모터 명령 writer 는 한 번에 하나만.** 이 바이너리를 돌릴 땐 biped_emb.py 를 끌 것.
//
//   실행:
//     ./biped_deploy                       # 실기(SHM). Emb 기동 후 5초 뒤
//     ./biped_deploy --mock                # 하드웨어 없이 로직 검증
//     ./biped_deploy --mjcf ../biped_from_quad.mjcf --T 30
//
// ★안전장치는 전부 emb/app/biped_emb.py 에서 이식했다. 실측으로 다듬어진 것들이라
//   재발명하지 않는다 — 워치독 · tilt/토크/속도 E-stop(래치) · 종료 시 limp 반복기록.
#include <mujoco/mujoco.h>
#include "biped_control.hpp"
#include "state_estimator.hpp"
#include "deploy_hw.hpp"
#include <Eigen/Dense>
#include <csignal>
#include <ctime>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace bipedhw;

static volatile std::sig_atomic_t g_stop = 0;
static void on_sig(int s){ g_stop = s; }

static double now_s(){
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec*1e-9;
}
static void sleep_s(double s){
  if(s <= 0) return;
  struct timespec ts{ (time_t)s, (long)((s-(double)(time_t)s)*1e9) };
  nanosleep(&ts, nullptr);
}

// RPY(rad) → quat(wxyz). hw_interface.rpy_to_quat 과 동일(ZYX).
static void rpy_to_quat(double r, double p, double y, double* q){
  double cr=std::cos(r/2), sr=std::sin(r/2), cp=std::cos(p/2), sp=std::sin(p/2),
         cy=std::cos(y/2), sy=std::sin(y/2);
  q[0]=cr*cp*cy + sr*sp*sy; q[1]=sr*cp*cy - cr*sp*sy;
  q[2]=cr*sp*cy + sr*cp*sy; q[3]=cr*cp*sy - sr*sp*cy;
}

// ★어떤 경로로 죽어도 **실제로** 무여자로 만든다.
//   bridge_enable(0) 만으로는 정지가 아니다 — g_enabled 플래그만 바꿀 뿐 SHM 명령버퍼는
//   그대로라, Emb 가 마지막 명령을 1kHz 로 영원히 재전송한다(shm_bridge.cpp:115).
//   Kp=Kd=0 을 **반복 기록**해야 진짜 limp 다. (biped_emb.safe_shutdown 과 같은 이유)
static void safe_shutdown(HwIface& hw, int n){
  std::vector<float> z(n,0.f), q(n,0.f);
  int ok=0;
  hw.enable(0);
  for(int i=0;i<25;i++){
    if(hw.write_pos(q.data(), z.data(), z.data(), n)==0) ok++;
    sleep_s(0.002);
  }
  if(ok==0){
    std::fprintf(stderr,
      "\n%s\n!! limp 실패 — SHM 에 무여자 명령을 한 번도 쓰지 못했다."
      "\n!! Emb 는 마지막 명령을 1kHz 로 계속 재전송한다. **모터 전원을 차단할 것**.\n%s\n",
      std::string(68,'!').c_str(), std::string(68,'!').c_str());
  } else {
    std::printf("[deploy] 종료 — 무여자(Kp=Kd=0) 명령 %d/25 회 기록 완료.\n", ok);
  }
}

int main(int argc, char** argv){
  std::string mjcf   = "../biped_from_quad.mjcf";     // ★배포는 점발(1pt). §8-g 참조
  std::string cfg_p  = "../emb/config/biped_emb.yaml";
  std::string cmd_p  = "/tmp/biped_cmd.json";
  std::string stt_p  = "/tmp/biped_state.json";
  bool mock = false; double T = 1e12;
  for(int i=1;i<argc;i++){
    std::string a = argv[i];
    if(a=="--mock") mock = true;
    else if(a=="--mjcf" && i+1<argc) mjcf = argv[++i];
    else if(a=="--config" && i+1<argc) cfg_p = argv[++i];
    else if(a=="--cmd" && i+1<argc) cmd_p = argv[++i];
    else if(a=="--state" && i+1<argc) stt_p = argv[++i];
    else if(a=="--T" && i+1<argc) T = atof(argv[++i]);
    else { std::printf("사용법: %s [--mock] [--mjcf X] [--config X] [--cmd X] [--state X] [--T s]\n", argv[0]); return 2; }
  }
  if(const char* e=getenv("QUAD_CMD"))   cmd_p = e;
  if(const char* e=getenv("QUAD_STATE")) stt_p = e;

  // ── 설정 ──
  EmbCfg cfg; std::string err;
  if(!load_cfg(cfg_p, cfg, err)){ std::printf("✗ %s\n", err.c_str()); return 1; }
  const int NCH = cfg.n_channel;
  const double dt = 1.0/cfg.ctrl_hz;
  JointMap jm(cfg);
  std::printf("[deploy] config=%s · joints=%d · n_channel=%d · ctrl_hz=%.0f\n",
              cfg_p.c_str(), jm.n_leg, NCH, cfg.ctrl_hz);
  { std::string s; for(auto& j : cfg.joints){
      char b[96]; std::snprintf(b,sizeof b," %s(ch%d,s%+.0f,off%+.1f)", j.name.c_str(), j.channel, j.sign, j.offset_deg);
      s += b; }
    std::printf("[deploy] 매핑:%s\n", s.c_str()); }
  if(!cfg.installed.empty()){
    std::string s; for(int ch : cfg.installed){ char b[8]; std::snprintf(b,sizeof b," %d",ch); s+=b; }
    std::printf("[deploy] 실장 채널:%s (나머지는 명령은 나가지만 통신 없어 무효)\n", s.c_str());
  }

  // ── 모델 + 컨트롤러 ──
  char merr[1000]={0};
  mjModel* m = mj_loadXML(mjcf.c_str(), nullptr, merr, 1000);
  if(!m){ std::printf("✗ 모델 로드 실패(%s): %s\n", mjcf.c_str(), merr); return 1; }
  mjData* d = mj_makeData(m);
  BipedControl c(m,d); c.reset();
  const int NJ = m->nq - 7, NU = m->nu;
  if(NJ != jm.n_leg){
    std::printf("✗ 모델 관절수(%d) ≠ config joints(%d) — 모델/설정 불일치\n", NJ, jm.n_leg);
    mj_deleteData(d); mj_deleteModel(m); return 1;
  }
  std::printf("[deploy] 모델=%s (nq=%d nv=%d nu=%d, 점발 1pt) · cmode=%d\n",
              mjcf.c_str(), m->nq, m->nv, m->nu, c.cmode);

  BipedEstimator est;
  { std::vector<int> fg={c.sph[0],c.sph[1]};
    std::vector<double> fr={m->geom_size[c.sph[0]*3], m->geom_size[c.sph[1]*3]};
    est.init(m,fg,fr); est.reset(Eigen::Vector3d(0,0,d->qpos[2])); }

  // ── 하드웨어 ──
  HwIface* hw = mock ? (HwIface*)new MockHw(NCH, dt)
                     : (HwIface*)new ShmHw(cfg.lib, NCH);
  if(!hw->init(cfg.recv_wait_ms)){
    auto* sh = dynamic_cast<ShmHw*>(hw);
    std::printf("✗ 하드웨어 초기화 실패\n  %s\n", sh? sh->err.c_str() : "?");
    delete hw; mj_deleteData(d); mj_deleteModel(m); return 1;
  }
  std::printf("[deploy] backend=%s\n", hw->name());

  for(int s : {SIGINT,SIGTERM,SIGHUP,SIGQUIT}) std::signal(s, on_sig);

  // ── 상태 ──
  std::vector<float> q_ch(NCH), dq_ch(NCH), tau_ch(NCH), kp_ch(NCH), kd_ch(NCH), zero(NCH,0.f);
  std::vector<double> q_ctrl(NJ), dq_ctrl(NJ), tau_ctrl(NU);
  std::vector<float> hold_ch(NCH, 0.f);
  HwState hs;
  std::string mode = "off", prev_mode = "off", last_raw;
  bool estop = false, wd_tripped = false;
  double tau_over_t0 = -1, last_cmd_t = now_s(), last_pub = 0, hz_ema = cfg.ctrl_hz;
  Cmd cmd; double body_h = 0.5;
  const double watchdog_s = cfg.watchdog_ms/1000.0;

  hw->enable(0);
  std::printf("[deploy] 모드: off/hold/stand/walk. GUI 로 조종(%s).\n", cmd_p.c_str());
  std::printf("[deploy] ⚠ jog·home 은 Python 앱 담당. writer 는 한 번에 하나만.\n");

  double t0 = now_s(), prev_loop = t0; long long k = 0;
  int rc = 0;
  while(!g_stop && (now_s()-t0) < T){
    double lt = now_s();

    // ① 센서
    hw->read(hs);
    jm.ch_to_q_ctrl(hs.q_deg.data(),  q_ctrl.data());
    jm.ch_to_dq_ctrl(hs.dq_dps.data(), dq_ctrl.data());
    const double D2R = JointMap::D2R;
    double rpy[3] = { hs.rpy[0]*(cfg.imu_deg?D2R:1.0),
                      hs.rpy[1]*(cfg.imu_deg?D2R:1.0),
                      hs.rpy[2]*(cfg.imu_deg?D2R:1.0) };
    double gyro[3]= { hs.gyr[0]*(cfg.imu_deg?D2R:1.0),
                      hs.gyr[1]*(cfg.imu_deg?D2R:1.0),
                      hs.gyr[2]*(cfg.imu_deg?D2R:1.0) };
    double quat[4]; rpy_to_quat(rpy[0],rpy[1],rpy[2],quat);
    double tilt = std::hypot(rpy[0],rpy[1]) * JointMap::R2D;

    // ② 명령 폴링(~50Hz)
    if(k % (long long)std::max(1.0, 0.02/dt) == 0){
      Cmd nc;
      if(read_cmd(cmd_p, nc)){
        bool fresh = (nc.raw != last_raw);          // ★"파일이 읽히는가"가 아니라 "내용이 바뀌는가"
        last_raw = nc.raw;                          //   (biped_emb.read_cmd_fresh 와 같은 이유)
        if(fresh) last_cmd_t = lt;
        cmd = nc; body_h = nc.body_h;
        std::string nm = nc.mode;
        if(nm=="reset") nm = "hold";
        if(nm!="off" && nm!="hold" && nm!="stand" && nm!="walk") nm = "off";   // jog/home 등 → off
        // ★E-stop 래치는 명시적 off 로만 해제. 그 전까지 모드변경 무시.
        if(estop){
          if(nm=="off"){ estop=false; std::printf("[deploy] E-stop 래치 해제(off 수신) — 재무장 가능\n"); }
          else nm = "off";
        }
        if(nm != mode){
          prev_mode = mode; mode = nm;
          hw->enable(mode=="off" ? 0 : 1);
          if(mode=="hold"){ hold_ch = hs.q_deg; jm.clamp_ch(hold_ch.data()); }
          if(mode=="stand" || mode=="walk"){
            c.reset(); c.com_ref_z = body_h;
            est.reset(Eigen::Vector3d(0,0,d->qpos[2]));
          }
          std::printf("[deploy] 모드 %s → %s\n", prev_mode.c_str(), mode.c_str());
        }
        c.vx_cmd = (mode=="walk") ? cmd.v  : 0.0;
        c.vy_cmd = (mode=="walk") ? cmd.vy : 0.0;
        c.wz_cmd = (mode=="walk") ? cmd.w  : 0.0;
      }
    }

    // ③ 워치독 — 명령 두절이면 limp. 전이를 반드시 출력한다(데드코드 방지).
    bool wd = (mode!="off") && (lt - last_cmd_t) > watchdog_s;
    if(wd != wd_tripped){
      wd_tripped = wd;
      std::printf(wd ? "[deploy] 워치독 트립 — 명령 두절 %.2fs > %.2fs → limp\n"
                     : "[deploy] 워치독 해제 — 명령 복귀 (%.2f/%.2f)\n",
                  lt-last_cmd_t, watchdog_s);
      std::fflush(stdout);
    }
    if(mode!="off") hw->enable(wd ? 0 : 1);

    // ④ E-stop (tilt / 토크 / 속도) — 전부 래치. biped_emb.py 이식.
    if(!estop && mode!="off" && tilt > cfg.tilt_estop_deg){
      std::printf("[deploy] ⛔ E-STOP: tilt %.0f° > %.0f° → limp·래치\n", tilt, cfg.tilt_estop_deg);
      estop = true; mode = "off"; hw->enable(0);
    }
    if(!estop && mode!="off"){
      double tau_pk=0, vel_pk=0; int tch=0, vch=0;
      for(int i=0;i<NCH;i++){
        if(std::fabs(hs.tau_nm[i])>tau_pk){ tau_pk=std::fabs(hs.tau_nm[i]); tch=i; }
        if(std::fabs(hs.dq_dps[i])>vel_pk){ vel_pk=std::fabs(hs.dq_dps[i]); vch=i; }
      }
      // 토크는 **연속 초과**만 트립(착지 충격 같은 순간 스파이크를 살린다)
      if(tau_pk > cfg.tau_trip_nm){
        if(tau_over_t0 < 0) tau_over_t0 = lt;
        else if((lt-tau_over_t0)*1000.0 >= cfg.tau_trip_ms){
          std::printf("[deploy] ⛔ E-STOP: ch%d 토크 %.2fNm > %.2fNm 가 %.0fms 연속 → limp·래치\n",
                      tch, tau_pk, cfg.tau_trip_nm, cfg.tau_trip_ms);
          estop = true; mode="off"; hw->enable(0);
        }
      } else tau_over_t0 = -1;
      // 속도는 즉시 트립(폭주를 지연시킬 이유가 없다)
      if(!estop && vel_pk > cfg.vel_trip_dps){
        std::printf("[deploy] ⛔ E-STOP: ch%d 속도 %.0fdps > %.0fdps → limp·래치\n",
                    vch, vel_pk, cfg.vel_trip_dps);
        estop = true; mode="off"; hw->enable(0);
      }
    } else tau_over_t0 = -1;
    if(estop) hw->enable(0);

    // ⑤ 모드 디스패치
    if(mode=="off"){
      for(int i=0;i<NCH;i++) q_ch[i]=0.f;
      hw->write_pos(q_ch.data(), zero.data(), zero.data(), NCH);     // enable=0 → 브리지가 0 토크
    } else if(mode=="hold"){
      jm.kp_ch(kp_ch.data()); jm.kd_ch(kd_ch.data());
      hw->write_pos(hold_ch.data(), kp_ch.data(), kd_ch.data(), NCH);
    } else {  // stand / walk — 모델기반
      // 접촉: 실기엔 발 힘센서가 없다. 게이트 위상(스탠스 다리)을 접촉으로 쓴다.
      //   ⚠추정에 쓰는 접촉이 제어기 자신의 계획이라 순환처럼 보이지만, 힘센서 없는
      //     운동학 오도메트리의 표준 관행이다. 힘센서가 생기면 여기를 교체할 것.
      std::vector<bool> cts(2,false);
      if(c.cmode==1){ cts[0]=cts[1]=true; }               // 평발 정적 = 양발
      else { cts[c.swing==0?1:0] = true; }                // 점발 = 스탠스 다리만
      est.estimate(m, q_ctrl.data(), dq_ctrl.data(), quat, gyro, cts, dt);

      // 모델에 주입 → mj_forward → 제어
      d->qpos[0]=est.p[0]; d->qpos[1]=est.p[1]; d->qpos[2]=est.p[2];
      for(int a=0;a<4;a++) d->qpos[3+a]=quat[a];
      for(int j=0;j<NJ;j++) d->qpos[7+j]=q_ctrl[j];
      d->qvel[0]=est.v[0]; d->qvel[1]=est.v[1]; d->qvel[2]=est.v[2];
      for(int a=0;a<3;a++) d->qvel[3+a]=gyro[a];
      for(int j=0;j<NJ;j++) d->qvel[6+j]=dq_ctrl[j];
      mj_forward(m,d);
      c.com_ref_z = body_h;
      c.control(dt);

      // 토크 → 채널. ★tau_max_frac 로 한 번 더 클램프(컨트롤러 내부 클립과 별개의 상위 안전망).
      for(int i=0;i<NU;i++){
        double lim = m->jnt_actfrcrange[(i+1)*2+1] * cfg.tau_max_frac;   // freejoint 다음이 액추에이터 관절
        if(lim<=0) lim = 80.0;
        tau_ctrl[i] = std::max(-lim, std::min(lim, d->ctrl[i]));
      }
      jm.q_ctrl_to_ch(q_ctrl.data(), q_ch.data());        // 위치/속도는 참고값(kp=kd=0 이라 무영향)
      jm.dq_ctrl_to_ch(dq_ctrl.data(), dq_ch.data());
      jm.tau_ctrl_to_ch(tau_ctrl.data(), tau_ch.data());
      // ★순수 토크: kp=kd=0. 드라이버가 tau_ff 만 실행한다.
      hw->write_mit(q_ch.data(), dq_ch.data(), tau_ch.data(), zero.data(), zero.data(), NCH);
    }

    // ⑥ 상태 발행(~20Hz)
    double period = lt - prev_loop; prev_loop = lt;
    if(period > 0) hz_ema = 0.98*hz_ema + 0.02*(1.0/period);
    if(lt - last_pub > 0.05){
      last_pub = lt;
      int n_ok=0, n_fault=0, n_dead=0, n_absent=0;
      std::string health="[", inst="[", qs="[";
      for(int i=0;i<jm.n_leg;i++){
        int ch = cfg.joints[i].channel;
        bool ins = cfg.installed_has(ch);
        const char* h = !ins ? "absent"
                      : (ch>=(int)hs.connected.size() || !hs.connected[ch]) ? "dead"
                      : (ch<(int)hs.status.size() && hs.status[ch]!=0) ? "fault" : "ok";
        if(!ins) n_absent++; else if(!std::strcmp(h,"ok")) n_ok++;
        else if(!std::strcmp(h,"fault")) n_fault++; else n_dead++;
        char b[64];
        std::snprintf(b,sizeof b,"%s\"%s\"", i?",":"", h); health += b;
        std::snprintf(b,sizeof b,"%s%s", i?",":"", ins?"true":"false"); inst += b;
        std::snprintf(b,sizeof b,"%s%.2f", i?",":"", hs.q_deg[ch]); qs += b;
      }
      health+="]"; inst+="]"; qs+="]";
      char buf[1600];
      std::snprintf(buf,sizeof buf,
        "{\"mode\":\"%s\",\"backend\":\"%s\",\"q_leg_deg\":%s,"
        "\"rpy_deg\":[%.2f,%.2f,%.2f],\"tilt_deg\":%.2f,\"loop_hz\":%.1f,"
        "\"motors_on\":%s,\"health\":%s,\"installed\":%s,"
        "\"n_ok\":%d,\"n_fault\":%d,\"n_dead\":%d,\"n_absent\":%d,\"n_installed\":%d,"
        "\"est_x\":%.3f,\"est_z\":%.3f,\"estop\":%s}",
        mode.c_str(), hw->name(), qs.c_str(), rpy[0]*JointMap::R2D, rpy[1]*JointMap::R2D,
        rpy[2]*JointMap::R2D, tilt, hz_ema, (mode!="off"&&!wd)?"true":"false",
        health.c_str(), inst.c_str(), n_ok, n_fault, n_dead, n_absent,
        (int)(jm.n_leg-n_absent), est.p[0], est.p[2], estop?"true":"false");
      write_state(stt_p, buf);
    }

    // ⑦ 실시간 페이싱(절대시각 — 누적 드리프트 없음)
    k++;
    sleep_s(t0 + k*dt - now_s());
  }

  if(g_stop) std::printf("\n[deploy] 신호 %d 수신 → 안전종료\n", (int)g_stop);
  safe_shutdown(*hw, NCH);
  delete hw; mj_deleteData(d); mj_deleteModel(m);
  return rc;
}
