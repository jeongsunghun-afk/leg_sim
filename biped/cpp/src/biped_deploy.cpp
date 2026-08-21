// biped_deploy.cpp — ★C++ 실기 배포 (§9, 핸드오프 미완료 #4).
//
//   biped_sim.cpp 의 `mj_step` 자리에 **실모터 read/write** 를 넣은 것이다.
//   데이터흐름:  HW.read → 관절매핑(deg→rad) → 추정(leg-odom) → 모델 주입 → mj_forward
//              → BipedControl.control → d->ctrl(토크) → 관절매핑(rad→deg) → HW.write_mit
//
//   모드: off / hold / home / jog / stand / walk
//     jog·home 은 위치제어(kp/kd, τ_ff 없음) · stand/walk 는 WBIC 토크(+kd 플로어).
//     emb/app/biped_emb.py 와 **같은 config 키**를 읽는다(jog.max_speed_dps / home.*).
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
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <sched.h>
#include <sys/mman.h>
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
    // ⚠"무여자" 라 쓰지 않는다 — SHM 에 드라이브 disable 경로가 **없다**(벤더 확인 2026-08-14).
    //   보장되는 건 명령토크 0 까지고, 드라이브는 여전히 여자 상태다.
    std::printf("[deploy] ⚠**체중을 받고 있었다면 지금 주저앉는다** — 무여자는 낙하다.\n"
                "         다시 세우려면: 크레인으로 들어올림 → home → 접지 → hold → stand\n");
    std::printf("[deploy] 종료 — 명령토크 0(Kp=Kd=τ=0) %d/25 회 기록 완료.\n"
                "         ⚠드라이브는 여전히 여자 상태다 — 축이 안 풀리면 물리 리셋/전원 차단.\n", ok);
  }
}

int main(int argc, char** argv){
  std::string mjcf   = "../biped_from_quad.mjcf";     // ★배포는 점발(1pt). §8-g 참조
  std::string cfg_p  = "../emb/config/biped_emb.yaml";
  std::string cmd_p  = "/tmp/biped_cmd.json";
  std::string stt_p  = "/tmp/biped_state.json";
  bool mock = false; double T = 1e12; std::string start_mode = "off";
  for(int i=1;i<argc;i++){
    std::string a = argv[i];
    if(a=="--mock") mock = true;
    else if(a=="--mjcf" && i+1<argc) mjcf = argv[++i];
    else if(a=="--config" && i+1<argc) cfg_p = argv[++i];
    else if(a=="--cmd" && i+1<argc) cmd_p = argv[++i];
    else if(a=="--state" && i+1<argc) stt_p = argv[++i];
    else if(a=="--T" && i+1<argc) T = atof(argv[++i]);
    // ★biped_emb.py 와 **같은 이름의 옵션**을 둔다 — 두 제어기의 기동 규약을 맞춘다.
    //   off  = 무여자로 시작(기본·권장). 명령파일에 뭐가 남아 있든 무시한다.
    //   hold = 생존확인 통과 후 **현재 자세를 그대로** 잡는다(로봇이 정지해 있어야 한다).
    else if(a=="--start-mode" && i+1<argc) start_mode = argv[++i];
    else { std::printf("사용법: %s [--mock] [--mjcf X] [--config X] [--cmd X] [--state X] [--T s]\n"
                       "        [--start-mode off|hold]   기본 off (기동 즉시 무장 방지)\n", argv[0]); return 2; }
  }
  if(start_mode!="off" && start_mode!="hold"){   // start_mode 검증
    std::printf("✗ --start-mode 는 off 또는 hold 다 (받은 값: %s)\n", start_mode.c_str()); return 2; }
  if(const char* e=getenv("QUAD_CMD"))   cmd_p = e;
  if(const char* e=getenv("QUAD_STATE")) stt_p = e;

  // ★★실시간 우선순위 (2026-08-20). 500Hz 루프가 일반 우선순위로 돌면 밀린다 —
  //   실측: 28~51ms 스톨(20~25틱 유실). 이 Pi 는 RobotEmbedded 가 1kHz 로 CPU 90% 를
  //   쓰고 데스크톱(gnome-shell·Xwayland·모니터)까지 얹혀 있어 경합이 실재한다.
  //   ⚠기본 사용자는 rtprio 한도가 0 이라 실패한다 — 그때는 경고만 하고 계속 돈다.
  //     sudo setcap cap_sys_nice+ep <바이너리>
  //     ⚠**재빌드하면 사라진다**(재링크가 파일을 새로 만든다). 실측 확인했다.
  //   ⚠SCHED_FIFO 는 이 스레드가 CPU 를 독점할 수 있다는 뜻이다. 500Hz 루프는
  //     대부분 sleep 이라 안전하지만, 무한루프 버그가 나면 기기가 멎는다.
  { const int rp = getenv("RT_PRIO") ? atoi(getenv("RT_PRIO")) : 80;
    if(rp > 0){
      struct sched_param sp; sp.sched_priority = rp;
      if(sched_setscheduler(0, SCHED_FIFO, &sp) == 0){
        std::printf("[deploy] 실시간 우선순위 **SCHED_FIFO %d** 적용\n", rp);
        if(mlockall(MCL_CURRENT|MCL_FUTURE) != 0)
          std::printf("[deploy] ⚠mlockall 실패 — 페이지폴트로 지터가 남을 수 있다\n");
      } else {
        std::printf("[deploy] ⚠실시간 우선순위 실패(%s) — **루프가 밀릴 수 있다**(실측 28~51ms 스톨).\n"
                    "         sudo setcap cap_sys_nice+ep %s\n"
                    "         ⚠**재빌드(재링크)하면 사라진다** — 소스를 고칠 때마다 다시 걸 것.\n"
                    "         끄려면 RT_PRIO=0\n", std::strerror(errno), argv[0]);
      } } }

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
  // ★"점발 1pt" 하드코딩이었다(2026-08-13 수정). 평발 MJCF 를 줘도 그렇게 찍혀서,
  //   운전자가 이 줄로 무엇이 올라왔는지 확인하는데 **거짓말을 하고 있었다.**
  //   cmode 는 heel 구 유무로 자동 결정된다(BipedControl: cmode = has_heel ? 1 : 0).
  std::printf("[deploy] 모델=%s (nq=%d nv=%d nu=%d) · cmode=%d **%s**\n",
              mjcf.c_str(), (int)m->nq, (int)m->nv, (int)m->nu, c.cmode,   // ★mjtSize=long → %d 경고
              c.cmode==1 ? "2점 평발(정적 자세유지)" : "1점 점발(stepping 보행)");

  // ★마찰 전방보상 상태를 **반드시 찍는다**. 켜져 있어도 꺼져 있어도 겉보기 동작이
  //   같아서, 지연보상 때 "설정하면 켜진다" 고 9일간 오해했던 것과 같은 함정이다.
  //   2점 평발에서만 켜진다 — 이 값이 20.73s 낙상을 60s 무낙상(tilt 0.1°)으로 바꿨다.
  { const double fc = getenv("FRIC_COMP") ? atof(getenv("FRIC_COMP")) : 1.0;
    const double fv = getenv("FRIC_V0")   ? atof(getenv("FRIC_V0"))   : 0.20;
    const bool all  = getenv("FRIC_ALL_MODES") && atoi(getenv("FRIC_ALL_MODES"));
    if(fc>0 && (c.cmode==1 || all))
      std::printf("[deploy] 마찰보상 **ON** — ×%.2f · v0=%.2f rad/s (JFRIC 0.827/0.604/0.871/0.639)\n", fc, fv);
    else if(fc>0)
      std::printf("[deploy] 마찰보상 OFF — 1점 보행 모드다(2점 평발에서만 켠다)\n");
    else
      std::printf("[deploy] ⚠마찰보상 **OFF**(FRIC_COMP=0) — 2점 stand 는 ~20s 에 넘어진다\n"); }

  BipedEstimator est;
  { std::vector<int> fg={c.sph[0],c.sph[1]};
    std::vector<double> fr={m->geom_size[c.sph[0]*3], m->geom_size[c.sph[1]*3]};
    est.init(m,fg,fr); est.reset(Eigen::Vector3d(0,0,d->qpos[2])); }

  // ── 지연보상(Smith predictor) ────────────────────────────────────────────
  // ★2026-08-14 신설. 종전엔 **시뮬에만** 있었다(deploy_loop.hpp) — 그런데 그 헤더를
  //   include 하는 건 biped_sim/biped_view 뿐이고, 실기 writer 인 이 바이너리는 아니다.
  //   그래서 `LAT_COMP_MS=8.4` 를 줘도 **아무 일도 일어나지 않았다.**
  //   STABILITY_MAP.md 가 그 값을 "배포 조건" 이라 적어둔 탓에 켜져 있다고 오해했다.
  //
  // 원리: 지금 읽는 센서는 T_d 전의 상태이고, 그 사이 내보낸 토크는 아직 안 나타났다.
  //       ⇒ **마지막 명령토크를 유지한 채** 모델을 T_d 만큼 굴려 "지금"을 만들어 제어한다.
  //   왕복지연 실측 **8.39±0.79 ms**(emb/pace/RESULTS.md ⑥). 모델 timestep 2ms → 4 step.
  //   ⚠LCOMP 는 제어주기가 아니라 **m->opt.timestep** 으로 나눈다 — mj_step 이 그만큼
  //     전진하기 때문이다. 지금은 둘 다 2ms 라 같지만 ctrl_hz 를 바꾸면 갈라진다.
  //
  // ★기본 **켜짐 8.4ms**(2026-08-14 근거 확보). 끄려면 `LAT_COMP_MS=0`.
  //   biped_sim 배포경로(EST_CTRL=1 · ACT_LAT_MS=8.4 · T=15s)에서 잰 값이다:
  //     보상없음  vx 0.05/0.15/0.20 에서 낙상 1/1/1  → 켜면 **전부 0**
  //     stand     base·tilt 가 소수점까지 동일 — **완전 중립**(해가 없다)
  //     보상값 오차 6.8/8.4/10.0/16.0 ms 전부 0 낙상(6.8·vx0.20 만 1)
  //       ⇒ 실측 ±0.79ms 는 물론 과보상 쪽으로도 안전하다. 위험한 건 **덜** 주는 쪽.
  //   ⚠속도잡음 증폭도 무시할 만하다: ENCDQ_N 0.0368 rad/s × 8.4ms = **0.018°**.
  const double MDT = m->opt.timestep;
  const double lat_comp_ms = getenv("LAT_COMP_MS") ? atof(getenv("LAT_COMP_MS")) : 8.4;
  // 안전망 — 예측이 튀면 **버리고 실측을 쓴다**. 시뮬판엔 이 가드가 없다(발산해도
  // 넘어질 뿐이지만, 실기에선 그 토크가 그대로 모터로 나간다).
  const double LC_MAX_DEG = getenv("LAT_COMP_MAX_DEG") ? atof(getenv("LAT_COMP_MAX_DEG")) : 5.0;
  const double LC_MAX_Z   = getenv("LAT_COMP_MAX_Z")   ? atof(getenv("LAT_COMP_MAX_Z"))   : 0.05;
  // ★운동학 외삽(`LAT_COMP_KIN=1`) — **이게 이 기기의 기본값이어야 한다.**
  //   동역학 롤아웃은 mj_step 4회인데 이 Pi 에서 mj_step 이 **604 µs** 다(실측 2026-08-14,
  //   두 모델 모두). 4회 = 2.42ms > 제어주기 2.00ms ⇒ **주기를 못 지킨다**
  //   (mock 실측: 504Hz → 285Hz). 지연을 보상하려다 지연을 만드는 셈이다.
  //   외삽은 q += q̇·T 라 사실상 공짜다(실측 501Hz 유지). 8.4ms 에서 빠지는 건 ½q̈T² 항뿐이고,
  //   시뮬 비교에서 **동역학과 낙상수가 동률**이었다(4속도 전부 0). ⇒ 기본값.
  //   `LAT_COMP_KIN=0` 으로 동역학 롤아웃을 강제할 수 있다(오프라인 연구용. 주기를 못 지킨다).
  const bool LC_KIN = getenv("LAT_COMP_KIN") ? atoi(getenv("LAT_COMP_KIN"))!=0 : true;
  int LCOMP = 0; mjData* dpred = nullptr;
  if(lat_comp_ms > 0){
    LCOMP = (int)std::lround(lat_comp_ms/1000.0/MDT);
    if(LCOMP > 0 && LC_KIN){
      std::printf("[deploy] 지연보상 **ON(운동학 외삽)** — %.2fms. 안전망 |Δq|<%.1f° · |Δz|<%.0fmm\n",
                  lat_comp_ms, LC_MAX_DEG, LC_MAX_Z*1e3);
    } else if(LCOMP > 0){
      dpred = mj_makeData(m);
      std::printf("[deploy] 지연보상 **ON(동역학 롤아웃)** — %.2fms = %d step(모델 dt %.1fms)."
                  " 안전망 |Δq|<%.1f° · |Δz|<%.0fmm\n"
                  "[deploy] ⚠⚠ mj_step %d회 ≈ %.2fms > 제어주기 %.2fms — **주기를 못 지킨다.**"
                  " 이 기기에선 LAT_COMP_KIN=1 을 쓸 것.\n",
                  lat_comp_ms, LCOMP, MDT*1e3, LC_MAX_DEG, LC_MAX_Z*1e3,
                  LCOMP, LCOMP*0.604, dt*1e3);
    } else {
      std::printf("[deploy] ⚠LAT_COMP_MS=%.2f 는 모델 1 step(%.1fms) 미만 → 보상 안 함\n",
                  lat_comp_ms, MDT*1e3);
    }
  } else {
    std::printf("[deploy] ⚠지연보상 **OFF**(LAT_COMP_MS=0) — 실측지연 8.39ms 가 보상되지 않는다.\n");
  }
  std::vector<double> u_prev(NU, 0.0);      // ★in-flight 토크. 롤아웃이 이걸 유지한다
  long lc_n = 0, lc_skip = 0; bool lc_warned = false;

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

  // ★★IMU 생존 확인 — tilt E-stop 은 IMU 가 있어야만 동작한다.
  //   이 로봇은 SHM fIMUBuf 가 전부 0 인데 IsUpdatedIMU() 는 1 을 반환한다
  //   (emb/IMU_RECOVERY.md). 그러면 tilt ≡ 0 이라 임계 40° 에 **영원히 도달 못 하고**
  //   tilt E-stop 이 **완전히 무력**해진다. 조용히 넘어가면 보호장치가 있다고
  //   착각하게 되므로 기동 시 명시적으로 경고하고 상태에도 노출한다.
  //   ⚠"값이 0" 보다 "신선한 0" 이 더 위험하다 — freshness 검사로 안 걸러진다.
  bool imu_dead = false;
  { HwState s0; hw->read(s0);
    double a = std::fabs(s0.acc[0])+std::fabs(s0.acc[1])+std::fabs(s0.acc[2]);
    double r = std::fabs(s0.rpy[0])+std::fabs(s0.rpy[1])+std::fabs(s0.rpy[2]);
    // 정상 IMU 는 정지 중에도 가속도계에 중력 ~9.81 이 반드시 잡힌다. 그게 없으면 죽은 것.
    imu_dead = (a < 0.5 && r < 1e-9);
    if(imu_dead){
      std::fprintf(stderr,
        "\n%s\n"
        "!! ⚠⚠ IMU 가 전부 0 → **tilt E-stop 무력**(tilt≡0 이라 임계에 도달 불가).\n"
        "!!    남은 런타임 보호는 워치독 · 토크트립 · 속도트립 **뿐**이다.\n"
        "!!    stand/walk 는 자세 피드백이 필요하다 — IMU 없이 돌리는 것은 위험하다.\n"
        "!!    원인·조치: emb/IMU_RECOVERY.md\n%s\n\n",
        std::string(72,'!').c_str(), std::string(72,'!').c_str());
    } else {
      std::printf("[deploy] IMU 정상 — |acc|합 %.2f (중력 감지). tilt E-stop 유효.\n", a);
    }
  }

  // ── 상태 ──
  std::vector<float> q_ch(NCH), dq_ch(NCH), tau_ch(NCH), kp_ch(NCH), kd_ch(NCH), zero(NCH,0.f);
  // ★모드별 **실제로 내보낸 위치명령**. 발행해서 모니터가 "명령 vs 측정" 을 보여줄 수 있게 한다.
  //   종전엔 안 냈다 — stand 가 순수 토크(kp=0)라 명령각이 무의미했기 때문이다.
  //   그런데 STAND_KP_FLOOR 를 켜면서 **의미가 생겼다**(2026-08-20). 처짐을 보려면 필요하다.
  std::vector<float> qcmd_ch(NCH, 0.f);
  // ★실제로 내보낸 게인도 같이 기록한다. 종전엔 발행값이 **0.0 하드코딩**이었다 —
  //   그래서 모니터의 "≈Nm"(위치오차→토크 환산)도, "kp 걸렸는데 안 따라옴" 판정도
  //   전부 죽어 있었다. 실제로 HL_foot 이 명령 +100° 인데 채널 −0.28° 에 머물러도
  //   모니터는 아무 경고를 못 냈다(2026-08-20).
  std::vector<float> kpcmd_ch(NCH, 0.f), kdcmd_ch(NCH, 0.f);
  // ★★위치모드(home·hold) 게인 배율 (2026-08-20). 위치오차를 줄이려면 kp 를 올린다.
  //   ⚠**두 가지가 같이 움직인다**:
  //   ① 토크트립 — kp_ch 1 당 0.0175 Nm/deg 다(실측). kp 500 이면 8.75 Nm/deg 라
  //      **1.7° 오차에서 트립(15Nm)** 이 걸린다. 게인을 올릴수록 트립이 예민해진다.
  //   ② 감쇠비 ζ ∝ kd/√kp — kp 만 5배 올리면 ζ 가 **2.24배 떨어진다.**
  //      지금도 속도잡음(±3~7dps)이 kd 를 타고 틱틱거리는데, 그게 진동으로 커진다.
  //   ⇒ kd 를 **√배율**만큼 같이 올리는 것을 기본으로 한다(ζ 보존). 따로 주려면 POS_KD_SCALE.
  //   ⚠stand 는 이 배율을 안 쓴다 — 거기선 WBIC 와 싸우면 안 되고 STAND_KP_FLOOR 가 따로 있다.
  const double POS_KP = getenv("POS_KP_SCALE") ? atof(getenv("POS_KP_SCALE")) : 1.0;
  const double POS_KD = getenv("POS_KD_SCALE") ? atof(getenv("POS_KD_SCALE"))
                                               : std::sqrt(std::max(1e-9, POS_KP));
  if(POS_KP != 1.0 || POS_KD != 1.0){
    const double tau_per_deg = 0.0175 * POS_KP;   // kp_ch 100 기준 축의 값
    std::printf("[deploy] 위치게인 배율 kp×%.2f · kd×%.2f — hip kp_ch %.0f→%.0f\n"
                "         ⚠트립까지 %.2f° (hip 기준 %.1fNm/deg · τ_trip %.0fNm)\n",
                POS_KP, POS_KD, cfg.joints[0].kp, cfg.joints[0].kp*POS_KP,
                cfg.tau_trip_nm/(cfg.joints[0].kp*tau_per_deg/100.0*100.0),
                cfg.joints[0].kp*tau_per_deg/100.0*100.0, cfg.tau_trip_nm);
  }
  std::vector<double> q_ctrl(NJ), dq_ctrl(NJ), tau_ctrl(NU);
  // ★토크 통계 누적기 — 루프율로 쌓고 발행마다 비운다(위 ① 주석 참조).
  std::vector<double> ts_sum(jm.n_leg,0.0), ts_sq(jm.n_leg,0.0),
                      ts_min(jm.n_leg, 1e300), ts_max(jm.n_leg,-1e300);
  long ts_n = 0;
  auto ts_reset = [&]{
    std::fill(ts_sum.begin(),ts_sum.end(),0.0); std::fill(ts_sq.begin(),ts_sq.end(),0.0);
    std::fill(ts_min.begin(),ts_min.end(), 1e300);
    std::fill(ts_max.begin(),ts_max.end(),-1e300);
    ts_n=0;
  };
  std::vector<float> hold_ch(NCH, 0.f);
  HwState hs;
  // ★home 램프 상태 (2026-08-20 신설). stand 자세로 **속도제한을 걸어** 이동한다.
  //   왜 필요한가: `stand` 는 WBIC posture task 라 램프가 없다. 0° 자세에서 바로 누르면
  //   발목이 채널각 **100.4°** 를 한꺼번에 요구한다(커플링·gear_k 1.2 때문에 모델 −59.8°
  //   가 채널 100.4° 가 된다). 속도트립 200dps 에 그냥 걸린다.
  //   ⇒ home 으로 먼저 그 자세까지 S-curve 로 간 뒤 hold→접지→stand 순서로 간다.
  std::vector<float> home_from(NCH,0.f), home_to(NCH,0.f);
  double home_t0=0, home_T=0; bool home_done=false;
  // ★jog 램프 상태 (2026-08-21). **등속 램프**다 — home 의 S-curve 를 쓰지 않는다.
  //   S-curve 는 "A→B 1회 이동" 전제인데 jog 목표는 슬라이더를 끄는 동안 **매 틱 바뀐다.**
  //   그래서 살아 있는 목표를 속도클램프로 따라간다(biped_emb.py control/jog.py 와 같은 방식).
  //   ⚠등속 램프는 출발·도착에서 가속도가 불연속이다(config 주석). jog 는 축 하나씩 천천히
  //     쓰는 검증용이라 실무상 문제가 안 됐지만, 알고 쓰는 것과 모르고 쓰는 것은 다르다.
  std::vector<double> jog_q(NJ, 0.0);        // 램프 중인 명령(**모델각 deg**)
  double jog_prev_t = 0; bool jog_init = false;
  bool ground_refused=false;      // ★접지 가드 거부 래치(로그 폭주 방지)
  bool still_warned=false;        // ★정지 확인 거부 래치
  // ★★무장 직후 트레이스 (2026-08-20). hold 진입 즉시 속도트립이 반복되는데
  //   원인 후보(게인 점프 / 낡은 측정값 / 부호 / 한쪽 다리)를 말로 가릴 수 없다.
  //   무장 순간부터 0.5초를 **매 틱** CSV 로 남긴다. 트립이 나도 파일은 남는다.
  FILE* trc=nullptr; double trc_t0=0;
  bool have_state=false; int ok_reads=0; double live_t0=0;
  std::string boot_mode="off"; bool mode_locked=false;   // ★기동 시 잔여명령 잠금   // ★센서 준비·생존 확인
  // ★★EtherCAT 동결 감지 (2026-08-20 실기). Emb 는 OP 를 잃어도 프로세스가 계속 돌고
  //   **마지막 버퍼를 재발행하며 갱신 플래그까지 1 로 세운다**(memory: emb-ethercat-freeze).
  //   그래서 health=ok · n_ok=8 · n_fault=0 인데 값만 얼어붙는다 — 침묵 실패다.
  //   실측(2026-08-20): **왼다리 4축만** 동결. dq 가 227dps 로 굳어 있는데 각도는 불변이라
  //   정지가드가 "움직이는 중" 으로 오판했다. 원인을 엉뚱하게 짚게 만든다.
  //   ⇒ 채널별로 (q,dq,tau) 가 **한 번도 안 바뀐 시간**을 세어 직접 이름 붙인다.
  std::vector<float> prv_q(NCH,1e9f), prv_dq(NCH,0.f), prv_t(NCH,0.f);
  std::vector<double> frz_t(NCH,0.0);  bool frz_warned=false; int frozen_now=0;
  // ★★stand 진입 블렌드 (2026-08-20 실기). hold→stand 는 위치제어(kp 100/50/80/30)에서
  //   **kp=kd=0 순수토크**로 한 틱에 바뀐다. 그 순간 WBIC 토크가 조금만 모자라도
  //   그대로 주저앉는다(실기 관측). 시뮬은 α=1 이라 안 드러난다 — 실기는 토크 스케일이
  //   ±10% 미검증이고 마찰도 있다.
  //   ⇒ 위치게인을 내리면서 WBIC 토크를 올린다. MIT 모드는 둘을 동시에 받으므로
  //     블렌드 중에는 위치제어가 받쳐 주고, 끝나면 순수토크가 된다.
  double stand_t0=0, stand_T=0; std::vector<float> stand_hold(NCH,0.f), stand_to(NCH,0.f), stand_ref(NCH,0.f);
  std::string mode = "off", prev_mode = "off", last_raw;
  bool estop = false, wd_tripped = false;
  double tau_over_t0 = -1, last_cmd_t = now_s(), last_pub = 0, hz_ema = cfg.ctrl_hz;
  Cmd cmd; double body_h = 0.5;
  const double watchdog_s = cfg.watchdog_ms/1000.0;

  hw->enable(0);
  std::printf("[deploy] 모드: off/hold/**home**/**jog**/stand/walk. GUI 로 조종(%s).\n"
              "[deploy] home = %s 자세로 %.0fdps S-curve 이동(램프 %.1fs 설정).\n"
              "[deploy] jog  = 축별 목표각 추종 · %.0fdps 등속 램프 · 관절한계 클램프.\n",
              cmd_p.c_str(), c.cmode==1?"2점 평발 stand":"1점 점발",
              cfg.home_speed_dps, cfg.home_min_time_s, cfg.jog_speed_dps);
  // ★위 두 값은 **실제 파싱된 것**을 찍는다 — config 가 안 읽혀도 기본값으로 조용히
  //   도는 걸 막는다. 2026-08-21 까지 실제로 그랬다(키가 safety 분기에 있어 무시됐다).
  std::printf("[deploy] ⚠ 모터 명령 writer 는 한 번에 하나만 — biped_emb.py 와 동시 실행 금지.\n");

  // ★★기동 시 명령파일에 남아 있는 모드를 **그대로 실행하지 않는다** (2026-08-20).
  //   GUI 는 마지막 상태를 파일에 남긴다. 실제로 seq 14871 짜리 `"mode":"stand"` 가
  //   남아 있었고, biped_deploy 는 기동 즉시 그걸 받아 무장했다 —
  //   매달린 채 stand → 접지가드가 hold 로 되돌림 → 흔들리는 로봇에 hold → 속도트립.
  //   기동할 때마다 이 고리를 돌았다.
  //   ⇒ 시작 시점의 내용을 **이미 본 것**으로 기록해 두고, 운전자가 버튼을 눌러
  //     내용이 **바뀌어야** 받는다. biped_emb.py 의 --start-mode off 와 같은 의도다.
  { Cmd c0; if(read_cmd(cmd_p, c0)){ last_raw = c0.raw;
      // ⚠내용 비교만으로는 못 막는다 — GUI 가 seq 를 올리며 **주기적으로 재발행**하므로
      //   같은 stand 도 매번 "새 명령" 이 된다(실측: 그대로 무장했다).
      //   ⇒ **모드**를 잠근다. 운전자가 다른 모드를 한 번 고르기 전까지 무시한다.
      boot_mode = c0.mode; mode_locked = (c0.mode != "off");
      if(c0.mode != "off")
        std::printf("[deploy] ⚠명령파일에 **%s** 가 남아 있다 — 무시하고 off 로 시작한다.\n"
                    "         GUI 에서 버튼을 다시 눌러야 반영된다(기동 즉시 무장 방지).\n",
                    c0.mode.c_str()); } }
  double t0 = now_s(), prev_loop = t0; long long k = 0; bool overrun_warned=false;
  int rc = 0;
  while(!g_stop && (now_s()-t0) < T){
    double lt = now_s();

    // ① 센서
    hw->read(hs);
    jm.ch_to_q_ctrl(hs.q_deg.data(),  q_ctrl.data());
    jm.ch_to_dq_ctrl(hs.dq_dps.data(), dq_ctrl.data());
    // ★★2026-08-20 **토크 통계는 루프율(500Hz)에서 낸다** — 발행(20Hz)에서 내면 안 된다.
    //   상태는 20Hz 로 나가므로 나이퀴스트가 10Hz 다. 그런데 이 로봇이 발산한 대역은
    //   **30~65Hz** 였다(a117c44: |dq| 196→322 dps). 모니터가 받은 표본으로 표준편차를
    //   계산하면 그 대역이 통째로 접혀 **리플을 과소평가**한다 — 정작 봐야 할 것을 못 본다.
    //   ⇒ 여기서 창(발행주기)마다 누적하고 std·min·max 를 실어 보낸다. 러닝 합이라
    //     비용은 무시할 수준이고, 모니터는 그리기만 하면 된다.
    //   ★min/max 를 같이 내는 이유: 이번 발산은 **첨두** 현상이라 평균·표준편차로는
    //     안 잡힌다. 창 안의 극값이 남아야 한다.
    {
      std::vector<double> tnow(jm.n_leg);
      jm.ch_to_tau_joint(hs.tau_nm.data(), tnow.data());
      for(int i=0;i<jm.n_leg;i++){
        const double v = tnow[i];
        ts_sum[i]+=v; ts_sq[i]+=v*v;
        if(v<ts_min[i]) ts_min[i]=v;
        if(v>ts_max[i]) ts_max[i]=v;
      }
      ts_n++;
    }
    const double D2R = JointMap::D2R;
    double rpy[3] = { hs.rpy[0]*(cfg.imu_deg?D2R:1.0),
                      hs.rpy[1]*(cfg.imu_deg?D2R:1.0),
                      hs.rpy[2]*(cfg.imu_deg?D2R:1.0) };
    double gyro[3]= { hs.gyr[0]*(cfg.imu_deg?D2R:1.0),
                      hs.gyr[1]*(cfg.imu_deg?D2R:1.0),
                      hs.gyr[2]*(cfg.imu_deg?D2R:1.0) };
    double quat[4]; rpy_to_quat(rpy[0],rpy[1],rpy[2],quat);
    double tilt = std::hypot(rpy[0],rpy[1]) * JointMap::R2D;

    // ★★**센서가 채워지기 전에는 명령을 받지 않는다** (2026-08-20, 트립 5회의 진짜 원인).
    //   기동 직후 첫 틱에 SHM 읽기가 아직 0 이었는데, 명령파일에 이전 hold/home 이
    //   남아 있어 **그 0 을 측정각으로 믿고** 래치했다. 그러면 명령이 0° 가 되어
    //   실제 ±25~85° 에서 0 으로 내리꽂는다 — 트레이스 실측:
    //       ch2 q 25.88 · 명령 0.00 · τ −18.2 Nm · dq −340dps
    //       ch6 q −85.12 · 명령 0.00 · τ +18.5 Nm · dq +460dps
    //   "최대이동 100.4°"(=0°에서 Qflat8 까지) 가 같은 사실을 가리키고 있었다.
    //   ⇒ 유효한 읽기가 몇 틱 쌓이기 전에는 off 를 유지한다.
    // ★동결 판정 — 값이 하나도 안 바뀌면 그 채널의 정지시간을 누적한다.
    for(int i=0;i<NCH;i++){
      if(hs.q_deg[i]==prv_q[i] && hs.dq_dps[i]==prv_dq[i] && hs.tau_nm[i]==prv_t[i]) frz_t[i]+=dt;
      else frz_t[i]=0.0;
      prv_q[i]=hs.q_deg[i]; prv_dq[i]=hs.dq_dps[i]; prv_t[i]=hs.tau_nm[i];
    }
    // ⚠**mock 은 제외한다** (2026-08-21). MockHw 는 명령이 없으면 값이 안 변하는 게 정상이라
    //   전 채널이 동결로 잡히고 **무장이 막힌다** — `--mock` 으로는 off 말고 아무 모드도
    //   못 들어가서 배포경로를 오프로봇으로 검증할 수 없었다.
    //   기동 생존확인(아래)은 이미 `!mock` 로 제외돼 있었다. 이쪽만 빠져 있어 짝이 안 맞았다.
    { std::string fz; int nfz=0;
      for(int i=0;i<NCH;i++) if(!mock && cfg.installed_has(i) && frz_t[i]>0.5){
        char b[16]; std::snprintf(b,sizeof b," ch%d",i); fz+=b; nfz++; }
      if(nfz && !frz_warned){ frz_warned=true;
        // ★몇 축이 얼었는지로 **어느 구간이 끊겼는지** 가른다 (2026-08-20).
        //   EtherCAT 슬레이브는 8축 **하나**(LAN9252)다. 그게 끊기면 8축이 **전부** 언다.
        //   4축만 얼면 MCU 는 살아 있고 그 아래 **FDCAN(해당 다리)** 이 끊긴 것이다 —
        //   MCU 가 마지막 값을 EtherCAT 으로 계속 올리므로 갱신 플래그·health 는 정상으로 보인다.
        const bool all_ch = (nfz >= (int)cfg.joints.size());
        std::fprintf(stderr,
          "\n%s\n!! ⛔⛔ **통신 동결** — 다음 채널이 0.5초 넘게 값이 하나도 안 바뀐다:%s\n"
          "!!   %s\n"
          "!!   ⚠health=ok · n_fault=0 으로 보인다 — **믿으면 안 된다.**\n"
          "!!   복구: 모터 전원 OFF/ON → Emb 재기동. %s\n%s\n\n",
          std::string(72,'!').c_str(), fz.c_str(),
          all_ch ? "8축 **전부** → EtherCAT(EMB↔MCU) 구간이 끊겼다."
                 : "일부만 얼었다 ⇒ **EtherCAT 이 아니다**(슬레이브는 8축 하나뿐이다).\n"
                   "!!   MCU 는 살아 있고 그 아래 **FDCAN(그 다리)** 이 끊긴 것이다.",
          all_ch ? "" : "\n!!   ⚠전원 재투입으로 잠깐 풀려도 재발한다 — 배선·커넥터·종단저항을 볼 것.",
          std::string(72,'!').c_str());
      } else if(!nfz) frz_warned=false;
      if(nfz && mode=="off"){ /* 동결 중에는 무장을 막는다 */ }
      frozen_now = nfz; }

    // ★★기동 시 **생존 확인**: 명령을 받기 전에 전 축이 실제로 갱신되는지 본다.
    //   동결은 세 번 다 같은 4채널이었고, 매번 **무장을 시도한 뒤에야** 드러났다.
    //   그때는 이미 E-stop 이 걸리고 로봇이 흔들린 뒤다. 먼저 확인하는 게 맞다.
    //   1초 동안 각 채널이 한 번이라도 바뀌는지 세고, 안 바뀐 채널이 있으면 무장을 막는다.
    if(!have_state){
      if(hs.mask != 0) ok_reads++; else ok_reads = 0;
      if(ok_reads == 1) live_t0 = lt;
      if(ok_reads >= 5 && lt-live_t0 < 1.0){ /* 아직 관찰 중 */ }
      else if(ok_reads >= 5){
        std::string dead; int nd=0;
        // ⚠mock 은 값이 정지해 있는 게 정상이라 이 검사를 건너뛴다(오탐).
        //   실기 엔코더는 정지 중에도 항상 미세하게 떨린다 — 그게 이 검사의 근거다.
        for(int i=0;i<NCH;i++) if(!mock && cfg.installed_has(i) && frz_t[i] > 0.9){
          char b[16]; std::snprintf(b,sizeof b," ch%d",i); dead+=b; nd++; }
        if(nd){
          std::fprintf(stderr,
            "\n%s\n!! ⛔⛔ **기동 생존확인 실패** — 1초 동안 값이 한 번도 안 바뀐 채널:%s\n"
            "!!   %s\n"
            "!!   **명령을 받지 않는다.** 배선을 고치고 전원 OFF/ON 후 다시 띄울 것.\n"
            "!!   (강제로 진행: LIVE_CHECK=0 — 죽은 축은 제어가 안 된다)\n%s\n\n",
            std::string(72,'!').c_str(), dead.c_str(),
            nd >= (int)cfg.joints.size() ? "8축 전부 → EtherCAT(EMB↔MCU)"
                                         : "일부만 → **FDCAN(그 다리)**. MCU 는 살아 있다.",
            std::string(72,'!').c_str());
          if(!(getenv("LIVE_CHECK") && atoi(getenv("LIVE_CHECK"))==0)){
            safe_shutdown(*hw, NCH); delete hw;
            mj_deleteData(d); mj_deleteModel(m); return 3;
          }
        }
        have_state = true;
        if(start_mode=="hold"){ mode="hold"; hold_ch=hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data());
          hw->enable(1);
          std::printf("[deploy] --start-mode hold — 현재 자세를 잡는다(무장).\n"); }
        std::printf("[deploy] 센서 준비됨 — 명령 수신 시작 (q_ch: %.1f %.1f %.1f %.1f ...)\n",
                    hs.q_deg[0], hs.q_deg[1], hs.q_deg[2], hs.q_deg[3]); }
    }
    // ② 명령 폴링(~50Hz)
    if(have_state && k % (long long)std::max(1.0, 0.02/dt) == 0){
      Cmd nc;
      if(read_cmd(cmd_p, nc)){
        bool fresh = (nc.raw != last_raw);          // ★"파일이 읽히는가"가 아니라 "내용이 바뀌는가"
        last_raw = nc.raw;                          //   (biped_emb.read_cmd_fresh 와 같은 이유)
        if(fresh) last_cmd_t = lt;
        cmd = nc; body_h = nc.body_h;
        std::string nm = nc.mode;
        if(nm=="reset") nm = "hold";
        if(nm!="off" && nm!="hold" && nm!="home" && nm!="jog" && nm!="stand" && nm!="walk") nm = "off";
        // ★E-stop 래치는 명시적 off 로만 해제. 그 전까지 모드변경 무시.
        if(estop){
          if(nm=="off"){ estop=false; std::printf("[deploy] E-stop 래치 해제(off 수신) — 재무장 가능\n"); }
          else nm = "off";
        }
        // ★기동 시 남아 있던 모드는 **다른 모드를 한 번 고를 때까지** 무시한다.
        if(mode_locked){
          if(nm == boot_mode) nm = "off";
          else { mode_locked = false;
                 std::printf("[deploy] 명령 잠금 해제 — %s 수신\n", nm.c_str()); }
        }
        if(nm!="stand" && nm!="walk") ground_refused = false;   // ★다른 모드 = 재시도 허용
        // ★★거부는 **래치**다 (2026-08-21, mock 실측으로 발견).
        //   종전엔 `ground_refused` 가 **검사만** 건너뛰게 했다. 그런데 GUI 는 stand 를
        //   20ms 마다 재전송하므로, 다음 폴에서 nm!=mode 로 다시 진입하고 검사는 건너뛴 채
        //   **그대로 stand 로 들어갔다.** 즉 거부가 한 틱만 유효했다.
        //   ⚠접지 검사도 같은 경로였다 — 이 커밋 이전엔 "접지 안 됨" 도 한 번만 막혔다.
        //   ⇒ 다른 모드를 한 번 고를 때까지 stand/walk 를 hold 로 눌러 둔다.
        if(ground_refused && (nm=="stand" || nm=="walk")) nm = "hold";
        // ★★**움직이는 중에는 무장하지 않는다** (2026-08-20 실기).
        //   속도트립은 *측정* 속도로 걸린다 — 우리가 명령을 안 줘도, 로봇이 이미
        //   무너지는 중이면 무장하는 순간 그대로 트립한다(ch6 201dps · ch2 204 · ch3 224).
        //   실제 경위: 접지 상태에서 Ctrl+C → safe_shutdown 이 kp=kd=τ=0 을 써서
        //   **로봇이 주저앉았고**, 그 낙하 도중에 hold 를 눌렀다.
        //   ⇒ 트립의 원인은 명령이 아니라 "움직이는 걸 잡으려 한 것" 이다. 미리 막는다.
        if(mode=="off" && nm!="off" && frozen_now>0){
          std::printf("[deploy] ⛔ **EtherCAT 동결 %d채널** — 무장 거부. 전원 OFF/ON 후 Emb 재기동.\n", frozen_now);
          nm = "off";
        } else if(mode=="off" && nm!="off"){
          double vmx=0; int vch=-1;
          for(int i=0;i<NCH;i++) if(cfg.installed_has(i) && std::fabs(hs.dq_dps[i])>vmx){
            vmx=std::fabs(hs.dq_dps[i]); vch=i; }
          const double vlim = cfg.vel_trip_dps * 0.25;      // 트립의 1/4 — 잔진동은 허용
          if(vmx > vlim){
            if(!still_warned){ still_warned = true;
              std::printf("[deploy] ⛔ **아직 움직이고 있다** — ch%d %.0fdps > %.0f. 무장 거부.\n"
                          "         로봇이 정지한 뒤에 누를 것(주저앉는 중이면 크레인으로 받칠 것).\n"
                          "         지금 무장하면 속도트립(%.0fdps)에 즉시 걸린다.\n",
                          vch, vmx, vlim, cfg.vel_trip_dps); }
            nm = "off";
          } else still_warned = false;
        }
        if(nm != mode){
          prev_mode = mode; mode = nm;
          if(((mode!="off" && prev_mode=="off") || mode=="stand" || mode=="walk") && !trc){
            trc = fopen("/tmp/arm_trace.csv","w"); trc_t0 = lt;
            if(trc){ fprintf(trc,"t");
              for(int i=0;i<NCH;i++) fprintf(trc,",q%d,dq%d,tau%d,cmd%d",i,i,i,i);
              fprintf(trc,"\n");
              std::printf("[deploy] 트레이스 → /tmp/arm_trace.csv (3초 — 블렌드 전체)\n"); } }
          hw->enable(mode=="off" ? 0 : 1);
          if(mode=="hold"){
            std::vector<float> raw = hs.q_deg;             // 클램프 전
            hold_ch = hs.q_deg; jm.clamp_ch_via_joint(hold_ch.data());
            // ★hold 진입은 "측정각을 그대로 목표로" 라 **오차 0 이어야** 한다.
            //   그런데 실기에서 진입 즉시 ch2(204dps)·ch3(224dps) 속도트립이 났다.
            //   원인 후보가 둘이라 찍어서 가른다:
            //     (a) clamp_ch_via_joint 의 ch→관절→ch 왕복이 값을 바꾼다(발목은 커플링이라
            //         calf 에 의존한다 — 왕복이 항등이 아닐 수 있다)
            //     (b) 왕복은 항등인데 **게인 인계 점프**다(Emb kd 5.0 → 우리 3.5/2.0).
            //   Δ 가 0 이면 (b), 0 이 아니면 (a) 다. 한 번만 찍는다.
            std::string dmsg; double dmx=0; int dch=-1;
            for(int i=0;i<NCH;i++){
              double dd = (double)hold_ch[i]-(double)raw[i];
              if(std::fabs(dd)>std::fabs(dmx)){ dmx=dd; dch=i; }
              char b[48]; std::snprintf(b,sizeof b," ch%d %+.3f", i, dd); dmsg += b;
            }
            std::printf("[deploy] hold 래치 — 클램프 Δ(목표−측정):%s\n", dmsg.c_str());
            if(std::fabs(dmx)>0.05)
              std::printf("[deploy] ⚠클램프가 ch%d 를 %+.3f° 옮겼다 — 그만큼 **계단 명령**이 된다.\n", dch, dmx);
            else
              std::printf("[deploy] ✓클램프 영향 없음(최대 %+.3f°) — 움직이면 **게인 인계 점프**다.\n", dmx);
            std::printf("[deploy]   인계 게인 kp/kd = hip %.0f/%.1f · thigh %.0f/%.1f · calf %.0f/%.1f · foot %.0f/%.1f\n",
                        cfg.joints[0].kp,cfg.joints[0].kd, cfg.joints[1].kp,cfg.joints[1].kd,
                        cfg.joints[2].kp,cfg.joints[2].kd, cfg.joints[3].kp,cfg.joints[3].kd);
          }
          if(mode=="jog"){
            // ★현재 **측정각에서** 시작한다(클램프 없이) — 명령 점프 방지.
            //   한계로 클램프해서 시작하면, 현재 자세가 jog 범위 밖일 때 진입 즉시
            //   한계까지 순간이동 명령이 나간다(= 막으려던 바로 그 점프).
            //   biped_emb.py Jogger.reset 이 같은 이유로 클램프를 뺐다(2026-08-10).
            jm.ch_to_q_joint(hs.q_deg.data(), jog_q.data());
            jog_prev_t = lt; jog_init = true;
            std::printf("[deploy] jog 진입 — 현재 자세에서 시작 · %.0f dps 제한 · 한계는 관절한계\n",
                        cfg.jog_speed_dps);
          }
          if(mode=="home"){
            home_from = hs.q_deg;
            std::vector<double> qt(NJ);
            for(int j=0;j<NJ;j++) qt[j] = (c.cmode==1 ? c.Qflat8[j] : c.Qhome8[j]);
            jm.q_ctrl_to_ch(qt.data(), home_to.data());
            jm.clamp_ch_via_joint(home_to.data());
            double mx=0;
            for(int i=0;i<NCH;i++) if(cfg.installed_has(i))
              mx = std::max(mx, (double)std::fabs(home_to[i]-home_from[i]));
            // smoothstep s=3u²−2u³ : 최대속도 1.5·Δ/T · 최대가속 6·Δ/T²
            //   ⇒ T 를 둘 다 만족하게 잡으면 트립 임계 안에서 끝난다.
            double T1 = 1.5*mx/std::max(1e-6, cfg.home_speed_dps);
            double T2 = std::sqrt(6.0*mx/std::max(1e-6, cfg.home_acc_dps2));
            home_T = std::max(std::max(T1,T2), cfg.home_min_time_s);
            home_t0 = lt; home_done = false;
            std::printf("[deploy] home → **%s** 자세 · 최대이동 %.1f° · %.1fs 램프"
                        "(S-curve · 최대 %.0fdps ≪ 트립 %.0f)\n",
                        c.cmode==1?"2점 평발 stand":"1점 점발 home",
                        mx, home_T, 1.5*mx/home_T, cfg.vel_trip_dps);
          }
          // ★★접지 확인 없이 stand 를 못 켜게 막는다 (2026-08-20 실기).
          //   매달린 채 stand 를 누르면 WBIC 가 요구하는 지면반력을 지면이 못 내줘
          //   QP 가 매 틱 실패하고(시뮬 실측 95%) 중력보상 폴백으로 떨어진다.
          //   실기에선 그 직전에 다리가 흔들려 속도트립이 난다(ch7 207dps).
          //   ⚠겉보기엔 "그냥 안 되는" 것처럼 보여 원인을 찾기 어렵다 — 그래서 막는다.
          //
          //   판별: **모델이 예측한 '매달림' 중력토크와 실측 토크를 비교**한다.
          //     매달림이면 다리 자중만 걸리므로 둘이 비슷하다.
          //     접지면 몸무게가 다리를 통해 내려와 실측이 훨씬 커진다.
          //   힘센서가 없어도 되고, 임계를 사람이 정하지 않아도 된다.
          // ★거부는 **래치**한다. 명령파일이 계속 stand 면 50Hz 로 재시도하며 로그가
          //   폭주한다(같은 함정이 hold↔stand 재전이에도 있었다 — 그때 남긴 주석 참조).
          //   래치는 운전자가 stand/walk 가 아닌 모드를 한 번 보내면 풀린다.
          if((mode=="stand" || mode=="walk") && prev_mode!="stand" && prev_mode!="walk"
             && !ground_refused){
            for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
            d->qpos[0]=d->qpos[1]=0; d->qpos[2]=0.5;
            d->qpos[3]=1; d->qpos[4]=d->qpos[5]=d->qpos[6]=0;
            for(int i=0;i<6;i++) d->qvel[i]=0.0;
            mj_forward(m,d);
            std::vector<double> tau_meas(NJ,0.0);
            jm.ch_to_tau_joint(hs.tau_nm.data(), tau_meas.data());
            double hang=0, meas=0;
            for(int j=0;j<NJ;j++){
              hang += std::fabs(d->qfrc_bias[6+j]);
              meas += std::fabs(tau_meas[j]);
            }
            const double ratio = (hang>1e-6) ? meas/hang : 0.0;
            // ★기준 1.25 — 1.5 는 **너무 높았다**(2026-08-20 정정). 모델로 재보니
            //   매달림 16.21 Nm · 체중전부 25.72 Nm 라 비가 **1.00~1.59** 밖에 안 움직인다.
            //   1.5 를 요구하면 체중의 **95%** 를 넘겨야 통과 = 크레인을 거의 다 내린 상태인데,
            //   hold 는 그 하중을 못 버티고 주저앉는다(kp 100/50/80/30 은 체중용이 아니다).
            //   ⇒ 1.25 ≈ 체중의 **42%**. "확실히 닿았다" 는 알 수 있고 hold 도 버틴다.
            const double need = getenv("GROUND_RATIO") ? atof(getenv("GROUND_RATIO")) : 1.25;
            std::printf("[deploy] 접지 확인 — 실측 |τ|합 %.2f Nm vs 매달림 예측 %.2f Nm · 비 %.2f (기준 %.2f)\n",
                        meas, hang, ratio, need);
            if(ratio < need){
              std::printf("[deploy] ⛔ **접지가 안 됐다** — stand 거부, hold 유지.\n"
                          "         크레인을 내려 발바닥을 붙이고 하중을 로봇에 넘긴 뒤 다시 누를 것.\n"
                          "         (강제로 진행하려면 GROUND_RATIO=0 — 매달림 stand 는 위험하다)\n");
              nm = "hold"; mode = "hold"; ground_refused = true;
            }
          }
          // ★★자세 거리 가드 (2026-08-21). 블렌드 목표가 기하 자세로 **이동**하게 바뀌면서
          //   생긴 위험을 막는다. 종전엔 목표가 측정각에 얼어 있어 아무리 멀어도 안 움직였다.
          //   지금은 2.5초 안에 그 거리를 쓸어버린다 — `home` 을 건너뛰면 위험하다:
          //       발목이 100° 떨어져 있으면 피크 ≈ 0.25·kp_joint 43.2·1.75rad ≈ **18.9 Nm**
          //       → 토크트립 15 Nm 초과
          //   ⚠"자동으로 home 을 돌리면 되지 않나" 는 **틀렸다.** home 은 매달린 채 하는
          //     동작이고 stand 는 접지 후다. stand 안에서 home 을 돌리면 **하중이 실린 채로
          //     다리를 크게 움직이게** 된다. 그래서 자동 실행이 아니라 **거부**가 맞다.
          if((mode=="stand" || mode=="walk") && !ground_refused){
            std::vector<double> qt0(NJ); std::vector<float> to0(NCH,0.f);
            for(int j=0;j<NJ;j++) qt0[j] = (c.cmode==1 ? c.Qflat8[j] : c.Qhome8[j]);
            jm.q_ctrl_to_ch(qt0.data(), to0.data());
            jm.clamp_ch_via_joint(to0.data());
            double far=0; int fch=-1;
            for(int j=0;j<NJ;j++){ int ch=cfg.joints[j].channel;
              double dd=std::fabs((double)to0[ch]-(double)hs.q_deg[ch]);
              if(dd>far){ far=dd; fch=ch; } }
            //   임계 15° — 처짐(채널 수 도)은 통과하고 "home 안 돌림"(채널 100°)은 걸린다.
            const double lim = getenv("STAND_POSE_LIM") ? atof(getenv("STAND_POSE_LIM")) : 15.0;
            if(far > lim){
              std::printf("[deploy] ⛔ **자세가 목표에서 멀다** — stand 거부, hold 유지.\n"
                          "         ch%d 가 %.1f° 어긋나 있다(허용 %.1f°). **먼저 home 을 돌릴 것.**\n"
                          "         (강제: STAND_POSE_LIM=999 — 블렌드가 그 거리를 2.5초에 쓸어 트립할 수 있다)\n",
                          fch, far, lim);
              nm = "hold"; mode = "hold"; ground_refused = true;
            }
          }
          if(mode=="stand" || mode=="walk"){
            stand_hold = hs.q_deg; jm.clamp_ch_via_joint(stand_hold.data());
            // ★★2026-08-21 블렌드 목표를 **측정각에 얼리지 않는다** — 기하 자세로 램프한다.
            //   종전엔 진입 순간 측정각(`stand_hold`)을 블렌드 내내 목표로 썼다. 그런데 그 값은
            //   **하중 상태에 따라 반대로 편향**된다:
            //       매달림 : PD 처짐만큼 아래로   (hip 5.247Nm/kp100 = 3.0° · thigh 2.6° …)
            //       접지   : 지면반력에 밀려 위로
            //   같은 의도 자세인데 언제 눌렀느냐로 hip 수 도가 갈리고, 그 편향이 블렌드 초반
            //   **kp 가 아직 살아 있는 구간**의 서보 목표가 된다.
            //   ⇒ 진입 시엔 측정각(계단 0), 블렌드가 끝날 때는 **Qflat8/Qhome8**(기하 진리).
            //     같은 smoothstep 계수를 쓰므로 추가 튜닝이 없다.
            //   ⚠목표를 처음부터 Qflat8 로 못 박으면 안 된다 — 로봇이 거기 없을 때 계단이 된다
            //     (0° 자세에서 발목 채널 100.4° 요구 → 속도트립. `home` 이 생긴 이유다).
            {
              std::vector<double> qt(NJ);
              for(int j=0;j<NJ;j++) qt[j] = (c.cmode==1 ? c.Qflat8[j] : c.Qhome8[j]);
              jm.q_ctrl_to_ch(qt.data(), stand_to.data());
              jm.clamp_ch_via_joint(stand_to.data());
            }
            // ★2.5초 (2026-08-20). 1.0초로는 ch7 이 202dps 로 임계를 1% 넘겼다.
            //   ⚠발목 채널은 구조상 제일 잘 걸린다 — raw각이 (foot+calf) 라
            //     채널속도 = (q̇_foot + q̇_calf)×1.2 로 **두 관절의 합**이 잡힌다.
            //     관절로는 작은 움직임도 채널로는 2배 넘게 보인다.
            stand_T = getenv("STAND_BLEND_S") ? atof(getenv("STAND_BLEND_S")) : 2.5;
            stand_t0 = lt;
            std::printf("[deploy] stand 진입 — 위치제어→토크 **%.1fs 블렌드**"
                        "(계단 전환은 주저앉는다)\n", stand_T);
            // ★★c.reset() 전에 **측정 자세를 모델에 주입**한다 (2026-08-20 실기).
            //   reset() 은 d->qpos 를 읽어 com_ref_xy · nominal_off · com_ref_z 를 잡는다.
            //   그런데 hold/home 분기는 d 에 아무것도 주입하지 않는다 — 그래서 그때까지
            //   d->qpos 는 **모델 기본자세**다. 그 상태로 reset 하면 WBIC 의 CoM 목표가
            //   실제 로봇과 몇 cm 어긋나고, 무장 순간 그 오차를 지우려 다리가 튄다
            //   (실기: 블렌드 중 ch2 204dps 트립. 실측 자세는 목표와 hip 7° · foot 12° 달랐다).
            for(int j=0;j<NJ;j++){ d->qpos[7+j]=q_ctrl[j]; d->qvel[6+j]=0.0; }
            d->qpos[0]=d->qpos[1]=0.0; d->qpos[2]=0.5;
            for(int a=0;a<4;a++) d->qpos[3+a]=quat[a];
            for(int i=0;i<6;i++) d->qvel[i]=0.0;
            mj_forward(m,d);
            c.reset();
            // ★2점 stand 는 **지금 높이를 유지**한다 — body_h(GUI 슬라이더)로 덮지 않는다.
            //   reset() 이 현재 CoM 높이를 잡아주는데 그걸 0.38/0.5 로 덮으면 무장 즉시
            //   수 cm 를 올리거나 내리려 든다(kp_z=200). 서 있는 걸 유지하는 게 목적이지
            //   높이를 바꾸는 게 아니다. 높이를 바꾸려면 STAND_H 로 명시할 것.
            if(c.cmode!=1) c.com_ref_z = body_h;
            else if(getenv("STAND_H")) c.com_ref_z = atof(getenv("STAND_H"));
            // ★밑창중심과 얼마나 어긋나 있는지 같이 찍는다. reset() 은 **지금 CoM 을 목표로**
            //   잡으므로(=움직이지 않음) 안전하지만, 그 지점이 지지면 밖이면 stand 해도 넘어진다.
            //   여유는 밑창 반길이 7.3cm 다 — 이 숫자가 그보다 크면 stand 를 걸면 안 된다.
            { Eigen::Vector3d fc = 0.5*(c.foot_center(0)+c.foot_center(1));
              const double ex = c.com_ref_xy[0]-fc[0], ey = c.com_ref_xy[1]-fc[1];
              std::printf("[deploy] stand 기준 — CoM 목표 xy(%.3f, %.3f) z %.3f (측정자세 = 지금 자세 유지)\n"
                          "[deploy]   밑창중심 대비 전후 %+.1f cm · 좌우 %+.1f cm  %s\n",
                          c.com_ref_xy[0], c.com_ref_xy[1], c.com_ref_z, ex*100, ey*100,
                          std::fabs(ex)<0.05 ? "✓지지면 안" : "⚠지지면 여유가 적다(반길이 7.3cm)"); }
            est.reset(Eigen::Vector3d(0,0,d->qpos[2]));
            // ★in-flight 토크도 같이 지운다. 안 지우면 직전 세션의 마지막 토크로
            //   첫 틱을 예측한다 — 무장 순간이 가장 위험한 자리다.
            std::fill(u_prev.begin(), u_prev.end(), 0.0);
            lc_n = lc_skip = 0; lc_warned = false;
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

    // ★트레이스 기록 — 무장 후 0.5초. 명령각은 hold/home 이 쓴 q_ch/hold_ch 다.
    if(trc){
      if(lt-trc_t0 <= 3.0){
        fprintf(trc,"%.4f", lt-trc_t0);
        for(int i=0;i<NCH;i++)
          fprintf(trc,",%.3f,%.1f,%.3f,%.3f", hs.q_deg[i], hs.dq_dps[i], hs.tau_nm[i],
                  (mode=="hold")? (double)hold_ch[i] : (double)q_ch[i]);
        fprintf(trc,"\n");
      } else { fclose(trc); trc=nullptr; std::printf("[deploy] 트레이스 저장 완료\n"); }
    }

    // ⑤ 모드 디스패치
    if(mode=="off"){
      for(int i=0;i<NCH;i++) q_ch[i]=0.f;
      qcmd_ch = q_ch; kpcmd_ch = zero; kdcmd_ch = zero;
      hw->write_pos(q_ch.data(), zero.data(), zero.data(), NCH);     // enable=0 → 브리지가 0 토크
    } else if(mode=="hold"){
      jm.kp_ch(kp_ch.data(), POS_KP); jm.kd_ch(kd_ch.data(), POS_KD);
      qcmd_ch = hold_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      hw->write_pos(hold_ch.data(), kp_ch.data(), kd_ch.data(), NCH);
    } else if(mode=="home"){
      // ★S-curve — 가감속이 0 에서 시작·끝나므로 속도트립을 만들지 않는다.
      double u = (home_T>0) ? (lt-home_t0)/home_T : 1.0;
      u = std::max(0.0, std::min(1.0, u));
      const double sf = u*u*(3.0-2.0*u);
      for(int i=0;i<NCH;i++)
        q_ch[i] = home_from[i] + (float)(sf*(double)(home_to[i]-home_from[i]));
      jm.kp_ch(kp_ch.data(), POS_KP); jm.kd_ch(kd_ch.data(), POS_KD);
      qcmd_ch = q_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      hw->write_pos(q_ch.data(), kp_ch.data(), kd_ch.data(), NCH);
      if(u>=1.0 && !home_done){
        home_done = true;
        std::printf("[deploy] home 도달 — 그 자세로 유지 중.\n"
                    "         다음: 크레인을 내려 접지 → **hold** 로 하중 이양 → stand\n");
      }
    } else if(mode=="jog"){
      // ★등속 램프로 **살아 있는 목표**를 따라간다. 목표는 GUI 의 jog_deg(모델각 deg).
      //   jog_deg 가 없으면(옛 GUI) 현재 램프값을 유지 — 갑자기 0 으로 끌려가지 않게.
      if(!jog_init){ jm.ch_to_q_joint(hs.q_deg.data(), jog_q.data()); jog_prev_t = lt; jog_init = true; }
      double el = lt - jog_prev_t; jog_prev_t = lt;
      el = std::max(0.0, std::min(el, 0.05));      // ⚠긴 정지 후 급이동 방지(Jogger.DT_CAP 과 동일)
      const double step = cfg.jog_speed_dps * el;
      if((int)cmd.jog_deg.size() >= NJ){
        for(int j=0;j<NJ;j++){
          double d0 = cmd.jog_deg[j] - jog_q[j];
          jog_q[j] += std::max(-step, std::min(step, d0));
        }
      }
      std::vector<double> qj = jog_q;
      jm.clamp_joint(qj.data());                   // 관절한계(모델각) — 채널이 아니라 여기서
      jm.q_joint_to_ch(qj.data(), q_ch.data());
      jm.kp_ch(kp_ch.data(), POS_KP); jm.kd_ch(kd_ch.data(), POS_KD);
      qcmd_ch = q_ch; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      hw->write_pos(q_ch.data(), kp_ch.data(), kd_ch.data(), NCH);
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

      // ★지연보상 — 마지막 명령토크를 유지한 채 LCOMP step 굴려 "지금"을 만든다.
      //   실패하면 **조용히 실측으로 되돌린다**(예측을 안 쓸 뿐, 제어는 계속된다).
      if(LCOMP>0){
        lc_n++;
        std::vector<double> pq(m->nq), pv(m->nv);
        if(dpred){                                   // (a) 동역학 롤아웃 — in-flight 토크 유지
          mju_copy(dpred->qpos, d->qpos, m->nq);
          mju_copy(dpred->qvel, d->qvel, m->nv);
          mju_zero(dpred->act, m->na); dpred->time = d->time;
          for(int l=0;l<LCOMP;l++){
            for(int i=0;i<NU;i++) dpred->ctrl[i] = u_prev[i];
            mj_step(m, dpred);
          }
          mju_copy(pq.data(), dpred->qpos, m->nq);
          mju_copy(pv.data(), dpred->qvel, m->nv);
        } else {                                     // (b) 운동학 외삽 — q += q̇·T (사실상 공짜)
          const double TT = lat_comp_ms/1000.0;
          mju_copy(pq.data(), d->qpos, m->nq);
          mju_copy(pv.data(), d->qvel, m->nv);        // 속도는 그대로(가속도를 안 쓴다)
          for(int i=0;i<3;i++) pq[i] += d->qvel[i]*TT;             // base 위치
          mju_quatIntegrate(pq.data()+3, d->qvel+3, TT);           // base 자세 ← 자이로
          for(int j=0;j<NJ;j++) pq[7+j] += d->qvel[6+j]*TT;        // 관절
        }
        // 가드 ① 유한성 ② 관절각 튐 ③ 몸통고도 튐.
        //   8.4ms 는 200dps 로 돌아도 1.7° 다 — 5° 를 넘으면 예측이 깨진 것이다.
        bool ok = true;
        for(int i=0;i<m->nq && ok;i++) ok = std::isfinite(pq[i]);
        for(int i=0;i<m->nv && ok;i++) ok = std::isfinite(pv[i]);
        if(ok) for(int j=0;j<NJ;j++)
          if(std::fabs(pq[7+j]-d->qpos[7+j])*JointMap::R2D > LC_MAX_DEG){ ok=false; break; }
        if(ok && std::fabs(pq[2]-d->qpos[2]) > LC_MAX_Z) ok = false;
        if(ok){
          mju_copy(d->qpos, pq.data(), m->nq);
          mju_copy(d->qvel, pv.data(), m->nv);
        } else {
          lc_skip++;
          // 한 번은 반드시 알린다 — 조용히 폴백하면 "보상이 켜져 있다" 고 착각한다.
          if(!lc_warned){ lc_warned = true;
            std::fprintf(stderr, "[deploy] ⚠지연보상 예측이 가드에 걸렸다 — 실측상태로 폴백."
                                 " (|Δq|>%.1f° 또는 |Δz|>%.0fmm 또는 비유한)\n",
                         LC_MAX_DEG, LC_MAX_Z*1e3); }
        }
      }
      mj_forward(m,d);
      c.com_ref_z = body_h;
      c.control(dt);

      // 토크 → 채널. ★tau_max_frac 로 한 번 더 클램프(컨트롤러 내부 클립과 별개의 상위 안전망).
      // ★2026-08-13 두 가지가 바뀌었다.
      //   ① 한계 출처: jnt_actfrcrange(+actuator_trnid 조회) → **actuator ctrlrange**.
      //      종전 주석이 "모델이 바뀌면 조용히 틀린 축의 토크한계를 쓴다" 고 경고했는데,
      //      발목을 tendon 액추에이터로 옮긴 것이 정확히 그 모델 변경이다 —
      //      trnid 가 관절이 아니라 **tendon id** 를 돌려줘 엉뚱한 관절 한계를 읽게 된다.
      //      ctrlrange 는 액추에이터당 하나라 조회 자체가 없어진다.
      //   ② d->ctrl 은 이제 **드라이브 토크**다. 한계도 드라이브 기준이라 여기서 바로 자른다.
      VectorXd u_drv(NU);
      for(int i=0;i<NU;i++){
        double lim = (m->actuator_ctrllimited[i] ? m->actuator_ctrlrange[i*2+1] : 0.0) * cfg.tau_max_frac;
        if(lim<=0) lim = 80.0;
        u_drv[i] = std::max(-lim, std::min(lim, d->ctrl[i]));
      }
      // ★다음 틱의 롤아웃이 유지할 in-flight 토크. **클램프 뒤** 값이어야 한다 —
      //   실제로 나가는 게 이것이고, 예측이 클램프 전 값을 쓰면 모델이 로봇보다 세진다.
      for(int i=0;i<NU;i++) u_prev[i] = u_drv[i];
      // ★관절토크로 되돌려서 넘긴다 — joint_map(tau_ctrl_to_ch)이 **자기가 전단**하므로
      //   드라이브 토크를 그대로 주면 전단이 두 번 걸려 τ_calf−2·τ_foot 이 나간다.
      VectorXd tj = bipedwbic::drive_to_tau(u_drv);
      for(int i=0;i<NU;i++) tau_ctrl[i] = tj[i];
      jm.q_ctrl_to_ch(q_ctrl.data(), q_ch.data());        // 위치/속도는 참고값(kp=kd=0 이라 무영향)
      jm.dq_ctrl_to_ch(dq_ctrl.data(), dq_ch.data());
      jm.tau_ctrl_to_ch(tau_ctrl.data(), tau_ch.data());
      // ★순수 토크: kp=kd=0. 드라이버가 tau_ff 만 실행한다.
      // ★블렌드: s=0 → 위치제어(hold 자세) · s=1 → 순수토크(WBIC)
      double sb = (stand_T>0) ? (lt-stand_t0)/stand_T : 1.0;
      sb = std::max(0.0, std::min(1.0, sb));
      const double bs = sb*sb*(3.0-2.0*sb);            // smoothstep
      // ★★**감쇠는 남긴다** (2026-08-20 실기). kp 는 0 으로 보내되 kd 는 유지한다.
      //   블렌드 끝(순수 kp=kd=0 토크)이 이 로봇에선 **불안정**했다 — 세 설정 모두에서
      //   위치게인이 빠지는 후반에 발산했다:
      //       지연ON·마찰ON  30.5Hz τmax 7.43 · 지연OFF 48.8Hz 16.12 · 마찰OFF 31.3Hz 13.21
      //       |dq| 6→12→19→44→196→322 dps (게인이 35% 아래로 내려간 뒤 폭발)
      //   8.4ms 지연은 30~65Hz 에서 위상지연 90~150° 라 순수 토크 피드백으로는 못 잡는다.
      //   ⚠kp 를 남기면 WBIC 와 싸운다(정지한 hold 자세로 되당긴다). kd 는 **속도만**
      //     억제하므로 목표와 싸우지 않고 그 대역만 먹는다. 그래서 kd 만 남긴다.
      //   calf 기준 kd_ch 3.5 = 관절 7.9 Nm·s/rad — 물리 감쇠(0)보다 압도적이다.
      //   STAND_KD_FLOOR 로 조절(0=종전 순수토크 · 1=설정 kd 전량).
      const double kdf = getenv("STAND_KD_FLOOR") ? atof(getenv("STAND_KD_FLOOR")) : 1.0;
      // ★★**kp 도 남긴다** (2026-08-20 실기: home 은 버티는데 stand 에서 처졌다).
      //   kd 는 **속도만** 잡으므로 처짐(정적 오차)에는 아무 도움이 안 된다.
      //   stand 는 kp=0 이라 관절 복원력이 **하나도 없고**, 자세를 잡는 건 모델 토크뿐이다:
      //       τ = h − Jᵀλ     h = **모델의** 중력항
      //   실제가 모델(13.9kg)보다 무겁거나 α(토크 스케일, ±10% 미검증)가 낮으면
      //   **토크가 모자라 처지는데 그걸 되돌릴 게 없다.**
      //   ⇒ feedforward 토크 + 관절 PD. 표준 구성이고, 2점 stand 는 PD 목표(stand_ref)와
      //     WBIC 목표가 **같은 자세**라 서로 싸우지 않는다.
      //   ⚠1점 보행(cmode=0)은 기본 0 이다 — 거기선 목표가 매 스텝 바뀌어 PD 가 방해한다.
      const double kpf = getenv("STAND_KP_FLOOR") ? atof(getenv("STAND_KP_FLOOR"))
                                                  : (c.cmode==1 ? 0.30 : 0.0);
      const double kd_scale = (1.0-bs) + bs*kdf;      // 블렌드 끝에서 kdf 로 수렴
      const double kp_scale = (1.0-bs) + bs*kpf;      // 블렌드 끝에서 kpf 로 수렴
      jm.kp_ch(kp_ch.data(), kp_scale); jm.kd_ch(kd_ch.data(), kd_scale);
      for(int i=0;i<NCH;i++) tau_ch[i] = (float)(bs*(double)tau_ch[i]);
      // 목표: 측정각 → 기하 자세로 블렌드와 **같은 계수**로 이동. bs=1 이면 순수 Qflat8.
      //   ⚠지금은 bs=1 에서 kp=0 이라 이 목표가 무영향이다. 그래도 측정각을 흘려보내지
      //     않는다 — STAND_KP_FLOOR 를 켜는 순간 **의미 있는 목표가 이미 들어가 있어야** 한다.
      //     측정각을 목표로 두면 err≡0(무효)이거나, 센서지연 때문에 err≈−q̇·τ = **음의 감쇠**가 된다.
      for(int i=0;i<NCH;i++)
        stand_ref[i] = stand_hold[i] + (float)(bs*(double)(stand_to[i]-stand_hold[i]));
      qcmd_ch = stand_ref; kpcmd_ch = kp_ch; kdcmd_ch = kd_ch;
      hw->write_mit(stand_ref.data(), zero.data(), tau_ch.data(),
                    kp_ch.data(), kd_ch.data(), NCH);
    }

    // ⑥ 상태 발행(~20Hz)
    double period = lt - prev_loop; prev_loop = lt;
    if(period > 0) hz_ema = 0.98*hz_ema + 0.02*(1.0/period);
    if(lt - last_pub > 0.05){
      last_pub = lt;
      int n_ok=0, n_fault=0, n_dead=0, n_absent=0;
      // ★ucStatus(=ERROR VECTOR 하위 8비트)를 **원값 그대로** 낸다 (2026-08-20).
      //   종전엔 "fault" 라는 라벨만 나가서, 무엇 때문에 걸렸는지 알 수 없었다.
      //   과전류·과온·엔코더·통신 중 뭔지 알아야 조치가 갈린다.
      // 명령각을 **모델각**으로 바꿔 낸다(측정 q_leg_deg 와 같은 좌표라 바로 비교된다)
      std::string qcmds="[", dqcmds="[";
      { std::vector<double> qc(jm.n_leg); jm.ch_to_q_joint(qcmd_ch.data(), qc.data());
        for(int i2=0;i2<jm.n_leg;i2++){ char b[24];
          std::snprintf(b,sizeof b,"%s%.3f", i2?",":"", qc[i2]); qcmds+=b;
          std::snprintf(b,sizeof b,"%s0.0", i2?",":""); dqcmds+=b; }
        qcmds+="]"; dqcmds+="]"; }
      std::string errs="[";
      std::string health="[", inst="[", qs="[", qchs="[";
      // ★2026-08-13 `q_leg_deg` 에 **채널각**을 넣고 있었다(모델각 계약 위반).
      //   emb/interface/state_pub.py 규약: q_leg_deg = **모델각**(MJCF qpos 와 같은 좌표계).
      //   채널각을 넣으면 뷰어·모니터가 **틀린 자세를 그린다** — gear_k(calf 1.5·foot 1.2)와
      //   커플링만큼 어긋난다. 실기에서 자세를 화면으로 보며 판단하므로 직격이다.
      //   ⇒ 여기서 한 번 변환하고, 채널각은 `q_ch_deg` 로 따로 낸다(캘리브레이션·진단용).
      //   ⚠`q_ch_deg` 가 없으면 diag/couple_check.py 가 "구버전" 이라며 거부한다.
      std::vector<double> q_leg(jm.n_leg);
      jm.ch_to_q_joint(hs.q_deg.data(), q_leg.data());
      for(int i=0;i<jm.n_leg;i++){
        int ch = cfg.joints[i].channel;
        bool ins = cfg.installed_has(ch);
        const char* h = !ins ? "absent"
                      : (ch>=(int)hs.connected.size() || !hs.connected[ch]) ? "dead"
                      : (ch<(int)hs.status.size() && hs.status[ch]!=0) ? "fault" : "ok";
        if(!ins) n_absent++; else if(!std::strcmp(h,"ok")) n_ok++;
        else if(!std::strcmp(h,"fault")) n_fault++; else n_dead++;
        { char b[16]; std::snprintf(b,sizeof b,"%s%d", i?",":"",
            ch<(int)hs.status.size()? hs.status[ch] : -1); errs += b; }
        char b[64];
        std::snprintf(b,sizeof b,"%s\"%s\"", i?",":"", h); health += b;
        std::snprintf(b,sizeof b,"%s%s", i?",":"", ins?"true":"false"); inst += b;
        std::snprintf(b,sizeof b,"%s%.2f", i?",":"", q_leg[i]);      qs   += b;   // 모델각
        std::snprintf(b,sizeof b,"%s%.2f", i?",":"", hs.q_deg[ch]);  qchs += b;   // 채널각
      }
      health+="]"; inst+="]"; qs+="]"; qchs+="]";
      // ★모니터링용 측정·명령 (2026-08-13). Python app(biped_emb.py)과 **같은 키·같은 단위**로 낸다
      //   — 모니터가 두 제어기를 구분하지 않아도 되게. 모델각[deg·deg/s] · 관절토크[Nm].
      //   ⚠이 경로는 **순수 토크모드**(kp=kd=0)라 위치·속도 "명령" 이 없다. q_ch/dq_ch 는
      //     측정에서 만든 참고값이라 명령으로 내보내면 거짓말이 된다 ⇒ 토크만 명령으로 낸다.
      //     kp/kd 를 0 으로 함께 내보내 모니터가 "위치명령 없음" 을 알 수 있게 한다.
      std::vector<double> tau_meas(jm.n_leg);
      jm.ch_to_tau_joint(hs.tau_nm.data(), tau_meas.data());
      std::string dqs="[", taus="[", taucs="[", kps="[", kds="[";
      for(int i=0;i<jm.n_leg;i++){
        char b[64]; const char* sep = i?",":"";
        std::snprintf(b,sizeof b,"%s%.2f", sep, dq_ctrl[i]*JointMap::R2D); dqs   += b;  // 측정 속도
        std::snprintf(b,sizeof b,"%s%.3f", sep, tau_meas[i]);             taus  += b;  // 측정 토크
        std::snprintf(b,sizeof b,"%s%.3f", sep, tau_ctrl[i]);             taucs += b;  // 명령 토크
        // ★채널게인 → **관절게인**: kp_joint = kp_ch · gear_k²  (emb/README "게인도 좌표가 둘")
        //   측정·명령이 전부 모델각이므로 게인도 관절 좌표여야 같이 읽힌다.
        { const auto& jj = cfg.joints[i]; const double k2 = jj.gear_k*jj.gear_k;
          std::snprintf(b,sizeof b,"%s%.2f", sep, (double)kpcmd_ch[jj.channel]*k2); kps += b; }
        { const auto& jj = cfg.joints[i]; const double k2 = jj.gear_k*jj.gear_k;
          std::snprintf(b,sizeof b,"%s%.2f", sep, (double)kdcmd_ch[jj.channel]*k2); kds += b; }
      }
      dqs+="]"; taus+="]"; taucs+="]"; kps+="]"; kds+="]";
      // ★창 통계(500Hz 누적) → 발행. 창이 비면(첫 틱) 빈 배열 대신 0 을 낸다.
      std::string tsd="[", tmn="[", tmx="[";
      for(int i=0;i<jm.n_leg;i++){
        char b[64]; const char* sep = i?",":"";
        double mu = ts_n? ts_sum[i]/ts_n : 0.0;
        double sd = ts_n? std::sqrt(std::max(0.0, ts_sq[i]/ts_n - mu*mu)) : 0.0;
        std::snprintf(b,sizeof b,"%s%.3f", sep, sd); tsd += b;
        std::snprintf(b,sizeof b,"%s%.3f", sep, ts_n? ts_min[i]:0.0); tmn += b;
        std::snprintf(b,sizeof b,"%s%.3f", sep, ts_n? ts_max[i]:0.0); tmx += b;
      }
      tsd+="]"; tmn+="]"; tmx+="]";
      const long ts_n_pub = ts_n;
      ts_reset();                       // 창을 비운다 — 다음 발행까지 다시 쌓는다
      char buf[4096];   // ★3072 → 4096 (2026-08-20): 토크 창통계 3배열(std·min·max) 추가
      std::snprintf(buf,sizeof buf,
        "{\"mode\":\"%s\",\"backend\":\"%s\",\"q_leg_deg\":%s,\"q_ch_deg\":%s,"
        "\"dq_leg_dps\":%s,\"tau_leg_nm\":%s,\"tau_cmd_nm\":%s,\"kp_leg\":%s,\"kd_leg\":%s,"
        // ★창 통계 — **500Hz 로 계산**한 값이다(발행 20Hz 표본이 아니라). tau_win_n 은
        //   그 창에 들어간 표본 수 = 통계의 신뢰도. 0 이면 통계를 읽지 말 것.
        "\"tau_std_nm\":%s,\"tau_min_nm\":%s,\"tau_max_nm\":%s,\"tau_win_n\":%ld,"
        "\"rpy_deg\":[%.2f,%.2f,%.2f],\"tilt_deg\":%.2f,\"loop_hz\":%.1f,"
        "\"motors_on\":%s,\"health\":%s,\"installed\":%s,"
        "\"n_ok\":%d,\"n_fault\":%d,\"n_dead\":%d,\"n_absent\":%d,\"n_installed\":%d,"
        "\"est_x\":%.3f,\"est_z\":%.3f,\"estop\":%s,\"tilt_estop_ok\":%s,"
        // ★WBIC QP 건강도 — **접지 판정의 유일한 지표**(biped_control.hpp 주석 참조).
        //   발이 덜 닿으면 QP 가 매 틱 실패하고 중력보상 폴백으로 떨어지는데,
        //   겉보기엔 안정돼 보인다. 이 셋이 그 상태를 드러낸다.
        // ★지연보상 상태. `lat_comp_ms`=0 이면 **꺼져 있다**(env 미설정).
        //   `lc_skip_pct`>0 은 예측이 가드에 걸려 실측으로 폴백 중이라는 뜻 —
        //   켜 놓고도 실제로는 안 걸린 상태라 반드시 보인다.
        "\"qp_fail_pct\":%.1f,\"qp_K\":%d,\"qp_cerr\":[%.4f,%.4f,%.4f],"
        // ★home 진행률(0~1) — GUI 가 이미 표시할 준비가 돼 있다(teleop_gui_biped:526).
        //   10초짜리 램프라 운전자가 "지금 얼마나 갔나" 를 볼 수 있어야 한다.
        "\"lat_comp_ms\":%.2f,\"lc_skip_pct\":%.1f,\"home_progress\":%.3f,\"err\":%s,"
        "\"q_cmd_deg\":%s,\"dq_cmd_dps\":%s}",
        mode.c_str(), hw->name(), qs.c_str(), qchs.c_str(),
        /* dq/tau/tau_cmd/kp/kd 는 다음 줄에서 이어진다 — 아래 5개 뒤에 창통계 4개 */
        dqs.c_str(), taus.c_str(), taucs.c_str(), kps.c_str(), kds.c_str(),
        tsd.c_str(), tmn.c_str(), tmx.c_str(), ts_n_pub,
        rpy[0]*JointMap::R2D, rpy[1]*JointMap::R2D,
        rpy[2]*JointMap::R2D, tilt, hz_ema, (mode!="off"&&!wd)?"true":"false",
        health.c_str(), inst.c_str(), n_ok, n_fault, n_dead, n_absent,
        (int)(jm.n_leg-n_absent), est.p[0], est.p[2], estop?"true":"false",
        imu_dead?"false":"true",
        c.qp_rate*100.0, c.qp_K, c.qp_cerr[0], c.qp_cerr[1], c.qp_cerr[2],
        LCOMP>0 ? lat_comp_ms : 0.0, lc_n ? 100.0*(double)lc_skip/(double)lc_n : 0.0,
        (mode=="home" && home_T>0) ? std::max(0.0,std::min(1.0,(lt-home_t0)/home_T)) : 0.0,
        (errs+"]").c_str(), qcmds.c_str(), dqcmds.c_str());
      write_state(stt_p, buf);
    }

    // ⑦ 실시간 페이싱(절대시각 — 누적 드리프트 없음)
    //   ★밀렸을 때 재동기: Pi 부하로 루프가 느려지면 t0+k·dt 가 현재보다 한참 과거가 되고,
    //     부하가 풀리는 순간 sleep 없이 수백 틱을 몰아쳐 돈다(제어주기 붕괴).
    //     10 틱 이상 밀리면 스케줄을 현재로 되맞추고, 지속되면 한 번 경고한다.
    k++;
    double lag = now_s() - (t0 + k*dt);
    if(lag > 10*dt){
      if(!overrun_warned){ overrun_warned = true;
        std::fprintf(stderr, "[deploy] ⚠ 루프가 %.0fms 밀렸다 — 제어주기 %.0fHz 를 못 지키고 있다."
                             " CPU 부하/우선순위 확인.\n", lag*1e3, cfg.ctrl_hz); }
      t0 = now_s() - k*dt;                 // 스케줄 재동기(캐치업 스핀 방지)
    }
    sleep_s(t0 + k*dt - now_s());
  }

  if(g_stop) std::printf("\n[deploy] 신호 %d 수신 → 안전종료\n", (int)g_stop);
  safe_shutdown(*hw, NCH);
  if(LCOMP>0 && lc_n)
    std::printf("[deploy] 지연보상 %.2fms — %ld 틱 중 %ld 폴백(%.1f%%)\n",
                lat_comp_ms, lc_n, lc_skip, 100.0*(double)lc_skip/(double)lc_n);
  if(dpred) mj_deleteData(dpred);
  delete hw; mj_deleteData(d); mj_deleteModel(m);
  return rc;
}
