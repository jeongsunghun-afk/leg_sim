#pragma once
// deploy_hw.hpp — 실기 배포용 하드웨어 계층 (§9, 핸드오프 미완료 #4).
//
//   [설정] emb/config/biped_emb.yaml 을 **Python 앱과 같은 파일**에서 읽는다.
//          다음 주 JOG 캘리브레이션이 그 파일의 sign/offset 을 고칠 텐데, C++ 가 따로
//          하드코딩하고 있으면 그때부터 두 실행체가 다른 로봇을 제어하게 된다.
//          → 의존성 없이 이 파일의 flow-mapping 서브셋만 파싱한다.
//   [I/O]  HwIface : ShmHw(libbipedshm) / MockHw(하드웨어 없는 데스크톱 검증)
//   [단위] SHM = deg · deg/s · Nm       컨트롤러 = rad · rad/s · Nm
//          q_ch = sign·rad2deg(q_ctrl) + offset      (joint_map.py 규약과 동일)
//          ⚠Kp/Kd 는 **Nm/rad 그대로 전달**한다(환산 금지). 드라이버 MIT 법칙이
//            tau = Kp·err[rad] + Kd·derr[rad/s] 라 이미 rad 기준이다.
//   [JSON] /tmp/biped_cmd.json 읽기 · /tmp/biped_state.json 쓰기 — 기존 GUI 와 호환.
//
// ⚠ 이 파일은 **실기 미검증**이다(2026-08-06 Emb 미기동). MockHw 로 로직만 검증했다.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace bipedhw {

// ─────────────────────────────────────────────────────────────────────────────
// 1. 설정 (biped_emb.yaml 서브셋)
// ─────────────────────────────────────────────────────────────────────────────
struct JointCfg {
  std::string name; int channel=0; double sign=1, offset_deg=0;
  // ★gear_k = 드라이버 감속비 오설정 보정(실제감속비/드라이버가정 7).
  //   hip·thigh 1.0 · calf 10.5/7=1.5 · foot 8.4/7=1.2. 미지정이면 1.0 = 종전 동작.
  //   sk() 로만 쓸 것 — sign 과 k 를 따로 곱하다 한쪽을 빠뜨리는 실수를 막는다.
  double gear_k=1;
  double sk() const { return sign*gear_k; }
  // ★기구 커플링: 이 관절이 couple_from 관절에 종속(발목이 링키지 구동).
  //   모델각 = raw − coef·모델각_src   ·   raw = 모델각 + coef·모델각_src
  std::string couple_from; double couple_coef=1; int couple_src=-1;   // src 는 load 시 해석
  double min_deg=-30, max_deg=30, kp=40, kd=2;
};

struct EmbCfg {
  // shm
  std::string lib = "";
  int n_channel = 10, recv_wait_ms = 2000;
  bool imu_deg = true;
  double ctrl_hz = 500;
  std::vector<JointCfg> joints;
  std::vector<int> installed;                 // meta.installed_channels
  // waist
  std::vector<int> waist_ch; std::vector<double> waist_hold;
  double waist_kp = 10, waist_kd = 0.5;
  // safety
  double tilt_estop_deg = 40, watchdog_ms = 500, tau_max_frac = 0.6;
  double tau_trip_nm = 8.0, tau_trip_ms = 50, vel_trip_dps = 200.0, vel_trip_ms = 20.0;
  // home 램프 — biped_emb.py 의 HomeTrajectory 와 **같은 설정 키**를 읽는다.
  //   두 제어기가 같은 자세로 같은 속도로 가야 운전자가 헷갈리지 않는다.
  double home_speed_dps = 15.0, home_acc_dps2 = 30.0, home_min_time_s = 0.6;
  // ★★home 목표자세 = `home.q_deg`(모델각 deg). biped_emb.py 와 **같은 값을 읽는다**.
  //   2026-08-21 신설. 종전 deploy 는 이 키를 안 읽고 MJCF 기하에서 뽑은 Qhome8
  //   (thigh +11.6° · calf −38.5°) 로 갔다. 그래서 같은 HOME 버튼인데 두 제어기가
  //   **다른 자세**로 갔다 — "1점 점발에서 홈이 이상하게 움직인다" 의 정체다.
  //   ⚠비었으면(키 없음) 종전대로 Qhome8 로 폴백한다. 2점 평발(cmode=1)은 이 값과
  //     무관하게 Qflat8 로 간다 — 거긴 발바닥이 지면과 평행해야 하는 **기하 조건**이고,
  //     0° 자세로 가면 stand 진입검사(15°)에 걸린다.
  std::vector<double> home_q_deg;
  double home_settle_deg = 0.5;      // 도달 판정[deg] — **측정각** 기준(biped_emb.py 와 동일)
  double jog_speed_dps  = 20.0;      // ★jog 등속 램프 속도(모델각 deg/s) — biped_emb.py 와 같은 키

  bool installed_has(int ch) const {
    return installed.empty() || std::find(installed.begin(), installed.end(), ch) != installed.end();
  }
};

// `{a: 1, b: xy}` 한 줄에서 key→value 문자열 추출. YAML flow-mapping 서브셋 전용.
inline std::map<std::string,std::string> parse_flow(const std::string& line){
  std::map<std::string,std::string> kv;
  auto b = line.find('{'), e = line.rfind('}');
  if(b==std::string::npos || e==std::string::npos || e<b) return kv;
  std::string in = line.substr(b+1, e-b-1);
  std::stringstream ss(in); std::string tok;
  while(std::getline(ss, tok, ',')){
    auto c = tok.find(':'); if(c==std::string::npos) continue;
    auto trim=[](std::string s){
      size_t i=s.find_first_not_of(" \t"); size_t j=s.find_last_not_of(" \t");
      return (i==std::string::npos)? std::string() : s.substr(i, j-i+1); };
    kv[trim(tok.substr(0,c))] = trim(tok.substr(c+1));
  }
  return kv;
}

// `key: value   # comment` 에서 value 만. 주석·따옴표 제거.
inline std::string scalar_of(const std::string& line){
  auto c = line.find(':'); if(c==std::string::npos) return "";
  std::string v = line.substr(c+1);
  auto h = v.find('#'); if(h!=std::string::npos) v = v.substr(0,h);
  size_t i=v.find_first_not_of(" \t\r"); size_t j=v.find_last_not_of(" \t\r");
  if(i==std::string::npos) return "";
  v = v.substr(i, j-i+1);
  if(v.size()>=2 && (v.front()=='"'||v.front()=='\'')) v = v.substr(1, v.size()-2);
  return v;
}

// `[0, 4]` → {0,4}
inline std::vector<double> parse_list(const std::string& v){
  std::vector<double> out; std::string s=v;
  auto b=s.find('['), e=s.rfind(']'); if(b==std::string::npos) return out;
  s = s.substr(b+1, e-b-1);
  std::stringstream ss(s); std::string t;
  while(std::getline(ss,t,',')) { try{ out.push_back(std::stod(t)); }catch(...){} }
  return out;
}

inline void resolve_couple(EmbCfg& c){
  for(size_t i=0;i<c.joints.size();i++){
    auto& j=c.joints[i];
    if(j.couple_from.empty()){ j.couple_src=-1; continue; }
    int s=-1; for(size_t m=0;m<c.joints.size();m++) if(c.joints[m].name==j.couple_from) s=(int)m;
    if(s<0) throw std::runtime_error(j.name+".couple_from not found: "+j.couple_from);
    if(s==(int)i) throw std::runtime_error(j.name+" coupled to itself");
    if(!c.joints[s].couple_from.empty()) throw std::runtime_error("couple chain unsupported: "+j.name);
    j.couple_src=s;
  }
}

inline bool load_cfg(const std::string& path, EmbCfg& c, std::string& err){
  std::ifstream f(path);
  if(!f){ err = "설정 파일을 열 수 없다: " + path; return false; }
  std::string line, section;
  auto getd=[&](const std::map<std::string,std::string>& kv, const char* k, double dv){
    auto it=kv.find(k); if(it==kv.end()) return dv; try{ return std::stod(it->second); }catch(...){ return dv; } };
  while(std::getline(f,line)){
    if(line.empty()) continue;
    std::string t = line; auto h=t.find('#'); if(h!=std::string::npos) t=t.substr(0,h);
    if(t.find_first_not_of(" \t\r")==std::string::npos) continue;
    bool top = (t[0]!=' ' && t[0]!='\t' && t[0]!='-');
    if(top){ section = t.substr(0, t.find(':')); continue; }

    auto trimmed = t.substr(t.find_first_not_of(" \t"));
    if(trimmed.rfind("- {",0)==0){                       // joints 리스트 항목
      if(section!="joints") continue;
      auto kv = parse_flow(trimmed);
      JointCfg j;
      j.name = kv.count("name")?kv["name"]:"";
      j.channel    = (int)getd(kv,"channel",0);
      j.sign       = getd(kv,"sign",1);
      j.offset_deg = getd(kv,"offset_deg",0);
      j.gear_k     = getd(kv,"gear_k",1);
      j.couple_from= kv.count("couple_from")? kv["couple_from"] : std::string();
      j.couple_coef= getd(kv,"couple_coef",1);
      if(j.gear_k <= 0) throw std::runtime_error("gear_k must be > 0: "+j.name);
      j.min_deg    = getd(kv,"min_deg",-30);
      j.max_deg    = getd(kv,"max_deg",30);
      j.kp         = getd(kv,"kp",40);
      j.kd         = getd(kv,"kd",2);
      c.joints.push_back(j);
      continue;
    }
    auto key = trimmed.substr(0, trimmed.find(':'));
    std::string val = scalar_of(trimmed);
    auto num=[&](double dv){ try{ return std::stod(val);}catch(...){ return dv; } };
    if(section=="shm"){
      if(key=="lib") c.lib = val;
      else if(key=="n_channel") c.n_channel = (int)num(10);
      else if(key=="recv_wait_ms") c.recv_wait_ms = (int)num(2000);
      else if(key=="imu_deg") c.imu_deg = (val=="true"||val=="True"||val=="1");
    } else if(section=="meta"){
      if(key=="ctrl_hz") c.ctrl_hz = num(500);
      else if(key=="installed_channels"){ for(double x : parse_list(val)) c.installed.push_back((int)x); }
    } else if(section=="waist"){
      if(key=="channels"){ for(double x : parse_list(val)) c.waist_ch.push_back((int)x); }
      else if(key=="hold_deg") c.waist_hold = parse_list(val);
      else if(key=="kp") c.waist_kp = num(10);
      else if(key=="kd") c.waist_kd = num(0.5);
    } else if(section=="safety"){
      if(key=="tilt_estop_deg") c.tilt_estop_deg = num(40);
      else if(key=="watchdog_ms") c.watchdog_ms = num(500);
      else if(key=="tau_max_frac") c.tau_max_frac = num(0.6);
      else if(key=="tau_trip_nm") c.tau_trip_nm = num(8.0);
      else if(key=="tau_trip_ms") c.tau_trip_ms = num(50);
      else if(key=="vel_trip_dps") c.vel_trip_dps = num(200.0);
      else if(key=="vel_trip_ms")  c.vel_trip_ms  = num(20.0);   // ★2026-08-24 — 종전엔 안 읽었다
    // ★★2026-08-21 `jog:`·`home:` 을 **제 블록에서** 읽는다.
    //   종전엔 이 세 키가 `safety` 분기 안에 있었다. 그런데 YAML 에서 그것들은 `jog:`·`home:`
    //   아래에 있으므로 **한 번도 읽히지 않았다** — 늘 기본값으로 떨어졌다.
    //   `home.max_speed_dps` 가 마침 기본값과 같은 15.0 이라 증상이 안 보였을 뿐이다
    //   (`jog.max_speed_dps: 20.0` 은 그냥 무시되고 있었다).
    //   ⚠section 추적은 파서에 이미 있었다(top-level 키에서 갱신). 분기만 잘못 놓였던 것이다.
    } else if(section=="jog"){
      if(key=="max_speed_dps") c.jog_speed_dps = num(20.0);
    } else if(section=="home"){
      if(key=="max_speed_dps") c.home_speed_dps = num(15.0);
      else if(key=="max_acc_dps2")  c.home_acc_dps2  = num(30.0);
      else if(key=="min_time_s")    c.home_min_time_s= num(0.6);
      else if(key=="q_deg")         c.home_q_deg     = parse_list(val);
      else if(key=="settle_deg")    c.home_settle_deg= num(0.5);
    }
  }
  if(c.joints.empty()){ err = "joints 를 하나도 못 읽었다 — 설정 형식 확인: " + path; return false; }
  try { resolve_couple(c); } catch(const std::exception& e){ err = e.what(); return false; }
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. 하드웨어 인터페이스
// ─────────────────────────────────────────────────────────────────────────────
struct HwState {
  std::vector<float> q_deg, dq_dps, tau_nm, cur_a;
  float rpy[3]={0,0,0}, acc[3]={0,0,0}, gyr[3]={0,0,0};
  std::vector<int> connected, status;
  int mask = 0;
};

struct HwIface {
  virtual ~HwIface(){}
  virtual bool init(int recv_wait_ms) = 0;
  virtual int  read(HwState& s) = 0;
  virtual int  write_pos(const float* q_des, const float* kp, const float* kd, int n) = 0;
  virtual int  write_mit(const float* q_des, const float* dq_des, const float* tau_ff,
                         const float* kp, const float* kd, int n) = 0;
  virtual int  enable(int on) = 0;
  virtual const char* name() const = 0;
};

// ── 실기: libbipedshm.so 동적 로드(dlopen) ───────────────────────────────────
//   ★정적 링크가 아니라 dlopen 을 쓰는 이유: 이 바이너리는 노트북에서도 빌드돼야 하는데
//     libbipedshm 은 Pi 전용(RobotSharedMem 필요)이다. 링크 시점 의존을 만들면
//     노트북 빌드가 통째로 깨진다. 실행 시점에만 필요하게 둔다.
struct ShmHw : HwIface {
  void* h=nullptr;
  int (*p_init)(int)=nullptr;
  int (*p_read)(float*,float*,float*,float*,float*,float*,float*,int*,int*)=nullptr;
  int (*p_wpos)(const float*,const float*,const float*,int)=nullptr;
  int (*p_wmit)(const float*,const float*,const float*,const float*,const float*,int)=nullptr;
  int (*p_en)(int)=nullptr;
  int n=10;
  std::string lib, err;

  ShmHw(const std::string& lib_, int nch): n(nch), lib(lib_) {}
  ~ShmHw() override;
  bool init(int recv_wait_ms) override;
  int read(HwState& s) override {
    s.q_deg.resize(n); s.dq_dps.resize(n); s.tau_nm.resize(n); s.cur_a.resize(n);
    s.connected.resize(n); s.status.resize(n);
    s.mask = p_read(s.q_deg.data(), s.dq_dps.data(), s.tau_nm.data(), s.cur_a.data(),
                    s.rpy, s.acc, s.gyr, s.connected.data(), s.status.data());
    return s.mask;
  }
  int write_pos(const float* q,const float* kp,const float* kd,int nn) override { return p_wpos(q,kp,kd,nn); }
  int write_mit(const float* q,const float* dq,const float* tf,const float* kp,const float* kd,int nn) override {
    return p_wmit(q,dq,tf,kp,kd,nn); }
  int enable(int on) override { return p_en(on); }
  const char* name() const override { return "shm"; }
};

// ── 데스크톱 검증용 mock (hal/mock_backend.py 와 같은 개념) ──────────────────
//   명령 위치로 속도제한 램프. enable=false 면 정지. kp=0 채널은 안 움직임(미배선 재현).
struct MockHw : HwIface {
  int n; double dt, max_step;
  std::vector<float> q, dq, q_des, kp;
  bool on=false;
  // ★고장 주입 — **안전장치를 실제로 발화시켜 검증**하기 위한 것.
  //   검증 안 된 E-stop 은 없느니만 못하다(있다고 착각하게 만든다).
  //   env: FAULT_AT_S(주입 시각) · FAULT_TILT_DEG · FAULT_TAU_NM · FAULT_VEL_DPS
  double t=0, fault_at=1e9, f_tilt=0, f_tau=0, f_vel=0;
  bool f_imu_dead=false;              // FAULT_IMU_DEAD=1 → IMU 전부 0(실기 현상 재현)
  MockHw(int nch, double dt_, double max_dps=60.0)
    : n(nch), dt(dt_), max_step(max_dps*dt_), q(nch,0.f), dq(nch,0.f), q_des(nch,0.f), kp(nch,0.f) {
    auto ev=[](const char* k, double dv){ const char* v=getenv(k); return v? atof(v) : dv; };
    fault_at = ev("FAULT_AT_S", 1e9);
    f_tilt = ev("FAULT_TILT_DEG",0); f_tau = ev("FAULT_TAU_NM",0); f_vel = ev("FAULT_VEL_DPS",0);
    f_imu_dead = ev("FAULT_IMU_DEAD",0) != 0;
  }
  bool init(int) override {
    std::printf("[MockHw] n_channel=%d dt=%.4f (SHM 없음 — 로직 검증용)\n", n, dt);
    if(fault_at<1e8) std::printf("[MockHw] 고장 주입 @%.1fs: tilt=%.0f° tau=%.1fNm vel=%.0fdps\n",
                                 fault_at, f_tilt, f_tau, f_vel);
    if(f_imu_dead) std::printf("[MockHw] 고장 주입: IMU 사망(전부 0)\n");
    return true; }
  int read(HwState& s) override {
    if(on){
      for(int i=0;i<n;i++){
        float e = (kp[i]>0)? (q_des[i]-q[i]) : 0.f;
        float st = std::max((float)-max_step, std::min((float)max_step, e));
        q[i]+=st; dq[i]=st/(float)dt;
      }
    } else for(int i=0;i<n;i++) dq[i]=0.f;
    s.q_deg=q; s.dq_dps=dq; s.tau_nm.assign(n,0.f); s.cur_a.assign(n,0.f);
    s.connected.assign(n,1); s.status.assign(n,0);
    s.rpy[0]=s.rpy[1]=s.rpy[2]=0; s.acc[0]=0; s.acc[1]=0; s.acc[2]=9.81f;
    s.gyr[0]=s.gyr[1]=s.gyr[2]=0;
    // ★실기 현상 재현: fIMUBuf 전부 0 인데 IsUpdatedIMU()=1 ("신선한 0").
    //   중력조차 안 잡히는 게 죽은 IMU 의 표식이다(emb/IMU_RECOVERY.md).
    if(f_imu_dead){ s.acc[0]=s.acc[1]=s.acc[2]=0.f; }
    t += dt;
    if(t >= fault_at){                       // 고장 주입(E-stop 발화 검증용)
      if(f_tilt>0) s.rpy[0] = (float)f_tilt;              // roll 로 tilt 만듦
      if(f_tau >0) s.tau_nm[0] = (float)f_tau;
      if(f_vel >0) s.dq_dps[0] = (float)f_vel;
    }
    s.mask = 0x11; return s.mask;
  }
  int write_pos(const float* qd,const float* k,const float*,int nn) override {
    if(on) for(int i=0;i<std::min(nn,n);i++){ q_des[i]=qd[i]; kp[i]=k[i]; } return 0; }
  int write_mit(const float* qd,const float*,const float*,const float* k,const float*,int nn) override {
    if(on) for(int i=0;i<std::min(nn,n);i++){ q_des[i]=qd[i]; kp[i]=k[i]; } return 0; }
  int enable(int o) override { on = o!=0; return 0; }
  const char* name() const override { return "mock"; }
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. 관절 매핑 (joint_map.py 규약)
// ─────────────────────────────────────────────────────────────────────────────
struct JointMap {
  const EmbCfg* c;
  int n_leg=0, n_channel=10;
  explicit JointMap(const EmbCfg& cfg): c(&cfg), n_leg((int)cfg.joints.size()), n_channel(cfg.n_channel) {}

  static constexpr double R2D = 57.29577951308232;
  static constexpr double D2R = 0.017453292519943295;

  // ── 모델각[deg] ↔ 채널각[deg] — Python joint_map 과 **같은 규약** ────────
  //   모델각 = (채널각 − offset)/(sign·k)        채널각 = 모델각·sign·k + offset
  //   ★k(gear_k) = 드라이버 감속비 오설정 보정. 2026-08-10 도입 — offset 만으로는
  //     기준자세 **한 점**만 맞고 기울기가 틀렸다(calf 1.5배·foot 1.2배 과대표시).
  //   ⚠토크만 반대로 **나눈다**: 보고토크 = 실제관절토크/k → 명령은 τ/k.
  //   ★k 는 실기 검증됨(2026-08-10, 육안). 상수 k 로 전 구간이 맞는다는 것이
  //     링키지가 아니라 **드라이버 고정 7:1** 임을 뒷받침한다. 근본 해결은 드라이버(RGA).
  //   GUI·jog·home·hold·표시는 전부 모델각. 채널각은 SHM 경계에서만.
  void ch_to_q_joint(const float* q_ch, double* out) const {
    std::vector<double> raw(n_leg);
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      raw[i] = ((double)q_ch[j.channel] - j.offset_deg)/j.sk(); }
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[i] = (j.couple_src<0)? raw[i] : raw[i] - j.couple_coef*raw[j.couple_src]; }
  }
  // ★±180 포화 — Python joint_map.q_joint_to_ch 와 **반드시 같아야 한다**.
  //   Emb 는 초과분을 클램프가 아니라 **래핑**한다(halGait.cpp:666-671,
  //   while(fPosition>180) fPosition-=360) → 181° 가 −179° 로 뒤집혀 반대편으로 날아간다.
  //   ⚠2026-08-10: Python 에만 넣고 여기를 빠뜨렸다. 관절한계 박스 꼭짓점에서
  //     py 180.00 vs cpp 240.09 로 **60° 어긋났다**. 당시 패리티 시험이 기준자세
  //     하나만 봐서 통과했다 — 그래서 biped_parity 를 만들어 한계 박스까지 훑는다.
  static constexpr double WRAP_DEG = 180.0;
  mutable long n_saturated = 0;               // 포화 발생 횟수(진단용)
  mutable int  last_saturated_ch = -1;

  void q_joint_to_ch(const double* q_j, float* out) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      const double raw = (j.couple_src<0)? q_j[i] : q_j[i] + j.couple_coef*q_j[j.couple_src];
      double v = raw*j.sk() + j.offset_deg;
      if(v >  WRAP_DEG){ v =  WRAP_DEG; n_saturated++; last_saturated_ch = j.channel; }
      if(v < -WRAP_DEG){ v = -WRAP_DEG; n_saturated++; last_saturated_ch = j.channel; }
      out[j.channel] = (float)v; }
    for(size_t k=0;k<c->waist_ch.size();k++)
      out[c->waist_ch[k]] = (float)(k<c->waist_hold.size()? c->waist_hold[k] : 0.0);
  }
  void clamp_joint(double* q_j) const {
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      q_j[i] = std::max(j.min_deg, std::min(j.max_deg, q_j[i])); }
  }

  // 컨트롤러(rad) → 채널(deg). 허리는 hold 각으로 고정.
  // ★q_joint_to_ch 에 **위임**한다(2026-08-10). 종전엔 같은 식을 두 벌 갖고 있어
  //   ±180 포화를 q_joint_to_ch 에만 넣었을 때 이 함수가 조용히 갈라졌다
  //   (한계 박스 내부에서 py −180.00 vs cpp −234.88, 최대 54.9° 차).
  //   Python 도 같은 이유로 위임 구조다(joint_map.py:q_ctrl_to_ch).
  void q_ctrl_to_ch(const double* q_rad, float* out) const {
    std::vector<double> q_deg(n_leg);
    for(int i=0;i<n_leg;i++) q_deg[i] = q_rad[i] * R2D;
    q_joint_to_ch(q_deg.data(), out);
  }
  void dq_ctrl_to_ch(const double* dq_rad, float* out) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      const double d = (j.couple_src<0)? dq_rad[i] : dq_rad[i] + j.couple_coef*dq_rad[j.couple_src];
      out[j.channel] = (float)(j.sk() * d * R2D); }
  }
  void tau_ctrl_to_ch(const double* tau, float* out) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    // ★토크: k 로 나누고, 커플링은 **전치**로 — 소스축이 목적축 토크를 떠안는다.
    std::vector<double> raw(tau, tau+n_leg);
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      if(j.couple_src>=0) raw[j.couple_src] -= j.couple_coef*tau[i]; }
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[j.channel] = (float)(j.sign * raw[i] / j.gear_k); }
  }
  // 채널(deg) → 컨트롤러(rad)
  void ch_to_q_ctrl(const float* q_ch, double* out) const {
    ch_to_q_joint(q_ch, out);
    for(int i=0;i<n_leg;i++) out[i] *= D2R;
  }
  // 보고 토크(채널) → **실제 관절토크**[Nm]. 실제 = 보고·k, 커플링은 (I+C)ᵀ.
  // ★Python joint_map.ch_to_tau_joint 의 미러 (2026-08-13 추가 — C++ 에만 없었다).
  //   각도와 **반대 방향**이다: 각도는 목적축에서 빼고, 토크는 소스축에 더한다.
  void ch_to_tau_joint(const float* tau_ch, double* out) const {
    std::vector<double> raw(n_leg);
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      raw[i] = (double)tau_ch[j.channel] * j.gear_k / j.sign; }
    for(int i=0;i<n_leg;i++) out[i]=raw[i];
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      if(j.couple_src>=0) out[j.couple_src] = raw[j.couple_src] + j.couple_coef*raw[i]; }
  }
  void ch_to_dq_ctrl(const float* dq_ch, double* out) const {
    std::vector<double> raw(n_leg);
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      raw[i] = (double)dq_ch[j.channel] / j.sk(); }
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[i] = ((j.couple_src<0)? raw[i] : raw[i] - j.couple_coef*raw[j.couple_src]) * D2R; }
  }
  // 게인 벡터(채널). ★Nm/rad 그대로 — 환산하지 않는다.
  void kp_ch(float* out, double scale=1.0) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    for(int i=0;i<n_leg;i++) out[c->joints[i].channel] = (float)(c->joints[i].kp*scale);
    for(int ch : c->waist_ch) out[ch] = (float)c->waist_kp;
  }
  void kd_ch(float* out, double scale=1.0) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    for(int i=0;i<n_leg;i++) out[c->joints[i].channel] = (float)(c->joints[i].kd*scale);
    for(int ch : c->waist_ch) out[ch] = (float)c->waist_kd;
  }
  // ★clamp_ch 삭제(2026-08-10). **모델각 한계(min/max_deg)를 채널각에 거는 것은 틀렸다.**
  //   offset 이 0 이던 시절엔 두 값이 같아 무해했지만, 영점 캘리브레이션 후에는 완전히
  //   다른 수다. 기준자세에서 hold 를 걸면 HR_foot 채널 87.69 → 62 로 잘려
  //   **모델각 21.4° 점프**(채널 25.7° × kp30 = 13.4 Nm — 트립 15Nm 바로 아래라 E-stop 도
  //   못 잡는다). Python 은 오늘 clamp_joint 로 고쳤고 C++ 만 남아 있었다.
  //   ⇒ 대체: ch_to_q_joint → clamp_joint → q_joint_to_ch (모델각 공간에서 클램프)
  void clamp_ch_via_joint(float* q_ch) const {
    std::vector<double> qj(n_leg);
    ch_to_q_joint(q_ch, qj.data());
    clamp_joint(qj.data());
    q_joint_to_ch(qj.data(), q_ch);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// 4. 명령/상태 JSON (기존 GUI 채널과 호환 — 최소 파서)
// ─────────────────────────────────────────────────────────────────────────────
struct Cmd {
  std::string mode = "off";
  double v=0, vy=0, w=0, body_h=0.5;
  long long seq=-1;
  std::vector<double> jog_deg;           // ★축별 목표각(**모델각 deg**). 없으면 비어 있다
  // ★위치모드 강성 배율 (2026-08-21). GUI 의 [강성] 버튼이 낸다.
  //   ⚠**−1 = 키 없음**이다(0 이 아니라). 0 은 "게인을 0 으로" 라는 유효한 값이고,
  //     옛 GUI 가 이 키를 안 보내는 것과 구분돼야 한다. 구분을 안 하면 옛 GUI 를 붙이는
  //     순간 hold 게인이 0 이 되어 **다리가 떨어진다.**
  double pos_kp_scale = -1.0;
  // ★kd 배율 — **음수 = 미지정**(제어기가 √kp 로 자동). GUI 가 명시하면 그 값을 쓴다.
  //   왜 따로 두나: ζ 보존(√kp)이 원칙이지만 **속도잡음이 지배하면** kd 를 덜 올려야 한다
  //   (kd × dq_noise 가 그대로 토크 리플이다). 어느 쪽이 지배인지는 기체마다 다르다.
  double pos_kd_scale = -1.0;
  // ★중력보상 배율(float 모드). **음수 = 미지정**(제어기 기본 1.0).
  //   중립점을 브래킷해 찾는 것이 목적이라 GUI 가 0.05 단위로 올리고 내린다.
  double grav_scale = -1.0;
  // ★★축별 중력배율 — 명령파일로도 준다(env GRAV_SCALE_JOINT 는 기동 시 1회뿐이라
  //   스윕 중에 못 바꾼다). 8개(관절 순서). 비어 있으면 env/공통값을 쓴다.
  //   ⇒ 이게 있어야 **2단 측정**이 된다: 1차로 축별 대략값을 잡고 그걸 심어 둔 뒤,
  //     2차에서 공통 배수만 훑으면 **어느 축도 폭주하지 않아** 창이 안 무너진다.
  std::vector<double> grav_scale_joint;
  std::string raw;                       // 워치독: 내용 변화 판정용
};

inline bool read_cmd(const std::string& path, Cmd& out){
  std::ifstream f(path); if(!f) return false;
  std::stringstream ss; ss << f.rdbuf(); std::string s = ss.str();
  if(s.empty()) return false;
  out.raw = s;
  auto num=[&](const char* key, double dv)->double{
    std::string k = std::string("\"")+key+"\"";
    auto p = s.find(k); if(p==std::string::npos) return dv;
    p = s.find(':', p); if(p==std::string::npos) return dv;
    try { return std::stod(s.substr(p+1)); } catch(...) { return dv; } };
  auto str=[&](const char* key, const std::string& dv)->std::string{
    std::string k = std::string("\"")+key+"\"";
    auto p = s.find(k); if(p==std::string::npos) return dv;
    p = s.find(':', p); if(p==std::string::npos) return dv;
    auto q1 = s.find('"', p); if(q1==std::string::npos) return dv;
    auto q2 = s.find('"', q1+1); if(q2==std::string::npos) return dv;
    return s.substr(q1+1, q2-q1-1); };
  // ★jog_deg 배열 — GUI 가 축별 목표각(**모델각 deg**)을 낸다. 없으면 빈 벡터.
  //   ⚠`num` 은 `"key":` 뒤를 stod 로 읽어 `[` 에서 실패하므로 배열엔 못 쓴다.
  auto list=[&](const char* key)->std::vector<double>{
    std::vector<double> out2;
    std::string k = std::string("\"")+key+"\"";
    auto p = s.find(k);      if(p==std::string::npos) return out2;
    auto b = s.find('[', p); if(b==std::string::npos) return out2;
    auto e = s.find(']', b); if(e==std::string::npos) return out2;
    std::stringstream ls(s.substr(b+1, e-b-1)); std::string t;
    while(std::getline(ls,t,',')){ try{ out2.push_back(std::stod(t)); }catch(...){} }
    return out2; };
  out.mode = str("mode","off");
  out.v = num("v",0); out.vy = num("vy",0); out.w = num("w",0);
  out.body_h = num("body_h",0.5); out.seq = (long long)num("seq",-1);
  out.jog_deg = list("jog_deg");
  out.pos_kp_scale = num("pos_kp_scale", -1.0);
  out.pos_kd_scale = num("pos_kd_scale", -1.0);
  out.grav_scale   = num("grav_scale",   -1.0);   // float(무중력) 모드 중력보상 배율
  out.grav_scale_joint = list("grav_scale_joint");
  return true;
}

// 원자적 교체(tmp → rename). GUI 가 반쯤 쓰인 파일을 읽지 않게.
inline void write_state(const std::string& path, const std::string& json){
  std::string tmp = path + ".tmp";
  { std::ofstream f(tmp); if(!f) return; f << json; }
  std::rename(tmp.c_str(), path.c_str());
}

} // namespace bipedhw

// ── ShmHw dlopen 구현 (헤더 하단: dlfcn 의존을 여기로 격리) ─────────────────
#include <dlfcn.h>
namespace bipedhw {
inline ShmHw::~ShmHw(){ if(h) dlclose(h); }
inline bool ShmHw::init(int recv_wait_ms){
  h = dlopen(lib.c_str(), RTLD_NOW);
  if(!h){ err = std::string("libbipedshm 로드 실패: ") + (dlerror()?dlerror():"?")
              + "\n  → Pi 에서 emb/hal/build_bridge.sh 를 먼저 실행할 것."; return false; }
  auto sym=[&](const char* s)->void*{ void* p=dlsym(h,s);
    if(!p) err = std::string("심볼 없음: ")+s; return p; };
  p_init=(int(*)(int))sym("bridge_init");
  p_read=(int(*)(float*,float*,float*,float*,float*,float*,float*,int*,int*))sym("bridge_read");
  p_wpos=(int(*)(const float*,const float*,const float*,int))sym("bridge_write_pos");
  p_wmit=(int(*)(const float*,const float*,const float*,const float*,const float*,int))sym("bridge_write_mit");
  p_en  =(int(*)(int))sym("bridge_enable");
  if(!p_init||!p_read||!p_wpos||!p_wmit||!p_en) return false;
  if(p_init(recv_wait_ms)!=0){
    err = "bridge_init 실패 — Emb(RobotEmbedded) 미기동이거나 halGait 초기화 미완료.\n"
          "  Emb 기동 후 5초(=100+4500 tick @1kHz 게이트) 기다린 뒤 재시도할 것.";
    return false;
  }
  return true;
}
} // namespace bipedhw
