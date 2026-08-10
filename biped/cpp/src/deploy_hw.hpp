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
  double tau_trip_nm = 8.0, tau_trip_ms = 50, vel_trip_dps = 200.0;

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
    }
  }
  if(c.joints.empty()){ err = "joints 를 하나도 못 읽었다 — 설정 형식 확인: " + path; return false; }
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
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[i] = ((double)q_ch[j.channel] - j.offset_deg)/j.sk(); }
  }
  void q_joint_to_ch(const double* q_j, float* out) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[j.channel] = (float)(q_j[i]*j.sk() + j.offset_deg); }
    for(size_t k=0;k<c->waist_ch.size();k++)
      out[c->waist_ch[k]] = (float)(k<c->waist_hold.size()? c->waist_hold[k] : 0.0);
  }
  void clamp_joint(double* q_j) const {
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      q_j[i] = std::max(j.min_deg, std::min(j.max_deg, q_j[i])); }
  }

  // 컨트롤러(rad) → 채널(deg). 허리는 hold 각으로 고정.
  void q_ctrl_to_ch(const double* q_rad, float* out) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[j.channel] = (float)(j.sk() * q_rad[i] * R2D + j.offset_deg); }
    for(size_t k=0;k<c->waist_ch.size();k++)
      out[c->waist_ch[k]] = (float)(k<c->waist_hold.size()? c->waist_hold[k] : 0.0);
  }
  void dq_ctrl_to_ch(const double* dq_rad, float* out) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[j.channel] = (float)(j.sk() * dq_rad[i] * R2D); }
  }
  void tau_ctrl_to_ch(const double* tau, float* out) const {
    for(int i=0;i<n_channel;i++) out[i]=0.f;
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[j.channel] = (float)(j.sign * tau[i] / j.gear_k); }   // ★토크는 k 로 나눈다
  }
  // 채널(deg) → 컨트롤러(rad)
  void ch_to_q_ctrl(const float* q_ch, double* out) const {
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[i] = ((double)q_ch[j.channel] - j.offset_deg) / j.sk() * D2R; }
  }
  void ch_to_dq_ctrl(const float* dq_ch, double* out) const {
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      out[i] = ((double)dq_ch[j.channel] / j.sk()) * D2R; }
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
  void clamp_ch(float* q_ch) const {
    for(int i=0;i<n_leg;i++){ const auto& j=c->joints[i];
      q_ch[j.channel] = (float)std::max(j.min_deg, std::min(j.max_deg, (double)q_ch[j.channel])); }
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// 4. 명령/상태 JSON (기존 GUI 채널과 호환 — 최소 파서)
// ─────────────────────────────────────────────────────────────────────────────
struct Cmd {
  std::string mode = "off";
  double v=0, vy=0, w=0, body_h=0.5;
  long long seq=-1;
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
  out.mode = str("mode","off");
  out.v = num("v",0); out.vy = num("vy",0); out.w = num("w",0);
  out.body_h = num("body_h",0.5); out.seq = (long long)num("seq",-1);
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
