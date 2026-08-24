// freeze_forensics.hpp — **통신 동결의 원인을 찾기 위한** 증거 수집기.
//
// ═══ 왜 필요한가 ═══
// 동결이 2026-08-20 부터 여덟 번 넘게 났는데 아직 원인을 모른다. 이유는 하나다:
// **증거가 남지 않는다.** 배너는 터미널 스크롤백에만 찍히고, 다음 재기동에 사라진다.
// 그래서 "hip 시험 중이었나 / 하중이 걸렸나 / 몇 초째였나" 를 매번 기억으로 맞춰 보다가
// 놓친다. 여덟 번의 데이터가 있는데도 상관관계를 못 본다.
//
// ★그리고 **Emb 쪽은 증거를 계산해서 버린다**(2026-08-24 확인).
//   ~/ZSource/RobotEmbedded/communications/commEtherCATm.cpp:
//       wkc = commEtherCATm_RoundTrip();
//       if (wkc < expectedWkc){ wkcErrorCount++; consecutiveWkcError++; ... }
//   즉 WKC(working counter) 오류를 **세고 있다.** 게터도 있다
//   (GetWkcErrorCount / GetCycleCount / GetRoundTripUsec / GetExpectedWKC).
//   그런데 commMCU.cpp:163 은
//       commEtherCATm_Proc();          // ← **반환값을 버린다**
//   이고, WKC 오류 경로에 printf 가 **하나도 없다.** 게터를 부르는 곳도 없다.
//   ⇒ EtherCAT 이 죽는 순간 Emb 는 **아무 로그도 남기지 않고** 마지막 버퍼를 계속
//     SHM 에 올린다(갱신 플래그까지 1로). 우리가 장님인 건 우연이 아니라 구조다.
//
// ═══ 여기서 모으는 것 ═══
// 이 파일은 **Emb 를 안 고치고** 밖에서 볼 수 있는 것을 전부 긁는다:
//
//   ① 직전 3초 링버퍼 → CSV.  동결 **전에** 무슨 일이 있었는지가 핵심이다.
//      토크가 튀었나 · 속도가 튀었나 · 명령이 컸나. 배너가 뜬 뒤엔 이미 늦다.
//   ② Emb 프로세스가 아직 CPU 를 쓰나 (/proc/<pid>/stat utime+stime).
//      **돌고 있다 = EtherCAT 이 죽었다** · **멈췄다 = Emb 가 스톨했다.** 결정적인 갈림길이고,
//      지금까지 이 둘을 구분한 적이 한 번도 없다.
//   ③ eth0 물리링크: carrier · rx_errors · rx_crc_errors · rx_over_errors.
//      **carrier=0 → 케이블/커넥터**다. carrier=1 인데 얼었다면 슬레이브/MCU 쪽이다.
//      CRC 오류가 늘고 있으면 전기적 노이즈(모터 전류)다 — 이게 지금 1순위 가설이다.
//   ④ 사건 로그 한 줄 → /tmp/biped_freeze_log.tsv (append). 재기동해도 남는다.
//
// ⚠읽기 전용이다. /proc·/sys 를 읽고 /tmp 에 쓴다. 모터엔 아무것도 안 쓴다.
// ⚠이걸로 원인이 **바로** 나오진 않는다. 나오는 건 **상관관계를 볼 수 있는 표**다.
//   여덟 번을 한 파일에서 나란히 보는 것이 지금 없는 것이고, 그게 이 코드의 목적이다.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <dirent.h>

namespace bipedhw {

// ── /sys, /proc 에서 한 줄 읽기 ──────────────────────────────────────────────
inline std::string read_line_file(const std::string& p){
  FILE* f = fopen(p.c_str(), "r"); if(!f) return "";
  char b[512]={0}; if(!fgets(b,sizeof b,f)){ fclose(f); return ""; }
  fclose(f);
  std::string s(b); while(!s.empty() && (s.back()=='\n'||s.back()=='\r')) s.pop_back();
  return s;
}
inline long long read_ll_file(const std::string& p, long long dv=-1){
  std::string s = read_line_file(p); if(s.empty()) return dv;
  try { return std::stoll(s); } catch(...) { return dv; }
}

// ── EtherCAT NIC 상태 ────────────────────────────────────────────────────────
//   ★인터페이스는 commEtherCATm.cpp:173 의 하드코딩 `"eth0"` 이다. 바뀌면 env 로 준다.
struct NicSnap {
  std::string ifname;
  long long carrier=-1, rx_err=-1, rx_crc=-1, rx_over=-1, rx_pkts=-1, tx_pkts=-1;
  static NicSnap take(){
    NicSnap n;
    n.ifname = getenv("ECAT_IFACE") ? getenv("ECAT_IFACE") : "eth0";
    const std::string b = "/sys/class/net/" + n.ifname;
    n.carrier = read_ll_file(b + "/carrier");
    n.rx_err  = read_ll_file(b + "/statistics/rx_errors");
    n.rx_crc  = read_ll_file(b + "/statistics/rx_crc_errors");
    n.rx_over = read_ll_file(b + "/statistics/rx_over_errors");
    n.rx_pkts = read_ll_file(b + "/statistics/rx_packets");
    n.tx_pkts = read_ll_file(b + "/statistics/tx_packets");
    return n;
  }
};

// ── Emb(RobotEmbedded) 프로세스 상태 ────────────────────────────────────────
//   ★핵심 판별기다. 동결 중에 Emb 의 CPU 시간이 **계속 늘면** Emb 는 돌고 있고
//     EtherCAT 이 죽은 것이다. **안 늘면** Emb 자체가 스톨/데드락이다.
//     지금까지 이 둘을 구분한 적이 없다 — 그래서 "전원 OFF/ON" 이라는 같은 처방만 반복했다.
struct EmbSnap {
  int pid=-1; long long jiffies=-1; std::string state="?"; int threads=-1;
  static int find_pid(const char* name="RobotEmbedded"){
    DIR* d = opendir("/proc"); if(!d) return -1;
    int found=-1;
    while(dirent* e = readdir(d)){
      if(e->d_name[0]<'0'||e->d_name[0]>'9') continue;
      std::string c = read_line_file(std::string("/proc/")+e->d_name+"/comm");
      if(c==name){ found = atoi(e->d_name); break; }
    }
    closedir(d); return found;
  }
  static EmbSnap take(int pid){
    EmbSnap s; s.pid = pid; if(pid<=0) return s;
    // /proc/<pid>/stat: comm 이 괄호 안에 있고 공백을 포함할 수 있으므로 ')' 뒤부터 센다.
    std::string st = read_line_file("/proc/"+std::to_string(pid)+"/stat");
    auto rp = st.rfind(')'); if(rp==std::string::npos || rp+2>=st.size()) return s;
    std::vector<std::string> f; { const char* p = st.c_str()+rp+2; std::string cur;
      for(; *p; ++p){ if(*p==' '){ f.push_back(cur); cur.clear(); } else cur+=*p; }
      if(!cur.empty()) f.push_back(cur); }
    // ')' 뒤 첫 필드가 state(3번), 그 뒤 utime=14 · stime=15 → 인덱스 11·12, threads=20 → 17
    if(f.size()>0)  s.state   = f[0];
    if(f.size()>12) s.jiffies = atoll(f[11].c_str()) + atoll(f[12].c_str());
    if(f.size()>17) s.threads = atoi(f[17].c_str());
    return s;
  }
};

// ── 직전 N 초 링버퍼 ────────────────────────────────────────────────────────
//   ★동결 **전**이 알고 싶은 것이다. 배너가 뜬 시점은 이미 0.5초 늦다.
//     500Hz × 3초 × 10채널 = 15000 표본. 메모리 ~2.4MB — 상관없다.
struct RingRec { double t; float q[16], dq[16], tau[16], cur[16], cmd[16]; };
struct Ring {
  std::vector<RingRec> buf; size_t n=0, head=0; int nch=10;
  void init(double secs, double dt, int nch_){
    nch = nch_;
    size_t cap = (size_t)std::max(2.0, secs/std::max(1e-6,dt));
    buf.assign(cap, RingRec{}); n=0; head=0;
  }
  void push(double t, const float* q, const float* dq, const float* tau,
            const float* cur, const float* cmd){
    if(buf.empty()) return;
    RingRec& r = buf[head];
    r.t = t;
    for(int i=0;i<nch && i<16;i++){
      r.q[i]=q?q[i]:0; r.dq[i]=dq?dq[i]:0; r.tau[i]=tau?tau[i]:0;
      r.cur[i]=cur?cur[i]:0; r.cmd[i]=cmd?cmd[i]:0; }
    head = (head+1)%buf.size();
    if(n<buf.size()) n++;
  }
  // 오래된 것부터 CSV 로. 반환=쓴 줄 수.
  size_t dump(const std::string& path) const {
    if(buf.empty()||n==0) return 0;
    FILE* f = fopen(path.c_str(),"w"); if(!f) return 0;
    fprintf(f,"t");
    for(int i=0;i<nch;i++) fprintf(f,",q%d,dq%d,tau%d,cur%d,cmd%d",i,i,i,i,i);
    fprintf(f,"\n");
    size_t start = (n<buf.size()) ? 0 : head;
    for(size_t k=0;k<n;k++){
      const RingRec& r = buf[(start+k)%buf.size()];
      fprintf(f,"%.4f", r.t);
      for(int i=0;i<nch;i++)
        fprintf(f,",%.3f,%.1f,%.3f,%.3f,%.3f", r.q[i],r.dq[i],r.tau[i],r.cur[i],r.cmd[i]);
      fprintf(f,"\n");
    }
    fclose(f); return n;
  }
  // 동결 직전 창(마지막 sec 초, 얼기 전 구간)의 채널별 |값| 최대. 원인 상관용.
  void peaks(double now, double win, double* tau_pk, double* dq_pk, double* cur_pk) const {
    for(int i=0;i<nch;i++){ tau_pk[i]=0; dq_pk[i]=0; cur_pk[i]=0; }
    if(buf.empty()||n==0) return;
    size_t start = (n<buf.size()) ? 0 : head;
    for(size_t k=0;k<n;k++){
      const RingRec& r = buf[(start+k)%buf.size()];
      if(now - r.t > win) continue;
      for(int i=0;i<nch;i++){
        double a=r.tau[i]<0?-r.tau[i]:r.tau[i];   if(a>tau_pk[i]) tau_pk[i]=a;
        double b=r.dq[i] <0?-r.dq[i] :r.dq[i];    if(b>dq_pk[i])  dq_pk[i]=b;
        double c=r.cur[i]<0?-r.cur[i]:r.cur[i];   if(c>cur_pk[i]) cur_pk[i]=c;
      }
    }
  }
};

// ── 사건 로그 (append) ──────────────────────────────────────────────────────
//   ⚠**append 다.** 재기동해도 남아야 상관관계를 볼 수 있다 — 그게 이 파일의 존재 이유다.
//   ⚠헤더는 파일이 없을 때만 쓴다.
inline void freeze_log_append(const std::string& path, const std::string& header,
                              const std::string& row){
  bool exists = false;
  { FILE* t = fopen(path.c_str(),"r"); if(t){ exists=true; fclose(t); } }
  FILE* f = fopen(path.c_str(),"a"); if(!f) return;
  if(!exists) fprintf(f,"%s\n", header.c_str());
  fprintf(f,"%s\n", row.c_str());
  fclose(f);
}

inline std::string wall_stamp(){
  time_t tt = time(nullptr); struct tm tmv; localtime_r(&tt,&tmv);
  char b[32]; strftime(b,sizeof b,"%Y-%m-%d %H:%M:%S",&tmv); return b;
}

} // namespace bipedhw
