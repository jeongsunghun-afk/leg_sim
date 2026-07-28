// tamols_online.hpp — receding-horizon 온라인 TAMOLS 플래너
//   매 replan: 현재 base 상태(x,y,z,yaw + 속도)와 발위치를 받아 짧은 horizon 트롯 계획을
//   solve_fast(warm-start)로 재풀이. 결과를 샘플링해 RSL-WBC가 추종할 (pose·vel·contact·발판) 버퍼 생성.
//   ★로컬 프레임: base_pose 시작=원점(컨트롤러가 현재 x,y로 앵커). 연속 전진 = 매 replan 앞으로 램프.
#pragma once
#include "tamols_jac.hpp"
#include <array>
#include <vector>

namespace tamols {

// 평지 cost map(지형 없음): 전부 0. gap은 별도 map 주입.
inline void flat_costmap(Grid& h, double& cell, int& map_size, int N = 41, double c = 0.05) {
  cell = c; map_size = N; h = Grid::Zero(N, N);
}

// 트롯 게이트 5-phase(전진 walk). 대각쌍 swing: P1{FL,RR}·P3{FR,RL}, 사이 all-contact.
//   ★off = horizon-shift 위상 오프셋: 매 replan 회전 → 실행 위상이 swing까지 순환(발 내딛기, stall 해소).
inline void set_trot_gait(TamolsState& st, double phase_dur = 0.2, int off = 0) {
  int P = 5; st.gait.resize(P);
  int cs[5][4] = {{1,1,1,1},{0,1,1,0},{1,1,1,1},{1,0,0,1},{1,1,1,1}};
  int ad[5][4] = {{0,0,0,0},{0,0,0,0},{1,0,0,1},{1,0,0,1},{1,1,1,1}};
  for (int k = 0; k < P; ++k) { int s = ((k + off) % P + P) % P; st.gait[k].duration = phase_dur;
    for (int i = 0; i < 4; ++i) { st.gait[k].contact[i] = cs[s][i]; st.gait[k].at_des[i] = ad[s][i]; } }
}

// ★walk(crawl) 게이트 4-phase: 한 발씩 swing → 3발 지지(정적 안정, 횡균형 강함).
//   LS crawl 순서 RR→FR→RL→FL(idx 3,1,2,0). off=horizon-shift 위상.
inline void set_walk_gait(TamolsState& st, double phase_dur = 0.2, int off = 0) {
  int P = 4; st.gait.resize(P);
  int cs[4][4] = {{1,1,1,0},{1,0,1,1},{1,1,0,1},{0,1,1,1}};   // swing: RR,FR,RL,FL
  for (int k = 0; k < P; ++k) { int s = ((k + off) % P + P) % P; st.gait[k].duration = phase_dur;
    for (int i = 0; i < 4; ++i) { st.gait[k].contact[i] = cs[s][i]; st.gait[k].at_des[i] = (cs[s][i]==0)?1:0; } }
}

// ── 온라인 replan: 현재 상태 → TamolsState(로컬, 앞으로 vadv 전진) → solve_fast(warm-start) ──
//   base0 = [z,yaw] 현재값(x,y=로컬 원점). v0 = 현재 base 속도(local x,y). foot_meas = 현재 발위치(로컬, base 기준 상대 xy + world z).
//   vadv = 명령 전진속도. 반환 = QpResult(수렴 여부).
struct OnlineCfg { double vadv = 0.4, phase_dur = 0.2; int rti_iter = 5; bool warm = false;
  double gap_x0 = -1, gap_x1 = -1;   // ★로컬 프레임 gap [x0,x1](base 기준). <0 = gap 없음(평지 walk)
  int phase_off = 0;                 // ★horizon-shift 위상 오프셋(매 replan 증가 → swing 실행)
  bool walk = false; };              // ★walk(crawl 4-phase 한발씩=정적안정) vs trot(5-phase 대각쌍)

inline QpResult online_replan(TamolsState& st, const Grid& h, double cell, int map_size,
                              double z0, double yaw0, double vx0, double vy0,
                              const Eigen::Matrix<double,4,3>& foot_meas, const OnlineCfg& cfg) {
  if (cfg.walk) set_walk_gait(st, cfg.phase_dur, cfg.phase_off);   // ★walk=한발씩 정적안정
  else          set_trot_gait(st, cfg.phase_dur, cfg.phase_off);   // ★horizon-shift 위상 회전(swing 실행)
  int P = st.num_phases(); double T = P * cfg.phase_dur, xf = cfg.vadv * T;
  st.base_pose << 0, 0, z0, 0, 0, yaw0;             // 로컬 원점서 시작(컨트롤러가 현재 x,y 앵커)
  st.base_vel  << vx0, vy0, 0, 0, 0, 0;
  st.p_meas = foot_meas;
  st.ref_vel = Vector3d(cfg.vadv, 0, 0);
  if (!cfg.warm || (int)st.a.size() != P) {         // cold: Hermite 전진 램프 init(정지출발 smootherstep)
    st.a.assign(P, MatrixXd::Zero(6, 4));
    // ★속도연속 cubic: x(0)=0·v(0)=vx0(초기조건 정합→elastic 방지) → x(T)=xf·v(T)=vadv(cruise 가속)
    double c1 = vx0, c3 = (cfg.vadv - vx0 - 2*(xf - vx0*T)/T)/(T*T), c2 = (xf - vx0*T - c3*T*T*T)/(T*T);
    auto xg = [&](double t){ return c1*t + c2*t*t + c3*t*t*t; };
    auto vg = [&](double t){ return c1 + 2*c2*t + 3*c3*t*t; };
    for (int k = 0; k < P; ++k) {
      st.a[k].col(0) = st.base_pose; st.a[k](2, 0) = z0;
      double t0 = k*cfg.phase_dur, x0 = xg(t0), x1 = xg(t0+cfg.phase_dur), v0 = vg(t0), v1 = vg(t0+cfg.phase_dur);
      st.a[k](0,0) = x0; st.a[k](0,1) = v0;
      st.a[k](0,2) = (3*(x1-x0)/cfg.phase_dur - 2*v0 - v1)/cfg.phase_dur;
      st.a[k](0,3) = (2*(x0-x1)/cfg.phase_dur + v0 + v1)/(cfg.phase_dur*cfg.phase_dur);
    }
    // 발판 = 명목 stance + Raibert(실제속도 기반 전진, 과신장 방지). ★gap in reach면 straddle
    double fwd0 = vx0 * (cfg.walk ? 0.4 : 0.3);   // 실제 vx0 기반 전진(계획속도 아님)=발이 base 안 앞섬
    for (int i = 0; i < 4; ++i) { st.p(i,0) = st.prm.hip_offsets(i,0) + fwd0; st.p(i,1) = st.prm.hip_offsets(i,1); st.p(i,2) = 0; }
    st.epsilon = VectorXd::Zero(P);
  }
  bool gap_near = cfg.gap_x0 >= 0 && cfg.gap_x0 < xf + 0.45 && cfg.gap_x1 > -0.15;   // gap이 horizon 전방 도달권
  if (gap_near) { double m = 0.04;
    st.p(0,0) = st.p(1,0) = cfg.gap_x1 + m;   // 앞발(FL,FR) = 갭 너머
    st.p(2,0) = st.p(3,0) = cfg.gap_x0 - m;   // 뒷발(RL,RR) = 갭 앞
  }
  QpOptions o; o.max_iter = cfg.rti_iter;
  o.rp_max = 0.06; o.zlo = 0.48; o.zhi = 0.55; o.yaw_max = 0.10; o.x_target = xf * 0.9;
  o.gap = gap_near; o.gap_lo = cfg.gap_x0; o.gap_hi = cfg.gap_x1;   // ★gap 회피 발판 제약(로컬)
  return solve_fast(st, h, cell, map_size, o);
}

// ── 계획 샘플링: TamolsState → (t, pose6, vel6, contact[4]) 시퀀스 + 발판[4] ──
//   dt 간격으로 스플라인 평가. RSL-WBC 추종 버퍼(tam_s/tam_c/tam_fh 등가) 채움.
inline void sample_plan(const TamolsState& st, double dt,
                        std::vector<std::array<double,12>>& samples,
                        std::vector<std::array<int,4>>& contacts,
                        Eigen::Matrix<double,4,3>& footholds) {
  samples.clear(); contacts.clear();
  double Ttot = 0; for (auto& g : st.gait) Ttot += g.duration;
  int N = std::max(1, (int)std::round(Ttot / dt));
  for (int n = 0; n <= N; ++n) {
    double t = std::min((double)n * dt, Ttot);
    // phase 찾기
    double acc = 0; int k = 0; double tau = t;
    for (int i = 0; i < st.num_phases(); ++i) { if (t <= acc + st.gait[i].duration || i == st.num_phases()-1) { k = i; tau = t - acc; break; } acc += st.gait[i].duration; }
    if (tau > st.gait[k].duration) tau = st.gait[k].duration;
    Vector6d pose = st.pos_at(k, tau), vel = st.vel_at(k, tau);
    std::array<double,12> s; for (int d = 0; d < 6; ++d) { s[d] = pose[d]; s[6+d] = vel[d]; }
    std::array<int,4> c; for (int i = 0; i < 4; ++i) c[i] = st.gait[k].contact[i];
    samples.push_back(s); contacts.push_back(c);
  }
  footholds = st.p;
}

} // namespace tamols
