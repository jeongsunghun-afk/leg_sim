// tamols_track.hpp — TAMOLS 플랜 → WBIC 추종 참조 변환
//   TAMOLS 해(base 스플라인·발판 p·게이트 스케줄)를 매 제어스텝의 WBIC 참조로 변환:
//   com_ref(base pos)·yaw_des·base_vel · contacts(stance) · swing(leg→목표 pos/vel).
//   WBIC: ctrl.com_ref/yaw_des 세팅 + ctrl.mpc_grf(x_ref,cs) → ctrl.wbic_track(contacts,swing,lam).
#pragma once
#include "tamols.hpp"
#include <map>
#include <vector>
#include <utility>

namespace tamols {

struct TrackRef {
  Vector3d com_ref;                                        // base 목표 위치
  double   yaw_des;                                        // base 목표 yaw
  Vector3d base_vel;                                       // base 목표 선속도
  std::vector<int> contacts;                               // stance leg 인덱스
  std::map<int, std::pair<Vector3d, Vector3d>> swing;      // swing leg → (목표pos, 목표vel)
  int      phase;                                          // 현재 위상
  double   tau;                                            // 위상 내 시간
  bool     done;                                           // horizon 종료
};

// t(누적시간)에서의 위상 k·tau 찾기
inline bool locate(const TamolsState& st, double t, int& phase, double& tau) {
  double acc = 0;
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    if (t < acc + Tk) { phase = k; tau = t - acc; return true; }
    acc += Tk;
  }
  phase = st.num_phases() - 1; tau = st.gait[phase].duration; return false;   // horizon 끝(마지막 홀드)
}

inline double total_duration(const TamolsState& st) {
  double s = 0; for (int k = 0; k < st.num_phases(); ++k) s += st.gait[k].duration; return s;
}

// swing 궤적: liftoff p0 → target pf, z=포물선 arc(step_h). s∈[0,1] 진행
inline Vector3d swing_pos(const Vector3d& p0, const Vector3d& pf, double s, double step_h) {
  Vector3d p = (1 - s) * p0 + s * pf;
  p(2) += step_h * 4.0 * s * (1.0 - s);                    // 최고점 s=0.5
  return p;
}
inline Vector3d swing_vel(const Vector3d& p0, const Vector3d& pf, double s, double sdot, double step_h) {
  Vector3d v = sdot * (pf - p0);
  v(2) += step_h * 4.0 * (1.0 - 2.0 * s) * sdot;
  return v;
}

// t에서의 WBIC 참조. liftoff = 각 다리의 swing 시작(이지) 위치(sim이 swing 진입 시 기록).
inline TrackRef eval(const TamolsState& st, double t,
                     const Eigen::Matrix<double, 4, 3>& liftoff, double step_h = 0.08) {
  TrackRef r; int k; double tau;
  r.done = !locate(st, t, k, tau); r.phase = k; r.tau = tau;
  Vector6d pose = st.pos_at(k, tau), vel = st.vel_at(k, tau);
  r.com_ref = pose.head<3>();
  r.yaw_des = pose(5);
  r.base_vel = vel.head<3>();
  double Tk = st.gait[k].duration, s = (Tk > 1e-9) ? (tau / Tk) : 0.0, sdot = (Tk > 1e-9) ? (1.0 / Tk) : 0.0;
  for (int i = 0; i < 4; ++i) {
    if (st.gait[k].contact[i]) {
      r.contacts.push_back(i);
    } else {
      Vector3d pf = st.p.row(i).transpose();               // 목표 foothold
      Vector3d p0 = liftoff.row(i).transpose();            // liftoff(이지) 위치
      r.swing[i] = {swing_pos(p0, pf, s, step_h), swing_vel(p0, pf, s, sdot, step_h)};
    }
  }
  return r;
}

} // namespace tamols
