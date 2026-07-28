// test_track.cpp — TAMOLS 플랜 → WBIC 참조 변환기 검증
//   Drake 해 로드 → 시간축 샘플: base 궤적 연속·접촉 스케줄·스윙 목표(발판 도달) 확인.
#include "tamols_track.hpp"
#include "constraints.hpp"   // load_solution
#include <cstdio>

using namespace tamols;

int main() {
  TamolsState st;
  if (!load_solution("/tmp/tamols_sol.txt", st)) { std::printf("★ sol 로드 실패\n"); return 1; }
  double T = total_duration(st);
  std::printf("[TAMOLS 플랜]  위상 %d개 · 총 %.3f s\n", st.num_phases(), T);

  // liftoff = 초기 측정 발위치(p_meas) 가정(1차 검증)
  Eigen::Matrix<double, 4, 3> liftoff = st.p_meas;

  // ① base 궤적 연속성: 촘촘히 샘플해 com_ref 점프(위상경계) 확인
  double max_jump = 0, prevx = 0, prevz = 0; bool first = true;
  int N = 400;
  for (int i = 0; i <= N; ++i) {
    double t = T * i / N;
    TrackRef r = eval(st, t, liftoff);
    if (!first) { max_jump = std::max(max_jump, std::fabs(r.com_ref(0) - prevx) + std::fabs(r.com_ref(2) - prevz)); }
    prevx = r.com_ref(0); prevz = r.com_ref(2); first = false;
  }
  double dt_step = T / N, vmax = 2.0;                 // 위상경계서도 연속이면 스텝당 이동 < v·dt
  bool cont_ok = max_jump < vmax * dt_step * 5;       // 여유 5배
  std::printf("  base 궤적 최대 스텝이동 = %.4f m (연속기준 %.4f) %s\n", max_jump, vmax * dt_step * 5, cont_ok ? "OK" : "★불연속");

  // ② 시작/끝 base 위치 (전진 확인)
  TrackRef r0 = eval(st, 0.0, liftoff), rT = eval(st, T - 1e-4, liftoff);
  std::printf("  base x: 시작 %.3f → 끝 %.3f (전진 %.3f m)\n", r0.com_ref(0), rT.com_ref(0), rT.com_ref(0) - r0.com_ref(0));

  // ③ 접촉 스케줄 프린트(위상별 stance/swing)
  std::printf("  위상별 접촉: ");
  double acc = 0;
  for (int k = 0; k < st.num_phases(); ++k) {
    TrackRef r = eval(st, acc + st.gait[k].duration * 0.5, liftoff);
    std::printf("[P%d T%.2f st=", k, st.gait[k].duration);
    for (int c : r.contacts) std::printf("%d", c);
    std::printf(" sw=");
    for (auto& kv : r.swing) std::printf("%d", kv.first);
    std::printf("] "); acc += st.gait[k].duration;
  }
  std::printf("\n");

  // ④ 스윙 목표가 발판 p로 수렴하는지 (swing 끝 tau→Tk서 목표≈p)
  double sw_err = 0; int sw_n = 0;
  acc = 0;
  for (int k = 0; k < st.num_phases(); ++k) {
    double Tk = st.gait[k].duration;
    for (int i = 0; i < 4; ++i) if (!st.gait[k].contact[i]) {
      TrackRef r = eval(st, acc + Tk * 0.999, liftoff);   // swing 거의 끝
      if (r.swing.count(i)) {
        Vector3d tgt = r.swing[i].first, pf = st.p.row(i).transpose();
        sw_err = std::max(sw_err, (tgt.head<2>() - pf.head<2>()).norm());   // xy 수렴(z는 arc)
        sw_n++;
      }
    }
    acc += Tk;
  }
  std::printf("  스윙끝 xy↔발판 최대오차 = %.4f m (%d개) %s\n", sw_err, sw_n, sw_err < 0.02 ? "OK" : "★");

  bool pass = cont_ok && (rT.com_ref(0) - r0.com_ref(0) > 0) && sw_err < 0.02;
  std::printf("  → 참조 변환기 %s\n", pass ? "동작 ✓" : "확인필요");
  return pass ? 0 : 1;
}
