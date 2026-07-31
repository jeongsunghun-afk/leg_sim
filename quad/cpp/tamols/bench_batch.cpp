// bench_batch.cpp — TAMOLS 배치 solve throughput 벤치 (in-loop RL 가능성 판정)
//   env별 TamolsState(warm-start 지속) + OnlineCfg를 OpenMP 병렬로 online_replan.
//   측정: 단일 warm solve, N-배치 병렬 throughput(solves/s) → 4096-env 터치다운율서 feasible env 수.
//   빌드: g++ -O3 -fopenmp -std=c++17 bench_batch.cpp -I/usr/include/eigen3 -I<eiq_inc> -L<eiq_lib> -leiquadprog -o bench_batch
#include "tamols_online.hpp"
#include <cstdio>
#include <chrono>
#include <vector>
#include <omp.h>
using namespace tamols;
using Clock = std::chrono::high_resolution_clock;
static double ms_since(Clock::time_point t){ return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }

int main(int argc, char** argv) {
  int N = argc > 1 ? std::atoi(argv[1]) : 512;   // env 수
  int M = argc > 2 ? std::atoi(argv[2]) : 20;    // warm replan 라운드
  int gap_frac_pct = argc > 3 ? std::atoi(argv[3]) : 50; // gap 지형 env %

  // 공유 heightmap (flat) — 각 env가 gap을 cfg로 주입(로컬 프레임이라 map은 공유 가능)
  Grid h; double cell; int ms; flat_costmap(h, cell, ms);
  Params prm;

  std::vector<TamolsState> st(N);
  std::vector<OnlineCfg> cfg(N);
  std::vector<double> vx0(N), yaw0(N);
  Eigen::Matrix<double,4,3> foot0;
  for (int i = 0; i < 4; ++i) { foot0(i,0)=prm.hip_offsets(i,0); foot0(i,1)=prm.hip_offsets(i,1); foot0(i,2)=0; }

  // per-env 변주: vx0∈[0,0.4], yaw0 소량, gap 위치 다양 (동일 solve 캐시 방지)
  for (int e = 0; e < N; ++e) {
    cfg[e].vadv = 0.4; cfg[e].phase_dur = 0.2; cfg[e].rti_iter = 5;
    vx0[e] = 0.4 * (double)(e % 5) / 4.0;
    yaw0[e] = 0.02 * ((e % 7) - 3);
    if ((e * 100 / N) % 100 < gap_frac_pct) {            // gap 지형 env
      double g0 = 0.30 + 0.02 * (e % 6);
      cfg[e].gap_x0 = g0; cfg[e].gap_x1 = g0 + 0.18;
    }
  }

  int nthreads = 0;
  #pragma omp parallel
  { if (omp_get_thread_num()==0) nthreads = omp_get_num_threads(); }

  // ── cold init (warm state 채우기), 병렬 ──
  auto tc = Clock::now();
  #pragma omp parallel for schedule(dynamic)
  for (int e = 0; e < N; ++e) {
    cfg[e].warm = false;
    online_replan(st[e], h, cell, ms, 0.52, yaw0[e], vx0[e], 0.0, foot0, cfg[e]);
    cfg[e].warm = true;
  }
  double cold_ms = ms_since(tc);

  // ── 단일 warm solve 타이밍 (1 스레드, env 0) ──
  {
    double best = 1e9;
    for (int r = 0; r < 10; ++r) { auto t=Clock::now(); online_replan(st[0], h, cell, ms, 0.52, yaw0[0], vx0[0], 0.0, foot0, cfg[0]); best=std::min(best, ms_since(t)); }
    std::printf("단일 warm solve (1thread): %.2f ms  → %.0f solves/s/core\n", best, 1000.0/best);
  }

  // ── 배치 warm solve throughput (OpenMP, M 라운드) ──
  int ok = 0, feas = 0;
  auto tb = Clock::now();
  for (int r = 0; r < M; ++r) {
    #pragma omp parallel for schedule(dynamic) reduction(+:ok,feas)
    for (int e = 0; e < N; ++e) {
      QpResult res = online_replan(st[e], h, cell, ms, 0.52, yaw0[e], vx0[e], 0.0, foot0, cfg[e]);
      if (r == M-1) { ok += res.ok ? 1 : 0; feas += (res.eq_viol<1e-2 && res.ineq_viol<1e-2) ? 1 : 0; }
    }
  }
  double batch_ms = ms_since(tb);
  double per_round = batch_ms / M;
  double throughput = 1000.0 * N / per_round;      // solves/s (배치 병렬)

  std::printf("\n=== TAMOLS 배치 벤치 (N=%d env, M=%d 라운드, gap %d%%, threads=%d) ===\n", N, M, gap_frac_pct, nthreads);
  std::printf("cold init(병렬)   : %.0f ms (%d env)\n", cold_ms, N);
  std::printf("배치 warm/라운드  : %.1f ms  (%d env 동시)\n", per_round, N);
  std::printf("throughput        : %.0f solves/s (병렬 %d코어)\n", throughput, nthreads);
  std::printf("feasible          : %d/%d (eq&ineq<1e-2), ok=%d/%d\n", feas, N, ok, N);

  // ── RL in-loop feasibility: 4096 env, 터치다운마다 재풀이 ──
  //   각 env 재풀이 주기 ≈ 스윙 주기(대각쌍) ~0.4s. 필요 throughput = env/주기.
  for (double period : {0.2, 0.4}) {
    double need_4096 = 4096.0 / period;
    double feasible_envs = throughput * period;
    std::printf("재풀이주기 %.1fs → 4096env 필요 %.0f solves/s | 현 throughput 지원 = %.0f env%s\n",
                period, need_4096, feasible_envs, feasible_envs>=4096?"  ✓4096가능":"");
  }
  return 0;
}
