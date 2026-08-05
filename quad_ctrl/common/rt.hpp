#pragma once
// RT(실시간) 스케줄링 셋업 — robot_main·tools 공용. Linux 전용.
//
//   왜 필요한가: 제어루프가 일반 우선순위면 다른 프로세스(RobotEmbedded 는 CPU ~90% 를 쓴다)에
//   선점당해 주기가 밀린다. 1kHz 루프에서 수 ms 지터는 곧 지연 예산(12ms) 잠식이다.
//
//   ★권한: SCHED_FIFO 는 root 또는 rtprio ulimit 필요. 이 Pi 는 rtprio 한도가 0 이라
//     **sudo 로 실행해야 걸린다**(RobotEmbedded 와 동일). 권한이 없으면 조용히 넘어가지 않고
//     실패를 명시적으로 알린다 — "RT 켠 줄 알았는데 안 켜져 있었다"가 제일 위험하다.
//
//   ★우선순위: 기본 80. 너무 높이면 시스템 데몬을 굶긴다. 우리 루프는 clock_nanosleep 으로
//     매 주기 양보하므로 폭주하지 않지만, 무한루프 버그와 겹치면 콘솔이 잠길 수 있다 —
//     그래서 SCHED_FIFO 를 쓰되 반드시 sleep 하는 루프에서만 쓸 것.
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>

namespace qc {

struct RtStatus {
  bool fifo = false;      // SCHED_FIFO 적용됨
  bool locked = false;    // mlockall 적용됨
  int  prio = 0;
  char why[192] = {0};    // 실패 사유(정직한 보고용)
};

// RT 우선순위 + 메모리 잠금 시도. 실패해도 실행은 계속하되 status 로 사실을 돌려준다.
inline RtStatus rt_setup(int prio = 80) {
  RtStatus st; st.prio = prio;

  // ① 페이지 폴트 제거 — 스왑/지연할당으로 인한 산발적 수 ms 스파이크 방지.
  //    memlock 한도만 충분하면 무권한으로도 된다.
  if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) st.locked = true;
  else std::snprintf(st.why, sizeof(st.why), "mlockall: %s", std::strerror(errno));

  // ② SCHED_FIFO — 실패 사유를 구분해서 남긴다(EPERM=권한, EINVAL=우선순위 범위).
  struct sched_param sp; std::memset(&sp, 0, sizeof(sp));
  sp.sched_priority = prio;
  if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) {
    st.fifo = true;
  } else {
    const int e = errno;
    const size_t n = std::strlen(st.why);
    std::snprintf(st.why + n, sizeof(st.why) - n, "%sSCHED_FIFO(prio %d): %s%s",
                  n ? " · " : "", prio, std::strerror(e),
                  e == EPERM ? " — sudo 로 실행하거나 rtprio ulimit 상향 필요" : "");
  }
  return st;
}

inline void rt_report(const RtStatus& st) {
  std::printf("[rt] SCHED_FIFO=%s(prio %d) · mlockall=%s\n",
              st.fifo ? "ON" : "off", st.prio, st.locked ? "ON" : "off");
  if (st.why[0]) std::printf("[rt] ⚠ %s\n", st.why);
}

}  // namespace qc
