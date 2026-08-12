#!/usr/bin/env python3
"""hwio.py — PACE/마찰시험용 안전 SHM I/O 계층.

설계 원칙 (실기 사고 방지):
  1. **가진은 전부 위치+게인(임피던스) 모드.** 생 토크명령을 쓰지 않는다.
     토크가 `Kp·err` 로 상한이 걸리므로 다리에서 폭주가 구조적으로 불가능하다.
     토크는 "명령" 이 아니라 **측정**해서 마찰을 뽑는다.
  2. **stale 감지 필수.** Emb 는 EtherCAT 이 OP 를 잃어도 마지막 버퍼를 계속 재발행하고
     updated flag 까지 세운다(commEtherCATm.cpp:520 조기 return). 얼어붙은 위치는
     "편차 0" 이라 한계검사를 영원히 통과한다 → 눈먼 채로 계속 명령하게 된다.
     여기서는 (pos,vel,tau) 무변화 지속시간을 **벽시계**로 재서 차단한다.
  3. **종료는 언제나 limp.** 명령을 "안 쓰는" 것은 정지가 아니다 —
     Emb 는 마지막 명령을 1kHz 로 영원히 재전송한다. 반드시 Kp=Kd=0 을 써야 한다.
     정상종료·예외·Ctrl-C·SIGTERM 전부 동일 경로로 limp 한다.
  4. **인가는 램프.** enable 상승엣지에서 측정각을 래치하고 게인을 0→목표로 램프한다
     (스텝 인가 시 토크 스파이크 방지).

단위: SHM 은 deg/deg·s⁻¹. 이 모듈 경계에서 deg 를 쓰고, 물리식은 rad 로 변환해 쓴다.
"""
from __future__ import annotations

import atexit
import contextlib
import ctypes as C
import math
import signal
import sys
import time
from dataclasses import dataclass, field

import numpy as np

F32P = C.POINTER(C.c_float)
I32P = C.POINTER(C.c_int)


def _p(a: np.ndarray) -> F32P:
    return a.ctypes.data_as(F32P)


def _ip(a: np.ndarray) -> I32P:
    return a.ctypes.data_as(I32P)


class SafetyAbort(RuntimeError):
    """안전조건 위반 — 호출측은 잡지 말고 종료시킬 것(limp 는 이미 수행됨)."""


@dataclass
class Sample:
    t: float
    q_deg: float          # 측정 위치
    dq_dps: float         # 측정 속도
    tau: float            # 측정 토크(보고 단위·보고 축 기준)
    cur: float            # 측정 전류
    q_cmd_deg: float      # 그때 내보낸 목표각
    kp: float
    kd: float


@dataclass
class Limits:
    q_min: float
    q_max: float
    tau_trip: float
    tau_trip_ms: float
    vel_trip: float
    err_max: float
    stale_ms: float
    kp_max: float
    kd_max: float


class Hardware:
    """libbipedshm(=hal/shm_bridge.cpp) 위의 안전 래퍼. 한 번에 한 축만 가진한다."""

    def __init__(self, lib_path: str, n_channel: int, rate_hz: float,
                 limits: Limits, recv_wait_ms: int = 3000, enable_ramp_s: float = 0.3,
                 hold_channels=(), hold_kp=0.0, hold_kd=0.0):   # kp/kd: float 또는 {ch: float}
        self.n = int(n_channel)
        self.dt = 1.0 / float(rate_hz)
        self.lim = limits
        self.enable_ramp_s = float(enable_ramp_s)
        # ── 시험축 외 홀드 (2026-08-06, 다리 조립 후 추가) ──────────────────
        #   ★왜 필요한가: 이 클래스는 원래 "한 번에 한 축만 가진" 이었고, 나머지 채널엔
        #     kp=kd=0(=limp)을 썼다. **다리가 없을 땐 그게 옳았다** — 출력축에 아무것도
        #     안 달려 있으니 홀드할 대상 자체가 없었다.
        #     다리를 조립한 뒤로는 그 전제가 깨진다:
        #       (a) PACE 가 빼내는 I_link 는 관절공간 질량행렬의 대각성분 M[i][i] 이고,
        #           그 정의가 **"다른 DOF 의 가속도가 0"** 이다. 무릎이 limp 인 채로
        #           thigh 를 흔들면 다리가 접히면서 강체 가정이 깨져 전혀 다른 값이 나온다.
        #       (b) 안전 — 하위 관절이 무여자면 중력으로 접힌다.
        #   ⇒ 시험축을 가진할 때 hold_channels 를 측정위치에 함께 잡아둔다.
        #   ⚠기본값은 빈 튜플 = 종전 동작 그대로. spec.yaml 의 safety.hold_others 로 켠다.
        self.hold_ch = tuple(int(c) for c in hold_channels)
        # ★홀드 게인은 **축별**로 줄 수 있다(2026-08-10).
        #   이유: 홀드축은 스프링이고 공진 f_n = √(kp·k²/I) 가 처프 대역 안에 들어오면
        #   그 위 주파수에서 홀드가 '자유' 처럼 행동해 식별값이 오염된다.
        #   축마다 I 와 gear_k 가 달라 필요한 kp 도 다르다 — 스칼라 하나로는 못 맞춘다.
        #   스칼라를 주면 종전대로 전 축 동일(하위호환).
        self.hold_kp = hold_kp if isinstance(hold_kp, dict) else float(hold_kp)
        self.hold_kd = hold_kd if isinstance(hold_kd, dict) else float(hold_kd)

        lib = C.CDLL(lib_path)
        lib.bridge_init.restype = C.c_int
        lib.bridge_init.argtypes = [C.c_int]
        lib.bridge_read.restype = C.c_int
        lib.bridge_read.argtypes = [F32P] * 7 + [I32P, I32P]
        lib.bridge_write_pos.restype = C.c_int
        lib.bridge_write_pos.argtypes = [F32P, F32P, F32P, C.c_int]
        lib.bridge_write_mit.restype = C.c_int      # q_des, dq_des, tau_ff, kp, kd, n
        lib.bridge_write_mit.argtypes = [F32P] * 5 + [C.c_int]
        lib.bridge_enable.restype = C.c_int
        lib.bridge_enable.argtypes = [C.c_int]
        self.lib = lib

        z = lambda k=self.n: np.zeros(k, np.float32)
        self._q, self._dq, self._tau, self._cur = z(), z(), z(), z()
        self._rpy, self._acc, self._gyr = z(3), z(3), z(3)
        self._conn = np.zeros(self.n, np.int32)
        self._stt = np.zeros(self.n, np.int32)

        self._prev = None                  # stale 판정용 직전 **전 채널** 스냅샷
        self._last_change_t = 0.0
        self._last_read_t = time.monotonic()
        # 이 간격을 넘겨 read 가 끊기면 그 구간은 stale 판정에서 제외한다(위 read 주석).
        #   제어루프 주기(2ms)보다 넉넉하되, 진짜 두절을 놓칠 만큼 크지 않게.
        self.stale_gap_s = 0.05
        # 신선도를 따지기 시작하는 **명령 누적변위**[deg]. 위 read() 4차 주석 참조.
        # 엔코더 분해능(~0.01°)의 50배 — 이만큼 명령했는데 1LSB 도 안 변하면 동결이다.
        self.stale_cmd_deg = 0.5
        self._cmd_at_change = None
        self._tau_over_since = None
        self._armed = False
        # ★홀드축 목표를 **한 번만** 래치한다 (2026-08-12).
        #   종전엔 arm() 이 매번 "지금 측정각" 을 홀드 목표로 삼았다. 그게 래칫이 된다:
        #     arm → 오차 0 → 중력이 kp·err 균형(thigh 실측 3.0°)까지 끌어내림
        #     → 다음 arm 이 **내려간 자리**를 다시 목표로 → 또 3.0° …
        #   토크프로브는 시행마다 arm 하므로(act_probe_torque_mode.py:47) 4시행 = 12.0°,
        #   err_max 12.0° 와 정확히 같다 → "홀드축 ch1 가 밀렸다" + 눈에 보이는 주저앉음.
        #   ⚠지그를 물리면 기구가 잡아서 **버그가 가려질 뿐** 사라지지 않는다.
        self._hold_target = None
        # ★채널별 트립 상한 (2026-08-12). 없으면 self.lim(=시험축 값)을 쓴다.
        #   ⚠종전엔 위치한계만 채널별이고 토크·속도·추종오차는 **시험축 값을 전 채널에**
        #     그대로 적용했다. foot 시험이면 τ_trip 이 foot 크기 8.0Nm 인데,
        #     hip 은 자세와 무관하게 **상시 5.25 Nm**(모델)을 문다 → 문턱의 66% 점유.
        #     hip 지그를 해제한 2026-08-12 부터 이게 실재 위험이 됐다.
        #   ⇒ 채널별로 준다. 같은 값을 두 곳에서 다르게 다루던 구조를 하나 더 없앤다.
        self.lim_ch: dict = {}
        # ★홀드축 **스톨 감지** (2026-08-12). 오늘 드라이버 파워단을 두 번 잃었다(ch7·ch4).
        #   기전: 축을 기구 스톱 쪽으로 명령하면 kp·err 가 계속 커지는데 축은 안 움직인다
        #   → 대전류가 지속 → 과전류 보호로 래치오프. 소프트웨어가 **죽을 때까지 밀었다.**
        #   ⚠단순한 "오차 크다" 로는 못 잡는다 — hip 은 정상 처짐이 3.5° 라 오탐한다.
        #     구분점은 **중력 대비 초과토크**다: 정상 처짐이면 kp·err ≈ 중력(균형),
        #     스톨이면 kp·err 가 중력+마찰을 넘는데도 안 움직인다.
        #   grav_fn(ch, q_ch) -> Nm 를 꽂으면 켜진다(없으면 검사 안 함 — 하위호환).
        self.grav_fn = None
        self.stall_margin_nm = 2.0      # 중력 대비 이만큼 초과하면 후보
        self.stall_vel_dps = 5.0        # 그런데 이보다 느리면 스톨
        self.stall_ms = 300.0           # 이 시간 지속되면 중단
        self._stall_since: dict = {}
        # ★스톨 감지는 **시험축에도** 건다 (2026-08-12 늦게 발견).
        #   위 검사는 check_hold() 안에 있고 그건 hold_ch 만 돈다. 그런데 --solo 는
        #   hold_ch 가 **빈 리스트**다 → 스톨·파워단 검사가 통째로 안 돈다.
        #   그 상태에서 시험축을 지키는 건 τ_trip(hip 12Nm)·err_max(12°) 뿐인데,
        #   2026-08-12 에 드라이버가 죽은 건 **10.6 Nm** 였다 — 문턱 아래다.
        #   즉 solo 로 hip 을 재는 동안 죽음의 경로가 열려 있었다.
        #   ⇒ _check_impl 에서도 본다. 다만 판정식이 다르다:
        #     홀드축은 kp·err(명령토크)로, 시험축은 **보고토크 τ** 로 본다.
        #     시험축은 τ_ff(중력)+kp·err 로 굴러서 kp·err 만으로는 총량이 안 나온다.
        self.stall_watch = True
        # 파워단 사망 판별 — 이만큼 명령했는데 보고 토크가 이 비율 미만이면 죽은 것이다.
        self.dead_cmd_nm = 3.0
        self.dead_ratio = 0.15
        self._q_cmd = np.zeros(self.n, np.float32)
        # ★상태 발행 훅 (2026-08-11) — PACE 시험 중에도 **뷰어가 자세를 볼 수 있게** 한다.
        #   writer 는 하나여야 해서 시험 중엔 biped_emb 를 끄는데, 그러면 발행자도 같이
        #   사라져 화면이 멎었다. 사람이 로봇 옆에서 토크시험을 돌리는 구간에서
        #   화면이 죽어 있는 건 곤란하다.
        #   ⚠별도 스레드로 안 만든다 — self._q 등 공유버퍼를 두 스레드가 만지면 찢어진다.
        #     read() 안에서 20Hz 로 스로틀한다(읽기는 어차피 매 틱 일어난다).
        self.publish_fn = None
        self.pub_period = 1.0 / 20.0
        self._pub_next = 0.0

        if lib.bridge_init(int(recv_wait_ms)) != 0:
            raise RuntimeError(
                "bridge_init 실패 — Emb 미기동이거나 halGait 초기화 미완료.\n"
                "  Emb 기동 후 5초(=100+4500 tick @1kHz 게이트) 기다린 뒤 재시도할 것.")

        # 종료 경로 일원화: 어떤 신호로 죽어도 limp 를 거친다.
        #   ★SIGHUP(ssh 끊김)·SIGQUIT 도 반드시 포함 — 기본동작이 즉사라 핸들러도 __exit__
        #     도 안 돌고, 그러면 Emb 가 마지막 명령(kp=40 + 처프 setpoint)을 1kHz 로
        #     영원히 재전송한다. atexit 는 정상/예외 종료의 최후 보루.
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass
        atexit.register(self.limp)

    # ── 종료 / limp ─────────────────────────────────────────────────────────
    def _hold_gain_of(self, ch: int) -> tuple[float, float]:
        """홀드축 게인. dict 면 축별, 스칼라면 전 축 동일(하위호환).

        ⚠KeyError 방지 — dict 에 없는 채널은 0(=무여자)이 아니라 **예외**로 드러낸다.
          조용히 0 을 주면 그 축이 홀드 안 된 채로 시험이 돌아 결과가 오염된다.
        """
        if isinstance(self.hold_kp, dict):
            if ch not in self.hold_kp or ch not in self.hold_kd:
                raise KeyError(f"hold_kp/hold_kd 에 채널 {ch} 가 없다 — spec.yaml 확인")
            return float(self.hold_kp[ch]), float(self.hold_kd[ch])
        return float(self.hold_kp), float(self.hold_kd)

    def _on_signal(self, *_):
        self.limp()
        raise SystemExit(130)

    def limits_for(self, ch: int):
        """채널 ch 에 적용할 Limits. 등록된 게 없으면 시험축 한계(self.lim)."""
        return self.lim_ch.get(int(ch), self.lim)

    def safe_hold(self, n_write: int = 25) -> int:
        """★트립 시 **제자리 정지**. limp 대신 쓴다 (2026-08-12, 사용자 지적).

        limp 는 8축 전부 kp=kd=0 이라 **매단 다리가 통째로 떨어진다.** 그러면 발이
        서로 겹친 자세로 착지하고, 다음 실행이 그 자세에서 시작해 또 트립한다 —
        오늘 하루 이 고리를 돌았다(실측: 늘어진 시작자세에서 좌우 발이 **−27mm 침투**).
        ⇒ 위치를 믿을 수 있는 트립(토크·속도·추종오차·스톨)에서는 **현재 측정각을
          명령으로 삼아 그 자리에 세운다.** 오차 0 에서 시작하므로 충격이 없다.
        ⚠stale·동결처럼 **위치를 못 믿는** 경우에는 쓰면 안 된다. 얼어붙은 값을 목표로
          잡으면 실제와 무관한 곳을 향해 밀게 된다. 그때는 limp 가 맞다.
        """
        self._armed = True
        ok = 0
        # ★**전 축**을 잡는다 — _hold_gains 는 hold_ch 만 채우므로 시험축이 빠진다.
        #   시험축을 놓으면 그 다리만 떨어져 결국 같은 충돌 자세가 된다.
        kp_v = np.zeros(self.n, np.float32); kd_v = np.zeros(self.n, np.float32)
        for c in range(self.n):
            try:
                _kp, _kd = self._hold_gain_of(c)
            except KeyError:
                continue                            # 게인이 정의 안 된 채널(미실장)은 건너뛴다
            kp_v[c] = min(_kp, self.lim.kp_max)
            kd_v[c] = min(_kd, self.lim.kd_max)
        for c in range(self.n):
            self._q_cmd[c] = float(self._q[c])     # ★지금 있는 자리
        for _ in range(n_write):
            try:
                self.lib.bridge_enable(1)
                if self.lib.bridge_write_pos(_p(self._q_cmd), _p(kp_v), _p(kd_v), self.n) == 0:
                    ok += 1
                time.sleep(self.dt)
            except Exception:
                pass
        if ok == 0:
            print("\n!! safe_hold 실패 — limp 로 전환한다", file=sys.stderr, flush=True)
            return self.limp()
        return ok

    def limp(self, n_write: int = 25) -> int:
        """Kp=Kd=0 을 반복 기록해 확실히 무여자로 만든다. 어떤 경로로든 마지막에 호출.

        ★성공 횟수를 반환하고, 0 이면 크게 경고한다. bridge_write_pos 는 SetMotorCommand16
          실패 시 즉시 -1 을 반환하고 남은 채널을 포기하므로(shm_bridge.cpp:104),
          실패를 삼키면 '실패한 limp' 와 '성공한 limp' 가 구분되지 않는다.
        """
        self._armed = False
        ok = 0
        zeros = np.zeros(self.n, np.float32)
        for _ in range(n_write):
            try:
                self.lib.bridge_enable(0)
                if self.lib.bridge_write_pos(_p(self._q_cmd), _p(zeros), _p(zeros), self.n) == 0:
                    ok += 1
                time.sleep(self.dt)
            except Exception:
                pass
        if ok == 0:
            print("\n" + "!" * 68 +
                  "\n!! limp 실패 — SHM 에 무여자 명령을 한 번도 쓰지 못했다."
                  "\n!! Emb 는 마지막 명령을 1kHz 로 계속 재전송한다. **모터 전원을 차단할 것**."
                  "\n" + "!" * 68, file=sys.stderr, flush=True)
        return ok

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.limp()
        return False

    # ── 읽기 ────────────────────────────────────────────────────────────────
    def read(self, ch: int) -> tuple[float, float, float, float]:
        self.lib.bridge_read(_p(self._q), _p(self._dq), _p(self._tau), _p(self._cur),
                             _p(self._rpy), _p(self._acc), _p(self._gyr),
                             _ip(self._conn), _ip(self._stt))
        q, dq = float(self._q[ch]), float(self._dq[ch])
        tau, cur = float(self._tau[ch]), float(self._cur[ch])

        now = time.monotonic()
        cur3 = (q, dq, tau)
        # ★stale 은 "**보고 있었는데** 안 변했다" 여야 한다. "안 보고 있던 시간" 을
        #   세면 안 된다 — 2026-08-12 실기에서 그렇게 오탐했다:
        #     시행 사이 time.sleep(0.4) 동안 read() 를 안 하는데, 그때 축은 무여자로
        #     정지해 있어 (q,dq,tau) 가 그대로다 → 재개 첫 read 에서
        #     "상태 정지 396ms > 150ms — EtherCAT OP 이탈 의심" 으로 시험이 중단됐다.
        #     396ms 가 sleep 0.4s 와 일치하는 게 증거다.
        #   ⇒ 직전 read 로부터 오래 지났으면 **판정을 유예**하고 기준시각을 새로 잡는다.
        #     샘플링을 안 한 구간은 신선도를 판단할 근거가 아니다.
        # ★신선도는 **전 채널**로 본다 (2026-08-12). 종전엔 방금 읽은 채널의
        #   (q,dq,tau) 만 봤는데, 그러면 **그 축이 가만히 있기만 해도 stale 이 뜬다.**
        #   실기: 중력추종 홀드가 성공해 축이 멈추자 q 일정 → G(q) 일정 → τ 일정 →
        #   dq 0 이 되어 셋 다 무변화 → "상태 정지 ch4 360ms" 로 트립했다.
        #   **정착이 잘 돼서 트립하는** 구조였다.
        #   ⇒ 진짜 EtherCAT 동결이면 **어느 채널도** 안 변한다. 한 축만 멈춘 것과 구분된다.
        #     (오늘 실측한 진짜 동결: 494표본 10초 동안 고유 조합 1개 · IMU 변화폭 0)
        gap = now - self._last_read_t
        self._last_read_t = now
        # ★신선도는 "**명령이 유의미하게 움직였는데** 측정이 안 따라오는가" 다
        #   (2026-08-12, 4차 수정 — 3차의 '명령 바이트가 바뀌었나' 로는 부족했다).
        #   3차: 가만히 있는 로봇은 (q,dq,tau,rpy) 가 그대로고 그건 정상이므로,
        #        명령이 정지해 있으면 판정을 안 했다. 작업자가 다리를 잡아 세운
        #        --solo 에서 "상태 정지 ch1 391ms" 로 트립한 걸 고친 것이다.
        #   4차: 그래도 **파단푸시에서 오탐했다** — 실기 ch2 "상태 정지 161ms".
        #        파단푸시는 명령을 0.6dps 로 기어가게 하므로 바이트는 매 틱 바뀐다
        #        → 3차 기준으로는 "명령이 움직이는 중". 그런데 **축이 안 움직이는 게
        #        측정법 자체**라 측정값도 그대로다. 즉 정상 동작이 동결로 보였다.
        #        150ms 동안 명령은 0.09° 밖에 안 간다 — 그 정도로는 엔코더 1LSB 도
        #        안 움직이는 게 당연하다.
        #   ⇒ 바이트 변화가 아니라 **누적 변위**로 본다: 마지막으로 측정값이 변한
        #     이후 명령이 stale_cmd_deg 이상 움직였을 때만 신선도를 따진다.
        #       파단푸시 0.6dps → 0.5° 가는 데 833ms (오탐 없음)
        #       홈복귀   15dps → 0.5° 가는 데  33ms (진짜 동결은 여전히 즉시 잡힘)
        allv = (self._q.tobytes(), self._dq.tobytes(), self._tau.tobytes(), self._rpy.tobytes())
        if gap > self.stale_gap_s or self._prev is None or allv != self._prev:
            self._prev = allv
            self._last_change_t = now
            self._cmd_at_change = self._q_cmd.astype(np.float64).copy()
        if self.publish_fn is not None and now >= self._pub_next:
            self._pub_next = now + self.pub_period
            try:                       # 발행 실패가 시험을 멈추면 안 된다
                self.publish_fn(self._q, self._rpy, self._armed)
            except Exception:
                pass
        return q, dq, tau, cur

    def stale_ms(self) -> float:
        """측정이 마지막으로 변한 뒤 지난 시간[ms]. 단, **명령이 유의미하게 움직인
        경우에만** 센다(read() 의 4차 주석 참조). 아직 안 움직였으면 0 을 낸다 —
        판단할 근거가 없는 것이지 신선한 것은 아니다."""
        if self._cmd_at_change is not None:
            moved = float(np.max(np.abs(self._q_cmd.astype(np.float64)
                                        - self._cmd_at_change)))
            if moved < self.stale_cmd_deg:
                return 0.0
        return (time.monotonic() - self._last_change_t) * 1e3

    def wait_fresh(self, timeout_s: float = 5.0, ch: int = 0) -> None:
        """명령 전 필수 — 값이 실제로 변하는지 확인. 플래그는 신선도를 증명하지 못한다."""
        t0 = time.monotonic()
        self._prev = None
        seen = 0
        while time.monotonic() - t0 < timeout_s:
            before = self._prev
            self.read(ch)
            if before is not None and self._prev != before:
                seen += 1
                if seen >= 5:
                    return
            time.sleep(self.dt)
        self.limp()
        raise SafetyAbort(
            "MotorStatus16 이 신선하지 않다(값 무변화). EtherCAT OP 이탈 의심.\n"
            "  Emb 는 스스로 복구하지 못한다 → Emb 재기동 필요. 모터 전원도 확인할 것.")

    # ── 안전검사 ────────────────────────────────────────────────────────────
    def _check(self, ch: int, q: float, dq: float, tau: float, q_cmd: float) -> None:
        """위반 시 **반드시 limp 후** SafetyAbort. 호출부(arm/step/step_response)가
        각자 limp 하는 구조였는데 arm·step_response·wait_fresh 가 빠져 있어서
        인가된 채로 예외가 올라갔다 → 여기서 일원화한다."""
        try:
            self._check_impl(ch, q, dq, tau, q_cmd)
        except SafetyAbort as e:
            # ★위치를 믿을 수 있으면 **제자리 정지**, 못 믿으면 limp (safe_hold 주석 참조).
            #   stale 은 값이 얼어붙은 것이라 그 위치를 목표로 잡으면 안 된다.
            (self.limp if "상태 정지" in str(e) else self.safe_hold)()
            raise

    def _check_impl(self, ch: int, q: float, dq: float, tau: float, q_cmd: float) -> None:
        L = self.limits_for(ch)          # ★채널별 상한(등록 없으면 시험축 self.lim)
        if self.stale_ms() > L.stale_ms:
            raise SafetyAbort(f"상태 정지 **ch{ch}** {self.stale_ms():.0f}ms > {L.stale_ms}ms "
                              f"— 위치를 신뢰할 수 없어 중단(EtherCAT OP 이탈 의심)")
        # ★어느 채널인지 반드시 찍는다 (2026-08-12). 종전엔 값만 있어서
        #   "추종오차 |12.10| > 12.0" 만 보고는 **어느 축인지 알 수 없었다** —
        #   8축을 손으로 추정하다 시간을 버렸다. 진단에 필요한 건 값이 아니라 축이다.
        if not (L.q_min <= q <= L.q_max):
            raise SafetyAbort(f"위치 한계 이탈 **ch{ch}** q={q:.2f}deg "
                              f"∉ [{L.q_min:.2f}, {L.q_max:.2f}]")
        if abs(dq) > L.vel_trip:
            raise SafetyAbort(f"속도 한계 **ch{ch}** |{dq:.1f}| > {L.vel_trip} deg/s")
        if abs(q_cmd - q) > L.err_max:
            raise SafetyAbort(f"추종오차 한계 **ch{ch}** |{q_cmd - q:.2f}| > {L.err_max} deg "
                              f"(명령 {q_cmd:.2f} · 측정 {q:.2f}) "
                              f"— 막힘·게인부족·기계간섭 의심")
        now = time.monotonic()
        if abs(tau) > L.tau_trip:
            self._tau_over_since = self._tau_over_since or now
            if (now - self._tau_over_since) * 1e3 > L.tau_trip_ms:
                raise SafetyAbort(f"토크 한계 **ch{ch}** |{tau:.2f}| > {L.tau_trip} 이 "
                                  f"{L.tau_trip_ms}ms 지속")
        else:
            self._tau_over_since = None

        # ★시험축 스톨 — τ_trip 보다 **훨씬 아래**에서 잡는다 (위 self.stall_watch 주석).
        #   오탐하지 않는 이유:
        #     · 등속 스윕 2dps — |dq|<5 는 맞지만 초과토크가 마찰(~0.6Nm) 뿐 → margin 2.0 미만
        #     · 가속 구간 120dps — 초과토크 I·α 가 2.4Nm 까지 가지만 **빠르게 움직인다** → 제외
        #     · 정상 처짐 — kp·err 가 중력과 균형이라 초과토크가 0 근처
        #   ⇒ "중력보다 2Nm 넘게 내면서 안 움직인다" 는 조합은 막힘밖에 없다.
        #   ⚠**의도적 정지압박**(토크램프·파단푸시)에서는 꺼야 한다 — intentional_push() 참조.
        if self.stall_watch and self.grav_fn is not None:
            g = abs(float(self.grav_fn(ch, q)))
            t_now = time.monotonic()
            if not (abs(tau) - g > self.stall_margin_nm and abs(dq) < self.stall_vel_dps):
                self._stall_since.pop(ch, None)
            elif (t_now - self._stall_since.setdefault(ch, t_now)) * 1e3 > self.stall_ms:
                self._stall_since.pop(ch, None)
                # ★**어디서** 멈췄는지 같이 찍는다 (2026-08-12 실기 ch1).
                #   위치 없이 "스톨" 만 보면 기구스톱인지·손인지·간섭인지 못 가린다.
                #   상자 끝과의 거리를 함께 주면 첫 줄만 읽고도 판별된다.
                _d = min(q - L.q_min, L.q_max - q)
                raise SafetyAbort(
                    f"시험축 **ch{ch}** 스톨 — 토크 {abs(tau):.2f}Nm 이 중력 {g:.2f}Nm 을 "
                    f"{abs(tau)-g:.2f}Nm 넘는데 속도 {dq:+.1f}dps 로 안 움직인다"
                    f"({self.stall_ms:.0f}ms 지속).\n"
                    f"  위치 {q:+.2f}° · 명령 {q_cmd:+.2f}° · 상자 [{L.q_min:+.1f}, "
                    f"{L.q_max:+.1f}]° — 가까운 상자 끝까지 {_d:+.2f}°.\n"
                    f"  {'상자 끝이다 → **기구 스톱**을 밀고 있다.' if _d < 3.0 else '상자 한가운데다 → 기구스톱이 아니라 **간섭·손·케이블**이다.'}\n"
                    f"  기구 스톱·간섭에 밀어붙이고 있다 — **이대로 두면 과전류로 드라이버가**\n"
                    f"  **래치오프된다**(2026-08-12 에 ch0 을 10.6Nm 로 밀다 실제로 그랬다).\n"
                    f"  τ_trip({L.tau_trip}Nm)은 그 사고보다 높아서 못 잡는다 — 그래서 이 검사가 있다.")

    # ── 쓰기 ────────────────────────────────────────────────────────────────
    def latch_hold(self, q_ch=None) -> np.ndarray:
        """홀드축 목표를 확정한다. 이후 arm() 은 **이 값을 재사용**한다(래칫 방지).

        q_ch=None 이면 현재 측정각으로 잡는다. 하니스는 HOME 정렬 직후 **홈 채널각**을
        넘겨 쓴다 — 그러면 처짐이 누적 대신 '홈 대비 일정 오차' 로 고정된다.
        """
        self._hold_target = (np.array([self.read(c)[0] for c in range(self.n)], float)
                             if q_ch is None else np.asarray(q_ch, float).copy())
        return self._hold_target

    def hold_drift(self) -> dict:
        """홀드 목표 대비 현재 측정각 편차[deg]. 처짐이 **보이게** 만든다."""
        if self._hold_target is None:
            return {}
        return {hc: float(self._q[hc]) - float(self._hold_target[hc])
                for hc in self.hold_ch if hc < self.n}

    def arm(self, ch: int, kp: float, kd: float) -> float:
        """측정각을 래치하고 게인을 0→목표로 램프해 인가. 래치된 각을 반환.

        hold_channels 가 있으면 그 축들도 **각자의 측정각**에 함께 래치·램프한다
        (I_link 의 강체 가정 성립 + 하위 관절 중력 붕괴 방지).
        """
        self.wait_fresh(ch=ch)
        q0 = self.read(ch)[0]
        self._q_cmd[:] = 0.0
        self._q_cmd[ch] = q0
        # ★홀드축은 "0" 이 아니라 **지금 있는 자리**에 잡는다. 0 으로 잡으면 인가 순간
        #   현재각만큼의 오차가 그대로 토크가 되어 다리가 홱 움직인다.
        # ★목표는 **첫 래치값 고정**이다. 매번 측정각을 다시 잡으면 래칫이 된다(위 주석).
        if self._hold_target is None:
            self.latch_hold()
        for hc in self.hold_ch:
            if hc != ch and hc < self.n:
                self._q_cmd[hc] = float(self._hold_target[hc])
        # ★enable 이전에 kp=kd=0 을 먼저 기록한다. bridge_enable 은 g_enabled 플래그만
        #   건드리고 SHM 버퍼는 안 쓰므로(shm_bridge.cpp:115), 이 전에 enable 하면
        #   죽은 writer 가 남긴 임의 게인·setpoint 가 순간 authoritative 가 된다.
        self._raw_write(ch, q0, 0.0, 0.0)
        self.lib.bridge_enable(1)
        self._armed = True

        n = max(1, int(self.enable_ramp_s / self.dt))
        for k in range(n):
            s = (k + 1) / n
            # ★홀드축은 **램프하지 않는다** (2026-08-12). 종전엔 hold_scale=s 로 같이 올렸는데,
            #   그러면 램프 0.3s 동안 홀드 게인이 0 에서 시작해 **축이 놓인다.**
            #   중력이 작은 축은 티가 안 났지만 hip 은 5.25Nm 이라 자유낙하한다(α≈135rad/s²):
            #     홈복귀 도착 오차 4.3° → arm 직후 **12.04°** → check_hold 트립.
            #   ⚠홀드축은 이미 goto_home 이 잡아둔 상태다. 램프할 이유가 없다.
            #     첫 arm(아무것도 안 잡힌 상태)에서도 홀드 명령은 **현재 측정각**이라
            #     오차 0 → 전 게인을 즉시 걸어도 충격이 없다.
            self._raw_write(ch, q0, kp * s, kd * s, hold_scale=1.0)
            q, dq, tau, _ = self.read(ch)
            self._check(ch, q, dq, tau, q0)
            time.sleep(self.dt)
        return q0

    def _raw_write_ff(self, ch: int, q_cmd_deg: float, kp: float, kd: float,
                      tau_ff: float) -> None:
        """_raw_write 와 같되 **tau_ff 를 함께** 보낸다(write_mit 경로)."""
        kp = min(max(kp, 0.0), self.lim.kp_max)
        kd = min(max(kd, 0.0), self.lim.kd_max)
        L = self.limits_for(ch)
        q_cmd_deg = min(max(q_cmd_deg, L.q_min), L.q_max)
        self._q_cmd[ch] = q_cmd_deg
        kp_v, kd_v = self._hold_gains(ch)
        kp_v[ch] = kp; kd_v[ch] = kd
        z = np.zeros(self.n, np.float32)
        tv = np.zeros(self.n, np.float32)
        tv[ch] = float(np.clip(tau_ff, -self.lim.tau_trip, self.lim.tau_trip))
        self.lib.bridge_write_mit(_p(self._q_cmd), _p(z), _p(tv), _p(kp_v), _p(kd_v), self.n)

    def _hold_gains(self, ch: int, scale: float = 1.0):
        """홀드축 kp/kd 벡터. 시험축 자리는 호출측이 덮어쓴다."""
        kp_v = np.zeros(self.n, np.float32)
        kd_v = np.zeros(self.n, np.float32)
        for hc in self.hold_ch:
            if hc != ch and hc < self.n:
                _kp, _kd = self._hold_gain_of(hc)
                kp_v[hc] = min(_kp * scale, self.lim.kp_max)
                kd_v[hc] = min(_kd * scale, self.lim.kd_max)
        return kp_v, kd_v

    def _raw_write(self, ch: int, q_cmd_deg: float, kp: float, kd: float,
                   hold_scale: float = 1.0) -> None:
        kp = min(max(kp, 0.0), self.lim.kp_max)      # 스케일 오류가 그대로 드라이버로 가지 않게
        kd = min(max(kd, 0.0), self.lim.kd_max)
        q_cmd_deg = min(max(q_cmd_deg, self.lim.q_min), self.lim.q_max)
        self._q_cmd[ch] = q_cmd_deg
        kp_v, kd_v = self._hold_gains(ch, hold_scale)
        kp_v[ch] = kp; kd_v[ch] = kd
        self.lib.bridge_write_pos(_p(self._q_cmd), _p(kp_v), _p(kd_v), self.n)

    @contextlib.contextmanager
    def intentional_push(self):
        """이 블록 안에서는 **시험축 스톨 감지를 끈다.**

        토크램프(순수토크 프로브)와 파단푸시는 "안 움직이는 축에 토크를 키워 간다"
        그 자체가 측정법이라, 스톨 감지가 켜져 있으면 **측정을 성공시킬 때마다** 튄다.
        ⚠끄는 대신 그 두 경로는 **자체 상한**이 있어야 한다:
            토크램프 → step_torque 의 tau_max 클램프
            파단푸시 → breakaway 의 tau_cap_nm(중력 대비 초과분 상한)
          상한 없이 이 블록을 쓰면 스톨 보호가 통째로 사라진다.
        """
        prev = self.stall_watch
        self.stall_watch = False
        try:
            yield
        finally:
            self.stall_watch = prev
            # 블록 안에서 쌓인 후보 시각을 버린다. 안 버리면 블록을 빠져나온 직후
            # **옛 t0** 로 즉시 stall_ms 를 초과해 오탐한다.
            self._stall_since.clear()

    def check_hold(self) -> None:
        """홀드축이 실제로 잡혀 있는지 확인. 밀려나면 측정이 무효이므로 중단.

        ★홀드가 조용히 실패하는 경로가 있다: 게인이 부족하거나 파워단이 죽은 축은
          중력으로 접히는데, 시험축 기준 검사(_check)는 그걸 절대 못 본다.
          그 상태의 I_total 은 강체 가정이 깨진 값이라 **측정 자체가 무의미**하다.
        """
        for hc in self.hold_ch:
            if hc >= self.n:
                continue
            Lh = self.limits_for(hc)
            err = abs(float(self._q_cmd[hc]) - float(self._q[hc]))
            # ★검사 순서 — **원인이 구체적인 것부터**. 2026-08-12 실기에서
            #   ch0 파워단이 죽었는데 err_max 가 먼저 걸려 "밀렸다" 로 떴다.
            #   그 문구는 '게인 부족·파워단 사망·기계간섭 중 하나' 라 원인을 셋으로 흩는다.
            #   사망은 kp·err 대 보고토크 비로 **단정**할 수 있으니 먼저 본다.
            # ★파워단 사망 — 큰 토크를 **명령했는데 보고 토크가 없다** (2026-08-12).
            #   오늘 세 번 겪었다(ch7 · ch4 두 번). 증상이 매번 다르게 위장됐다:
            #     "추종오차 12.10°" · "상태 정지 ch0 342ms" · "홀드축이 밀렸다"
            #   진짜 원인은 하나인데 **엉뚱한 축·엉뚱한 항목**으로 뜬다.
            #   판별은 간단하다: kp·err 가 크면 그만한 토크가 **보고돼야** 한다.
            #     정상 ch0: 오차 4.03° → 명령 7.03Nm · 보고 6.06Nm (비 0.86)
            #     사망 ch4: 오차 4.96° → 명령 8.66Nm · 보고 0.065Nm (비 **0.008**)
            #   ⚠스톨 감지로는 못 잡는다 — 죽은 축은 멈춰 있는 게 아니라 **떨어진다**(64dps).
            kp_h0, _ = self._hold_gain_of(hc)
            cmd_t = kp_h0 * abs(err) * math.pi / 180.0
            if cmd_t > self.dead_cmd_nm and abs(float(self._tau[hc])) < cmd_t * self.dead_ratio:
                self.limp()      # 죽은 축은 잡을 수 없다 — 나머지도 놓는 게 안전하다
                raise SafetyAbort(
                    f"홀드축 ch{hc} **파워단 사망** — kp·err = {cmd_t:.2f}Nm 을 명령했는데 "
                    f"보고 토크가 {self._tau[hc]:+.3f}Nm 뿐이다(비 "
                    f"{abs(float(self._tau[hc]))/cmd_t:.3f} < {self.dead_ratio}).\n"
                    f"  속도 {self._dq[hc]:+.1f}dps — 명령을 무시하고 중력에 끌려간다.\n"
                    f"  EtherCAT·텔레메트리는 정상인데 드라이버 파워단만 래치오프된 상태다.\n"
                    f"  **모터 전원 OFF → 3초 → ON** 후 Emb 재기동. Emb 만 재기동하면 안 풀린다.")
            if err > Lh.err_max:
                self.safe_hold()
                raise SafetyAbort(
                    f"홀드축 ch{hc} 가 밀렸다 — |명령−측정| {err:.2f}° > {Lh.err_max}°.\n"
                    f"  게인 부족·파워단 사망·기계간섭 중 하나다. 이 상태의 측정은 무효"
                    f"(하위 관절이 움직이면 I_link 의 강체 가정이 깨진다).")

            # ★스톨 — 중력보다 훨씬 큰 토크를 내는데 안 움직인다(위 주석)
            if self.grav_fn is not None:
                kp_h, _ = self._hold_gain_of(hc)
                cmd_tau = kp_h * abs(err) * math.pi / 180.0
                g = abs(float(self.grav_fn(hc, float(self._q[hc]))))
                stuck = (cmd_tau - g > self.stall_margin_nm
                         and abs(float(self._dq[hc])) < self.stall_vel_dps)
                t_now = time.monotonic()
                if not stuck:
                    self._stall_since.pop(hc, None)
                else:
                    t0 = self._stall_since.setdefault(hc, t_now)
                    if (t_now - t0) * 1e3 > self.stall_ms:
                        self.safe_hold()
                        raise SafetyAbort(
                            f"홀드축 ch{hc} **스톨** — 명령토크 {cmd_tau:.2f}Nm 이 중력 "
                            f"{g:.2f}Nm 을 {cmd_tau-g:.2f}Nm 넘는데 속도 "
                            f"{self._dq[hc]:+.1f}dps 로 안 움직인다({self.stall_ms:.0f}ms 지속).\n"
                            f"  기구 스톱에 밀어붙이고 있다 — **이대로 두면 과전류로 드라이버가**\n"
                            f"  **래치오프된다**(2026-08-12 ch7·ch4 에서 실제로 그랬다).\n"
                            f"  목표각을 스톱 안쪽으로 옮길 것(config hold_pose.by_test_ch).")
            if abs(float(self._dq[hc])) > Lh.vel_trip:
                self.safe_hold()
                raise SafetyAbort(f"홀드축 ch{hc} 속도 {self._dq[hc]:.0f} dps > "
                                  f"{Lh.vel_trip} — 잡혀 있지 않다")

    def release_test_axis(self, ch: int, n_write: int = 5) -> None:
        """**시험축만** 무여자로. 홀드축은 계속 잡아둔다.

        ★왜 limp() 를 못 쓰나 (2026-08-11) — limp 는 전 채널 kp=kd=0 이다. 다리가
          없을 땐 그게 곧 '안전' 이었지만, 지금은 매단 다리가 **통째로 떨어진다**.
          HOME 으로 정렬해 놓고 limp 하면 그 자리에서 다시 늘어져 충돌 자세로 돌아간다.
          시험 사이의 '해제' 는 시험축만 푸는 것이 맞다.
        ⚠최종 종료는 여전히 limp 다 — 그때는 사람이 붙어 있고, 매단 상태에서 계속
          토크를 물고 있는 것이 더 위험하다.
        """
        if not self.hold_ch:
            self.limp()
            return
        for _ in range(n_write):
            self._raw_write(ch, float(self._q[ch]), 0.0, 0.0)
            time.sleep(self.dt)

    def verify_driver_live(self, ch: int, kp: float = 40.0, kd: float = 2.0,
                           step_deg: float = 2.0, tau_floor: float = 0.15,
                           move_floor_deg: float = 0.05) -> None:
        """드라이버 파워단이 실제로 살아 있는지 확인. 죽어 있으면 SafetyAbort.

        ★왜 stale 검사로는 부족한가 (2026-08-05 실제 사고):
          EtherCAT·Emb·텔레메트리가 전부 정상인데 **드라이버 파워단만 래치오프**된
          상태가 존재한다. 명령 스트림이 끊기면(Emb 정지 등) 드라이버가 보호
          디스에이블로 들어가고 전원 재투입 전까지 안 풀린다. 이때 위치·속도·토크가
          계속 신선하게 갱신되므로 stale 검사는 통과한다 — 그런데 모터는 죽어 있다.
          그 상태로 순수토크 프로브를 돌려 "1.4 Nm 까지 미동 → 토크모드 미지원" 이라는
          **완전히 틀린 결론**을 낼 뻔했다.

        판정: 알려진 크기의 위치오차를 걸면 살아있는 축은 (a) 마찰 수준 이상의 토크를
        내고 (b) 명령을 따라간다. 죽은 축은 tau≈0.02, 미동(또는 중력으로 낙하)이다.

        ★**양방향**으로 잰다 (2026-08-12). 종전엔 +step 한 방향만 보고 "목표 쪽으로
          움직였나" 를 봤는데, **중력이 큰 축에서 오탐한다**:
            HR_hip — +2.0° 명령의 복원력 kp·step = 100×2° = 3.49 Nm 인데
                     중력이 5.25 Nm 라 축이 **반대로** 1.18° 밀렸다.
                     토크는 5.512 Nm(기준 0.15의 37배)로 파워단이 멀쩡한데도
                     "드라이버 미응답" 으로 중단했다. HL 은 중력 부호가 반대라 통과 —
                     좌우가 갈린 게 단서였다.
          ⇒ +step 과 −step 을 모두 명령하고 **두 정착점의 차이**를 본다.
            중력은 양쪽에 똑같이 실려 차분에서 상쇄된다(살아있으면 ≈ 2·step).
            죽은 축은 어느 쪽을 명령하든 중력 방향으로만 가므로 차이가 ≈ 0 이다.
        """
        q0 = self.arm(ch, kp, kd)
        n = max(1, int(0.8 / self.dt))
        settle = []
        for sgn in (+1.0, -1.0):
            tgt = q0 + sgn * abs(step_deg)
            for _ in range(n):
                s = self.step(ch, tgt, kp, kd)
                time.sleep(self.dt)
            settle.append(s.q_deg)
        spread = settle[0] - settle[1]          # 중력 상쇄된 순수 추종량
        self.release_test_axis(ch)
        if abs(s.tau) < tau_floor or spread < move_floor_deg:
            raise SafetyAbort(
                f"드라이버 미응답 — ±{abs(step_deg):.1f}° 명령(kp={kp:.0f})에 "
                f"토크 {s.tau:+.3f} Nm(기준 {tau_floor}), "
                f"양방향 정착 차이 {spread:+.3f}°(기준 {move_floor_deg}, 정상이면 "
                f"≈{2*abs(step_deg):.1f}°).\n"
                f"  EtherCAT·텔레메트리는 정상이나 **파워단이 래치오프**된 상태다.\n"
                f"  복구: Emb 종료 → 모터 전원 OFF/ON → Emb 재기동.\n"
                f"  (Emb 기동 직후 4.5초 램프에서 관절이 0°로 움직이면 복구 성공)")
        return None

    # ── 순수 토크 경로 ──────────────────────────────────────────────────────
    #   ⚠ 위치+게인 모드와 달리 **토크가 자기제한되지 않는다.** Kp=Kd=0 이면 위치 피드백이
    #     전혀 없어, 마찰(정지 0.71 Nm)을 넘는 토크는 관절을 계속 가속시킨다. 다리 미장착
    #     상태 관성이 0.0375 kg·m² 라 1 Nm 면 α=26.7 rad/s²(=1528 deg/s²) 다.
    #     반드시 tau_max 를 작게 잡고 위치·속도 한계를 매 틱 검사할 것.
    def step_torque(self, ch: int, tau_ff: float, tau_max: float,
                    kp: float = 0.0, kd: float = 0.0, q_des: float | None = None) -> Sample:
        """tau_ff 를 실어 보낸다. 기본은 Kp=Kd=0 인 **순수** 토크 명령.

        ★kp/kd/q_des 는 **게인→토크 핸드오프** 전용이다 (2026-08-12).
          중력이 큰 축(hip 5.25 · calf 0.81 Nm)은 게인을 먼저 0 으로 놓으면
          바이어스를 올리기도 전에 떨어진다 — 드라이런에서 hip 이 207 dps 로 트립했다.
          ⇒ kp·kd 를 홀드값에서 0 으로 내리면서 tau_ff 를 0→bias 로 올린다.
            그동안 q_des 를 **출발점에 고정**해야 kp·err 가 실제로 버틴다
            (기본값처럼 q_cmd=q 로 따라가면 오차가 0 이라 잡는 힘이 없다).
        """
        if not self._armed:
            raise RuntimeError("arm() 을 먼저 호출할 것")
        t = float(np.clip(tau_ff, -abs(tau_max), abs(tau_max)))
        z = np.zeros(self.n, np.float32)
        tv = np.zeros(self.n, np.float32); tv[ch] = t
        # 위치명령: 순수토크(kp=0)면 무의미하나 limp 복귀용으로 현재값 유지.
        # 핸드오프 중(kp>0)에는 **출발점 고정**이어야 버틴다 — q_des 로 준다.
        self._q_cmd[ch] = self._q[ch] if q_des is None else float(q_des)
        # ★순수토크는 **시험축만** 이다. 홀드축은 계속 잡아둔다 — 여기서 전 채널을 0 으로
        #   두면 다리 전체가 무여자가 되어 중력으로 접힌다(다리 조립 후 특히 위험).
        kp_v, kd_v = self._hold_gains(ch)
        kp_v[ch] = min(max(float(kp), 0.0), self.lim.kp_max)
        kd_v[ch] = min(max(float(kd), 0.0), self.lim.kd_max)
        try:
            self.lib.bridge_write_mit(_p(self._q_cmd), _p(z), _p(tv), _p(kp_v), _p(kd_v), self.n)
            q, dq, tau, cur = self.read(ch)
            self._check(ch, q, dq, tau, q)     # 추종오차 검사는 무의미 → q_cmd=q 로 무력화
            self.check_hold()
        except SafetyAbort:
            raise                              # _check 내부에서 이미 limp 함
        return Sample(time.monotonic(), q, dq, tau, cur, float(self._q_cmd[ch]),
                      float(kp), float(kd))

    def step(self, ch: int, q_cmd_deg: float, kp: float, kd: float,
             tau_ff: float = 0.0) -> Sample:
        """1틱: 명령 → 읽기 → 안전검사 → 샘플 반환. 위반 시 limp 후 SafetyAbort.

        ★tau_ff — **중력 피드포워드** (2026-08-12). 중력이 큰 축은 kp 가 그걸 혼자
          감당하느라 정작 축을 밀 여력이 없다:
              hip  max_push 2.5° × kp100 = 4.4Nm   그 자리 중력 4.2Nm → 남는 0.2Nm
              마찰 0.8Nm 을 못 넘어 **6시행 전부 미동**했다(2026-08-12 실기).
          τ_ff = G(q) 를 실으면 kp 는 잔차만 맡는다. 처짐도 같이 사라져 스트로크가 는다.
          ⚠tau_ff 가 0 이면 종전대로 write_pos 를 쓴다(경로를 안 바꾼다).
        """
        if not self._armed:
            raise RuntimeError("arm() 을 먼저 호출할 것")
        if tau_ff:
            self._raw_write_ff(ch, q_cmd_deg, kp, kd, float(tau_ff))
        else:
            self._raw_write(ch, q_cmd_deg, kp, kd)
        q, dq, tau, cur = self.read(ch)
        self._check(ch, q, dq, tau, self._q_cmd[ch])      # 위반 시 내부에서 limp 후 raise
        self.check_hold()                                 # 홀드축이 밀리면 측정 무효
        return Sample(time.monotonic(), q, dq, tau, cur, float(self._q_cmd[ch]), kp, kd)

    # ── 궤적 실행 ───────────────────────────────────────────────────────────
    def run_torque(self, ch: int, tau_fn, duration_s: float, tau_max: float,
                   drift_max_deg: float = 8.0, progress: str | None = None) -> list[Sample]:
        """★τ_ff 가진 루프 (kp=kd=0). `run` 의 토크판.

        왜 필요한가 — **위치처프는 순환이다.** 드라이버가 돌려주는 τ 가
        `kp·err + kd·derr` 로 R² 0.97 재구성되므로(지연 10ms 정렬 시), 그 τ 로 회귀하면
        우리 게인의 그림자를 식별하는 셈이 된다. `kp=kd=0` 이면 재구성할 항이 없고
        **우리가 넣은 τ_ff 가 곧 입력**이라 순환이 원천 소멸한다.

        ⚠**가진축은 복원력이 0 이다.** 위치제어가 아니므로 τ_ff 가 조금만 틀려도
          그 방향으로 계속 흘러간다. `_check` 의 추종오차 검사는 여기서 무력이므로
          **위치 드리프트 워치독**을 따로 둔다 — 시작각 대비 `drift_max_deg` 를 넘으면
          즉시 중단하고 limp 한다.
        ⚠중력은 축마다 다르다(HOME 기준): hip 4.96 · thigh 0.33 · calf 0.36 · **foot 0.10** Nm.
          tau_fn 에 그 상수를 실어야 흘러내리지 않는다. foot 은 사실상 없어도 된다 —
          그래서 **토크 경로 첫 시험은 foot 이 맞다**.
        ⚠홀드축은 step_torque 가 계속 잡아 준다(전 채널 무여자로 만들지 않는다).
        """
        out: list[Sample] = []
        q0 = self.read(ch)[0]
        t0 = time.monotonic()
        k = 0
        # ★스톨 감지를 끈다 — 여기서 "안 움직이는데 토크가 크다" 는 **정상**이다
        #   (파단 직전이 정확히 그 상태다). 상한은 step_torque 의 tau_max 가 쥔다.
        try:
            with self.intentional_push():
                while True:
                    t = time.monotonic() - t0
                    if t >= duration_s:
                        break
                    smp = self.step_torque(ch, float(tau_fn(t)), tau_max)
                    out.append(smp)
                    if abs(smp.q_deg - q0) > drift_max_deg:
                        self.limp()
                        raise SafetyAbort(
                            f"ch{ch} 위치 드리프트 {smp.q_deg - q0:+.2f}° > {drift_max_deg}° "
                            f"— τ_ff 가 중력을 못 이기거나 부호가 반대다. limp 함")
                    k += 1
                    if progress and k % max(1, int(1.0 / self.dt)) == 0:
                        print(f"    {progress} {t:5.1f}/{duration_s:.0f}s "
                              f"q={smp.q_deg:7.2f}(Δ{smp.q_deg-q0:+5.2f}) "
                              f"tau={smp.tau:6.3f}", flush=True)
                    nxt = t0 + k * self.dt
                    slp = nxt - time.monotonic()
                    if slp > 0:
                        time.sleep(slp)
        except SafetyAbort:
            raise
        except Exception:
            self.limp()
            raise
        return out

    def run(self, ch: int, qcmd_fn, duration_s: float, kp: float, kd: float,
            progress: str | None = None, tau_ff_fn=None) -> list[Sample]:
        """qcmd_fn(t)->목표각[deg] 를 duration_s 동안 실행하며 샘플을 모은다.
        절대시각 스케줄러(누적 드리프트 없음)."""
        out: list[Sample] = []
        t0 = time.monotonic()
        k = 0
        while True:
            t = time.monotonic() - t0
            if t >= duration_s:
                break
            out.append(self.step(ch, float(qcmd_fn(t)), kp, kd,
                             tau_ff=(tau_ff_fn(float(self._q[ch])) if tau_ff_fn else 0.0)))
            k += 1
            if progress and k % max(1, int(1.0 / self.dt)) == 0:
                print(f"    {progress} {t:5.1f}/{duration_s:.0f}s "
                      f"q={out[-1].q_deg:7.2f} tau={out[-1].tau:6.3f}", flush=True)
            nxt = t0 + k * self.dt
            slp = nxt - time.monotonic()
            if slp > 0:
                time.sleep(slp)
        return out

    def step_response(self, ch: int, delta_deg: float, kp: float, kd: float,
                      window_s: float = 0.30, settle_s: float = 0.6) -> dict:
        """왕복지연 측정용 스텝응답.

        루프주기(5ms)보다 고운 분해능이 필요하므로 **sleep 없이 타이트 폴링**한다.
        명령은 t=0 에 한 번만 쓴다 — Emb 는 MotCmd 를 래치해 1kHz 로 계속 재전송하므로
        매 틱 다시 쓸 필요가 없고, 다시 쓰면 쓰기시각이 흐려져 지연 측정이 오염된다.

        반환: t(=명령시각 기준 상대), q, dq, tau 배열 + baseline 통계.
        """
        q0 = self.read(ch)[0]
        base = []
        n_settle = max(1, int(settle_s / self.dt))
        for _ in range(n_settle):
            s = self.step(ch, q0, kp, kd)
            base.append((s.tau, s.q_deg))
            time.sleep(self.dt)
        tau_b = float(np.mean([b[0] for b in base[-n_settle // 2:]]))
        tau_sd = float(np.std([b[0] for b in base[-n_settle // 2:]]))

        ts, qs, dqs, taus = [], [], [], []
        t0 = time.monotonic()
        self._raw_write(ch, q0 + delta_deg, kp, kd)      # ★ 단 한 번
        while True:
            t = time.monotonic() - t0
            if t >= window_s:
                break
            q, dq, tau, _ = self.read(ch)
            ts.append(t); qs.append(q); dqs.append(dq); taus.append(tau)
            self._check(ch, q, dq, tau, q0 + delta_deg)

        # 원위치 복귀(램프) — 예외가 나도 반드시 수행
        try:
            self.goto(ch, q0, kp, kd, speed_dps=6.0)
        finally:
            pass
        return {"t": np.array(ts), "q": np.array(qs), "dq": np.array(dqs),
                "tau": np.array(taus), "tau_base": tau_b, "tau_base_sd": tau_sd,
                "q0": q0, "delta": delta_deg}

    def _raw_write_all(self, q_cmd_vec, kp, kd, q_box=None) -> None:
        """전 채널을 **동시에** 구동. goto_all 전용. kp/kd 는 스칼라 또는 {ch: 값}.

        ★축별 dict 를 받는 이유: 스칼라 하나로는 못 맞춘다. spec 의 kp 40 은 다리 미장착
          시절 값이라 hip 중력 4.96Nm 에 **7.1° 처진다**(4.96/40 rad) — 목표에 못 닿는다.
          배포게인(hip100/thigh50/calf50/foot30)을 그대로 쓰는 게 맞다.
        """
        kp_v = np.zeros(self.n, np.float32); kd_v = np.zeros(self.n, np.float32)
        for c in range(self.n):
            v = float(q_cmd_vec[c])
            # ★채널마다 한계가 다르다. self.lim 은 **시험축** 한계이므로 다축 경로에
            #   그걸 쓰면 안 된다 — 2026-08-11 에 foot 의 [−40,30] 이 전 채널에 적용돼
            #   calf/foot 목표가 최대 57° 잘려나갔다(예외 없이 조용히).
            #   q_box 가 없으면 클램프하지 않는다: 여기서 잘못 자르는 것이
            #   안 자르는 것보다 위험하고, 트립(토크·속도·추종오차)이 따로 있다.
            if q_box is not None and c in q_box:
                blo, bhi = q_box[c]
                v = min(max(v, blo), bhi)
            self._q_cmd[c] = v
            _kp, _kd = self._gain_at(c, kp, kd)
            kp_v[c] = min(max(_kp, 0.0), self.lim.kp_max)
            kd_v[c] = min(max(_kd, 0.0), self.lim.kd_max)
        self.lib.bridge_write_pos(_p(self._q_cmd), _p(kp_v), _p(kd_v), self.n)

    def _gain_at(self, c: int, kp, kd):
        """채널 c 의 (kp, kd). 스칼라면 그대로, dict 면 **없는 채널은 0**.

        ★없는 채널을 0 으로 두는 게 왜 맞나 — n_channel 은 10 이고 8~9 는 허리다.
          PACE 는 다리 8축만 다루므로 축별 게인 dict 의 키도 0~7 뿐이다. 예전엔
          kp[c] 로 바로 찍어 **KeyError: 8 로 죽었다**(2026-08-11 실기).
          허리를 0 으로 두는 것은 arm() 이 이미 하고 있는 동작과 같다
          (_hold_gains 는 hold_ch 에만 게인을 준다) — 즉 거동 변화가 없다.
        ⚠게인 0 인 채널은 goto_all 에서 **목표 이동도 하지 않는다**. 안 그러면
          나중에 누가 허리에 게인을 넣는 순간 조용히 같이 끌려간다.
        """
        _kp = float(kp.get(c, 0.0)) if isinstance(kp, dict) else float(kp)
        _kd = float(kd.get(c, 0.0)) if isinstance(kd, dict) else float(kd)
        return _kp, _kd

    # ── goto_all 은 삭제됨 (2026-08-11) ──────────────────────────────────────
    #   HOME 복귀는 pace/homing.py 의 goto_home() 이 담당하고, 그건 GUI 와 **같은**
    #   control/home.py:HomeTrajectory 를 쓴다. 여기 따로 두면 구현이 둘이 된다:
    #     · 채널각 직선보간 ≠ 모델각 직선보간 (calf→foot 커플링 때문)
    #     · 가속도 한계 없음 (HomeTrajectory 는 v·a 둘 다 지킨다)
    #     · jog 한계 클램프·"잘렸다" 보고 없음
    #   _raw_write_all 은 남는다 — goto_home 이 그걸 쓴다.
    def goto(self, ch: int, q_target_deg: float, kp: float, kd: float,
             speed_dps: float = 8.0, tau_ff_fn=None) -> None:
        """현재 위치에서 목표각까지 등속 램프(안전 이동)."""
        q0 = self.read(ch)[0]
        dist = q_target_deg - q0
        T = abs(dist) / max(speed_dps, 1e-6)
        if T < 1e-3:
            return
        self.run(ch, lambda t: q0 + dist * min(t / T, 1.0), T + 0.3, kp, kd,
                 tau_ff_fn=tau_ff_fn)


def samples_to_arrays(s: list[Sample]) -> dict[str, np.ndarray]:
    """샘플 리스트 → 배열 dict. t 는 0 기준으로 이동."""
    if not s:
        return {k: np.zeros(0) for k in ("t", "q", "dq", "tau", "cur", "q_cmd")}
    t0 = s[0].t
    return {
        "t":     np.array([x.t - t0 for x in s]),
        "q":     np.array([x.q_deg for x in s]),
        "dq":    np.array([x.dq_dps for x in s]),
        "tau":   np.array([x.tau for x in s]),
        "cur":   np.array([x.cur for x in s]),
        "q_cmd": np.array([x.q_cmd_deg for x in s]),
    }


DEG = math.pi / 180.0
