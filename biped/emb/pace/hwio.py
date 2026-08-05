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
                 limits: Limits, recv_wait_ms: int = 3000, enable_ramp_s: float = 0.3):
        self.n = int(n_channel)
        self.dt = 1.0 / float(rate_hz)
        self.lim = limits
        self.enable_ramp_s = float(enable_ramp_s)

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

        self._prev = None                  # stale 판정용 직전 (q,dq,tau)
        self._last_change_t = 0.0
        self._tau_over_since = None
        self._armed = False
        self._q_cmd = np.zeros(self.n, np.float32)

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
    def _on_signal(self, *_):
        self.limp()
        raise SystemExit(130)

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
        if self._prev is None or cur3 != self._prev:
            self._prev = cur3
            self._last_change_t = now
        return q, dq, tau, cur

    def stale_ms(self) -> float:
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
        except SafetyAbort:
            self.limp()
            raise

    def _check_impl(self, ch: int, q: float, dq: float, tau: float, q_cmd: float) -> None:
        L = self.lim
        if self.stale_ms() > L.stale_ms:
            raise SafetyAbort(f"상태 정지 {self.stale_ms():.0f}ms > {L.stale_ms}ms "
                              f"— 위치를 신뢰할 수 없어 중단(EtherCAT OP 이탈 의심)")
        if not (L.q_min <= q <= L.q_max):
            raise SafetyAbort(f"위치 한계 이탈 q={q:.2f}deg ∉ [{L.q_min}, {L.q_max}]")
        if abs(dq) > L.vel_trip:
            raise SafetyAbort(f"속도 한계 |{dq:.1f}| > {L.vel_trip} deg/s")
        if abs(q_cmd - q) > L.err_max:
            raise SafetyAbort(f"추종오차 한계 |{q_cmd - q:.2f}| > {L.err_max} deg "
                              f"(막힘·게인부족·기계간섭 의심)")
        now = time.monotonic()
        if abs(tau) > L.tau_trip:
            self._tau_over_since = self._tau_over_since or now
            if (now - self._tau_over_since) * 1e3 > L.tau_trip_ms:
                raise SafetyAbort(f"토크 한계 |{tau:.2f}| > {L.tau_trip} 이 "
                                  f"{L.tau_trip_ms}ms 지속")
        else:
            self._tau_over_since = None

    # ── 쓰기 ────────────────────────────────────────────────────────────────
    def arm(self, ch: int, kp: float, kd: float) -> float:
        """측정각을 래치하고 게인을 0→목표로 램프해 인가. 래치된 각을 반환."""
        self.wait_fresh(ch=ch)
        q0 = self.read(ch)[0]
        self._q_cmd[:] = 0.0
        self._q_cmd[ch] = q0
        # ★enable 이전에 kp=kd=0 을 먼저 기록한다. bridge_enable 은 g_enabled 플래그만
        #   건드리고 SHM 버퍼는 안 쓰므로(shm_bridge.cpp:115), 이 전에 enable 하면
        #   죽은 writer 가 남긴 임의 게인·setpoint 가 순간 authoritative 가 된다.
        self._raw_write(ch, q0, 0.0, 0.0)
        self.lib.bridge_enable(1)
        self._armed = True

        n = max(1, int(self.enable_ramp_s / self.dt))
        for k in range(n):
            s = (k + 1) / n
            self._raw_write(ch, q0, kp * s, kd * s)
            q, dq, tau, _ = self.read(ch)
            self._check(ch, q, dq, tau, q0)
            time.sleep(self.dt)
        return q0

    def _raw_write(self, ch: int, q_cmd_deg: float, kp: float, kd: float) -> None:
        kp = min(max(kp, 0.0), self.lim.kp_max)      # 스케일 오류가 그대로 드라이버로 가지 않게
        kd = min(max(kd, 0.0), self.lim.kd_max)
        q_cmd_deg = min(max(q_cmd_deg, self.lim.q_min), self.lim.q_max)
        self._q_cmd[ch] = q_cmd_deg
        kp_v = np.zeros(self.n, np.float32); kp_v[ch] = kp
        kd_v = np.zeros(self.n, np.float32); kd_v[ch] = kd
        self.lib.bridge_write_pos(_p(self._q_cmd), _p(kp_v), _p(kd_v), self.n)

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
        내고 (b) 목표 쪽으로 실제로 움직인다. 죽은 축은 tau≈0.02, 미동이다.
        """
        q0 = self.arm(ch, kp, kd)
        tgt = q0 + step_deg
        n = max(1, int(0.8 / self.dt))
        for _ in range(n):
            s = self.step(ch, tgt, kp, kd)
            time.sleep(self.dt)
        moved = (s.q_deg - q0) * (1 if step_deg > 0 else -1)
        self.limp()
        if abs(s.tau) < tau_floor or moved < move_floor_deg:
            raise SafetyAbort(
                f"드라이버 미응답 — {step_deg:+.1f}° 명령(kp={kp:.0f})에 "
                f"토크 {s.tau:+.3f} Nm(기준 {tau_floor}), 이동 {moved:+.3f}°(기준 {move_floor_deg}).\n"
                f"  EtherCAT·텔레메트리는 정상이나 **파워단이 래치오프**된 상태다.\n"
                f"  복구: Emb 종료 → 모터 전원 OFF/ON → Emb 재기동.\n"
                f"  (Emb 기동 직후 4.5초 램프에서 관절이 0°로 움직이면 복구 성공)")
        return None

    # ── 순수 토크 경로 ──────────────────────────────────────────────────────
    #   ⚠ 위치+게인 모드와 달리 **토크가 자기제한되지 않는다.** Kp=Kd=0 이면 위치 피드백이
    #     전혀 없어, 마찰(정지 0.71 Nm)을 넘는 토크는 관절을 계속 가속시킨다. 다리 미장착
    #     상태 관성이 0.0375 kg·m² 라 1 Nm 면 α=26.7 rad/s²(=1528 deg/s²) 다.
    #     반드시 tau_max 를 작게 잡고 위치·속도 한계를 매 틱 검사할 것.
    def step_torque(self, ch: int, tau_ff: float, tau_max: float) -> Sample:
        """Kp=Kd=0 + tau_ff 만 실어 보내는 순수 토크 명령."""
        if not self._armed:
            raise RuntimeError("arm() 을 먼저 호출할 것")
        t = float(np.clip(tau_ff, -abs(tau_max), abs(tau_max)))
        z = np.zeros(self.n, np.float32)
        tv = np.zeros(self.n, np.float32); tv[ch] = t
        self._q_cmd[ch] = self._q[ch]          # 위치명령은 무의미하나 limp 복귀용으로 현재값 유지
        try:
            self.lib.bridge_write_mit(_p(self._q_cmd), _p(z), _p(tv), _p(z), _p(z), self.n)
            q, dq, tau, cur = self.read(ch)
            self._check(ch, q, dq, tau, q)     # 추종오차 검사는 무의미 → q_cmd=q 로 무력화
        except SafetyAbort:
            raise                              # _check 내부에서 이미 limp 함
        return Sample(time.monotonic(), q, dq, tau, cur, float(self._q_cmd[ch]), 0.0, 0.0)

    def step(self, ch: int, q_cmd_deg: float, kp: float, kd: float) -> Sample:
        """1틱: 명령 → 읽기 → 안전검사 → 샘플 반환. 위반 시 limp 후 SafetyAbort."""
        if not self._armed:
            raise RuntimeError("arm() 을 먼저 호출할 것")
        self._raw_write(ch, q_cmd_deg, kp, kd)
        q, dq, tau, cur = self.read(ch)
        self._check(ch, q, dq, tau, self._q_cmd[ch])      # 위반 시 내부에서 limp 후 raise
        return Sample(time.monotonic(), q, dq, tau, cur, float(self._q_cmd[ch]), kp, kd)

    # ── 궤적 실행 ───────────────────────────────────────────────────────────
    def run(self, ch: int, qcmd_fn, duration_s: float, kp: float, kd: float,
            progress: str | None = None) -> list[Sample]:
        """qcmd_fn(t)->목표각[deg] 를 duration_s 동안 실행하며 샘플을 모은다.
        절대시각 스케줄러(누적 드리프트 없음)."""
        out: list[Sample] = []
        t0 = time.monotonic()
        k = 0
        while True:
            t = time.monotonic() - t0
            if t >= duration_s:
                break
            out.append(self.step(ch, float(qcmd_fn(t)), kp, kd))
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

    def goto(self, ch: int, q_target_deg: float, kp: float, kd: float,
             speed_dps: float = 8.0) -> None:
        """현재 위치에서 목표각까지 등속 램프(안전 이동)."""
        q0 = self.read(ch)[0]
        dist = q_target_deg - q0
        T = abs(dist) / max(speed_dps, 1e-6)
        if T < 1e-3:
            return
        self.run(ch, lambda t: q0 + dist * min(t / T, 1.0), T + 0.3, kp, kd)


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
