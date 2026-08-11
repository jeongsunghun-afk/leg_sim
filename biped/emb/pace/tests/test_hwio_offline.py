#!/usr/bin/env python3
"""test_hwio_offline.py — 하드웨어 없이 hwio 경로를 끝까지 돌리는 스모크 테스트.

★왜 만들었나 (2026-08-11)
  goto_all(HOME 정렬)이 실기에서 **두 번 연속** 죽었다. 둘 다 로봇을 세워 두고
  사람이 실행해야만 드러나는 형태였다:
      1회차  KeyError: 8              — 축별 게인 dict 에 허리(8,9) 키가 없다
      2회차  AttributeError: 'rate'   — self.rate 는 없다(주기는 self.dt)
  둘 다 **한 줄짜리 이름 오류**이고, 둘 다 `python3 -c "import"` 나 구문검사로는
  안 잡힌다. 실행경로를 실제로 밟아야만 나온다.
  ⇒ 공유 라이브러리를 스텁으로 갈아끼워 arm→goto_all→토크루프를 전부 돌린다.

⚠이 테스트가 **보장하지 않는** 것 — 물리다. 스텁 플랜트는 1차 지연이고 마찰·중력·
  백래시·통신지연이 없다. 여기 통과는 "코드가 끝까지 돈다" 이지 "값이 맞다" 가 아니다.
  안전한계(트립·stale·드리프트)의 동작만 확인한다.

실행: python3 tests/test_hwio_offline.py
"""
from __future__ import annotations

import ctypes as C
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PACE = os.path.dirname(HERE)
sys.path.insert(0, PACE)


# ── 가짜 libbipedshm ────────────────────────────────────────────────────────
class _Fn:
    """ctypes 함수 흉내 — restype/argtypes 대입을 받아 주고 호출은 파이썬으로."""

    def __init__(self, fn):
        self._fn = fn
        self.restype = None
        self.argtypes = None

    def __call__(self, *a):
        return self._fn(*a)


class FakeLib:
    """1차 지연 플랜트 + 중력. 채널별 독립.

    q̇ 를 (kp·err − kd·q̇ + τ_ff + τ_grav)/I 로 적분한다. 실물과 다르지만
    **코드경로를 밟는 것** 이 목적이므로 충분하다.
    """

    def __init__(self, n=10, q_init=None, tau_grav=None, dt=0.002):
        self.n = n
        self.dt = dt
        self.q = np.array(q_init if q_init is not None else np.zeros(n), float)
        self.dq = np.zeros(n)
        self.tau = np.zeros(n)
        self.tau_grav = np.array(tau_grav if tau_grav is not None else np.zeros(n), float)
        self.I = np.full(n, 0.06)
        # ★쿨롱 마찰 — 없으면 τ_ff 시험이 성립하지 않는다. 마찰이 0 이면 아무리 작은
        #   토크도 계속 가속해서 '브레이크어웨이' 라는 개념 자체가 사라진다.
        #   0.65 Nm 은 hip 에서 실측한 정지마찰(0.63~0.71)을 가져온 값이다.
        self.tau_c = np.full(n, 0.65)
        self.enabled = 0
        self.ticks = 0
        self.max_kp_seen = np.zeros(n)
        for name in ("bridge_init", "bridge_read", "bridge_write_pos",
                     "bridge_write_mit", "bridge_enable"):
            setattr(self, name, _Fn(getattr(self, "_" + name)))

    # -- 물리 --
    def _integrate(self, q_cmd, kp, kd, tau_ff):
        """τ = kp·(q_cmd−q) + kd·(0−q̇) + τ_ff + τ_grav,  q̈ = τ/I.  (rad 로 계산)

        감쇠는 **kd 로만** 준다. 예전엔 `dq *= 0.9` 라는 인공감쇠를 넣었는데, 그러면
        속도추종 지연이 kd 와 무관해져 60dps 에서 19° 나 뒤처졌다 — 실기 사양이 아니라
        스텁의 인공물이 추종오차 트립을 때리는 형태였다.
        """
        D = np.pi / 180.0
        act = (kp * (q_cmd - self.q) - kd * self.dq) * D
        if not self.enabled:                 # shm_bridge.cpp:112 — 전원 off = kp/kd/τ 0
            act = np.zeros(self.n)
            tau_ff = np.zeros(self.n)
        self.tau = act + tau_ff
        net = self.tau + self.tau_grav
        mov = np.abs(self.dq) > 1e-3                     # 운동/정지 마찰 분기
        net = np.where(mov, net - np.sign(self.dq) * self.tau_c,
                       np.where(np.abs(net) > self.tau_c,
                                net - np.sign(net) * self.tau_c, 0.0))
        ddq = net / self.I / D
        self.dq += ddq * self.dt
        self.q += self.dq * self.dt
        self.ticks += 1

    # -- ctypes 경계 --
    def _bridge_init(self, wait_ms):
        return 0

    def _bridge_enable(self, on):
        self.enabled = int(on)
        return 0

    def _bridge_read(self, qp, dqp, taup, curp, rpyp, accp, gyrp, connp, sttp):
        # stale 판정은 (q,dq,tau) 무변화를 본다. 정지 시에도 갱신되는 걸 흉내내려
        # 아주 작은 디더를 tau 에 준다(실물 센서 노이즈에 해당).
        d = 1e-4 if (self.ticks % 2) else -1e-4
        for i in range(self.n):
            qp[i] = float(self.q[i])
            dqp[i] = float(self.dq[i])
            taup[i] = float(self.tau[i] + d)
            curp[i] = float(self.tau[i])
            connp[i] = 1
            sttp[i] = 1
        for i in range(3):
            rpyp[i] = accp[i] = gyrp[i] = 0.0
        self.ticks += 1
        return 0

    def _bridge_write_pos(self, qp, kpp, kdp, n):
        q_cmd = np.array([qp[i] for i in range(n)])
        kp = np.array([kpp[i] for i in range(n)])
        kd = np.array([kdp[i] for i in range(n)])
        self.max_kp_seen = np.maximum(self.max_kp_seen, kp)
        self._integrate(q_cmd, kp, kd, np.zeros(n))
        return 0

    def _bridge_write_mit(self, qp, dqp, tffp, kpp, kdp, n):
        q_cmd = np.array([qp[i] for i in range(n)])
        kp = np.array([kpp[i] for i in range(n)])
        kd = np.array([kdp[i] for i in range(n)])
        tff = np.array([tffp[i] for i in range(n)])
        self._integrate(q_cmd, kp, kd, tff)
        return 0


# ── 픽스처 ──────────────────────────────────────────────────────────────────
def _load_spec():
    return yaml.safe_load(open(os.path.join(PACE, "spec.yaml")))


def _make_hw(fake, spec, hold, rate_hz=500.0, ramp_s=0.02, align=True):
    import hwio
    real_cdll = C.CDLL
    C.CDLL = lambda path: fake                     # ★스텁 주입
    try:
        s = spec["shm"]
        sf, g = spec["safety"], spec["gains"]
        import actuator_test as at
        # 실기와 동일: 정렬 구간은 기구한계, 시험 구간은 spec 여유폭
        lo, hi = (at._mech_limit_box()[3] if align
                  else at._ch_limit_box(spec, pin_home=True)[3])   # 시험축 = HL_foot
        lim = hwio.Limits(q_min=lo, q_max=hi,
                          tau_trip=float(sf["tau_trip_nm"]),
                          tau_trip_ms=float(sf["tau_trip_ms"]),
                          vel_trip=float(sf["vel_trip_dps"]),
                          err_max=float(sf["err_max_deg"]),
                          stale_ms=float(sf["stale_ms"]),
                          kp_max=float(g["kp_max"]), kd_max=float(g["kd_max"]))
        return hwio.Hardware(s["lib"], int(s["n_channel"]), rate_hz, lim,
                             int(s["recv_wait_ms"]), ramp_s,
                             hold_channels=hold,
                             hold_kp=sf.get("hold_kp", 40.0),
                             hold_kd=sf.get("hold_kd", 2.0))
    finally:
        C.CDLL = real_cdll


def _home_targets():
    sys.path.insert(0, os.path.join(os.path.dirname(PACE), "interface"))
    from joint_map import JointMap
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(PACE),
                                           "config", "biped_emb.yaml")))
    jm = JointMap(cfg)
    return jm.q_joint_to_ch([float(x) for x in cfg["home"]["q_deg"]]), jm


# ── 테스트 ──────────────────────────────────────────────────────────────────
FAILED: list[str] = []


def check(name, cond, detail=""):
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def t_goto_all_home():
    """실기에서 두 번 죽은 그 호출. 늘어진 자세 → HOME."""
    print("\n[1] goto_all — 늘어진 자세에서 HOME 정렬 (실기 재현)")
    spec = _load_spec()
    tgt, jm = _home_targets()
    n = int(spec["shm"]["n_channel"])
    # 제어기를 끄면 다리가 늘어진다. 실측 자세를 채널각으로 환산해 초기값으로 쓴다.
    q_droop = jm.q_joint_to_ch([-12., 39., -53., -31., 13., 33., -46., -24.])
    fake = FakeLib(n, q_init=q_droop, dt=1 / 500.)
    sf = spec.get("safety", {})
    hw = _make_hw(fake, spec, hold=[0, 1, 2, 4, 5, 6, 7])
    kp, kd = sf.get("hold_kp", 40.0), sf.get("hold_kd", 2.0)
    import actuator_test as at
    box = at._mech_limit_box()      # 실기와 동일: 정렬 구간은 **기구한계** 상자
    # (spec 상자를 쓰면 늘어진 ch3=+17.55° 가 상한 12.75° 로 잘려 시작부터 어긋난다 —
    #  정렬용 상자가 왜 기구한계여야 하는지가 여기서 드러난다.)
    hw.arm(3, 30.0, 2.0)
    # 실기는 8dps 지만 테스트 시간을 줄이려 60dps. 스텁 플랜트가 따라올 수 있는 상한이다
    # (400dps 는 스텁이 19° 뒤처져 추종오차 트립에 걸린다 — 실기 사양과 무관한 인공물).
    T = hw.goto_all(tgt, kp=kp, kd=kd, speed_dps=60.0, q_box=box)
    err = np.array([hw.read(c)[0] for c in range(n)]) - np.asarray(tgt, float)
    drv = sorted(kp) if isinstance(kp, dict) else list(range(n))
    check("예외 없이 완주", True, f"T={T:.2f}s")
    # ★잔류오차 = τ_c/kp 가 물리적으로 맞다(마찰 데드밴드). foot kp30·τ_c0.65 →
    #   0.65/30 rad = 1.24°. 실기의 hip 은 중력 4.96/kp100 = 2.84° 였다.
    #   ⇒ goto_all 의 tol_deg=5.0 은 이 잔류를 오탐하지 않으면서 '못 닿음' 과는 구분된다.
    check("구동축이 목표에 도달(마찰 데드밴드 τ_c/kp 이내)",
          float(np.max(np.abs(err[drv]))) < 2.0,
          f"최대오차 {np.max(np.abs(err[drv])):.3f}° (예상 τ_c/kp=1.24°)")
    check("허리(8,9)는 제자리", abs(fake.q[8] - q_droop[8]) < 1e-6
          and abs(fake.q[9] - q_droop[9]) < 1e-6)
    check("허리에 게인이 나가지 않음", fake.max_kp_seen[8] == 0.0
          and fake.max_kp_seen[9] == 0.0)
    # ★2026-08-11 조용한 버그 회귀검사: 시험축 한계가 전 채널에 적용되면 여기서 깨진다
    check("클램프에 목표가 잘리지 않음",
          all(box[c][0] - 1e-6 <= tgt[c] <= box[c][1] + 1e-6 for c in box),
          "전 채널 목표가 채널각 상자 안")
    hw.limp()


def t_scalar_gain():
    """스칼라 게인(하위호환) 경로도 살아 있는지."""
    print("\n[2] goto_all — 스칼라 게인(하위호환)")
    spec = _load_spec()
    n = int(spec["shm"]["n_channel"])
    home, _ = _home_targets()                      # ★HOME 채널각에서 출발(실기와 동일)
    fake = FakeLib(n, q_init=home, dt=1 / 500.)
    hw = _make_hw(fake, spec, hold=[0, 1])
    hw.arm(3, 30.0, 2.0)
    tgt = np.array(home, float); tgt[3] += 5.0
    hw.goto_all(tgt, kp=40.0, kd=2.0, speed_dps=60.0)
    check("스칼라 경로 완주 + 도달", abs(hw.read(3)[0] - tgt[3]) < 1.0,
          f"q3={hw.read(3)[0]:.3f}° (목표 {tgt[3]:.2f})")
    hw.limp()


def t_torque_loop():
    """τ_ff 루프 — 곧 실기에서 돌릴 경로."""
    print("\n[3] run_torque — τ_ff 램프 + 드리프트 워치독")
    spec = _load_spec()
    n = int(spec["shm"]["n_channel"])
    home, _ = _home_targets()
    fake = FakeLib(n, q_init=home, dt=1 / 500.)
    hw = _make_hw(fake, spec, hold=[0, 1, 2, 4, 5, 6, 7], align=False)
    hw.arm(3, 0.0, 0.0)
    # 실기에서 돌릴 것과 같은 램프: 0.25 Nm/s. 스텁 마찰 0.65 Nm 이므로 t≈2.6s 에서
    # 브레이크어웨이가 나야 한다 — 실기 판정 로직이 성립하는지까지 여기서 본다.
    smp = hw.run_torque(3, lambda t: 0.25 * t, duration_s=3.2, tau_max=1.4,
                        drift_max_deg=25.0)
    check("샘플 수집", len(smp) > 100, f"{len(smp)} 샘플")
    check("τ_ff 가 실제로 인가됨", abs(smp[-1].tau) > 0.05,
          f"tau_last={smp[-1].tau:+.3f} Nm")
    q0 = smp[0].q_deg
    brk = next((x for x in smp if abs(x.q_deg - q0) > 0.5), None)
    check("브레이크어웨이 검출", brk is not None and abs(0.25 * (brk.t - smp[0].t) - 0.65) < 0.2,
          f"τ_brk≈{0.25 * (brk.t - smp[0].t):.3f} Nm (스텁 마찰 0.65)" if brk else "미검출")

    # 워치독: 복원력이 0 이라 큰 τ 는 반드시 흘러간다 → 잡아야 한다
    import hwio
    fake2 = FakeLib(n, q_init=home, dt=1 / 500.)
    hw2 = _make_hw(fake2, spec, hold=[0, 1], align=False)
    hw2.arm(3, 0.0, 0.0)
    tripped = False
    try:
        hw2.run_torque(3, lambda t: 3.0, duration_s=5.0, tau_max=3.0, drift_max_deg=2.0)
    except hwio.SafetyAbort:
        tripped = True
    check("드리프트 워치독이 중단시킴", tripped)


def t_limp_and_signal():
    """종료 경로 — limp 가 실제로 kp=kd=0 을 쓰는지."""
    print("\n[4] limp")
    spec = _load_spec()
    n = int(spec["shm"]["n_channel"])
    home, _ = _home_targets()
    fake = FakeLib(n, q_init=home, dt=1 / 500.)
    hw = _make_hw(fake, spec, hold=[0, 1])
    hw.arm(3, 30.0, 2.0)
    fake.max_kp_seen[:] = 0.0
    ok = hw.limp()
    check("limp 성공 횟수 > 0", ok > 0, f"{ok}회")
    check("limp 중 게인 0", float(fake.max_kp_seen.max()) == 0.0)
    check("enable 해제", fake.enabled == 0)


if __name__ == "__main__":
    print("=" * 66)
    print("hwio 오프라인 스모크 — 스텁 SHM 위에서 실행경로를 끝까지 밟는다")
    print("=" * 66)
    for fn in (t_goto_all_home, t_scalar_gain, t_torque_loop, t_limp_and_signal):
        try:
            fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            FAILED.append(f"{fn.__name__}: {type(e).__name__}: {e}")
    print("\n" + "=" * 66)
    if FAILED:
        print(f"실패 {len(FAILED)}건:")
        for f in FAILED:
            print("  -", f)
        sys.exit(1)
    print("전부 통과 — 코드경로는 끝까지 돈다(물리 정합성은 이 테스트의 범위가 아니다)")
