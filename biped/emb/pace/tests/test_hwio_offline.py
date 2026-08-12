#!/usr/bin/env python3
"""test_hwio_offline.py — 하드웨어 없이 hwio 경로를 끝까지 돌리는 스모크 테스트.

★왜 만들었나 (2026-08-11)
  goto_all(HOME 정렬)이 실기에서 **두 번 연속** 죽었다. 둘 다 로봇을 세워 두고
  사람이 실행해야만 드러나는 형태였다:
      1회차  KeyError: 8              — 축별 게인 dict 에 허리(8,9) 키가 없다
      2회차  AttributeError: 'rate'   — self.rate 는 없다(주기는 self.dt)
      3회차  KeyError: 'kp'           — spec.gains.kp 는 이미 없어진 키였다
  둘 다 **한 줄짜리 이름 오류**이고, 둘 다 `python3 -c "import"` 나 구문검사로는
  안 잡힌다. 실행경로를 실제로 밟아야만 나온다.
  ⇒ 공유 라이브러리를 스텁으로 갈아끼워 arm→홈복귀→토크루프를 전부 돌린다.
  홈복귀는 GUI 와 같은 control/home.py:HomeTrajectory 를 쓰므로 그 배선까지 여기서 본다.

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
        # ★2026-08-12: 종전엔 self.ticks 패리티를 썼는데 _integrate 와 _bridge_read 가
        #   **둘 다** ticks 를 올린다 → 한 사이클에 +2 라 패리티가 **고정**된다.
        #   디더가 안 걸려서, 축이 완전히 멈춰 있는 구간(바이어스 정착 0.25s)에서
        #   stale 152ms 오탐이 났다. 실기는 σ_q 0.0044° · σ_dq 2.11dps 라 매 틱 변한다 —
        #   **스텁이 실물보다 조용하면 없는 고장을 만들어낸다.**
        self._dither = getattr(self, "_dither", 0) + 1
        d = 1e-4 if (self._dither % 2) else -1e-4
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
    """실기에서 죽었던 그 경로. 늘어진 자세 → HOME (GUI 와 동일 궤적)."""
    print("\n[1] goto_home — 늘어진 자세에서 HOME 복귀 (실기 재현, HomeTrajectory)")
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
    from homing import goto_home, make_homer
    import yaml as _y
    cfg = _y.safe_load(open(os.path.join(os.path.dirname(PACE), "config", "biped_emb.yaml")))
    box = at._mech_limit_box()      # 실기와 동일: 복귀 구간은 **기구한계** 상자
    # (spec 상자를 쓰면 늘어진 ch3=+17.55° 가 상한 12.75° 로 잘려 시작부터 어긋난다 —
    #  복귀용 상자가 왜 기구한계여야 하는지가 여기서 드러난다.)
    hw.arm(3, 30.0, 2.0)
    homer = make_homer(jm, cfg, hw.dt)
    # ★**실기와 같은 15dps** 를 쓴다(config home.max_speed_dps).
    #   60dps 로 올렸더니 스텁이 못 따라가 추종오차 13.9°>12° 로 트립했고, 30dps 로
    #   낮춰도 12.0~12.2° 로 **경계에서 흔들렸다**(스텁 지연이 벽시계 타이밍에 의존).
    #   테스트를 빠르게 하려고 실기와 다른 조건을 쓰면 이렇게 가짜 실패가 난다 —
    #   ⚠오늘만 두 번째다(합성검증 dur=0.6s 로 2단회귀를 통과시킨 것과 같은 실수).
    T = goto_home(hw, jm, homer, cfg, q_box=box)      # speed 미지정 = config 값(15dps)
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
    """상태 발행 — 시험 중 뷰어가 자세를 볼 수 있어야 한다."""
    print("\n[2] publish_fn — 시험 중 뷰어 상태 발행")
    import json, tempfile
    from state_pub import publish_state
    spec = _load_spec()
    n = int(spec["shm"]["n_channel"])
    home, jm = _home_targets()
    fake = FakeLib(n, q_init=home, dt=1 / 500.)
    hw = _make_hw(fake, spec, hold=[0, 1])
    path = os.path.join(tempfile.gettempdir(), "biped_state_pacetest.json")
    if os.path.exists(path):
        os.remove(path)
    hw.publish_fn = lambda q_ch, rpy, on: publish_state(
        "pace:HL_foot", jm.ch_to_q_joint(np.asarray(q_ch, float)),
        np.asarray(rpy, float), 1.0 / hw.dt, on, "pace", path=path)
    hw.pub_period = 0.0                      # 테스트에서는 매 read 마다
    hw.arm(3, 30.0, 2.0)
    for _ in range(5):
        hw.read(3)
    ok = os.path.exists(path)
    check("상태파일 생성", ok, path)
    if ok:
        d = json.load(open(path))
        check("스키마가 뷰어와 같음",
              all(k in d for k in ("mode", "q_leg_deg", "rpy_deg", "loop_hz",
                                   "motors_on", "backend")),
              f"mode={d.get('mode')} backend={d.get('backend')}")
        # ★모델각으로 실려야 한다 — 채널각을 그대로 흘리면 뷰어가 엉뚱한 자세를 그린다.
        check("모델각으로 발행(HOME 이면 0)",
              max(abs(x) for x in d["q_leg_deg"]) < 0.01,
              f"q_leg_deg={d['q_leg_deg']}")
        check("motors_on 반영", d["motors_on"] is True)
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


def t_friction_full():
    """★마찰 시험 **전 경로**를 스텁 위에서 끝까지 밟는다 (2026-08-12 추가).

    왜 필요했나 — `_breakaway` 가 `_ff` 라는 **클로저 이름을 모듈 수준에서** 참조하고
    있었다. `_ff` 는 measure_actuator_friction 안에 있으니 안 보인다 → NameError.
    3커밋 동안 안 걸린 이유는 오프라인 테스트가 (A)breakaway·(C)사인 을 **한 번도
    안 밟았기** 때문이다. 같은 부류가 (C) 에도 하나 더 있었다(goto 의 tau_ff_fn=ff).
    ⇒ 이제 A·B·C 를 다 밟는다. 물리값은 안 본다 — **NameError·KeyError·시그니처
      불일치**를 실기 전에 잡는 게 목적이다.

    ⚠spec 을 **줄여서** 돌린다. 실기값 그대로면 breakaway 만 80초다. 줄여도 밟는
      코드줄은 같다. 줄인 값은 아래 SHRINK 한 곳에만 있다.
    """
    print("\n[3b] measure_actuator_friction — A·B·C 전 경로 (실기 실행 전 관문)")
    import copy, tempfile
    spec = copy.deepcopy(_load_spec())
    n = int(spec["shm"]["n_channel"])
    home, _ = _home_targets()

    fr = spec["friction"]                                    # ── SHRINK ──
    fr.pop("by_ch", None)
    # 파단이 **일어나도록** 잡는다: 스텁 마찰 0.65Nm ÷ kp 30Nm/rad = 1.24° 가 필요하다.
    # 램프도 느려야 한다 — q_ref 는 t>0.3s 에 래치되므로 그 전에 풀리면 검출을 못 한다
    # (1.24° ÷ 2dps = 0.62s > 0.3s ✓). 종전 0.6°/6dps 는 0.31Nm 이라 영영 안 풀렸다.
    fr["breakaway"].update(max_push_deg=4.0, ramp_dps=2.0, trials=1)
    fr["sweep"].update(stroke_deg=4.0, speeds_dps=[5, 10, 20],
                       accel_skip_s=0.05, min_dwell_samples=10)
    fr["sine"].update(amplitude_deg=2.0, frequency_hz=1.0, cycles=1.0)

    fake = FakeLib(n, q_init=home, dt=1 / 500.)
    hw = _make_hw(fake, spec, hold=[], align=False)          # solo 와 동일 조건
    # ★**실기와 같은 조건**으로 건다 (2026-08-12). grav_fn/tau_ff_fn 을 안 꽂으면
    #   measure_actuator_friction 이 표 경로로 빠져 실기와 다른 코드를 밟는다.
    #   실제로 그래서 `_base → hw.grav_fn → _ff → _base` **무한 재귀**를 못 잡았다.
    #   실기는 actuator_test 가 홈복귀 전에 둘 다 걸어 놓고 들어온다.
    _gt0 = (spec["torque_mode"].get("tau_grav_table") or {})
    hw.grav_fn = (lambda c, q, _t=_gt0:
                  float(np.interp(q, _t[c]["q_ch"], _t[c]["tau"])) if c in _t else 0.0)
    hw.tau_ff_fn = lambda c, q: float(hw.grav_fn(3, q)) if c == 3 else 0.0
    j = {"ch": 3, "name": "HL_foot", "gear": 7.0, "q_min": -27.84, "q_max": 48.0}

    import tests.act_measure_friction as amf  # noqa: F401
    from act_measure_friction import measure_actuator_friction, swing_str
    check("swing_str 환산", swing_str(40.0, 80.0) == "±20°·1Hz", swing_str(40.0, 80.0))
    with tempfile.TemporaryDirectory() as td:
        html, res = measure_actuator_friction(hw, spec, j, td, log=lambda *a: None)
    # ⚠물리값은 **보지 않는다**. 스텁의 마찰모형은 실기와 다르므로 jfric/jdamp 가
    #   실기값과 맞을 이유가 없다(여기서 jdamp 는 14 로 나온다 — 스텁 특성이다).
    #   이 시험이 지키는 건 "끝까지 돈다" 뿐이다.
    check("A·B·C 전부 완주", isinstance(html, str) and len(html) > 200)
    for k in ("tau_static", "jfric", "jdamp"):
        check(f"결과 키 {k}", k in res, f"{res.get(k)}")


def t_measure_gravity():
    """★중력 실측 — 마찰이 상쇄되고 중력만 남는가 (2026-08-12 추가).

    이게 없으면 --solo 가 성립하지 않는다. 중력표는 "다른 관절 = neutral" 가정으로
    만든 것인데 solo 는 하위 관절이 늘어져 있어 HL_thigh 에서 1.27Nm 틀렸고, 그게
    스톨 감지의 가짜 초과토크가 되어 홈복귀에서 시험을 세 번 죽였다.
    스텁은 중력을 알고 있으므로 **정답을 아는 상태로** 검증할 수 있다.
    """
    print("\n[3c] measure_gravity — 위·아래 접근 평균으로 마찰을 상쇄한다")
    spec = _load_spec()
    n = int(spec["shm"]["n_channel"])
    home, _ = _home_targets()
    for g_true in (-1.5, +0.8, 0.0):
        tg = np.zeros(n)
        tg[3] = g_true                       # 스텁이 거는 물리 중력(부하)
        fake = FakeLib(n, q_init=home, tau_grav=tg, dt=1 / 500.)
        hw = _make_hw(fake, spec, hold=[], align=False)
        # ★**실기와 같은 조건**으로 건다 — grav_fn 을 안 꽂으면 스톨 감지가 통째로
        #   꺼져서 시험이 통과해 버린다. 처음에 그렇게 짰다가 실기에서 터질 뻔했다.
        #   일부러 **틀린 표**(진짜의 절반)를 꽂는다. 실기의 thigh 상황 그대로다.
        hw.grav_fn = lambda c, q, _g=g_true: (-_g * 0.5 if c == 3 else 0.0)
        hw.arm(3, 30.0, 2.0)
        g = hw.measure_gravity(3, 30.0, 2.0, delta_deg=4.0, settle_s=0.5)
        hw.limp()
        # 모터가 들어야 하는 토크 = −(물리 부하). 스텁 마찰 0.65 는 상쇄돼야 한다.
        check(f"중력 {-g_true:+.2f}Nm 복원 (마찰 0.65 상쇄 · 표는 절반만 맞음)",
              abs(g - (-g_true)) < 0.15, f"측정 {g:+.3f}")


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


def t_hold_no_ratchet():
    """홀드 목표가 시행마다 **재래치되지 않는지** — 처짐 래칫 회귀시험.

    ★2026-08-12 실기: 지그 없이 --tests torque 를 돌리면 thigh 가 눈에 띄게 주저앉았다.
      원인은 arm() 이 홀드 목표를 "지금 측정각" 으로 잡은 것이다:
        arm → 오차 0 → 중력이 kp·err 균형까지 끌어내림 → **다음 arm 이 그 자리를 목표로**
      토크프로브는 시행마다 arm 하므로(4회) 처짐이 선형 누적된다. 실측 1회 3.0° →
      4시행 12.0° = err_max 12.0°, 즉 'ch1 홀드축이 밀렸다' 트립 지점과 정확히 같다.
      지그를 물리면 기구가 잡아서 **가려질 뿐** 버그는 남는다.
    """
    print("\n[5] 홀드 목표 래칫")
    spec = _load_spec()
    n = int(spec["shm"]["n_channel"])
    g = np.zeros(n)
    g[1], g[5] = -2.06, -2.01                 # MJCF 에서 뽑은 thigh 중력토크
    fake = FakeLib(n, tau_grav=g, dt=1 / 500.)
    hw = _make_hw(fake, spec, hold=[0, 1, 2, 4, 5, 6, 7])
    tgt = hw.latch_hold(np.zeros(n))
    pos = []
    for _ in range(6):                         # 토크프로브보다 넉넉히
        hw.arm(3, 0.0, 0.0)
        for _ in range(600):
            hw._raw_write(3, float(fake.q[3]), 0.0, 0.0)
        hw.release_test_axis(3)
        pos.append(float(fake.q[1]))
    check("홀드 목표가 arm 마다 안 바뀐다",
          float(np.max(np.abs(hw._hold_target - tgt))) == 0.0)
    # 처짐은 한 번 일어나고 **수렴**해야 한다. 누적이면 뒤로 갈수록 계속 커진다.
    late = abs(pos[-1] - pos[-2])
    check("처짐이 누적되지 않고 수렴", late < 0.05, f"마지막 증분 {late:.3f}°")
    check("총 처짐이 err_max 안", abs(pos[-1]) < spec["safety"]["err_max_deg"],
          f"{abs(pos[-1]):.2f}° < {spec['safety']['err_max_deg']}°")
    drift = hw.hold_drift()
    check("hold_drift 가 처짐을 보고한다", abs(drift.get(1, 0.0)) > 0.5,
          f"ch1 {drift.get(1, 0.0):+.2f}°")


if __name__ == "__main__":
    print("=" * 66)
    print("hwio 오프라인 스모크 — 스텁 SHM 위에서 실행경로를 끝까지 밟는다")
    print("=" * 66)
    for fn in (t_goto_all_home, t_scalar_gain, t_torque_loop, t_measure_gravity,
               t_friction_full, t_limp_and_signal, t_hold_no_ratchet):
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
