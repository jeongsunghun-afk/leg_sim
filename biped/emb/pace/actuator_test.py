#!/usr/bin/env python3
"""actuator_test.py — 액추에이터 자동시험 하니스 (마찰 + PACE 식별) → HTML 리포트.

레퍼런스 motorcortex-python-tools/automatic_testing_examples/actuator_test.py 의 구조를 계승:
  연결 → engage → 테스트들 실행(각자 HTML 조각 반환) → disengage → 템플릿 렌더 → 리포트.
차이: motorcortex WebSocket 대신 SHM(libbipedshm), PDF(weasyprint) 대신 HTML.

사용법
  # 하드웨어 없이 추정기 수학만 검증(모터 무동작·안전)
  python3 actuator_test.py --selftest

  # 실기 — Emb 기동 후 5초 이상 지난 뒤 실행
  python3 actuator_test.py --ch 0 --tests friction
  python3 actuator_test.py --ch 0 --tests friction,pace
  python3 actuator_test.py --all                  # spec 의 installed_channels 전부

★시퀀스 (2026-08-11)
      arm → **HOME 복귀** → (지그 설치 대기) → 한계 조임 → 드라이버 생존확인 → 시험
  · HOME 복귀는 **지그 유무와 무관하게 항상** 돈다. 지그가 물려 있으면 편차가 작아
    즉시 끝난다. 제어기를 끄면 다리가 늘어지고, 그 자세는 두 발이 22mm 파고든
    **충돌 상태**라 어떤 경우에도 여기서 벗어나야 한다.
  · 궤적은 **GUI 홈복귀와 같은 구현**을 쓴다(control/home.py:HomeTrajectory, pace/homing.py).
    모델각 공간에서 보간하고 v·a 한계를 둘 다 지킨다. 하니스가 따로 짜지 않는다.
  · 시작 시 **영점 검증**을 보고한다 — HOME 자세의 모델각은 정의상 전부 0 이어야 한다.
  · 시험 중 **뷰어 상태를 계속 발행**한다(interface/state_pub.py, biped_emb 와 같은 파일).
    writer 는 하나여야 해서 biped_emb 를 끄지만, 화면은 살아 있다.

      python3 actuator_test.py --ch 3 --tests torque --jig

⚠ 실행 전 확인
  1. Emb 기동 후 5초 경과 (halGait 초기화 게이트 = 100+4500 tick @1kHz).
  2. **모터 명령 writer 는 한 번에 하나만** — app/biped_emb.py, RobotTestGait, mot_test 종료.
  3. 로봇 거치(크레인/스탠드). 시험 중 관절이 스펙 한계 안에서 왕복한다.
  4. 종료(정상·예외·Ctrl-C·SIGTERM) 시 항상 limp 로 빠진다.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

from hwio import DEG, Hardware, Limits, SafetyAbort  # noqa: E402
from homing import goto_home, make_homer            # noqa: E402  GUI 와 동일 궤적
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "interface"))
from state_pub import publish_state                 # noqa: E402  뷰어와 동일 스키마


def load_spec(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def todo_scan(spec: dict) -> list[str]:
    """TODO 로 남은 스펙 항목 수집 — 없는 값을 추측하지 않기 위해 명시적으로 보고."""
    out = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str) and node.strip() == "TODO":
            out.append(path)
    walk(spec)
    return out


# ── 셀프테스트: 합성 데이터로 추정기 검증 (하드웨어 불필요) ──────────────────
def selftest() -> int:
    """알려진 파라미터로 신호를 만들어 추정기가 되찾는지 확인한다.
    실기 데이터를 믿기 전에 '추정기 자체가 맞는가' 를 먼저 분리 검증하는 목적."""
    from scipy.signal import savgol_filter
    print("=== 셀프테스트: 합성 데이터 → 추정기 ===\n")
    ok = True

    # --- 1. PACE 회귀 ------------------------------------------------------
    I_true, b_true, tc_true = 0.0123, 0.184, 0.512
    A_true, B_true, c_true = 0.31, -0.07, 0.021
    dt, T = 1 / 200, 30.0
    t = np.arange(0, T, dt)
    amp, f0, f1 = 6 * DEG, 0.2, 4.0
    k = (f1 - f0) / T
    ph = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    w = 2 * np.pi * (f0 + k * t)
    q = amp * np.sin(ph)
    dq = amp * w * np.cos(ph)
    ddq = -amp * w * w * np.sin(ph)
    eps = 0.02
    tau = (I_true * ddq + b_true * dq + tc_true * np.tanh(dq / eps)
           + A_true * np.sin(q) + B_true * np.cos(q) + c_true)
    rng = np.random.default_rng(0)
    tau_n = tau + rng.normal(0, 0.01, t.size)              # 토크 노이즈
    dq_n = dq + rng.normal(0, 15 * DEG, t.size)            # ★실측 수준 속도노이즈

    dq_f = savgol_filter(dq_n, 31, 3)
    ddq_f = savgol_filter(dq_n, 31, 3, deriv=1, delta=dt)
    W = np.column_stack([ddq_f, dq_f, np.tanh(dq_f / eps),
                         np.sin(q), np.cos(q), np.ones_like(q)])
    th, *_ = np.linalg.lstsq(W, tau_n, rcond=None)
    for nm, got, exp, tol in (("I_total", th[0], I_true, 0.25),
                              ("JDAMP", th[1], b_true, 0.25),
                              ("JFRIC", th[2], tc_true, 0.15)):
        err = abs(got - exp) / abs(exp)
        good = err < tol
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {nm:8s} 참값 {exp:8.4f} → 추정 {got:8.4f} "
              f"(상대오차 {err*100:5.1f}%, 허용 {tol*100:.0f}%)")

    # --- 2. 마찰 양방향 상쇄 ----------------------------------------------
    print()
    grav, bias = 0.44, 0.03                                # 마찰과 섞이는 성분
    v = np.array([2, 5, 10, 20, 40, 70]) * DEG
    f_true = tc_true + b_true * v
    tau_p = +f_true + grav + bias
    tau_n2 = -f_true + grav + bias
    f_rec = (tau_p - tau_n2) / 2
    g_rec = (tau_p + tau_n2) / 2
    M = np.column_stack([np.ones_like(v), v])
    co, *_ = np.linalg.lstsq(M, f_rec, rcond=None)
    for nm, got, exp, tol in (("JFRIC", co[0], tc_true, 0.02),
                              ("JDAMP", co[1], b_true, 0.02),
                              ("중력+bias", float(np.mean(g_rec)), grav + bias, 0.02)):
        err = abs(got - exp) / abs(exp)
        good = err < tol
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {nm:9s} 참값 {exp:8.4f} → 복원 {got:8.4f} "
              f"(상대오차 {err*100:5.2f}%)")

    # --- 3. 상쇄를 안 하면 얼마나 틀리는가 (이 설계의 근거) ---------------
    naive, *_ = np.linalg.lstsq(M, tau_p, rcond=None)      # +방향만 쓴 경우
    print(f"\n  [참고] 양방향 상쇄 없이 +방향만 회귀하면 "
          f"JFRIC {naive[0]:.4f} (참값 {tc_true:.4f}, "
          f"{(naive[0]/tc_true-1)*100:+.0f}% 오차) — 중력·바이어스가 마찰로 둔갑한다")

    # --- 4. FRF 추정기 (공진·감쇠·강성) ---------------------------------
    print()
    sys.path.insert(0, os.path.join(HERE, "tests"))
    from act_measure_frf import selftest as frf_selftest
    ok &= frf_selftest()

    print(f"\n=== 셀프테스트 {'통과' if ok else '실패'} ===")
    return 0 if ok else 1


# ── 실기 실행 ────────────────────────────────────────────────────────────────
def preflight(spec: dict, log=print) -> None:
    import subprocess
    r = subprocess.run(["pgrep", "-x", "RobotEmbedded"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit("✗ Emb(RobotEmbedded) 미기동. 기동 후 5초 기다린 뒤 재실행할 것.")
    r = subprocess.run(["pgrep", "-f", "biped_emb.py|RobotTestGait|mot_test"],
                       capture_output=True)
    if r.returncode == 0:
        raise SystemExit("✗ 다른 모터 명령 writer 가 실행 중이다. 종료 후 재실행할 것.\n"
                         f"  {r.stdout.decode().strip()}")
    todos = todo_scan(spec)
    if todos:
        log("⚠ spec.yaml 미확정(TODO) 항목 — 관련 환산은 '미확정' 으로 보고된다:")
        for t in todos:
            log(f"    · {t}")
        log("")


def _gain(v):
    """홀드 게인 정규화 — 스칼라면 float, dict 면 **키를 int 로** 맞춘 dict.

    ★yaml 은 `{0: 100.0, ...}` 을 int 키로 읽지만, 문자열 키로 쓰인 설정도 있을 수 있어
      여기서 한 번 정규화한다. Hardware._hold_gain_of 는 채널 int 로 조회한다.
    """
    if isinstance(v, dict):
        return {int(k): float(x) for k, x in v.items()}
    return float(v)


_JIG_DEV: dict = {}          # 마지막 precheck 의 (편차, 기준) — 오류 메시지용


def _jig_precheck(hw, jm, tgt_ch, spec: dict, log=print, tol_override=None) -> bool:
    """지금 자세가 이미 HOME 인가? 겸사겸사 **영점 검증**이 된다.

    ★지그 자세 = HOME = 영점이 같은 자세다(사용자 확인, 2026-08-11).
      그래서 지그를 **먼저** 물려 두면 자세가 이미 HOME 이고, 시험 전에 모터를
      한 번도 움직일 필요가 없다 — 매단 로봇에서 이게 가장 안전한 순서다.
      (반대로 지그가 물린 채 goto_all 을 부르면 모터가 지그와 싸운다.)

    ★모델각으로 보고하는 이유 — 지그 자세의 모델각은 **정의상 전부 0** 이어야 한다.
      여기서 나오는 편차가 곧 영점 오차다. 시험 전에 공짜로 얻는 교정 검증이다.
    """
    tol = float(tol_override if tol_override is not None
                else spec.get("safety", {}).get("home_tol_deg", 3.0))
    q_ch = np.array([hw.read(c)[0] for c in range(hw.n)], float)
    q_j = np.asarray(jm.ch_to_q_joint(q_ch), float)
    d_ch = np.array([q_ch[c] - float(tgt_ch[c]) for c in jm.ch])
    emax = float(np.max(np.abs(d_ch)))
    at_home = emax <= tol
    _JIG_DEV.update(dev=emax, tol=tol)
    log(f"  자세 확인 — HOME 대비 최대 {emax:.2f}° (판정 기준 {tol}°)")
    log("    " + "  ".join(f"ch{c}{d_ch[i]:+.2f}" for i, c in enumerate(jm.ch)))
    # 모델각은 지그 자세에서 **정의상 전부 0** 이어야 한다. 편차가 크든 작든 항상
    # 찍는다 — 어긋났을 때야말로 이 표가 원인(어느 축의 영점인지)을 알려준다.
    log(("  ✓ 이미 HOME — **모터 이동 없음**. " if at_home else "  ⚠ ") +
        "영점 검증(모델각, 0 이어야 함):")
    log("    " + "  ".join(f"{n}{q_j[i]:+.2f}" for i, n in
                           enumerate(x["name"] for x in
                                     sorted(spec["joints"], key=lambda y: int(y["ch"])))))
    return at_home


def _jig_lower_gains(hw, spec: dict, log=print) -> None:
    """지그가 이미 물려 있는 경우 — 대기 없이 홀드게인만 낮춘다."""
    frac = float(spec.get("safety", {}).get("jig_hold_frac", 0.2))
    scale = lambda v: ({k: x * frac for k, x in v.items()}
                       if isinstance(v, dict) else v * frac)
    hw.hold_kp, hw.hold_kd = scale(hw.hold_kp), scale(hw.hold_kd)
    hw.check_hold()
    log(f"  ✓ 지그 물린 상태 확인 — 홀드게인 {frac:.0%} 로 인하(추락 방지용만 남김)")


def _jig_engage(hw, spec: dict, hold, joint, log=print) -> None:
    """HOME 정렬 뒤 **지그 설치**를 기다렸다가 홀드게인을 낮춘다.

    ★왜 지그가 모터 홀드보다 나은가 — 홀드축은 모터로 잡으면 **스프링**이다.
      고유진동수 f_n = √(kp·k²/I) 가 배포게인 기준 hip 3.19 · thigh 2.53 · calf 5.08 ·
      foot 4.49 Hz 로, 처프 대역(0.2~4 Hz) **안**에 있다. 그 위에서는 홀드축이
      '자유' 처럼 굴어 시험축 식별값을 오염시킨다. 기구 고정은 이 항 자체를 없앤다.
      부수적으로 홀드 민감도(Schur 보정 −1.4~−35.6%)도 사라진다.

    ⚠ 순서가 중요하다: **정렬 → 지그 → 게인 낮춤**.
      지그를 먼저 물리면 HOME 으로 갈 수가 없고, 게인을 먼저 낮추면 다리가 떨어진다.

    ⚠ 게인을 0 으로 만들지 않는다. 지그가 하중을 받되 모터는 **추락 방지용**으로만
      남긴다(기본 20%). 지그가 훨씬 강성이 크므로 이 정도 병렬 스프링은 식별에
      기여하지 않고, 지그가 미끄러졌을 때 다리를 잡아 준다.
      → 그 미끄러짐은 check_hold() 가 그대로 **지그 슬립 감지기**로 동작해 잡는다.
    """
    frac = float(spec.get("safety", {}).get("jig_hold_frac", 0.2))
    scale = lambda v: ({k: x * frac for k, x in v.items()}
                       if isinstance(v, dict) else v * frac)
    log("\n" + "─" * 66)
    log(f"  ★지그 모드 — 지금 자세(HOME)에서 홀드축 {hold} 를 기구적으로 고정할 것.")
    log(f"    설치가 끝나면 홀드게인을 {frac:.0%} 로 낮춘다(추락 방지용만 남김).")
    log(f"    시험축 {joint['name']}(ch{joint['ch']})는 **물리지 말 것** — 그게 측정 대상이다.")
    log("─" * 66)
    try:
        input("  지그 설치 완료 후 Enter (중단은 Ctrl+C): ")
    except (EOFError, KeyboardInterrupt):
        hw.limp()
        raise SystemExit("지그 설치 취소 — limp 함")
    hw.hold_kp, hw.hold_kd = scale(hw.hold_kp), scale(hw.hold_kd)
    hw.check_hold()          # 지그 물린 뒤에도 자세가 유지되는지 즉시 확인
    log(f"  ✓ 홀드게인 {frac:.0%} 로 인하 — 이후 check_hold 는 **지그 슬립 감지기**로 동작한다\n")


def _jointmap():
    """emb 의 JointMap 을 그대로 쓴다 — 환산식을 여기 복사하지 않는다."""
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "interface"))
    from joint_map import JointMap
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(HERE),
                                           "config", "biped_emb.yaml")))
    return JointMap(cfg), cfg


_BOX_CACHE: dict = {}


def _mech_limit_box() -> dict:
    """**기구 한계**(biped_emb.yaml joints.min_deg/max_deg, 모델각) → 채널각 상자.

    ★spec 의 q_min/q_max 와 목적이 다르다:
        spec       = 시험 중 여기 진폭의 **보수적 여유폭**(hip ±20 등)
        biped_emb  = 실제로 갈 수 있는 **기구 한계**(hip ±35 등)
      자세 정렬은 늘어진 자세에서 출발하는데, 그 자세는 시험 여유폭 밖이다:
        늘어진 calf 는 모델각 −53/−46° 로 spec 의 ±40 을 넘는다.
      정렬에 spec 상자를 쓰면 arm 이 곧바로 '위치 한계 이탈' 로 트립한다.
      ⇒ 정렬 구간은 기구 한계, 시험 구간은 spec 한계. 상자를 **두 개** 쓴다.
    """
    if "mech" in _BOX_CACHE:
        return _BOX_CACHE["mech"]
    jm, cfg = _jointmap()
    js = sorted(cfg["joints"], key=lambda x: int(x["channel"]))
    lo_j = np.array([float(x["min_deg"]) for x in js])
    hi_j = np.array([float(x["max_deg"]) for x in js])
    n = len(js)
    lo = np.full(jm.n_channel, np.inf)
    hi = np.full(jm.n_channel, -np.inf)
    for m in range(1 << n):
        qj = np.where([(m >> i) & 1 for i in range(n)], hi_j, lo_j)
        v = np.asarray(jm.q_joint_to_ch(qj), float)
        lo = np.minimum(lo, v)
        hi = np.maximum(hi, v)
    box = {int(x["channel"]): (float(lo[int(x["channel"])]), float(hi[int(x["channel"])]))
           for x in js}
    _BOX_CACHE["mech"] = box
    return box


def _ch_limit_box(spec: dict, pin_home: bool = False) -> dict:
    """spec.joints 의 q_min/q_max(=**모델각** 여유폭) → **채널각** 한계상자.

    ★왜 필요한가 — 2026-08-11 발견, **예외 없이 조용히 틀리는** 버그였다.
      이 하니스는 SHM 을 직접 읽고 쓴다. 즉 hwio 가 다루는 q 는 전부 **채널각**이다.
      그런데 spec 의 q_min/q_max 는 홈 주변 **모델각** 여유폭으로 적혀 있다.
      영점 재교정 전에는 offset≈0·k=1 이라 두 값이 사실상 같아서 구분이 안 드러났다.
      지금은 calf/foot 의 offset 이 −66.6/−83.3 이고 gear_k 도 1.5/1.2 다. 그대로 쓰면:
          ch2 목표 −66.56 → [−40, 40] 로 잘려 −40.00  (26.6° 어긋남)
          ch3 목표 −83.25 → [−40, 30] 로 잘려 −40.00  (43.2° 어긋남)
          ch6 목표 +85.69 → [−40, 40] 로 잘려 +40.00  (45.7° 어긋남)
          ch7 목표 +87.00 → [−40, 30] 로 잘려 +30.00  (57.0° 어긋남)
      HOME 정렬이 **엉뚱한 자세로 가 놓고 성공했다고 보고**한다. 충돌 회피가 목적인
      정렬이 그 자체로 위험해지는 형태다.

    ⚠커플링 때문에 foot 의 채널각은 calf **모델각**에도 의존한다 → 상자의 상은 상자가
      아니다. 여기서는 꼭짓점 2^8 전수의 상을 감싸는 최소상자를 쓴다. 즉 **바깥쪽으로
      보수적**이다 — 정당한 명령을 잘못 자르지 않는 쪽. 폭주 방어는 q 상자가 아니라
      토크·속도·추종오차·드리프트 트립이 맡는다.

    pin_home=True — 다른 축을 **HOME 에 고정**한 채 그 축만 범위를 훑는다.
      이게 **시험 중 실제 조건**이다(홀드축은 잡혀 있다). 커플링 여유가 안 붙어
      상자가 훨씬 좁고, 그만큼 위치한계 트립이 실제로 일을 한다.
      예: HL_foot 은 합집합이면 [−167.25, 12.75](폭 180°)인데 pin 하면
      [−119.25, −35.25](폭 84° = 모델각 70° × gear_k 1.2) — 합집합 쪽은 폭이 커플링
      여유(=calf 120°)에 먹혀 위치보호가 사실상 무력해진다.
      다축 이동(goto_all)은 소스축이 **움직이므로** pin 을 쓰면 안 된다.
    """
    key = (id(spec), bool(pin_home))
    if key in _BOX_CACHE:
        return _BOX_CACHE[key]
    jm, cfg = _jointmap()
    js = sorted(spec["joints"], key=lambda x: int(x["ch"]))
    lo_j = np.array([float(x["q_min"]) for x in js])
    hi_j = np.array([float(x["q_max"]) for x in js])
    n = len(js)
    lo = np.full(jm.n_channel, np.inf)
    hi = np.full(jm.n_channel, -np.inf)
    if pin_home:
        home = np.array([float(x) for x in cfg["home"]["q_deg"]])
        for i in range(n):                        # 축 하나만 훑고 나머지는 HOME
            for e in (lo_j[i], hi_j[i]):
                qj = home.copy(); qj[i] = e
                v = np.asarray(jm.q_joint_to_ch(qj), float)
                lo[js[i]["ch"]] = min(lo[js[i]["ch"]], v[js[i]["ch"]])
                hi[js[i]["ch"]] = max(hi[js[i]["ch"]], v[js[i]["ch"]])
    else:
        for m in range(1 << n):                   # 꼭짓점 전수
            qj = np.where([(m >> i) & 1 for i in range(n)], hi_j, lo_j)
            v = np.asarray(jm.q_joint_to_ch(qj), float)
            lo = np.minimum(lo, v)
            hi = np.maximum(hi, v)
    box = {int(js[i]["ch"]): (float(lo[js[i]["ch"]]), float(hi[js[i]["ch"]]))
           for i in range(n)}
    _BOX_CACHE[key] = box
    return box


def main() -> int:
    ap = argparse.ArgumentParser(description="액추에이터 마찰/PACE 자동시험")
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--ch", type=int, action="append", help="시험할 SHM 채널(반복 가능)")
    ap.add_argument("--all", action="store_true", help="spec 의 installed_channels 전부")
    ap.add_argument("--no-home", action="store_true",
                    help="시험 전 HOME 정렬을 생략한다(권장하지 않음 — 아래 사유)")
    ap.add_argument("--home-speed", type=float, default=None,
                    help="HOME 복귀 속도[deg/s] 상한. 미지정이면 biped_emb.yaml 의 "
                         "home.max_speed_dps 를 그대로 쓴다(GUI 와 동일).")
    ap.add_argument("--home-tol", type=float, default=None,
                    help="HOME 판정 허용편차[deg] (기본 spec.safety.home_tol_deg=3.0). "
                         "--jig 에서 이 값을 넘으면 이동하지 않고 중단한다.")
    ap.add_argument("--tests", default="friction",
                help="friction,torque,backlash,frf,latency,pace 중 콤마구분")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--selftest", action="store_true", help="하드웨어 없이 추정기만 검증")
    ap.add_argument("--jig", action="store_true",
                    help="지그로 홀드축을 **기구적으로** 고정하고 측정축만 푼다(권장). "
                         "HOME 정렬 후 지그 설치를 기다렸다가 홀드게인을 낮춘다.")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    spec = load_spec(a.spec)
    chans = list(spec["meta"]["installed_channels"]) if a.all else (a.ch or [])
    if not chans:
        raise SystemExit("--ch 또는 --all 을 지정할 것 (--selftest 는 하드웨어 불필요)")
    tests = [t.strip() for t in a.tests.split(",") if t.strip()]
    # ★이름이 틀리면 **조용히 아무것도 안 돈다** — 아래 디스패치가 전부 `in tests` 정확일치라
    #   오타 하나면 HOME 정렬만 하고 끝난다(2026-08-11: `--tests torque_mode` 로 실제 발생.
    #   맞는 이름은 `torque`). 정렬은 로봇을 움직이므로 "안 돌았다" 를 알아채기도 어렵다.
    KNOWN = ("friction", "torque", "backlash", "frf", "latency", "pace")
    bad = [t for t in tests if t not in KNOWN]
    if bad:
        import difflib
        hint = "\n".join(
            f"    {t!r} → {difflib.get_close_matches(t, KNOWN, 1, 0.3) or list(KNOWN)}"
            for t in bad)
        raise SystemExit(f"✗ 모르는 시험 이름: {bad}\n  가능한 값: {', '.join(KNOWN)}\n{hint}")

    plotdir = os.path.join(a.out, "plots")
    os.makedirs(plotdir, exist_ok=True)
    preflight(spec)

    from act_identify_pace import identify_pace
    from act_measure_friction import measure_actuator_friction
    from act_measure_latency import measure_latency_and_backlash
    from act_probe_torque_mode import probe_torque_mode
    from act_measure_backlash import measure_backlash
    from act_measure_frf import measure_frf

    jmap = {int(j["ch"]): j for j in spec["joints"]}
    sf, g = spec["safety"], spec["gains"]
    fragments, summary = [], []

    installed = set(spec["meta"]["installed_channels"])
    for ch in chans:
        if ch not in jmap:
            raise SystemExit(f"ch{ch} 가 spec.joints 에 없다")
        # ★미장착 채널을 arm 하면 물리적으로 없는 모터에 명령하게 된다.
        if ch not in installed:
            raise SystemExit(
                f"✗ ch{ch}({jmap[ch]['name']}) 는 meta.installed_channels{sorted(installed)} 에 "
                f"없다 — 미장착 축이다. 장착 후 spec.yaml 을 갱신할 것.")
        j = jmap[ch]
        # ★모델각 한계 → 채널각 한계. 두 상자를 쓴다:
        #   box      — 소스축이 움직이는 다축 이동(goto_all)용. 합집합이라 넉넉하다.
        #   box_pin  — 시험축용. 홀드축이 HOME 에 잡혀 있는 **실제 시험 조건**이라 좁다.
        box = _mech_limit_box()                   # 정렬 구간 — 늘어진 자세를 포함해야 한다
        box_pin = _ch_limit_box(spec, pin_home=True)
        lo, hi = box_pin[ch]
        _mk = lambda a, b: Limits(
            q_min=a, q_max=b,
            tau_trip=float(sf["tau_trip_nm"]), tau_trip_ms=float(sf["tau_trip_ms"]),
            vel_trip=float(sf["vel_trip_dps"]), err_max=float(sf["err_max_deg"]),
            stale_ms=float(sf["stale_ms"]),
            kp_max=float(g["kp_max"]), kd_max=float(g["kd_max"]))
        lim_align = _mk(*box[ch])                 # 정렬용(기구 한계)
        lim_test = _mk(lo, hi)                    # 시험용(spec 여유폭, 홀드축 HOME 고정 기준)
        lim = lim_align                           # ★정렬을 먼저 하므로 느슨한 쪽으로 시작
        print(f"\n{'='*70}\n{j['name']} (ch{ch})  한계 모델각∈[{j['q_min']},{j['q_max']}] "
              f"→ **채널각**∈[{lo:.2f},{hi:.2f}]deg  τ_trip={sf['tau_trip_nm']}Nm\n{'='*70}")

        # ★시험축 외 홀드 대상 = 실장된 채널 − 시험축. 다리 조립 후 필수(spec.safety 주석 참조).
        hold = sorted(installed - {ch}) if bool(sf.get("hold_others", False)) else []
        # ★hold 가 비어도 정의해 둔다 — HOME 정렬이 이 값을 쓴다(예전엔 if 안에 있어서
        #   hold_others=false 면 NameError 였다).
        _kp, _kd = _gain(sf.get("hold_kp", 40.0)), _gain(sf.get("hold_kd", 2.0))
        _kp_ch = _kp[ch] if isinstance(_kp, dict) else _kp     # 시험축의 배포게인
        _kd_ch = _kd[ch] if isinstance(_kd, dict) else _kd
        if hold:
            _fmt = (lambda g: "축별 " + " ".join(f"ch{c}:{g[c]:g}" for c in sorted(g))
                    if isinstance(g, dict) else f"{g:g}")
            print(f"  [{j['name']}] 홀드축 {hold} 를 측정위치에 고정\n"
                  f"            kp = {_fmt(_kp)}\n"
                  f"            kd = {_fmt(_kd)}\n"
                  f"            — I_link 강체가정 성립 + 하위관절 붕괴 방지")
        with Hardware(spec["shm"]["lib"], spec["shm"]["n_channel"], spec["shm"]["rate_hz"],
                      lim, int(spec["shm"]["recv_wait_ms"]),
                      float(g["enable_ramp_s"]),
                      hold_channels=hold,
                      # ★dict(축별) 도 스칼라도 그대로 넘긴다 — Hardware 가 둘 다 받는다.
                      #   여기서 float() 로 감싸면 축별 게인이 TypeError 로 죽는다(2026-08-11).
                      hold_kp=_gain(sf.get("hold_kp", 40.0)),
                      hold_kd=_gain(sf.get("hold_kd", 2.0))) as hw:
            try:
                # ── ★HOME 정렬 (2026-08-11 추가) ─────────────────────────
                #   왜 필요한가 — 실기에서 바로 걸렸다:
                #     제어기를 끄면 다리가 **무여자로 늘어진다**. 하니스는 그 늘어진 자세를
                #     그대로 잡으므로 시험이 **충돌 상태에서 시작**한다.
                #     실측: 늘어진 자세에서 두 발 구가 22mm 파고든 상태였다
                #           (HL_sphere ↔ HR_sphere dist −0.0223). HOME 에서는 충돌 0.
                #   ⇒ arm 직후 **전 축을 HOME 으로 동시 이동**한 뒤 시험을 시작한다.
                #     목표 채널각은 JointMap 으로 뽑는다 — 수식을 여기 복사하지 않는다
                #     (sign·gear_k·offset·커플링·±180 포화가 전부 반영돼야 한다).
                if not a.no_home:
                    try:
                        _jm, _c = _jointmap()
                        _tgt = _jm.q_joint_to_ch([float(x) for x in _c["home"]["q_deg"]])
                        print(f"  [{j['name']}] HOME 정렬 — 목표 채널각 "
                              f"{[round(float(_tgt[c]), 2) for c in _jm.ch]}")
                        # ★목표가 채널각 한계 안에 있는지 **먼저** 확인한다. 예전엔 시험축
                        #   한계(foot [−40,30])가 전 채널에 적용돼 목표가 최대 57° 잘려도
                        #   조용히 "도착" 했다. 이제는 상자를 채널별로 넘기고, 그래도 밖이면
                        #   자르지 말고 **멈춘다** — 자세를 못 맞추면 시험할 이유가 없다.
                        _bad = [(c, float(_tgt[c]), box[c]) for c in _jm.ch
                                if c in box and not (box[c][0] - 1e-6 <= _tgt[c] <= box[c][1] + 1e-6)]
                        if _bad:
                            raise SystemExit(
                                "HOME 목표가 채널각 한계 밖이다 — spec.joints 의 q_min/q_max"
                                "(모델각) 또는 calib 을 확인할 것:\n" + "\n".join(
                                    f"    ch{c}: 목표 {t:+.2f}° ∉ [{b[0]:.2f}, {b[1]:.2f}]"
                                    for c, t, b in _bad))
                        # ★arm 이 먼저다 — enable 없이 복귀를 부르면 SHM 에 kp=kd=0 이
                        #   나가(shm_bridge.cpp:112) 모터가 전혀 안 움직인다.
                        #   시험축도 홀드게인으로 arm 한다(복귀 중엔 시험축도 '홀드' 다).
                        _log = lambda m: print(f"  [{j['name']}]{m}")
                        # ★뷰어 상태 발행 — 시험 중에는 biped_emb 를 끄므로(writer 는 하나)
                        #   여기서 발행하지 않으면 **화면이 멎는다**. 사람이 옆에 있는 구간이다.
                        _mode = f"pace:{j['name']}"
                        hw.publish_fn = lambda q_ch, rpy, on, _m=_mode, _j=_jm: publish_state(
                            _m, _j.ch_to_q_joint(np.asarray(q_ch, float)),
                            np.asarray(rpy, float), 1.0 / hw.dt, on, "pace")
                        # 자세 보고 + 영점 검증(모델각은 0 이어야 한다). 판정과 무관하게 찍는다.
                        _jig_precheck(hw, _jm, _tgt, spec, log=_log, tol_override=a.home_tol)
                        hw.arm(ch, _kp_ch, _kd_ch)
                        # ★복귀는 **지그 유무와 무관하게 항상** 돈다(사용자 결정 2026-08-11).
                        #   지그가 물려 있으면 편차가 작아 즉시 끝나거나 생략된다.
                        #   궤적은 GUI 홈복귀와 **같은 구현**(control/home.py:HomeTrajectory).
                        _homer = make_homer(_jm, _c, hw.dt)
                        goto_home(hw, _jm, _homer, _c, q_box=box, log=_log,
                                  speed_dps=a.home_speed)
                        if a.jig:
                            _jig_engage(hw, spec, hold, j)
                        # ★정렬이 끝났으니 한계를 **시험용(좁은)** 으로 조인다.
                        hw.lim = lim_test
                        print(f"  [{j['name']}] 한계 전환 → 시험용 채널각 "
                              f"[{lim_test.q_min:.2f}, {lim_test.q_max:.2f}]")
                    except Exception as e:
                        hw.limp()
                        print(f"  ✗ HOME 정렬 실패({type(e).__name__}: {e}) — limp 하고 중단.")
                        print(f"    자세를 모르면 충돌 상태에서 시험이 시작될 수 있다.")
                        raise
                else:
                    # ★--no-home 이면 자세를 모른다 → 한계를 조이면 지금 자리가 이미
                    #   시험상자 밖일 수 있다. 느슨한 기구한계를 유지하고 그 사실을 알린다.
                    print(f"  [{j['name']}] ⚠--no-home — 자세 정렬을 건너뛴다. "
                          f"한계는 기구값 [{lim_align.q_min:.2f}, {lim_align.q_max:.2f}] 유지.\n"
                          f"    늘어진 자세면 링크가 간섭한 상태일 수 있다(실측 −22mm).")

                # ★모든 시험 앞에서 파워단 생존을 확인한다. 텔레메트리 신선도(stale 검사)
                #   만으로는 부족하다 — EtherCAT·Emb·값갱신이 전부 정상인데 드라이버
                #   파워단만 래치오프된 상태가 실재하고, 그 상태의 측정은 전부 무효다.
                #   (2026-08-05: 그 상태에서 "순수토크 미지원" 이라는 틀린 결론이 나왔다.
                #    대조군 — 위치+게인으로 같은 크기 토크를 걸어본 것 — 이 잡아냈다.)
                print(f"  [{j['name']}] 드라이버 생존 확인…", flush=True)
                # ★게인은 **그 축의 배포값**을 쓴다. 예전엔 g["kp"] 를 읽었는데 spec 에
                #   gains.kp 가 없어져(축별 hold_kp 로 옮겨감) KeyError 로 죽었다.
                #   시험축은 지그에 물리지 않으므로 인하 없이 배포값 그대로가 맞다.
                hw.verify_driver_live(ch, kp=_kp_ch, kd=_kd_ch)
                print(f"  [{j['name']}] ✓ 파워단 정상 — 시험 시작")
                if "friction" in tests:
                    html, res = measure_actuator_friction(hw, spec, j, plotdir)
                    fragments.append(html); summary.append(("friction", res))
                if "torque" in tests:
                    res = probe_torque_mode(hw, spec, j)
                    summary.append(("torque", res))
                if "backlash" in tests:
                    html, res = measure_backlash(hw, spec, j, plotdir)
                    fragments.append(html); summary.append(("backlash", res))
                if "frf" in tests:
                    html, res = measure_frf(hw, spec, j, plotdir)
                    fragments.append(html); summary.append(("frf", res))
                if "latency" in tests:
                    html, res = measure_latency_and_backlash(hw, spec, j, plotdir)
                    fragments.append(html); summary.append(("latency", res))
                if "pace" in tests:
                    html, res = identify_pace(hw, spec, j, plotdir, a.out)
                    fragments.append(html); summary.append(("pace", res))
            except SafetyAbort as e:
                print(f"\n✗ 안전 중단 (limp 완료): {e}")
                fragments.append(f'<div class="warn"><b>{j["name"]} 안전 중단</b><br>{e}</div>')

    env = Environment(loader=FileSystemLoader(os.path.join(HERE, "templates")))
    out = env.get_template("base.html").render(
        tests=fragments, datetime=time.strftime("%Y-%m-%d %H:%M:%S"))
    path = os.path.join(a.out, "output.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"\n{'='*70}\n리포트: {path}")
    for kind, r in summary:
        if kind == "friction":
            print(f"  {r['name']:9s} 마찰  JFRIC={r['jfric']:.4f} Nm  "
                  f"JDAMP={r['jdamp']:.4f} Nm·s/rad  τ_s={r['tau_static']:.4f} Nm  R²={r['r2']:.3f}")
        elif kind == "torque":
            tb = f"{r['tau_break_mean']:.3f} Nm" if r["tau_break_mean"] else "—"
            print(f"  {r['name']:9s} 토크모드  {'지원됨' if r['supported'] else '미지원'}  파단토크={tb}")
        elif kind == "backlash":
            bl = f"{r['backlash_deg']:.4f}deg" if r["backlash_deg"] is not None else "유의미한 유격 없음"
            st = f"{r['stiffness_nm_per_deg']:.3f}Nm/deg" if r["stiffness_nm_per_deg"] else "—"
            print(f"  {r['name']:9s} 백래시  {bl}  강성={st}  루프폭={r['loop_width_deg']:.4f}deg")
        elif kind == "frf":
            fn = f"{r['f_n']:.2f}Hz" if r["f_n"] else "미검출"
            z = f"{r['zeta']:.3f}" if r["zeta"] else "—"
            kk = f"{r['k_nm_per_deg']:.2f}Nm/deg" if r["k_nm_per_deg"] else "—"
            print(f"  {r['name']:9s} FRF  f_n={fn}  zeta={z}  k={kk}  coh={r['coh_mean']:.2f}")
        elif kind == "latency":
            lm = f"{r['lost_motion_deg']:.4f}deg" if r["lost_motion_deg"] else "미검출"
            gd = f"{r['group_delay_ms']:.2f}ms" if r["group_delay_ms"] else "—"
            print(f"  {r['name']:9s} 지연  T_rt={r['t_rt_ms']:.2f}±{r['t_rt_sd']:.2f}ms  "
                  f"군지연={gd}  lost motion={lm}")
        else:
            ri = f"{r['rotor_i']:.3e}" if r["rotor_i"] is not None else "미확정(I_link 필요)"
            print(f"  {r['name']:9s} PACE  ROTOR_I={ri}  JDAMP={r['jdamp']:.4f}  "
                  f"JFRIC={r['jfric']:.4f}  R²={r['r2']:.3f} cond={r['cond']:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
