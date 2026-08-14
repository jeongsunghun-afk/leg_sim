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

from dataclasses import replace
import numpy as np
import yaml
from jinja2 import Environment, FileSystemLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

from hwio import DEG, Hardware, Limits, SafetyAbort  # noqa: E402
from homing import goto_home, make_homer            # noqa: E402  GUI 와 동일 궤적
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "interface"))
from state_pub import publish_state, leg_extra                 # noqa: E402  뷰어와 동일 스키마


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


def _torque_html(r: dict) -> str:
    """토크모드 프로브 결과를 리포트 조각으로. **파단토크는 채널토크다** — k 배해야 관절값.

    ★2026-08-11: 이 함수가 없어서 torque 결과가 리포트에 전혀 안 남았다.
      다른 시험은 (html, res) 를 돌려주는데 probe_torque_mode 만 res 만 돌려주는
      비대칭이 원인이었다. 터미널을 닫으면 측정이 사라지는 상태였다.
    """
    k = float(r.get("gear_k", 1.0))
    rows = "".join(
        f"<tr><td>{'+' if t['dir']>0 else '−'}</td>"
        f"<td class=numeric>{t['tau_break']:.3f}</td>"
        f"<td class=numeric>{t['tau_break']*k:.3f}</td>"
        f"<td class=numeric>{t['tau_peak']:.3f}</td>"
        f"<td class=numeric>{t['dq_max']:.1f}</td>"
        f"<td>{'파단' if t['moved'] else '미동'}</td></tr>"
        for t in r.get("trials", []) if t.get("tau_break") is not None)
    tb = r.get("tau_break_mean")
    concl = ""
    if tb:
        vals = [t["tau_break"] for t in r["trials"] if t.get("tau_break")]
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        concl = (f"<p><b>평균 파단토크 {tb:.3f} ± {sd:.3f} Nm(채널)</b> → "
                 f"관절 <b>{tb*k:.3f} Nm</b> (k={k}) · "
                 f"모터축 <b>{tb*k/float(r.get('gear', 8.4)):.4f} Nm</b></p>")
    ok = r.get("supported")
    return f"""
<h2>{r['name']} (ch{r['ch']}) — 순수 토크모드 프로브</h2>
<p class=dim>Kp=Kd=0 으로 두고 τ_ff 만 올려 파단(breakaway)을 찾는다. 움직이면 드라이버가
τ_ff 를 실제로 쓴다는 뜻이고, 그 파단토크는 위치모드로 잰 정지마찰과 <b>일치해야</b> 한다.</p>
<p><b>{'✅ 순수 토크모드 지원됨' if ok else '❌ 미지원(τ_ff 무시)'}</b></p>
<table><tr><th>방향</th><th>파단τ[Nm 채널]</th><th>파단τ[Nm 관절]</th>
<th>최대τ</th><th>최대q̇[dps]</th><th>결과</th></tr>{rows}</table>
{concl}
<div class=warn><b>해석 주의</b><ul>
<li>원시값은 <b>채널토크</b>다. 드라이버가 전 축 7:1 로 가정하므로 관절토크 = τ_ch·k.</li>
<li>축끼리 비교하려면 <b>모터축</b>(관절토크÷N)으로 환산할 것 — 8축이 같은 모터다.</li>
<li>이 시험은 <b>마찰</b>을 잰다. 관성(I_total)은 재지 않는다 — 0.3° 에서 즉시 중단하므로
가속 구간이 없다. 관성은 <code>--tests inertia</code>(2단 토크법)로 잰다.</li>
</ul></div>
"""


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



def _sweep_hold_kp(hw, spec, j, gains_csv: str, cfg):
    """커플링 원천축의 홀드게인을 바꿔가며 토크프로브를 반복해 **마찰 vs 강성**을 가른다.

    ★왜 필요한가 (2026-08-12)
      foot 벨트가 calf 관절을 지난다(=couple_from). foot 토크의 반작용이 calf 를 밀고,
      우리는 **foot 채널각**으로 움직임을 판정하는데 q_ch_foot=(q_foot+q_calf)·s·k 라
      calf 변형이 그대로 섞인다. calf 는 모터로 잡혀 있고 모터 홀드는 **스프링**이다.
          Δq_ch_foot = τ · k_f²/(k_c²·kp_src)
      판정문턱 0.312° 를 순전히 탄성으로 채우는 토크가 kp_src=80 에서 0.681 Nm 인데
      실측 파단이 0.64~0.73 Nm 이라 **구분이 안 된다**(kp 80 이 하필 축퇴점이다).

    ⇒ kp_src 를 바꾸면 갈린다:
        탄성이면  τ_break ∝ kp_src   (40→0.34 · 80→0.68 · 160→1.36)
        마찰이면  τ_break 불변       (전부 ~0.64)
      ★내려가는 쪽이 특히 깨끗하다 — 40 에서 0.34 vs 0.64 는 2배 차이라 못 헷갈린다.
      ⚠160 의 탄성예측 1.36 은 tau_max 1.4 에 거의 닿는다. 안 움직이고 끝나면
        그것 자체가 "탄성" 쪽 증거다(마찰이면 0.64 에서 진작 움직였어야 한다).
    """
    from act_probe_torque_mode import probe_torque_mode
    # ★커플링 정보는 **spec.yaml 이 아니라 biped_emb.yaml** 에 있다 (2026-08-12).
    #   j 는 spec.joints 항목이라 couple_from 이 없다 — j.get("couple_from") 은 항상 None 이다.
    #   ⚠드라이런은 통과했는데 실기가 "couple_from 이 없다" 로 빠졌다. 테스트에 config 에서
    #     뽑은 j 를 먹였기 때문이다 — **생산 경로와 다른 데이터로 검증한** 전형적 실패다.
    #   ⇒ 이름으로 config 를 조회한다. 커플링·gear_k 의 단일 출처는 biped_emb.yaml 이다.
    ent = next((x for x in cfg["joints"] if x["name"] == j["name"]), {})
    src = ent.get("couple_from")
    if not src:
        print(f"  [{j['name']}] ⚠biped_emb.yaml 에 couple_from 이 없다 — "
              f"스윕할 원천축이 없어 1회만 돈다.")
        return probe_torque_mode(hw, spec, j)
    src_ch = next(int(x["channel"]) for x in cfg["joints"] if x["name"] == src)
    if not isinstance(hw.hold_kp, dict):
        print(f"  [{j['name']}] ⚠홀드게인이 스칼라다 — 축별 스윕 불가. 1회만 돈다.")
        return probe_torque_mode(hw, spec, j)
    gains = [float(g) for g in gains_csv.split(",") if g.strip()]
    kp_max = float(spec["gains"]["kp_max"])
    over = [g for g in gains if g > kp_max]
    if over:
        # ★조용히 클램프하면 "게인을 올렸는데 값이 안 변한다" 를 마찰이라 오독한다.
        print(f"  [{j['name']}] ⚠kp_max {kp_max} 초과 {over} — **그대로 두면 클램프되어 "
              f"판정이 오염된다**. 제외하고 진행한다.")
        gains = [g for g in gains if g <= kp_max]
    kf = float(ent.get("gear_k", j.get("gear_k", 1.0)))
    kc = float(next(x for x in cfg["joints"] if x["name"] == src).get("gear_k", 1.0))
    thr = float(spec.get("torque_mode", {}).get("move_thresh_deg", 0.30)) * 1.04  # 양자화 여유
    kp0 = float(hw.hold_kp[src_ch])
    print(f"\n  [{j['name']}] ★홀드게인 스윕 — 원천축 {src}(ch{src_ch}) kp {gains} "
          f"(현재 {kp0:.0f})")
    print(f"           탄성이면 τ_break ∝ kp, 마찰이면 불변. k_f={kf} k_c={kc}")
    rows, out = [], None
    try:
        for g in gains:
            hw.hold_kp[src_ch] = g
            pred = np.deg2rad(thr) * kc ** 2 * g / kf ** 2
            print(f"\n  ── {src} kp={g:.0f} — 탄성 예측 τ_break = {pred:.3f} Nm ──")
            r = probe_torque_mode(hw, spec, j)
            tb = r.get("tau_break_mean")
            rows.append((g, pred, tb, r))
            if abs(g - kp0) < 1e-9:
                out = r
    finally:
        hw.hold_kp[src_ch] = kp0        # ★어떤 경로로 끝나도 원상복구
    out = out or (rows[-1][3] if rows else None)

    print(f"\n{'='*70}\n  판정 — {j['name']} 파단토크가 마찰인가 강성인가\n{'='*70}")
    print(f"  {'kp_src':>7}{'탄성예측':>10}{'실측':>9}{'실측/예측':>11}")
    for g, pred, tb, _ in rows:
        print(f"  {g:>7.0f}{pred:>10.3f}"
              + (f"{tb:>9.3f}{tb/pred:>11.2f}" if tb else f"{'미동':>9}{'':>11}"))
    got = [(g, tb) for g, pred, tb, _ in rows if tb]
    if len(got) >= 2:
        (g1, t1), (g2, t2) = got[0], got[-1]
        slope = (t2 - t1) / (g2 - g1) * (g1 + g2) / (t1 + t2)     # 무차원 탄력도
        print(f"\n  게인 {g1:.0f}→{g2:.0f} 에서 τ_break {t1:.3f}→{t2:.3f} · 탄력도 {slope:+.2f}")
        print("  " + ("⇒ **탄성이 지배한다** — 잰 것은 마찰이 아니라 직렬 강성이다. "
                      "판정문턱·측정방식을 고쳐야 한다." if slope > 0.5 else
                      "⇒ **마찰이 지배한다** — τ_break 을 정지마찰로 써도 된다." if slope < 0.2 else
                      "⇒ **혼재** — 마찰과 탄성이 섞였다. τ_break 은 마찰의 상한으로만 쓸 것."))
    else:
        print("\n  ⚠유효 표본이 부족하다 — 판정 불가.")
    if out is not None:
        out["_sweep"] = [(g, pred, tb) for g, pred, tb, _ in rows]
    return out

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
                help="friction,torque,inertia,backlash,frf,latency,pace 중 콤마구분")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--selftest", action="store_true", help="하드웨어 없이 추정기만 검증")
    ap.add_argument("--solo", action="store_true",
                    help="★**측정축만 제어**한다. 나머지 축은 kp=kd=0 으로 완전히 놓는다 — "
                         "작업자가 손으로 잡는 전제(2026-08-12 사용자 결정). "
                         "홈복귀도 측정축만 움직이고, 트립 검사도 측정축만 본다. "
                         "⚠하위 관절이 무여자라 I_link 강체가정이 깨진다 — inertia·pace 에는 쓰지 말 것.")
    ap.add_argument("--sweep-hold-kp", default=None, metavar="G1,G2,...",
                    help="★파단토크가 마찰인지 **직렬 강성**인지 가르는 시험. 시험축의 "
                         "커플링 원천축(couple_from, foot→calf)의 홀드게인을 이 값들로 "
                         "바꿔가며 토크프로브를 반복한다. τ_break 이 게인에 비례하면 "
                         "탄성(=강성을 잰 것), 안 변하면 마찰이다. 예: --sweep-hold-kp 40,80,160")
    ap.add_argument("--pose", choices=("home", "neutral"), default="home",
                    help="시험 중 홀드 자세. neutral = **thigh 중력중립각**(+21.7°)으로 "
                         "옮겨 잡는다 — 그 자세에서 thigh 중력토크가 0 이라 처지지 않는다 "
                         "(HOME 은 2.36° 처짐). 지그 없이 돌릴 때 권장.")
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
    KNOWN = ("friction", "torque", "inertia", "backlash", "frf", "latency", "pace")
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
    from act_measure_inertia_torque import measure_inertia_torque

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
        # ★채널별 τ_trip (2026-08-12). 시험축 문턱(foot 8.0Nm)을 전 채널에 쓰면 hip 이
        #   상시 5.25Nm 라 문턱의 66% 를 점유한다 — hip 지그 해제로 실재 위험이 됐다.
        #   위치한계는 채널별 상자로 이미 갈라놨는데 토크만 안 갈라져 있었다.
        _tb = {int(k): float(v) for k, v in (sf.get("tau_trip_by_ch") or {}).items()}
        lim_ch = {c: replace(_mk(*box[c]), tau_trip=_tb[c]) for c in _tb if c in box}
        if lim_ch:
            print(f"  채널별 τ_trip: " + " ".join(
                f"ch{c}:{v.tau_trip:.0f}Nm" for c, v in sorted(lim_ch.items()))
                + f"  (나머지 {sf['tau_trip_nm']}Nm)")
        print(f"\n{'='*70}\n{j['name']} (ch{ch})  한계 모델각∈[{j['q_min']},{j['q_max']}] "
              f"→ **채널각**∈[{lo:.2f},{hi:.2f}]deg  τ_trip={sf['tau_trip_nm']}Nm\n{'='*70}")

        # ★시험축 외 홀드 대상 = 실장된 채널 − 시험축. 다리 조립 후 필수(spec.safety 주석 참조).
        # ★--solo 면 홀드축이 **없다**. 작업자가 손으로 잡으므로 모터로 잡지 않는다.
        #   hold_ch 가 비면 check_hold 가 아무것도 검사하지 않으므로 "홀드축이 밀렸다"
        #   같은 트립이 원천적으로 안 난다 — 손 위치는 오차가 크게 나는 게 정상이다.
        hold = ([] if a.solo
                else sorted(installed - {ch}) if bool(sf.get("hold_others", False)) else [])
        if a.solo:
            print(f"  [{j['name']}] ★--solo — **이 축만 제어**한다. 나머지 7축은 무여자다."
                  f"\n            반대편 다리는 작업자가 손으로 잡을 것. 홈복귀도 이 축만 움직인다.")
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
                        hw.publish_fn = lambda q_ch, rpy, on, raw=None, _m=_mode, _j=_jm: publish_state(
                            _m, _j.ch_to_q_joint(np.asarray(q_ch, float)),
                            np.asarray(rpy, float), 1.0 / hw.dt, on, "pace",
                            extra=leg_extra(_j, **(raw or {})))
                        # 자세 보고 + 영점 검증(모델각은 0 이어야 한다). 판정과 무관하게 찍는다.
                        hw.lim_ch = lim_ch            # ★채널별 트립 상한 적용
                        # ★스톨 감지용 중력 조회 — 표는 채널각으로 색인돼 있다.
                        #   이게 있어야 "정상 처짐" 과 "스톱에 밀어붙임" 이 구분된다.
                        _gt = spec["torque_mode"].get("tau_grav_table") or {}
                        # ★보정은 **상수가 아니라 각도의 함수**다 (2026-08-14).
                        #   종전엔 한 점에서 잰 오프셋을 전 구간에 썼다. calf·foot 은
                        #   보정이 0.01~0.19Nm 이라 통했는데 **thigh 에서 무너졌다**:
                        #     실측 +0.170 vs 표 +1.412 @ +36.41° → 보정 −1.242 Nm
                        #   thigh 는 중력 기울기가 ~0.1 Nm/° 라 시험구간 ±50° 에서 중력이
                        #   5Nm 변한다. 상수 오프셋으로는 못 따라간다. 실제로 파단이
                        #   0.197Nm 에 뜨고(비정상적으로 낮다) 축이 상자를 넘어 달아났다.
                        #   ⚠표가 틀린 건 **크기만이 아니라 모양**이다 — 표는 다른 관절이
                        #     neutral 일 때 뽑았는데 solo 는 calf −61°·foot +52° 로 늘어져
                        #     있어 thigh 가 지는 하중분포 자체가 다르다.
                        #   ⇒ 여러 각도에서 재서 **보간**한다. 점이 1개면 종전과 같다.
                        _gbias = {}                   # {ch: (q_pts[], corr_pts[])}
                        def _grav(c, q, _t=_gt, _b=_gbias):
                            base = (float(np.interp(q, _t[c]["q_ch"], _t[c]["tau"]))
                                    if c in _t else 0.0)
                            e = _b.get(c)
                            if e is None:
                                return base
                            qp, cp = e
                            # 바깥은 끝값 고정(np.interp 기본) — 외삽하지 않는다
                            return base + float(np.interp(q, qp, cp))
                        hw.grav_fn = _grav
                        _jig_precheck(hw, _jm, _tgt, spec, log=_log, tol_override=a.home_tol)
                        hw.arm(ch, _kp_ch, _kd_ch)
                        # ★홈복귀 **전에** 중력을 실측해 표를 보정한다 (2026-08-12).
                        #   표는 "다른 관절 = neutral" 가정으로 만든 것인데 --solo 는
                        #   하위 관절이 늘어져 있다. HL_thigh 에서 표 1.63 vs 진짜 2.90 Nm.
                        #   그 1.27Nm 이 홈복귀 중 스톨 감지의 가짜 초과토크가 되어
                        #   시험을 죽였다(실측 3.75Nm: 진짜 대비 초과 0.85=마찰인데
                        #   표 대비로는 2.12 > 2.0).
                        #   ⚠파단 뒤에 보정하는 것으로는 늦다 — 홈복귀부터 필요하다.
                        #   ⚠MuJoCo 확인: 이 오차는 축 각도에 거의 무관하다
                        #     (thigh 0~32° 에서 −1.30~−0.92). 상수 오프셋으로 충분하다.
                        # ★정렬 상자를 **현재 측정각까지 늘린다** (2026-08-12 실기 HR_calf).
                        #   goto_home 은 box_eff 로 그렇게 하는데 **arm() 의 _check 는
                        #   안 늘린다** — 같은 값을 두 곳에서 다르게 다루던 자리다.
                        #   HR_calf 가 −95.38 로 늘어져 상자 −93.0 밖이라 arm 에서 죽었다.
                        #   ⚠보호는 유지된다: 현재 자리는 허용하되 더 바깥으로는 못 간다.
                        #     정렬의 목적 자체가 상자 밖 자세를 안으로 데려오는 것이다.
                        _q_arm = float(hw.read(ch)[0])
                        _La = hw.limits_for(ch)
                        if not (_La.q_min <= _q_arm <= _La.q_max):
                            hw.lim_ch[ch] = replace(_La, q_min=min(_La.q_min, _q_arm),
                                                    q_max=max(_La.q_max, _q_arm))
                            _log(f"  ⚠정렬 상자를 현재각까지 확장 {_q_arm:+.2f}° "
                                 f"(원래 [{_La.q_min:+.1f}, {_La.q_max:+.1f}]) — "
                                 f"무여자 늘어짐이 관절한계 밖이다. config 한계를 볼 것.")
                        if a.solo and ch in _gt:
                            _q_now = float(hw.read(ch)[0])
                            _g_tbl = float(np.interp(_q_now, _gt[ch]["q_ch"], _gt[ch]["tau"]))
                            _g_meas = hw.measure_gravity(ch, _kp_ch, _kd_ch,
                                                         ff_fn=lambda q: hw.grav_fn(ch, q))
                            if _g_meas != _g_meas:      # nan — 상자 여유 부족
                                _log(f"  ⚠중력 실측 생략 — {_q_now:+.2f}° 가 상자 "
                                     f"[{hw.limits_for(ch).q_min:+.1f}, "
                                     f"{hw.limits_for(ch).q_max:+.1f}] 끝이라 움직일 자리가"
                                     f" 없다. **표 값을 그대로 쓴다**(오차가 남을 수 있다).")
                                _g_meas = _g_tbl
                            _qs_g, _cs_g = [_q_now], [_g_meas - _g_tbl]
                            _log(f"  중력 실측 {_g_meas:+.3f} vs 표 {_g_tbl:+.3f} Nm "
                                 f"@ {_q_now:+.2f}° → 보정 {_cs_g[0]:+.3f} Nm")
                            # ★추가 점 — 시험구간을 훑어 보정의 **모양**을 잡는다.
                            _npt = int(spec["torque_mode"].get("grav_probe_points", 3))
                            if _npt > 1:
                                # ★탐침 범위는 **시험이 실제로 쓸 폭**으로 제한한다
                                #   (2026-08-14 수정). 종전엔 정렬상자 전체를 훑었는데
                                #   thigh 는 그게 [−64.7, +135.2] 로 **200° 폭**이라
                                #   탐침점이 −58.7 / +35.2 / **+129.2** 로 잡혔다.
                                #   ⚠+129° 는 RL_INTERFACE 가 "URDF 가 틀렸다면 하드스톱에
                                #     부딪힌다(미확인)" 로 표시한 구간이다. 게다가 홈복귀
                                #     **전**이라 다른 7축이 무여자다 — 총 283° 를 매달린
                                #     다리로 훑게 된다. 필요도 없고 위험하다.
                                #   ⇒ 현재각 중심 ±span/2 로 자르고 상자로 clip 한다.
                                #     관성시험 이동폭이 63~70° 이므로 70 이면 덮는다.
                                #   ⚠보간은 바깥을 **끝값으로 고정**한다(외삽 안 함) —
                                #     탐침 밖으로 나가도 마지막 보정값이 유지된다.
                                _Lg = hw.limits_for(ch)
                                _m = 6.0                      # 상자 끝 여유
                                _sp = float(spec["torque_mode"].get("grav_probe_span_deg", 70.0))
                                _plo = max(_Lg.q_min + _m, _q_now - _sp / 2)
                                _phi = min(_Lg.q_max - _m, _q_now + _sp / 2)
                                if _phi - _plo < 10.0:        # 자리가 없으면 포기한다
                                    _log(f"    중력 탐침 생략 — 쓸 폭이 {_phi-_plo:.1f}° 뿐이다")
                                    _cand = []
                                else:
                                    _cand = np.linspace(_plo, _phi, _npt)
                                    # 가까운 점부터 — 총 이동거리를 줄인다
                                    _cand = sorted(_cand, key=lambda v: abs(v - _q_now))
                                    _tot = 0.0; _cur = _q_now
                                    for _v in _cand:
                                        _tot += abs(_v - _cur); _cur = _v
                                    _log(f"    중력 탐침 {len(_cand)}점 "
                                         f"[{_plo:+.1f}, {_phi:+.1f}]° · 총 이동 {_tot:.0f}°"
                                         f" ≈ {_tot/15:.0f}초 (다른 축은 무여자다)")
                                for _qp in _cand:
                                    if abs(_qp - _q_now) < 4.0:
                                        continue              # 이미 잰 자리
                                    try:
                                        hw.goto(ch, float(_qp), _kp_ch, _kd_ch, speed_dps=15.0)
                                        _gm = hw.measure_gravity(
                                            ch, _kp_ch, _kd_ch,
                                            ff_fn=lambda q: hw.grav_fn(ch, q))
                                    except Exception as _e:
                                        _log(f"    중력 탐침 {_qp:+.1f}° 실패({type(_e).__name__}) — 건너뛴다")
                                        continue
                                    if _gm != _gm:
                                        continue
                                    _gt_p = float(np.interp(_qp, _gt[ch]["q_ch"], _gt[ch]["tau"]))
                                    _qs_g.append(float(_qp)); _cs_g.append(_gm - _gt_p)
                                    _log(f"    보정 @{_qp:+7.1f}° = {_gm - _gt_p:+.3f} Nm")
                                _o = np.argsort(_qs_g)
                                _qs_g = list(np.asarray(_qs_g)[_o])
                                _cs_g = list(np.asarray(_cs_g)[_o])
                                _spread = max(_cs_g) - min(_cs_g)
                                _log(f"  중력보정 {len(_qs_g)}점 · 폭 {_spread:.3f} Nm"
                                     + ("  ★상수로는 못 맞는 축이다" if _spread > 0.5 else ""))
                            _gbias[ch] = (np.asarray(_qs_g), np.asarray(_cs_g))
                            if max(abs(np.asarray(_cs_g))) > 3.0:
                                _log(f"  ⚠보정이 3Nm 을 넘는다 — 표를 다시 뽑을 것"
                                     f"(tools/gen_grav_table.py). 일단 진행한다.")
                            # ★기본 FF 로 건다 — 이 뒤 **모든 쓰기**가 자동으로 태운다
                            #   (홈복귀·arm·verify_driver_live·파단·스윕 전부).
                            #   호출부마다 꿰다가 verify_driver_live 를 빠뜨려 thigh 가
                            #   17.5° 튀었다. grav_fn 이 갱신되면 여기도 따라간다.
                            hw.tau_ff_fn = (lambda c, q, _c=ch:
                                            float(hw.grav_fn(_c, q)) if c == _c else 0.0)
                            hw.arm(ch, _kp_ch, _kd_ch)
                        # ★복귀는 **지그 유무와 무관하게 항상** 돈다(사용자 결정 2026-08-11).
                        #   지그가 물려 있으면 편차가 작아 즉시 끝나거나 생략된다.
                        #   궤적은 GUI 홈복귀와 **같은 구현**(control/home.py:HomeTrajectory).
                        # ★"상태 정지"(SHM 동결) 는 **한 번 재시도한다** (2026-08-12).
                        #   실기 ch3 홈복귀 중 365ms 동결로 세션이 끝났는데, **바로 다음
                        #   축(ch1)이 정상으로 돌았다** — 전원사이클이 필요한 OP 이탈이
                        #   아니라 일과성 정지였다. 그걸 못 넘겨서 foot 을 못 쟀다.
                        #   ⚠판별은 SHM 이 **스스로 돌아오는가** 다. 진짜 OP 이탈은 안 돌아온다.
                        #     wait_fresh 가 5초 안에 성공하면 일과성, 실패하면 진짜다.
                        #   ⚠재시도는 limp 된 **실제 위치에서 다시 계획**한다. 눈먼 채로
                        #     이어가지 않는다 — homer 를 새로 만드는 이유가 그것이다.
                        for _try in range(2):
                            try:
                                goto_home(hw, _jm, make_homer(_jm, _c, hw.dt), _c,
                                          q_box=box, log=_log, speed_dps=a.home_speed,
                                          **({"only_ch": ch, "kp": _kp_ch, "kd": _kd_ch}
                                             if a.solo else {}))
                                break
                            except SafetyAbort as _e:
                                if "상태 정지" not in str(_e) or _try:
                                    raise
                                _log(f"  ⚠SHM 이 일시 정지했다 — {str(_e).splitlines()[0]}")
                                _log("    SHM 이 스스로 돌아오는지 5초 기다린다"
                                     "(돌아오면 일과성, 아니면 EtherCAT OP 이탈이다)…")
                                hw.wait_fresh(timeout_s=5.0, ch=ch)   # 실패 시 여기서 중단
                                _log("    ✓ 복구됨 — **실제 위치에서 다시 계획**해 재시도한다."
                                     " 이 동결은 기록해 둘 것(원인 미제).")
                        # ★홀드 자세 (2026-08-12, 사용자 제안).
                        #   HOME 은 thigh 중력토크 −2.06Nm 이라 kp50 에서 2.36° 처진다.
                        #   thigh 를 **중력중립각**으로 옮기면 그 토크가 0 이 되어 처짐 자체가
                        #   없어진다(막는 게 아니라 없애는 쪽). foot 중력은 0.033→0.038Nm 로
                        #   사실상 그대로라 파단 측정에는 영향이 없고, 링크 간섭도 없다.
                        # ★--solo 면 다른 축을 못 움직이므로 자세 덮어쓰기가 무의미하다.
                        _hp = {} if a.solo else _c.get("hold_pose", {})
                        _pose = _hp.get(f"{a.pose}_deg")
                        # ★시험축별 덮어쓰기 — 시험하지 않는 쪽 hip 을 바깥으로 재껴
                        #   두 다리가 부딪히지 않게 한다(config 주석 참조).
                        #   ⚠홀드축만 덮어쓴다. 시험축 자신을 옮기면 측정 자세가 바뀐다.
                        _ov = {int(k): float(v) for k, v in
                               (_hp.get("by_test_ch", {}).get(ch) or {}).items()}
                        _ov.pop(ch, None)
                        if _ov:
                            _pose = list(_pose or _c["home"]["q_deg"])
                            for _c2, _v in _ov.items():
                                _pose[_c2] = _v
                            _log(f"    ★시험축별 자세 — " + " · ".join(
                                f"{_jm.names[c2]} {v:+.1f}°" for c2, v in sorted(_ov.items()))
                                + "  (반대 다리를 바깥으로 재껴 충돌 회피)")
                        if _pose and (a.pose != "home" or _ov):
                            _log(f"    홀드 자세 → {a.pose} "
                                 f"(thigh {_pose[1]:+.1f}° — 중력중립, 처짐 2.36°→0)")
                            goto_home(hw, _jm, make_homer(_jm, _c, hw.dt, q_deg=_pose),
                                      _c, q_box=box, log=_log, speed_dps=a.home_speed,
                                      **({"only_ch": ch, "kp": _kp_ch, "kd": _kd_ch}
                                         if a.solo else {}))
                        # ★홀드 목표를 **여기서 한 번** 확정한다 → 이후 arm() 이 재사용.
                        #   안 하면 arm() 이 매번 '지금 처진 자리' 를 목표로 삼아 래칫이 된다.
                        _tgt_ch = _jm.q_joint_to_ch(np.asarray(
                            _pose if (_pose and (a.pose != "home" or _ov))
                            else _c["home"]["q_deg"], float))
                        hw.latch_hold(_tgt_ch)
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
                    # ★fragments.append 가 빠져 있었다(2026-08-11) — 터미널에만 찍히고
                    #   리포트에는 안 남았다. 다른 시험은 전부 (html, res) 를 돌려주는데
                    #   이것만 res 만 돌려주는 비대칭이 원인이었다.
                    if a.sweep_hold_kp:
                        res = _sweep_hold_kp(hw, spec, j, a.sweep_hold_kp, _jointmap()[1])
                    else:
                        res = probe_torque_mode(hw, spec, j)
                    # ★파단토크를 관성시험에 넘긴다 — 준위를 축별로 자동설계하려면
                    #   그 축 자신의 파단값이 필요하다(HL 0.674 · HR 0.753 로 다르다).
                    j["_tau_break"] = res.get("tau_break_mean")
                    fragments.append(_torque_html(res))
                    summary.append(("torque", res))
                if "inertia" in tests:
                    j["_ch_box"] = box_pin[ch]     # 방향별 시작점 계산용(채널각 한계)
                    html, res = measure_inertia_torque(hw, spec, j, plotdir)
                    fragments.append(html); summary.append(("inertia", res))
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
        elif kind == "inertia":
            if "I_joint" in r:
                e = f"  MJCF 대비 {r['err_pct']:+.1f}%" if "err_pct" in r else ""
                print(f"  {r['name']:9s} 관성  I_ch={r['I_ch']:.5f}  "
                      f"I_joint={r['I_joint']:.5f} kg·m²{e}")
            else:
                print(f"  {r['name']:9s} 관성  측정 실패(표본 부족)")
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
