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


def main() -> int:
    ap = argparse.ArgumentParser(description="액추에이터 마찰/PACE 자동시험")
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--ch", type=int, action="append", help="시험할 SHM 채널(반복 가능)")
    ap.add_argument("--all", action="store_true", help="spec 의 installed_channels 전부")
    ap.add_argument("--tests", default="friction",
                help="friction,torque,backlash,latency,pace 중 콤마구분")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--selftest", action="store_true", help="하드웨어 없이 추정기만 검증")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    spec = load_spec(a.spec)
    chans = list(spec["meta"]["installed_channels"]) if a.all else (a.ch or [])
    if not chans:
        raise SystemExit("--ch 또는 --all 을 지정할 것 (--selftest 는 하드웨어 불필요)")
    tests = [t.strip() for t in a.tests.split(",") if t.strip()]

    plotdir = os.path.join(a.out, "plots")
    os.makedirs(plotdir, exist_ok=True)
    preflight(spec)

    from act_identify_pace import identify_pace
    from act_measure_friction import measure_actuator_friction
    from act_measure_latency import measure_latency_and_backlash
    from act_probe_torque_mode import probe_torque_mode
    from act_measure_backlash import measure_backlash

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
        lim = Limits(q_min=float(j["q_min"]), q_max=float(j["q_max"]),
                     tau_trip=float(sf["tau_trip_nm"]), tau_trip_ms=float(sf["tau_trip_ms"]),
                     vel_trip=float(sf["vel_trip_dps"]), err_max=float(sf["err_max_deg"]),
                     stale_ms=float(sf["stale_ms"]),
                     kp_max=float(g["kp_max"]), kd_max=float(g["kd_max"]))
        print(f"\n{'='*70}\n{j['name']} (ch{ch})  한계 q∈[{j['q_min']},{j['q_max']}]deg "
              f"τ_trip={sf['tau_trip_nm']}Nm\n{'='*70}")

        with Hardware(spec["shm"]["lib"], spec["shm"]["n_channel"], spec["shm"]["rate_hz"],
                      lim, int(spec["shm"]["recv_wait_ms"]),
                      float(g["enable_ramp_s"])) as hw:
            try:
                # ★모든 시험 앞에서 파워단 생존을 확인한다. 텔레메트리 신선도(stale 검사)
                #   만으로는 부족하다 — EtherCAT·Emb·값갱신이 전부 정상인데 드라이버
                #   파워단만 래치오프된 상태가 실재하고, 그 상태의 측정은 전부 무효다.
                #   (2026-08-05: 그 상태에서 "순수토크 미지원" 이라는 틀린 결론이 나왔다.
                #    대조군 — 위치+게인으로 같은 크기 토크를 걸어본 것 — 이 잡아냈다.)
                print(f"  [{j['name']}] 드라이버 생존 확인…", flush=True)
                hw.verify_driver_live(ch, kp=float(g["kp"]), kd=float(g["kd"]))
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
