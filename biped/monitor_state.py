#!/usr/bin/env python3
"""monitor_state.py — 관절별 **측정 vs 명령** 실시간 모니터 (위치·속도·토크).

왜 터미널인가 (2026-08-13):
  Pi 4 에서 이미 MuJoCo 뷰어(biped_monitor) + dearpygui GUI + 500Hz 제어루프가 돈다.
  이 저장소엔 **루프가 밀리면 jog 속도제한이 20dps → 500dps 로 뚫린다**는 실측 기록이
  있다(app/biped_emb.py 페이싱 주석). 세 번째 무거운 GUI 는 그 위험을 키운다.
  ⇒ 표시는 stdlib + ANSI 로만 한다. 의존성 0, CPU 무시할 수준, SSH 로도 보인다.

읽는 것: /tmp/biped_state.json (QUAD_STATE) — **읽기 전용**. 아무것도 쓰지 않는다.
  제어기가 발행하는 키를 그대로 쓴다:
    q_leg_deg  dq_leg_dps  tau_leg_nm      측정 (모델각 deg · deg/s · 관절 Nm)
    q_cmd_deg  dq_cmd_dps  tau_cmd_nm      명령 (같은 단위)
    kp_leg kd_leg health installed mode loop_hz ...
  ⚠단위는 전부 **모델각**이다(채널각 아님). q_ch_deg 는 --ch 로 따로 볼 수 있다.

없는 키는 `—` 로 뜬다. 두 경우가 있다:
  · 제어기가 구버전 → 재시작하면 나온다
  · **C++ 배포(biped_deploy)는 순수 토크모드**라 위치·속도 "명령" 이 원래 없다
    (kp=kd=0). 그때 q명령/dq명령 이 비는 건 정상이고, 토크만 비교하면 된다.

사용:
  python3 monitor_state.py                 # 기본 10Hz
  python3 monitor_state.py --hz 20         # 갱신율
  python3 monitor_state.py --ch            # 채널각도 같이 표시(캘리브레이션용)
  python3 monitor_state.py --no-color
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")
FALLBACK_NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot",
                  "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]

# ── 경고 임계 (env 로 조정) ────────────────────────────────────────────────
#   ★위치오차는 그 자체보다 **토크로 환산했을 때** 의미가 있다 — 드라이버 MIT 법칙이
#     τ = kp·err[rad] + kd·derr 라 kp 1 당 0.0175 Nm/deg 다. kp100 축에서 2° 면 3.5 Nm.
#     그래서 축마다 다른 kp 를 곱해 **토크 기여**로 판정한다. 고정 각도임계는 hip 과
#     foot 을 같은 잣대로 재게 되어 한쪽이 늘 빨갛거나 늘 조용해진다.
WARN_TAU = float(os.environ.get("MON_WARN_TAU", "3.0"))    # 위치오차의 토크환산[Nm] 경고
BAD_TAU = float(os.environ.get("MON_BAD_TAU", "8.0"))      # 〃 위험
WARN_DQ = float(os.environ.get("MON_WARN_DQ", "30"))       # 속도[dps] 경고
BAD_DQ = float(os.environ.get("MON_BAD_DQ", "120"))        # 〃 (속도트립 200 의 60%)
WARN_TAU_M = float(os.environ.get("MON_WARN_TAU_M", "8.0"))   # 측정토크[Nm] 경고
BAD_TAU_M = float(os.environ.get("MON_BAD_TAU_M", "15.0"))    # 〃 (토크트립 15 와 동일)
STALE_MS = float(os.environ.get("MON_STALE_MS", "500"))    # 이보다 오래된 상태 = 끊김
KP_TO_NM_PER_DEG = 0.0175                                  # [실측 2026-08-05] kp 1 = 0.0175 Nm/deg

C = {"r": "\033[31m", "y": "\033[33m", "g": "\033[32m", "d": "\033[2m",
     "b": "\033[1m", "c": "\033[36m", "x": "\033[0m", "R": "\033[41m\033[97m"}


def paint(s, col, on=True):
    return f"{C[col]}{s}{C['x']}" if (on and col) else s


def lvl(v, warn, bad):
    a = abs(v)
    return "r" if a >= bad else ("y" if a >= warn else "")


def fmt(v, w=7, p=2):
    return "—".rjust(w) if v is None else f"{v:+{w}.{p}f}"


def get(st, key, n):
    """길이 n 의 실수 리스트로. 없거나 길이가 모자라면 None 리스트."""
    v = st.get(key)
    if not isinstance(v, list) or len(v) < n:
        return [None] * n
    try:
        return [float(x) for x in v[:n]]
    except (TypeError, ValueError):
        return [None] * n


def load_names():
    try:
        import yaml
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "emb", "config", "biped_emb.yaml")
        return [j["name"] for j in yaml.safe_load(open(p))["joints"]]
    except Exception:
        return FALLBACK_NAMES          # config 를 못 읽어도 모니터는 떠야 한다


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=10.0, help="갱신율(기본 10)")
    ap.add_argument("--ch", action="store_true", help="채널각도 같이 표시")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="한 프레임만 그리고 끝낸다(시험·스크립트용. 커서제어 없음)")
    a = ap.parse_args()
    col = not a.no_color and sys.stdout.isatty()
    names = load_names()
    nj = len(names)
    period = 1.0 / max(a.hz, 0.5)

    peak_dq = [0.0] * nj          # 세션 중 최대 |Δ| — 순간 스파이크는 눈으로 못 잡는다
    peak_eq = [0.0] * nj
    peak_tm = [0.0] * nj
    if not a.once:
        sys.stdout.write("\033[2J")
    try:
        while True:
            t0 = time.perf_counter()
            try:
                age_ms = (time.time() - os.path.getmtime(STATE)) * 1e3
                st = json.load(open(STATE))
            except Exception as e:
                sys.stdout.write("\033[H\033[J")
                print(paint(f"  상태파일을 못 읽는다: {STATE}", "r", col))
                print(f"  ({e})\n  제어기가 떠 있는지 확인:  pgrep -f 'biped_emb.py|biped_deploy'")
                if a.once:
                    return 1
                time.sleep(0.5)
                continue

            qm = get(st, "q_leg_deg", nj);   qc = get(st, "q_cmd_deg", nj)
            dm = get(st, "dq_leg_dps", nj);  dc = get(st, "dq_cmd_dps", nj)
            tm = get(st, "tau_leg_nm", nj);  tc = get(st, "tau_cmd_nm", nj)
            kp = get(st, "kp_leg", nj);      kd = get(st, "kd_leg", nj)
            qch = get(st, "q_ch_deg", nj)
            # ★ucStatus 원값 = MD80 ERROR VECTOR 하위 8bit(벤더 확인 2026-08-14).
            #   'fault' 라는 것만 알고 왜인지 못 보던 걸 숫자로 드러낸다.
            stt = get(st, "stt_raw", nj)
            health = st.get("health") or ["?"] * nj
            inst = st.get("installed") or [True] * nj

            out = [] if a.once else ["\033[H"]
            stale = age_ms > STALE_MS
            hdr = (f" mode={st.get('mode','?'):<6} backend={st.get('backend','?'):<6} "
                   f"loop={st.get('loop_hz',0):6.1f}Hz  tilt={st.get('tilt_deg',0):5.2f}°  "
                   f"age={age_ms:6.1f}ms")
            out.append(paint(f"{hdr:<92}", "R" if stale else "b", col) + "\n")
            flags = []
            if stale:
                flags.append(paint(f"*** STALE {age_ms:.0f}ms — 제어기가 발행을 멈췄다 ***", "R", col))
            if st.get("estop_latched") or st.get("estop"):
                flags.append(paint(f"E-STOP 래치: {st.get('estop_reason','?')}", "R", col))
            if st.get("wd_trip"):
                flags.append(paint("워치독 트립", "r", col))
            if st.get("write_fail"):
                flags.append(paint(f"write_fail={st['write_fail']}", "r", col))
            if st.get("tilt_estop_ok") is False:
                flags.append(paint("IMU 죽음 → tilt E-stop 무력", "y", col))
            # ★QP 건강도 = **접지 판정**. 발이 덜 닿으면 QP 가 매 틱 실패하고 중력보상
            #   폴백으로 떨어지는데 겉보기엔 안정돼 보인다(매달림 실측 95% vs 접지 0.05%).
            _qf = st.get("qp_fail_pct")
            if _qf is not None:
                _ce = st.get("qp_cerr") or [0, 0, 0]
                _cn = (sum(v * v for v in _ce)) ** 0.5 * 1e3
                _t = (f"QP실패 {_qf:.0f}% · K={st.get('qp_K','?')} · com_err {_cn:.0f}mm")
                flags.append(paint(_t + ("  ← 접지 불량(폐루프 죽음)" if _qf >= 50 else ""),
                                   "R" if _qf >= 50 else ("y" if _qf >= 20 else "g"), col))
            out.append(("  " + "   ".join(flags) if flags else "  " + paint("이상 없음", "g", col)) + "\n")

            chh = "   q_ch" if a.ch else ""
            out.append(paint(
                f"\n  {'축':<9}{'q측정':>8}{'q명령':>8}{'Δq':>7}{'≈Nm':>6} │"
                f"{'dq측정':>8}{'dq명령':>8}{'Δdq':>7} │{'τ측정':>8}{'τ명령':>8}{'Δτ':>7} │"
                f"{'kp':>5}{'kd':>5}{chh:>8}  st  err\n", "c", col))
            out.append("  " + "─" * (90 + (8 if a.ch else 0)) + "\n")

            for i in range(nj):
                eq = None if (qm[i] is None or qc[i] is None) else qm[i] - qc[i]
                ed = None if (dm[i] is None or dc[i] is None) else dm[i] - dc[i]
                et = None if (tm[i] is None or tc[i] is None) else tm[i] - tc[i]
                # 위치오차 → 토크환산(축별 kp 반영). kp 가 없으면 환산 불가.
                enm = None if (eq is None or kp[i] is None) else eq * kp[i] * KP_TO_NM_PER_DEG
                if eq is not None:
                    peak_eq[i] = max(peak_eq[i], abs(eq))
                if dm[i] is not None:
                    peak_dq[i] = max(peak_dq[i], abs(dm[i]))
                if tm[i] is not None:
                    peak_tm[i] = max(peak_tm[i], abs(tm[i]))
                c_eq = lvl(enm, WARN_TAU, BAD_TAU) if enm is not None else ""
                c_dm = lvl(dm[i], WARN_DQ, BAD_DQ) if dm[i] is not None else ""
                c_tm = lvl(tm[i], WARN_TAU_M, BAD_TAU_M) if tm[i] is not None else ""
                h = str(health[i]) if i < len(health) else "?"
                c_h = {"ok": "g", "fault": "r", "dead": "r", "absent": "d"}.get(h, "y")
                nm = names[i] if (i >= len(inst) or inst[i]) else paint(names[i], "d", col)
                row = (f"  {nm:<9}{fmt(qm[i])}{fmt(qc[i])}"
                       f"{paint(fmt(eq), c_eq, col)}{paint(fmt(enm, 6, 1), c_eq, col)} │"
                       f"{paint(fmt(dm[i], 8, 1), c_dm, col)}{fmt(dc[i], 8, 1)}{fmt(ed, 7, 1)} │"
                       f"{paint(fmt(tm[i], 8, 2), c_tm, col)}{fmt(tc[i], 8, 2)}{fmt(et, 7, 2)} │"
                       f"{fmt(kp[i], 5, 0)}{fmt(kd[i], 5, 1)}")
                if a.ch:
                    row += fmt(qch[i], 8, 1)
                ev = "" if stt[i] is None else f" 0x{int(stt[i]):02X}"
                c_ev = "r" if (stt[i] and int(stt[i])) else "d"
                out.append(row + "  " + paint(h[:1], c_h, col) + paint(ev, c_ev, col) + "\n")

            out.append("\n  " + paint("세션 최대 │ ", "d", col)
                       + paint(f"|Δq| {max(peak_eq):5.2f}°  |dq| {max(peak_dq):6.1f}dps  "
                               f"|τ| {max(peak_tm):5.2f}Nm", "d", col) + "\n")
            out.append(paint("  Ctrl-C 종료 · 읽기전용(아무것도 쓰지 않는다) · 단위=모델각\n", "d", col))
            if not a.once:
                out.append("\033[J")
            sys.stdout.write("".join(out))
            sys.stdout.flush()
            if a.once:
                return 0
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
