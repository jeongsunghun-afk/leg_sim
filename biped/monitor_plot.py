#!/usr/bin/env python3
"""monitor_plot.py — 축 하나를 골라 **위치·속도·토크**를 그래프로 본다 (측정 vs 명령).

왜 축 하나만 그리나: 8축 × 3량 × 2계열 = 48줄이면 아무것도 안 보인다.
  판단은 늘 "어느 축이 이상한가" → "그 축이 왜" 순서로 간다. 앞은 값 표(monitor_state.py)나
  3D 뷰어가 하고, **뒤를 이 창이 한다.**
  ★이력은 **전 축을 항상 버퍼링**한다 — 축을 바꿔도 과거가 살아 있다. 이상을 본 뒤에
    그 축으로 옮기면 이미 늦은 상황이 되면 안 되기 때문이다.

읽는 것: /tmp/biped_state.json (QUAD_STATE) — **읽기 전용**. 아무것도 쓰지 않는다.
  q_leg_deg / dq_leg_dps / tau_leg_nm      측정
  q_cmd_deg / dq_cmd_dps / tau_cmd_nm      명령
  단위는 전부 **모델각**(deg·deg/s)과 **관절토크**(Nm). 채널각 아님.

⚠**C++ 배포(biped_deploy)는 순수 토크모드**라 위치·속도 "명령" 이 없다(kp=kd=0).
  그때 명령 곡선이 안 그려지는 건 정상이고, 창에 그렇게 표시된다.

사용:
  python3 monitor_plot.py                  # 기본 10초 창 · 30Hz
  python3 monitor_plot.py --win 20 --hz 20
  GUI_FONT=/경로/폰트.ttf python3 monitor_plot.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import dearpygui.dearpygui as dpg

STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")
FALLBACK_NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot",
                  "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
KP_TO_NM_PER_DEG = 0.0175          # [실측 2026-08-05] 드라이버 MIT: kp 1 = 0.0175 Nm/deg
STALE_MS = 500.0

# 계열 3종 × (측정 키, 명령 키, 단위, 플롯 라벨, y축 최소 표시폭)
#   ★최소 표시폭이 왜 필요한가: fit_axis_data 는 데이터 범위에 **딱** 맞춘다.
#     값이 0 근처에서만 놀면 축이 ±0.001 로 붕괴해 잡음이 산맥처럼 보이고,
#     0 이 화면 밖으로 나가 부호 감각도 사라진다. 아래 폭과 **0 포함**을 강제한다.
SIG = [("q",   "q_leg_deg",   "q_cmd_deg",   "deg",   "위치 [deg]",    4.0),
       ("dq",  "dq_leg_dps",  "dq_cmd_dps",  "deg/s", "속도 [deg/s]", 20.0),
       ("tau", "tau_leg_nm",  "tau_cmd_nm",  "Nm",    "토크 [Nm]",     4.0)]


# ── CSV 기록 ────────────────────────────────────────────────────────────────
#   ★왜 필요한가: 평발 stand 는 **검증용** 시험이다. 눈으로 본 것만 남으면 중력보상
#     잔차(→토크 스케일 α)·좌우 비대칭·발목 커플링 잔차를 나중에 정량으로 못 판다.
#   ★**신선한 표본만** 기록한다(상태파일 mtime 이 바뀔 때만). 같은 값을 반복해 넣으면
#     표본율이 뻥튀기돼 수치미분·FFT 가 전부 틀어진다.
#   ⚠빈 칸은 "그 시점에 그 값이 발행되지 않았다" 는 뜻이다(예: 순수 토크모드의 위치명령).
#     0 으로 채우면 "명령이 0 이었다" 와 구분이 안 된다.
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_KEYS = ("q_leg_deg", "q_cmd_deg", "dq_leg_dps", "dq_cmd_dps",
            "tau_leg_nm", "tau_cmd_nm", "kp_leg", "kd_leg")
LOG_SUFFIX = ("q_m", "q_c", "dq_m", "dq_c", "tau_m", "tau_c", "kp", "kd")


def log_header(names):
    h = ["t_s", "epoch", "mode", "loop_hz", "tilt_deg", "estop", "wd_trip", "write_fail"]
    for n in names:
        h += [f"{n}_{s}" for s in LOG_SUFFIX]
    return h


def log_row(t_s, mt, st, nj):
    c = {k: col(st, k, nj) for k in LOG_KEYS}
    r = [f"{t_s:.4f}", f"{mt:.4f}", st.get("mode", ""), st.get("loop_hz", ""),
         st.get("tilt_deg", ""), int(bool(st.get("estop_latched") or st.get("estop"))),
         int(bool(st.get("wd_trip"))), st.get("write_fail", "")]
    for i in range(nj):
        for k in LOG_KEYS:
            v = c[k][i]
            r.append("" if v is None else f"{v:.4f}")     # 빈 칸 = 미발행. 0 과 구분된다
    return r


def log_open(path, names):
    """(파일, writer, 절대경로). 상대경로는 logs/ 아래로. 실패는 호출자가 받는다."""
    if not path:
        path = "biped_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    if not os.path.isabs(path):
        os.makedirs(LOG_DIR, exist_ok=True)
        path = os.path.join(LOG_DIR, path)
    f = open(path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(log_header(names))
    return f, w, os.path.abspath(path)


def load_names():
    try:
        import yaml
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "emb", "config", "biped_emb.yaml")
        return [j["name"] for j in yaml.safe_load(open(p))["joints"]]
    except Exception:
        return FALLBACK_NAMES


def col(st, key, n):
    v = st.get(key)
    if not isinstance(v, list) or len(v) < n:
        return [None] * n
    out = []
    for x in v[:n]:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            out.append(None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--win", type=float, default=10.0, help="시간창[s] (기본 10)")
    ap.add_argument("--hz", type=float, default=30.0, help="표본율 (기본 30)")
    ap.add_argument("--log", nargs="?", const="", default=None, metavar="파일",
                    help="시작하자마자 CSV 기록. 경로 생략 시 자동이름(logs/biped_날짜_시각.csv)")
    a = ap.parse_args()
    names = load_names()
    nj = len(names)
    maxlen = max(60, int(a.win * a.hz * 1.4))

    # 이력: [축][계열] → (측정 deque, 명령 deque). 시간축은 공용.
    ts: deque = deque(maxlen=maxlen)
    hist = {k: ([[deque(maxlen=maxlen) for _ in range(nj)],
                 [deque(maxlen=maxlen) for _ in range(nj)]]) for k, *_ in SIG}
    sel = [0]                      # 선택된 축 (리스트 = 콜백에서 쓰기 위함)
    t0 = [None]                    # 첫 표본 시각(파일 mtime 기준)
    last_mtime = [0.0]
    st_cache = {}                  # 마지막으로 읽은 상태(갱신 없을 때 그대로 재사용)

    dpg.create_context()
    # ★한글 폰트 — teleop_gui_biped.py 와 **같은 규약**(그쪽 주석 참조).
    #   ⚠add_font_range_hint 는 dearpygui 2.x 에서 deprecated no-op 이라 쓰지 않는다.
    fcands = [os.environ.get("GUI_FONT", ""),
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]
    fpath = next((f for f in fcands if f and os.path.exists(f)), None)
    kf = None
    if fpath:
        with dpg.font_registry():
            kf = dpg.add_font(fpath, 17)
    else:
        print("[monitor] ⚠ CJK 폰트를 못 찾았다 — 한글 라벨이 깨져 보인다.\n"
              "          설치: sudo apt install fonts-noto-cjk  (또는 GUI_FONT=/경로.ttf)")

    def on_sel(_s, val):
        sel[0] = names.index(val) if val in names else 0

    # ── 기록 상태 ──────────────────────────────────────────────────────────
    #   ⚠파일을 닫는 책임이 여기 있다. 정지 버튼·창 종료 **양쪽**에서 닫아야 한다 —
    #     안 닫으면 마지막 버퍼가 안 나가서 시험 끝부분이 통째로 사라진다.
    rec = {"f": None, "w": None, "path": "", "n": 0, "t0": None, "last_flush": 0.0}

    def rec_stop():
        if rec["f"]:
            try:
                rec["f"].flush()
                rec["f"].close()
            except Exception:
                pass
        rec.update(f=None, w=None, n=0, t0=None)

    def rec_start(path=""):
        rec_stop()
        try:
            f, w, ap_ = log_open(path, names)
            rec.update(f=f, w=w, path=ap_, n=0, t0=None, last_flush=time.time())
            print(f"[monitor] 기록 시작: {ap_}")
        except Exception as e:
            rec["path"] = f"기록 실패: {e}"
            print(f"[monitor] ⚠ 기록 실패: {e}")

    def on_rec(_s=None, _v=None):
        if rec["f"]:
            p = rec["path"]
            rec_stop()
            rec["path"] = p
            dpg.set_item_label("rec_btn", "● 기록 시작")
        else:
            rec_start()
            dpg.set_item_label("rec_btn", "■ 기록 정지")

    with dpg.window(tag="main"):
        dpg.add_text("", tag="hdr")
        dpg.add_text("", tag="flags", color=(255, 170, 60))
        dpg.add_separator()
        dpg.add_text("축 선택 — 고른 축의 위치·속도·토크만 그린다 (이력은 전 축 유지)")
        dpg.add_radio_button(names, tag="sel", default_value=names[0],
                             horizontal=True, callback=on_sel)
        dpg.add_separator()
        # ★기록은 **전 축**을 남긴다 — 화면은 한 축만 그려도 로그는 8축 전부다.
        #   시험 중에는 어느 축이 문제인지 모르고, 끝난 뒤에 알게 되기 때문이다.
        with dpg.group(horizontal=True):
            dpg.add_button(label="● 기록 시작", tag="rec_btn", width=130, callback=on_rec)
            dpg.add_text("", tag="rec_info")
        dpg.add_separator()
        dpg.add_text("", tag="readout")
        for key, _mk, _ck, unit, label, _fl in SIG:
            with dpg.plot(label=label, height=185, width=-1, tag=f"plot_{key}"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="t [s]", tag=f"x_{key}")
                dpg.add_plot_axis(dpg.mvYAxis, label=unit, tag=f"y_{key}")
                dpg.add_line_series([], [], label="측정", parent=f"y_{key}", tag=f"s_{key}_m")
                dpg.add_line_series([], [], label="명령", parent=f"y_{key}", tag=f"s_{key}_c")
    if kf:
        dpg.bind_font(kf)

    dpg.create_viewport(title="biped 값 모니터 — 측정 vs 명령", width=880, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("main", True)
    if a.log is not None:                  # --log 를 줬으면 즉시 기록 시작
        rec_start(a.log)
        dpg.set_item_label("rec_btn", "■ 기록 정지")

    # ── 표집 스레드 ────────────────────────────────────────────────────────
    #   ★★왜 스레드인가 (2026-08-13): 종전엔 렌더 루프 안에서 표집했다. 그러면 표본율이
    #     **프레임율에 묶인다** — Pi 에서 3개 플롯을 그리면 ~10fps 라, 제어기가 50Hz 로
    #     발행해도 10Hz 만 잡혔다(실측: writer 50Hz → CSV 10Hz).
    #     ⚠상태파일은 **스트림이 아니라 스냅샷**이다. 덮어써지므로 나중에 몰아 읽을 수 없다
    #       — 놓친 표본은 영영 없다. 그래서 "프레임당 여러 번 읽기" 로는 못 고친다.
    #     ⇒ 표집을 렌더에서 떼어내 별도 스레드가 500Hz 로 폴링한다. 그래프는 느리게 그려도
    #       **기록은 제어기 발행율을 그대로 따라간다.**
    #   ⚠스레드는 dpg 를 절대 건드리지 않는다(dpg 호출은 메인 스레드 전용).
    #     파이썬 자료구조만 만지고, 렌더 쪽과는 lock 으로 나눈다.
    lock = threading.Lock()
    stop_ev = threading.Event()
    err_s = [""]
    age_s = [1e9]

    def sampler():
        while not stop_ev.is_set():
            try:
                mt = os.path.getmtime(STATE)
                age_s[0] = (time.time() - mt) * 1e3
                # ★파일이 갱신됐을 때만 표본을 넣는다. 같은 값을 반복해 넣으면 그래프가
                #   "제어기가 살아 있다" 는 착각을 준다 — STALE 은 곡선이 멈춰야 보인다.
                if mt > last_mtime[0]:
                    st = json.load(open(STATE))
                    with lock:
                        last_mtime[0] = mt
                        if t0[0] is None:
                            t0[0] = mt
                        ts.append(mt - t0[0])
                        for key, mk, ck, _u, _l, _fl in SIG:
                            mv, cv = col(st, mk, nj), col(st, ck, nj)
                            for k2 in range(nj):
                                hist[key][0][k2].append(mv[k2])
                                hist[key][1][k2].append(cv[k2])
                        # ★기록도 신선한 표본에서만 — 그리고 표본당 **한 행**이다.
                        if rec["w"] is not None:
                            if rec["t0"] is None:
                                rec["t0"] = mt
                            rec["w"].writerow(log_row(mt - rec["t0"], mt, st, nj))
                            rec["n"] += 1
                            # 1초마다 flush — 매 행이면 느리고, 안 하면 죽을 때 끝이 날아간다
                            if time.time() - rec["last_flush"] > 1.0:
                                rec["f"].flush()
                                rec["last_flush"] = time.time()
                        st_cache.clear()
                        st_cache.update(st)
                err_s[0] = ""
            except Exception as e:
                err_s[0] = str(e)
            time.sleep(0.002)          # 500Hz 폴링 상한. 실제 표본은 파일 갱신율이 정한다

    threading.Thread(target=sampler, daemon=True).start()

    while dpg.is_dearpygui_running():
        if err_s[0] or not st_cache:
            dpg.set_value("hdr", f"상태파일을 못 읽는다: {STATE}")
            dpg.set_value("flags", f"{err_s[0]}   —   제어기가 떠 있는지: "
                                   f"pgrep -f 'biped_emb.py|biped_deploy'")
            dpg.render_dearpygui_frame()
            continue
        with lock:                     # 스냅샷만 뜨고 바로 놓는다(렌더 중 lock 을 쥐지 않는다)
            st = dict(st_cache)
            tl = list(ts)
            snap = {k: (list(hist[k][0][sel[0]]), list(hist[k][1][sel[0]])) for k, *_ in SIG}
            rec_n, rec_t0, rec_path, rec_on = rec["n"], rec["t0"], rec["path"], rec["f"] is not None
        age_ms = age_s[0]
        mt = last_mtime[0]

        i = sel[0]
        stale = age_ms > STALE_MS
        dpg.set_value("hdr", f"mode={st.get('mode','?'):<6} backend={st.get('backend','?'):<6} "
                             f"loop={st.get('loop_hz',0):6.1f}Hz  tilt={st.get('tilt_deg',0):5.2f}°  "
                             f"age={age_ms:6.1f}ms   [{names[i]}]")
        fl = []
        if stale:
            fl.append(f"*** STALE {age_ms:.0f}ms — 제어기가 발행을 멈췄다 ***")
        if st.get("estop_latched") or st.get("estop"):
            fl.append(f"E-STOP 래치: {st.get('estop_reason','?')}")
        if st.get("wd_trip"):
            fl.append("워치독 트립")
        if st.get("write_fail"):
            fl.append(f"write_fail={st['write_fail']}")
        if st.get("tilt_estop_ok") is False:
            fl.append("IMU 죽음 → tilt E-stop 무력")
        dpg.set_value("flags", "   ".join(fl) if fl else "이상 없음")

        # 곡선 갱신 — None 표본은 건너뛴다(명령이 없는 순수 토크모드 대응).
        #   snap 은 lock 안에서 뜬 스냅샷이다. 여기서 hist 를 직접 읽으면 표집 스레드와
        #   경합해 길이가 어긋난 채 zip 되어 곡선이 시간축에서 밀린다.
        for key, _mk, _ck, _u, _l, floor in SIG:
            for tag, ys in ((f"s_{key}_m", snap[key][0]), (f"s_{key}_c", snap[key][1])):
                xy = [(x, y) for x, y in zip(tl, ys) if y is not None]
                dpg.set_value(tag, [[p[0] for p in xy], [p[1] for p in xy]])
            if tl:
                dpg.set_axis_limits(f"x_{key}", max(0.0, tl[-1] - a.win), tl[-1] + 1e-3)
            # ★y축은 **0 을 항상 포함**하고 **최소폭**을 지킨다(SIG 의 마지막 값).
            #   fit_axis_data 만 쓰면 값이 0 근처일 때 축이 붕괴해 잡음이 산맥이 되고
            #   0 이 화면 밖으로 나가 음/양 감각이 사라진다.
            vs = [v for v in snap[key][0] if v is not None]
            vs += [v for v in snap[key][1] if v is not None]
            lo, hi = (min(vs), max(vs)) if vs else (0.0, 0.0)
            lo, hi = min(lo, 0.0), max(hi, 0.0)
            if hi - lo < floor:
                mid = 0.5 * (lo + hi)
                lo, hi = min(mid - floor / 2, 0.0), max(mid + floor / 2, 0.0)
            pad = 0.08 * (hi - lo)
            dpg.set_axis_limits(f"y_{key}", lo - pad, hi + pad)

        # 숫자 판독 — 그래프는 추세, 숫자는 현재값. 둘 다 필요하다.
        kp = col(st, "kp_leg", nj)[i]
        parts = []
        for key, mk, ck, unit, _l, _fl in SIG:
            m = col(st, mk, nj)[i]
            c = col(st, ck, nj)[i]
            e = None if (m is None or c is None) else m - c
            f = lambda v, p=2: "—" if v is None else f"{v:+.{p}f}"
            parts.append(f"{key}: 측정 {f(m)} / 명령 {f(c)} / Δ {f(e)} {unit}")
        eq = None
        qm, qc = col(st, "q_leg_deg", nj)[i], col(st, "q_cmd_deg", nj)[i]
        if qm is not None and qc is not None and kp is not None:
            eq = (qm - qc) * kp * KP_TO_NM_PER_DEG
        # 기록 상태 — 몇 표본 담겼는지가 보여야 "돌고 있나" 를 안 물어보게 된다.
        if rec_on:
            dur = 0.0 if rec_t0 is None else (mt - rec_t0)
            hz = (rec_n / dur) if dur > 0.5 else 0.0
            dpg.set_value("rec_info", f"● 기록중  {rec_n}표본 · {dur:.1f}s · {hz:.0f}Hz · {rec_path}")
        elif rec_path:
            dpg.set_value("rec_info", f"정지됨 — 저장: {rec_path}")
        else:
            dpg.set_value("rec_info", "미기록 (8축 전부 CSV 로 남는다. 화면은 한 축만 그려도)")

        parts.append("위치오차≈ " + ("—" if eq is None else f"{eq:+.1f} Nm")
                     + f"  (kp {kp if kp is not None else '—'})")
        dpg.set_value("readout", "   │   ".join(parts))

        dpg.render_dearpygui_frame()

    stop_ev.set()       # 표집 스레드 정지 — rec 파일을 닫기 **전에** 세운다(경합 방지)
    time.sleep(0.05)
    rec_stop()          # ★창을 닫아도 파일을 닫는다 — 안 닫으면 마지막 버퍼가 날아간다
    if rec["path"]:
        print(f"[monitor] 기록 저장: {rec['path']}")
    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
