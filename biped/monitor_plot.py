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
import json
import os
import time
from collections import deque

import dearpygui.dearpygui as dpg

STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")
FALLBACK_NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot",
                  "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
KP_TO_NM_PER_DEG = 0.0175          # [실측 2026-08-05] 드라이버 MIT: kp 1 = 0.0175 Nm/deg
STALE_MS = 500.0

# 계열 3종 × (측정 키, 명령 키, 단위, 플롯 라벨)
SIG = [("q",   "q_leg_deg",   "q_cmd_deg",   "deg",   "위치 [deg]"),
       ("dq",  "dq_leg_dps",  "dq_cmd_dps",  "deg/s", "속도 [deg/s]"),
       ("tau", "tau_leg_nm",  "tau_cmd_nm",  "Nm",    "토크 [Nm]")]


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

    with dpg.window(tag="main"):
        dpg.add_text("", tag="hdr")
        dpg.add_text("", tag="flags", color=(255, 170, 60))
        dpg.add_separator()
        dpg.add_text("축 선택 — 고른 축의 위치·속도·토크만 그린다 (이력은 전 축 유지)")
        dpg.add_radio_button(names, tag="sel", default_value=names[0],
                             horizontal=True, callback=on_sel)
        dpg.add_separator()
        dpg.add_text("", tag="readout")
        for key, _mk, _ck, unit, label in SIG:
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

    # ★표본율(--hz)은 **버퍼 길이**에만 쓴다. 실제 표본은 상태파일이 갱신될 때만 들어가므로
    #   제어기 발행율(기본 50Hz)이 상한이다. 렌더는 dearpygui 의 vsync 에 맡긴다.
    while dpg.is_dearpygui_running():
        try:
            mt = os.path.getmtime(STATE)
            age_ms = (time.time() - mt) * 1e3
            # ★파일이 갱신됐을 때만 표본을 넣는다. 같은 값을 반복해 넣으면 그래프가
            #   "제어기가 살아 있다" 는 착각을 준다 — STALE 은 곡선이 멈춰야 보인다.
            fresh = mt > last_mtime[0]
            if fresh:
                last_mtime[0] = mt
                st = json.load(open(STATE))
                if t0[0] is None:
                    t0[0] = mt
                ts.append(mt - t0[0])
                for key, mk, ck, _u, _l in SIG:
                    mv, cv = col(st, mk, nj), col(st, ck, nj)
                    for i in range(nj):
                        hist[key][0][i].append(mv[i])
                        hist[key][1][i].append(cv[i])
                st_cache.clear(); st_cache.update(st)
            st = st_cache
        except Exception as e:
            dpg.set_value("hdr", f"상태파일을 못 읽는다: {STATE}")
            dpg.set_value("flags", f"{e}   —   제어기가 떠 있는지: pgrep -f 'biped_emb.py|biped_deploy'")
            dpg.render_dearpygui_frame()
            continue

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

        # 곡선 갱신 — None 표본은 건너뛴다(명령이 없는 순수 토크모드 대응)
        tl = list(ts)
        for key, _mk, _ck, _u, _l in SIG:
            for tag, buf in ((f"s_{key}_m", hist[key][0][i]), (f"s_{key}_c", hist[key][1][i])):
                ys = list(buf)
                xy = [(x, y) for x, y in zip(tl, ys) if y is not None]
                dpg.set_value(tag, [[p[0] for p in xy], [p[1] for p in xy]])
            if tl:
                dpg.set_axis_limits(f"x_{key}", max(0.0, tl[-1] - a.win), tl[-1] + 1e-3)
            dpg.fit_axis_data(f"y_{key}")

        # 숫자 판독 — 그래프는 추세, 숫자는 현재값. 둘 다 필요하다.
        kp = col(st, "kp_leg", nj)[i]
        parts = []
        for key, mk, ck, unit, _l in SIG:
            m = col(st, mk, nj)[i]
            c = col(st, ck, nj)[i]
            e = None if (m is None or c is None) else m - c
            f = lambda v, p=2: "—" if v is None else f"{v:+.{p}f}"
            parts.append(f"{key}: 측정 {f(m)} / 명령 {f(c)} / Δ {f(e)} {unit}")
        eq = None
        qm, qc = col(st, "q_leg_deg", nj)[i], col(st, "q_cmd_deg", nj)[i]
        if qm is not None and qc is not None and kp is not None:
            eq = (qm - qc) * kp * KP_TO_NM_PER_DEG
        parts.append("위치오차≈ " + ("—" if eq is None else f"{eq:+.1f} Nm")
                     + f"  (kp {kp if kp is not None else '—'})")
        dpg.set_value("readout", "   │   ".join(parts))

        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
