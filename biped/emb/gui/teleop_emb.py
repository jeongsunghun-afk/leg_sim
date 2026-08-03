"""gui/teleop_emb.py — biped 실기(Emb) 텔레옵 GUI. dearpygui.

★핵심 = per-axis JOG 패널: 8관절 슬라이더로 축별 목표각(deg) 발행 → 각축이 잘 움직이는지 검증.
   실시간 측정각(state.q_leg_deg)을 옆에 표시 → 명령 vs 실제 비교로 부호·오프셋·한계 확인.
모드 버튼: Off(limp) / JOG / Hold / Stand / Walk / RESET.  Walk 는 조이스틱(v/vy/w)+몸통높이.
채널·jog 한계는 config(biped_emb.yaml)에서 로드 → app 과 동일 계약.

명령 채널: /tmp/biped_cmd.json  {mode, jog_deg[8], v, vy, w, body_h}
상태 채널: /tmp/biped_state.json {mode, q_leg_deg[8], rpy_deg, tilt_deg, loop_hz, motors_on, backend}
실행(proxddp env): /home/jsh/miniforge3/envs/proxddp/bin/python gui/teleop_emb.py
"""
import os, json
import numpy as np
import dearpygui.dearpygui as dpg
import yaml

CMD    = os.environ.get("QUAD_CMD",   "/tmp/biped_cmd.json")
STATE  = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")
HERE   = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.environ.get("CONFIG", os.path.join(os.path.dirname(HERE), "config", "biped_emb.yaml"))

with open(CONFIG) as f:
    cfg = yaml.safe_load(f)
JOINTS = cfg["joints"]
NAMES  = [j["name"] for j in JOINTS]
NJ     = len(JOINTS)
FRAC   = float(cfg.get("jog", {}).get("range_frac", 0.5))
JMIN   = [j["min_deg"] * FRAC for j in JOINTS]      # jog 안전 범위(중립 근처)
JMAX   = [j["max_deg"] * FRAC for j in JOINTS]

VMAX   = float(os.environ.get("VMAX", "0.15"))
VY_MAX = float(os.environ.get("VY_MAX", "0.10"))
WZ_MAX = float(os.environ.get("WZ_MAX", "0.30"))
H_MIN, H_MAX, H_DEF = 0.36, 0.54, 0.42


class Pub:
    def __init__(self, path=CMD):
        self.path = path
        self.cmd = {"mode": "off", "jog_deg": [0.0] * NJ,
                    "v": 0.0, "vy": 0.0, "w": 0.0, "body_h": H_DEF}
        self._pub()

    def set(self, **kw):
        self.cmd.update(kw); self._pub()

    def set_jog(self, i, val):
        self.cmd["jog_deg"][i] = float(val); self._pub()

    def _pub(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cmd, f)
        os.replace(tmp, self.path)


class JoyPad:
    """가상 조이스틱(walk 전용). 좌드래그=축[-1,1]·놓으면 복귀."""
    def __init__(self, tag, size, on_change, x_only=False):
        self.tag = tag; self.sz = size; self.R = size * 0.5 - 16; self.c = size / 2
        self.on_change = on_change; self.x_only = x_only; self.active = False

    def build(self, label=""):
        with dpg.drawlist(width=self.sz, height=self.sz, tag=self.tag):
            dpg.draw_circle([self.c, self.c], self.R, color=(80, 90, 120), fill=(28, 30, 42), thickness=2)
            dpg.draw_line([self.c - self.R, self.c], [self.c + self.R, self.c], color=(58, 62, 84))
            dpg.draw_line([self.c, self.c - self.R], [self.c, self.c + self.R], color=(58, 62, 84))
            if label:
                dpg.draw_text([8, 6], label, color=(120, 140, 170), size=14)
            dpg.draw_circle([self.c, self.c], self.sz * 0.15, color=(250, 195, 75),
                            fill=(238, 178, 58), tag=self.tag + "_k")

    def _loc(self):
        m = dpg.get_mouse_pos(local=False); r = dpg.get_item_rect_min(self.tag)
        return m[0] - r[0], m[1] - r[1]

    def press(self):
        if dpg.is_item_hovered(self.tag):
            self.active = True; self.move()

    def move(self):
        if not self.active:
            return
        lx, ly = self._loc(); dx = max(-self.R, min(self.R, lx - self.c)); dy = max(-self.R, min(self.R, ly - self.c))
        if self.x_only:
            dy = 0
        dpg.configure_item(self.tag + "_k", center=[self.c + dx, self.c + dy])
        self.on_change(dx / self.R, -dy / self.R)

    def release(self):
        if self.active:
            self.active = False
            dpg.configure_item(self.tag + "_k", center=[self.c, self.c]); self.on_change(0.0, 0.0)

    def clear(self):
        self.active = False
        dpg.configure_item(self.tag + "_k", center=[self.c, self.c]); self.on_change(0.0, 0.0)


pub = Pub()
_expo = lambda a: a * abs(a)


def on_left(ax, ay):
    pub.set(v=round(_expo(ay) * VMAX, 3), vy=round(_expo(-ax) * VY_MAX, 3))


def on_right(ax, _ay):
    pub.set(w=round(_expo(-ax) * WZ_MAX, 3))


def on_height(_, val):
    pub.set(body_h=round(val, 3))


def on_jog(sender, val, i):
    pub.set_jog(i, val)


def jog_zero():
    for i in range(NJ):
        dpg.set_value(f"jog_{i}", 0.0)
    pub.set(jog_deg=[0.0] * NJ)


def set_mode(mode):
    if mode == "reset":
        left.clear(); right.clear(); pub.set(mode="reset", v=0.0, vy=0.0, w=0.0)
        return
    if mode == "jog":                       # jog 진입 = 슬라이더를 현재 측정각으로 맞춰 점프 방지
        try:
            with open(STATE) as f:
                q = json.load(f).get("q_leg_deg", [0.0] * NJ)
            for i in range(NJ):
                v = float(np.clip(q[i], JMIN[i], JMAX[i]))
                dpg.set_value(f"jog_{i}", v); pub.cmd["jog_deg"][i] = v
        except Exception:
            pass
    if mode in ("stand", "walk", "off", "hold"):
        left.clear(); right.clear(); pub.set(v=0.0, vy=0.0, w=0.0)
    pub.set(mode=mode)


left  = JoyPad("joyL", 150, on_left)
right = JoyPad("joyR", 150, on_right, x_only=True)

dpg.create_context()
_FONT = os.environ.get("GUI_FONT", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_kf = None
if os.path.exists(_FONT):
    with dpg.font_registry():
        _kf = dpg.add_font(_FONT, 17)
with dpg.theme() as _dark:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (22, 24, 32))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (46, 52, 74))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (66, 74, 104))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 224, 235))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
with dpg.theme() as _stop:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 45, 45))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 65, 65))
with dpg.theme() as _go:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 70))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 160, 95))

with dpg.window(tag="main"):
    dpg.add_text("biped EMB teleop  —  각축 검증(JOG) + 모델기반(Stand/Walk)", color=(150, 200, 255))
    dpg.add_separator()
    dpg.add_text("모드", color=(170, 175, 195))
    with dpg.group(horizontal=True):
        _o = dpg.add_button(label="Off 전원", width=90, callback=lambda: set_mode("off")); dpg.bind_item_theme(_o, _stop)
        dpg.add_button(label="JOG 검증", width=90, callback=lambda: set_mode("jog"))
        dpg.add_button(label="Hold", width=70, callback=lambda: set_mode("hold"))
        dpg.add_button(label="Stand 서기", width=100, callback=lambda: set_mode("stand"))
        _w = dpg.add_button(label="Walk 이동", width=100, callback=lambda: set_mode("walk")); dpg.bind_item_theme(_w, _go)
        _r = dpg.add_button(label="RESET", width=80, callback=lambda: set_mode("reset")); dpg.bind_item_theme(_r, _stop)
    dpg.add_text("Off=limp(토크0) · JOG=축별 저속 위치검증 · Stand/Walk=모델기반(jog 검증 후)",
                 color=(150, 155, 175))
    dpg.add_separator()

    dpg.add_text("● JOG — 각축 검증 (슬라이더=목표각° · 실측° · ●=상태LED)", color=(255, 205, 120))
    with dpg.group(horizontal=True):
        dpg.add_button(label="모두 0 (home)", width=110, callback=jog_zero)
        dpg.add_text("LED: 초록=정상 · 노랑=에러(ucStatus) · 회색=무통신(죽음). 임베디드 보고 반영",
                     color=(120, 130, 150))
    LED_R = 7
    for i, nm in enumerate(NAMES):
        with dpg.group(horizontal=True):
            with dpg.drawlist(width=2 * LED_R + 6, height=2 * LED_R + 6, tag=f"leddl_{i}"):
                dpg.draw_circle([LED_R + 3, LED_R + 3], LED_R, fill=(70, 70, 78),
                                color=(30, 30, 36), tag=f"led_{i}")   # 기본 회색(무통신)
            dpg.add_text(f"{nm:9s}", color=(190, 195, 210))
            dpg.add_slider_float(tag=f"jog_{i}", default_value=0.0, min_value=JMIN[i], max_value=JMAX[i],
                                 width=250, format="%.1f", user_data=i,
                                 callback=lambda s, v, u: on_jog(s, v, u))
            dpg.add_text("--.-", tag=f"meas_{i}", color=(150, 220, 150))
    dpg.add_separator()

    dpg.add_text("● WALK — 이동 명령(모델기반)", color=(150, 220, 255))
    with dpg.group(horizontal=True):
        with dpg.group():
            left.build("전후/측방"); dpg.add_text("좌스틱: 위=전진 좌우=측방", color=(120, 130, 150))
        with dpg.group():
            right.build("선회"); dpg.add_text("우스틱: 좌우=선회", color=(120, 130, 150))
        with dpg.group():
            dpg.add_text("몸통 높이 [m]")
            dpg.add_slider_float(tag="h_sl", default_value=H_DEF, min_value=H_MIN, max_value=H_MAX,
                                 width=180, callback=on_height)
    dpg.add_separator()
    dpg.add_text("-", tag="state", color=(150, 220, 150))

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=lambda: (left.press(), right.press()))
    dpg.add_mouse_drag_handler(callback=lambda: (left.move(), right.move()))
    dpg.add_mouse_release_handler(callback=lambda: (left.release(), right.release()))

dpg.bind_theme(_dark)
if _kf is not None:
    dpg.bind_font(_kf)
dpg.create_viewport(title="biped EMB teleop", width=760, height=640)
dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window("main", True)

while dpg.is_dearpygui_running():
    try:
        with open(STATE) as f:
            st = json.load(f)
        q = st.get("q_leg_deg", [0.0] * NJ)
        health = st.get("health", ["dead"] * NJ)
        LED = {"ok": (60, 210, 90), "fault": (235, 200, 60), "dead": (70, 70, 78)}
        for i in range(min(NJ, len(q))):
            dpg.set_value(f"meas_{i}", f"{q[i]:+6.1f}")
        for i in range(min(NJ, len(health))):      # LED: 초록=정상 · 노랑=에러 · 회색=무통신
            dpg.configure_item(f"led_{i}", fill=LED.get(health[i], (70, 70, 78)))
        line = ("mode=%s  backend=%s  정상 %d / 에러 %d / 두절 %d / %d  tilt=%.1f°  loop=%.0fHz"
                % (st.get("mode", "-"), st.get("backend", "-"),
                   st.get("n_ok", 0), st.get("n_fault", 0), st.get("n_dead", NJ), NJ,
                   st.get("tilt_deg", 0), st.get("loop_hz", 0)))
        if "est_z" in st:
            line += "   est_z=%.3f est_x=%+.2f" % (st["est_z"], st.get("est_x", 0))
        dpg.set_value("state", line)
    except Exception:
        dpg.set_value("state", "(app/biped_emb.py 대기중…)")
    dpg.render_dearpygui_frame()

dpg.destroy_context()
