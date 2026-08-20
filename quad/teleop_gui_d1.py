#!/usr/bin/env python3
"""D1(OCS2 NMPC+WBC) 전용 텔레옵 GUI — A GUI(teleop_gui_17dof.py)와 별도 관리.
   ★D1이 실제 읽는 명령만 띄운다: v/vy/w(조이스틱)·gait(trot/walk/bound)·mode(Ready/보행)·reset.
   A-전용(whip·TAMOLS·Jump·Sit·허리핸들·body/step height·ground-pose)은 없음(D1 미소비).
   명령 채널=CMDFILE(/tmp/quad_cmd.json), D1(test02legMujoco/d1_deploy)이 50Hz 소비.
"""
import os, json, math
import dearpygui.dearpygui as dpg

CMD_PATH = os.environ.get('QUAD_CMD', '/tmp/quad_cmd.json')
VMAX, WMAX = 0.6, 0.9

# ── 발행기: D1이 읽는 필드만(v/vy/w/mode/gait/reset_seq/home_seq) ──
class SportClientD1:
    def __init__(self, path=CMD_PATH):
        self.path = path; self.vmax = VMAX; self.wmax = WMAX
        self.cmd = {'v': 0.0, 'vy': 0.0, 'w': 0.0, 'mode': 'stand_up', 'gait': 'trot',
                    'reset_seq': 0, 'home_seq': 0, 'vmax': VMAX}   # 시작=Ready(stance). 보행 눌러야 이동
        self._pub()
    def _pub(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f: json.dump(self.cmd, f)
        os.replace(tmp, self.path)
    def Move(self, vx, vy, vyaw): self.cmd.update(v=vx, vy=vy, w=vyaw); self._pub()
    def StopMove(self): self.cmd.update(v=0.0, vy=0.0, w=0.0); self._pub()
    def SetGait(self, g): self.cmd['gait'] = str(g); self._pub()
    def WalkSpeed(self, v): self.vmax = float(v); self.cmd['vmax'] = float(v); self._pub()
    def SetMode(self, m):
        self.cmd['mode'] = m
        if m != 'move': self.cmd.update(v=0.0, vy=0.0, w=0.0)   # 자세전환=속도0
        self._pub()
    def Ready(self):    # 서기(stance)+홈 복귀(home_seq 상승엣지 → D1 초기포즈+발판 리셋)
        self.cmd['mode'] = 'stand_up'; self.cmd.update(v=0.0, vy=0.0, w=0.0)
        self.cmd['home_seq'] = int(self.cmd.get('home_seq', 0)) + 1; self._pub()
    def Reset(self):    # 리셋(넘어짐 복구): reset_seq 상승엣지 → D1 초기포즈+발판 리셋
        self.cmd['reset_seq'] = int(self.cmd.get('reset_seq', 0)) + 1
        self.cmd.update(v=0.0, vy=0.0, w=0.0, mode='stand_up'); self._pub()


sc = SportClientD1()


# ── 가상 조이스틱(teleop_gui_17dof JoyPad 동일) ──
class JoyPad:
    def __init__(self, tag, size, on_change, x_only=False):
        self.tag = tag; self.sz = size; self.R = size * 0.5 - 16; self.c = size / 2
        self.on_change = on_change; self.x_only = x_only; self.active = False; self.latched = False
    def build(self):
        with dpg.drawlist(width=self.sz, height=self.sz, tag=self.tag):
            dpg.draw_circle([self.c, self.c], self.R, color=(80, 90, 120), fill=(28, 30, 42), thickness=2)
            dpg.draw_line([self.c - self.R, self.c], [self.c + self.R, self.c], color=(58, 62, 84))
            dpg.draw_line([self.c, self.c - self.R], [self.c, self.c + self.R], color=(58, 62, 84))
            dpg.draw_circle([self.c, self.c], self.R * 0.5, color=(52, 56, 76))
            dpg.draw_circle([self.c, self.c], self.sz * 0.15, color=(250, 195, 75), fill=(238, 178, 58), tag=self.tag + '_k')
    def _loc(self):
        m = dpg.get_mouse_pos(local=False); r = dpg.get_item_rect_min(self.tag)
        return m[0] - r[0], m[1] - r[1]
    def press(self):
        if dpg.is_item_hovered(self.tag): self.active = True; self.move()
    def move(self):
        if not self.active: return
        lx, ly = self._loc(); dx = lx - self.c; dy = ly - self.c; d = math.hypot(dx, dy)
        if d > self.R and d > 0: dx *= self.R / d; dy *= self.R / d
        if self.x_only: dy = 0
        dpg.configure_item(self.tag + '_k', center=[self.c + dx, self.c + dy])
        self.on_change(dx / self.R, -dy / self.R)
    def release(self):
        if self.active:
            self.active = False
            if not self.latched:
                dpg.configure_item(self.tag + '_k', center=[self.c, self.c]); self.on_change(0.0, 0.0)
    def toggle_latch(self):
        if not dpg.is_item_hovered(self.tag): return
        self.latched = not self.latched
        dpg.configure_item(self.tag + '_k', fill=(95, 210, 120) if self.latched else (238, 178, 58))
        if not self.latched:
            dpg.configure_item(self.tag + '_k', center=[self.c, self.c]); self.on_change(0.0, 0.0)
    def clear_latch(self):
        self.latched = False; self.active = False
        dpg.configure_item(self.tag + '_k', center=[self.c, self.c], fill=(238, 178, 58)); self.on_change(0.0, 0.0)


_last_left = [0.0, 0.0]; _last_right = [0.0, 0.0]

def _status():
    dpg.set_value('stat', 'v=%.2f  vy=%.2f  w=%.2f  |  mode=%s  gait=%s'
                  % (sc.cmd['v'], sc.cmd['vy'], sc.cmd['w'], sc.cmd['mode'], sc.cmd['gait']))

def _left(ax, ay):                       # 좌스틱: 전후(ay)/측방(ax) — 풀스케일=속도게이지
    _last_left[:] = [ax, ay]; sc.Move(ay * sc.vmax, -ax * sc.vmax, sc.cmd['w']); _status()

def _right(ax, ay):                      # 우스틱: 선회(ax)
    _last_right[:] = [ax, ay]; sc.Move(sc.cmd['v'], sc.cmd['vy'], -ax * sc.wmax); _status()

def _set_walk_speed(v):
    sc.WalkSpeed(v); _left(_last_left[0], _last_left[1])

# ★D1 게이트 프리셋: 조이스틱 풀스케일(gait별 D1 안정 상한). walk<trot<bound(bound가 D1선 최고속).
def _gait_preset(gait):
    vmax = {'walk': 0.4, 'trot': 0.6, 'bound': 0.8}.get(gait, 0.6)
    if dpg.does_item_exist('ws'): dpg.set_value('ws', vmax)
    _set_walk_speed(vmax)

def _mode_btn(m): sc.SetMode(m); _status()

left = JoyPad('joyL', 200, _left)
right = JoyPad('joyR', 200, _right, x_only=True)

def _key(sender, app_data):              # 키보드 백업: 화살표=이동, ,/. =선회, X=STOP
    k = app_data; s = 0.05
    if k == dpg.mvKey_Up:      sc.Move(min(sc.vmax, sc.cmd['v'] + s), sc.cmd['vy'], sc.cmd['w'])
    elif k == dpg.mvKey_Down:  sc.Move(max(-sc.vmax, sc.cmd['v'] - s), sc.cmd['vy'], sc.cmd['w'])
    elif k == dpg.mvKey_Left:  sc.Move(sc.cmd['v'], min(sc.vmax, sc.cmd['vy'] + s), sc.cmd['w'])
    elif k == dpg.mvKey_Right: sc.Move(sc.cmd['v'], max(-sc.vmax, sc.cmd['vy'] - s), sc.cmd['w'])
    elif k == dpg.mvKey_Comma:  sc.Move(sc.cmd['v'], sc.cmd['vy'], min(sc.wmax, sc.cmd['w'] + s))
    elif k == dpg.mvKey_Period: sc.Move(sc.cmd['v'], sc.cmd['vy'], max(-sc.wmax, sc.cmd['w'] - s))
    elif k in (dpg.mvKey_X, dpg.mvKey_Spacebar): sc.StopMove(); left.clear_latch(); right.clear_latch()
    else: return
    _status()


dpg.create_context()
# 한글 폰트(없으면 ??로 표기) — A GUI와 동일
_FONT = os.environ.get('GUI_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
_kf = None
if os.path.exists(_FONT):
    with dpg.font_registry():
        _kf = dpg.add_font(_FONT, 18)
with dpg.theme() as _dark:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (24, 26, 34))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 224, 235))
with dpg.theme() as _stop_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 45, 45))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 65, 65))
with dpg.theme() as _walk_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 70))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 160, 95))

with dpg.window(tag='main'):
    dpg.add_text('D1 (OCS2 NMPC+WBC) Teleop', color=(120, 200, 255))
    dpg.add_text('★D1 배선 명령만: 조이스틱 v/vy/w · 게이트 trot/walk/bound · 모드 Ready/보행 · Reset', color=(150, 155, 175))
    dpg.add_separator()
    with dpg.group(horizontal=True):
        with dpg.group():
            dpg.add_text('이동  (좌스틱: ↕전후 / ↔측방)'); left.build()
        dpg.add_spacer(width=20)
        with dpg.group():
            dpg.add_text('선회  (우스틱: ↔좌우)'); right.build()
    dpg.add_text('★우클릭=조이스틱 고정(초록). 전진 유지하며 조향. 재우클릭/X=해제', color=(250, 195, 75))
    dpg.add_separator()
    dpg.add_text('모션', color=(170, 175, 195))
    with dpg.group(horizontal=True):
        _b = dpg.add_button(label='RESET', width=90, callback=lambda: (sc.Reset(), _status()))
        dpg.bind_item_theme(_b, _stop_theme)
        dpg.add_button(label='Ready 서기(정지)', width=140, callback=lambda: (sc.Ready(), _status()))
        _wb = dpg.add_button(label='▶ 보행(이동)', width=120, callback=lambda: _mode_btn('move'))
        dpg.bind_item_theme(_wb, _walk_theme)
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text('게이트:', color=(170, 175, 195))
        dpg.add_button(label='trot 대각', width=95,
                       callback=lambda: (sc.SetGait('trot'), _gait_preset('trot'), _status()))
        dpg.add_button(label='walk 순차', width=95,
                       callback=lambda: (sc.SetGait('walk'), _gait_preset('walk'), _status()))
        dpg.add_button(label='bound 바운드', width=110,
                       callback=lambda: (sc.SetGait('bound'), _gait_preset('bound'), _status()))
        dpg.add_text('(버튼=속도게이지 자동세팅. walk 0.4 / trot 0.6 / bound 0.8)', color=(120, 125, 145))
    dpg.add_separator()
    dpg.add_text('속도', color=(170, 175, 195))
    dpg.add_slider_float(label='Walk Speed [m/s]  (조이스틱 풀스케일)', tag='ws',
                         min_value=0.0, max_value=1.2, default_value=VMAX,
                         callback=lambda s, a: _set_walk_speed(a))
    dpg.add_separator()
    dpg.add_text('', tag='stat', color=(180, 210, 160))

with dpg.handler_registry():
    dpg.add_key_press_handler(callback=_key)
    dpg.add_mouse_down_handler(button=dpg.mvMouseButton_Left, callback=lambda s, a: (left.press(), right.press()))
    dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=lambda s, a: (left.move(), right.move()))
    dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=lambda s, a: (left.release(), right.release()))
    dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=lambda s, a: (left.toggle_latch(), right.toggle_latch()))

_status()
dpg.create_viewport(title='D1 Teleop (OCS2 NMPC+WBC)', width=480, height=560)
dpg.setup_dearpygui()
if _kf is not None:
    dpg.bind_font(_kf)
dpg.bind_theme(_dark)
dpg.show_viewport()
dpg.set_primary_window('main', True)
dpg.start_dearpygui()
dpg.destroy_context()
