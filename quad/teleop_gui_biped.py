"""biped 전용 슬림 GUI (17-DOF GUI 규약) — dearpygui.

명령을 JSON 채널(/tmp/biped_cmd.json)로 발행 → biped_run.py 소비.
★듀얼 조이스틱: 좌스틱=전후(vx)/측방(vy) · 우스틱=선회(wz). 버튼 Stand/Walk/RESET.
실행(proxddp env): /home/jsh/miniforge3/envs/proxddp/bin/python teleop_gui_biped.py
  ① 컨트롤러: python biped_run.py   ② GUI: 위 명령
"""
import os, json, math
import dearpygui.dearpygui as dpg

CMD    = os.environ.get('QUAD_CMD',   '/tmp/biped_cmd.json')
STATE  = os.environ.get('QUAD_STATE', '/tmp/biped_state.json')
VMAX   = 0.15         # 전진 상한[m/s] (★안전 하향 0.2→0.15=로버스트 범위. 0.20은 marginal)
VY_MAX = 0.10         # 좌우 상한[m/s] (★body-frame 게이트 수정 후 vy 0.12까지 안정→0.10 캡. 십자라 순수 측방)
WZ_MAX = 0.30         # 선회 상한[rad/s] (★제자리 0.4·주행중 0.3 안정(body-frame 수정 후). turn rate head_lead로 ~2.5°/s 포화)
H_MIN, H_MAX, H_DEF = 0.36, 0.52, 0.50
H_DEF_1PT, H_DEF_2PT = 0.50, 0.42       # ★접촉모드별 기본 몸통높이(점발/평발)


class Pub:
    def __init__(self, path=CMD):
        self.path = path
        self.cmd = {'v': 0.0, 'vy': 0.0, 'w': 0.0, 'body_h': H_DEF, 'mode': 'stand', 'contact': '1pt'}
        self._pub()

    def set(self, **kw):
        self.cmd.update(kw); self._pub()

    def _pub(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.cmd, f)
        os.replace(tmp, self.path)


class JoyPad:
    """가상 조이스틱(17-DOF GUI 참조). 좌드래그=축[-1,1]·놓으면 복귀·우클릭=고정."""
    def __init__(self, tag, size, on_change, x_only=False, cross_only=False):
        self.tag = tag; self.sz = size; self.R = size * 0.5 - 16; self.c = size / 2
        self.on_change = on_change; self.x_only = x_only; self.cross_only = cross_only
        self.active = False; self.latched = False

    def build(self, label=''):
        with dpg.drawlist(width=self.sz, height=self.sz, tag=self.tag):
            dpg.draw_circle([self.c, self.c], self.R, color=(80, 90, 120), fill=(28, 30, 42), thickness=2)
            dpg.draw_line([self.c - self.R, self.c], [self.c + self.R, self.c], color=(58, 62, 84))
            dpg.draw_line([self.c, self.c - self.R], [self.c, self.c + self.R], color=(58, 62, 84))
            if label:
                dpg.draw_text([8, 6], label, color=(120, 140, 170), size=14)
            dpg.draw_circle([self.c, self.c], self.sz * 0.15, color=(250, 195, 75),
                            fill=(238, 178, 58), tag=self.tag + '_k')

    def _loc(self):
        m = dpg.get_mouse_pos(local=False); r = dpg.get_item_rect_min(self.tag)
        return m[0] - r[0], m[1] - r[1]

    def press(self):
        if dpg.is_item_hovered(self.tag):
            self.active = True; self.move()

    def move(self):
        if not self.active:
            return
        lx, ly = self._loc(); dx = lx - self.c; dy = ly - self.c
        dx = max(-self.R, min(self.R, dx)); dy = max(-self.R, min(self.R, dy))
        if self.x_only:
            dy = 0
        if self.cross_only:                                 # ★십자(4-way): 우세 축만 = vx·vy 동시 금지(대각 marginal 방지)
            if abs(dx) >= abs(dy): dy = 0
            else: dx = 0
        dpg.configure_item(self.tag + '_k', center=[self.c + dx, self.c + dy])
        self.on_change(dx / self.R, -dy / self.R)           # ax 우=+, ay 위=+

    def release(self):
        if self.active:
            self.active = False
            if not self.latched:
                dpg.configure_item(self.tag + '_k', center=[self.c, self.c]); self.on_change(0.0, 0.0)

    def toggle_latch(self):
        if not dpg.is_item_hovered(self.tag):
            return
        self.latched = not self.latched
        dpg.configure_item(self.tag + '_k', fill=(95, 210, 120) if self.latched else (238, 178, 58))
        if not self.latched:
            dpg.configure_item(self.tag + '_k', center=[self.c, self.c]); self.on_change(0.0, 0.0)

    def clear(self):
        self.latched = False; self.active = False
        dpg.configure_item(self.tag + '_k', center=[self.c, self.c], fill=(238, 178, 58))
        self.on_change(0.0, 0.0)


pub = Pub()
_expo = lambda a: a * abs(a)          # 중앙 미세·끝 최대


def on_left(ax, ay):                  # 좌스틱: 전후(ay=vx) · 측방(ax=vy)
    v = round(_expo(ay) * VMAX, 3)
    vy = round(_expo(-ax) * VY_MAX, 3)   # 스틱 좌 = +vy(좌)
    pub.set(v=v, vy=vy)
    dpg.set_value('spd_sl', v); dpg.set_value('vy_sl', vy)


def on_right(ax, _ay):                # 우스틱: 선회(ax=wz)
    w = round(_expo(-ax) * WZ_MAX, 3)    # 스틱 우 = 우선회(−wz)
    pub.set(w=w); dpg.set_value('turn_sl', w)


def on_vx(_, val): pub.set(v=round(val, 3))
def on_vy(_, val): pub.set(vy=round(val, 3))
def on_turn(_, val): pub.set(w=round(val, 3))
def on_height(_, val): pub.set(body_h=round(val, 3))


def set_contact(contact):
    """★접촉모드 전환: 1pt=점발(동적보행)·2pt=평발(정적 양발지지). 모델 리로드 + 기본높이 세팅."""
    left.clear(); right.clear()
    h = H_DEF_2PT if contact == '2pt' else H_DEF_1PT
    pub.set(contact=contact, mode='stand', body_h=h, v=0.0, vy=0.0, w=0.0)
    dpg.set_value('h_sl', h)
    dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)
    dpg.set_value('contact_lbl', '접촉: ' + ('2점 평발(정적 서기 전용 — 걸으려면 1점)' if contact == '2pt'
                  else '1점 점발(동적 보행)')
                  + '  · 전환=목표자세 재정착 (C++ 배포경로 run_gui_cpp)')


def set_mode(mode):
    if mode == 'reset':
        left.clear(); right.clear()
        pub.set(mode='reset', v=0.0, vy=0.0, w=0.0)
        dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)
        return
    pub.set(mode=mode)
    if mode != 'walk':
        left.clear(); right.clear(); pub.set(v=0.0, vy=0.0, w=0.0)
        dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)


left  = JoyPad('joyL', 190, on_left, cross_only=True)   # ★십자만(전후 XOR 측방, 대각 금지)
right = JoyPad('joyR', 190, on_right, x_only=True)

dpg.create_context()
_FONT = os.environ.get('GUI_FONT', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
_kf = None
if os.path.exists(_FONT):
    with dpg.font_registry():
        _kf = dpg.add_font(_FONT, 18)
with dpg.theme() as _dark:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (22, 24, 32))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (46, 52, 74))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (66, 74, 104))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 224, 235))
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
with dpg.theme() as _stop:                 # RESET·Off 전원 = 빨강(17-DOF 규약)
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (170, 45, 45))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (210, 65, 65))
with dpg.theme() as _walk:                  # Walk 이동 = 초록
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 70))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 160, 95))

with dpg.window(tag='main'):
    dpg.add_text('biped teleop  —  MPC + WBIC (event-DCM)', color=(150, 200, 255))
    dpg.add_separator()
    with dpg.group(horizontal=True):
        with dpg.group():
            left.build('전후/측방')
            dpg.add_text('좌: 위아래=전후 · 좌우=측방 (십자=하나씩)', color=(120, 130, 150))
        with dpg.group():
            right.build('선회')
            dpg.add_text('우: 좌우=선회 (★점발 한계로 매우 약함 ~1-2°/s·측방도 60%)', color=(120, 130, 150))
        with dpg.group():
            dpg.add_text('vx [m/s]')
            dpg.add_slider_float(tag='spd_sl', default_value=0.0, min_value=-VMAX, max_value=VMAX, width=180, callback=on_vx)
            dpg.add_text('vy [m/s] (측방)')
            dpg.add_slider_float(tag='vy_sl', default_value=0.0, min_value=-VY_MAX, max_value=VY_MAX, width=180, callback=on_vy)
            dpg.add_text('wz [rad/s] (선회)')
            dpg.add_slider_float(tag='turn_sl', default_value=0.0, min_value=-WZ_MAX, max_value=WZ_MAX, width=180, callback=on_turn)
            dpg.add_text('몸통 높이 [m]')
            dpg.add_slider_float(tag='h_sl', default_value=H_DEF, min_value=H_MIN, max_value=H_MAX, width=180, callback=on_height)
    dpg.add_spacer(height=8)
    dpg.add_text('접촉 모드', color=(170, 175, 195))
    with dpg.group(horizontal=True):   # ★1점/2점 접촉모드(모델 리로드)
        _c1 = dpg.add_button(label='1점 점발', width=110, callback=lambda: set_contact('1pt'))
        _c2 = dpg.add_button(label='2점 평발', width=110, callback=lambda: set_contact('2pt'))
        dpg.bind_item_theme(_c2, _walk)
    dpg.add_text('접촉: 1점 점발(동적보행)', tag='contact_lbl', color=(150, 155, 175))
    dpg.add_spacer(height=8)
    dpg.add_text('모션', color=(170, 175, 195))
    with dpg.group(horizontal=True):   # ★버튼 순서(17-DOF 규약): RESET → Off전원 → Stand서기 → Walk이동
        _rb = dpg.add_button(label='RESET', width=90, callback=lambda: set_mode('reset'))
        dpg.bind_item_theme(_rb, _stop)
        _ob = dpg.add_button(label='Off 전원', width=100, callback=lambda: set_mode('off'))
        dpg.bind_item_theme(_ob, _stop)
        dpg.add_button(label='Stand 서기', width=110, callback=lambda: set_mode('stand'))
        _wb = dpg.add_button(label='Walk 이동', width=110, callback=lambda: set_mode('walk'))
        dpg.bind_item_theme(_wb, _walk)
    dpg.add_text('복구 순서: 전원(Off) → 서기(Stand) → 이동(Walk)   · Off=모터 토크차단(limp), 실HW=motor disable',
                 color=(150, 155, 175))
    dpg.add_separator()
    dpg.add_text('-', tag='state', color=(150, 220, 150))

with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=lambda: (left.press(), right.press()))
    dpg.add_mouse_drag_handler(callback=lambda: (left.move(), right.move()))
    dpg.add_mouse_release_handler(callback=lambda: (left.release(), right.release()))
    dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=lambda: (left.toggle_latch(), right.toggle_latch()))

dpg.bind_theme(_dark)
if _kf is not None:
    dpg.bind_font(_kf)
dpg.create_viewport(title='biped teleop', width=680, height=430)
dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window('main', True)

while dpg.is_dearpygui_running():
    try:
        with open(STATE) as f:
            st = json.load(f)
        line = ('mode=%s  높이%.2f  vx%+.2f vy%+.2f wz%+.2f  yaw%+.0f°  tilt%.1f°  (%+.1f,%+.1f)'
                % (st['mode'], st['base_z'], st['vx_cmd'], pub.cmd['vy'], st.get('wz_cmd', 0),
                   st.get('yaw', 0), st['tilt'], st['x'], st.get('y', 0)))
        if 'est_perr' in st:                       # biped_deploy 실행 시 = leg-odometry 추정오차(GT 대비)
            line += '\n추정(leg-odom) 오차: pos %.1fcm  vel %.3fm/s   EST(%+.2f,%+.2f)' % (
                st['est_perr']*100, st['est_verr'], st.get('est_x', 0), st.get('est_y', 0))
        dpg.set_value('state', line)
    except Exception:
        dpg.set_value('state', '(biped_run.py 대기중…)')
    dpg.render_dearpygui_frame()

dpg.destroy_context()
