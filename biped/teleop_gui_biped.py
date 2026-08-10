"""biped 전용 슬림 GUI (17-DOF GUI 규약) — dearpygui.

명령을 JSON 채널(/tmp/biped_cmd.json)로 발행 → biped_run.py 소비.
★듀얼 조이스틱: 좌스틱=전후(vx)/측방(vy) · 우스틱=선회(wz). 버튼 Stand/Walk/RESET.
실행(proxddp env): /home/jsh/miniforge3/envs/proxddp/bin/python teleop_gui_biped.py
  ① 컨트롤러: python biped_run.py   ② GUI: 위 명령
"""
import os, json, math, socket, time
import dearpygui.dearpygui as dpg

CMD    = os.environ.get('QUAD_CMD',   '/tmp/biped_cmd.json')
STATE  = os.environ.get('QUAD_STATE', '/tmp/biped_state.json')
# ★Isaac Sim 배선: TELEOP_UDP="host:port" 설정 시 명령을 UDP로도 발행(원격 play.py 수신).
_UDP_TGT  = os.environ.get('TELEOP_UDP')          # 예: 192.168.1.205:9999
_udp_sock = None; _udp_addr = None
if _UDP_TGT:
    _uh, _up = _UDP_TGT.split(':'); _udp_addr = (_uh, int(_up))
    _udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# ★범위 env 오버라이드(hind_leg는 vx~2.0·yaw~0.5라 기본보다 넓게 쓸 수 있음)
VMAX   = float(os.environ.get('VMAX',   '0.15'))  # 전진 상한[m/s]
VY_MAX = float(os.environ.get('VY_MAX', '0.10'))  # 좌우 상한[m/s]
WZ_MAX = float(os.environ.get('WZ_MAX', '0.30'))  # 선회 상한[rad/s]
H_MIN, H_MAX, H_DEF = 0.36, 0.54, 0.38  # 슬라이더 전체범위·시작(2점)기본
H_DEF_1PT, H_DEF_2PT = 0.50, 0.38       # ★접촉모드별 기본 몸통높이(점발/평발, 접촉구2배 자연높이)

# ── 각축(JOG) 검증용 관절 정의 — emb/config/biped_emb.yaml 있으면 로드, 없으면 기본값 ──
#   실기(app/biped_emb.py) 배포 시 축별 목표각·통신 LED로 각 모터 확인. sim에선 inert(무해).
JOG_NAMES = ['HL_hip', 'HL_thigh', 'HL_calf', 'HL_foot', 'HR_hip', 'HR_thigh', 'HR_calf', 'HR_foot']
JOG_LIM   = [(-17, 17), (-67, 32), (-27, 32), (-40, 20)] * 2   # jog 안전범위(deg)=mjcf range×0.5
try:
    import yaml
    _cfgp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emb', 'config', 'biped_emb.yaml')
    _cfg = yaml.safe_load(open(_cfgp))
    _frac = float(_cfg.get('jog', {}).get('range_frac', 0.5))
    JOG_NAMES = [j['name'] for j in _cfg['joints']]
    JOG_LIM = [(j['min_deg'] * _frac, j['max_deg'] * _frac) for j in _cfg['joints']]
except Exception:
    pass
NJ = len(JOG_NAMES)


class Pub:
    def __init__(self, path=CMD):
        self.path = path
        # ★시작 모드 — 'stand' 는 절대 금지. Pub() 이 생성 시점에 곧바로 _pub() 하므로
        #   (뷰포트 생성 전!) **기동 4초 내 자동 무장**되고 모델기반 제어가 돌아버린다.
        #
        # ★2026-08-07: 'off' → 'hold'. 실기에서 Emb 가 4.5초 램프로 잡아둔 자세를
        #   GUI 가 off 를 쏘는 순간 놓아버려 **다리가 떨어졌다**(hip 중력토크 4.96 Nm).
        #   hold 는 "지금 그 자리를 유지" 라 인계 시 움직임이 0 이다.
        #   ⚠stand/walk 와 달리 hold 는 모델기반 제어가 아니다 — 측정각 임피던스 유지뿐.
        #   sim 에서는 컨트롤러가 hold 를 모르면 무시하므로 무해하다.
        self.cmd = {'v': 0.0, 'vy': 0.0, 'w': 0.0, 'body_h': H_DEF, 'mode': 'hold', 'contact': '2pt',
                    'jog_deg': [0.0] * NJ, 'seq': 0}
        self._pub()

    def set_jog(self, i, val):
        self.cmd['jog_deg'][i] = float(val); self._pub()

    def set(self, **kw):
        self.cmd.update(kw)
        # ★스틱/슬라이더로 非零 속도가 들어오면 자동 walk 전환.
        #   (안 그러면 sim이 mode=stand를 보고 속도를 0으로 무시 → "명령이 안 먹는" 증상)
        # ★'off' 를 자동승격 대상에서 뺐다. off 는 "아직 무장 안 함" 상태이므로
        #   조이스틱 한 번에 stand 를 건너뛰고 walk 로 뛰어드는 것을 막는다.
        if self.cmd.get('mode') in ('stand',) and (
            abs(self.cmd.get('v', 0.0)) > 1e-3 or abs(self.cmd.get('vy', 0.0)) > 1e-3
            or abs(self.cmd.get('w', 0.0)) > 1e-3):
            self.cmd['mode'] = 'walk'
        self._pub()

    def _pub(self):
        # ★seq 를 증가시킨다. emb 앱의 워치독은 "파일이 읽히는가" 가 아니라
        #   "명령 내용이 바뀌는가" 로 살아있음을 판정하므로(biped_emb.read_cmd_fresh),
        #   정적 파일은 통신두절과 구분되지 않는다. seq 가 그 구분을 만든다.
        self.cmd['seq'] = int(self.cmd.get('seq', 0)) + 1
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.cmd, f)
        os.replace(tmp, self.path)
        if _udp_sock is not None:                      # ★Isaac Sim으로 UDP 발행
            try: _udp_sock.sendto(json.dumps(self.cmd).encode(), _udp_addr)
            except Exception: pass


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


def on_jog(sender, val, i):           # 각축 슬라이더 → 목표각(deg) 발행
    pub.set_jog(i, val)


def jog_zero():                       # 전체 0(home)
    for i in range(NJ):
        dpg.set_value(f'jog_{i}', 0.0)
    pub.set(jog_deg=[0.0] * NJ)


def set_mode(mode):
    if mode == 'reset':
        left.clear(); right.clear()
        pub.set(mode='reset', v=0.0, vy=0.0, w=0.0)
        dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)
        return
    if mode == 'walk':          # ★걸으려면 1점 점발로 자동 전환(평발=발목토크0라 걷기 부적합)
        pub.set(contact='1pt', body_h=H_DEF_1PT, mode='walk')
        dpg.set_value('h_sl', H_DEF_1PT)
        return
    if mode == 'stand':         # ★서려면 2점 평발로 자동 전환(밑창ZMP=안정 서기)
        left.clear(); right.clear()
        pub.set(contact='2pt', body_h=H_DEF_2PT, mode='stand', v=0.0, vy=0.0, w=0.0)
        dpg.set_value('h_sl', H_DEF_2PT)
        dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)
        return
    if mode == 'jog':           # ★각축 검증 진입: 슬라이더를 현재 실측각으로 정렬(명령 점프 방지)
        try:
            q = json.load(open(STATE)).get('q_leg_deg', [0.0] * NJ)
            for i in range(NJ):
                v = float(max(JOG_LIM[i][0], min(JOG_LIM[i][1], q[i])))
                dpg.set_value(f'jog_{i}', v); pub.cmd['jog_deg'][i] = v
        except Exception:
            pass
    pub.set(mode=mode)          # off/jog/hold 등
    left.clear(); right.clear(); pub.set(v=0.0, vy=0.0, w=0.0)
    dpg.set_value('spd_sl', 0); dpg.set_value('vy_sl', 0); dpg.set_value('turn_sl', 0)


left  = JoyPad('joyL', 190, on_left, cross_only=True)   # ★십자만(전후 XOR 측방, 대각 금지)
right = JoyPad('joyR', 190, on_right, x_only=True)

dpg.create_context()
# ★한글 폰트 — GUI(dearpygui) 는 TTF 를 로드하므로 한글이 정상이다.
#   ⚠깨져 보였던 것은 **MuJoCo 뷰어** 쪽이다(mjr_overlay = ASCII 전용 비트맵 폰트).
#     둘은 별개다 — 뷰어 HUD 는 영문으로 바꿨고(biped_monitor.cpp), 여기는 TTF 를 쓴다.
#   폰트 파일이 없으면 라벨이 깨져 JOG 검증에서 축 이름을 못 읽으므로 기동 시 경고한다.
_FONT_CANDS = [os.environ.get('GUI_FONT', ''),
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
               '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
               '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc']
_FONT = next((f for f in _FONT_CANDS if f and os.path.exists(f)), None)
_kf = None
if _FONT:
    with dpg.font_registry():
        _kf = dpg.add_font(_FONT, 18)
    # ⚠add_font_range_hint 는 쓰지 않는다 — dearpygui 2.x 에서 **deprecated no-op** 이다
    #   ("character ranges are now automatic"). 부르면 경고만 나오고 효과가 없다.
    print(f'[gui] 한글 폰트: {_FONT}')
else:
    print('[gui] ⚠ CJK 폰트를 못 찾았다 — 한글 라벨이 깨져 보인다.\n'
          '      설치: sudo apt install fonts-noto-cjk   (또는 GUI_FONT=/경로/폰트.ttf)')
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
with dpg.theme() as _home:                  # Home 복귀 = 파랑(이동은 하지만 자율보행은 아님)
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 85, 140))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 115, 185))

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
    dpg.add_text('모션', color=(170, 175, 195))
    with dpg.group(horizontal=True):   # RESET → Off전원 → Stand(2점서기) → Walk(1점보행)
        _rb = dpg.add_button(label='RESET', width=90, callback=lambda: set_mode('reset'))
        dpg.bind_item_theme(_rb, _stop)
        _ob = dpg.add_button(label='Off 전원', width=100, callback=lambda: set_mode('off'))
        dpg.bind_item_theme(_ob, _stop)
        dpg.add_button(label='JOG 검증', width=90, callback=lambda: set_mode('jog'))   # ★각축 검증(실기)
        with dpg.group():              # ★Home=정해진 홈 자세로 S-curve 복귀(emb/control/home.py)
            _hb = dpg.add_button(label='Home 복귀', width=100, callback=lambda: set_mode('home'))
            dpg.add_text('(S-curve)', color=(120, 130, 150))
        dpg.bind_item_theme(_hb, _home)
        dpg.add_button(label='Hold', width=70, callback=lambda: set_mode('hold'))
        with dpg.group():              # ★Stand=2점접촉 서기(자동 전환)
            dpg.add_button(label='Stand 서기', width=110, callback=lambda: set_mode('stand'))
            dpg.add_text('(2점접촉)', color=(120, 130, 150))
        with dpg.group():              # ★Walk=1점접촉 보행(자동 전환)
            _wb = dpg.add_button(label='Walk 이동', width=110, callback=lambda: set_mode('walk'))
            dpg.add_text('(1점접촉)', color=(120, 130, 150))
        dpg.bind_item_theme(_wb, _walk)
    dpg.add_text('복구 순서: 전원(Off) → [Home 복귀] → 서기(Stand) → 이동(Walk)   · Off=모터 토크차단(limp), 실HW=motor disable',
                 color=(150, 155, 175))
    dpg.add_text('Home=정해진 홈 자세(emb/config/biped_emb.yaml: home.q_deg)로 전축 동시 S-curve 이동 · Hold=지금 그 자리를 잡기',
                 color=(150, 155, 175))
    dpg.add_separator()
    # ── ★각축(JOG) 패널: 8관절 슬라이더(모터 1:1) + 실측 + 통신 상태 LED ──
    dpg.add_text('● 각축 JOG 검증 (슬라이더=목표각° · 실측° · ●=상태LED)', color=(255, 205, 120))
    with dpg.group(horizontal=True):
        # ★라벨에서 'home' 을 뺐다 — 위 [Home 복귀] 버튼과 전혀 다른 동작이다.
        #   이건 JOG 슬라이더를 0 으로 놓는 것(등속 램프, 축마다 도착시각 제각각)이고,
        #   [Home 복귀] 는 home 모드의 S-curve 동시도착 궤적이다.
        dpg.add_button(label='슬라이더 모두 0', width=130, callback=jog_zero)
        dpg.add_text('LED 초록=정상·노랑=에러·빨강=두절(배선O)·어두움=미장착 · 실기(app/biped_emb.py)서 각 모터 확인',
                     color=(120, 130, 150))
    _LED_R = 7
    for i, nm in enumerate(JOG_NAMES):
        with dpg.group(horizontal=True):
            with dpg.drawlist(width=2 * _LED_R + 6, height=2 * _LED_R + 6, tag=f'leddl_{i}'):
                dpg.draw_circle([_LED_R + 3, _LED_R + 3], _LED_R, fill=(70, 70, 78),
                                color=(30, 30, 36), tag=f'led_{i}')
            dpg.add_text(f'{nm:9s}', color=(190, 195, 210))
            dpg.add_slider_float(tag=f'jog_{i}', default_value=0.0,
                                 min_value=JOG_LIM[i][0], max_value=JOG_LIM[i][1],
                                 width=240, format='%.1f', user_data=i,
                                 callback=lambda s, v, u: on_jog(s, v, u))
            dpg.add_text('--.-', tag=f'meas_{i}', color=(150, 220, 150))
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
dpg.create_viewport(title='biped teleop', width=700, height=800)
dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window('main', True)

# ★absent(미장착)를 dead(배선됐는데 두절)와 다른 색으로 둔다 — 전자는 정상, 후자는 고장이다.
#   같은 회색으로 뭉뚱그리면 "원래 없는 축" 과 "죽은 축" 이 구분되지 않는다.
_LED = {'ok': (60, 210, 90), 'fault': (235, 200, 60),
        'dead': (150, 90, 90), 'absent': (48, 50, 58)}
_last_file_hb = [0.0]          # ★파일 하트비트 타이머(리스트=클로저 없이 가변)
while dpg.is_dearpygui_running():
    try:
        with open(STATE) as f:
            st = json.load(f)
        if 'health' in st or 'q_leg_deg' in st:          # ── emb(app/biped_emb) 상태: LED+실측 ──
            q = st.get('q_leg_deg', [0.0] * NJ)
            health = st.get('health', ['dead'] * NJ)
            for i in range(min(NJ, len(q))):
                dpg.set_value(f'meas_{i}', f'{q[i]:+6.1f}')
            for i in range(min(NJ, len(health))):
                dpg.configure_item(f'led_{i}', fill=_LED.get(health[i], (70, 70, 78)))
            # ★분모는 **실장축 수**. 미장착을 분모에 넣으면 정상인데도 "8중 2" 로 보인다.
            n_inst = st.get('n_installed', NJ)
            line = ('[emb] mode=%s  backend=%s  정상%d/에러%d/두절%d/%d  tilt%.1f°  loop%.0fHz'
                    % (st.get('mode', '-'), st.get('backend', '-'), st.get('n_ok', 0),
                       st.get('n_fault', 0), st.get('n_dead', n_inst), n_inst,
                       st.get('tilt_deg', 0), st.get('loop_hz', 0)))
            if st.get('n_absent'):
                line += '  · 미장착%d' % st['n_absent']
            if 'home_progress' in st:            # home 모드 진행률 + 실제 도달 여부
                # ★진행률(명령 기준)과 도달(측정 기준)을 따로 보여준다 — 궤적이 끝나도
                #   부하·마찰로 실제로는 안 들어와 있을 수 있고, 그게 중요한 정보다.
                line += '\nHome 궤적 %3.0f%%%s  %s' % (
                    st['home_progress'] * 100,
                    ' (완료)' if st.get('home_done') else '',
                    '✓ 홈 도달' if st.get('home_at_goal') else '… 미도달')
        else:                                            # ── sim(biped_run/view) 상태 ──
            line = ('mode=%s  높이%.2f  vx%+.2f vy%+.2f wz%+.2f  yaw%+.0f°  tilt%.1f°  (%+.1f,%+.1f)'
                    % (st.get('mode', '-'), st.get('base_z', 0), st.get('vx_cmd', 0), pub.cmd['vy'],
                       st.get('wz_cmd', 0), st.get('yaw', 0), st.get('tilt', 0),
                       st.get('x', 0), st.get('y', 0)))
            if 'est_perr' in st:                         # biped_deploy = leg-odometry 추정오차(GT 대비)
                line += '\n추정(leg-odom) 오차: pos %.1fcm  vel %.3fm/s   EST(%+.2f,%+.2f)' % (
                    st['est_perr']*100, st['est_verr'], st.get('est_x', 0), st.get('est_y', 0))
        dpg.set_value('state', line)
    except Exception:
        dpg.set_value('state', '(컨트롤러 대기중…)')
    # ★연속 발행: 스틱을 가만히 눌러 유지해도(=drag 이벤트 없음) 명령이 계속 전송되게.
    #   dpg drag 핸들러는 마우스가 움직일 때만 발화 → 정지 유지 시 패킷 끊김 → sim이 옛 명령 유지/누락.
    #   매 프레임 현재 pub.cmd를 UDP로 재전송(≈60Hz)해 이벤트 타이밍 의존 제거.
    if _udp_sock is not None:
        try: _udp_sock.sendto(json.dumps(pub.cmd).encode(), _udp_addr)
        except Exception: pass
    # ★파일 채널에도 동일한 하트비트(20Hz). 위 주석이 UDP 에 대해 지적한 문제
    #   ("이벤트가 없으면 패킷이 끊긴다")가 **파일 경로에도 똑같이 있었는데 안 고쳐져
    #   있었다.** emb 앱 워치독은 이 하트비트로 GUI 생존을 판정한다 — 없으면 워치독이
    #   jog 램프 중(무이벤트 1.5s) 오작동한다.
    _now = time.time()
    if _now - _last_file_hb[0] > 0.05:
        _last_file_hb[0] = _now
        try: pub._pub()
        except Exception: pass
    dpg.render_dearpygui_frame()

dpg.destroy_context()
