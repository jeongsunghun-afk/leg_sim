"""biped 전용 슬림 GUI (17-DOF GUI 규약) — dearpygui.

명령을 JSON 채널(/tmp/biped_cmd.json)로 발행 → biped_run.py 소비.
★듀얼 조이스틱: 좌스틱=전후(vx)/측방(vy) · 우스틱=선회(wz). 버튼: 정지·현자세 · 2점 평발 stand · 점발 보행.
실행(proxddp env): /home/jsh/miniforge3/envs/proxddp/bin/python teleop_gui_biped.py
  ① 컨트롤러: python biped_run.py   ② GUI: 위 명령
"""
import os, json, math, socket, time, subprocess, threading
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
    # ★축별 예외(jog_min_deg/jog_max_deg)를 반드시 반영한다 — 2026-08-10.
    #   여기는 emb/interface/joint_map.py 의 jog 한계 규칙을 **복제**한 코드다(GUI 는 별도
    #   venv 라 joint_map 을 import 하지 않는다). 그래서 그쪽만 고치면 여기가 조용히 어긋난다.
    #   실제로 어긋났었다: calf 를 [−55, 32.5] 로 넓혔는데 GUI 는 min_deg×0.5 = −27.5 를 써서,
    #   구조적 한계(−55°)에 서 있는 무릎이 [JOG 검증] 진입 순간 **+27.5° 명령**을 받았다.
    #   ⚠joint_map 의 규칙을 바꾸면 여기도 같이 고칠 것.
    JOG_LIM = [(float(j.get('jog_min_deg', j['min_deg'] * _frac)),
                float(j.get('jog_max_deg', j['max_deg'] * _frac))) for j in _cfg['joints']]
except Exception:
    pass
NJ = len(JOG_NAMES)

# ── ★위치모드 강성 배율 (2026-08-21) ────────────────────────────────────────
#   home/hold/jog 의 kp 에 곱하는 배율. **stand/walk 는 안 쓴다**(WBIC 와 싸우면 안 된다).
#   왜 GUI 로 빼는가: 접지시켜 하중을 걸며 "이 자세를 지키려면 얼마나 세야 하나" 를 찾는
#   작업이라, 매번 제어기를 껐다 켜며 env 를 바꾸는 게 실무상 불가능하다.
#
#   ⚠올릴수록 **토크트립까지의 각도가 줄어든다.** τ_trip ÷ (kp_joint·π/180·배율) 이다.
#     그래서 버튼마다 **가장 예민한 축의 트립각**을 같이 찍는다 — 숫자만 보고 올리면
#     접지 순간 그 축이 먼저 트립한다(calf 는 kp_joint 180 이라 ×5 에서 1.0° 다).
#   ⚠kd 는 제어기가 **√배율**로 같이 올린다(ζ ∝ kd/√kp 보존). GUI 가 따로 안 보낸다.
KP_STEPS = [1.0, 2.0, 3.0, 4.0, 5.0]
try:
    _tt = float(_cfg.get('safety', {}).get('tau_trip_nm', 15.0))
    # kp_joint = kp_ch·gear_k²  (emb/README "게인도 좌표가 둘")
    _kpj = [(j['name'], float(j['kp']) * float(j.get('gear_k', 1.0)) ** 2) for j in _cfg['joints']]
    _tight = max(_kpj, key=lambda t: t[1])      # kp_joint 가 가장 큰 축 = 트립각이 가장 작다
    KP_TRIP = [(_tight[0], _tt / (_tight[1] * math.pi / 180.0 * s)) for s in KP_STEPS]
except Exception:
    KP_TRIP = [('?', float('nan'))] * len(KP_STEPS)


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
        # ★pos_kp_scale=1.0 으로 시작한다 — GUI 를 띄우는 것만으로 강성이 바뀌면 안 된다.
        #   (제어기 쪽 env POS_KP_SCALE 은 **이 값이 도착하는 순간 덮인다.** GUI 를 쓸 거면
        #    env 로 주지 말고 여기 버튼으로 줄 것.)
        self.cmd = {'v': 0.0, 'vy': 0.0, 'w': 0.0, 'body_h': H_DEF, 'mode': 'hold', 'contact': '2pt',
                    'jog_deg': [0.0] * NJ, 'pos_kp_scale': 1.0, 'seq': 0}
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


def set_kp_scale(s):
    """위치모드(home/hold/jog) 강성 배율. 제어기가 1초에 걸쳐 램프한다."""
    pub.set(pos_kp_scale=float(s))
    for k, v in enumerate(KP_STEPS):
        dpg.bind_item_theme(f'kpbtn_{k}', _kp_on if abs(v - s) < 1e-6 else _kp_off)
    who, trip = KP_TRIP[KP_STEPS.index(s)] if s in KP_STEPS else ('?', float('nan'))
    dpg.set_value('kp_lbl', f'현재 ×{s:g}  ·  kd×{math.sqrt(s):.2f}(ζ 보존)  ·  '
                            f'트립 예민축 {who} {trip:.2f}°')
    dpg.configure_item('kp_lbl', color=(210, 120, 100) if trip < 2.0 else (150, 155, 175))


def jog_zero():                       # 전체 0(home)
    for i in range(NJ):
        dpg.set_value(f'jog_{i}', 0.0)
    pub.set(jog_deg=[0.0] * NJ)


# ── ★제어기 재기동 (통신 두절 복구) ──────────────────────────────────────
#   왜 필요한가: 실기 브링업 중 SHM/제어기가 이따금 끊긴다. 그때마다 터미널로 가서
#   프로세스를 죽이고 다시 띄우는 게 번거롭다. GUI 에서 한 번에 복구한다.
#
#   ⚠**RESET 버튼을 덮어쓰지 않았다.** RESET 은 이미 `jogger.reset(q_leg)` → HOLD 라는
#     안전 동작을 한다(jog 램프를 실측각으로 재시드해 점프를 막는다). 그걸 잃으면 안 된다.
#   ★RobotEmbedded 도 띄운다(2026-08-11) — setcap 이 적용돼 일반 사용자로 실행된다:
#       cap_net_admin,cap_net_raw=eip   (getcap 으로 확인)
#     SHM 세그먼트는 root 소유지만 perms 666 이라 비루트 attach 에 문제가 없다.
#     죽어 있을 때만 띄우고, 살아 있으면 건드리지 않는다.
#   ⚠두 번 눌러야 실행된다(오조작 방지) — 프로세스를 띄우고 모터를 물리는 동작이다.
# _EMB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emb')
# _EMB_BIN = os.path.expanduser('~/ZSource/RobotEmbedded/build/src/RobotEmbedded')
# _restart_armed = [0.0]


# def _proc_pids(prefix):
#     """명령줄이 prefix 로 **시작**하는 PID. bash 래퍼·자기 자신이 안 걸리게 한다."""
#     try:
#         out = subprocess.run(['ps', '-eo', 'pid=,args='], capture_output=True, text=True).stdout
#     except Exception:
#         return []
#     pids = []
#     for line in out.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         pid, _, args = line.partition(' ')
#         if args.strip().startswith(prefix):
#             pids.append(int(pid))
#     return pids
#
#
# def _pgrep_x(name):
#     """프로세스 **이름**이 정확히 일치하는 PID (명령줄 prefix 로는 sudo 래퍼를 못 잡는다)."""
#     try:
#         r = subprocess.run(['pgrep', '-x', name], capture_output=True, text=True)
#         return [int(x) for x in r.stdout.split()]
#     except Exception:
#         return []
#
#
# def _restart_worker():
#     def say(m):
#         print(f'[gui] {m}', flush=True)
#         try:
#             dpg.set_value('restart_msg', m)
#         except Exception:
#             pass
#
#     # ★RobotEmbedded 검출은 프로세스 **이름**으로 한다 — 명령줄 prefix 로는 `sudo ./src/...`
#     #   형태를 못 잡는다(처음에 그렇게 짰다가 실기에서 빈 배열이 나와 발견).
#     if not _pgrep_x('RobotEmbedded'):
#         # ★setcap 이 돼 있어 **일반 사용자로 띄울 수 있다**(2026-08-11 확인):
#         #     cap_net_admin,cap_net_raw=eip  ·  SHM perms 666 이라 비루트 attach 도 된다.
#         #   그 전에는 sudo 가 필요해 GUI 에서 못 띄웠다.
#         if not os.path.exists(_EMB_BIN):
#             say(f'✗ RobotEmbedded 바이너리가 없다: {_EMB_BIN}'); return
#         say('RobotEmbedded 기동 중 …')
#         try:
#             subprocess.Popen(['./src/RobotEmbedded'], cwd=os.path.dirname(os.path.dirname(_EMB_BIN)),
#                              stdout=open('/tmp/robotembedded.log', 'w'),
#                              stderr=subprocess.STDOUT, start_new_session=True)
#         except Exception as e:
#             say(f'✗ RobotEmbedded 기동 실패: {e} — setcap 확인'); return
#         # 초기화 게이트: halGait 는 수신 100틱 + 램프 4500틱 @1kHz ≈ 4.6초 동안 SHM 명령을
#         # 무시한다. 그 전에 제어기를 붙이면 첫 명령이 버려진다 ⇒ 충분히 기다린다.
#         for i in range(80):
#             time.sleep(0.1)
#             if not _pgrep_x('RobotEmbedded'):
#                 say('✗ RobotEmbedded 가 곧바로 죽었다 — /tmp/robotembedded.log 확인'); return
#             if i * 0.1 > 6.0:
#                 break
#         say('RobotEmbedded 준비됨(초기화 게이트 6s 대기 완료)')
#
#     old = _proc_pids('python3 app/biped_emb.py')
#     if old:
#         say(f'제어기 종료 {old} …')
#         for pid in old:
#             try:
#                 os.kill(pid, 15)
#             except Exception:
#                 pass
#         for _ in range(40):                       # 최대 4초 대기
#             time.sleep(0.1)
#             if not _proc_pids('python3 app/biped_emb.py'):
#                 break
#         else:
#             say('✗ 이전 제어기가 안 죽는다 — 수동 확인 필요'); return
#
#     say('제어기 기동 중 …')
#     try:
#         # ★hold 로 시작한다. off 로 띄우면 Emb 게이트가 열리는 순간 무여자 명령이 먹혀
#         #   다리가 풀린다(2026-08-10 실기에서 겪음).
#         subprocess.Popen(['python3', 'app/biped_emb.py', '--start-mode', 'hold'],
#                          cwd=_EMB_DIR, stdout=open('/tmp/biped_emb.log', 'w'),
#                          stderr=subprocess.STDOUT, start_new_session=True)
#     except Exception as e:
#         say(f'✗ 기동 실패: {e}'); return
#
#     for _ in range(100):                          # state 가 신선해질 때까지 최대 10초
#         time.sleep(0.1)
#         try:
#             if time.time() - os.path.getmtime(STATE) < 0.5:
#                 st = json.load(open(STATE))
#                 say(f"✅ 복구됨 — mode={st.get('mode')} n_ok={st.get('n_ok')}/8 "
#                     f"{st.get('loop_hz', 0):.0f}Hz")
#                 return
#         except Exception:
#             pass
#     say('✗ 기동했으나 state 가 갱신되지 않는다 — /tmp/biped_emb.log 확인')
#
#
# def on_restart():
#     now = time.time()
#     if now - _restart_armed[0] > 3.0:             # 1차 클릭 = 무장
#         _restart_armed[0] = now
#         dpg.set_value('restart_msg', '⚠ 3초 안에 한 번 더 누르면 제어기를 재기동한다')
#         return
#     _restart_armed[0] = 0.0
#     threading.Thread(target=_restart_worker, daemon=True).start()
#
#
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
        # ★실패하면 **jog 로 들어가지 않는다**(2026-08-10). 종전엔 except: pass 라
        #   상태를 못 읽어도 그대로 jog 로 진입했고, 그때 jog_deg 는 초기값 0 이다.
        #   영점을 잡은 지금 그건 "전 축 모델 0° 로 가라" 는 뜻이라 무릎이 55° 펴진다.
        #   시드에 실패했다는 건 명령 점프를 막을 근거가 없다는 뜻이므로 fail-closed 가 맞다.
        try:
            q = json.load(open(STATE))['q_leg_deg']
            if len(q) < NJ:
                raise ValueError(f'q_leg_deg 길이 {len(q)} < {NJ}')
            for i in range(NJ):
                v = float(max(JOG_LIM[i][0], min(JOG_LIM[i][1], q[i])))
                if abs(v - float(q[i])) > 0.5:
                    raise ValueError(
                        f'{JOG_NAMES[i]} 실측 {q[i]:+.1f}° 가 jog 한계 '
                        f'[{JOG_LIM[i][0]:+.1f},{JOG_LIM[i][1]:+.1f}] 밖 — 진입 시 '
                        f'{abs(v-float(q[i])):.1f}° 움직인다. 한계를 넓히거나 자세를 먼저 옮길 것')
                dpg.set_value(f'jog_{i}', v); pub.cmd['jog_deg'][i] = v
        except Exception as e:
            msg = f'JOG 진입 취소 — {e}'
            print(f'[gui] {msg}', flush=True)          # /tmp/teleop_gui_biped.log
            try:
                dpg.set_value('state', msg)            # 갱신 루프가 곧 덮어쓸 수 있다
            except Exception:
                pass
            return                                     # ★mode 를 안 보낸다 = 현재 모드 유지
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
#   ★고정 경로 목록만으로는 못 찾는다 — 사용자가 `~/.local/share/fonts` 에 깔아둔 폰트를
#     놓친다(2026-08-21 실측: 이 기기는 시스템 경로 셋이 다 없고 NanumGothic 이 홈에 있었다).
#     그래서 고정 목록 뒤에 **fontconfig 질의**를 붙인다. 그러면 어디에 깔았든 잡힌다.
def _fc_korean():
    """fc-list 로 한글 글리프가 있는 TTF/TTC 를 찾는다. fontconfig 이 없으면 조용히 포기."""
    import shutil, subprocess
    if not shutil.which('fc-list'):
        return []
    try:
        out = subprocess.run(['fc-list', ':lang=ko', 'file'],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return []
    paths = []
    for line in out.splitlines():
        p = line.split(':')[0].strip()
        if p.lower().endswith(('.ttf', '.ttc', '.otf')):
            paths.append(p)
    return paths

_FONT_CANDS = [os.environ.get('GUI_FONT', ''),
               '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
               '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
               '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
               os.path.expanduser('~/.local/share/fonts/NanumGothic-Regular.ttf')]
_FONT = next((f for f in _FONT_CANDS if f and os.path.exists(f)), None)
if _FONT is None:                                   # 마지막 수단 — 시스템에 뭐가 있든 찾는다
    _FONT = next((f for f in _fc_korean() if os.path.exists(f)), None)
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
with dpg.theme() as _kp_on:                 # 강성 배율 — 선택된 단계(주황)
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (185, 110, 40))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (215, 140, 60))
with dpg.theme() as _kp_off:                # 강성 배율 — 나머지(어둡게)
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (44, 48, 62))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 76, 96))

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
    # ★2026-08-14 라벨 정리 — 이름이 동작을 오해시키고 있었다.
    #   · 'RESET' → **'정지·현자세'**: 프로세스와 아무 상관이 없다. 하는 일은
    #     `jogger.reset(q_leg)` → HOLD, 즉 "jog 램프를 **현재 측정각**으로 재시드하고
    #     그 자리에 선다". 램프를 재시드하는 게 핵심이라 다음 JOG 에서 계단이 안 생긴다.
    #     'RESET' 이라 써 두니 '제어기 재기동'과 헷갈렸다 — 둘은 전혀 다른 동작이다.
    #   · '제어기 재기동' → **'제어기 재시작(프로세스)'**: 진짜로 프로세스를 죽였다 띄운다.
    #     SHM/통신 두절 복구용. 오조작 방지로 3초 안에 두 번 눌러야 실행된다.
    #   · 'Stand 서기' → **'2점 평발 stand'** · 'Walk 이동' → **'점발 보행'**
    #     접촉모드가 동작 이름 안에 들어가야 한다. 평발/점발은 발 자세가 53° 다르고
    #     (밑창 수평 vs toe-down) 전환은 매달아서 해야 하는 별개 작업이다.
    with dpg.group(horizontal=True):   # 정지 → Off전원 → JOG → 재시작
        _rb = dpg.add_button(label='정지·현자세', width=110, callback=lambda: set_mode('reset'))
        dpg.bind_item_theme(_rb, _stop)
        _ob = dpg.add_button(label='Off 전원', width=100, callback=lambda: set_mode('off'))
        dpg.bind_item_theme(_ob, _stop)
        dpg.add_button(label='JOG 검증', width=90, callback=lambda: set_mode('jog'))   # ★각축 검증(실기)
        # ★'제어기 재시작' 버튼 **삭제** (2026-08-14). 코드는 아래에 주석으로 남긴다.
        #   ⚠이 버튼은 `emb_ctl.sh` 를 **우회하고 RobotEmbedded 를 직접 띄웠다**:
        #       subprocess.Popen(['./src/RobotEmbedded'], stdout=open('/tmp/robotembedded.log','w'))
        #     ⇒ ① awk 로그필터 없음 — 시간당 1.7GB 로 자란다. 게다가 파일명이 달라
        #          (`robotembedded.log`) **아무도 안 보는 자리**에 쌓인다.
        #       ② 중복기동 가드 없음 — emb_ctl 은 "중복 기동은 EtherCAT 버스를 깬다" 며 막는다.
        #       ③ 신선도(MotorStatus16) 확인 없음 · stdbuf 없음(버퍼링으로 로그가 안 보임).
        #   ⇒ 기동/재기동은 **`emb/diag/emb_ctl.sh` 한 곳으로** 모은다. GUI 는 모드만 바꾼다.
        #     복구:  cd ~/simulation/biped/emb && diag/emb_ctl.sh stop && diag/emb_ctl.sh start
        with dpg.group():              # ★Home=정해진 홈 자세로 S-curve 복귀(emb/control/home.py)
            _hb = dpg.add_button(label='Home 복귀', width=100, callback=lambda: set_mode('home'))
            dpg.add_text('(S-curve)', color=(120, 130, 150))
        dpg.bind_item_theme(_hb, _home)
        dpg.add_button(label='Hold', width=70, callback=lambda: set_mode('hold'))
        with dpg.group():              # ★2점 평발 = 정적 자세유지(보행 안 함)
            dpg.add_button(label='2점 평발 stand', width=130, callback=lambda: set_mode('stand'))
            dpg.add_text('(밑창 접지·정적)', color=(120, 130, 150))
        with dpg.group():              # ★1점 점발 = stepping 보행
            _wb = dpg.add_button(label='점발 보행', width=110, callback=lambda: set_mode('walk'))
            dpg.add_text('(발끝 1점·동적)', color=(120, 130, 150))
        dpg.bind_item_theme(_wb, _walk)
    dpg.add_text('복구 순서: Off 전원 → Home 복귀 → Hold → (접지·하중전달) → 2점 평발 stand'
                 '   · Off=명령토크 0 (Kp=Kd=τ=0)', color=(150, 155, 175))
    dpg.add_text('⚠매달린 채로 stand/보행을 켜지 말 것 — GRF 를 전제한 QP 라 해가 안 나오고 '
                 '중력보상 폴백으로 떨어진다(겉보기엔 안정돼 보인다). 매달려서 되는 건 off/jog/home/hold 뿐.',
                 color=(210, 150, 90))
    dpg.add_text('Home=제어기가 정한 자세로 전축 동시 S-curve 이동 · Hold=지금 그 자리를 잡기\n'
                 '  목표자세: 1점 점발 = config home.q_deg(전축 0, biped_emb.py 와 동일) · '
                 '2점 평발 = Qflat8(발목 채널 100.4°, 밑창 수평)',
                 color=(150, 155, 175))
    # ── ★위치모드 강성 5단계 (2026-08-21) ────────────────────────────────────
    #   home/hold/jog 의 kp 배율. **stand/walk 에는 안 걸린다**(WBIC 와 싸우면 안 된다).
    #   쓰는 자리: home 으로 자세를 잡고 hold 로 굳힌 뒤 **크레인을 내려 하중을 실을 때**,
    #   그 자세를 지키려면 얼마나 세야 하는지 여기서 올려 가며 찾는다.
    #   ⚠제어기가 1초에 걸쳐 램프한다 — 계단으로 바꾸면 벌어져 있던 축의 토크가
    #     그 자리에서 배율만큼 튀어 접지 중에 τ_trip 이 바로 걸린다.
    dpg.add_text('● 위치모드 강성 (home·hold·jog 의 kp 배율 — stand/walk 무관)',
                 color=(255, 205, 120))
    with dpg.group(horizontal=True):
        for _k, _s in enumerate(KP_STEPS):
            _who, _tr = KP_TRIP[_k]
            dpg.add_button(label=f'×{_s:g}', width=52, tag=f'kpbtn_{_k}',
                           callback=lambda _a, _b, u=_s: set_kp_scale(u))
            with dpg.tooltip(f'kpbtn_{_k}'):
                dpg.add_text(f'kp×{_s:g} · kd×{math.sqrt(_s):.2f}\n'
                             f'가장 예민한 축 {_who} — {_tr:.2f}° 에서 토크트립')
        dpg.add_text('', tag='kp_lbl', color=(150, 155, 175))
    dpg.add_text('⚠올릴수록 자세는 잘 지키지만 **토크트립까지의 각도가 줄어든다** '
                 '(τ_trip ÷ kp_joint). 접지시키며 하중이 실릴 때 여기 걸리기 쉽다 — ×3 부터 시작할 것.',
                 color=(210, 150, 90))
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
            # ★슬라이더 양끝에 jog 한계를 숫자로 박아 둔다. 이 한계는 축마다 다르고
            #   (config 의 jog_min_deg/jog_max_deg 예외), 관절한계와도 다르다 —
            #   화면에 안 쓰여 있으면 "왜 여기서 안 넘어가지" 를 매번 config 를 열어 확인해야 한다.
            dpg.add_text(f'{JOG_LIM[i][0]:>6.1f}', color=(120, 130, 150))
            dpg.add_slider_float(tag=f'jog_{i}', default_value=0.0,
                                 min_value=JOG_LIM[i][0], max_value=JOG_LIM[i][1],
                                 width=240, format='%.1f', user_data=i,
                                 callback=lambda s, v, u: on_jog(s, v, u))
            dpg.add_text(f'{JOG_LIM[i][1]:<6.1f}', color=(120, 130, 150))
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
set_kp_scale(1.0)      # ★강성 버튼 초기 선택 표시(×1). Pub 기본값과 반드시 일치시킬 것.

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
