"""
leg_sim_v5.py  —  4족 보행 Gait 시뮬레이터
v4 (단일 다리, 점프/보행) → v5 (4족 동시 시각화, GaitController 검증)

[구조]
  GaitScheduler  : 다리별 stance/swing 위상 관리 (trot / walk)
  swing_foot_pos : Cubic Bezier swing 궤적
  stance_foot_pos: 선형 stance (발 지면 고정)
  analytical_ik  : 발끝 xyz → joint angles 4개 (leg_sim_v4 동일)

[시각화]
  좌: 3D 몸체 + 4다리 + 발끝 궤적
  우상: 보행 위상 다이어그램 (swing=색/stance=회색)
  우중: 발끝 Z 높이 (swing 확인)
  우하: Hip Pitch (th2) 관절각
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# ── 한글 폰트
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 0. 파라미터
# ══════════════════════════════════════════════════════════════
GAIT_TYPE     = 'trot'   # 'trot' | 'walk'
DT            = 0.005    # 제어 주기 [s] (200Hz)
N_CYCLES      = 4        # 시뮬레이션 보행 사이클 수
GAIT_PERIOD   = 0.6      # 1주기 시간 [s]
SWING_RATIO   = 0.4      # swing 비율 (나머지는 stance)
STEP_HEIGHT   = 0.04     # 발 들어올림 높이 [m]
STRIDE_LENGTH = 0.06     # 보폭 [m]
BODY_VX       = 0.1      # 전진 속도 [m/s]

# 몸체 크기 [m]
BODY_L = 0.30   # 앞뒤 길이
BODY_W = 0.20   # 좌우 너비

# DH 파라미터 (leg_sim_v4 동일, 4관절 사용)
DH_PARAMS = [
    (-math.pi/2, 0.0,   0.0   ),   # Joint 1: Hip Abduction
    (0.0,        0.21,  0.0075),   # Joint 2: Hip Pitch
    (0.0,        0.21,  0.0   ),   # Joint 3: Knee
    (0.0,        0.148, 0.0   ),   # Joint 4: Ankle
]
_A2 = DH_PARAMS[1][1]
_A3 = DH_PARAMS[2][1]
_A4 = DH_PARAMS[3][1]
_D2 = DH_PARAMS[1][2]

Q_HOME   = [math.radians(a) for a in [0, -150, -90, 90]]
PHI_HOME = Q_HOME[1] + Q_HOME[2] + Q_HOME[3]   # ankle 각도 합 고정

# 다리 레이아웃
LEG_NAMES  = ['FL', 'FR', 'RL', 'RR']
LEG_COLORS = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00']

# 힙 위치 오프셋 (몸체 프레임 기준)
LEG_HIP_OFFSETS = np.array([
    [+BODY_L/2, +BODY_W/2, 0.0],   # FL: 앞왼쪽
    [+BODY_L/2, -BODY_W/2, 0.0],   # FR: 앞오른쪽
    [-BODY_L/2, +BODY_W/2, 0.0],   # RL: 뒤왼쪽
    [-BODY_L/2, -BODY_W/2, 0.0],   # RR: 뒤오른쪽
])

# 위상 오프셋 [FL, FR, RL, RR]
PHASE_OFFSETS = {
    'trot': [0.0, 0.5, 0.5, 0.0],    # FL+RR 동시, FR+RL 동시
    'walk': [0.0, 0.25, 0.75, 0.5],  # 순차
}

# ══════════════════════════════════════════════════════════════
# 1. 기구학 (leg_sim_v4 참조)
# ══════════════════════════════════════════════════════════════

def _dh_matrix(alpha, a, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ], dtype=float)

def forward_kinematics(thetas):
    """joint angles(4) → 관절 원점 리스트 5개 [힙~발끝]"""
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (alpha, a, d) in enumerate(DH_PARAMS):
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
        pts.append(T[:3, 3].copy())
    return pts

def analytical_ik(Px, Py, Pz, phi=PHI_HOME, elbow_up=True):
    """발끝 xyz (다리 로컬 프레임) → joint angles 4개. 실패 시 None"""
    D2 = Px**2 + Py**2 - _D2**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)
    theta1 = math.atan2(-Px, Py) - math.atan2(R, _D2)
    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s    = c1 * Px + s1 * Py
    Z      = -Pz
    x3     = x_s - _A4 * math.cos(phi)
    z3     = Z   - _A4 * math.sin(phi)
    cos_th3 = (x3**2 + z3**2 - _A2**2 - _A3**2) / (2.0 * _A2 * _A3)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = -math.acos(cos_th3) if elbow_up else math.acos(cos_th3)
    theta2  = (math.atan2(z3, x3)
               - math.atan2(_A3 * math.sin(theta3),
                             _A2 + _A3 * math.cos(theta3)))
    theta4  = phi - theta2 - theta3

    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    return [wrap(theta1), wrap(theta2), wrap(theta3), wrap(theta4)]

# ══════════════════════════════════════════════════════════════
# 2. Gait Scheduler & Foot Trajectory
# ══════════════════════════════════════════════════════════════

class GaitScheduler:
    def __init__(self, gait=GAIT_TYPE, period=GAIT_PERIOD, swing_ratio=SWING_RATIO):
        self.period      = period
        self.swing_ratio = swing_ratio
        self.offsets     = PHASE_OFFSETS[gait]

    def phase(self, leg, t):
        return (t / self.period + self.offsets[leg]) % 1.0

    def is_swing(self, leg, t):
        return self.phase(leg, t) < self.swing_ratio

    def swing_t(self, leg, t):
        p = self.phase(leg, t)
        return p / self.swing_ratio if p < self.swing_ratio else 0.0

    def stance_t(self, leg, t):
        p = self.phase(leg, t)
        if p >= self.swing_ratio:
            return (p - self.swing_ratio) / (1.0 - self.swing_ratio)
        return 0.0


def swing_foot_pos(sw_t, p_start, p_end, step_height=STEP_HEIGHT):
    """Cubic Bezier swing 궤적 (들어올림 + 전진 + 착지)"""
    t  = sw_t
    p1 = p_start + np.array([0, 0, step_height])
    p2 = p_end   + np.array([0, 0, step_height])
    return ((1-t)**3 * p_start
            + 3*(1-t)**2*t * p1
            + 3*(1-t)*t**2 * p2
            + t**3 * p_end)


def stance_foot_pos(st_t, p_contact, body_vel, stance_dur):
    """Stance: 발 지면 고정 (몸체 전진 = 발 상대적 후방 이동)"""
    return p_contact - body_vel * (st_t * stance_dur)

# ══════════════════════════════════════════════════════════════
# 3. 궤적 사전 계산
# ══════════════════════════════════════════════════════════════
N_FRAMES   = int(N_CYCLES * GAIT_PERIOD / DT)
sched      = GaitScheduler()
stance_dur = GAIT_PERIOD * (1.0 - SWING_RATIO)
body_vel   = np.array([BODY_VX, 0.0, 0.0])

# 홈 자세 발끝 위치 (다리 로컬 프레임)
home_foot = np.array(forward_kinematics(Q_HOME)[4])

print("─" * 55)
print(f"궤적 계산 중...  [{GAIT_TYPE}]  {N_CYCLES}사이클  {N_FRAMES}프레임")
print(f"  홈 발끝: X={home_foot[0]*1e3:.1f}mm  "
      f"Y={home_foot[1]*1e3:.1f}mm  Z={home_foot[2]*1e3:.1f}mm")

# 기록 배열
joint_hist = np.zeros((N_FRAMES, 4, 4))   # [frame, leg, joint]
foot_hist  = np.zeros((N_FRAMES, 4, 3))   # [frame, leg, xyz] 월드 프레임
phase_hist = np.zeros((N_FRAMES, 4))      # [frame, leg] 0~1
swing_flag = np.zeros((N_FRAMES, 4), dtype=bool)

# 다리별 상태
foot_contact    = [home_foot.copy() for _ in range(4)]
foot_sw_start   = [home_foot.copy() for _ in range(4)]
foot_local_prev = [home_foot.copy() for _ in range(4)]
prev_swing      = [sched.is_swing(leg, 0) for leg in range(4)]

for fi in range(N_FRAMES):
    t = fi * DT
    for leg in range(4):
        is_sw = sched.is_swing(leg, t)
        phase_hist[fi, leg] = sched.phase(leg, t)
        swing_flag[fi, leg] = is_sw

        # 상태 전환 감지
        if is_sw and not prev_swing[leg]:
            # swing 시작: 현재 발 위치를 swing 출발점으로
            foot_sw_start[leg] = foot_local_prev[leg].copy()
        if not is_sw and prev_swing[leg]:
            # stance 시작: 이전 발 위치를 contact점으로
            foot_contact[leg] = foot_local_prev[leg].copy()

        # 발끝 위치 계산 (로컬 프레임)
        if is_sw:
            sw_t     = sched.swing_t(leg, t)
            # 착지 목표: 홈 위치 + 보폭 절반 앞
            p_end    = home_foot + np.array([STRIDE_LENGTH * 0.5, 0, 0])
            foot_loc = swing_foot_pos(sw_t, foot_sw_start[leg], p_end)
        else:
            st_t     = sched.stance_t(leg, t)
            foot_loc = stance_foot_pos(st_t, foot_contact[leg], body_vel, stance_dur)

        foot_local_prev[leg] = foot_loc.copy()
        prev_swing[leg]      = is_sw

        # 월드 프레임 = 힙 오프셋 + 로컬 발끝
        foot_hist[fi, leg] = LEG_HIP_OFFSETS[leg] + foot_loc

        # IK
        q = analytical_ik(foot_loc[0], foot_loc[1], foot_loc[2])
        if q is None:
            q = list(Q_HOME)
        joint_hist[fi, leg] = q

print(f"완료.")
print("─" * 55)

# ══════════════════════════════════════════════════════════════
# 4. 시각화 설정
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(17, 9))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(3, 2, figure=fig, wspace=0.35, hspace=0.60,
                        left=0.04, right=0.97, top=0.93, bottom=0.06)

_dark = '#16213e'
_gray = 'gray'

def _style_ax(ax, title, xlabel='프레임', ylabel=''):
    ax.set_facecolor(_dark)
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors=_gray)
    ax.grid(True, alpha=0.25, color=_gray)
    for sp in ax.spines.values():
        sp.set_edgecolor(_gray)

# ── 3D 뷰 (왼쪽 전체)
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_facecolor(_dark)
reach = 0.65
ax3d.set_xlim(-reach, reach)
ax3d.set_ylim(-0.5, 0.5)
ax3d.set_zlim(-0.6, 0.15)
ax3d.set_xlabel('X (m)', color='white', labelpad=4)
ax3d.set_ylabel('Y (m)', color='white', labelpad=4)
ax3d.set_zlabel('Z (m)', color='white', labelpad=4)
ax3d.tick_params(colors=_gray)
ax3d.set_title(f'4족 {GAIT_TYPE.upper()}  |  Vx={BODY_VX}m/s  |  T={GAIT_PERIOD}s',
               color='white', fontsize=10)
ax3d.view_init(elev=20, azim=-55)
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False

# 몸체 사각형
_bc = np.array([
    [+BODY_L/2, +BODY_W/2, 0],
    [+BODY_L/2, -BODY_W/2, 0],
    [-BODY_L/2, -BODY_W/2, 0],
    [-BODY_L/2, +BODY_W/2, 0],
    [+BODY_L/2, +BODY_W/2, 0],
])
ax3d.plot(_bc[:,0], _bc[:,1], _bc[:,2], '-', color='white', lw=2.5, alpha=0.7)

# 지면 (발끝 Z 기준)
gnd_z = home_foot[2]
xx, yy = np.meshgrid([-reach, reach], [-0.5, 0.5])
ax3d.plot_surface(xx, yy, np.full_like(xx, gnd_z), alpha=0.12, color='#888888')

# 힙 점 표시
for leg in range(4):
    h = LEG_HIP_OFFSETS[leg]
    ax3d.plot([h[0]], [h[1]], [h[2]], 'o', color=LEG_COLORS[leg], markersize=7, alpha=0.8)

# 다리 링크 선 (힙→J1→J2→J3→발끝 = 4구간)
leg_links = []
for leg in range(4):
    lns = [ax3d.plot([], [], [], '-o', color=LEG_COLORS[leg],
                     lw=2.5, markersize=5)[0] for _ in range(4)]
    leg_links.append(lns)

# 발끝 trace (최근 1주기)
TRACE_LEN  = int(GAIT_PERIOD / DT)
leg_traces = [ax3d.plot([], [], [], '-', color=LEG_COLORS[leg],
                        lw=1.2, alpha=0.6)[0] for leg in range(4)]
trace_buf  = [[[], [], []] for _ in range(4)]

# swing 중 발끝 마커
swing_dots = [ax3d.plot([], [], [], 'o', color=LEG_COLORS[leg],
                        markersize=9, alpha=0.9)[0] for leg in range(4)]

info_text = ax3d.text2D(0.02, 0.98, "", transform=ax3d.transAxes,
                         color='white', fontfamily='monospace', fontsize=7.5, va='top')

# ── 위상 다이어그램 (우상)
ax_phase = fig.add_subplot(gs[0, 1])
_style_ax(ax_phase, f'보행 위상  [{GAIT_TYPE}]  (밝음=swing)', ylabel='다리')
ax_phase.set_xlim(0, N_FRAMES)
ax_phase.set_ylim(-0.5, 3.5)
ax_phase.set_yticks([0, 1, 2, 3])
ax_phase.set_yticklabels(LEG_NAMES[::-1], color='white')

# 위상 다이어그램 — 사전에 컬러맵으로 미리 그림
for leg in range(4):
    row = 3 - leg   # 위에서부터 FL, FR, RL, RR
    # swing 구간 찾아서 사각형 표시
    in_sw   = False
    sw_start = 0
    for fi in range(N_FRAMES):
        if swing_flag[fi, leg] and not in_sw:
            sw_start = fi
            in_sw    = True
        elif not swing_flag[fi, leg] and in_sw:
            ax_phase.barh(row, fi - sw_start, left=sw_start, height=0.7,
                          color=LEG_COLORS[leg], alpha=0.85)
            in_sw = False
    if in_sw:
        ax_phase.barh(row, N_FRAMES - sw_start, left=sw_start, height=0.7,
                      color=LEG_COLORS[leg], alpha=0.85)

phase_cursor = ax_phase.axvline(x=0, color='white', lw=1.5, ls='--')

# ── 발끝 Z 높이 (우중)
ax_z = fig.add_subplot(gs[1, 1])
_style_ax(ax_z, '발끝 Z 높이 [m]', ylabel='Z [m]')
ax_z.set_xlim(0, N_FRAMES)
_fr = np.arange(N_FRAMES)
for leg in range(4):
    ax_z.plot(_fr, foot_hist[:, leg, 2], lw=1.8,
              color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_z.axhline(gnd_z, color='white', lw=0.8, ls=':', alpha=0.5, label='ground')
ax_z.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
            edgecolor=_gray, ncol=5)
z_cursor = ax_z.axvline(x=0, color='white', lw=1.5, ls='--')

# ── Hip Pitch (th2) 관절각 (우하)
ax_ang = fig.add_subplot(gs[2, 1])
_style_ax(ax_ang, 'Hip Pitch th2 [deg]', ylabel='[deg]')
ax_ang.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_ang.plot(_fr, np.degrees(joint_hist[:, leg, 1]), lw=1.8,
                color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_ang.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
              edgecolor=_gray, ncol=4)
ang_cursor = ax_ang.axvline(x=0, color='white', lw=1.5, ls='--')

all_cursors = [phase_cursor, z_cursor, ang_cursor]

# ══════════════════════════════════════════════════════════════
# 5. 애니메이션
# ══════════════════════════════════════════════════════════════

def init_anim():
    for leg in range(4):
        for ln in leg_links[leg]:
            ln.set_data([], [])
            ln.set_3d_properties([])
        leg_traces[leg].set_data([], [])
        leg_traces[leg].set_3d_properties([])
        swing_dots[leg].set_data([], [])
        swing_dots[leg].set_3d_properties([])
    info_text.set_text('')
    return []


def animate(fi):
    t = fi * DT

    for leg in range(4):
        q   = joint_hist[fi, leg]
        pts = forward_kinematics(q)           # 로컬 프레임
        hip = LEG_HIP_OFFSETS[leg]            # 월드 오프셋

        # 4개 링크 업데이트
        for k in range(4):
            A = hip + pts[k]
            B = hip + pts[k+1]
            leg_links[leg][k].set_data([A[0], B[0]], [A[1], B[1]])
            leg_links[leg][k].set_3d_properties([A[2], B[2]])

        # swing 중 발끝 강조
        pe = foot_hist[fi, leg]
        if swing_flag[fi, leg]:
            swing_dots[leg].set_data([pe[0]], [pe[1]])
            swing_dots[leg].set_3d_properties([pe[2]])
        else:
            swing_dots[leg].set_data([], [])
            swing_dots[leg].set_3d_properties([])

        # 발끝 궤적 trace (최근 1주기)
        trace_buf[leg][0].append(pe[0])
        trace_buf[leg][1].append(pe[1])
        trace_buf[leg][2].append(pe[2])
        tx = trace_buf[leg][0][-TRACE_LEN:]
        ty = trace_buf[leg][1][-TRACE_LEN:]
        tz = trace_buf[leg][2][-TRACE_LEN:]
        leg_traces[leg].set_data(tx, ty)
        leg_traces[leg].set_3d_properties(tz)

    # 커서 갱신
    for cur in all_cursors:
        cur.set_xdata([fi, fi])

    # 텍스트 정보
    sw_str = "  ".join(
        f"{LEG_NAMES[l]}:{'SW' if swing_flag[fi, l] else 'ST'}"
        for l in range(4)
    )
    deg = np.degrees(joint_hist[fi])
    msg = (f"t={t:.3f}s\n{sw_str}\n\n"
           + "\n".join(
               f"{LEG_NAMES[leg]} "
               f"th1={deg[leg,0]:+5.1f}° "
               f"th2={deg[leg,1]:+6.1f}° "
               f"th3={deg[leg,2]:+6.1f}° "
               f"th4={deg[leg,3]:+5.1f}°"
               for leg in range(4)
           ))
    info_text.set_text(msg)
    return []


ani = FuncAnimation(
    fig, animate, frames=N_FRAMES,
    init_func=init_anim,
    interval=DT * 1000,
    blit=False, repeat=True
)

plt.suptitle(
    f'4족 보행 시뮬레이터  |  {GAIT_TYPE.upper()}  |  '
    f'T={GAIT_PERIOD}s  swing={int(SWING_RATIO*100)}%  '
    f'stride={STRIDE_LENGTH*100:.0f}cm  h={STEP_HEIGHT*100:.0f}cm  '
    f'Vx={BODY_VX}m/s',
    color='white', fontsize=10
)
plt.show()
