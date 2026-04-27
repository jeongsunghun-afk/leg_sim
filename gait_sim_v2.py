"""
gait_sim_v2.py  —  4족 보행 Gait 시뮬레이터
v2: FR/FL 기구학을 leg_IK3_FB 기준으로 수정
  · Joint1 α = +π/2 (was -π/2)
  · θ1 공식 수정, Z 부호 제거, a4+a5 차감
  · FR/FL 5관절 (foot 링크 a5=0.045 추가)
  · elbow_up=False (Q_HOME θ3>0 branch)
"""

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []
mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════
# 0. 파라미터
# ══════════════════════════════════════════════════════════════
GAIT_TYPE   = 'trot'
DT          = 0.002
N_CYCLES    = 4

V           = 0.5
T           = 0.5
STEP_HEIGHT = 0.15
D           = 0.50

D_MIN        = V * T
STRIDE_D     = 0.50
assert STRIDE_D > D_MIN, f"stride({STRIDE_D}m) ≤ d_min({D_MIN}m)"
T_SW         = T * (1.0 - D)
T_ST         = T * D
STEP_LENGTH  = STRIDE_D / 2.0 - V * T_SW

BODY_VX      = V
GAIT_PERIOD  = T
SWING_RATIO  = 1.0 - D
STRIDE_LENGTH = STRIDE_D

BODY_FWD_F =  0.250
BODY_FWD_H = -0.250
BODY_LAT   =  0.050
BODY_X_H   = -0.100

# ── DH 파라미터 ──────────────────────────────────────────────
# FR/FL: 5관절, Joint1 α=+π/2  (leg_IK3_FB 기준)
DH_FRONT = [
    (+math.pi/2, 0.0,   0.0,    ),   # Joint 1: Hip Abduction  α=+π/2
    (0.0,        0.21,  0.0075, ),   # Joint 2: Hip Pitch
    (0.0,        0.235, 0.0,    ),   # Joint 3: Knee
    (0.0,        0.1,   0.0,    ),   # Joint 4: Lower leg
    (0.0,        0.045, 0.0,    ),   # Joint 5: Foot
]
# HR/HL: 4관절, Joint1 α=-π/2  (기존 유지)
DH_HIND = [
    (-math.pi/2, 0.0,   0.0,    ),   # Joint 1: Hip Abduction
    (0.0,        0.21,  0.0075, ),   # Joint 2: Hip Pitch
    (0.0,        0.21,  0.0,    ),   # Joint 3: Knee
    (0.0,        0.148, 0.0,    ),   # Joint 4: Lower leg
]

# front IK 상수 (DH_FRONT와 동기화)
_A2_F = 0.21; _A3_F = 0.235; _A4_F = 0.1; _A5_F = 0.045; _D2_F = 0.0075

# ── Q_HOME ──────────────────────────────────────────────────
Q_HOME_FRONT_DEG = [0.0, 157.5, 22.5, 30.6583, 59.3417]   # 5관절
Q_HOME_HIND_DEG  = [0.0, -150.0, -90.0, 90.0]             # 4관절
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]

PHI_FRONT    = Q_HOME_FRONT[1] + Q_HOME_FRONT[2] + Q_HOME_FRONT[3]
PHI_HIND     = Q_HOME_HIND[1]  + Q_HOME_HIND[2]  + Q_HOME_HIND[3]
THETA5_FRONT = PHI_FRONT + Q_HOME_FRONT[4]   # θ2+θ3+θ4+θ5 고정

Q_HOME_PER_LEG   = [Q_HOME_FRONT, Q_HOME_FRONT, Q_HOME_HIND, Q_HOME_HIND]
PHI_PER_LEG      = [PHI_FRONT, PHI_FRONT, PHI_HIND, PHI_HIND]

# ── 다리 레이아웃 ────────────────────────────────────────────
LEG_NAMES        = ['FR', 'FL', 'HR', 'HL']
LEG_COLORS       = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00']
LEG_DH           = [DH_FRONT, DH_FRONT, DH_HIND, DH_HIND]
N_JOINTS_PER_LEG = [5, 5, 4, 4]
N_JOINTS_MAX     = max(N_JOINTS_PER_LEG)

LEG_HIP_OFFSETS = np.array([
    [+BODY_FWD_F, -BODY_LAT, 0.0     ],   # FR
    [+BODY_FWD_F, +BODY_LAT, 0.0     ],   # FL
    [+BODY_FWD_H, -BODY_LAT, BODY_X_H],   # HR
    [+BODY_FWD_H, +BODY_LAT, BODY_X_H],   # HL
])

PHASE_OFFSETS = {
    'trot': [0.0, 0.5, 0.5, 0.0],
    'walk': [0.0, 0.5, 0.75, 0.25],
}

# ══════════════════════════════════════════════════════════════
# 1. 기구학
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

def forward_kinematics(thetas, dh=None):
    if dh is None:
        dh = DH_FRONT
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (alpha, a, d) in enumerate(dh):
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
        pts.append(T[:3, 3].copy())
    return pts

def analytical_ik_front(Px, Py, Pz, phi, theta5_target):
    """FR/FL IK: α=+π/2, 5관절 (leg_IK3_FB 기준)"""
    D2 = Px**2 + Py**2 - _D2_F**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)

    theta1 = math.atan2(_D2_F, -R) - math.atan2(-Py, Px)

    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py

    x3 = x_s - _A4_F * math.cos(phi) - _A5_F * math.cos(theta5_target)
    z3 = Pz   - _A4_F * math.sin(phi) - _A5_F * math.sin(theta5_target)

    cos_th3 = (x3**2 + z3**2 - _A2_F**2 - _A3_F**2) / (2.0 * _A2_F * _A3_F)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = math.acos(cos_th3)   # elbow_up=False

    theta2 = (math.atan2(z3, x3)
              - math.atan2(_A3_F * math.sin(theta3), _A2_F + _A3_F * math.cos(theta3)))
    theta4 = phi - theta2 - theta3
    theta5 = theta5_target - (theta2 + theta3 + theta4)

    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi
    return [wrap(theta1), wrap(theta2), wrap(theta3), wrap(theta4), wrap(theta5)]

def analytical_ik_hind(Px, Py, Pz, phi, dh):
    """HR/HL IK: α=-π/2, 4관절 (기존)"""
    a2 = dh[1][1]; a3 = dh[2][1]; a4 = dh[3][1]; d2 = dh[1][2]

    D2 = Px**2 + Py**2 - d2**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)
    theta1 = math.atan2(-Px, Py) - math.atan2(R, d2)

    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py
    Z   = -Pz

    x2 = x_s - a4 * math.cos(phi)
    z2 = Z   - a4 * math.sin(phi)

    cos_th3 = (x2**2 + z2**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = -math.acos(cos_th3)   # elbow_up=True
    theta2  = (math.atan2(z2, x2)
               - math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3)))
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
    t  = sw_t
    p1 = p_start + np.array([0, 0, step_height])
    p2 = p_end   + np.array([0, 0, step_height])
    return ((1-t)**3 * p_start
            + 3*(1-t)**2*t * p1
            + 3*(1-t)*t**2 * p2
            + t**3 * p_end)


def stance_foot_pos(st_t, p_contact, body_vel, stance_dur):
    return p_contact - body_vel * (st_t * stance_dur)

# ══════════════════════════════════════════════════════════════
# 3. 궤적 사전 계산
# ══════════════════════════════════════════════════════════════
N_FRAMES   = int(N_CYCLES * GAIT_PERIOD / DT)
sched      = GaitScheduler()
stance_dur = GAIT_PERIOD * (1.0 - SWING_RATIO)
body_vel   = np.array([BODY_VX, 0.0, 0.0])

# 홈 발끝: DH→sim 변환
# front(FR/FL): Rz(π) 적용 → [-DH_Z, -DH_Y, DH_X]
# hind(HR/HL) : 기존     → [ DH_Z,  DH_Y, DH_X]
def _dh_to_sim(pts_dh, is_front):
    if is_front:
        return np.array([-pts_dh[2], -pts_dh[1], pts_dh[0]])
    else:
        return np.array([ pts_dh[2],  pts_dh[1], pts_dh[0]])

home_foot_per_leg = [
    _dh_to_sim(forward_kinematics(Q_HOME_PER_LEG[leg], dh=LEG_DH[leg])[-1], leg < 2)
    for leg in range(4)
]
home_foot = home_foot_per_leg[0]   # FR 기준 (지면 높이 등)

joint_hist = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
foot_hist  = np.zeros((N_FRAMES, 4, 3))
phase_hist = np.zeros((N_FRAMES, 4))
swing_flag = np.zeros((N_FRAMES, 4), dtype=bool)

foot_contact    = [home_foot_per_leg[leg].copy() for leg in range(4)]
foot_sw_start   = [home_foot_per_leg[leg].copy() for leg in range(4)]
foot_local_prev = [home_foot_per_leg[leg].copy() for leg in range(4)]
prev_swing      = [sched.is_swing(leg, 0) for leg in range(4)]

print("─" * 55)
print(f"궤적 계산 중...  [{GAIT_TYPE}]  {N_CYCLES}사이클  {N_FRAMES}프레임")
print(f"  v={V}m/s  T={T}s  D={D}  →  d_min={D_MIN:.3f}m  d={STRIDE_D}m")
print(f"  T_sw={T_SW:.3f}s  T_st={T_ST:.3f}s  step(body)={STEP_LENGTH*1e3:.1f}mm  h={STEP_HEIGHT*1e3:.0f}mm")
print(f"  홈 발끝(FR): X={home_foot[0]*1e3:.1f}mm  "
      f"Y={home_foot[1]*1e3:.1f}mm  Z={home_foot[2]*1e3:.1f}mm")

for fi in range(N_FRAMES):
    t = fi * DT
    for leg in range(4):
        is_sw = sched.is_swing(leg, t)
        phase_hist[fi, leg] = sched.phase(leg, t)
        swing_flag[fi, leg] = is_sw

        if is_sw and not prev_swing[leg]:
            foot_sw_start[leg] = foot_local_prev[leg].copy()
        if not is_sw and prev_swing[leg]:
            foot_contact[leg] = foot_local_prev[leg].copy()

        if is_sw:
            sw_t     = sched.swing_t(leg, t)
            p_end    = home_foot_per_leg[leg] + np.array([STEP_LENGTH, 0, 0])
            foot_loc = swing_foot_pos(sw_t, foot_sw_start[leg], p_end)
        else:
            st_t     = sched.stance_t(leg, t)
            foot_loc = stance_foot_pos(st_t, foot_contact[leg], body_vel, stance_dur)

        foot_local_prev[leg] = foot_loc.copy()
        prev_swing[leg]      = is_sw
        foot_hist[fi, leg]   = LEG_HIP_OFFSETS[leg] + foot_loc

        # IK: sim→DH 변환 후 호출
        # front: Rz(π) 역변환 → Px=sim_Z, Py=-sim_Y, Pz=-sim_X
        # hind : 기존          → Px=sim_Z, Py= sim_Y, Pz= sim_X
        if leg < 2:   # FR, FL: 5관절 front IK
            q = analytical_ik_front(foot_loc[2], -foot_loc[1], -foot_loc[0],
                                    PHI_FRONT, THETA5_FRONT)
            if q is None:
                q = list(Q_HOME_FRONT)
        else:         # HR, HL: 4관절 hind IK
            q = analytical_ik_hind(foot_loc[2], foot_loc[1], foot_loc[0],
                                   PHI_HIND, dh=DH_HIND)
            if q is None:
                q = list(Q_HOME_HIND)

        nj = N_JOINTS_PER_LEG[leg]
        joint_hist[fi, leg, :nj] = q[:nj]

print("완료.")
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

def _style_ax(ax, title, xlabel='Frame', ylabel=''):
    ax.set_facecolor(_dark)
    ax.set_title(title, color='white', fontsize=9)
    ax.set_xlabel(xlabel, color='white', fontsize=8)
    ax.set_ylabel(ylabel, color='white', fontsize=8)
    ax.tick_params(colors=_gray)
    ax.grid(True, alpha=0.25, color=_gray)
    for sp in ax.spines.values():
        sp.set_edgecolor(_gray)

ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_facecolor(_dark)
reach = 0.65
ax3d.set_xlim(-reach, reach)
ax3d.set_ylim(-0.5, 0.5)
ax3d.set_zlim(-0.65, 0.15)
ax3d.set_xlabel('X (m)', color='white', labelpad=4)
ax3d.set_ylabel('Y (m)', color='white', labelpad=4)
ax3d.set_zlabel('Z (m)', color='white', labelpad=4)
ax3d.tick_params(colors=_gray)
ax3d.set_title(f'Gait Sim v2  [{GAIT_TYPE.upper()}]  v={V}m/s  T={T}s  D={D}',
               color='white', fontsize=10)
ax3d.view_init(elev=20, azim=-55)
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False

_bc = np.array([
    LEG_HIP_OFFSETS[0], LEG_HIP_OFFSETS[2],
    LEG_HIP_OFFSETS[3], LEG_HIP_OFFSETS[1],
    LEG_HIP_OFFSETS[0],
])
ax3d.plot(_bc[:,0], _bc[:,1], _bc[:,2], '-', color='white', lw=2.5, alpha=0.7)

gnd_z = home_foot[2]
xx, yy = np.meshgrid([-reach, reach], [-0.5, 0.5])
ax3d.plot_surface(xx, yy, np.full_like(xx, gnd_z), alpha=0.12, color='#888888')

_AX_COLORS = ['#ff4444', '#44ff44', '#4444ff']
_P       = np.array([[ 0, 0, 1],[0, 1,0],[1,0,0]], dtype=float)  # hind: DH→sim
_P_FRONT = np.array([[ 0, 0,-1],[0,-1,0],[1,0,0]], dtype=float)  # front: Rz(π) 후 DH→sim

for leg in range(4):
    h = LEG_HIP_OFFSETS[leg]
    ax3d.plot([h[0]], [h[1]], [h[2]], 'o', color=LEG_COLORS[leg], markersize=7, alpha=0.8)
    ax3d.text(h[0], h[1], h[2]+0.02, LEG_NAMES[leg], color=LEG_COLORS[leg], fontsize=7)

_BASE_FRAME_LEN = 0.12
for ax_i, lbl in enumerate(['X (fwd)', 'Y (lat)', 'Z (up)']):
    dv = np.zeros(3); dv[ax_i] = _BASE_FRAME_LEN
    ax3d.quiver(0, 0, 0, dv[0], dv[1], dv[2],
                color=_AX_COLORS[ax_i], linewidth=2.5, arrow_length_ratio=0.25)
    ax3d.text(dv[0]*1.15, dv[1]*1.15, dv[2]*1.15,
              lbl, color=_AX_COLORS[ax_i], fontsize=8, fontweight='bold')
ax3d.plot([0], [0], [0], 'w+', markersize=12, markeredgewidth=2.5, zorder=10)

# 다리 링크: 각 다리 관절 수에 맞게 생성
leg_links = []
for leg in range(4):
    nj = N_JOINTS_PER_LEG[leg]
    lns = [ax3d.plot([], [], [], '-o', color=LEG_COLORS[leg],
                     lw=2.5, markersize=5)[0] for _ in range(nj)]
    leg_links.append(lns)

TRACE_LEN  = int(GAIT_PERIOD / DT)
leg_traces = [ax3d.plot([], [], [], '-', color=LEG_COLORS[leg],
                        lw=1.2, alpha=0.6)[0] for leg in range(4)]
trace_buf  = [[[], [], []] for _ in range(4)]

swing_dots = [ax3d.plot([], [], [], 'o', color=LEG_COLORS[leg],
                        markersize=9, alpha=0.9)[0] for leg in range(4)]

FRAME_LEN   = 0.035
_jf_quivers = [
    [[None, None, None] for _ in range(N_JOINTS_PER_LEG[leg] + 1)]
    for leg in range(4)
]

info_text = ax3d.text2D(0.02, 0.98, "", transform=ax3d.transAxes,
                         color='white', fontfamily='monospace', fontsize=7.5, va='top')

# ── 위상 다이어그램
ax_phase = fig.add_subplot(gs[0, 1])
_style_ax(ax_phase, f'Gait Phase  [{GAIT_TYPE}]  (Bright=Swing)', ylabel='Leg')
ax_phase.set_xlim(0, N_FRAMES)
ax_phase.set_ylim(-0.5, 3.5)
ax_phase.set_yticks([0, 1, 2, 3])
ax_phase.set_yticklabels(LEG_NAMES[::-1], color='white')

for leg in range(4):
    row = 3 - leg
    in_sw = False; sw_start = 0
    for fi in range(N_FRAMES):
        if swing_flag[fi, leg] and not in_sw:
            sw_start = fi; in_sw = True
        elif not swing_flag[fi, leg] and in_sw:
            ax_phase.barh(row, fi-sw_start, left=sw_start, height=0.7,
                          color=LEG_COLORS[leg], alpha=0.85)
            in_sw = False
    if in_sw:
        ax_phase.barh(row, N_FRAMES-sw_start, left=sw_start, height=0.7,
                      color=LEG_COLORS[leg], alpha=0.85)

phase_cursor = ax_phase.axvline(x=0, color='white', lw=1.5, ls='--')

# ── 발끝 Z 높이
ax_z = fig.add_subplot(gs[1, 1])
_style_ax(ax_z, 'Step height [m]', ylabel='Z [m]')
ax_z.set_xlim(0, N_FRAMES)
_fr = np.arange(N_FRAMES)
for leg in range(4):
    ax_z.plot(_fr, foot_hist[:, leg, 2], lw=1.8,
              color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_z.axhline(gnd_z, color='white', lw=0.8, ls=':', alpha=0.5, label='ground')
ax_z.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
            edgecolor=_gray, ncol=5)
z_cursor = ax_z.axvline(x=0, color='white', lw=1.5, ls='--')

# ── Hip Pitch (th2)
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
        nj     = N_JOINTS_PER_LEG[leg]
        q      = joint_hist[fi, leg, :nj]
        pts_dh = forward_kinematics(q, dh=LEG_DH[leg])
        pts    = [_dh_to_sim(p, leg < 2) for p in pts_dh]
        hip    = LEG_HIP_OFFSETS[leg]

        for k in range(nj):
            A = hip + pts[k]
            B = hip + pts[k+1]
            leg_links[leg][k].set_data([A[0], B[0]], [A[1], B[1]])
            leg_links[leg][k].set_3d_properties([A[2], B[2]])

        pe = foot_hist[fi, leg]
        if swing_flag[fi, leg]:
            swing_dots[leg].set_data([pe[0]], [pe[1]])
            swing_dots[leg].set_3d_properties([pe[2]])
        else:
            swing_dots[leg].set_data([], [])
            swing_dots[leg].set_3d_properties([])

        trace_buf[leg][0].append(pe[0])
        trace_buf[leg][1].append(pe[1])
        trace_buf[leg][2].append(pe[2])
        leg_traces[leg].set_data(trace_buf[leg][0][-TRACE_LEN:],
                                  trace_buf[leg][1][-TRACE_LEN:])
        leg_traces[leg].set_3d_properties(trace_buf[leg][2][-TRACE_LEN:])

        Pmat = _P_FRONT if leg < 2 else _P
        T_dh = np.eye(4)
        for j in range(nj + 1):
            orig_sim = Pmat @ T_dh[:3, 3]
            pos = hip + orig_sim
            for ax_i in range(3):
                dv = Pmat @ T_dh[:3, ax_i]
                if ax_i == 2:
                    dv = -dv
                if _jf_quivers[leg][j][ax_i] is not None:
                    _jf_quivers[leg][j][ax_i].remove()
                _jf_quivers[leg][j][ax_i] = ax3d.quiver(
                    pos[0], pos[1], pos[2],
                    dv[0]*FRAME_LEN, dv[1]*FRAME_LEN, dv[2]*FRAME_LEN,
                    color=_AX_COLORS[ax_i], linewidth=1.0, arrow_length_ratio=0.3
                )
            if j < nj:
                T_dh = T_dh @ _dh_matrix(
                    LEG_DH[leg][j][0], LEG_DH[leg][j][1],
                    LEG_DH[leg][j][2], float(q[j])
                )

    for cur in all_cursors:
        cur.set_xdata([fi, fi])

    sw_str = "  ".join(
        f"{LEG_NAMES[l]}:{'SW' if swing_flag[fi, l] else 'ST'}"
        for l in range(4)
    )
    deg   = np.degrees(joint_hist[fi])
    lines = []
    for leg in range(4):
        d = deg[leg]
        if N_JOINTS_PER_LEG[leg] == 5:
            lines.append(f"{LEG_NAMES[leg]} "
                         f"th1={d[0]:+5.1f}° th2={d[1]:+6.1f}° th3={d[2]:+6.1f}° "
                         f"th4={d[3]:+5.1f}° th5={d[4]:+5.1f}°")
        else:
            lines.append(f"{LEG_NAMES[leg]} "
                         f"th1={d[0]:+5.1f}° th2={d[1]:+6.1f}° th3={d[2]:+6.1f}° "
                         f"th4={d[3]:+5.1f}°")
    info_text.set_text(f"t={t:.3f}s\n{sw_str}\n\n" + "\n".join(lines))
    return []


ani = FuncAnimation(
    fig, animate, frames=N_FRAMES,
    init_func=init_anim,
    interval=DT * 1000,
    blit=False, repeat=True
)

plt.suptitle(
    f'Gait Sim v2  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  '
    f'd={STRIDE_D}m(d_min={D_MIN:.2f}m)  '
    f'T_sw={T_SW:.2f}s  step={STEP_LENGTH*1e3:.0f}mm  h={STEP_HEIGHT*1e2:.0f}cm',
    color='white', fontsize=9
)
plt.show()
