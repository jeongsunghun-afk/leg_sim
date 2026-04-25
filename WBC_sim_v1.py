"""
WBC_sim_v1.py  —  4족 보행 Gait 시뮬레이터
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

[지금 TODO로 남긴 부분]
  phi (ankle 각도) 실제 값 조정
  swing 착지 목표점 계산 (스트라이드 방향 × 길이)
  MCX 전송 연결
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
GAIT_TYPE   = 'trot'   # 'trot' | 'walk'
DT          = 0.002    # 제어 주기 [s] (500Hz)
N_CYCLES    = 4        # 시뮬레이션 보행 사이클 수

# ── 보행 기본 파라미터 ────────────────────────────────────────────
V           = 0.5    # 보행속도          v    [m/s]
T           = 0.5    # 보행주기          T    [s]
STEP_HEIGHT = 0.15   # 보행높이                [m]

# ── duty factor D (stance 비율) ──────────────────────────────────
#   평보(Walk)  : swing 25~40%  →  D = 0.60~0.75
#   속보(Trot)  : swing 50%     →  D = 0.50
#   구보(Gallop): swing 60~80%  →  D = 0.20~0.40
D = 0.50             # duty factor (stance 비율), trot

# ── 유도 파라미터 ─────────────────────────────────────────────────
#   최소 보행길이   d_min = v · T
#   실제 보행길이   d  >  d_min
#   swing 주기      T_sw = T · (1 - D)
#   보폭(body frame) step = d/2 - v · T_sw
D_MIN        = V * T              # 최소 보행길이 d_min [m]
STRIDE_D     = 0.50               # 실제 보행길이 d     [m]  (반드시 d > d_min)
assert STRIDE_D > D_MIN, f"stride({STRIDE_D}m) ≤ d_min({D_MIN}m)"
T_SW         = T * (1.0 - D)     # swing 주기    T_sw  [s]
T_ST         = T * D             # stance 주기   T_st  [s]
STEP_LENGTH  = STRIDE_D / 2.0 - V * T_SW  # 보폭 (body frame) [m]

# ── 기존 변수 매핑 (GaitScheduler / 궤적 계산 호환) ─────────────
BODY_VX      = V
GAIT_PERIOD  = T
SWING_RATIO  = 1.0 - D           # GaitScheduler: swing 비율 = 1 - D
STRIDE_LENGTH = STRIDE_D

# ── 몸체 힙 마운트 위치 (robot base frame: z=전후, y=좌우, x=상하) ──
# FR/FL: z=+250mm, x=  0mm, y=±50mm
# HR/HL: z=-250mm, x=-100mm, y=±50mm
BODY_FWD_F =  0.250   # 전방 힙 z [m]
BODY_FWD_H = -0.250   # 후방 힙 z [m]
BODY_LAT   =  0.050   # 좌우 힙 y [m]
BODY_X_H   = -0.100   # 후방 힙 x offset [m]

# ── DH 파라미터 (α, a, d) — FR/FL 및 HR/HL 공통 (4관절) ─────────
DH_FRONT = [
    (-math.pi/2, 0.0,   0.0,    ),   # Joint 1: Hip Abduction
    (0.0,        0.21,  0.0075, ),   # Joint 2: Hip Pitch
    (0.0,        0.235, 0.0,    ),   # Joint 3: Knee
    (0.0,        0.1,   0.0,    ),   # Joint 4: Lower leg
    # (0.0,      0.045, 0.0,    ),   # Joint 5: Foot (추후 5관절 확장 시 활성화)
]
DH_HIND = [
    (-math.pi/2, 0.0,   0.0,    ),   # Joint 1: Hip Abduction
    (0.0,        0.21,  0.0075, ),   # Joint 2: Hip Pitch
    (0.0,        0.21, 0.0,    ),   # Joint 3: Knee
    (0.0,        0.148,   0.0,    ),   # Joint 4: Lower leg
    # (0.0,      0.045, 0.0,    ),   # Joint 5: Foot (추후 5관절 확장 시 활성화)
]
# ── 홈 각도 — 전방/후방 다리 별도 설정 [deg → rad] ──────────────
# θ2: 물리각 + 90° 바이어스 (robot base: 중력=-x, DH zero=+x 수평)
Q_HOME_FRONT_DEG = [0.0, 157.5,  22.5, 30.6583]   # FR / FL
Q_HOME_HIND_DEG  = [0.0, -150.0, -90.0, 90.0  ]   # HR / HL (leg_sim_v4 기준)
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]

PHI_FRONT = Q_HOME_FRONT[1] + Q_HOME_FRONT[2] + Q_HOME_FRONT[3]
PHI_HIND  = Q_HOME_HIND[1]  + Q_HOME_HIND[2]  + Q_HOME_HIND[3]

# 다리 인덱스별 홈 각도 / phi [FR, FL, HR, HL]
Q_HOME_PER_LEG = [Q_HOME_FRONT, Q_HOME_FRONT, Q_HOME_HIND, Q_HOME_HIND]
PHI_PER_LEG    = [PHI_FRONT, PHI_FRONT, PHI_HIND, PHI_HIND]

# ── 다리 레이아웃 [FR, FL, HR, HL] ──────────────────────────────
LEG_NAMES  = ['FR', 'FL', 'HR', 'HL']
LEG_COLORS = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00']
LEG_DH     = [DH_FRONT, DH_FRONT, DH_HIND, DH_HIND]

# 힙 위치 오프셋 — sim frame: x=전후(=robot_z), y=좌우(=robot_y), z=상하(=robot_x)
# HR/HL 의 robot_x=-100mm → sim_Z=BODY_X_H 로 반영 (전방힙 대비 100mm 낮음)
LEG_HIP_OFFSETS = np.array([
    [+BODY_FWD_F, -BODY_LAT, 0.0     ],   # FR: sim_Z=0 (기준)
    [+BODY_FWD_F, +BODY_LAT, 0.0     ],   # FL: sim_Z=0
    [+BODY_FWD_H, -BODY_LAT, BODY_X_H],   # HR: sim_Z=robot_X=-100mm
    [+BODY_FWD_H, +BODY_LAT, BODY_X_H],   # HL: sim_Z=robot_X=-100mm
])

# 위상 오프셋 [FR, FL, HR, HL]
PHASE_OFFSETS = {
    'trot': [0.0, 0.5, 0.5, 0.0],   # FR+HL 동시, FL+HR 동시
    'walk': [0.0, 0.5, 0.75, 0.25], # 순차
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

def forward_kinematics(thetas, dh=None):
    """joint angles → 관절 원점 리스트 [힙~발끝]  (DH 미지정 시 DH_FRONT 사용)"""
    if dh is None:
        dh = DH_FRONT
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (alpha, a, d) in enumerate(dh):
        T = T @ _dh_matrix(alpha, a, d, thetas[i])
        pts.append(T[:3, 3].copy())
    return pts

def analytical_ik(Px, Py, Pz, phi, dh, elbow_up=True):
    """발끝 xyz (다리 로컬 프레임) → joint angles 4개. 실패 시 None
    phi = θ2+θ3+θ4 (발끝 방위각 고정 조건)
    dh  = 해당 다리의 DH 파라미터 테이블 (링크 길이 추출)
    """
    a2 = dh[1][1]; a3 = dh[2][1]; a4 = dh[3][1]; d2 = dh[1][2]

    D2 = Px**2 + Py**2 - d2**2
    if D2 < 0:
        return None
    R      = math.sqrt(D2)
    theta1 = math.atan2(-Px, Py) - math.atan2(R, d2)
    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s    = c1 * Px + s1 * Py
    Z      = -Pz

    x2 = x_s - a4 * math.cos(phi)
    z2 = Z   - a4 * math.sin(phi)

    cos_th3 = (x2**2 + z2**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = -math.acos(cos_th3) if elbow_up else math.acos(cos_th3)
    theta2  = (math.atan2(z2, x2)
               - math.atan2(a3 * math.sin(theta3),
                             a2 + a3 * math.cos(theta3)))
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

# 홈 자세 발끝 위치 — DH 프레임 → sim 프레임 변환 [DH_Z, DH_Y, DH_X]
# DH_X=robot_X(up/down), DH_Z=robot_Z(forward=sim_X), DH_X=sim_Z(vertical)
home_foot_per_leg = [
    np.array(forward_kinematics(Q_HOME_PER_LEG[leg], dh=LEG_DH[leg])[-1])[[2, 1, 0]]
    for leg in range(4)
]
home_foot = home_foot_per_leg[0]   # 출력/시각화 기준 (FR)

N_JOINTS = len(DH_FRONT)   # 현재 4

print("─" * 55)
print(f"궤적 계산 중...  [{GAIT_TYPE}]  {N_CYCLES}사이클  {N_FRAMES}프레임")
print(f"  v={V}m/s  T={T}s  D={D}  →  d_min={D_MIN:.3f}m  d={STRIDE_D}m")
print(f"  T_sw={T_SW:.3f}s  T_st={T_ST:.3f}s  step(body)={STEP_LENGTH*1e3:.1f}mm  h={STEP_HEIGHT*1e3:.0f}mm")
print(f"  홈 발끝(FR): X={home_foot[0]*1e3:.1f}mm  "
      f"Y={home_foot[1]*1e3:.1f}mm  Z={home_foot[2]*1e3:.1f}mm")

# 기록 배열
joint_hist = np.zeros((N_FRAMES, 4, N_JOINTS))  # [frame, leg, joint]
foot_hist  = np.zeros((N_FRAMES, 4, 3))          # [frame, leg, xyz] 월드 프레임
phase_hist = np.zeros((N_FRAMES, 4))
swing_flag = np.zeros((N_FRAMES, 4), dtype=bool)

# 다리별 상태 (각 다리의 홈 발끝으로 초기화)
foot_contact    = [home_foot_per_leg[leg].copy() for leg in range(4)]
foot_sw_start   = [home_foot_per_leg[leg].copy() for leg in range(4)]
foot_local_prev = [home_foot_per_leg[leg].copy() for leg in range(4)]
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
            # 착지 목표: 다리별 홈(neutral) + step = d/2 - v·T_sw
            p_end    = home_foot_per_leg[leg] + np.array([STEP_LENGTH, 0, 0])
            foot_loc = swing_foot_pos(sw_t, foot_sw_start[leg], p_end)
        else:
            st_t     = sched.stance_t(leg, t)
            foot_loc = stance_foot_pos(st_t, foot_contact[leg], body_vel, stance_dur)

        foot_local_prev[leg] = foot_loc.copy()
        prev_swing[leg]      = is_sw

        # 월드 프레임 = 힙 오프셋 + 로컬 발끝
        foot_hist[fi, leg] = LEG_HIP_OFFSETS[leg] + foot_loc

        # IK: sim 프레임 → DH 프레임 변환 후 호출
        # sim_X=DH_Z, sim_Y=DH_Y, sim_Z=DH_X → IK 입력 (DH_X, DH_Y, DH_Z) = (sim_Z, sim_Y, sim_X)
        q = analytical_ik(foot_loc[2], foot_loc[1], foot_loc[0],
                          phi=PHI_PER_LEG[leg], dh=LEG_DH[leg])
        if q is None:
            q = list(Q_HOME_PER_LEG[leg])
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
ax3d.set_title(f'4족 {GAIT_TYPE.upper()}  |  v={V}m/s  T={T}s  D={D}',
               color='white', fontsize=10)
ax3d.view_init(elev=20, azim=-55)
ax3d.xaxis.pane.fill = ax3d.yaxis.pane.fill = ax3d.zaxis.pane.fill = False

# 몸체 사각형 (힙 위치 기반)
_bc = np.array([
    LEG_HIP_OFFSETS[0],   # FR
    LEG_HIP_OFFSETS[2],   # HR
    LEG_HIP_OFFSETS[3],   # HL
    LEG_HIP_OFFSETS[1],   # FL
    LEG_HIP_OFFSETS[0],   # FR (닫기)
])
ax3d.plot(_bc[:,0], _bc[:,1], _bc[:,2], '-', color='white', lw=2.5, alpha=0.7)

# 지면 (발끝 Z 기준)
gnd_z = home_foot[2]
xx, yy = np.meshgrid([-reach, reach], [-0.5, 0.5])
ax3d.plot_surface(xx, yy, np.full_like(xx, gnd_z), alpha=0.12, color='#888888')

# 좌표축 색상 / DH→sim 변환 행렬 (이후 base frame & 관절 quiver 공용)
_AX_COLORS = ['#ff4444', '#44ff44', '#4444ff']   # X=빨강, Y=초록, Z=파랑
_P = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=float)  # DH→sim 좌표 변환 (X↔Z)

# 힙 점 표시
for leg in range(4):
    h = LEG_HIP_OFFSETS[leg]
    ax3d.plot([h[0]], [h[1]], [h[2]], 'o', color=LEG_COLORS[leg], markersize=7, alpha=0.8)
    ax3d.text(h[0], h[1], h[2]+0.02, LEG_NAMES[leg], color=LEG_COLORS[leg], fontsize=7)

# Robot base frame 좌표축 (원점: 몸체 중심)
_BASE_FRAME_LEN = 0.12
_BASE_LABELS    = ['X (fwd)', 'Y (lat)', 'Z (up)']
for ax_i in range(3):
    dv = np.zeros(3); dv[ax_i] = _BASE_FRAME_LEN
    ax3d.quiver(0, 0, 0, dv[0], dv[1], dv[2],
                color=_AX_COLORS[ax_i], linewidth=2.5, arrow_length_ratio=0.25)
    ax3d.text(dv[0] * 1.15, dv[1] * 1.15, dv[2] * 1.15,
              _BASE_LABELS[ax_i], color=_AX_COLORS[ax_i], fontsize=8, fontweight='bold')
ax3d.plot([0], [0], [0], 'w+', markersize=12, markeredgewidth=2.5, zorder=10)

# 다리 링크 선 (N_JOINTS 구간)
leg_links = []
for leg in range(4):
    lns = [ax3d.plot([], [], [], '-o', color=LEG_COLORS[leg],
                     lw=2.5, markersize=5)[0] for _ in range(N_JOINTS)]
    leg_links.append(lns)

# 발끝 trace (최근 1주기)
TRACE_LEN  = int(GAIT_PERIOD / DT)
leg_traces = [ax3d.plot([], [], [], '-', color=LEG_COLORS[leg],
                        lw=1.2, alpha=0.6)[0] for leg in range(4)]
trace_buf  = [[[], [], []] for _ in range(4)]

# swing 중 발끝 마커
swing_dots = [ax3d.plot([], [], [], 'o', color=LEG_COLORS[leg],
                        markersize=9, alpha=0.9)[0] for leg in range(4)]

# 관절 좌표축 quiver: [leg][frame 0=base..N_JOINTS][axis 0=X 1=Y 2=Z]
FRAME_LEN   = 0.035   # 화살표 길이 [m]
_jf_quivers = [[[None, None, None] for _ in range(N_JOINTS + 1)] for _ in range(4)]

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
        pts_dh = forward_kinematics(q, dh=LEG_DH[leg])   # DH 프레임
        # DH → sim 프레임: [DH_Z, DH_Y, DH_X]
        pts = [np.array([p[2], p[1], p[0]]) for p in pts_dh]
        hip = LEG_HIP_OFFSETS[leg]

        # N_JOINTS 링크 업데이트
        for k in range(N_JOINTS):
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

        # 관절 좌표축 시각화 (DH → sim 변환)
        T_dh = np.eye(4)
        for j in range(N_JOINTS + 1):
            orig_sim = _P @ T_dh[:3, 3]
            pos = hip + orig_sim
            for ax_i in range(3):
                dv = _P @ T_dh[:3, ax_i]   # DH 축 벡터 → sim 프레임
                if ax_i == 2:
                    dv = -dv  # DH_Z 반전: det(_P)=-1 보정 → 오른손법칙 복원, joint1 +X = base +Z 유지
                if _jf_quivers[leg][j][ax_i] is not None:
                    _jf_quivers[leg][j][ax_i].remove()
                _jf_quivers[leg][j][ax_i] = ax3d.quiver(
                    pos[0], pos[1], pos[2],
                    dv[0] * FRAME_LEN, dv[1] * FRAME_LEN, dv[2] * FRAME_LEN,
                    color=_AX_COLORS[ax_i], linewidth=1.0, arrow_length_ratio=0.3
                )
            if j < N_JOINTS:
                T_dh = T_dh @ _dh_matrix(
                    LEG_DH[leg][j][0], LEG_DH[leg][j][1],
                    LEG_DH[leg][j][2], float(q[j])
                )

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
    f'v={V}m/s  T={T}s  D={D}  '
    f'd={STRIDE_D}m(d_min={D_MIN:.2f}m)  '
    f'T_sw={T_SW:.2f}s  step={STEP_LENGTH*1e3:.0f}mm  h={STEP_HEIGHT*1e2:.0f}cm',
    color='white', fontsize=9
)
plt.show()
