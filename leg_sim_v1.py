"""
leg_statics_v2.py
sim2.py 기반 3D 다리 시뮬레이터 + 궤적 애니메이션
leg_dynamics_sargent_v2.py의 궤적 생성 방식 참조

[궤적 모드]
  TRAJ_MODE = 'jump' : 수직 점프 (준비→도약→착지), 5차 다항식 스플라인
  TRAJ_MODE = 'gait' : 보행 (Stance+Swing), 발끝 X-Z 평면 궤적

[좌표 기준]
  힙 = 원점.  다리가 -X 방향으로 신전.
  굴곡(Crouch): 발끝 X 증가 (+X, 다리 수축)
  신전(Push)  : 발끝 X 감소 (-X, 다리 연장)
  들어올림    : 발끝 Z 증가

[IK]
  해석적 역기구학 (leg_IK3.py 참조)
  φ = θ2+θ3+θ4 유지,  θ5 = theta5_target - φ
"""

import numpy as np
import math
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
TRAJ_MODE = 'jump'   # 'jump' | 'gait'
DT        = 0.005    # 시간 간격 [s] 5ms → 200Hz 제어 주기 가정
N_STEPS   = 240      # waypoint 총 개수 (총 시간 = N_STEPS × DT)
# 총 궤적 시간 = n × DT = 240 × 5ms = 1.2초

# DH 파라미터 [alpha, a(m), d(m)]
DH_PARAMS = [
    (-math.pi/2, 0.0,    0.0    ),   # Joint 1 : Hip Abduction
    (0.0,        0.21,   0.0075 ),   # Joint 2 : Hip Pitch
    (0.0,        0.21,   0.0    ),   # Joint 3 : Knee
    (0.0,        0.148,  0.0    ),   # Joint 4 : Ankle
    (0.0,        0.0,  0.0    ),   # Joint 5 : Toe
]

Q_INIT = [math.radians(a) for a in [0,  -90,   0,  0,  0]]
Q_HOME = [math.radians(a) for a in [0, -150, -90, 90, 60]]

# ══════════════════════════════════════════════════════════════
# 1. FK / IK (해석적, leg_IK3.py 참조)
# ══════════════════════════════════════════════════════════════

# 링크 파라미터 (DH_PARAMS와 동기화)
_A2 = DH_PARAMS[1][1]
_A3 = DH_PARAMS[2][1]
_A4 = DH_PARAMS[3][1]
_D2 = DH_PARAMS[1][2]

def get_dh_matrix(alpha, a, d, theta):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ], dtype=float)

def forward_kinematics(thetas):
    T = np.eye(4)
    pts = [np.zeros(3)]
    for i, (alpha, a, d) in enumerate(DH_PARAMS):
        T = T @ get_dh_matrix(alpha, a, d, thetas[i])
        pts.append(T[:3, 3].copy())
    return pts

def analytical_ik(Px, Py, Pz, phi, theta5_target, elbow_up=True):
    """
    해석적 역기구학 (leg_IK3.py 참조)
      θ1 = atan2(-Px, Py) - atan2(R, d2),  R = sqrt(Px²+Py²-d2²)
      x3 = x_s - a4·cos(φ),  z3 = Z - a4·sin(φ)
      cos(θ3) = (x3²+z3²-a2²-a3²) / (2·a2·a3)
      θ2 = atan2(z3, x3) - atan2(a3·sinθ3, a2+a3·cosθ3)
      θ4 = φ - θ2 - θ3,  θ5 = theta5_target - φ
    """
    D2 = Px**2 + Py**2 - _D2**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)

    theta1 = math.atan2(-Px, Py) - math.atan2(R, _D2)

    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py
    Z   = -Pz

    x3 = x_s - _A4 * math.cos(phi)
    z3 = Z   - _A4 * math.sin(phi)

    cos_th3 = (x3**2 + z3**2 - _A2**2 - _A3**2) / (2.0 * _A2 * _A3)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = -math.acos(cos_th3) if elbow_up else math.acos(cos_th3)

    theta2 = (math.atan2(z3, x3)
              - math.atan2(_A3 * math.sin(theta3),
                           _A2 + _A3 * math.cos(theta3)))

    theta4 = phi - theta2 - theta3
    theta5 = theta5_target - (theta2 + theta3 + theta4)

    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    return [wrap(theta1), wrap(theta2), wrap(theta3), wrap(theta4), wrap(theta5)]

# ══════════════════════════════════════════════════════════════
# 2. 궤적 생성  (leg_dynamics_sargent_v2.py 참조)
# ══════════════════════════════════════════════════════════════

def solve_quintic_spline(t0, tf, p0, v0, a0, pf, vf, af):
    """5차 다항식 계수 계산 (경계조건: 위치·속도·가속도)"""
    T_mat = np.array([
        [1, t0, t0**2,    t0**3,    t0**4,    t0**5],
        [0,  1, 2*t0,  3*t0**2,  4*t0**3,  5*t0**4],
        [0,  0,    2,    6*t0, 12*t0**2, 20*t0**3 ],
        [1, tf, tf**2,    tf**3,    tf**4,    tf**5],
        [0,  1, 2*tf,  3*tf**2,  4*tf**3,  5*tf**4],
        [0,  0,    2,    6*tf, 12*tf**2, 20*tf**3 ],
    ])
    return np.linalg.solve(T_mat, np.array([p0, v0, a0, pf, vf, af]))

def eval_quintic(c, t):
    pos = c[0]+c[1]*t    +c[2]*t**2    +c[3]*t**3    +c[4]*t**4    +c[5]*t**5
    vel = c[1]+2*c[2]*t  +3*c[3]*t**2  +4*c[4]*t**3  +5*c[5]*t**4
    acc = 2*c[2]+6*c[3]*t+12*c[4]*t**2+20*c[5]*t**3
    return pos, vel, acc

def make_jump_trajectory(start, h_crouch=0.04, h_jump=0.06, n=240):
    """
    수직 점프 궤적 (발끝 X축 방향 이동)
    Phase 1 (준비):  발끝 X 증가  → 다리 굴곡, 힙 하강
    Phase 2 (도약):  발끝 X 감소  → 다리 신전, 힙 상승
    Phase 3 (착지):  발끝 X 복귀
    """
    n1 = n // 3;  n2 = n // 6;  n3 = n - n1 - n2
    x0       = start[0]
    x_crouch = x0 + h_crouch      # +X: 굴곡
    x_push   = x0 - h_jump        # -X: 신전
    T1 = n1*DT;  T2 = n2*DT;  T3 = n3*DT

    c1 = solve_quintic_spline(0, T1, x0,       0, 0, x_crouch, 0, 0)
    c2 = solve_quintic_spline(0, T2, x_crouch, 0, 0, x_push,   0, 0)
    c3 = solve_quintic_spline(0, T3, x_push,   0, 0, x0,       0, 0)

    pts = []
    for j in range(n1):
        x, _, _ = eval_quintic(c1, j*DT)
        pts.append([x, start[1], start[2]])
    for j in range(n2):
        x, _, _ = eval_quintic(c2, j*DT)
        pts.append([x, start[1], start[2]])
    for j in range(n3):
        x, _, _ = eval_quintic(c3, j*DT)
        pts.append([x, start[1], start[2]])

    return np.array(pts), (n1, n1 + n2)

def make_gait_trajectory(start, step_x=0.06, lift=0.04, n=120):
    """
    보행 궤적 (X-Z 평면)
    Stance (전반): 발끝 X 후방 이동  (+X, 힙 전진에 대응)
    Swing  (후반): 발끝 X 전방 이동  (-X) + Z 들어올림
    """
    n_stance = n // 2;  n_swing = n - n_stance
    t1 = np.linspace(0, 1, n_stance)

    # Stance: x가 +X 방향으로 선형 이동, z 고정
    sx = start[0] + step_x * t1
    sz = np.full(n_stance, start[2])

    # Swing: x가 -X 방향으로 복귀, z 사인 들어올림
    t2 = np.linspace(0, np.pi, n_swing)
    wx = (start[0] + step_x) - step_x * (1 - np.cos(t2)) / 2
    wz = start[2] + lift * np.sin(t2)

    # 마지막 Swing 끝이 start[0]으로 정확히 복귀
    wx[-1] = start[0]
    wz[-1] = start[2]

    pts_stance = np.column_stack([sx, np.full(n_stance, start[1]), sz])
    pts_swing  = np.column_stack([wx, np.full(n_swing,  start[1]), wz])
    return np.vstack([pts_stance, pts_swing]), None

# ══════════════════════════════════════════════════════════════
# 3. 사전 계산: IK로 관절각 이력 생성
# ══════════════════════════════════════════════════════════════
print("궤적 IK 계산 중...")
toe_start = np.array(forward_kinematics(Q_HOME)[4])   # O4 (ankle)
print(f"  O4 시작점 (Q_HOME FK): "
      f"X={toe_start[0]*1000:.1f}mm  Y={toe_start[1]*1000:.1f}mm  Z={toe_start[2]*1000:.1f}mm")

if TRAJ_MODE == 'jump':
    trajectory, phase_idx = make_jump_trajectory(
        toe_start, h_crouch=0.04, h_jump=0.06, n=N_STEPS)
    mode_label = '수직 점프 (준비→도약→착지)'
else:
    trajectory, phase_idx = make_gait_trajectory(
        toe_start, step_x=0.06, lift=0.04, n=N_STEPS)
    mode_label = '보행 (Stance→Swing)'

# 각 궤적 점에 대해 해석적 IK 수행
theta_hist = [Q_HOME[:]]
current    = Q_HOME[:]
for target in trajectory:
    phi          = current[1] + current[2] + current[3]
    theta5_target = current[1] + current[2] + current[3] + current[4]
    result = analytical_ik(target[0], target[1], target[2],
                           phi, theta5_target, elbow_up=True)
    if result is not None:
        current = result
    theta_hist.append(current[:])

theta_hist = np.array(theta_hist)   # (n_frames, 5)
n_frames   = len(theta_hist)
print(f"완료: {n_frames} 프레임  |  모드: {mode_label}")

# ── 궤적 .txt 저장 (th1~th4, 단위: deg)
_out_path = f"trajectory_{TRAJ_MODE}.txt"
_header   = f"# {mode_label}  |  {n_frames} frames  |  DT={DT}s\n"
_header  += "frame\tth1_deg\tth2_deg\tth3_deg\tth4_deg"
_data_deg = np.degrees(theta_hist[:, :4])   # (n_frames, 4)
_rows     = np.column_stack([np.arange(n_frames), _data_deg])
np.savetxt(_out_path, _rows, delimiter='\t', header=_header,
           fmt=['%d', '%.6f', '%.6f', '%.6f', '%.6f'], comments='')
print(f"궤적 저장 완료: {_out_path}")

# 발끝 FK 이력
toe_hist = np.array([forward_kinematics(th)[4] for th in theta_hist])   # O4

# ══════════════════════════════════════════════════════════════
# 4. 시각화 설정
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 8))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(2, 2, figure=fig, wspace=0.4, hspace=0.45,
                        left=0.05, right=0.97, top=0.93, bottom=0.08)

# ── 3D 다리 뷰
ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_facecolor('#16213e')
reach = 0.55
ax3d.set_xlim(-reach, reach)
ax3d.set_ylim(-reach, reach)
ax3d.set_zlim(-reach, reach)
ax3d.set_xlabel('X (m)', color='white', labelpad=6)
ax3d.set_ylabel('Y (m)', color='white', labelpad=6)
ax3d.set_zlabel('Z (m)', color='white', labelpad=6)
ax3d.tick_params(colors='gray')
ax3d.set_title(f'3D 다리  |  {mode_label}', color='white', fontsize=10)
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False

# ── 관절각 그래프
ax_ang = fig.add_subplot(gs[0, 1])
ax_ang.set_facecolor('#16213e')
ang_min = min(math.degrees(theta_hist[:, k].min()) for k in range(4)) - 10
ang_max = max(math.degrees(theta_hist[:, k].max()) for k in range(4)) + 10
ax_ang.set_xlim(0, n_frames)
ax_ang.set_ylim(ang_min, ang_max)
ax_ang.set_xlabel('프레임', color='white', fontsize=9)
ax_ang.set_ylabel('[deg]', color='white', fontsize=9)
ax_ang.set_title('관절각 변화 (θ1~θ4)', color='white', fontsize=10)
ax_ang.tick_params(colors='gray')
ax_ang.grid(True, alpha=0.25, color='gray')
for sp in ax_ang.spines.values():
    sp.set_edgecolor('gray')

# ── 발끝 X·Y·Z 위치 vs 프레임
ax_traj = fig.add_subplot(gs[1, 1])
ax_traj.set_facecolor('#16213e')
ax_traj.set_xlim(0, n_frames)
_traj_min = min(toe_hist[:, k].min() for k in range(3)) - 0.02
_traj_max = max(toe_hist[:, k].max() for k in range(3)) + 0.02
ax_traj.set_ylim(_traj_min, _traj_max)
ax_traj.set_xlabel('프레임', color='white', fontsize=9)
ax_traj.set_ylabel('pos (m)', color='white', fontsize=9)
ax_traj.set_title('발끝 X·Y·Z 위치', color='white', fontsize=10)
ax_traj.tick_params(colors='gray')
ax_traj.grid(True, alpha=0.25, color='gray')
for sp in ax_traj.spines.values():
    sp.set_edgecolor('gray')
# 목표 X 궤적 표시
ax_traj.plot(range(len(trajectory)), trajectory[:, 0],
             '--', color='#888888', lw=1.2, alpha=0.6, label='목표 X')

# ── 링크 컬러
_COLORS     = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00', '#cc88ff']
_ANG_COLORS = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00']
_AXIS_COLORS = ['#ff4444', '#44ff44', '#4444ff']
AXIS_LEN    = 0.06

# 3D 링크 라인
link_lines = [
    ax3d.plot([], [], [], '-o', color=_COLORS[i], lw=3, markersize=8)[0]
    for i in range(5)
]
trace_line, = ax3d.plot([], [], [], '-', color='#ff88aa', lw=1.5, alpha=0.8, label='발끝 궤적')
info_text   = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes,
                          color='white', fontfamily='monospace', fontsize=8,
                          va='top')
phase_text  = ax3d.text2D(0.02, 0.72, "", transform=ax3d.transAxes,
                          color='yellow', fontsize=9, fontweight='bold', va='top')

# 관절각 라인
ang_data  = [[] for _ in range(4)]
ang_lines = [
    ax_ang.plot([], [], lw=1.8, color=_ANG_COLORS[k], label=f'θ{k+1}')[0]
    for k in range(4)
]
ax_ang.legend(loc='upper right', fontsize=8,
              facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

# 위상 구분선
if phase_idx:
    for p_i, p_c, p_l in zip(phase_idx,
                               ['orangered', 'royalblue'],
                               ['도약', '착지']):
        ax_ang.axvline(x=p_i + 1, color=p_c, lw=1.2, ls=':', alpha=0.8)
        ax_traj.axvline(x=0, color=p_c, lw=0)   # dummy (색만 일치)

# 발끝 궤적 선 (오른쪽 그래프) - X, Y, Z
traj_line_x, = ax_traj.plot([], [], '-', color='#ff88aa', lw=2.0, label='X')
traj_line_y, = ax_traj.plot([], [], '-', color='#88ff88', lw=2.0, label='Y')
traj_line_z, = ax_traj.plot([], [], '-', color='#8899ff', lw=2.0, label='Z')
traj_dot,    = ax_traj.plot([], [], 'o', color='white', ms=6, zorder=5)
ax_traj.legend(loc='upper right', fontsize=8,
               facecolor='#1a1a2e', labelcolor='white', edgecolor='gray')

# 좌표축 quiver (시작점 + 끝점 각 3개)
_frame_quivers = [
    ax3d.quiver(0, 0, 0, 1, 0, 0, length=AXIS_LEN, color=c, linewidth=1.5)
    for _ in range(2) for c in _AXIS_COLORS
]

def _draw_frame(T, quivers_xyz):
    orig = T[:3, 3]
    for j, q in enumerate(quivers_xyz):
        axis = T[:3, j]
        q.remove()
        quivers_xyz[j] = ax3d.quiver(
            orig[0], orig[1], orig[2],
            axis[0], axis[1], axis[2],
            length=AXIS_LEN, color=_AXIS_COLORS[j],
            linewidth=1.5, arrow_length_ratio=0.3
        )
    return quivers_xyz

# ══════════════════════════════════════════════════════════════
# 5. 애니메이션
# ══════════════════════════════════════════════════════════════
trace_x, trace_y, trace_z = [], [], []
traj_x2, traj_y2, traj_z2 = [], [], []

def init_anim():
    for ln in link_lines:
        ln.set_data([], [])
        ln.set_3d_properties([])
    trace_line.set_data([], [])
    trace_line.set_3d_properties([])
    for ln in ang_lines:
        ln.set_data([], [])
    traj_line_x.set_data([], [])
    traj_line_y.set_data([], [])
    traj_line_z.set_data([], [])
    traj_dot.set_data([], [])
    info_text.set_text('')
    phase_text.set_text('')
    return []

def animate(i):
    global _frame_quivers
    thetas = theta_hist[i]

    # FK
    T04 = np.eye(4)
    for k, (alpha, a, d) in enumerate(DH_PARAMS[:4]):
        T04 = T04 @ get_dh_matrix(alpha, a, d, thetas[k])
    pts = forward_kinematics(thetas)

    # 링크 업데이트
    for k in range(5):
        A, B = pts[k], pts[k+1]
        link_lines[k].set_data([A[0], B[0]], [A[1], B[1]])
        link_lines[k].set_3d_properties([A[2], B[2]])

    # O4 궤적 (3D)
    Pe = pts[4]   # O4 (ankle)
    trace_x.append(Pe[0]); trace_y.append(Pe[1]); trace_z.append(Pe[2])
    trace_line.set_data(trace_x, trace_y)
    trace_line.set_3d_properties(trace_z)

    # 좌표축 갱신
    _frame_quivers[:3] = _draw_frame(np.eye(4), _frame_quivers[:3])
    _frame_quivers[3:] = _draw_frame(T04, _frame_quivers[3:])

    # 관절각 그래프
    deg = [math.degrees(thetas[k]) for k in range(4)]
    for k in range(4):
        ang_data[k].append(deg[k])
        ang_lines[k].set_data(range(len(ang_data[k])), ang_data[k])

    # 발끝 X·Y·Z 위치 vs 프레임
    traj_x2.append(Pe[0]); traj_y2.append(Pe[1]); traj_z2.append(Pe[2])
    frames = range(len(traj_x2))
    traj_line_x.set_data(frames, traj_x2)
    traj_line_y.set_data(frames, traj_y2)
    traj_line_z.set_data(frames, traj_z2)
    traj_dot.set_data([i], [Pe[0]])

    # 위상 텍스트
    if phase_idx:
        if i <= phase_idx[0]:     p_str = "Phase 1: 준비 (Crouch)"
        elif i <= phase_idx[1]:   p_str = "Phase 2: 도약 (Push-off)"
        else:                     p_str = "Phase 3: 착지 복귀"
    else:
        n_half = n_frames // 2
        p_str = "Stance" if i < n_half else "Swing"
    phase_text.set_text(p_str)

    # 정보 텍스트
    msg = (f"θ1:{deg[0]:+6.1f}°  θ2:{deg[1]:+6.1f}°\n"
           f"θ3:{deg[2]:+6.1f}°  θ4:{deg[3]:+6.1f}°\n\n"
           f"O4 (mm)\n"
           f"X:{Pe[0]*1000:+7.1f}\n"
           f"Y:{Pe[1]*1000:+7.1f}\n"
           f"Z:{Pe[2]*1000:+7.1f}")
    info_text.set_text(msg)

    return []

ani = FuncAnimation(
    fig, animate, frames=n_frames,
    init_func=init_anim,
    interval=DT * 1000,   # ms
    blit=False, repeat=True
)

plt.show()
