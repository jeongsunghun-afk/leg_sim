#역기구학 감쇠 최소자승법
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 키보드 기본 단축키 충돌 방지
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []

mpl.rcParams['font.family'] = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════════
# 1. DH 파라미터 & FK (순수 행렬 연산)
# ══════════════════════════════════════════════════════════════════

# [alpha (rad), a (m), d (m)]  ← sim.py 기준 (a 방향 링크)
DH_PARAMS = [
    (-math.pi/2, 0.0,    0.0    ),   # Joint 1 : Hip Abduction
    (0.0,        0.21,   0.0075 ),   # Joint 2 : Hip Pitch
    (0.0,        0.21,   0.0    ),   # Joint 3 : Knee
    (0.0,        0.148,  0.0    ),   # Joint 4 : Ankle
#    (0.0,        0.045,  0.0    ),   # Joint 5 : Toe
    (0.0,        0.0,  0.0    ),   # Joint 5 : Toe
]

# 관절 각도 (sim.py 동일)
Q_INIT = [math.radians(a) for a in [0, -90,   0,  0,  0]]   # 초기 위치
Q_HOME = [math.radians(a) for a in [0, -150, -90, 90, 60]]   # 홈 위치 오프셋

def get_dh_matrix(alpha, a, d, theta):
    """Standard DH 변환행렬 생성"""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1]
    ], dtype=float)

def forward_kinematics(thetas):
    """5개 관절각을 받아 모든 관절의 3D 좌표 반환"""
    T = np.eye(4)
    pts = [np.zeros(3)]  # 원점(O0)
    for i, (alpha, a, d) in enumerate(DH_PARAMS):
        T = T @ get_dh_matrix(alpha, a, d, thetas[i])
        pts.append(T[:3, 3].copy())
    return pts

# ══════════════════════════════════════════════════════════════════
# 2. 역기구학 (야코비안 의사역행렬)
# ══════════════════════════════════════════════════════════════════

def compute_jacobian(thetas):
    """위치 야코비안 (3×4) 계산: theta5 고정, joint 1~4만 사용"""
    T = np.eye(4)
    transforms = [T.copy()]
    for i, (alpha, a, d) in enumerate(DH_PARAMS):
        T = T @ get_dh_matrix(alpha, a, d, thetas[i])
        transforms.append(T.copy())
    p_e = transforms[-1][:3, 3]  # 끝점은 여전히 joint5 포함 위치
    J = np.zeros((3, 4))
    for j in range(4):              # joint 1~4만 계산
        z_j = transforms[j][:3, 2]
        p_j = transforms[j][:3, 3]
        J[:, j] = np.cross(z_j, p_e - p_j)
    return J

def ik_step(thetas, delta_p, damping=0.0001):
    """감쇠 최소자승법 IK: theta5 고정, joint 1~4만 업데이트"""
    J = compute_jacobian(thetas)
    lam2 = damping ** 2
    delta_theta = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(3), delta_p)
    result = thetas[:]
    for i in range(4):
        result[i] += delta_theta[i]
    deg = [math.degrees(result[i]) for i in range(5)]
    print(f"th1:{deg[0]:.2f}  th2:{deg[1]:.2f}  th3:{deg[2]:.2f}  th4:{deg[3]:.2f}  th5:{deg[4]:.2f}")
    return result
    

# ══════════════════════════════════════════════════════════════════
# 3. 초기값 및 상태 변수
# ══════════════════════════════════════════════════════════════════

DELTA_ANGLE = math.radians(5.0)  # 한 번 누를 때 5도씩 회전
DELTA_POS   = 0.005               # IK: 한 번 누를 때 5mm
angles = Q_HOME[:]               # 초기 표시 자세

# ══════════════════════════════════════════════════════════════════
# 4. 시각화 설정 (Matplotlib 3D)
# ══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor('#1a1a2e')

ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_facecolor('#16213e')

# 축 범위 설정
reach = 0.5
ax3d.set_xlim(-reach, reach)
ax3d.set_ylim(-reach, reach)
ax3d.set_zlim(-reach, reach)
ax3d.set_xlabel('X (m)', color='white')
ax3d.set_ylabel('Y (m)', color='white')
ax3d.set_zlabel('Z (m)', color='white')
ax3d.tick_params(colors='gray')

_COLORS = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00', '#cc88ff']
_AXIS_COLORS = ['#ff4444', '#44ff44', '#4444ff']  # x=빨강, y=초록, z=파랑
AXIS_LEN = 0.06  # 좌표축 화살표 길이 (m)

link_lines = [ax3d.plot([], [], [], '-o', color=_COLORS[i], lw=3, markersize=8)[0] for i in range(5)]
info_text = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes, color='white', fontfamily='monospace')

# 시작점·끝점 좌표축 quiver (x,y,z 각 3개씩 × 2 프레임 = 6개)
_frame_quivers = [
    ax3d.quiver(0, 0, 0, 1, 0, 0, length=AXIS_LEN, color=c, linewidth=1.5)
    for _ in range(2) for c in _AXIS_COLORS
]

def _draw_frame(T, quivers_xyz):
    """변환행렬 T의 x,y,z 축을 quiver 3개에 반영"""
    orig = T[:3, 3]
    for j, q in enumerate(quivers_xyz):
        axis = T[:3, j]  # j=0:x, 1:y, 2:z
        q.remove()
        quivers_xyz[j] = ax3d.quiver(
            orig[0], orig[1], orig[2],
            axis[0], axis[1], axis[2],
            length=AXIS_LEN, color=_AXIS_COLORS[j],
            linewidth=1.5, arrow_length_ratio=0.3
        )
    return quivers_xyz

def update_plot():
    global _frame_quivers

    T = np.eye(4)
    for i, (alpha, a, d) in enumerate(DH_PARAMS):
        T = T @ get_dh_matrix(alpha, a, d, angles[i])
    print("=== 동차변환행렬 (0A4) ===")
    print(np.round(T, 4))

    pts = forward_kinematics(angles)
    for i in range(5):
        A, B = pts[i], pts[i+1]
        link_lines[i].set_data([A[0], B[0]], [A[1], B[1]])
        link_lines[i].set_3d_properties([A[2], B[2]])

    # 시작점(O0) 좌표축: 항상 단위행렬
    _frame_quivers[:3] = _draw_frame(np.eye(4), _frame_quivers[:3])
    # 끝점(O5) 좌표축: 현재 T
    _frame_quivers[3:] = _draw_frame(T, _frame_quivers[3:])

    Pe = pts[-1]
    deg = [math.degrees(a) for a in angles]

    msg = (f"JOINT ANGLES (deg)\n"
           f"θ1:{deg[0]:.1f}  θ2:{deg[1]:.1f}  θ3:{deg[2]:.1f}\n"
           f"θ4:{deg[3]:.1f}  θ5:{deg[4]:.1f}\n\n"
           f"TOE POS (m)\n"
           f"X:{Pe[0]:.3f} Y:{Pe[1]:.3f} Z:{Pe[2]:.3f}")
    info_text.set_text(msg)
    fig.canvas.draw_idle()

# ══════════════════════════════════════════════════════════════════
# 5. 키보드 이벤트 (FK + IK)
# ══════════════════════════════════════════════════════════════════

def on_key(event):
    global angles
    key = event.key

    # ── FK: 관절각 직접 조작 ──────────────────────────────────────
    if   key == '1': angles[0] += DELTA_ANGLE
    elif key == 'q': angles[0] -= DELTA_ANGLE
    elif key == '2': angles[1] += DELTA_ANGLE
    elif key == 'w': angles[1] -= DELTA_ANGLE
    elif key == '3': angles[2] += DELTA_ANGLE
    elif key == 'e': angles[2] -= DELTA_ANGLE
    elif key == '4': angles[3] += DELTA_ANGLE
    elif key == 'r': angles[3] -= DELTA_ANGLE
    elif key == '5': angles[4] += DELTA_ANGLE
    elif key == 't': angles[4] -= DELTA_ANGLE

    # ── IK: 발끝 위치 직접 조작 (a/s/d: +x/+y/+z, A/S/D: -x/-y/-z) ──
    elif key == 'a': angles[:] = ik_step(angles, [ DELTA_POS, 0, 0])
    elif key == 'z': angles[:] = ik_step(angles, [-DELTA_POS, 0, 0])
    elif key == 's': angles[:] = ik_step(angles, [0,  DELTA_POS, 0])
    elif key == 'x': angles[:] = ik_step(angles, [0, -DELTA_POS, 0])
    elif key == 'd': angles[:] = ik_step(angles, [0, 0,  DELTA_POS])
    elif key == 'c': angles[:] = ik_step(angles, [0, 0, -DELTA_POS])

    # ── 리셋 ─────────────────────────────────────────────────────
    elif key == 'h': angles[:] = Q_INIT[:]
    elif key == 'p': angles[:] = Q_HOME[:]

    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot()
plt.show()