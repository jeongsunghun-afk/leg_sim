#역기구학 해석적 방법
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 키보드 기본 단축키 충돌 방지
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False

# ══════════════════════════════════════════════════════════════════
# 1. DH 파라미터 & FK
# ══════════════════════════════════════════════════════════════════

DH_PARAMS = [
    (-math.pi/2, 0.0,   0.0,    ),   # Joint 1: Hip Abduction
    (0.0,        0.21,  0.0075, ),   # Joint 2: Hip Pitch
    (0.0,        0.235, 0.0,    ),   # Joint 3: Knee
    (0.0,        0.1,   0.0,    ),   # Joint 4: Lower leg
    (0.0,      0.045, 0.0,    ),   # Joint 5: Foot (추후 5관절 확장 시 활성화)
]
# IK에서 사용하는 링크 파라미터 (DH_PARAMS와 동기화)
_A2 = 0.21
_A3 = 0.235
_A4 = 0.1
_D2 = 0.0075

Q_INIT = [math.radians(a) for a in [0, -90,   0,  0,  0]]
Q_HOME = [math.radians(a) for a in [0.0, 157.5,  22.5, 30.6583, 0]]


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

# ══════════════════════════════════════════════════════════════════
# 2. 해석적 역기구학 (ss.py analytical_ik)
#
#  목표: O5 위치 (Px, Py, Pz) → [θ1, θ2, θ3, θ4, θ5]
#  (DH_PARAMS에서 a5=0이므로 O5 = O4)
#
#  φ = θ2+θ3+θ4 : 발목 피치 방향각 (현재값 유지)
#  theta5_target = θ2+θ3+θ4+θ5 : 발끝 최종 누적각 (현재값 유지)
#
#  유도식:
#    θ1 = atan2(-Px, Py) - atan2(R, d2),  R = sqrt(Px²+Py²-d2²)
#    x_s = c1·Px + s1·Py,  Z = -Pz
#    x3  = x_s - a4·cos(φ),  z3 = Z - a4·sin(φ)
#    cos(θ3) = (x3²+z3²-a2²-a3²) / (2·a2·a3)
#    θ3 = ±acos(cos_θ3)   (elbow_up=True → 음수)
#    θ2 = atan2(z3, x3) - atan2(a3·sinθ3, a2+a3·cosθ3)
#    θ4 = φ - θ2 - θ3
#    θ5 = theta5_target - (θ2+θ3+θ4)
# ══════════════════════════════════════════════════════════════════

def analytical_ik(Px, Py, Pz, phi, theta5_target, elbow_up=True):
    D2 = Px**2 + Py**2 - _D2**2
    if D2 < 0:
        return None                      # 도달 불가
    R = math.sqrt(D2)

    # θ1
    theta1 = math.atan2(-Px, Py) - math.atan2(R, _D2)

    # 사지 평면 좌표
    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py
    Z   = -Pz

    # 2-링크 목표점 (a4 제거)
    x3 = x_s - _A4 * math.cos(phi)
    z3 = Z   - _A4 * math.sin(phi)

    # θ3
    cos_th3 = (x3**2 + z3**2 - _A2**2 - _A3**2) / (2.0 * _A2 * _A3)
    cos_th3 = max(-1.0, min(1.0, cos_th3))
    theta3  = -math.acos(cos_th3) if elbow_up else math.acos(cos_th3)

    # θ2
    theta2 = (math.atan2(z3, x3)
              - math.atan2(_A3 * math.sin(theta3),
                           _A2 + _A3 * math.cos(theta3)))

    # θ4, θ5
    theta4 = phi - theta2 - theta3
    theta5 = theta5_target - (theta2 + theta3 + theta4)

    # [-π, π] 정규화
    def wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    result = [wrap(theta1), wrap(theta2), wrap(theta3), wrap(theta4), wrap(theta5)]
    deg = [math.degrees(r) for r in result]
    print(f"th1:{deg[0]:.2f}  th2:{deg[1]:.2f}  th3:{deg[2]:.2f}  th4:{deg[3]:.2f}  th5:{deg[4]:.2f}")
    return result


def ik_move(thetas, delta_p):
    """현재 자세에서 발끝을 delta_p 만큼 이동하는 해석적 IK"""
    pts    = forward_kinematics(thetas)
    target = pts[-1] + np.array(delta_p)   # O5 목표 (a5=0 이므로 O4와 동일)

    phi           = thetas[1] + thetas[2] + thetas[3]   # 발목 방향 유지
    theta5_target = thetas[1] + thetas[2] + thetas[3] + thetas[4]

    result = analytical_ik(target[0], target[1], target[2],
                           phi, theta5_target, elbow_up=True)
    return result if result is not None else thetas[:]

# ══════════════════════════════════════════════════════════════════
# 3. 초기값 및 상태 변수
# ══════════════════════════════════════════════════════════════════

DELTA_ANGLE = math.radians(5.0)
DELTA_POS   = 0.005
angles = Q_HOME[:]

# ══════════════════════════════════════════════════════════════════
# 4. 시각화 설정 (Matplotlib 3D)
# ══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(12, 7))
fig.patch.set_facecolor('#1a1a2e')

ax3d = fig.add_subplot(111, projection='3d')
ax3d.set_facecolor('#16213e')

reach = 0.5
ax3d.set_xlim(-reach, reach)
ax3d.set_ylim(-reach, reach)
ax3d.set_zlim(-reach, reach)
ax3d.set_xlabel('X (m)', color='white')
ax3d.set_ylabel('Y (m)', color='white')
ax3d.set_zlabel('Z (m)', color='white')
ax3d.tick_params(colors='gray')
ax3d.set_title('sim4 — Analytical IK', color='white', fontsize=10)

_COLORS      = ['#00d4ff', '#00ff99', '#ff6b35', '#ffcc00', '#cc88ff']
_AXIS_COLORS = ['#ff4444', '#44ff44', '#4444ff']
AXIS_LEN     = 0.06

link_lines = [ax3d.plot([], [], [], '-o', color=_COLORS[i], lw=3, markersize=8)[0]
              for i in range(5)]
info_text  = ax3d.text2D(0.05, 0.95, "", transform=ax3d.transAxes,
                          color='white', fontfamily='monospace')

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

    _frame_quivers[:3] = _draw_frame(np.eye(4), _frame_quivers[:3])
    _frame_quivers[3:] = _draw_frame(T, _frame_quivers[3:])

    Pe  = pts[-1]
    deg = [math.degrees(a) for a in angles]
    msg = (f"JOINT ANGLES (deg)\n"
           f"θ1:{deg[0]:.1f}  θ2:{deg[1]:.1f}  θ3:{deg[2]:.1f}\n"
           f"θ4:{deg[3]:.1f}  θ5:{deg[4]:.1f}\n\n"
           f"TOE POS (m)\n"
           f"X:{Pe[0]:.3f} Y:{Pe[1]:.3f} Z:{Pe[2]:.3f}\n\n"
           f"IK: Analytical")
    info_text.set_text(msg)
    fig.canvas.draw_idle()

# ══════════════════════════════════════════════════════════════════
# 5. 키보드 이벤트
#
#  FK (관절각 직접):  1/q  2/w  3/e  4/r  5/t  (+/- 5deg)
#  IK (발끝 이동):    a/z(±X)  s/x(±Y)  d/c(±Z)  (5mm)
#  리셋:              h=초기자세  p=홈자세
# ══════════════════════════════════════════════════════════════════

def on_key(event):
    global angles
    key = event.key

    # ── FK ───────────────────────────────────────────────────────
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

    # ── 해석적 IK ────────────────────────────────────────────────
    elif key == 'a': angles[:] = ik_move(angles, [ DELTA_POS, 0, 0])
    elif key == 'z': angles[:] = ik_move(angles, [-DELTA_POS, 0, 0])
    elif key == 's': angles[:] = ik_move(angles, [0,  DELTA_POS, 0])
    elif key == 'x': angles[:] = ik_move(angles, [0, -DELTA_POS, 0])
    elif key == 'd': angles[:] = ik_move(angles, [0, 0,  DELTA_POS])
    elif key == 'c': angles[:] = ik_move(angles, [0, 0, -DELTA_POS])

    # ── 리셋 ─────────────────────────────────────────────────────
    elif key == 'h': angles[:] = Q_INIT[:]
    elif key == 'p': angles[:] = Q_HOME[:]

    update_plot()

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot()
plt.show()
