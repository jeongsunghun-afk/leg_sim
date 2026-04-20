"""
WBC_sim_v1.py
4DOF 다리 시뮬레이터 - 자코비안 IK + 동역학 토크 + 수직 점프 궤적

  관심점: L3 끝점 (joints[3])
  θ4: 고정

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[궤적 모드]
  TRAJ_MODE = 'gait' : 보행 스텝 (Stance + Swing)
  TRAJ_MODE = 'jump' : 수직 점프 (준비 → 도약 → 착지)

[수직 점프 궤적 구간]
  Phase 1 (준비/Crouch) : y0  → y0 - h_crouch  n_frames//3
  Phase 2 (도약/Push)   : y0 - h_crouch → y0 + h_jump  n_frames//6
  Phase 3 (착지/Land)   : y0 + h_jump → y0  나머지
  x: 고정 (수직 이동만)
  각 구간: 5차 스플라인, 경계 v=a=0

[동역학 방정식]
  τ = M(q)q̈ + C(q,q̇)q̇ + G(q)

[동역학 파라미터]
  masses   [kg]     링크 질량
  lc_ratio [-]      COM 위치 비율 (0=근위관절, 1=원위)
  I_cm     [kg·m²]  COM 기준 관성모멘트 (균일봉: m·L²/12)
  G_ACC    [m/s²]   중력가속도
  DT       [s]      제어 주기 (θ̈ 수치미분에 사용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.font_manager as fm

# ── 한글 폰트
_font_path = fm.findfont(fm.FontProperties(family='NanumGothic'))
if 'NanumGothic' not in _font_path:
    fm.fontManager.addfont('/home/jsh/.local/share/fonts/NanumGothic-Regular.ttf')
mpl.rcParams['font.family']        = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False
for key in mpl.rcParams:
    if key.startswith("keymap."):
        mpl.rcParams[key] = []

# ══════════════════════════════════════════
# 파라미터
# ══════════════════════════════════════════
L    = np.array([120.0, 120.0, 148.0, 45.0])   # 링크 길이 [mm]
L_m  = L / 1000.0                               # 링크 길이 [m]
THETA_INIT = np.deg2rad([300.0, 270.0, 90.0, 60.0])
DT        = 0.005   # 제어 주기 [s]
TRAJ_MODE = 'jump'  # ← 'gait' | 'jump'

# ── 동역학 파라미터
masses   = np.array([2.0,  1.5,  1.0, 0.5])    # 링크 질량 [kg]
lc_ratio = np.array([0.5,  0.5,  0.5, 0.5])    # COM 위치 비율
lc_m     = L_m * lc_ratio                       # COM 오프셋 [m]
I_cm     = masses * L_m**2 / 12                 # 관성모멘트 [kg·m²]
G_ACC    = 9.81                                  # 중력가속도 [m/s²]

# ══════════════════════════════════════════
# 순기구학
# ══════════════════════════════════════════
def fk(theta, L=L):
    alpha  = np.cumsum(theta)
    joints = np.zeros((5, 2))
    for i in range(4):
        joints[i+1] = joints[i] + L[i] * np.array([np.cos(alpha[i]), np.sin(alpha[i])])
    return joints

# ══════════════════════════════════════════
# 자코비안 (L3 끝점, θ1~θ3)
# ══════════════════════════════════════════
def jacobian(theta, L=L):
    alpha = np.cumsum(theta[:3])
    J_abs = np.array([
        [-L[0]*np.sin(alpha[0]), -L[1]*np.sin(alpha[1]), -L[2]*np.sin(alpha[2])],
        [ L[0]*np.cos(alpha[0]),  L[1]*np.cos(alpha[1]),  L[2]*np.cos(alpha[2])]
    ])
    T = np.array([[1,0,0],[1,1,0],[1,1,1]], dtype=float)
    return J_abs, J_abs @ T

# ══════════════════════════════════════════
# 보행 궤적
# ══════════════════════════════════════════
def make_gait_trajectory(start, step_x=80.0, lift=45.0, n=120):
    t1 = np.linspace(0, 1, n // 2)
    sx = start[0] - step_x/2 * t1
    sy = np.full(n//2, start[1])
    t2 = np.linspace(0, np.pi, n//2)
    cx = start[0] - step_x/4
    wx = cx + step_x/2 * np.cos(np.pi - t2)
    wy = start[1] + lift * np.sin(t2)
    return np.vstack([np.column_stack([sx, sy]), np.column_stack([wx, wy])])

# ══════════════════════════════════════════
# 자코비안 IK (DLS)
# ══════════════════════════════════════════
def ik_jacobian(theta_init, trajectory, L=L,
                max_iter=80, tol=0.3, lam=4.0, step=0.25):
    theta   = theta_init.copy()
    history = [theta.copy()]
    for target in trajectory:
        for _ in range(max_iter):
            ee  = fk(theta, L)[3]
            err = target - ee
            if np.linalg.norm(err) < tol:
                break
            _, J  = jacobian(theta, L)
            J_dls = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(2))
            theta[:3] += J_dls @ err * step
        history.append(theta.copy())
    return np.array(history)

# ══════════════════════════════════════════
# 5차 다항식 스플라인 (참조 코드)
# ══════════════════════════════════════════
def solve_quintic_spline(t0, tf, p0, v0, a0, pf, vf, af):
    """
    S(t) = Σ c_k · t^k  (k=0..5)
    경계 조건 [위치, 속도, 가속도] × [시작, 끝]
    """
    T_mat = np.array([
        [1, t0,   t0**2,    t0**3,    t0**4,    t0**5],
        [0,  1, 2*t0,    3*t0**2,  4*t0**3,  5*t0**4],
        [0,  0,   2,    6*t0,    12*t0**2, 20*t0**3 ],
        [1, tf,   tf**2,    tf**3,    tf**4,    tf**5],
        [0,  1, 2*tf,    3*tf**2,  4*tf**3,  5*tf**4],
        [0,  0,   2,    6*tf,    12*tf**2, 20*tf**3 ]
    ])
    return np.linalg.solve(T_mat, np.array([p0, v0, a0, pf, vf, af]))

def eval_quintic(c, t):
    """위치, 속도, 가속도 반환"""
    pos = c[0]+c[1]*t    +c[2]*t**2    +c[3]*t**3     +c[4]*t**4     +c[5]*t**5
    vel = c[1]+2*c[2]*t  +3*c[3]*t**2  +4*c[4]*t**3   +5*c[5]*t**4
    acc = 2*c[2]+6*c[3]*t+12*c[4]*t**2 +20*c[5]*t**3
    return pos, vel, acc

# ══════════════════════════════════════════
# 수직 점프 궤적 (5차 스플라인 3구간)
# ══════════════════════════════════════════
def make_jump_trajectory(start, h_crouch=40.0, h_jump=80.0, n=120):
    """
    수직 점프 궤적 (x 고정, y만 변화)

    [좌표계 기준]
      힙(origin) 고정, 발끝(L3)이 힙 아래에 위치 (y < 0)
      다리 신장 = 발끝이 힙에서 멀어짐 = y 감소 (↓)
      다리 굴곡 = 발끝이 힙에 가까워짐  = y 증가 (↑)

    Phase 1 (준비/Crouch) : y0 → y0 + h_crouch  [n//3 frames]
      - 다리 굴곡 (발끝 ↑) → 관절 압축, 에너지 축적

    Phase 2 (도약/Push-off): y0 + h_crouch → y0 - h_jump  [n//6 frames]
      - 다리 신장 (발끝 ↓) → 지면을 미는 방향, 큰 신장 토크 발생
      - 짧은 구간 = 높은 각가속도 → 큰 관성 토크

    Phase 3 (착지/Landing): y0 - h_jump → y0  [나머지 frames]
      - 다리 굴곡 복귀 (발끝 ↑) → 완충

    각 구간 경계: 속도=0, 가속도=0  (5차 스플라인)

    start    : L3 끝점 초기 위치 [mm]
    h_crouch : 준비 굴곡량 [mm]
    h_jump   : 도약 신장량 [mm] (초기 위치 기준 하방)
    n        : 전체 프레임 수
    """
    n1 = n // 3         # 준비 구간
    n2 = n // 6         # 도약 구간 (짧고 빠름)
    n3 = n - n1 - n2    # 착지 구간

    x0, y0   = start
    y_crouch = y0 + h_crouch   # 굴곡 시 발끝 ↑ (힙에 가까워짐)
    y_push   = y0 - h_jump     # 신장 시 발끝 ↓ (힙에서 멀어짐 = 지면 밀기)

    T1 = n1 * DT
    T2 = n2 * DT
    T3 = n3 * DT

    c1 = solve_quintic_spline(0, T1, y0,       0, 0, y_crouch, 0, 0)  # 준비: 굴곡
    c2 = solve_quintic_spline(0, T2, y_crouch, 0, 0, y_push,   0, 0)  # 도약: 신장 (지면 밀기)
    c3 = solve_quintic_spline(0, T3, y_push,   0, 0, y0,       0, 0)  # 착지: 굴곡 복귀

    pts = []
    for j in range(n1):
        y, _, _ = eval_quintic(c1, j * DT);  pts.append([x0, y])
    for j in range(n2):
        y, _, _ = eval_quintic(c2, j * DT);  pts.append([x0, y])
    for j in range(n3):
        y, _, _ = eval_quintic(c3, j * DT);  pts.append([x0, y])

    phase_idx = (n1, n1 + n2)
    return np.array(pts), phase_idx

# ══════════════════════════════════════════
# θ̇, θ̈ 산출 (구간별 5차 스플라인)
# ══════════════════════════════════════════
def smooth_and_diff(theta_hist, dt):
    """
    IK 궤적 → 5차 스플라인 보간 → θ, θ̇, θ̈ 해석적 산출
    θ4 고정.
    """
    N     = len(theta_hist)
    tf    = (N - 1) * dt
    t_arr = np.linspace(0, tf, N)

    theta_s = theta_hist.copy().astype(float)
    dtheta  = np.zeros_like(theta_hist, dtype=float)
    ddtheta = np.zeros_like(theta_hist, dtype=float)

    for k in range(3):
        c = solve_quintic_spline(0, tf,
                                 theta_hist[0, k], 0, 0,
                                 theta_hist[-1, k], 0, 0)
        for i, t in enumerate(t_arr):
            theta_s[i, k], dtheta[i, k], ddtheta[i, k] = eval_quintic(c, t)

    theta_s[:,  3] = theta_hist[0, 3]
    dtheta[:,   3] = 0.0
    ddtheta[:,  3] = 0.0
    return theta_s, dtheta, ddtheta

# ══════════════════════════════════════════
# 동역학 모델링  τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
# ══════════════════════════════════════════
def _com_jac(theta_full, link_idx):
    alpha = np.cumsum(theta_full)
    Jc    = np.zeros((2, 3))
    for j in range(min(link_idx + 1, 3)):
        dx = dy = 0.0
        for k in range(j, link_idx):
            dx -= L_m[k] * np.sin(alpha[k])
            dy += L_m[k] * np.cos(alpha[k])
        dx -= lc_m[link_idx] * np.sin(alpha[link_idx])
        dy += lc_m[link_idx] * np.cos(alpha[link_idx])
        Jc[0, j] = dx
        Jc[1, j] = dy
    return Jc

def mass_matrix_fn(theta_full):
    """3×3 질량 행렬 M(q) [kg·m²]"""
    M = np.zeros((3, 3))
    for i in range(4):
        Jc = _com_jac(theta_full, i)
        M += masses[i] * Jc.T @ Jc
        Jw = np.zeros(3)
        Jw[:min(i + 1, 3)] = 1.0
        M += I_cm[i] * np.outer(Jw, Jw)
    return M

def gravity_vec(theta_full):
    """중력 벡터 G(q) [N·m]"""
    alpha = np.cumsum(theta_full)
    G = np.zeros(3)
    for i in range(4):
        for j in range(min(i + 1, 3)):
            dy  = sum(L_m[k] * np.cos(alpha[k]) for k in range(j, i))
            dy += lc_m[i] * np.cos(alpha[i])
            G[j] += masses[i] * G_ACC * dy
    return G

def coriolis_vec(theta_full, dtheta3, eps=1e-7):
    """코리올리·원심력 C(q,q̇)q̇ [N·m] (Christoffel, 수치미분)"""
    n  = 3
    dM = np.zeros((n, n, n))
    for k in range(n):
        tp, tm = theta_full.copy(), theta_full.copy()
        tp[k] += eps;  tm[k] -= eps
        dM[k] = (mass_matrix_fn(tp) - mass_matrix_fn(tm)) / (2 * eps)
    C = np.zeros(n)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                Gamma = 0.5 * (dM[k,i,j] + dM[j,i,k] - dM[i,j,k])
                C[i] += Gamma * dtheta3[j] * dtheta3[k]
    return C

def compute_torques(theta_s, dtheta, ddtheta):
    """τ = M(q)q̈ + C(q,q̇)q̇ + G(q) [N·m]"""
    N       = len(theta_s)
    torques = np.zeros((N, 3))
    for i in range(N):
        M          = mass_matrix_fn(theta_s[i])
        G          = gravity_vec(theta_s[i])
        C          = coriolis_vec(theta_s[i], dtheta[i, :3])
        torques[i] = M @ ddtheta[i, :3] + C + G
    return torques

# ══════════════════════════════════════════
# 사전 계산
# ══════════════════════════════════════════
print("궤적 IK 계산 중...")
init_joints = fk(THETA_INIT)
toe_start   = init_joints[3].copy()

phase_idx = None
if TRAJ_MODE == 'jump':
    trajectory, phase_idx = make_jump_trajectory(
        toe_start, h_crouch=40.0, h_jump=80.0, n=120)
    mode_label = '수직 점프 (준비→도약→착지)'
else:
    trajectory = make_gait_trajectory(toe_start, step_x=80, lift=45, n=120)
    mode_label = '보행 스텝 (Stance+Swing)'

theta_hist = ik_jacobian(THETA_INIT, trajectory)
joint_hist = np.array([fk(th) for th in theta_hist])

print("5차 스플라인 보간 + 토크 계산 중...")
theta_s, dtheta, ddtheta = smooth_and_diff(theta_hist, DT)
torque_hist = compute_torques(theta_s, dtheta, ddtheta)
print(f"완료: {len(theta_hist)} 프레임  |  모드: {mode_label}")
print(f"  최대 토크: τ1={np.abs(torque_hist[:,0]).max():.2f}  "
      f"τ2={np.abs(torque_hist[:,1]).max():.2f}  "
      f"τ3={np.abs(torque_hist[:,2]).max():.2f}  [N·m]")

# ══════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════
fig = plt.figure(figsize=(15, 9))
gs  = gridspec.GridSpec(3, 3, figure=fig, wspace=0.42, hspace=0.52)

ax_leg    = fig.add_subplot(gs[:, :2])
ax_angle  = fig.add_subplot(gs[0, 2])
ax_ee     = fig.add_subplot(gs[1, 2])
ax_torque = fig.add_subplot(gs[2, 2])

# ── 다리 축
margin = 60
x_all  = np.concatenate([joint_hist[:, :, 0].ravel(), trajectory[:, 0]])
y_all  = np.concatenate([joint_hist[:, :, 1].ravel(), trajectory[:, 1]])
ax_leg.set_xlim(x_all.min()-margin, x_all.max()+margin)
ax_leg.set_ylim(y_all.min()-margin, y_all.max()+margin)
ax_leg.set_aspect('equal');  ax_leg.grid(True, alpha=0.25)
ground_y = toe_start[1]
ax_leg.axhline(y=ground_y, color='saddlebrown', lw=1.5, ls='--', alpha=0.6)
ax_leg.fill_between([x_all.min()-margin, x_all.max()+margin],
                    ground_y-20, ground_y, color='saddlebrown', alpha=0.12)
ax_leg.plot(trajectory[:, 0], trajectory[:, 1], 'g--', lw=1.2, alpha=0.5, label='목표 궤적')
ax_leg.plot(*toe_start, 'ko', ms=9, zorder=5, label='초기 발끝')

# 점프 구간 경계 표시
if phase_idx is not None:
    for p_idx, p_label, p_col in zip(
            phase_idx, ['도약 시작', '착지 시작'], ['orangered', 'royalblue']):
        ax_leg.plot(*trajectory[p_idx], 'D', color=p_col, ms=10, zorder=7, label=p_label)

leg_line,   = ax_leg.plot([], [], 'o-', color='steelblue', lw=3.5, ms=8, zorder=4)
trace_line, = ax_leg.plot([], [], '-',  color='tomato',    lw=2,   alpha=0.8, label='실제 궤적')
target_dot, = ax_leg.plot([], [], '*',  color='lime',      ms=13,  zorder=6,  label='현재 목표')
ax_leg.set_title(f'4DOF 다리 - {mode_label}  +  동역학 토크', fontsize=11)
ax_leg.set_xlabel('x [mm]');  ax_leg.set_ylabel('y [mm]')
ax_leg.legend(loc='upper right', fontsize=7)

info_text = ax_leg.text(0.02, 0.04, '', transform=ax_leg.transAxes, fontsize=7.5,
                        va='bottom', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))

# ── 관절각
n_frames = len(theta_hist)
ax_angle.set_xlim(0, n_frames);  ax_angle.set_ylim(-30, 370)
ax_angle.grid(True, alpha=0.3)
ax_angle.set_title('관절각 변화', fontsize=10)
ax_angle.set_xlabel('프레임');    ax_angle.set_ylabel('[deg]')
line_t = [ax_angle.plot([], [], lw=1.8, label=f'θ{i+1}')[0] for i in range(4)]
# 점프 구간 경계
if phase_idx:
    for p, c in zip(phase_idx, ['orangered', 'royalblue']):
        ax_angle.axvline(x=p, color=c, lw=1.2, ls=':', alpha=0.8)
ax_angle.legend(fontsize=8)

# ── L3 끝점 좌표
y_ee_all = joint_hist[:, 3, :]
ax_ee.set_xlim(0, n_frames)
ax_ee.set_ylim(y_ee_all.min()-10, y_ee_all.max()+10)
ax_ee.grid(True, alpha=0.3)
ax_ee.set_title('L3 끝점 좌표 (관심점)', fontsize=10)
ax_ee.set_xlabel('프레임');  ax_ee.set_ylabel('[mm]')
line_ex, = ax_ee.plot([], [], lw=1.8, label='x', color='steelblue')
line_ey, = ax_ee.plot([], [], lw=1.8, label='y', color='tomato')
if phase_idx:
    for p, c in zip(phase_idx, ['orangered', 'royalblue']):
        ax_ee.axvline(x=p, color=c, lw=1.2, ls=':', alpha=0.8)
ax_ee.legend(fontsize=8)

# ── 토크
t_max = np.abs(torque_hist).max() * 1.2 + 0.1
ax_torque.set_xlim(0, n_frames);  ax_torque.set_ylim(-t_max, t_max)
ax_torque.grid(True, alpha=0.3)
ax_torque.axhline(0, color='k', lw=0.8, ls='--')
ax_torque.set_title('관절 토크  τ = Mq̈ + Cq̇ + G', fontsize=10)
ax_torque.set_xlabel('프레임');  ax_torque.set_ylabel('[N·m]')
colors_tau = ['steelblue', 'tomato', 'seagreen']
line_tau   = [ax_torque.plot([], [], lw=1.8, color=colors_tau[i],
                              label=f'τ{i+1}')[0] for i in range(3)]
if phase_idx:
    for p, c in zip(phase_idx, ['orangered', 'royalblue']):
        ax_torque.axvline(x=p, color=c, lw=1.2, ls=':', alpha=0.8)
ax_torque.legend(fontsize=8)

# ── 애니메이션 상태
trace_x, trace_y    = [], []
ang_data             = [[] for _ in range(4)]
ee_x_data, ee_y_data = [], []
tau_data             = [[] for _ in range(3)]


def init_anim():
    leg_line.set_data([], []);  trace_line.set_data([], []);  target_dot.set_data([], [])
    for ln in line_t + line_tau + [line_ex, line_ey]:
        ln.set_data([], [])
    info_text.set_text('')
    return (leg_line, trace_line, target_dot,
            *line_t, line_ex, line_ey, *line_tau, info_text)


def animate(i):
    if i >= n_frames:
        return (leg_line, trace_line, target_dot,
                *line_t, line_ex, line_ey, *line_tau, info_text)

    joints = joint_hist[i]
    leg_line.set_data(joints[:, 0], joints[:, 1])

    ee = joints[3]
    trace_x.append(ee[0]);  trace_y.append(ee[1])
    trace_line.set_data(trace_x, trace_y)
    target_dot.set_data([trajectory[min(i, len(trajectory)-1), 0]],
                         [trajectory[min(i, len(trajectory)-1), 1]])

    th = theta_hist[i]
    for k in range(4):
        ang_data[k].append(np.rad2deg(th[k]))
        line_t[k].set_data(range(len(ang_data[k])), ang_data[k])

    ee_x_data.append(ee[0]);  ee_y_data.append(ee[1])
    line_ex.set_data(range(len(ee_x_data)), ee_x_data)
    line_ey.set_data(range(len(ee_y_data)), ee_y_data)

    tau = torque_hist[i]
    for k in range(3):
        tau_data[k].append(tau[k])
        line_tau[k].set_data(range(len(tau_data[k])), tau_data[k])

    # 현재 구간 표시
    if phase_idx:
        if i < phase_idx[0]:
            phase_str = "Phase 1: 준비 (Crouch)"
        elif i < phase_idx[1]:
            phase_str = "Phase 2: 도약 (Push-off)"
        else:
            phase_str = "Phase 3: 착지 (Landing)"
    else:
        phase_str = ""

    J_abs, _ = jacobian(th, L)
    info_text.set_text(
        f"{phase_str}\n\n"
        "J_abs (2×3):\n"
        f"[{J_abs[0,0]:+6.1f} {J_abs[0,1]:+6.1f} {J_abs[0,2]:+6.1f}]\n"
        f"[{J_abs[1,0]:+6.1f} {J_abs[1,1]:+6.1f} {J_abs[1,2]:+6.1f}]\n\n"
        "토크 [N·m]:\n"
        f"  τ1 = {tau[0]:+.3f}\n"
        f"  τ2 = {tau[1]:+.3f}\n"
        f"  τ3 = {tau[2]:+.3f}\n\n"
        f"θ4 = fixed ({np.rad2deg(th[3]):.1f}°)"
    )

    return (leg_line, trace_line, target_dot,
            *line_t, line_ex, line_ey, *line_tau, info_text)


ani = FuncAnimation(fig, animate, init_func=init_anim,
                    frames=n_frames, interval=40, blit=True)

plt.suptitle(f'4DOF Leg Simulation  |  {mode_label}  |  τ = M(q)q̈ + C(q,q̇)q̇ + G(q)',
             fontsize=11)
plt.show()
