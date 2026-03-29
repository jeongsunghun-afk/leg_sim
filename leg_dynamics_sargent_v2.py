"""
leg_sim_v7.py
4DOF 다리 시뮬레이터 - 부유기반(Floating-Base) 동역학 모델

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[물리 모델: Floating-Base]
  L1 시작점(힙)이 수직으로 자유 이동.
  접촉 구간: 발이 지면에 고정, 힙이 위아래 이동.

[좌표계]
  힙 기준 상대 좌표로 FK/IK 수행 (기존과 동일).
  힙 글로벌 위치: y_hip_global = -(발끝 y_relative) [발=지면=0 기준]
  힙 가속도: a_hip_y = -d²(FK_y_rel)/dt²

[동역학 방정식]
  τ = M(q)q̈ + C(q,q̇)q̇ + G(q) - J_m^T · F_grf       [N·m]

[GRF 산출: 전체 시스템 뉴턴 법칙]
  F_grf - M_sys·g = M_sys · a_com_sys
  F_grf_y = M_sys·(g + a_hip_y) + Σ mᵢ·aᵢ_com_rel_y   [N]
  F_grf_x = M_sys·a_hip_x   + Σ mᵢ·aᵢ_com_rel_x       [N]

  여기서:
    M_sys  = M_BODY + M_LEG_TOTAL
    a_hip  = 발 고정 조건에서 힙 가속도 (수치미분)
    a_i_rel = 힙 기준 링크i COM 가속도 (자코비안 @ ddtheta)

[단위]
  동역학 자코비안: L_m[m] → J_m.T @ F_grf [N·m] ✓
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
L    = np.array([120.0, 120.0, 148.0, 45.0])  # [mm] FK/IK용
L_m  = L / 1000.0                              # [m]  동역학 계산용
THETA_INIT = np.deg2rad([300.0, 270.0, 90.0, 60.0])
DT        = 0.005
TRAJ_MODE = 'jump'

# ── 동역학 파라미터
masses   = np.array([2.0, 1.5, 1.0, 0.5])     # 링크 질량 [kg]
lc_ratio = np.array([0.5, 0.5, 0.5, 0.5])
lc_m     = L_m * lc_ratio
I_cm     = masses * L_m**2 / 12
G_ACC    = 9.81
M_LEG    = masses.sum()                         # 다리 총 질량 [kg]
M_BODY   = 10.0                                 # 몸통 질량 [kg]  ← 조정 가능
M_SYS    = M_BODY + M_LEG                       # 전체 시스템 질량 [kg]

# ══════════════════════════════════════════
# 순기구학 (힙 기준 상대 좌표, mm)
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
    """L 단위에 따라 J 단위 결정: L=L(mm)→J[mm], L=L_m(m)→J[m]"""
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
# 5차 다항식 스플라인
# ══════════════════════════════════════════
def solve_quintic_spline(t0, tf, p0, v0, a0, pf, vf, af):
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
    pos = c[0]+c[1]*t    +c[2]*t**2    +c[3]*t**3     +c[4]*t**4     +c[5]*t**5
    vel = c[1]+2*c[2]*t  +3*c[3]*t**2  +4*c[4]*t**3   +5*c[5]*t**4
    acc = 2*c[2]+6*c[3]*t+12*c[4]*t**2 +20*c[5]*t**3
    return pos, vel, acc

# ══════════════════════════════════════════
# 수직 점프 궤적
# ══════════════════════════════════════════
def make_jump_trajectory(start, h_crouch=40.0, h_jump=80.0, n=120):
    """
    힙 기준 발끝(L3) 상대 궤적.
    Phase 1(준비): 발끝 ↑ (힙-발 거리 감소 = 다리 굴곡)
    Phase 2(도약): 발끝 ↓ (힙-발 거리 증가 = 다리 신장, 힙 상승)
    Phase 3(착지): 발끝 ↑ 복귀
    """
    n1 = n // 3;  n2 = n // 6;  n3 = n - n1 - n2
    x0, y0   = start
    y_crouch = y0 + h_crouch
    y_push   = y0 - h_jump
    T1 = n1*DT;  T2 = n2*DT;  T3 = n3*DT
    c1 = solve_quintic_spline(0, T1, y0,       0, 0, y_crouch, 0, 0)
    c2 = solve_quintic_spline(0, T2, y_crouch, 0, 0, y_push,   0, 0)
    c3 = solve_quintic_spline(0, T3, y_push,   0, 0, y0,       0, 0)
    pts = []
    for j in range(n1): y, _, _ = eval_quintic(c1, j*DT); pts.append([x0, y])
    for j in range(n2): y, _, _ = eval_quintic(c2, j*DT); pts.append([x0, y])
    for j in range(n3): y, _, _ = eval_quintic(c3, j*DT); pts.append([x0, y])
    return np.array(pts), (n1, n1 + n2)

# ══════════════════════════════════════════
# θ̇, θ̈ 산출 (5차 스플라인)
# ══════════════════════════════════════════
def smooth_and_diff(theta_hist, dt):
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
# 동역학 모델 (링크 관성 / 중력 / 코리올리)
# ══════════════════════════════════════════
def _com_jac(theta_full, link_idx):
    """link_idx번 링크 COM 자코비안 (2×3) [L_m 단위 = m]"""
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
    M = np.zeros((3, 3))
    for i in range(4):
        Jc = _com_jac(theta_full, i)
        M += masses[i] * Jc.T @ Jc
        Jw = np.zeros(3);  Jw[:min(i + 1, 3)] = 1.0
        M += I_cm[i] * np.outer(Jw, Jw)
    return M

def gravity_vec(theta_full):
    alpha = np.cumsum(theta_full)
    G = np.zeros(3)
    for i in range(4):
        for j in range(min(i + 1, 3)):
            dy  = sum(L_m[k] * np.cos(alpha[k]) for k in range(j, i))
            dy += lc_m[i] * np.cos(alpha[i])
            G[j] += masses[i] * G_ACC * dy
    return G

def coriolis_vec(theta_full, dtheta3, eps=1e-7):
    n  = 3;  dM = np.zeros((n, n, n))
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

# ══════════════════════════════════════════
# 힙 가속도 (발 지면 고정 시)
# ══════════════════════════════════════════
def compute_hip_accel(theta_s, contact_mask):
    """
    발이 지면에 고정된 경우 힙 가속도 [m/s²]

    y_foot_global = y_hip + FK_y_rel = 0  (발 = 지면)
    → y_hip = -FK_y_rel  [m]
    → a_hip_y = -d²(FK_y_rel)/dt²

    FK_y_rel = joint_hist[:,3,1] [mm] → /1000 [m]
    """
    N = len(theta_s)
    # 힙 기준 발끝 y (상대, m)
    y_foot_rel = np.array([fk(theta_s[i])[3, 1] for i in range(N)]) / 1000.0

    # 수치 2차 미분
    vel_y = np.gradient(y_foot_rel, DT)
    acc_y = np.gradient(vel_y, DT)

    a_hip = np.zeros((N, 2))
    for i in range(N):
        if contact_mask[i]:
            a_hip[i, 1] = -acc_y[i]   # a_hip_y = -d²(FK_y)/dt²
            # x: 수직 점프에서 힙 수평 이동 없음 → 0

    return a_hip   # [m/s²]

# ══════════════════════════════════════════
# Floating-Base 동역학 토크 + GRF
# ══════════════════════════════════════════
def compute_torques_floating(theta_s, dtheta, ddtheta, contact_mask):
    """
    [Floating-Base 역동역학]
    τ = M(q)q̈ + C(q,q̇)q̇ + G(q) - J_m^T · F_grf      [N·m]

    GRF (뉴턴 2법칙 전체 시스템):
      F_grf_y = M_sys·g + M_sys·a_hip_y + Σ mᵢ·Jc_i_y·q̈   [N]
      F_grf_x = M_sys·a_hip_x + Σ mᵢ·Jc_i_x·q̈             [N]

      (코리올리 항 Σmᵢ·Jċ_i·q̇ 포함시 더 정확하나,
       v=a=0 경계조건 궤적에서 영향 작음 → 생략)

    단위: J_m은 L_m[m] 사용 → J_m^T·F [m·N = N·m] ✓
    """
    N        = len(theta_s)
    torques  = np.zeros((N, 3))
    tau_free = np.zeros((N, 3))
    grf_hist = np.zeros((N, 2))
    hip_acc  = compute_hip_accel(theta_s, contact_mask)  # [m/s²]

    for i in range(N):
        M_mat = mass_matrix_fn(theta_s[i])
        G     = gravity_vec(theta_s[i])
        C     = coriolis_vec(theta_s[i], dtheta[i, :3])
        tf    = M_mat @ ddtheta[i, :3] + C + G
        tau_free[i] = tf

        if contact_mask[i]:
            # 다리 링크 COM 가속도 합산 (힙 기준 상대, m/s²)
            # Σ mᵢ·Jc_i·q̈  (2-vector)
            sum_m_a_rel = np.zeros(2)
            for j in range(4):
                Jc_j = _com_jac(theta_s[i], j)          # 2×3, [m]
                sum_m_a_rel += masses[j] * (Jc_j @ ddtheta[i, :3])

            # GRF [N]
            F_grf = np.array([
                M_SYS * hip_acc[i, 0] + sum_m_a_rel[0],
                M_SYS * (G_ACC + hip_acc[i, 1]) + sum_m_a_rel[1]
            ])
            grf_hist[i] = F_grf

            # 동역학 자코비안 [m] → F·J^T 단위 N·m
            _, J_m = jacobian(theta_s[i], L=L_m)
            torques[i] = tf - J_m.T @ F_grf
        else:
            torques[i] = tf  # 비접촉: GRF=0

    return torques, tau_free, grf_hist, hip_acc

# ══════════════════════════════════════════
# 힙 글로벌 위치 계산 (시각화용)
# ══════════════════════════════════════════
def compute_hip_global(theta_s, contact_mask):
    """
    접촉 시: y_hip_global = -(FK_y_rel / 1000)  [m → mm 로 반환]
    비접촉: 마지막 접촉 시점의 힙 위치 + 낙하 탄도
    """
    N = len(theta_s)
    y_foot_rel_mm = np.array([fk(theta_s[i])[3, 1] for i in range(N)])  # [mm]
    hip_y = np.zeros(N)  # [mm]

    last_contact_idx  = 0
    last_hip_vel_mm_s = 0.0

    for i in range(N):
        if contact_mask[i]:
            hip_y[i] = -y_foot_rel_mm[i]
            last_contact_idx  = i
            last_hip_vel_mm_s = (np.gradient(hip_y[:i+1], DT)[-1]
                                 if i > 0 else 0.0)
        else:
            dt_flight = (i - last_contact_idx) * DT
            hip_y[i]  = (hip_y[last_contact_idx]
                         + last_hip_vel_mm_s * dt_flight
                         - 0.5 * G_ACC * 1000 * dt_flight**2)

    return hip_y  # [mm]

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
    n_traj = len(trajectory)
    contact_mask = np.array([i <= phase_idx[1] for i in range(n_traj + 1)])
else:
    trajectory = make_gait_trajectory(toe_start, step_x=80, lift=45, n=120)
    mode_label = '보행 (Stance+Swing)'
    n_traj = len(trajectory)
    contact_mask = np.array([i <= n_traj // 2 for i in range(n_traj + 1)])

theta_hist = ik_jacobian(THETA_INIT, trajectory)
joint_hist = np.array([fk(th) for th in theta_hist])

print("5차 스플라인 + Floating-Base 동역학 계산 중...")
theta_s, dtheta, ddtheta    = smooth_and_diff(theta_hist, DT)
torque_hist, tau_free_hist, grf_hist, hip_acc_hist = compute_torques_floating(
    theta_s, dtheta, ddtheta, contact_mask)
hip_y_global = compute_hip_global(theta_s, contact_mask)   # [mm]

print(f"완료: {len(theta_hist)} 프레임  |  모드: {mode_label}")
print(f"  전체 질량: M_body={M_BODY:.1f}kg  M_leg={M_LEG:.1f}kg  M_sys={M_SYS:.1f}kg")
print(f"  최대 토크: τ1={np.abs(torque_hist[:,0]).max():.2f}  "
      f"τ2={np.abs(torque_hist[:,1]).max():.2f}  "
      f"τ3={np.abs(torque_hist[:,2]).max():.2f}  [N·m]")
print(f"  최대 GRF:  Fx={np.abs(grf_hist[:,0]).max():.1f} N  "
      f"Fy={np.abs(grf_hist[:,1]).max():.1f} N")
print(f"  힙 최대 상승: {hip_y_global.max() - hip_y_global[0]:.1f} mm")
print(f"  체중: {M_SYS*G_ACC:.1f} N")

# ══════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(4, 3, figure=fig, wspace=0.45, hspace=0.62)

ax_leg    = fig.add_subplot(gs[:, :2])
ax_angle  = fig.add_subplot(gs[0, 2])
ax_torque = fig.add_subplot(gs[1, 2])
ax_grf    = fig.add_subplot(gs[2, 2])
ax_hip    = fig.add_subplot(gs[3, 2])

# ── 다리 축 (힙 글로벌 위치 포함)
margin = 80
x_all  = np.concatenate([joint_hist[:,:,0].ravel(), trajectory[:,0]])
y_rel_all = np.concatenate([joint_hist[:,:,1].ravel(), trajectory[:,1]])
# 힙 글로벌 위치 오프셋 적용
y_global_max = hip_y_global.max() + y_rel_all.max()
y_global_min = hip_y_global.min() + y_rel_all.min()
ax_leg.set_xlim(x_all.min()-margin, x_all.max()+margin)
ax_leg.set_ylim(y_global_min-margin, y_global_max+margin)
ax_leg.set_aspect('equal');  ax_leg.grid(True, alpha=0.25)

# 지면선 (y_global=0)
ax_leg.axhline(y=0, color='saddlebrown', lw=2.0, ls='-', alpha=0.7, label='지면')
ax_leg.fill_between([x_all.min()-margin, x_all.max()+margin],
                    -30, 0, color='saddlebrown', alpha=0.15)
ax_leg.plot(trajectory[:,0], trajectory[:,1] + hip_y_global[:len(trajectory)],
            'g--', lw=1.2, alpha=0.5, label='목표 궤적(글로벌)')

if phase_idx is not None:
    for p_idx, p_lbl, p_col in zip(
            phase_idx, ['도약 시작', '착지 시작'], ['orangered', 'royalblue']):
        ax_leg.plot(trajectory[p_idx, 0],
                    trajectory[p_idx, 1] + hip_y_global[p_idx],
                    'D', color=p_col, ms=10, zorder=7, label=p_lbl)

leg_line,  = ax_leg.plot([], [], 'o-', color='steelblue', lw=3.5, ms=8, zorder=4)
hip_dot,   = ax_leg.plot([], [], 's', color='darkorange', ms=12, zorder=6, label='힙(이동)')
trace_line,= ax_leg.plot([], [], '-', color='tomato', lw=1.8, alpha=0.8, label='발끝 궤적')
grf_quiv   = ax_leg.quiver([0],[0],[0],[0],
                           color='magenta', scale=1, scale_units='xy',
                           angles='xy', width=0.005, zorder=8)

ax_leg.set_title(f'4DOF 다리 - Floating-Base  |  {mode_label}', fontsize=11)
ax_leg.set_xlabel('x [mm]');  ax_leg.set_ylabel('y [mm] (글로벌)')
ax_leg.legend(loc='upper right', fontsize=7)
info_text = ax_leg.text(0.02, 0.04, '', transform=ax_leg.transAxes, fontsize=7.5,
                        va='bottom', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.85))

# ── 관절각
n_frames = len(theta_hist)
ax_angle.set_xlim(0, n_frames);  ax_angle.set_ylim(-30, 370)
ax_angle.grid(True, alpha=0.3)
ax_angle.set_title('관절각 변화', fontsize=10)
ax_angle.set_xlabel('프레임');  ax_angle.set_ylabel('[deg]')
line_t = [ax_angle.plot([], [], lw=1.8, label=f'θ{i+1}')[0] for i in range(4)]
if phase_idx:
    for p, c in zip(phase_idx, ['orangered', 'royalblue']):
        ax_angle.axvline(x=p+1, color=c, lw=1.2, ls=':', alpha=0.8)
ax_angle.legend(fontsize=8)

# ── 관절 토크
t_max = max(np.abs(torque_hist).max(), np.abs(tau_free_hist).max()) * 1.2 + 0.5
ax_torque.set_xlim(0, n_frames);  ax_torque.set_ylim(-t_max, t_max)
ax_torque.grid(True, alpha=0.3)
ax_torque.axhline(0, color='k', lw=0.8, ls='--')
ax_torque.set_title('관절 토크  τ = M·q̈+C·q̇+G - J^T·F_grf', fontsize=9)
ax_torque.set_xlabel('프레임');  ax_torque.set_ylabel('[N·m]')
colors_tau = ['steelblue', 'tomato', 'seagreen']
line_tau  = [ax_torque.plot([], [], lw=1.8, color=colors_tau[k],
                            label=f'τ{k+1}')[0] for k in range(3)]
line_tauf = [ax_torque.plot([], [], lw=1.0, color=colors_tau[k], ls='--', alpha=0.4,
                            label=f'τ{k+1}_free')[0] for k in range(3)]
if phase_idx:
    for p, c in zip(phase_idx, ['orangered', 'royalblue']):
        ax_torque.axvline(x=p+1, color=c, lw=1.2, ls=':', alpha=0.8)
ax_torque.legend(fontsize=7, ncol=2)

# ── GRF
grf_ref = M_SYS * G_ACC
grf_max = max(np.abs(grf_hist).max() * 1.2, grf_ref * 1.5) + 1.0
ax_grf.set_xlim(0, n_frames);  ax_grf.set_ylim(-grf_ref*0.3, grf_max)
ax_grf.grid(True, alpha=0.3)
ax_grf.axhline(0, color='k', lw=0.8, ls='--')
ax_grf.axhline(y=grf_ref, color='gray', lw=1.0, ls=':',
               label=f'체중 {grf_ref:.1f} N ({M_SYS:.0f}kg)', alpha=0.8)
ax_grf.set_title(f'GRF  F_grf = M_sys·(g+a_hip) + Σmᵢ·aᵢ', fontsize=9)
ax_grf.set_xlabel('프레임');  ax_grf.set_ylabel('[N]')
line_gx, = ax_grf.plot([], [], lw=1.8, color='darkorange', label='F_x')
line_gy, = ax_grf.plot([], [], lw=1.8, color='purple',     label='F_y')
if phase_idx:
    for p, c in zip(phase_idx, ['orangered', 'royalblue']):
        ax_grf.axvline(x=p+1, color=c, lw=1.2, ls=':', alpha=0.8)
ax_grf.legend(fontsize=8)

# ── 힙 글로벌 위치
ax_hip.set_xlim(0, n_frames)
ax_hip.set_ylim(hip_y_global.min()-20, hip_y_global.max()+20)
ax_hip.grid(True, alpha=0.3)
ax_hip.axhline(y=hip_y_global[0], color='gray', lw=1.0, ls=':', alpha=0.7)
ax_hip.set_title('힙 글로벌 높이 (발=지면=0)', fontsize=10)
ax_hip.set_xlabel('프레임');  ax_hip.set_ylabel('[mm]')
line_hip, = ax_hip.plot([], [], lw=2.0, color='darkorange', label='y_hip')
if phase_idx:
    for p, c in zip(phase_idx, ['orangered', 'royalblue']):
        ax_hip.axvline(x=p+1, color=c, lw=1.2, ls=':', alpha=0.8)
ax_hip.legend(fontsize=8)

# ── 애니메이션 상태
trace_x, trace_y     = [], []
ang_data              = [[] for _ in range(4)]
tau_data              = [[] for _ in range(3)]
tauf_data             = [[] for _ in range(3)]
gx_data, gy_data      = [], []
hip_data              = []
GRF_SCALE = 1.0   # mm/N

def init_anim():
    leg_line.set_data([], []);  hip_dot.set_data([], []);  trace_line.set_data([], [])
    for ln in line_t + line_tau + line_tauf + [line_gx, line_gy, line_hip]:
        ln.set_data([], [])
    info_text.set_text('')
    return (leg_line, hip_dot, trace_line,
            *line_t, *line_tau, *line_tauf,
            line_gx, line_gy, line_hip, info_text)

def animate(i):
    if i >= n_frames:
        return (leg_line, hip_dot, trace_line,
                *line_t, *line_tau, *line_tauf,
                line_gx, line_gy, line_hip, info_text)

    # 힙 글로벌 위치
    y_hip = hip_y_global[i]

    # 다리 관절 글로벌 좌표
    joints_rel = joint_hist[i]
    joints_gl  = joints_rel.copy()
    joints_gl[:, 1] += y_hip   # y를 힙 글로벌 위치만큼 이동
    leg_line.set_data(joints_gl[:, 0], joints_gl[:, 1])
    hip_dot.set_data([joints_gl[0, 0]], [joints_gl[0, 1]])

    # 발끝 궤적 (글로벌)
    ee_gl = joints_gl[3]
    trace_x.append(ee_gl[0]);  trace_y.append(ee_gl[1])
    trace_line.set_data(trace_x, trace_y)

    # GRF 화살표
    grf = grf_hist[i]
    if contact_mask[i] and np.linalg.norm(grf) > 0.5:
        grf_quiv.set_offsets([[ee_gl[0], ee_gl[1]]])
        grf_quiv.set_UVC([grf[0]*GRF_SCALE], [grf[1]*GRF_SCALE])
    else:
        grf_quiv.set_UVC([0], [0])

    th = theta_hist[i]
    for k in range(4):
        ang_data[k].append(np.rad2deg(th[k]))
        line_t[k].set_data(range(len(ang_data[k])), ang_data[k])

    tau   = torque_hist[i]
    tau_f = tau_free_hist[i]
    for k in range(3):
        tau_data[k].append(tau[k]);   line_tau[k].set_data(range(len(tau_data[k])),  tau_data[k])
        tauf_data[k].append(tau_f[k]); line_tauf[k].set_data(range(len(tauf_data[k])), tauf_data[k])

    gx_data.append(grf[0]);  gy_data.append(grf[1])
    line_gx.set_data(range(len(gx_data)), gx_data)
    line_gy.set_data(range(len(gy_data)), gy_data)

    hip_data.append(y_hip)
    line_hip.set_data(range(len(hip_data)), hip_data)

    if phase_idx:
        if i <= phase_idx[0]+1: phase_str = "Phase 1: 준비 (Crouch) — 접촉"
        elif i <= phase_idx[1]+1: phase_str = "Phase 2: 도약 (Push-off) — 접촉"
        else:                   phase_str = "Phase 3: 착지 복귀 — 비접촉"
    else:
        phase_str = ""

    a_hip_y = hip_acc_hist[i, 1]
    contact_str = "접촉" if contact_mask[i] else "비접촉"
    info_text.set_text(
        f"{phase_str}\n\n"
        f"힙 높이: {y_hip:+.1f} mm\n"
        f"힙 가속: a_hip_y={a_hip_y:+.2f} m/s²\n\n"
        f"GRF [{contact_str}]:\n"
        f"  Fx={grf[0]:+.1f} N\n"
        f"  Fy={grf[1]:+.1f} N\n\n"
        f"토크 [N·m]:\n"
        f"  τ1={tau[0]:+.3f}  (free:{tau_f[0]:+.3f})\n"
        f"  τ2={tau[1]:+.3f}  (free:{tau_f[1]:+.3f})\n"
        f"  τ3={tau[2]:+.3f}  (free:{tau_f[2]:+.3f})"
    )

    return (leg_line, hip_dot, trace_line,
            *line_t, *line_tau, *line_tauf,
            line_gx, line_gy, line_hip, info_text)

ani = FuncAnimation(fig, animate, init_func=init_anim,
                    frames=n_frames, interval=40, blit=True)
plt.suptitle(
    f'4DOF Leg Simulation  |  Floating-Base  |  {mode_label}\n'
    f'M_body={M_BODY:.0f}kg  M_leg={M_LEG:.0f}kg  M_sys={M_SYS:.0f}kg  '
    f'체중={M_SYS*G_ACC:.0f}N   |   τ = M·q̈+C·q̇+G - J^T·F_grf',
    fontsize=9)
plt.show()
