"""
gait_sim_v5.py  —  4족 보행 Gait 시뮬레이터
v5: FR/FL hip 기준 z축 180° 보정 + Swing/Stance 위상 기반 Joint4 각도 제어
    · FR/FL만 DH→sim 변환 시 x/y 반전
    · FK/프레임 렌더링/홈 포인트에 동일 변환 적용
    · FR/FL 5관절 (foot 링크 a5=0.045 추가)
    · Joint5 자동 조정으로 wrist 각도 제약 유지
"""

import math
import time
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
STEP_HEIGHT = 0.2
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
# HR/HL: 5관절, Joint1 α=-π/2
DH_HIND = [
    (-math.pi/2, 0.0,   0.0,    ),   # Joint 1: Hip Abduction
    (0.0,        0.21,  0.0075, ),   # Joint 2: Hip Pitch
    (0.0,        0.21,  0.0,    ),   # Joint 3: Knee
    (0.0,        0.148, 0.0,    ),   # Joint 4: Lower leg
    (0.0,        0.045, 0.0,    ),   # Joint 5: Foot
]

# front IK 상수 (DH_FRONT와 동기화)
_A2_F = 0.21; _A3_F = 0.235; _A4_F = 0.1; _A5_F = 0.045; _D2_F = 0.0075

# ── Q_HOME ──────────────────────────────────────────────────
Q_HOME_FRONT_DEG = [0.0, 157.5, 22.5, 30.6583, 59.3417]   # 5관절
Q_HOME_HIND_DEG  = [0.0, -150.0, -90.0, 90.0, 60.0]       # 5관절
Q_HOME_FRONT = [math.radians(a) for a in Q_HOME_FRONT_DEG]
Q_HOME_HIND  = [math.radians(a) for a in Q_HOME_HIND_DEG]

PHI_FRONT    = Q_HOME_FRONT[1] + Q_HOME_FRONT[2] + Q_HOME_FRONT[3]
PHI_HIND     = Q_HOME_HIND[1]  + Q_HOME_HIND[2]  + Q_HOME_HIND[3]
THETA5_FRONT = PHI_FRONT + Q_HOME_FRONT[4]   # θ2+θ3+θ4+θ5 고정 (전방)
THETA5_HIND  = PHI_HIND  + Q_HOME_HIND[4]    # θ2+θ3+θ4+θ5 고정 (후방)

Q_HOME_PER_LEG   = [Q_HOME_FRONT, Q_HOME_FRONT, Q_HOME_HIND, Q_HOME_HIND]
PHI_PER_LEG      = [PHI_FRONT, PHI_FRONT, PHI_HIND, PHI_HIND]
TRAJ_PT_IDX_PER_LEG = [4, 4, 4, 4]      # 전 다리: Joint4 축 기준 출력

# ── 다리 레이아웃 ────────────────────────────────────────────
LEG_NAMES        = ['FR', 'FL', 'HR', 'HL']
LEG_COLORS       = ['#00d4ff', '#ff6b35', '#00ff99', '#ffcc00']
LEG_DH           = [DH_FRONT, DH_FRONT, DH_HIND, DH_HIND]
N_JOINTS_PER_LEG = [5, 5, 5, 5]
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


def _dh_to_sim(vec, front_leg=False):
    # Right-handed remap (det=+1): [x_dh, y_dh, z_dh] -> [z_dh, -y_dh, x_dh]
    sim = np.array([vec[2], -vec[1], vec[0]], dtype=float)
    if front_leg:
        sim[:2] *= -1.0
    return sim


def _sim_to_dh(vec, front_leg=False):
    sim = np.array(vec, dtype=float)
    if front_leg:
        sim[:2] *= -1.0
    # Inverse of _dh_to_sim
    return np.array([sim[2], -sim[1], sim[0]], dtype=float)


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


_fk_front_home      = forward_kinematics(Q_HOME_FRONT, DH_FRONT)
_FRONT_J4_TO_J5_DH  = np.array(_fk_front_home[5]) - np.array(_fk_front_home[4])
_FRONT_J4_TO_J5_SIM = _dh_to_sim(_FRONT_J4_TO_J5_DH, front_leg=True)

_fk_hind_home      = forward_kinematics(Q_HOME_HIND, DH_HIND)
_HIND_J4_TO_J5_DH  = np.array(_fk_hind_home[5]) - np.array(_fk_hind_home[4])
_HIND_J4_TO_J5_SIM = _dh_to_sim(_HIND_J4_TO_J5_DH, front_leg=False)

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

def analytical_ik_hind(Px, Py, Pz, phi, dh, theta5_target=None):
    """HR/HL IK: α=-π/2, 4관절 해 (theta5_target 지정 시 J5 위치 타겟)"""
    a2 = dh[1][1]; a3 = dh[2][1]; a4 = dh[3][1]; d2 = dh[1][2]

    D2 = Px**2 + Py**2 - d2**2
    if D2 < 0:
        return None
    R = math.sqrt(D2)
    theta1 = math.atan2(-Px, Py) - math.atan2(R, d2)

    c1, s1 = math.cos(theta1), math.sin(theta1)
    x_s = c1 * Px + s1 * Py
    Z   = -Pz

    if theta5_target is not None:
        a5 = dh[4][1]
        x2 = x_s - a4 * math.cos(phi) - a5 * math.cos(theta5_target)
        z2 = Z   - a4 * math.sin(phi) - a5 * math.sin(theta5_target)
    else:
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
    # 7차 Bezier: P0=P1=P2=start, P5=P6=P7=end
    # → B'=0, B''=0 at both endpoints (C² 연속, 각가속도 스파이크 제거)
    t = sw_t
    h = np.array([0.0, 0.0, step_height])
    A  = p_start
    B  = p_end
    P3 = A + (B - A) * (1/3) + h   # 상승
    P4 = A + (B - A) * (2/3) + h   # 하강
    b  = 1.0 - t
    cA  = b**7 + 7*b**6*t + 21*b**5*t**2   # P0+P1+P2 합산 계수
    cP3 = 35*b**4*t**3
    cP4 = 35*b**3*t**4
    cB  = 21*b**2*t**5 + 7*b*t**6 + t**7   # P5+P6+P7 합산 계수
    return cA*A + cP3*P3 + cP4*P4 + cB*B


def stance_foot_pos(st_t, p_contact, body_vel, stance_dur):
    # smootherstep: 시작/끝 속도·가속도=0, 중간 최대 속도
    s = st_t**3 * (10.0 - 15.0*st_t + 6.0*st_t**2)
    return p_contact - body_vel * stance_dur * s

# ══════════════════════════════════════════════════════════════
# 3. 궤적 사전 계산
# ══════════════════════════════════════════════════════════════
N_FRAMES   = int(N_CYCLES * GAIT_PERIOD / DT)
sched      = GaitScheduler()
stance_dur = GAIT_PERIOD * (1.0 - SWING_RATIO)
body_vel   = np.array([BODY_VX, 0.0, 0.0])

# 홈 기준점: FR/FL은 Joint4 축, HR/HL은 말단 축
home_foot_per_leg = [
    _dh_to_sim(
        forward_kinematics(Q_HOME_PER_LEG[leg], dh=LEG_DH[leg])[TRAJ_PT_IDX_PER_LEG[leg]],
        front_leg=(leg < 2)
    )
    for leg in range(4)
]
home_foot = home_foot_per_leg[0]   # FR 기준 (지면 높이 등)

joint_hist = np.zeros((N_FRAMES, 4, N_JOINTS_MAX))
foot_hist  = np.zeros((N_FRAMES, 4, 3))
phase_hist = np.zeros((N_FRAMES, 4))
swing_flag = np.zeros((N_FRAMES, 4), dtype=bool)
frame_calc_time = np.zeros(N_FRAMES, dtype=float)

# 궤적 자동 제한: 각 조인트 최대 각속도 한계(rad/s)
JOINT_VEL_LIMIT_RAD_S = np.array([14.66, 15.91, 15.91, 14.66, 14.66], dtype=float)
#VEL_LIMIT_MARGIN = 0.98
VEL_LIMIT_MARGIN = 999 # 제한 최적화 비활성화
MAX_TRAJ_OPT_ITERS = 6

print("─" * 55)
print(f"궤적 계산 중...  [{GAIT_TYPE}]  {N_CYCLES}사이클  {N_FRAMES}프레임")
print(f"  v={V}m/s  T={T}s  D={D}  →  d_min={D_MIN:.3f}m  d={STRIDE_D}m")
print(f"  T_sw={T_SW:.3f}s  T_st={T_ST:.3f}s  step(body)={STEP_LENGTH*1e3:.1f}mm  h={STEP_HEIGHT*1e3:.0f}mm")
print(f"  홈 기준점(FR:J4): X={home_foot[0]*1e3:.1f}mm  "
      f"Y={home_foot[1]*1e3:.1f}mm  Z={home_foot[2]*1e3:.1f}mm")

traj_scale = 1.0
height_scale = 1.0
opt_iter_used = 0
peak_per_joint = np.zeros(N_JOINTS_MAX, dtype=float)
calc_total = 0.0

for opt_iter in range(1, MAX_TRAJ_OPT_ITERS + 1):
    opt_iter_used = opt_iter

    joint_hist.fill(0.0)
    foot_hist.fill(0.0)
    phase_hist.fill(0.0)
    swing_flag.fill(False)
    frame_calc_time.fill(0.0)

    # Stance 시작 다리(t=0에서 is_swing=False)의 contact 위치를
    # home이 아닌 home+step_length로 초기화 (swing이 막 끝난 착지 위치)
    _step_vec = np.array([STEP_LENGTH * traj_scale, 0.0, 0.0])
    foot_contact    = [
        home_foot_per_leg[leg].copy() + (np.zeros(3) if sched.is_swing(leg, 0) else _step_vec)
        for leg in range(4)
    ]
    foot_sw_start   = [home_foot_per_leg[leg].copy() for leg in range(4)]
    foot_local_prev = [foot_contact[leg].copy() for leg in range(4)]
    prev_swing      = [sched.is_swing(leg, 0) for leg in range(4)]

    calc_start = time.perf_counter()
    for fi in range(N_FRAMES):
        frame_start = time.perf_counter()
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
                sw_t  = sched.swing_t(leg, t)
                p_end = home_foot_per_leg[leg] + np.array([STEP_LENGTH * traj_scale, 0, 0])
                foot_loc = swing_foot_pos(sw_t, foot_sw_start[leg], p_end,
                                         step_height=STEP_HEIGHT * height_scale)
            else:
                st_t = sched.stance_t(leg, t)
                foot_loc = stance_foot_pos(st_t, foot_contact[leg], body_vel * traj_scale, stance_dur)

            foot_local_prev[leg] = foot_loc.copy()
            prev_swing[leg]      = is_sw
            foot_hist[fi, leg]   = LEG_HIP_OFFSETS[leg] + foot_loc

            # IK: J4 궤적점에 J4→J5 오프셋을 더해 J5(발끝) 기준으로 환산
            if leg < 2:   # FR, FL: 5관절 front IK (leg_IK3_FB 기준)
                foot_ik_sim = foot_loc + _FRONT_J4_TO_J5_SIM
                foot_dh = _sim_to_dh(foot_ik_sim, front_leg=True)
                q = analytical_ik_front(foot_dh[0], foot_dh[1], foot_dh[2],
                                        PHI_FRONT, THETA5_FRONT)
                if q is None:
                    q = list(Q_HOME_FRONT)
            else:         # HR, HL: 5관절 hind IK + Joint5 고정(60deg)
                foot_ik_sim = foot_loc + _HIND_J4_TO_J5_SIM
                foot_dh = _sim_to_dh(foot_ik_sim, front_leg=False)
                q_h = analytical_ik_hind(foot_dh[0], foot_dh[1], foot_dh[2],
                                         PHI_HIND, dh=DH_HIND, theta5_target=THETA5_HIND)
                if q_h is None:
                    q = list(Q_HOME_HIND)
                else:
                    q = list(q_h) + [Q_HOME_HIND[4]]

            nj = N_JOINTS_PER_LEG[leg]
            joint_hist[fi, leg, :nj] = q[:nj]
        frame_calc_time[fi] = time.perf_counter() - frame_start

    calc_total = time.perf_counter() - calc_start

    # 각도 언랩 후 미분해서 각속도(rad/s) 계산
    joint_hist_unwrapped = np.unwrap(joint_hist, axis=0)
    joint_vel_hist = np.zeros_like(joint_hist)
    joint_vel_hist[1:, :, :] = (joint_hist_unwrapped[1:, :, :] - joint_hist_unwrapped[:-1, :, :]) / DT

    # 프레임별 축(th1~th5) 각속도: 다리 4개 중 최대 절대값
    joint_vel_axis_max_abs = np.max(np.abs(joint_vel_hist), axis=1)

    peak_per_joint = np.max(np.abs(joint_vel_hist), axis=(0, 1))
    ratio_per_joint = peak_per_joint / JOINT_VEL_LIMIT_RAD_S
    worst_ratio = float(np.max(ratio_per_joint))

    if worst_ratio <= VEL_LIMIT_MARGIN:
        break

    # 속도는 궤적 스케일에 거의 비례하므로 비율 기반으로 감쇠
    scale_decay = max(0.60, min(0.98 / worst_ratio, 0.98))
    traj_scale *= scale_decay
    height_scale *= scale_decay

print("완료.")
print("궤적 제한 최적화:")
print(f"  iterations={opt_iter_used}/{MAX_TRAJ_OPT_ITERS}  traj_scale={traj_scale:.4f}  height_scale={height_scale:.4f}")
print("  joint peak/limit [rad/s]: "
      + "  ".join(f"th{i+1}:{peak_per_joint[i]:.3f}/{JOINT_VEL_LIMIT_RAD_S[i]:.2f}"
                  for i in range(N_JOINTS_MAX)))
print("연산 시간 통계(프레임 기준):")
print(f"  총 계산 시간={calc_total*1e3:.2f}ms  평균={frame_calc_time.mean()*1e3:.4f}ms"
      f"  최대={frame_calc_time.max()*1e3:.4f}ms  지터(std)={frame_calc_time.std()*1e3:.4f}ms")
print(f"  목표 주기 DT={DT*1e3:.3f}ms  오버런 프레임={int(np.sum(frame_calc_time > DT))}/{N_FRAMES}")

print("조인트 통계 (FR / HR):")
for leg in [0, 2]:   # FR=0, HR=2
    nj  = N_JOINTS_PER_LEG[leg]
    vel = joint_vel_hist[:, leg, :nj]
    acc = np.zeros_like(vel)
    acc[1:] = (vel[1:] - vel[:-1]) / DT
    print(f"  {LEG_NAMES[leg]}:")
    for j in range(nj):
        w = vel[:, j];  a = acc[:, j]
        print(f"    th{j+1}:  vel  peak={np.max(np.abs(w)):8.3f} rad/s   rms={np.sqrt(np.mean(w*w)):8.3f} rad/s")
        print(f"          acc  peak={np.max(np.abs(a)):8.3f} rad/s²  rms={np.sqrt(np.mean(a*a)):8.3f} rad/s²")
print("─" * 55)

# FR(leg=0) 각속도·각가속도·각저크
joint_vel_FR = joint_vel_hist[:, 0, :]                        # (N_FRAMES, 5)
joint_acc_FR = np.zeros_like(joint_vel_FR)
joint_acc_FR[1:] = (joint_vel_FR[1:] - joint_vel_FR[:-1]) / DT
joint_jrk_FR = np.zeros_like(joint_acc_FR)
joint_jrk_FR[1:] = (joint_acc_FR[1:] - joint_acc_FR[:-1]) / DT

# ══════════════════════════════════════════════════════════════
# 4. 시각화 설정
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 9))
fig.patch.set_facecolor('#1a1a2e')
gs  = gridspec.GridSpec(3, 3, figure=fig, wspace=0.38, hspace=0.65,
                        left=0.03, right=0.98, top=0.93, bottom=0.07)

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
ax3d.set_title(f'Gait Sim v5  [{GAIT_TYPE.upper()}]  v={V}m/s  T={T}s  D={D}',
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

_fr = np.arange(N_FRAMES)
axis_colors = ['#ff6b6b', '#ffd166', '#06d6a0', '#4cc9f0', '#f72585']

# ── (1,1) 위상 다이어그램
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

# ── (2,1) 발끝 Z 높이
ax_z = fig.add_subplot(gs[1, 1])
_style_ax(ax_z, 'Step height [m]', ylabel='Z [m]')
ax_z.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_z.plot(_fr, foot_hist[:, leg, 2], lw=1.8,
              color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_z.axhline(gnd_z, color='white', lw=0.8, ls=':', alpha=0.5, label='ground')
ax_z.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
            edgecolor=_gray, ncol=5)
z_cursor = ax_z.axvline(x=0, color='white', lw=1.5, ls='--')

# ── (3,1) Lower Leg (th4)
ax_ang = fig.add_subplot(gs[2, 1])
_style_ax(ax_ang, 'Lower Leg th4 [deg]', ylabel='[deg]')
ax_ang.set_xlim(0, N_FRAMES)
for leg in range(4):
    ax_ang.plot(_fr, np.degrees(joint_hist[:, leg, 3]), lw=1.8,
                color=LEG_COLORS[leg], label=LEG_NAMES[leg])
ax_ang.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
              edgecolor=_gray, ncol=4)
ang_cursor = ax_ang.axvline(x=0, color='white', lw=1.5, ls='--')

# ── (1,2) FR Joint Angular Velocity (th1~th5)
ax_w = fig.add_subplot(gs[0, 2])
_style_ax(ax_w, 'FR Joint Angular Velocity [rad/s]', ylabel='[rad/s]')
ax_w.set_xlim(0, N_FRAMES)
for j in range(N_JOINTS_MAX):
    ax_w.plot(_fr, joint_vel_FR[:, j], lw=1.6,
              color=axis_colors[j % len(axis_colors)], label=f'th{j+1}')
ax_w.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
            edgecolor=_gray, ncol=5)
w_cursor = ax_w.axvline(x=0, color='white', lw=1.5, ls='--')

# ── (2,2) FR Joint Angular Acceleration (th1~th5)
ax_acc = fig.add_subplot(gs[1, 2])
_style_ax(ax_acc, 'FR Joint Angular Acceleration [rad/s²]', ylabel='[rad/s²]')
ax_acc.set_xlim(0, N_FRAMES)
for j in range(N_JOINTS_MAX):
    ax_acc.plot(_fr, joint_acc_FR[:, j], lw=1.6,
                color=axis_colors[j % len(axis_colors)], label=f'th{j+1}')
ax_acc.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
              edgecolor=_gray, ncol=5)
acc_cursor = ax_acc.axvline(x=0, color='white', lw=1.5, ls='--')

# ── (3,2) FR Joint Jerk (th1~th5)
ax_jrk = fig.add_subplot(gs[2, 2])
_style_ax(ax_jrk, 'FR Joint Jerk [rad/s³]', ylabel='[rad/s³]')
ax_jrk.set_xlim(0, N_FRAMES)
for j in range(N_JOINTS_MAX):
    ax_jrk.plot(_fr, joint_jrk_FR[:, j], lw=1.6,
                color=axis_colors[j % len(axis_colors)], label=f'th{j+1}')
ax_jrk.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
              edgecolor=_gray, ncol=5)
jrk_cursor = ax_jrk.axvline(x=0, color='white', lw=1.5, ls='--')

all_cursors = [phase_cursor, z_cursor, ang_cursor, w_cursor, acc_cursor, jrk_cursor]

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
        pts    = [_dh_to_sim(p, front_leg=(leg < 2)) for p in pts_dh]
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

        T_dh = np.eye(4)
        for j in range(nj + 1):
            orig_sim = _dh_to_sim(T_dh[:3, 3], front_leg=(leg < 2))
            pos = hip + orig_sim
            for ax_i in range(3):
                dv = _dh_to_sim(T_dh[:3, ax_i], front_leg=(leg < 2))
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
    f'Gait Sim v5  |  {GAIT_TYPE.upper()}  |  '
    f'v={V}m/s  T={T}s  D={D}  '
    f'd={STRIDE_D}m(d_min={D_MIN:.2f}m)  '
    f'T_sw={T_SW:.2f}s  step={STEP_LENGTH*1e3:.0f}mm  h={STEP_HEIGHT*1e2:.0f}cm',
    color='white', fontsize=9
)
plt.show()
