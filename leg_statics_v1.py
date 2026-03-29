"""
leg_static.py
4DOF 다리 정역학 - 피드포워드 토크 제어 + GRF 역계산 검증

  τ_cmd  = Kp(θt - θa) + Kd(θ̇t - θ̇a) + τ_ff

  τ_ff   = J^T · λ_des                       [N·m]  (원하는 GRF → 토크)
  λ_calc = (J·J^T)^{-1} · J · τ_cmd         [N]    (τ_cmd → GRF 역산)

  완벽 추종 (θt = θa):  τ_cmd = τ_ff  →  λ_calc = λ_des  ✓
  추종 오차 존재 시:     PD 보정항이 λ_calc 에 영향 (비교 가능)

자코비안 (이미지 수식):
  J_abs = [[-L1·sin(a), -L2·sin(a+b), -L3·sin(a+b+c)],
           [ L1·cos(a),  L2·cos(a+b),  L3·cos(a+b+c)]]
  J_rel = J_abs @ T,  T = [[1,0,0],[1,1,0],[1,1,1]]
  τ = J_rel^T · λ  (L_m[m] 기준 → N·m)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.font_manager as fm

# ── 한글 폰트
_fp = fm.findfont(fm.FontProperties(family='NanumGothic'))
if 'NanumGothic' not in _fp:
    fm.fontManager.addfont('/home/jsh/.local/share/fonts/NanumGothic-Regular.ttf')
mpl.rcParams['font.family']        = 'NanumGothic'
mpl.rcParams['axes.unicode_minus'] = False
for k in mpl.rcParams:
    if k.startswith("keymap."):
        mpl.rcParams[k] = []

# ══════════════════════════════════════════
# 파라미터
# ══════════════════════════════════════════
L   = np.array([120.0, 120.0, 148.0, 45.0])  # 링크 길이 [mm]
L_m = L / 1000.0                              # 링크 길이 [m]  ← 자코비안용

THETA = np.deg2rad([300.0, 270.0, 90.0, 60.0])  # 목표 관절각 θt (고정)

# PD 제어 게인 (θ1~θ3)
Kp = np.array([100.0, 100.0, 80.0])   # [N·m/rad]
Kd = np.array([ 10.0,  10.0,  8.0])   # [N·m·s/rad]

# 시뮬레이션
DT      = 0.005   # 제어 주기 [s]
TAU_LAG = 0.03    # 실제 각도 1차 지연 상수 [s] (추종 오차 생성)
INIT_ERR = np.deg2rad([3.0, -2.5, 2.0, 0.0])  # 초기 관절각 오차 [rad]

# 접촉력 프로파일
F_PEAK = 50.0   # [N]
N      = 120    # 프레임 수


# ══════════════════════════════════════════
# 순기구학 / 자코비안
# ══════════════════════════════════════════
def fk(theta):
    alpha  = np.cumsum(theta)
    joints = np.zeros((5, 2))
    for i in range(4):
        joints[i+1] = joints[i] + L[i] * np.array([np.cos(alpha[i]), np.sin(alpha[i])])
    return joints


def jacobian(theta):
    """
    J_abs, J_rel (2×3, L_m[m] 기준)
    τ = J_rel^T · λ  →  [N·m]
    """
    a = np.cumsum(theta[:3])
    J_abs = np.array([
        [-L_m[0]*np.sin(a[0]), -L_m[1]*np.sin(a[1]), -L_m[2]*np.sin(a[2])],
        [ L_m[0]*np.cos(a[0]),  L_m[1]*np.cos(a[1]),  L_m[2]*np.cos(a[2])]
    ])
    T = np.array([[1, 0, 0],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=float)
    return J_abs, J_abs @ T


# ══════════════════════════════════════════
# 사전 계산
# ══════════════════════════════════════════
joints    = fk(THETA)
toe       = joints[3]           # L3 끝점 [mm]
_, J      = jacobian(THETA)     # J: 2×3 [m/rad]  (고정 자세 → 고정 J)

JJT     = J @ J.T               # 2×2
JJT_inv = np.linalg.inv(JJT)    # (J·J^T)^{-1}

# ── 접촉력 프로파일: λy = 0 → F_PEAK → 0 [N]
lam_des = np.column_stack([
    np.zeros(N),
    F_PEAK * np.sin(np.linspace(0, np.pi, N))
])  # N×2

# ── τ_ff = J^T · λ_des  (N×3)
#    lam_des @ J  ≡  (J^T @ lam_des_i)^T 각 행
tau_ff = lam_des @ J            # N×3  [N·m]

# ── 목표각 θt (고정)
theta_t  = np.tile(THETA, (N, 1))   # N×4
dtheta_t = np.zeros((N, 4))         # 고정 자세 → 속도=0

# ── 실제각 θa: 1차 지연으로 θt 추종 (초기 오차에서 수렴)
theta_a  = np.zeros((N, 4))
dtheta_a = np.zeros((N, 4))
theta_a[0] = THETA + INIT_ERR
for i in range(1, N):
    theta_a[i]  = theta_a[i-1] + (DT / TAU_LAG) * (theta_t[i-1] - theta_a[i-1])
    dtheta_a[i] = (theta_a[i] - theta_a[i-1]) / DT

# ── PD 토크: τ_pd = Kp*(θt-θa) + Kd*(dθt-dθa)  (θ1~θ3, N×3)
err_pos = theta_t[:, :3] - theta_a[:, :3]    # N×3
err_vel = dtheta_t[:, :3] - dtheta_a[:, :3]  # N×3
tau_pd  = Kp * err_pos + Kd * err_vel         # N×3  [N·m]

# ── 최종 토크 명령: τ_cmd = τ_pd + τ_ff
tau_cmd = tau_pd + tau_ff                      # N×3  [N·m]

# ── GRF 역산: λ_calc = (J·J^T)^{-1} · J · τ_cmd  (N×2)
lam_calc = (JJT_inv @ J @ tau_cmd.T).T        # N×2  [N]

# ── 콘솔 출력
print("=== 자코비안 (고정 자세) ===")
print(f"J_rel (2×3) [m/rad]:\n{np.round(J, 4)}")
print(f"\n발끝 위치: ({toe[0]:.1f}, {toe[1]:.1f}) mm")
print(f"\n=== 피크 프레임 (i=60) ===")
i60 = N // 2
print(f"λ_des   = [{lam_des[i60,0]:+.2f}, {lam_des[i60,1]:+.2f}] N")
print(f"τ_ff    = [{tau_ff[i60,0]:+.4f}, {tau_ff[i60,1]:+.4f}, {tau_ff[i60,2]:+.4f}] N·m")
print(f"τ_pd    = [{tau_pd[i60,0]:+.4f}, {tau_pd[i60,1]:+.4f}, {tau_pd[i60,2]:+.4f}] N·m")
print(f"τ_cmd   = [{tau_cmd[i60,0]:+.4f}, {tau_cmd[i60,1]:+.4f}, {tau_cmd[i60,2]:+.4f}] N·m")
print(f"λ_calc  = [{lam_calc[i60,0]:+.2f}, {lam_calc[i60,1]:+.2f}] N")
print(f"λ 오차  = [{lam_calc[i60,0]-lam_des[i60,0]:+.2f}, "
      f"{lam_calc[i60,1]-lam_des[i60,1]:+.2f}] N  (PD 영향)")


# ══════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
gs  = gridspec.GridSpec(3, 3, figure=fig, wspace=0.42, hspace=0.55)

ax_leg = fig.add_subplot(gs[:, :2])   # 다리 (좌측 큰 창)
ax_tau = fig.add_subplot(gs[0, 2])    # τ 그래프
ax_lam = fig.add_subplot(gs[1, 2])    # λ_des vs λ_calc
ax_err = fig.add_subplot(gs[2, 2])    # θ 추종 오차

# ── 다리 축 ──────────────────────────────
pad = 60
ax_leg.set_xlim(joints[:, 0].min() - pad, joints[:, 0].max() + pad)
ax_leg.set_ylim(joints[:, 1].min() - pad * 1.2, joints[:, 1].max() + pad * 0.6)
ax_leg.set_aspect('equal')
ax_leg.grid(True, alpha=0.25)

ax_leg.axhline(y=toe[1], color='saddlebrown', lw=2, alpha=0.7)
ax_leg.fill_between([joints[:, 0].min() - pad, joints[:, 0].max() + pad],
                    joints[:, 1].min() - pad * 1.2, toe[1],
                    color='saddlebrown', alpha=0.10, label='지면')

# 다리 링크 (고정)
ax_leg.plot(joints[:, 0], joints[:, 1],
            'o-', color='steelblue', lw=4, ms=10, zorder=4, label='링크 (고정)')
ax_leg.plot(0, 0, 'ks', ms=12, zorder=6, label='기반 (고정)')
ax_leg.plot(*toe, 'ro', ms=12, zorder=7, label='발끝 (접촉점)')
for j, name in enumerate(['무릎', '발목', '발가락', '발끝']):
    ax_leg.annotate(name, joints[j+1], fontsize=7.5, color='dimgray',
                    xytext=(6, 4), textcoords='offset points')

# λ_des 화살표 (동적)
GRF_SCALE = 1.0   # mm/N
grf_line, = ax_leg.plot([], [], '-',  color='magenta', lw=3.5, zorder=8)
grf_tip,  = ax_leg.plot([], [], '^',  color='magenta', ms=13,  zorder=9, label='λ_des (GRF)')

ax_leg.set_title('4DOF 다리  |  고정 자세  |  τ_cmd = Kp·Δθ + Kd·Δθ̇ + J^T·λ', fontsize=10)
ax_leg.set_xlabel('x [mm]')
ax_leg.set_ylabel('y [mm]')
ax_leg.legend(loc='upper right', fontsize=8)

info = ax_leg.text(0.02, 0.03, '', transform=ax_leg.transAxes, fontsize=8.5,
                   va='bottom', family='monospace',
                   bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.90))

# ── τ 그래프 ─────────────────────────────
t_max = max(np.abs(tau_cmd).max(), np.abs(tau_ff).max()) * 1.25 + 0.01
ax_tau.set_xlim(0, N)
ax_tau.set_ylim(-t_max * 0.3, t_max)
ax_tau.axhline(0, color='k', lw=0.8, ls='--')
ax_tau.grid(True, alpha=0.3)
ax_tau.set_title('토크  [N·m]', fontsize=10)
ax_tau.set_xlabel('프레임')
ax_tau.set_ylabel('[N·m]')
colors = ['steelblue', 'tomato', 'seagreen']
# τ_cmd (실선), τ_ff (점선), τ_pd (점점선)
lines_cmd = [ax_tau.plot([], [], lw=2.2, color=colors[k], label=f'τ_cmd{k+1}')[0]
             for k in range(3)]
lines_ff  = [ax_tau.plot([], [], lw=1.2, ls='--', color=colors[k], alpha=0.6,
                          label=f'τ_ff{k+1}')[0] for k in range(3)]
lines_pd  = [ax_tau.plot([], [], lw=1.0, ls=':', color=colors[k], alpha=0.5,
                          label=f'τ_pd{k+1}')[0] for k in range(3)]
ax_tau.legend(fontsize=6.5, ncol=3)

# ── λ 비교 그래프 ─────────────────────────
lam_max = max(np.abs(lam_des).max(), np.abs(lam_calc).max()) * 1.3 + 1.0
ax_lam.set_xlim(0, N)
ax_lam.set_ylim(-lam_max * 0.15, lam_max)
ax_lam.axhline(0, color='k', lw=0.8, ls='--')
ax_lam.grid(True, alpha=0.3)
ax_lam.set_title('λ_des  vs  λ_calc = (JJᵀ)⁻¹·J·τ_cmd  [N]', fontsize=9)
ax_lam.set_xlabel('프레임')
ax_lam.set_ylabel('[N]')
line_ld,  = ax_lam.plot([], [], lw=2.5, color='purple',     label='λ_des y (목표)')
line_lc,  = ax_lam.plot([], [], lw=1.8, ls='--', color='orangered', label='λ_calc y (역산)')
ax_lam.axhline(y=F_PEAK, color='gray', lw=0.8, ls=':', alpha=0.7)
ax_lam.legend(fontsize=8)

# ── θ 추종 오차 ───────────────────────────
err_deg = np.rad2deg(err_pos)   # N×3
err_max = np.abs(err_deg).max() * 1.3 + 0.1
ax_err.set_xlim(0, N)
ax_err.set_ylim(-err_max * 0.1, err_max)
ax_err.axhline(0, color='k', lw=0.8, ls='--')
ax_err.grid(True, alpha=0.3)
ax_err.set_title('추종 오차  θt - θa  [deg]', fontsize=10)
ax_err.set_xlabel('프레임')
ax_err.set_ylabel('[deg]')
lines_err = [ax_err.plot([], [], lw=2.0, color=colors[k], label=f'Δθ{k+1}')[0]
             for k in range(3)]
ax_err.legend(fontsize=8)

# ── 애니메이션 버퍼 ─────────────────────
bufs = {
    'cmd':  [[] for _ in range(3)],
    'ff':   [[] for _ in range(3)],
    'pd':   [[] for _ in range(3)],
    'ld':   [], 'lc':   [],
    'err':  [[] for _ in range(3)],
}


def init_anim():
    grf_line.set_data([], [])
    grf_tip.set_data([], [])
    for ln in lines_cmd + lines_ff + lines_pd + lines_err + [line_ld, line_lc]:
        ln.set_data([], [])
    info.set_text('')
    return (grf_line, grf_tip,
            *lines_cmd, *lines_ff, *lines_pd,
            line_ld, line_lc, *lines_err, info)


def animate(i):
    lam_d = lam_des[i]
    lam_c = lam_calc[i]
    tff   = tau_ff[i]
    tpd   = tau_pd[i]
    tcmd  = tau_cmd[i]
    ep    = err_deg[i]

    # GRF 화살표 (λ_des)
    if lam_d[1] > 0.5:
        tip_y = toe[1] + lam_d[1] * GRF_SCALE
        grf_line.set_data([toe[0], toe[0]], [toe[1], tip_y])
        grf_tip.set_data([toe[0]], [tip_y])
    else:
        grf_line.set_data([], [])
        grf_tip.set_data([], [])

    # 토크
    for k in range(3):
        bufs['cmd'][k].append(tcmd[k])
        bufs['ff'][k].append(tff[k])
        bufs['pd'][k].append(tpd[k])
        fr = range(len(bufs['cmd'][k]))
        lines_cmd[k].set_data(fr, bufs['cmd'][k])
        lines_ff[k].set_data(fr, bufs['ff'][k])
        lines_pd[k].set_data(fr, bufs['pd'][k])

    # λ 비교
    bufs['ld'].append(lam_d[1])
    bufs['lc'].append(lam_c[1])
    fr2 = range(len(bufs['ld']))
    line_ld.set_data(fr2, bufs['ld'])
    line_lc.set_data(fr2, bufs['lc'])

    # θ 오차
    for k in range(3):
        bufs['err'][k].append(ep[k])
        lines_err[k].set_data(range(len(bufs['err'][k])), bufs['err'][k])

    # 텍스트
    info.set_text(
        f"τ_cmd = Kp·Δθ + Kd·Δθ̇ + J^T·λ\n"
        f"λ_calc = (JJᵀ)⁻¹·J·τ_cmd\n"
        f"─────────────────────────\n"
        f"λ_des  = [{lam_d[0]:+6.2f}, {lam_d[1]:+6.2f}] N\n"
        f"λ_calc = [{lam_c[0]:+6.2f}, {lam_c[1]:+6.2f}] N\n"
        f"Δλy    = {lam_c[1]-lam_d[1]:+.2f} N  (PD 기여)\n\n"
        f"τ_ff  = [{tff[0]:+.3f}, {tff[1]:+.3f}, {tff[2]:+.3f}]\n"
        f"τ_pd  = [{tpd[0]:+.3f}, {tpd[1]:+.3f}, {tpd[2]:+.3f}]\n"
        f"τ_cmd = [{tcmd[0]:+.3f}, {tcmd[1]:+.3f}, {tcmd[2]:+.3f}]\n\n"
        f"Δθ [deg]: [{ep[0]:+.2f}, {ep[1]:+.2f}, {ep[2]:+.2f}]"
    )

    return (grf_line, grf_tip,
            *lines_cmd, *lines_ff, *lines_pd,
            line_ld, line_lc, *lines_err, info)


ani = FuncAnimation(fig, animate, init_func=init_anim,
                    frames=N, interval=40, blit=True)

plt.suptitle(
    f'정역학 피드포워드 제어  |  τ_cmd = Kp·Δθ + Kd·Δθ̇ + J^T·λ_des\n'
    f'λ_calc = (J·Jᵀ)⁻¹·J·τ_cmd  (역산 GRF, λ_des와 비교)',
    fontsize=11
)
plt.show()
