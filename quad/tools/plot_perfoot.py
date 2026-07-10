"""발별(HL/HR/FL/FR) 각속도·토크 그래프 — backup/graphs/joints_phase_*.png 양식.
4행(발)×2열(각속도 좌·토크 우), phase 음영(warmup/accel/steady), 관절별 한계 점선, 총질량 제목.
실행: python plot_perfoot.py <npz> <out.png> <V> [mjcf]"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

npz = sys.argv[1] if len(sys.argv) > 1 else '/tmp/v10.npz'
out = sys.argv[2] if len(sys.argv) > 2 else '/home/jsh/문서/jsh/simulation/quad/joints_perfoot.png'
V = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
mjcf = sys.argv[4] if len(sys.argv) > 4 else 'quad_real_17dof_sphere.mjcf'

# ★관절 한계 = 감속비 기반: 토크=motor_peak·N, 속도=motor_noload/N (N=기어). [[02leg-motor-spec]]
#   nominal 7/7/10.5/14 → 토크 84/84/126/168, 속도 29.6/29.6/19.7/14.8. GEAR_FOOT 등 env로 재기어 반영.
MOTOR_PEAK = 12.0; MOTOR_NOLOAD = 207.0            # 12Nm·207rad/s (84/7, 29.6*7)
GEARMAP = {'hip': 7.0, 'thigh': 7.0, 'calf': 10.5, 'foot': 14.0}
GEARN = {jt: GEARMAP[jt] * float(os.environ.get('GEAR_' + jt.upper(), '1.0')) for jt in GEARMAP}
TL = {jt: MOTOR_PEAK * GEARN[jt] for jt in GEARMAP}       # 재기어 토크한계
WL = {jt: MOTOR_NOLOAD / GEARN[jt] for jt in GEARMAP}     # 재기어 속도한계
JCOL = {'hip': 'C0', 'thigh': 'C1', 'calf': 'C2', 'foot': 'C3'}
JORDER = ['hip', 'thigh', 'calf', 'foot']
LEGS = ['HL', 'HR', 'FL', 'FR']

d = np.load(npz)
t = d['t']; tau = d['tau']; dq = d['dq']
names = [str(n) for n in d['names']]
idx = {n: i for i, n in enumerate(names)}

# 총질량 (모델 로드)
mass = None
try:
    import mujoco
    _m = mujoco.MjModel.from_xml_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mjcf', mjcf))   # tools/ → quad/mjcf/
    mass = float(_m.body_subtreemass[0])
except Exception as e:
    print('mass 로드 실패:', e)

# phase 경계: WARMUP(제자리 0.6s) → ACCEL(속도램프, ACC=0.6m/s²) → STEADY
WARM = 0.6; ACC = 0.6
accel_end = WARM + V / ACC

fig, axes = plt.subplots(len(LEGS), 2, figsize=(14, 11), sharex=True)
for r, leg in enumerate(LEGS):
    ax_w, ax_t = axes[r, 0], axes[r, 1]
    for ax in (ax_w, ax_t):
        ax.axvspan(0, WARM, color='gray', alpha=0.12)
        ax.axvspan(WARM, accel_end, color='orange', alpha=0.10)
        ax.axvspan(accel_end, t[-1], color='green', alpha=0.08)
        ax.grid(alpha=0.3)
    for jt in JORDER:
        nm = '%s_%s' % (leg, jt)
        if nm not in idx:
            continue
        i = idx[nm]; c = JCOL[jt]
        ax_w.plot(t, dq[:, i], lw=0.7, color=c, label=jt)
        ax_t.plot(t, tau[:, i], lw=0.7, color=c, label=jt)
    # 한계 점선(관절별 색)
    for jt in JORDER:
        ax_w.axhline(WL[jt], ls='--', lw=0.6, color=JCOL[jt], alpha=0.7); ax_w.axhline(-WL[jt], ls='--', lw=0.6, color=JCOL[jt], alpha=0.7)
        ax_t.axhline(TL[jt], ls='--', lw=0.6, color=JCOL[jt], alpha=0.5); ax_t.axhline(-TL[jt], ls='--', lw=0.6, color=JCOL[jt], alpha=0.5)
    ax_w.set_ylabel('%s  ω[rad/s]' % leg); ax_t.set_ylabel('%s  τ[Nm]' % leg)
    ax_w.set_ylim(-45, 45); ax_t.set_ylim(-180, 180)
    ax_w.legend(fontsize=7, ncol=4, loc='upper left')
    if r == 0:
        ax_w.set_title('ang.vel @%.1fm/s (dash=limit)' % V)
        ax_t.set_title('torque @%.1fm/s (dash=Peak)' % V)
axes[-1, 0].set_xlabel('t [s]'); axes[-1, 1].set_xlabel('t [s]')
mtxt = ('mass %.2f kg  |  ' % mass) if mass else ''
mc = 'MOTOR_CURVE' if os.environ.get('MOTOR_CURVE') else 'no motor-curve'
fig.suptitle('02_Leg @%.1fm/s  |  %sfoot gear %.1f:1 (limits v=%.1f t=%.0fNm, %s)  |  gray=warmup orange=accel green=steady'
             % (V, mtxt, GEARN['foot'], WL['foot'], TL['foot'], mc), fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=110)
print('총질량: %s kg' % (('%.2f' % mass) if mass else '?'))
print('▶ 그래프 저장: %s' % out)
