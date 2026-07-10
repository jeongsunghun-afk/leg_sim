"""기립 MuJoCo 물리기반 궤적최적화 (predictive sampling / MPPI).
   물리엔진이 haunch(엉덩이)·발 등 모든 접촉을 자동처리 → contact-implicit 정공법.
   관절기준궤적 q_ref(t)를 샘플링·롤아웃(PD추종+중력보상)·비용평가·MPPI갱신으로 개선.
   sit→stand 기립을 발견. 출력: /tmp/getup_stand.npz (WBC/PD 추종 포맷).
실행: /home/jsh/miniforge3/envs/proxddp/bin/python getup_mppi.py
"""
import os, numpy as np, mujoco
from scipy.ndimage import gaussian_filter1d
_HERE = os.path.dirname(os.path.abspath(__file__))
QUAD = os.path.normpath(os.path.join(_HERE, '..', '..'))   # offline/getup/ → quad/
MJCF = os.path.join(QUAD, 'mjcf', 'quad_real_17dof_waist_sphere.mjcf')
np.random.seed(0)

m = mujoco.MjModel.from_xml_path(MJCF); nu = m.nu
qs = np.array([float(x) for x in open('/tmp/q_sit.txt').read().split()])       # 정착 sit
qst = np.array([float(x) for x in open('/tmp/q_stand.txt').read().split()])    # 정착 stand
qj_sit = qs[7:7+nu]; qj_stand = qst[7:7+nu]
TAU = np.array([84,84,126,96]*4 + [84.])[:nu]   # 우리 모터셋(발목 8:1=96), waist=84
LEGS = ['HL','HR','FL','FR']
fg = {L: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, L+'_sphere') for L in LEGS}

T = 160; dt = 0.01; sub = max(1, round(dt/m.opt.timestep))
KP = 120.0; KD = 4.0   # ★C++ 추종(GETUP_TRAJ_KP/KD)과 일치 → MPPI 결과를 C++이 그대로 재현

def rollout(qref, record=False):
    """sit에서 q_ref 궤적을 PD+중력보상 추종. 비용 반환(작을수록 좋음)."""
    d = mujoco.MjData(m); d.qpos[:] = qs; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    cost = 0.0; maxtilt = 0.0; zs = []
    ztgt = np.linspace(qs[2], 0.515, T)   # base z 목표 프로파일(상승)
    log = []
    for k in range(T):
        for _ in range(sub):
            tau = d.qfrc_bias[6:6+nu] + KP*(qref[k]-d.qpos[7:7+nu]) - KD*d.qvel[6:6+nu]
            d.ctrl[:] = np.clip(tau, -TAU, TAU); mujoco.mj_step(m, d)
        w,x,y,z = d.qpos[3:7]; tilt = np.degrees(np.arccos(max(-1,min(1,1-2*(x*x+y*y)))))
        maxtilt = max(maxtilt, tilt); zs.append(d.qpos[2])
        cost += -8000.0*d.qpos[2] + 60.0*max(0.0,tilt-45)**2  # ★높이 보상 + tilt 45°↑ 강한 페널티(gather bounce 완화)
        if record: log.append((d.qpos.copy(), d.qvel.copy()))
    zf = d.qpos[2]; zend = np.mean(zs[-15:])
    cost += -3e5*zend                                          # ★종료 높이 대폭 보상(기립 강제)
    if maxtilt > 110: cost += 1e5*(maxtilt-110)               # 완전전복만 페널티
    return (cost, zf, maxtilt, log) if record else cost

# ── 노미널: ★작동하는 kinematic gather 궤적으로 seed(핵심=gather로 CoM 전진. 단순보간은 gather없어 캡). ──
_kt = open('/tmp/getup_traj.txt').read().split('\n')
_hdr = _kt[0].split(); _N = int(_hdr[0])
nominal = np.array([[float(v) for v in _kt[1+k].split()][1:1+nu] for k in range(_N)])  # q[17](phase 제외)
T = _N; dt = float(_hdr[1]); sub = max(1, round(dt/m.opt.timestep))
print('seed=kinematic gather 궤적 (%d프레임)' % T)

# ── MPPI ──
Nsamp = 48; ITERS = 18; sigma = 0.22; lam = 8000.0
print('MPPI: T=%d Nsamp=%d iters=%d' % (T, Nsamp, ITERS))
c0 = rollout(nominal); print('노미널 비용=%.0f' % c0)
for it in range(ITERS):
    eps = np.random.randn(Nsamp, T, nu) * sigma
    for s in range(Nsamp):    # 시간축 스무딩(부드러운 궤적)
        eps[s] = gaussian_filter1d(eps[s], sigma=6, axis=0)
    costs = np.array([rollout(nominal + eps[s]) for s in range(Nsamp)])
    beta = costs.min(); wts = np.exp(-(costs-beta)/lam); wts /= wts.sum()
    nominal = nominal + np.einsum('s,stu->tu', wts, eps)
    nominal = gaussian_filter1d(nominal, sigma=2, axis=0)
    cbest, zf, mt, _ = rollout(nominal, record=True)
    print('  iter%2d best=%.0f zf=%.3f maxtilt=%.1f (평균샘플=%.0f)' % (it, cbest, zf, mt, costs.mean()))
    sigma *= 0.92

# ── 최종 궤적 저장 ──
cbest, zf, mt, log = rollout(nominal, record=True)
print('최종: zf=%.3f maxtilt=%.1f' % (zf, mt))
fq = np.array([l[0] for l in log])          # full qpos
Q = fq[:, 7:7+nu]
DQ = np.array([l[1][6:6+nu] for l in log])
d2 = mujoco.MjData(m); com = np.zeros((T,3)); bz = np.zeros(T)
for k in range(T):
    d2.qpos[:] = fq[k]; mujoco.mj_forward(m, d2); com[k] = d2.subtree_com[0].copy(); bz[k] = fq[k][2]
comv = np.gradient(com, dt, axis=0); acom = np.gradient(comv, dt, axis=0)
mj_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(nu)]
np.savez('/tmp/getup_stand.npz', q=nominal, dq=DQ, tau=np.zeros((T-1,nu)),
         base_z=bz, sched=np.array(['B']*T, dtype=object), dt=dt,
         com_ref=com, comv_ref=comv, acom_ref=acom, full_qpos=fq, mj_names=np.array(mj_names))
print('저장: /tmp/getup_stand.npz (MPPI %d스텝)' % T)
# ★C++ 뷰어 추종용 txt(phase,q[17],dq[17]). MPPI는 no-FF PD로 최적화 → dq=0(C++ 속도항=순수감쇠로 일치). 끝 chunk=wbic 서기홀드.
_sched = [0]*(T-15) + [2]*15
with open('/tmp/getup_traj.txt','w') as f:
    f.write('%d %g\n' % (T, dt))
    for k in range(T):
        f.write('%d ' % _sched[k] + ' '.join('%.6f'%v for v in nominal[k])
                + ' ' + ' '.join('0.0' for _ in range(nu)) + '\n')
print('저장: /tmp/getup_traj.txt (C++ 추종용, MPPI refined %d프레임)' % T)
