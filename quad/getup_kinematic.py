"""기립 기구학 준정적 궤적 생성 (OCP 대안). 앉기→서기.
   base pose(sit→stand 보간) 고정 + 발 목표(앞발고정·뒷발 순차착지)를 Jacobian IK로 풀어 관절궤적.
   발을 지면에 정확히 두므로 관통 원천차단. CoM은 지지폴리곤 안(정적안정). WBC(getup_track.py)로 추종.
실행: /home/jsh/miniforge3/envs/proxddp/bin/python getup_kinematic.py
"""
import os, numpy as np, mujoco
_HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.join(_HERE, 'quad_real_17dof_waist_sphere.mjcf')
LEGS = ['HL', 'HR', 'FL', 'FR']   # MuJoCo legqp 순서
JT = ['hip', 'thigh', 'calf', 'foot']

m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
fgid = {L: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, '%s_sphere' % L) for L in LEGS}
frad = {L: m.geom_size[fgid[L]][0] for L in LEGS}
qadr = {L: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, '%s_%s_joint' % (L, jt))] for jt in JT] for L in LEGS}
vadr = {L: [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, '%s_%s_joint' % (L, jt))] for jt in JT] for L in LEGS}

def foot_pos(L):   # sphere 접촉점(바닥)
    return d.geom_xpos[fgid[L]] - np.array([0, 0, frad[L]])

def set_base(pos, quat):
    d.qpos[0:3] = pos; d.qpos[3:7] = quat

def slerp(q0, q1, t):   # wxyz
    dot = np.dot(q0, q1)
    if dot < 0: q1 = -q1; dot = -dot
    if dot > 0.9995: q = q0 + t*(q1-q0)
    else:
        th = np.arccos(np.clip(dot,-1,1)); q = (np.sin((1-t)*th)*q0 + np.sin(t*th)*q1)/np.sin(th)
    return q/np.linalg.norm(q)

def ik_feet(targets, iters=200):
    """현 base 고정, 각 다리 3관절(hip/thigh/calf)로 발을 target(3D)에 — foot관절은 고정값 유지."""
    for _ in range(iters):
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        for L in LEGS:
            e = targets[L] - foot_pos(L)
            if np.linalg.norm(e) < 1e-4:
                continue
            J = np.zeros((3, m.nv)); mujoco.mj_jac(m, d, J, None, d.geom_xpos[fgid[L]], m.geom_bodyid[fgid[L]])
            Jl = J[:, vadr[L][:3]]                     # hip/thigh/calf
            dq = 0.5 * Jl.T @ np.linalg.solve(Jl @ Jl.T + 1e-4*np.eye(3), e)
            for c in range(3):
                d.qpos[qadr[L][c]] += dq[c]

# ── 자세 정의 ──
qs = np.array([float(x) for x in open('/tmp/q_sit.txt').read().split()])     # 정착 sit
qst = np.array([float(x) for x in open('/tmp/q_stand.txt').read().split()])  # 정착 stand
# stand 발위치(전 발 착지목표) + sit 발위치
d.qpos[:] = qst; mujoco.mj_forward(m, d)
foot_stand = {L: foot_pos(L).copy() for L in LEGS}
base_stand = d.qpos[0:3].copy()
d.qpos[:] = qs; mujoco.mj_forward(m, d)
foot_sit = {L: foot_pos(L).copy() for L in LEGS}
base_sit = d.qpos[0:3].copy()
# foot 관절(발목) 목표: sit→stand 보간(발목은 IK 안하고 스케줄로)
foot_ank_sit = {L: qs[qadr[L][3]] for L in LEGS}
foot_ank_st = {L: qst[qadr[L][3]] for L in LEGS}
print('sit base z=%.3f 뒷발z=%.3f / stand base z=%.3f' % (base_sit[2], foot_sit['HL'][2], base_stand[2]))

# ── 궤적 스케줄 ──
NA1, NA2, NB, dt = 40, 40, 90, 0.01
N = NA1 + NA2 + NB; NA = NA1 + NA2
def lerp(a, b, t): return a + (b - a) * np.clip(t, 0, 1)

quat_sit = qs[3:7].copy(); quat_id = np.array([1., 0, 0, 0])
def base_prof(k):   # (bx, bz, quat) 프로파일 — 실제 sit자세→레벨 slerp
    if k < NA1:      bz = lerp(base_sit[2], 0.16, k/NA1)
    elif k < NA1+NA2: bz = lerp(0.16, 0.20, (k-NA1)/NA2)
    else:            bz = lerp(0.20, base_stand[2], (k-NA1-NA2)/NB)
    bx = lerp(base_sit[0], base_stand[0], (k+1)/N)
    oq = slerp(quat_sit, quat_id, min(1.0, (k+1)/NA))   # 자세는 Phase A 동안 레벨링
    return bx, bz, oq

traj_q = []; sched = []
for k in range(N):
    ph = 'A1' if k < NA1 else ('A2' if k < NA1+NA2 else 'B')
    tt = (k/NA1) if ph == 'A1' else ((k-NA1)/NA2 if ph == 'A2' else (k-NA1-NA2)/NB)
    bx, bz, oq = base_prof(k)
    set_base([bx, 0, bz], oq)
    # 발 목표: 앞발=stand위치 고정. 뒷발=순차 착지(sit위치→stand위치, z는 위→0)
    tg = {}
    for L in LEGS:
        if L in ('FL', 'FR'):
            tg[L] = foot_stand[L].copy()
        elif L == 'HL':
            s = tt if ph == 'A1' else 1.0
            tg[L] = lerp(foot_sit[L], foot_stand[L], s)
        else:  # HR
            s = tt if ph == 'A2' else (0.0 if ph == 'A1' else 1.0)
            tg[L] = lerp(foot_sit[L], foot_stand[L], s)
    # 발목 관절 보간(IK 전에 세팅)
    for L in LEGS:
        d.qpos[qadr[L][3]] = lerp(foot_ank_sit[L], foot_ank_st[L], (k+1)/N)
    ik_feet(tg)
    traj_q.append(d.qpos[7:7+m.nu].copy())
    sched.append(ph)

traj_q = np.array(traj_q)
traj_dq = np.gradient(traj_q, dt, axis=0)
# CoM 기준(WBC용) + base_z
com = np.zeros((N, 3)); bz_arr = np.zeros(N); full_qpos = np.zeros((N, m.nq))
for k in range(N):
    # base 재세팅(위 프로파일과 동일) + 관절
    pass
# com은 각 스텝의 full state로 계산: base 프로파일 재현
for k in range(N):
    bx, bz, oq = base_prof(k)
    set_base([bx,0,bz],oq); d.qpos[7:7+m.nu]=traj_q[k]; mujoco.mj_forward(m,d)
    com[k]=d.subtree_com[0].copy(); bz_arr[k]=d.qpos[2]; full_qpos[k]=d.qpos.copy()
comv = np.gradient(com, dt, axis=0); acom = np.gradient(comv, dt, axis=0)
mj_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
np.savez('/tmp/getup_stand.npz', q=traj_q, dq=traj_dq, tau=np.zeros((N-1, m.nu)),
         base_z=bz_arr, sched=np.array(sched, dtype=object), dt=dt,
         com_ref=com, comv_ref=comv, acom_ref=acom, full_qpos=full_qpos, mj_names=np.array(mj_names))
print('저장: /tmp/getup_stand.npz (기구학 %d스텝, base %.3f→%.3f, CoM %.3f→%.3f)'
      % (N, bz_arr[0], bz_arr[-1], com[0,2], com[-1,2]))
