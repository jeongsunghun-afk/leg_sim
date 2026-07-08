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

# ── 궤적 스케줄: G(gather 앞으로 모으기) → A1(HL착지) → A2(HR착지) → B(일어서기) ──
#   ★사용자 통찰: CoM이 뒤(엉덩이)에 있어 바로 일어서면 뒤로 넘어감 → 먼저 base를 앞으로 전진+낮춤+앞숙임해
#     앞/뒤다리 접으며 CoM을 발 지지면 위로 가져온 뒤(gather) 일어선다(사람이 상체 숙였다 서듯).
NG  = int(os.environ.get('NG', 45))
NA1 = int(os.environ.get('NA1', 35))
NA2 = int(os.environ.get('NA2', 35))
NB  = int(os.environ.get('NB', 90))
dt = 0.01
N = NG + NA1 + NA2 + NB
def lerp(a, b, t): return a + (b - a) * np.clip(t, 0, 1)

quat_sit = qs[3:7].copy(); quat_id = np.array([1., 0, 0, 0])
GX = float(os.environ.get('GATHER_X', base_stand[0]))   # gather 목표 base x(CoM 전진)
GZ = float(os.environ.get('GATHER_Z', 0.20))            # gather 낮춤 높이
LEAN = float(os.environ.get('LEAN', 0.10))              # gather 앞숙임(rad, nose-down)→CoM 전진
quat_lean = np.array([np.cos(LEAN/2), 0., np.sin(LEAN/2), 0.])   # nose-down(+y)

def base_prof(k):   # (bx, bz, quat)
    if k < NG:                        # gather: 전진+낮춤+앞숙임
        t = (k+1)/NG
        return lerp(base_sit[0],GX,t), lerp(base_sit[2],GZ,t), slerp(quat_sit,quat_lean,t)
    elif k < NG+NA1+NA2:              # 뒷발 순차착지(전진 자세 유지)
        return GX, GZ, quat_lean
    else:                             # 일어서기: 상승+레벨
        t = (k-NG-NA1-NA2)/NB
        return lerp(GX,base_stand[0],t), lerp(GZ,base_stand[2],t), slerp(quat_lean,quat_id,t)

traj_q = []; sched = []
for k in range(N):
    ph = 'A1' if k < NG+NA1 else ('A2' if k < NG+NA1+NA2 else 'B')
    bx, bz, oq = base_prof(k)
    set_base([bx, 0, bz], oq)
    tg = {}
    for L in LEGS:
        if L in ('FL', 'FR'):
            tg[L] = foot_stand[L].copy()                                    # 앞발 고정(gather서 앞다리 접힘)
        elif L == 'HL':
            s = 0.0 if k < NG else (min(1.0,(k-NG)/NA1) if k < NG+NA1 else 1.0)
            tg[L] = lerp(foot_sit[L], foot_stand[L], s)                     # HL: A1서 착지
        else:  # HR
            s = 0.0 if k < NG+NA1 else (min(1.0,(k-NG-NA1)/NA2) if k < NG+NA1+NA2 else 1.0)
            tg[L] = lerp(foot_sit[L], foot_stand[L], s)                     # HR: A2서 착지
    for L in LEGS:
        d.qpos[qadr[L][3]] = lerp(foot_ank_sit[L], foot_ank_st[L], (k+1)/N)
    ik_feet(tg)
    traj_q.append(d.qpos[7:7+m.nu].copy())
    sched.append(ph)

traj_q = np.array(traj_q)
traj_dq = np.gradient(traj_q, dt, axis=0)
com = np.zeros((N, 3)); bz_arr = np.zeros(N); full_qpos = np.zeros((N, m.nq))
for k in range(N):
    bx, bz, oq = base_prof(k)
    set_base([bx,0,bz],oq); d.qpos[7:7+m.nu]=traj_q[k]; mujoco.mj_forward(m,d)
    com[k]=d.subtree_com[0].copy(); bz_arr[k]=d.qpos[2]; full_qpos[k]=d.qpos.copy()
comv = np.gradient(com, dt, axis=0); acom = np.gradient(comv, dt, axis=0)
mj_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
np.savez('/tmp/getup_stand.npz', q=traj_q, dq=traj_dq, tau=np.zeros((N-1, m.nu)),
         base_z=bz_arr, sched=np.array(sched, dtype=object), dt=dt,
         com_ref=com, comv_ref=comv, acom_ref=acom, full_qpos=full_qpos, mj_names=np.array(mj_names))
# ★C++ 뷰어 추종용 텍스트 내보내기: N dt \n (phase 0=A1/1=A2/2=B, q[17]) × N
_pc = {'A1':0,'A2':1,'B':2}
with open('/tmp/getup_traj.txt','w') as f:
    f.write('%d %g\n' % (N, dt))
    for k in range(N):   # phase, q[17], dq[17](속도 피드포워드)
        f.write('%d ' % _pc[sched[k]] + ' '.join('%.6f'%v for v in traj_q[k])
                + ' ' + ' '.join('%.6f'%v for v in traj_dq[k]) + '\n')
print('저장: /tmp/getup_traj.txt (C++ 추종용, %d프레임)' % N)
d.qpos[:]=qst; mujoco.mj_forward(m,d); fxs=[foot_pos(L)[0] for L in LEGS]
print('저장: /tmp/getup_stand.npz (%d스텝 G%d/A1%d/A2%d/B%d, base %.3f→%.3f)'
      % (N, NG, NA1, NA2, NB, bz_arr[0], bz_arr[-1]))
print('  ★CoM x: sit%+.3f → gather%+.3f → stand%+.3f  (지지발 x %.3f~%.3f, CoM이 이 안이어야 안 넘어짐)'
      % (com[0,0], com[NG-1,0], com[-1,0], min(fxs), max(fxs)))
