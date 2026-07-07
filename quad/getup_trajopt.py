"""offline 기립 궤적 최적화 (sit→stand). crocoddyl FullDynamics OCP.
   17-DOF(허리fixed 16leg) 02_Leg_UFDF_260703. 발=sphere바닥 접촉프레임.
   핵심: sit_home·crouch_home 모두 발을 동일 nominal XY에 둠 → 발 고정 접촉으로 몸만 올림(발 재배치 불필요).
   출력: getup.npz (q/dq/tau/base_z/sched) — wbic_jump/open-loop 추종용(점프 포맷 호환).
실행: BASE_Z0=0.5234 /home/jsh/miniforge3/envs/proxddp/bin/python getup_trajopt.py
"""
import os, re, sys, time
import numpy as np
import pinocchio as pin
import crocoddyl
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_SRC = os.path.join(_HERE, '../02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf')
MJCF = os.path.join(_HERE, 'quad_real_17dof_waist_sphere.mjcf')
LEGS = ['FL', 'FR', 'HL', 'HR']          # pinocchio 관절순서
JT = ['hip', 'thigh', 'calf', 'foot']
MU = 0.7
PEAK = {'hip': 84.0, 'thigh': 84.0, 'calf': 126.0, 'foot': 96.0}   # 우리 모터셋(발목 8:1 재기어=96)

# ── 1) pinocchio 모델(허리 fixed, freeflyer) ──
def build_model():
    u = open(URDF_SRC).read()
    u = re.sub(r'package://[^/]+/meshes/', '', u)
    u = re.sub(r'<joint\b.*?</joint>',
               lambda m: m.group(0).replace('type="revolute"', 'type="fixed"')
               if 'FB_waist_joint' in m.group(0) else m.group(0), u, flags=re.S)
    open('/tmp/getup.urdf', 'w').write(u)
    model = pin.buildModelFromUrdf('/tmp/getup.urdf', pin.JointModelFreeFlyer())
    return model

# ── 2) MuJoCo로 발 sphere offset + sit/stand qpos 계산 ──
def mj_states():
    m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
    fgid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, '%s_sphere' % L) for L in LEGS]
    # sphere 접촉점 오프셋(contact_link 로컬): geom_pos - [0,0,r]
    sole = {}
    for i, L in enumerate(LEGS):
        r = m.geom_size[fgid[i]][0]
        sole[L] = m.geom_pos[fgid[i]].copy() - np.array([0, 0, r])
    return m, d, sole, fgid

# 관절각(MuJoCo 이름기준) → mujoco qpos 세팅 + 발 지면(min z=0) 되도록 base 조정
def set_pose(m, d, joints, base_pitch, legs_qadr):
    d.qpos[:] = 0; d.qpos[3] = np.cos(base_pitch/2); d.qpos[5] = -np.sin(base_pitch/2)
    d.qpos[2] = 0.6
    for L in LEGS:
        for k, jt in enumerate(JT):
            d.qpos[legs_qadr[L][k]] = joints[L][k]
    mujoco.mj_forward(m, d)
    # 발 최저 z를 0으로: base_z 조정
    fz = min(_foot_z_mj(m, d, L) for L in LEGS)
    d.qpos[2] -= fz
    mujoco.mj_forward(m, d)
    return d.qpos.copy()

def _foot_z_mj(m, d, L):
    fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, '%s_sphere' % L)
    r = m.geom_size[fid][0]
    return d.geom_xpos[fid][2] - r

def mj_qadr(m):
    return {L: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, '%s_%s_joint' % (L, jt))]
               for jt in JT] for L in LEGS}

# ── 3) MuJoCo qpos → pinocchio q (관절 이름매핑) ──
def mj2pin_q(model, m, mqpos, legs_qadr):
    q = np.zeros(model.nq)
    q[0:3] = mqpos[0:3]
    w, x, y, z = mqpos[3:7]; q[3:7] = [x, y, z, w]      # wxyz→xyzw
    for L in LEGS:
        for k, jt in enumerate(JT):
            jn = '%s_%s_joint' % (L, jt)
            q[model.joints[model.getJointId(jn)].idx_q] = mqpos[legs_qadr[L][k]]
    return q

def add_foot_frames(model, sole):
    for L in LEGS:
        fr = model.frames[model.getFrameId(L + '_foot_contact_link')]
        pl = fr.placement * pin.SE3(np.eye(3), sole[L])
        model.addFrame(pin.Frame(L + '_foot', fr.parentJoint, fr.parentFrame, pl, pin.FrameType.OP_FRAME))

# ── 4) crocoddyl OCP ──
def contact_model(state, actuation, foot_fid, stance):
    cm = crocoddyl.ContactModelMultiple(state, actuation.nu)
    for L in stance:
        c = crocoddyl.ContactModel3D(state, foot_fid[L], np.zeros(3), pin.LOCAL_WORLD_ALIGNED, actuation.nu, np.array([0., 50.]))
        cm.addContact('c_' + L, c)
    return cm

def action(state, actuation, foot_fid, stance, x_ref, tau_lim, dt, w_term=False, swing=None):
    # stance=접촉발 리스트, swing={L: 목표 3D위치}=자유발(내려 심기 목표)
    cm = contact_model(state, actuation, foot_fid, stance)
    cost = crocoddyl.CostModelSum(state, actuation.nu)
    nu = actuation.nu
    w_base = [1.0, 1.0, 50.0 if w_term else 25.0, 80.0, 80.0, 20.0]   # base pos(z강조=가라앉음방지)+ori
    w_state = np.array(w_base + [2.0]*nu + [1.0]*6 + [0.1]*nu)
    cost.addCost('xreg', crocoddyl.CostModelResidual(
        state, crocoddyl.ActivationModelWeightedQuad(w_state**2),
        crocoddyl.ResidualModelState(state, x_ref, nu)), 1.0 if not w_term else 200.0)
    cost.addCost('ureg', crocoddyl.CostModelResidual(
        state, crocoddyl.ResidualModelControl(state, nu)), 1e-3)
    for L in stance:
        fc = crocoddyl.FrictionCone(np.eye(3), MU, 4, False)
        fc_act = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(fc.lb, fc.ub))
        fc_res = crocoddyl.ResidualModelContactFrictionCone(state, foot_fid[L], fc, nu, True)
        cost.addCost('fc_' + L, crocoddyl.CostModelResidual(state, fc_act, fc_res), 2.0)
    # 모든 비접촉발에 z≥0 clearance(관통방지). 스윙발은 추가로 목표(착지점) 유도.
    zb = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(
        np.array([-1e3, -1e3, 0.0]), np.array([1e3, 1e3, 1e3])))
    for L in LEGS:
        if L in stance:
            continue
        tgt = (swing.get(L) if swing else None)
        ref = tgt if tgt is not None else np.zeros(3)   # clearance는 z만 봄
        res = crocoddyl.ResidualModelFrameTranslation(state, foot_fid[L], ref, nu)
        cost.addCost('cl_'+L, crocoddyl.CostModelResidual(state, zb, res), 400.0)
        if tgt is not None:   # 스윙발=착지점 유도
            res2 = crocoddyl.ResidualModelFrameTranslation(state, foot_fid[L], tgt, nu)
            cost.addCost('sw_'+L, crocoddyl.CostModelResidual(state, res2), 20.0)
    tau_act = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-tau_lim, tau_lim))
    cost.addCost('taulim', crocoddyl.CostModelResidual(
        state, tau_act, crocoddyl.ResidualModelControl(state, nu)), 8.0)
    diff = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actuation, cm, cost, 0.0, True)
    return crocoddyl.IntegratedActionModelEuler(diff, dt)

def main():
    model = build_model()
    m, d, sole, fgid = mj_states()
    add_foot_frames(model, sole)
    data = model.createData()
    state = crocoddyl.StateMultibody(model)
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    nu = actuation.nu
    foot_fid = {L: model.getFrameId(L + '_foot') for L in LEGS}
    tau_lim = np.array([next((PEAK[t] for t in PEAK if t in model.names[j]), 80.0)
                        for j in range(2, model.njoints)])
    print('nq=%d nv=%d nu=%d tau_lim=%s' % (model.nq, model.nv, nu, tau_lim[:4]))

    legs_qadr = mj_qadr(m)
    # 실제 정착 qpos 덤프 로드(trot_sim DUMP_QPOS). nq=24(waist 포함), 이름매핑으로 waist skip.
    q_stand_mj = np.array([float(x) for x in open('/tmp/q_stand.txt').read().split()])
    q_sit_mj = np.array([float(x) for x in open('/tmp/q_sit.txt').read().split()])
    x_stand = np.concatenate([mj2pin_q(model, m, q_stand_mj, legs_qadr), np.zeros(model.nv)])
    x_sit = np.concatenate([mj2pin_q(model, m, q_sit_mj, legs_qadr), np.zeros(model.nv)])
    print('sit  base z=%.3f pitch(quat)=%.2f' % (q_sit_mj[2], q_sit_mj[3]))
    print('stand base z=%.3f' % q_stand_mj[2])
    # 발이 stand/sit서 지면(z≈0)인지 확인
    for tag, q in [('sit', x_sit[:model.nq]), ('stand', x_stand[:model.nq])]:
        pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
        fz = {L: data.oMf[foot_fid[L]].translation[2] for L in LEGS}
        print('  %s 발z: %s' % (tag, {k: round(v, 3) for k, v in fz.items()}))

    # stand 발위치(뒷발 착지목표)
    pin.forwardKinematics(model, data, x_stand[:model.nq]); pin.updateFramePlacements(model, data)
    foot_tgt = {L: data.oMf[foot_fid[L]].translation.copy() for L in LEGS}
    # ★3단계 OCP(뒷발 순차착지로 항상 ≥3접촉=정적안정):
    #   A1) FL,FR 지지 + HL 내려심기  A2) FL,FR,HL 지지 + HR 내려심기  B) 4발 기립
    dt = 0.01; NA1 = 35; NA2 = 35; NB = 80
    front = ['FL', 'FR']
    N = NA1 + NA2 + NB; NA = NA1 + NA2
    z0 = x_sit[2]; zA = 0.17; zS = x_stand[2]
    def xref(k, bz):   # x_stand 관절 목표(다리 펴 심게) + base_z 프로파일
        x = x_stand.copy(); x[2] = bz; return x
    runs = []
    for k in range(NA1):   # A1: HL 내려심기 (FL,FR 지지)
        bz = z0 + (zA-z0)*0.5*k/NA1
        runs.append(action(state, actuation, foot_fid, front, xref(k, bz), tau_lim, dt, swing={'HL': foot_tgt['HL']}))
    for k in range(NA2):   # A2: HR 내려심기 (FL,FR,HL 지지=3접촉 안정)
        bz = z0 + (zA-z0)*(0.5+0.5*k/NA2)
        runs.append(action(state, actuation, foot_fid, front+['HL'], xref(NA1+k, bz), tau_lim, dt, swing={'HR': foot_tgt['HR']}))
    for k in range(NB):    # B: 4발 기립
        bz = zA + (zS-zA)*k/NB
        runs.append(action(state, actuation, foot_fid, LEGS, xref(NA+k, bz), tau_lim, dt))
    term = action(state, actuation, foot_fid, LEGS, x_stand, tau_lim, dt, w_term=True)
    problem = crocoddyl.ShootingProblem(x_sit, runs, term)
    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    solver.th_stop = 1e-5
    t0 = time.time()
    done = solver.solve([x_sit]*(N+1), [np.zeros(nu)]*N, 800, False, 1e-9)
    xs = np.array(solver.xs); us = np.array(solver.us)
    print('\n수렴=%s iter=%d %.0fms cost=%.3f' % (done, solver.iter, (time.time()-t0)*1e3, solver.cost))
    # base z 궤적
    bz = xs[:, 2]
    print('base z: %.3f → %.3f (목표 %.3f)  최종tilt=' % (bz[0], bz[-1], q_stand_mj[2]), end='')
    qf = xs[-1, :model.nq]; Rf = pin.Quaternion(qf[6], qf[3], qf[4], qf[5]).toRotationMatrix()
    print('%.1f°' % np.degrees(np.arccos(np.clip(Rf[2, 2], -1, 1))))
    # ── MuJoCo 순서로 변환 저장(추종용) ──
    # pin 관절 idx → MuJoCo 액추에이터(관절) 순서 매핑(이름기준)
    mj_jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    pin_of_mj = []   # 각 MuJoCo 액추에이터의 pin q/v/u 인덱스(없으면 -1=waist)
    for an in mj_jnames:
        jn = an + '_joint'
        if model.existJointName(jn):
            jid = model.getJointId(jn); pin_of_mj.append(model.joints[jid].idx_v - 6)  # u/v 순서(base 6 제외)
        else:
            pin_of_mj.append(-1)   # waist 등
    NT = len(xs)
    qmj = np.zeros((NT, m.nu)); dqmj = np.zeros((NT, m.nu)); taumj = np.zeros((NT-1, m.nu))
    for k in range(NT):
        for a, p in enumerate(pin_of_mj):
            if p >= 0:
                qmj[k, a] = xs[k, 7 + p]           # q: base 7 + joint
                dqmj[k, a] = xs[k, model.nq + 6 + p]
                if k < NT-1: taumj[k, a] = us[k, p]
    bz = xs[:, 2]
    sched = ['A1']*NA1 + ['A2']*NA2 + ['B']*(NB+1)   # A1=FL,FR / A2=FL,FR,HL / B=4발
    # ── CoM 기준(WBC 추종용): MuJoCo qpos에 궤적 세팅 → subtree_com ──
    full_qpos = np.zeros((NT, m.nq))
    com = np.zeros((NT, 3))
    for k in range(NT):
        full_qpos[k, 0:3] = xs[k, 0:3]
        qx, qy, qz, qw = xs[k, 3:7]; full_qpos[k, 3:7] = [qw, qx, qy, qz]   # xyzw→wxyz
        full_qpos[k, 7:7+m.nu] = qmj[k]
        d.qpos[:] = full_qpos[k]; d.qvel[:] = 0; mujoco.mj_forward(m, d)
        com[k] = d.subtree_com[0].copy()
    comv = np.gradient(com, dt, axis=0)
    acom = np.gradient(comv, dt, axis=0)
    np.savez('/tmp/getup_stand.npz', xs=xs, us=us, dt=dt,
             q=qmj, dq=dqmj, tau=taumj, base_z=bz, sched=np.array(sched, dtype=object),
             mj_names=np.array(mj_jnames), full_qpos=full_qpos,
             com_ref=com, comv_ref=comv, acom_ref=acom)
    print('저장: /tmp/getup_stand.npz (q/dq/tau/com_ref %d스텝, CoM %.3f→%.3f)' % (NT, com[0,2], com[-1,2]))

if __name__ == '__main__':
    os.environ.setdefault('BASE_Z0', '0.5234')
    main()
