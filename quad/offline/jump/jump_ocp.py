#!/usr/bin/env python3
# J1 — 17-DoF 점프 OCP (crocoddyl FDDP 다상). RPET_JUMP_MPC.md §2.
#   push(4접촉, CoM vz↑) → flight(무접촉=자유동역학, base 자세비용 → 최적화가 다리로 각운동량 관리) → land(4접촉).
#   ★flight가 핵심: 접촉 제거 = base underactuated, 자세비용을 다리모션으로만 만족 → 각운동량 인지 궤적.
#   허리(FB_waist) lock=16 leg DOF. 출력=/tmp/jump_ocp.npz (q,dq,tau,sched) → wbic_jump 추종용.
import os, numpy as np, pinocchio as pin, crocoddyl
np.set_printoptions(precision=3, suppress=True)
G=9.81
VX=float(os.environ.get('JUMP_VX','0.0'))   # ★전방 이륙속도(m/s). 0=수직 제자리 점프. base 위치는 비용0이라 vx만으로 탄도 전진
HERE=os.path.dirname(os.path.abspath(__file__))
QUAD=os.path.normpath(os.path.join(HERE,'..','..'))   # offline/jump/ → quad/
URDF=os.path.join(QUAD,'..','02_Leg_UFDF_260703_2','urdf','02_Leg_UFDF_260703_3.urdf')
PKG=os.path.join(QUAD,'..')   # simulation/ = package://02_Leg_UFDF_260703_2/ 탐색 루트

full=pin.RobotWrapper.BuildFromURDF(URDF,[PKG],pin.JointModelFreeFlyer())
# 허리 lock → reduced
wj=full.model.getJointId('FB_waist_joint')
q_full=pin.neutral(full.model)
rm,[rd]=pin.buildReducedModel(full.model,[full.geometryModel if False else full.collision_model],[wj],q_full) if False else (None,[None])
model=pin.buildReducedModel(full.model,[wj],q_full)
data=model.createData()
nq,nv=model.nq,model.nv
FEET=['FL_foot_contact_link','FR_foot_contact_link','HL_foot_contact_link','HR_foot_contact_link']
fids=[model.getFrameId(f) for f in FEET]
print(f"[OCP] reduced nq={nq} nv={nv} 다리DOF={nv-6} feet={fids}")

# ── crouch/stand q0: 검증된 MuJoCo crouch 자세 → pinocchio 매핑 ──
import mujoco
mjm=mujoco.MjModel.from_xml_path(os.path.join(QUAD,'mjcf','quad_real_17dof_waist_sphere.mjcf')); mjd=mujoco.MjData(mjm)
mfgid=[mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_GEOM,f+'_sphere') for f in ['HL','HR','FL','FR']]
mfr=[mjm.geom_size[g][0] for g in mfgid]
def mj_crouch(base_z):
    if mjm.nkey>0: mujoco.mj_resetDataKeyframe(mjm,mjd,0)
    else: mjd.qpos[:]=0; mjd.qpos[3]=1
    mjd.qpos[2]=0.55; mujoco.mj_forward(mjm,mjd)
    fxy=[mjd.geom_xpos[g][:2].copy() for g in mfgid]; mjd.qpos[2]=base_z
    for _ in range(300):
        mujoco.mj_kinematics(mjm,mjd)
        for i,L in enumerate(['HL','HR','FL','FR']):
            p=mjd.geom_xpos[mfgid[i]].copy(); p[2]-=mfr[i]
            e=np.array([fxy[i][0],fxy[i][1],0.0])-p
            jp=np.zeros((3,mjm.nv)); mujoco.mj_jac(mjm,mjd,jp,None,mjd.geom_xpos[mfgid[i]],mjm.geom_bodyid[mfgid[i]])
            cols=[mjm.jnt_dofadr[mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_JOINT,f'{L}_{jn}_joint')] for jn in ['hip','thigh','calf','foot']]
            Jl=jp[:,cols]; dq=0.5*Jl.T@np.linalg.solve(Jl@Jl.T+1e-4*np.eye(3),e)
            qa=[mjm.jnt_qposadr[mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_JOINT,f'{L}_{jn}_joint')] for jn in ['hip','thigh','calf','foot']]
            for k,a in enumerate(qa): mjd.qpos[a]+=dq[k]
    mujoco.mj_forward(mjm,mjd)
    return mjd.qpos.copy()
def mj2pin(qpos):
    q=pin.neutral(model)
    q[0:3]=qpos[0:3]; q[3:7]=[qpos[4],qpos[5],qpos[6],qpos[3]]   # wxyz→xyzw
    for jid in range(1,model.njoints):
        nm=model.names[jid]; mjid=mujoco.mj_name2id(mjm,mujoco.mjtObj.mjOBJ_JOINT,nm)
        if mjid>=0: q[model.idx_qs[jid]]=qpos[mjm.jnt_qposadr[mjid]]
    return q
q_crouch=mj2pin(mj_crouch(0.30))
q_stand =mj2pin(mj_crouch(0.50))
x0=np.concatenate([q_crouch, np.zeros(nv)])
print(f"[OCP] crouch base_z={q_crouch[2]:.3f} stand base_z={q_stand[2]:.3f}")

state=crocoddyl.StateMultibody(model)
actu=crocoddyl.ActuationModelFloatingBase(state)
nu=actu.nu
# 토크한계(다리 peak, foot 8:1=96)
tau_lim=np.zeros(nu)
order=[]  # actuation nu 순서 = model joint 순서(base 제외)
for jid in range(1,model.njoints):
    nm=model.names[jid]
    for k,lim in [('hip',84),('thigh',84),('calf',126),('foot',96)]:
        if k in nm:
            for _ in range(model.joints[jid].nv): tau_lim[len(order)]=lim; order.append(nm)
tau_lim=np.where(tau_lim>0,tau_lim,100.0)

def make_contacts(active):
    cm=crocoddyl.ContactModelMultiple(state,nu)
    if active:
        for f,fid in zip(FEET,fids):
            c=crocoddyl.ContactModel3D(state,fid,np.zeros(3),pin.LOCAL_WORLD_ALIGNED,nu,np.array([0.,50.]))
            cm.addContact(f,c)
    return cm

def costs(vz_tar=None, vx_tar=0.0, up=True, xreg_to=None, wpush=0.0, term=False):
    cs=crocoddyl.CostModelSum(state,nu)
    xref=np.concatenate([xreg_to if xreg_to is not None else q_stand, np.zeros(nv)])
    wx=np.array([0]*3+[300,300,300]+[1.]*(nv-6) + [1.]*3+[10,10,10]+[0.1]*(nv-6))  # base 자세 강조
    cs.addCost('xreg',crocoddyl.CostModelResidual(state,
        crocoddyl.ActivationModelWeightedQuad(wx**2),
        crocoddyl.ResidualModelState(state,xref,nu)), 2.0 if term else 0.2)
    cs.addCost('ureg',crocoddyl.CostModelResidual(state,crocoddyl.ResidualModelControl(state,nu)),1e-3)
    tb=crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(-tau_lim,tau_lim))
    cs.addCost('taulim',crocoddyl.CostModelResidual(state,tb,crocoddyl.ResidualModelControl(state,nu)),1.0)
    if vz_tar is not None:   # ★이륙속도 유도(push): base 선속도 vz(+전방 vx) 강가중
        vtar=np.zeros(nv); vtar[2]=vz_tar; vtar[0]=vx_tar
        wv=np.zeros(2*nv); wv[nv+2]=1.0     # state tangent(2nv) 중 base linear vz
        if vx_tar!=0.0: wv[nv+0]=1.0        # base linear vx(전방)
        cs.addCost('pushv',crocoddyl.CostModelResidual(state,
            crocoddyl.ActivationModelWeightedQuad(wv**2),
            crocoddyl.ResidualModelState(state,np.concatenate([q_stand,vtar]),nu)), wpush)
    return cs

def run_model(dt, contact, **kw):
    cm=make_contacts(contact); cs=costs(**kw)
    dam=crocoddyl.DifferentialActionModelContactFwdDynamics(state,actu,cm,cs,0.,True)
    return crocoddyl.IntegratedActionModelEuler(dam,dt)

dt=0.01
vz_tk=np.sqrt(2*G*0.15)   # apex 0.15m 목표
T_fly=2*vz_tk/G; D=VX*T_fly    # ★전방 점프 비행시간·전진거리(탄도)
n_push, n_land = 22, 40
n_fly=max(6,int(2*vz_tk/G/dt))
models=[]; sched=[]
for _ in range(n_push): models.append(run_model(dt,True, vz_tar=vz_tk, vx_tar=VX, wpush=4.0)); sched.append('push')
for _ in range(n_fly):  models.append(run_model(dt,False, xreg_to=q_crouch)); sched.append('flight')  # tuck+자세
for _ in range(n_land): models.append(run_model(dt,True, xreg_to=q_stand)); sched.append('land')
terminal=run_model(dt,True, xreg_to=q_stand, term=True)
print(f"[OCP] 스케줄 push={n_push} flight={n_fly} land={n_land} (T={(n_push+n_fly+n_land)*dt:.2f}s)")

problem=crocoddyl.ShootingProblem(x0,models,terminal)
solver=crocoddyl.SolverFDDP(problem)
# ★점프형 warm-start(문서 필수): push=crouch→stand+vz↑, flight=탄도, land=stand
xs=[];
for k in range(n_push):
    a=(k+1)/n_push; q=(1-a)*q_crouch+a*q_stand; v=np.zeros(nv); v[2]=a*vz_tk; v[0]=a*VX
    xs.append(np.concatenate([q,v]))
for k in range(n_fly):
    tt=k*dt; q=q_crouch.copy(); q[2]=q_stand[2]+vz_tk*tt-0.5*G*tt*tt; q[0]=VX*tt
    v=np.zeros(nv); v[2]=vz_tk-G*tt; v[0]=VX
    xs.append(np.concatenate([q,v]))
qsf=q_stand.copy(); qsf[0]=D    # 착지 후 전방 위치(base 위치비용=0이라 warm-start만)
for k in range(n_land+1): xs.append(np.concatenate([qsf,np.zeros(nv)]))
xs=[x0]+xs[:len(models)]                       # 길이 N+1
us=problem.quasiStatic(xs[:-1])
print("[OCP] FDDP 풀이…")
ok=solver.solve(xs,us,200,False,1e-4)
xs=np.array(solver.xs); us=np.array(solver.us)
base_z=xs[:,2]; print(f"[OCP] 수렴={ok} iter={solver.iter} cost={solver.cost:.2f}")
print(f"[OCP] base_z: 시작 {base_z[0]:.3f} → peak {base_z.max():.3f} (apex {base_z.max()-q_crouch[2]:.3f}m) → 끝 {base_z[-1]:.3f}")
# 착지 tilt
def tilt(x):
    q=x[:nq]; pin.forwardKinematics(model,data,q); R=pin.Quaternion(q[6],q[3],q[4],q[5]).matrix()
    return np.degrees(np.arccos(np.clip(R[2,2],-1,1)))
print(f"[OCP] tilt: peak중 {tilt(xs[n_push+n_fly//2]):.1f}° 착지 {tilt(xs[n_push+n_fly]):.1f}°")
np.savez('/tmp/jump_ocp.npz', xs=xs, us=us, sched=np.array(sched), dt=dt, nq=nq, nv=nv)
print("[OCP] 저장 /tmp/jump_ocp.npz")
