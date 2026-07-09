#!/usr/bin/env python3
# J2 — OCP 점프궤적 추종(배포). RPET_JUMP_MPC.md §3.
#   /tmp/jump_ocp.npz (pinocchio reduced xs,us) → MuJoCo 17dof 매핑 → τ_ff + PD 추종 → 실제 점프 검증.
#   상: push/flight=τ_ff+관절PD / touchdown 후=저-PD 흡수 → 홀드. base_z peak·tilt·falls 측정.
import os, numpy as np, mujoco, pinocchio as pin
HERE=os.path.dirname(os.path.abspath(__file__))
G=9.81
Z=np.load('/tmp/jump_ocp.npz', allow_pickle=True)
xs, us, sched, dt = Z['xs'], Z['us'], Z['sched'], float(Z['dt']); nq_p, nv_p = int(Z['nq']), int(Z['nv'])
N=len(us)

# pinocchio reduced 모델(허리 lock) — 매핑용 조인트 이름/인덱스
URDF=os.path.join(HERE,'..','02_Leg_UFDF_260703_2','urdf','02_Leg_UFDF_260703_3.urdf')
full=pin.RobotWrapper.BuildFromURDF(URDF,[os.path.join(HERE,'..')],pin.JointModelFreeFlyer())
pm=pin.buildReducedModel(full.model,[full.model.getJointId('FB_waist_joint')],pin.neutral(full.model))

# MuJoCo 17dof
m=mujoco.MjModel.from_xml_path(os.path.join(HERE,'quad_real_17dof_waist_sphere.mjcf')); d=mujoco.MjData(m)
fgid=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f+'_sphere') for f in ['HL','HR','FL','FR']]
GEARF=0.5714  # foot 8:1(배포)
# 관절명→ pin q/v 인덱스, MuJoCo qpos/qvel/ctrl 인덱스
LEGJ=[f'{L}_{jn}_joint' for L in ['FL','FR','HL','HR'] for jn in ['hip','thigh','calf','foot']]
pin_qi={}; pin_vi={}; pin_ui={}
for jid in range(1,pm.njoints):
    nm=pm.names[jid]
    if pm.joints[jid].nv!=1: continue           # freeflyer(base, nv=6) 제외 — u는 다리 16개만
    pin_qi[nm]=pm.idx_qs[jid]; pin_vi[nm]=pm.idx_vs[jid]; pin_ui[nm]=pm.idx_vs[jid]-6
mj_qa={jn:m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,jn)] for jn in LEGJ}
mj_va={jn:m.jnt_dofadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,jn)] for jn in LEGJ}
mj_ai={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a):a for a in range(m.nu)}

def node_ref(k):   # k번째 노드 → MuJoCo q_ref[nu], dq_ref[nu], tau_ff[nu]
    x=xs[k]; u=us[min(k,N-1)]; qp=x[:nq_p]; vp=x[nq_p:]
    qref=np.zeros(m.nu); dqref=np.zeros(m.nu); tff=np.zeros(m.nu)
    for jn in LEGJ:
        act=jn.replace('_joint',''); ai=mj_ai.get(act)
        if ai is None: continue
        qref[ai]=qp[pin_qi[jn]]; dqref[ai]=vp[pin_vi[jn]]; tff[ai]=u[pin_ui[jn]]
    return qref,dqref,tff

# ── C++ 통합용 export: MuJoCo 순서 궤적 (/tmp/jump_traj.txt) — C++ 점프모드가 thrust/flight 재생 ──
with open('/tmp/jump_traj.txt','w') as f:
    f.write(f"{N} {dt}\n")
    for k in range(N):
        qr,dqr,tf=node_ref(k)
        ph=0 if (k<len(sched) and sched[k]=='push') else (1 if (k<len(sched) and sched[k]=='flight') else 2)
        f.write(f"{ph} "+" ".join(f"{v:.6f}" for v in qr)+" "
                +" ".join(f"{v:.6f}" for v in dqr)+" "+" ".join(f"{v:.6f}" for v in tf)+"\n")
print("[TRACK] export /tmp/jump_traj.txt (C++ 점프모드 통합용, MuJoCo 순서)")

# ── MuJoCo에 crouch(노드0) teleport ──
q0=xs[0][:nq_p]
if m.nkey>0: mujoco.mj_resetDataKeyframe(m,d,0)
d.qpos[0:3]=q0[0:3]; d.qpos[3:7]=[q0[6],q0[3],q0[4],q0[5]]   # xyzw→wxyz
for jn in LEGJ: d.qpos[mj_qa[jn]]=q0[pin_qi[jn]]
d.qvel[:]=0; mujoco.mj_forward(m,d)
zc=d.qpos[2]; print(f"[TRACK] crouch teleport base_z={zc:.3f}  노드수 N={N} dt={dt}")

KP=float(os.environ.get('JT_KP','120')); KD=float(os.environ.get('JT_KD','4'))
tau_peak={'hip':84,'thigh':84,'calf':126,'foot':96}
def peak(ai):
    nm=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,ai)
    for k,v in tau_peak.items():
        if k in nm: return v
    return 100
PK=np.array([peak(a) for a in range(m.nu)])

substep=int(round(dt/m.opt.timestep))   # OCP dt(10ms) : MuJoCo dt
zpeak=zc; tiltmax=0; landed=False; falls=0
def tiltdeg():
    R=np.zeros(9); mujoco.mju_quat2Mat(R,d.qpos[3:7]); return np.degrees(np.arccos(np.clip(R[8],-1,1)))
for k in range(N):
    qref,dqref,tff=node_ref(k)
    ph=sched[k] if k<len(sched) else 'land'
    for _ in range(max(1,substep)):
        for ai in range(m.nu):
            if landed and ph=='land':   # 착지후=저-PD 흡수/홀드
                tau=d.qfrc_bias[6+ai]+45*(qref[ai]-d.qpos[7+ai])-6*d.qvel[6+ai]
            else:                        # push/flight=τ_ff + PD
                tau=tff[ai]+KP*(qref[ai]-d.qpos[7+ai])+KD*(dqref[ai]-d.qvel[6+ai])
            d.ctrl[ai]=np.clip(tau,-PK[ai],PK[ai])
        mujoco.mj_step(m,d)
        zpeak=max(zpeak,d.qpos[2]); tiltmax=max(tiltmax,tiltdeg())
        if tiltdeg()>60 or d.qpos[2]<0.15: falls+=1
        # touchdown: flight 후 발접촉
        if ph in ('flight','land') and not landed:
            nc=sum(any(d.contact[ci].geom1==g or d.contact[ci].geom2==g for g in fgid) for ci in range(d.ncon))
            if nc>=2 and d.qpos[2]<zpeak-0.02: landed=True
# 착지후 홀드 잠깐
for _ in range(400):
    for ai in range(m.nu):
        d.ctrl[ai]=np.clip(d.qfrc_bias[6+ai]+80*(node_ref(N-1)[0][ai]-d.qpos[7+ai])-6*d.qvel[6+ai],-PK[ai],PK[ai])
    mujoco.mj_step(m,d); tiltmax=max(tiltmax,tiltdeg())
    if tiltdeg()>60 or d.qpos[2]<0.15: falls+=1
print(f"[TRACK] base_z peak={zpeak:.3f} (clearance {zpeak-zc:.3f}m) 최종={d.qpos[2]:.3f}")
print(f"[TRACK] tilt_max={tiltmax:.1f}°  falls={falls}  착지감지={landed}")
print(f"[TRACK] 결과: {'✅ 실제로 뛰고 착지' if (zpeak-zc>0.08 and d.qpos[2]>0.30 and tiltmax<40) else '❌ 미흡(높이/착지/tilt 확인)'}")
