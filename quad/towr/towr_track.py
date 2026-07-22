#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOWR Phase1 — 오프라인 TOWR 궤적을 MuJoCo에서 추종(크로싱 실증).
개루프 재생은 부양 base가 불안정 → SRBD 힘-기반 균형 추종(WBIC-lite):
  ▸ base pose/vel 피드백 → 보정 렌치 W_des(TOWR base 궤적 추종 + 중력보상)
  ▸ W_des를 지지발 GRF로 분배(grasp map 의사역) + 마찰콘 클립 → τ=Jᵀf
  ▸ 스윙발: TOWR 발궤적 IK 관절각 PD
= "오프라인 trajopt + 균형 추종" 패턴(점프/기립 계보).

실행: /home/jsh/simple-mpc/.pixi/envs/default/bin/python towr_track.py
  env TRAJ=traj_step.json MJCF=지형씬 VIEW=0
"""
import numpy as np, mujoco as mj, pinocchio as pin, json, os

TRAJ = os.environ.get('TRAJ','/home/jsh/문서/jsh/simulation/quad/towr/traj_step.json')
URDF = "/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf"
MJCF = os.environ.get('MJCF','/home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf')
FEET = ['FL','FR','HL','HR']
FRAME = {f: f+'_foot_contact_link' for f in FEET}
_R = 0.025
MASS = 38.016; G = 9.81; INER = np.diag([0.941,2.521,2.236])
MU = 0.6

pm = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
pdat = pm.createData()
fid = {f: pm.getFrameId(FRAME[f]) for f in FEET}
NQ, NV = pm.nq, pm.nv
_PIN2MJ = np.array([8,9,10,11,12, 13,14,15,16, 0,1,2,3, 4,5,6,7])

def leg_ik(q_init, base_SE3, foot_world):
    q = q_init.copy(); q[0:3]=base_SE3.translation; q[3:7]=pin.Quaternion(base_SE3.rotation).coeffs()
    for _ in range(60):
        pin.forwardKinematics(pm,pdat,q); pin.updateFramePlacements(pm,pdat)
        err=np.zeros(12); Js=np.zeros((12,NV))
        for i,f in enumerate(FEET):
            p=pdat.oMf[fid[f]].translation+pdat.oMf[fid[f]].rotation@np.array([0,0,-_R])
            err[3*i:3*i+3]=foot_world[f]-p
            Js[3*i:3*i+3]=pin.computeFrameJacobian(pm,pdat,q,fid[f],pin.LOCAL_WORLD_ALIGNED)[:3]
        if np.linalg.norm(err)<1e-5: break
        Jl=Js[:,6:]; dq=Jl.T@np.linalg.solve(Jl@Jl.T+1e-6*np.eye(12),err)
        qv=np.zeros(NV); qv[6:]=dq; q=pin.integrate(pm,q,0.5*qv)
    return q, np.linalg.norm(err)

def so3_err(R_d, R):
    """orientation 오차(world) = log(R_d R^T) 벡터."""
    return pin.log3(R_d @ R.T)

def main():
    d=json.load(open(TRAJ)); dt_t=d['dt']; N=d['N']
    P=np.array(d['P']); Th=np.array(d['Th'])
    Ft={f:np.array(d['Ft'][f]) for f in FEET}
    Fr={f:np.array(d['Fr'][f]) for f in FEET}
    con={f:d['contact'][f] for f in FEET}
    print("[TRACK] TOWR:", os.path.basename(TRAJ),"kind=",d['kind'],"N=",N)
    # TOWR base 속도·가속(finite diff)
    Pd=np.gradient(P,dt_t,axis=1); Pdd=np.gradient(Pd,dt_t,axis=1)

    # 오프라인 IK → 스윙발 관절 참조
    q_des=np.zeros((N+1,NQ)); ike=0; qp=pin.neutral(pm); qp[2]=P[2,0]
    for k in range(N+1):
        Rb=pin.rpy.rpyToMatrix(Th[0,k],Th[1,k],Th[2,k])
        qk,e=leg_ik(qp,pin.SE3(Rb,P[:,k]),{f:Ft[f][:,k] for f in FEET}); ike=max(ike,e)
        q_des[k]=qk; qp=qk
    print("[TRACK] 오프라인 IK 완료 최대발오차=%.4f"%ike)

    m=mj.MjModel.from_xml_path(MJCF); data=mj.MjData(m); sim_dt=m.opt.timestep
    sub=max(1,int(round(dt_t/sim_dt)))
    print("[TRACK] MJCF=%s sim_dt=%.4f sub=%d"%(os.path.basename(MJCF),sim_dt,sub))
    data.qpos[0:3]=P[:,0]; data.qpos[3:7]=[1,0,0,0]
    _u=np.zeros(m.nu); _u[_PIN2MJ]=q_des[0][7:]; data.qpos[7:7+m.nu]=_u
    mj.mj_forward(m,data)
    bid={f:mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,FRAME[f]) for f in FEET}

    # 게인
    # ★안정 게인(crawl 검증): 강한 자세권한+부드러운 위치·스윙 → tilt<7° 완주
    KP_P=float(os.environ.get('KP_P','300')); KD_P=float(os.environ.get('KD_P','30'))   # base pos(가속단위)
    KP_R=float(os.environ.get('KP_R','1600'));KD_R=float(os.environ.get('KD_R','70'))   # base ori(강)
    KP_J=float(os.environ.get('KP_J','20'));  KD_J=float(os.environ.get('KD_J','1'))    # 스윙 관절(약, 반력↓)
    view=os.environ.get('VIEW','0')!='0'; v=None
    if view:
        import mujoco.viewer as mjv; v=mjv.launch_passive(m,data)

    fell=False
    for s in range((N)*sub):
        k=min(s//sub,N-1); fr=(s%sub)/sub
        # 측정 base
        p=data.qpos[0:3].copy(); quat=data.qpos[3:7]  # wxyz
        R=np.array([[0.,0,0],[0,0,0],[0,0,0]]); mj.mju_quat2Mat(R.ravel(),quat); R=R.reshape(3,3)
        v_lin=data.qvel[0:3].copy(); w_ang=data.qvel[3:6].copy()
        # TOWR 참조(보간)
        pd=(1-fr)*P[:,k]+fr*P[:,k+1]; vd=(1-fr)*Pd[:,k]+fr*Pd[:,k+1]; ad=(1-fr)*Pdd[:,k]+fr*Pdd[:,k+1]
        thd=(1-fr)*Th[:,k]+fr*Th[:,k+1]; Rd=pin.rpy.rpyToMatrix(*thd)
        # ── 원하는 base 렌치(world): 가속ff + 중력 + 피드백 ──
        F_des=MASS*(ad + KP_P*(pd-p) + KD_P*(vd-v_lin)) + MASS*np.array([0,0,G])
        M_des=INER@(KP_R*so3_err(Rd,R) - KD_R*w_ang)
        W=np.concatenate([F_des,M_des])
        # ── 지지발 GRF 분배: W = Σ[ f_i ; (r_i-p)×f_i ] ──
        st=[f for f in FEET if con[f][k]]
        fvec={}
        if st:
            Gm=np.zeros((6,3*len(st)))
            for i,f in enumerate(st):
                r=data.xpos[bid[f]]-p
                Gm[0:3,3*i:3*i+3]=np.eye(3)
                Gm[3:6,3*i:3*i+3]=np.array([[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]])
            fsol=Gm.T@np.linalg.solve(Gm@Gm.T+1e-4*np.eye(6),W)
            for i,f in enumerate(st):
                fz=max(fsol[3*i+2],0.0); lim=MU*fz
                fvec[f]=np.array([np.clip(fsol[3*i],-lim,lim),np.clip(fsol[3*i+1],-lim,lim),np.clip(fz,0,2*MASS*G)])
        # ── 토크: 중력보상(+선택적 스윙 CT) + 지지발 -Jᵀf + 스윙발 PD ──
        tau=np.zeros(m.nu)
        qpin=np.zeros(NQ); qpin[0:3]=p; qpin[3:7]=[quat[1],quat[2],quat[3],quat[0]]
        qpin[7:]=data.qpos[7:7+m.nu][_PIN2MJ]
        g_tau=pin.computeGeneralizedGravity(pm,pdat,qpin)
        tau[_PIN2MJ] += g_tau[6:]
        qd=(1-fr)*q_des[k]+fr*q_des[k+1]; _qd=np.zeros(m.nu); _qd[_PIN2MJ]=qd[7:]
        for f in FEET:
            if f in fvec:
                jacp=np.zeros((3,m.nv)); mj.mj_jacBody(m,data,jacp,None,bid[f])
                tau += -(jacp[:,6:6+m.nu].T @ fvec[f])
        # 스윙발(비지지) 관절 PD — 해당 다리 관절만
        LEGJ={'FL':[9,10,11,12],'FR':[13,14,15,16],'HL':[0,1,2,3],'HR':[4,5,6,7]}  # pin leg 인덱스
        for f in FEET:
            if not con[f][k]:
                for pj in LEGJ[f]:
                    ai=_PIN2MJ[pj]
                    tau[ai]+= KP_J*(_qd[ai]-data.qpos[7+ai]) - KD_J*data.qvel[6+ai]
        if m.actuator_forcelimited.any():
            data.ctrl[:]=np.clip(tau,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
        else: data.ctrl[:]=tau
        mj.mj_step(m,data)
        if s%(sub*5)==0:
            z=data.qpos[2]; til=np.degrees(np.arccos(np.clip(1-2*(data.qpos[4]**2+data.qpos[5]**2),-1,1)))
            print("  s=%4d t=%.2f x=%+.3f y=%+.3f z=%.3f tilt=%.1f | TOWRx=%.3f TOWRz=%.3f"
                  %(s,s*sim_dt,data.qpos[0],data.qpos[1],z,til,pd[0],pd[2]),flush=True)
            if z<0.20 or til>50: print("[TRACK] ❌낙상 @%.2fs (z=%.2f tilt=%.0f)"%(s*sim_dt,z,til)); fell=True; break
        if v is not None: v.sync()
    til=np.degrees(np.arccos(np.clip(1-2*(data.qpos[4]**2+data.qpos[5]**2),-1,1)))
    print("[TRACK] %s 최종 x=%.3f (TOWR목표%.3f) z=%.3f tilt=%.1f"
          %("❌낙상" if fell else "✅완주",data.qpos[0],P[0,-1],data.qpos[2],til))

if __name__=='__main__': main()
