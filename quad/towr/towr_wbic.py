#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOWR Phase0c-3 — 풀 QP-WBIC 추종기(fast cadence 갭 크로싱 관문).
WBIC-lite(준정적 gravity+GRF)가 fast cadence를 못 잡는 한계를 full-dynamics WBIC로 해결.

QP (변수 x=[q̈(nv), f(3·ns)]):
  min  w_b·‖q̈₀₆−a_base_des‖² + w_sw·Σ‖J_sw q̈−a_sw_des‖² + w_f·‖f−f_ref‖² + ε‖q̈‖²
  s.t. base 동역학:  M₀₆ q̈ − Σ Jc_fᵀ|₀₆ f = −bias₀₆        (부양 base 무구동)
       접촉:        Jc_f q̈ = 0  (지지발 무가속, J̇q̇≈0)
       마찰콘:      f_z≥f_min, |f_xy|≤μ f_z, f_z≤f_max
  → τ_act = M₆: q̈ + bias₆: − Σ Jc_fᵀ|₆: f  (해 복원)
MuJoCo 규약: qvel[0:3]=world 선속도, qvel[3:6]=local 각속도. Jc=mj_jacBody(world).

실행: /home/jsh/simple-mpc/.pixi/envs/default/bin/python towr_wbic.py
  env TRAJ=traj_crawl_platgap.json MJCF=지형씬 VIEW=0
"""
import numpy as np, mujoco as mj, pinocchio as pin, json, os, proxsuite

TRAJ = os.environ.get('TRAJ','/home/jsh/문서/jsh/simulation/quad/towr/traj_crawl_platgap.json')
URDF = "/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf"
MJCF = os.environ.get('MJCF','/home/jsh/문서/jsh/simulation/quad/mjcf/quad_terrain_platgap.mjcf')
FEET=['FL','FR','HL','HR']; FRAME={f:f+'_foot_contact_link' for f in FEET}; _R=0.025
MASS=38.016; G=9.81; MU=0.6

# pinocchio(IK 전용)
pm=pin.buildModelFromUrdf(URDF,pin.JointModelFreeFlyer()); pdat=pm.createData()
pfid={f:pm.getFrameId(FRAME[f]) for f in FEET}; NQ,NV=pm.nq,pm.nv
_PIN2MJ=np.array([8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7])

def leg_ik(q_init,base_SE3,foot_world):
    q=q_init.copy(); q[0:3]=base_SE3.translation; q[3:7]=pin.Quaternion(base_SE3.rotation).coeffs()
    for _ in range(60):
        pin.forwardKinematics(pm,pdat,q); pin.updateFramePlacements(pm,pdat)
        err=np.zeros(12); Js=np.zeros((12,NV))
        for i,f in enumerate(FEET):
            p=pdat.oMf[pfid[f]].translation+pdat.oMf[pfid[f]].rotation@np.array([0,0,-_R])
            err[3*i:3*i+3]=foot_world[f]-p
            Js[3*i:3*i+3]=pin.computeFrameJacobian(pm,pdat,q,pfid[f],pin.LOCAL_WORLD_ALIGNED)[:3]
        if np.linalg.norm(err)<1e-5: break
        Jl=Js[:,6:]; dq=Jl.T@np.linalg.solve(Jl@Jl.T+1e-6*np.eye(12),err)
        qv=np.zeros(NV); qv[6:]=dq; q=pin.integrate(pm,q,0.5*qv)
    return q,np.linalg.norm(err)

def so3_err(R_d,R): return pin.log3(R_d@R.T)

def main():
    d=json.load(open(TRAJ)); dt_t=d['dt']; N=d['N']
    P=np.array(d['P']); Th=np.array(d['Th'])
    Ft={f:np.array(d['Ft'][f]) for f in FEET}; Fr={f:np.array(d['Fr'][f]) for f in FEET}
    con={f:d['contact'][f] for f in FEET}
    print("[WBIC] TOWR:",os.path.basename(TRAJ),"kind=",d.get('kind'),"N=",N)
    Pd=np.gradient(P,dt_t,axis=1); Pdd=np.gradient(Pd,dt_t,axis=1)
    Ftd={f:np.gradient(Ft[f],dt_t,axis=1) for f in FEET}                 # 발 속도참조
    Ftdd={f:np.gradient(Ftd[f],dt_t,axis=1) for f in FEET}
    Thd=np.gradient(Th,dt_t,axis=1)

    # 오프라인 IK → 초기자세용
    q_des=np.zeros((N+1,NQ)); ike=0; qp=pin.neutral(pm); qp[2]=P[2,0]
    for k in range(N+1):
        Rb=pin.rpy.rpyToMatrix(Th[0,k],Th[1,k],Th[2,k])
        qk,e=leg_ik(qp,pin.SE3(Rb,P[:,k]),{f:Ft[f][:,k] for f in FEET}); ike=max(ike,e); q_des[k]=qk; qp=qk
    print("[WBIC] 오프라인 IK 최대발오차=%.4f"%ike)

    m=mj.MjModel.from_xml_path(MJCF); data=mj.MjData(m); sim_dt=m.opt.timestep
    _st=float(os.environ.get('STIFF','0.002'))          # ★강체접촉 정합(B의 STIFF, WBIC 강체가정↔soft sphere)
    m.geom_solref[:,0]=_st; m.geom_solref[:,1]=1.0
    sub=max(1,int(round(dt_t/sim_dt)))
    print("[WBIC] MJCF=%s sim_dt=%.4f sub=%d"%(os.path.basename(MJCF),sim_dt,sub))
    data.qpos[0:3]=P[:,0]; data.qpos[3:7]=[1,0,0,0]
    _u=np.zeros(m.nu); _u[_PIN2MJ]=q_des[0][7:]; data.qpos[7:7+m.nu]=_u; mj.mj_forward(m,data)
    bid={f:mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,FRAME[f]) for f in FEET}

    # WBIC 게인/가중
    KP_P=float(os.environ.get('KP_P','150')); KD_P=float(os.environ.get('KD_P','25'))
    KP_R=float(os.environ.get('KP_R','600')); KD_R=float(os.environ.get('KD_R','50'))
    KP_S=float(os.environ.get('KP_S','150')); KD_S=float(os.environ.get('KD_S','25'))  # swing foot
    W_B=float(os.environ.get('W_B','10')); W_BR=float(os.environ.get('W_BR','80'))  # base 위치·자세 분리
    W_SW=float(os.environ.get('W_SW','3'))
    W_F=float(os.environ.get('W_F','0.3')); W_QA=float(os.environ.get('W_QA','1e-4'))
    F_MIN=float(os.environ.get('F_MIN','5')); F_MAX=2.0*MASS*G
    view=os.environ.get('VIEW','0')!='0'; vv=None
    if view:
        import mujoco.viewer as mjv; vv=mjv.launch_passive(m,data)

    nv=m.nv; Mmat=np.zeros((nv,nv)); fell=False
    _JD=float(os.environ.get('JDOT','0'))                             # J̇q̇ 유한차분 on/off(노이즈 주의)
    Jprev={f:None for f in FEET}
    for s in range((N)*sub):
        k=min(s//sub,N-1); fr=(s%sub)/sub
        p=data.qpos[0:3].copy(); quat=data.qpos[3:7]
        R=np.zeros(9); mj.mju_quat2Mat(R,quat); R=R.reshape(3,3)
        v_w=data.qvel[0:3].copy(); w_l=data.qvel[3:6].copy()          # 선(world)·각(local)
        # 참조(보간)
        pd=(1-fr)*P[:,k]+fr*P[:,k+1]; vd=(1-fr)*Pd[:,k]+fr*Pd[:,k+1]; ad=(1-fr)*Pdd[:,k]+fr*Pdd[:,k+1]
        thd=(1-fr)*Th[:,k]+fr*Th[:,k+1]; Rd=pin.rpy.rpyToMatrix(*thd)
        # 동역학 양
        mj.mj_fullM(m,Mmat,data.qM); bias=data.qfrc_bias.copy()
        st=[f for f in FEET if con[f][k]]; sw=[f for f in FEET if not con[f][k]]
        ns=len(st); nx=nv+3*ns
        Jc={}; Jdqd={}
        for f in FEET:
            J=np.zeros((3,nv)); mj.mj_jacBody(m,data,J,None,bid[f]); Jc[f]=J
            Jdqd[f]=_JD*(((J-Jprev[f])/sim_dt)@data.qvel) if Jprev[f] is not None else np.zeros(3)  # J̇q̇
            Jprev[f]=J.copy()
        # ── 비용 H,g ──
        H=np.zeros((nx,nx)); g=np.zeros(nx)
        # base task: q̈[0:3] world 선가속, q̈[3:6] local 각가속
        a_lin=ad+KP_P*(pd-p)+KD_P*(vd-v_w)
        a_ang=R.T@(KP_R*so3_err(Rd,R))-KD_R*w_l
        Wb=np.diag([W_B,W_B,W_B,W_BR,W_BR,W_BR])         # 위치 W_B, 자세 W_BR(자세 강조)
        a_base=np.concatenate([a_lin,a_ang])
        Jb=np.zeros((6,nx)); Jb[:,0:6]=np.eye(6)
        H+=Jb.T@Wb@Jb; g+=-Jb.T@Wb@a_base
        # swing task
        for f in sw:
            pf=data.xpos[bid[f]]; vf=Jc[f]@data.qvel
            pf_d=(1-fr)*Ft[f][:,k]+fr*Ft[f][:,k+1]; vf_d=(1-fr)*Ftd[f][:,k]+fr*Ftd[f][:,k+1]
            af_d=(1-fr)*Ftdd[f][:,k]+fr*Ftdd[f][:,k+1]
            a_sw=af_d+KP_S*(pf_d-pf)+KD_S*(vf_d-vf) - Jdqd[f]     # Jc q̈ = a_sw − J̇q̇
            Jsw=np.zeros((3,nx)); Jsw[:,0:nv]=Jc[f]
            H+=W_SW*Jsw.T@Jsw; g+=-W_SW*Jsw.T@a_sw
        # force reg to TOWR
        for i,f in enumerate(st):
            fref=(1-fr)*Fr[f][:,k]+fr*Fr[f][:,min(k+1,N)]
            Sf=np.zeros((3,nx)); Sf[:,nv+3*i:nv+3*i+3]=np.eye(3)
            H+=W_F*Sf.T@Sf; g+=-W_F*Sf.T@fref
        # 접촉 무가속 = 고가중 소프트 task(하드 등식은 마찰콘과 충돌해 infeasible 위험)
        W_C=float(os.environ.get('W_C','1e3'))
        for i,f in enumerate(st):
            Jct=np.zeros((3,nx)); Jct[:,0:nv]=Jc[f]
            H+=W_C*Jct.T@Jct; g+=-W_C*Jct.T@(-Jdqd[f])
        H+=W_QA*np.eye(nx); H[nv:,nv:]+=1e-6*np.eye(3*ns)
        H=0.5*(H+H.T)
        # ── 등식: base 동역학(부양 무구동)만 하드 ──
        neq=6; A=np.zeros((neq,nx)); b=np.zeros(neq)
        A[0:6,0:nv]=Mmat[0:6,:]
        for i,f in enumerate(st): A[0:6,nv+3*i:nv+3*i+3]=-Jc[f][:,0:6].T
        b[0:6]=-bias[0:6]
        # ── 부등식 마찰콘: l ≤ C x ≤ u ──
        nin=5*ns; C=np.zeros((nin,nx)); l=np.full(nin,-1e20); u=np.full(nin,1e20)
        for i,f in enumerate(st):
            base=nv+3*i; row=5*i
            C[row,base+2]=1; l[row]=F_MIN; u[row]=F_MAX                 # f_z∈[F_MIN,F_MAX]
            C[row+1,base+0]=1; C[row+1,base+2]=-MU; u[row+1]=0          # f_x-μf_z≤0
            C[row+2,base+0]=-1; C[row+2,base+2]=-MU; u[row+2]=0         # -f_x-μf_z≤0
            C[row+3,base+1]=1; C[row+3,base+2]=-MU; u[row+3]=0
            C[row+4,base+1]=-1; C[row+4,base+2]=-MU; u[row+4]=0
        # ── QP 풀기(proxsuite) ──
        qp=proxsuite.proxqp.dense.QP(nx,neq,nin)
        qp.settings.eps_abs=1e-6; qp.settings.max_iter=200; qp.settings.verbose=False
        qp.init(H,g,A,b,C,l,u); qp.solve()
        x=qp.results.x
        if x is None or np.any(np.isnan(x)):
            qacc=np.zeros(nv); fsol={}
        else:
            qacc=x[0:nv]; fsol={f:x[nv+3*i:nv+3*i+3] for i,f in enumerate(st)}
        # ── τ_act 복원: M₆: q̈ + bias₆: − Σ Jc_fᵀ|₆: f ──
        tau_full=Mmat[6:,:]@qacc+bias[6:]
        for f in st: tau_full-=Jc[f][:,6:].T@fsol[f]
        tau=tau_full.copy()   # mj qvel[6:]=mjcf 관절순=액추에이터순(PIN2MJ 불요)
        if m.actuator_forcelimited.any():
            data.ctrl[:]=np.clip(tau,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
        else: data.ctrl[:]=tau
        mj.mj_step(m,data)
        if s%(sub*5)==0:
            z=data.qpos[2]; til=np.degrees(np.arccos(np.clip(1-2*(data.qpos[4]**2+data.qpos[5]**2),-1,1)))
            print("  s=%4d t=%.2f x=%+.3f y=%+.3f z=%.3f tilt=%.1f | TOWRx=%.3f"
                  %(s,s*sim_dt,data.qpos[0],data.qpos[1],z,til,pd[0]),flush=True)
            if z<0.20 or til>50: print("[WBIC] ❌낙상 @%.2fs (z=%.2f tilt=%.0f)"%(s*sim_dt,z,til)); fell=True; break
        if vv is not None: vv.sync()
    til=np.degrees(np.arccos(np.clip(1-2*(data.qpos[4]**2+data.qpos[5]**2),-1,1)))
    print("[WBIC] %s 최종 x=%.3f (TOWR목표%.3f) z=%.3f tilt=%.1f"
          %("❌낙상" if fell else "✅완주",data.qpos[0],P[0,-1],data.qpos[2],til))

if __name__=='__main__': main()
