#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOWR → B의 WBIC 브리지 — 오프라인 TOWR 궤적을 B(quad_centroidal_17dof)의 성숙
whole-body ID(KinodynamicsID/TSID)로 추종. 내 QP-WBIC 프로토타입이 못한 접촉전이·
soft접촉·마찰콘·발고정을 B의 검증된 TSID가 처리.

방식: B와 동일 셋업(model_handler·KinodynamicsID·MujocoRobot device) 재현 →
  MPC 계획 대신 TOWR 궤적을 참조(q,v,a,contact,forces)로 kino_ID.setTarget → solve → execute.
  + phase-based leash 재생(참조를 실제 전진에 동기화, 개루프 runaway 방지).

상태(2026-07-22): ★slow/quasi-static 참조=견고 추종(평지 crawl Tg0.80 tilt1.7° 완주,
  base z 유지). fast cadence(Tg0.40) 갭 크로싱=미완(fast 스윙영역서 정체/크라우치/전복).
  근본원인=개루프 재생의 벽: B 게이트는 MPC 매10ms 재계획(폐루프)이라 안정, 개루프 TOWR
  재생은 fast서 상태불일치 누적→회복불가. 슬로우는 불일치 작아 추종OK.
  → 갭 크로싱 정답=TOWR footholds/timing을 B의 MPC에 참조주입(폐루프 유지, D1/perceptive식),
     TOWR→WBIC 직접 개루프가 아니라. 브리지=slow지형 배포용으로 유효.

실행: (CONDA_PREFIX 필요)
  export CONDA_PREFIX=/home/jsh/simple-mpc/.pixi/envs/default
  MJCF=../mjcf/quad_terrain_platgap.mjcf TRAJ=../towr/traj_crawl_platgap.json VIEW=0 \
    /home/jsh/simple-mpc/.pixi/envs/default/bin/python towr_track_B.py
"""
import numpy as np, mujoco as _mj, os as _os0, json, time
_GO2_MJCF=_os0.environ.get("MJCF","/home/jsh/문서/jsh/simulation/quad/mjcf/quad_terrain_platgap.mjcf")
_PIN2MJ=[8,9,10,11,12,13,14,15,16,0,1,2,3,4,5,6,7]

class MujocoRobot:                                        # B와 동일(측정=pin규약, execute=PIN2MJ remap)
    def __init__(self, q0, dt_simu, view=False):
        self.m=_mj.MjModel.from_xml_path(_GO2_MJCF); self.m.opt.timestep=dt_simu
        _st=float(_os0.environ.get("STIFF","0.002")); self.m.geom_solref[:,0]=_st; self.m.geom_solref[:,1]=1.0
        _rl=float(_os0.environ.get("REAR_LOCK","500"))
        if _rl>0:
            for _jn in ("FL_foot_joint","FR_foot_joint","HL_foot_joint","HR_foot_joint"):
                _jid=_mj.mj_name2id(self.m,_mj.mjtObj.mjOBJ_JOINT,_jn)
                if _jid>=0: self.m.jnt_stiffness[_jid]=_rl; self.m.dof_damping[self.m.jnt_dofadr[_jid]]=_rl*0.2
        _wl=float(_os0.environ.get("WAIST_LOCK","2000"))
        if _wl>0:
            _wj=_mj.mj_name2id(self.m,_mj.mjtObj.mjOBJ_JOINT,"FB_waist_joint")
            if _wj>=0: self.m.jnt_stiffness[_wj]=_wl; self.m.dof_damping[self.m.jnt_dofadr[_wj]]=_wl*0.2
        self.d=_mj.MjData(self.m); self.nu=self.m.nu; self._set(q0); self.viewer=None
        if view:
            import mujoco.viewer as _v; self.viewer=_v.launch_passive(self.m,self.d)
    def _set(self,q):
        self.d.qpos[0:3]=q[0:3]; x,y,z,w=q[3:7]; self.d.qpos[3:7]=[w,x,y,z]
        _tmp=np.zeros(self.nu); _tmp[_PIN2MJ]=q[7:7+self.nu]; self.d.qpos[7:7+self.nu]=_tmp
        self.d.qvel[:]=0.0; _mj.mj_forward(self.m,self.d)
    def measureState(self):
        d=self.d; qp=np.zeros(self.m.nq); vp=np.zeros(self.m.nv)
        qp[0:3]=d.qpos[0:3]; w,x,y,z=d.qpos[3:7]; qp[3:7]=[x,y,z,w]
        R=np.zeros(9); _mj.mju_quat2Mat(R,d.qpos[3:7]); R=R.reshape(3,3)
        vp[0:3]=R.T@d.qvel[0:3]; vp[3:6]=d.qvel[3:6]
        qp[7:]=np.asarray(d.qpos[7:7+self.nu])[_PIN2MJ]; vp[6:]=np.asarray(d.qvel[6:6+self.nu])[_PIN2MJ]
        return qp, vp
    def execute(self,tau):
        _um=np.zeros(self.nu); _um[_PIN2MJ]=np.asarray(tau).ravel()[:self.nu]; self.d.ctrl[:]=_um
        _mj.mj_step(self.m,self.d)
        if self.viewer: self.viewer.sync()

# ── simple_mpc 셋업(B와 동일) ──
from simple_mpc import RobotModelHandler, RobotDataHandler, KinodynamicsID, KinodynamicsIDSettings
import pinocchio as _pin
URDF="/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf"
base_joint_name="root_joint"; _R=0.025
_M=_pin.buildModelFromUrdf(URDF,_pin.JointModelFreeFlyer())
for _L in ['FL','FR','HL','HR']:
    _fr=_M.frames[_M.getFrameId(_L+"_foot_contact_link")]
    _pl=_fr.placement*_pin.SE3(np.eye(3),np.array([0,0,-_R]))
    _M.addFrame(_pin.Frame(_L+"_foot",_fr.parentJoint,_fr.parentFrame,_pl,_pin.FrameType.OP_FRAME))
_data=_M.createData(); _q=_pin.neutral(_M); _q[2]=0.50
_fid={_L:_M.getFrameId(_L+"_foot") for _L in ['FL','FR','HL','HR']}
_ftgt={'FL':[0.30,0.16,0.0],'FR':[0.30,-0.16,0.0],'HL':[-0.30,0.16,0.0],'HR':[-0.30,-0.16,0.0]}
for _it in range(400):
    _pin.forwardKinematics(_M,_data,_q); _pin.updateFramePlacements(_M,_data); _pin.computeJointJacobians(_M,_data,_q)
    _err=np.zeros(12); _J=np.zeros((12,_M.nv))
    for _i,_L in enumerate(['FL','FR','HL','HR']):
        _err[3*_i:3*_i+3]=np.array(_ftgt[_L])-_data.oMf[_fid[_L]].translation
        _J[3*_i:3*_i+3]=_pin.getFrameJacobian(_M,_data,_fid[_L],_pin.LOCAL_WORLD_ALIGNED)[:3]
    _J[:,:6]=0.0
    if np.linalg.norm(_err)<1e-5: break
    _q=_pin.integrate(_M,_q,0.5*np.linalg.lstsq(_J,_err,rcond=None)[0])
_M.referenceConfigurations["standing"]=_q.copy()
model_handler=RobotModelHandler(_M,"standing",base_joint_name)
for _L in ['FL','FR','HL','HR']: model_handler.addPointFoot(_L+"_foot",base_joint_name)
data_handler=RobotDataHandler(model_handler)
NQ=model_handler.getModel().nq; NV=model_handler.getModel().nv
print("[BRIDGE] model nq=%d nv=%d standing_z=%.3f"%(NQ,NV,_q[2]),flush=True)

# KinodynamicsID(=B의 WBIC)
s=KinodynamicsIDSettings()
s.kp_base=float(_os0.environ.get("KP_BASE","40.0")); s.kp_posture=float(_os0.environ.get("KP_POSTURE","10.0"))  # ★40=firm base(7=처짐)
s.kp_contact=float(_os0.environ.get("KP_CONTACT","10.0")); s.w_base=float(_os0.environ.get("W_BASE","100.0"))
s.w_posture=float(_os0.environ.get("W_POSTURE","1.0")); s.w_contact_force=float(_os0.environ.get("W_CFORCE","1.0"))
s.w_contact_motion=float(_os0.environ.get("W_CMOTION","1.0")); s.friction_coefficient=float(_os0.environ.get("FRICOEF","0.8"))
dt_simu=0.001
kino_ID=KinodynamicsID(model_handler,dt_simu,s)

# ── IK: TOWR base+발 → pin q(관절순=model_handler) ──
_fidk={_L:_M.getFrameId(_L+"_foot") for _L in ['FL','FR','HL','HR']}
def node_ik(q_init, base_pos, R, foot_world):
    q=q_init.copy(); q[0:3]=base_pos; q[3:7]=_pin.Quaternion(R).coeffs()
    for _ in range(80):
        _pin.forwardKinematics(_M,_data,q); _pin.updateFramePlacements(_M,_data); _pin.computeJointJacobians(_M,_data,q)
        err=np.zeros(12); Jk=np.zeros((12,_M.nv))
        for i,L in enumerate(['FL','FR','HL','HR']):
            err[3*i:3*i+3]=foot_world[L]-_data.oMf[_fidk[L]].translation
            Jk[3*i:3*i+3]=_pin.getFrameJacobian(_M,_data,_fidk[L],_pin.LOCAL_WORLD_ALIGNED)[:3]
        Jk[:,:6]=0.0
        if np.linalg.norm(err)<1e-6: break
        q=_pin.integrate(_M,q,0.5*np.linalg.lstsq(Jk,err,rcond=None)[0])
    return q, np.linalg.norm(err)

def main():
    d=json.load(open(_os0.environ.get("TRAJ","/home/jsh/문서/jsh/simulation/quad/towr/traj_crawl_platgap.json")))
    dt_t=d['dt']; N=d['N']; P=np.array(d['P']); Th=np.array(d['Th'])
    FEET=['FL','FR','HL','HR']
    Ft={f:np.array(d['Ft'][f]) for f in FEET}; Fr={f:np.array(d['Fr'][f]) for f in FEET}
    con={f:d['contact'][f] for f in FEET}
    print("[BRIDGE] TOWR:",_os0.path.basename(_os0.environ.get("TRAJ","platgap")),"N=",N,"dt=",dt_t,flush=True)
    Pd=np.gradient(P,dt_t,axis=1); Pdd=np.gradient(Pd,dt_t,axis=1); Thd=np.gradient(Th,dt_t,axis=1)

    # 노드별 pin q_des(IK) + finite-diff v_des,a_des(관절) ; base는 TOWR
    Q=np.zeros((N+1,NQ)); ike=0; qp=_M.referenceConfigurations["standing"].copy()
    for k in range(N+1):
        Rk=_pin.rpy.rpyToMatrix(Th[0,k],Th[1,k],Th[2,k])
        qk,e=node_ik(qp,P[:,k],Rk,{f:Ft[f][:,k] for f in FEET}); ike=max(ike,e); Q[k]=qk; qp=qk
    print("[BRIDGE] 오프라인 IK 최대발오차=%.4f"%ike,flush=True)
    Qj=Q[:,7:]; Vj=np.gradient(Qj,dt_t,axis=0); Aj=np.gradient(Vj,dt_t,axis=0)  # 관절 v,a
    Thdd=np.gradient(Thd,dt_t,axis=1)

    device=MujocoRobot(Q[0], dt_simu, view=_os0.environ.get("VIEW","0")!="0")
    sub=int(round(dt_t/dt_simu))                                       # 20
    print("[BRIDGE] sub=%d STIFF=%s"%(sub,_os0.environ.get("STIFF","0.002")),flush=True)

    def ref_at(k):
        """TOWR 노드 k → (q_des(pin), v_des(pin,base local), a_des, contact[4], forces[4])."""
        Rk=_pin.rpy.rpyToMatrix(Th[0,k],Th[1,k],Th[2,k])
        q_des=Q[k].copy()
        v_des=np.zeros(NV); a_des=np.zeros(NV)
        v_des[0:3]=Rk.T@Pd[:,k]; v_des[3:6]=Rk.T@Thd[:,k]; v_des[6:]=Vj[k]     # base twist=local
        a_des[0:3]=Rk.T@Pdd[:,k]; a_des[3:6]=Rk.T@Thdd[:,k]; a_des[6:]=Aj[k]
        cs=[bool(con[f][k]) for f in FEET]
        fs=[Fr[f][:,k].copy() for f in FEET]
        return q_des,v_des,a_des,cs,fs

    # ★phase-based 재생(leash): 참조를 실제 전진에 동기화(개루프 시간재생은 뒤처지면 참조가 달아남).
    #   로봇 x가 참조 x보다 뒤처지면 phase 진행률↓ → 게이트 전체가 느려지되 동기 유지.
    K_LEASH=float(_os0.environ.get("LEASH","6.0")); LAG0=float(_os0.environ.get("LAG0","0.05"))
    RMIN=float(_os0.environ.get("RMIN","0.05"))
    fell=False; t=0.0; pk=0.0; nsub=0
    maxsteps=int(N*sub*4)                                 # leash로 느려질 수 있어 여유
    while pk < N-1e-6 and nsub < maxsteps:
        k0=int(pk); al=pk-k0; k1=min(k0+1,N)
        q0,v0,a0,cs0,fs0=ref_at(k0); q1,v1,a1,_,fs1=ref_at(k1)
        q_i=q0.copy(); q_i[0:3]=(1-al)*q0[0:3]+al*q1[0:3]; q_i[7:]=(1-al)*q0[7:]+al*q1[7:]
        q_i[3:7]=q0[3:7]
        v_i=(1-al)*v0+al*v1; a_i=(1-al)*a0+al*a1
        f_i=[(1-al)*fs0[i]+al*fs1[i] for i in range(4)]
        q_meas,v_meas=device.measureState()
        kino_ID.setTarget(q_i, v_i, a_i, cs0, f_i)
        tau=kino_ID.solve(t, q_meas, v_meas)
        device.execute(tau); t+=dt_simu; nsub+=1
        ref_x=(1-al)*P[0,k0]+al*P[0,k1]; act_x=device.d.qpos[0]      # leash: 참조 x vs 실제 x
        lag=ref_x-act_x
        rate=float(np.clip(1.0-K_LEASH*max(lag-LAG0,0.0), RMIN, 1.0))
        pk += rate*(dt_simu/dt_t)
        if nsub%200==0:
            dd=device.d; z=dd.qpos[2]
            til=np.degrees(np.arccos(np.clip(1-2*(dd.qpos[4]**2+dd.qpos[5]**2),-1,1)))
            print("  t=%.2f pk=%.1f x=%+.3f y=%+.3f z=%.3f tilt=%.1f | refx=%.3f lag=%.3f rate=%.2f"
                  %(t,pk,dd.qpos[0],dd.qpos[1],z,til,ref_x,lag,rate),flush=True)
            if z<0.20 or til>50: print("[BRIDGE] ❌낙상 @%.2fs (z=%.2f tilt=%.0f)"%(t,z,til)); fell=True; break
    dd=device.d; til=np.degrees(np.arccos(np.clip(1-2*(dd.qpos[4]**2+dd.qpos[5]**2),-1,1)))
    _done=(pk>=N-1.0) and not fell
    print("[BRIDGE] %s 최종 x=%.3f (TOWR목표%.3f) z=%.3f tilt=%.1f pk=%.1f/%d"
          %("✅완주" if _done else ("❌낙상" if fell else "⚠️정체"),dd.qpos[0],P[0,-1],dd.qpos[2],til,pk,N),flush=True)

if __name__=='__main__': main()
