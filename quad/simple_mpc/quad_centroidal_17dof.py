import numpy as np
import mujoco as _mj
import os as _os0
# ★점접촉 MJCF(OCP 점접촉 가정과 일치) — 메시발은 OCP와 불일치해 발산. fulldynamics와 동일.
_GO2_MJCF=_os0.environ.get("MJCF","/home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf")
# ★17-DOF remap: pin순(waist,FL,FR,HL,HR) → mjcf순(HL,HR,waist,FL,FR). pin idx→mjcf idx.
#   pin: waist=0·FL1-4·FR5-8·HL9-12·HR13-16  /  mjcf: HL0-3·HR4-7·waist8·FL9-12·FR13-16
_PIN2MJ=[8, 9,10,11,12, 13,14,15,16, 0,1,2,3, 4,5,6,7]
class MujocoRobot:
    """simple-mpc device(BulletRobot) 인터페이스를 MuJoCo로 구현. 토크는 KinodynamicsID(TSID) 출력.
       pin↔mujoco: go2 관절 순서 동일(재정렬 X), 베이스 quat [w,x,y,z]↔[x,y,z,w], lin world→local(R^T)."""
    def __init__(self, q0, dt_simu, view=False):
        self.m=_mj.MjModel.from_xml_path(_GO2_MJCF); self.m.opt.timestep=dt_simu
        import os as _o2                                   # 접촉모델 매칭(컨트롤러 강체가정 ↔ MuJoCo soft)
        _lms=float(_o2.environ.get('LEG_MASS_SCALE','1.0'))   # ★다리무게 가설: 물리(MuJoCo) 다리링크 질량/관성 스케일
        if _lms!=1.0:
            for _b in range(self.m.nbody):
                _bn=_mj.mj_id2name(self.m,_mj.mjtObj.mjOBJ_BODY,_b) or ''
                if any(_s in _bn for _s in ('hip','thigh','calf','foot')):
                    self.m.body_mass[_b]*=_lms; self.m.body_inertia[_b]*=_lms
            _mj.mj_setConst(self.m,_mj.MjData(self.m))
            print('[LEG_MASS-MJ] 다리링크 ×%.2f → 총질량 %.1fkg'%(_lms,self.m.body_mass.sum()),flush=True)
        _bad=float(_o2.environ.get('BODY_ADD','0'))   # ★바디무게 추가(다리비율↓, centroidal 검증)
        if _bad!=0.0:
            _bb=_mj.mj_name2id(self.m,_mj.mjtObj.mjOBJ_BODY,'base'); _m0=self.m.body_mass[_bb]; _mn=_m0+_bad
            self.m.body_inertia[_bb]*=(_mn/_m0); self.m.body_mass[_bb]=_mn; _mj.mj_setConst(self.m,_mj.MjData(self.m))
            print('[BODY_ADD-MJ] base %.2f→%.2fkg 총%.1fkg 다리비율%.0f%%'%(_m0,_mn,self.m.body_mass.sum(),100*(1-self.m.body_mass[_bb]/self.m.body_mass.sum())),flush=True)
        if _o2.environ.get("CONE"): self.m.opt.cone=int(_o2.environ["CONE"])
        if _o2.environ.get("STIFF"): self.m.geom_solref[:,0]=float(_o2.environ["STIFF"]); self.m.geom_solref[:,1]=1.0
        _rl=float(_o2.environ.get("REAR_LOCK","0"))   # ★뒷발목 물리잠금(4-DOF→3-DOF 대칭화, 강성)
        if _rl>0:
            for _jn in ("HL_foot_joint","HR_foot_joint"):
                _jid=_mj.mj_name2id(self.m,_mj.mjtObj.mjOBJ_JOINT,_jn)
                if _jid>=0:
                    self.m.jnt_stiffness[_jid]=_rl; self.m.dof_damping[self.m.jnt_dofadr[_jid]]=_rl*0.2
            print("[REAR_LOCK] 뒷발목 stiffness=%.0f (대칭3-DOF화)"%_rl,flush=True)
        self.d=_mj.MjData(self.m); self.nu=self.m.nu
        self._set(q0); self.viewer=None
        if view:
            import mujoco.viewer as _v; self.viewer=_v.launch_passive(self.m,self.d)
    def _set(self,q):
        self.d.qpos[0:3]=q[0:3]; x,y,z,w=q[3:7]; self.d.qpos[3:7]=[w,x,y,z]
        import numpy as _np0;
        _tmp=_np0.zeros(self.nu); _tmp[_PIN2MJ]=q[7:7+self.nu]; self.d.qpos[7:7+self.nu]=_tmp; self.d.qvel[:]=0.0
        _mj.mj_forward(self.m,self.d)
    def initializeJoints(self,q0): self._set(q0)
    def resetState(self,q0): self._set(q0)
    def measureState(self):
        d=self.d; import numpy as _np
        qp=_np.zeros(self.m.nq); vp=_np.zeros(self.m.nv)
        qp[0:3]=d.qpos[0:3]; w,x,y,z=d.qpos[3:7]; qp[3:7]=[x,y,z,w]
        R=_np.zeros(9); _mj.mju_quat2Mat(R,d.qpos[3:7]); R=R.reshape(3,3)
        vp[0:3]=R.T@d.qvel[0:3]; vp[3:6]=d.qvel[3:6]
        qp[7:]=_np.asarray(d.qpos[7:7+self.nu])[_PIN2MJ]; vp[6:]=_np.asarray(d.qvel[6:6+self.nu])[_PIN2MJ]
        return qp, vp
    def execute(self,tau):
        import numpy as _np
        _um=_np.zeros(self.nu); _um[_PIN2MJ]=_np.asarray(tau).ravel()[:self.nu]; self.d.ctrl[:]=_um
        _mj.mj_step(self.m,self.d)
        if self.viewer:
            d=self.d; m=self.m
            _R=_np.zeros(9); _mj.mju_quat2Mat(_R,d.qpos[3:7]); _R=_R.reshape(3,3)
            _vact=float((_R.T@d.qvel[0:3])[0])                      # 실제 전진속도(base local x)
            _fext=max((float(_np.linalg.norm(d.xfrc_applied[b,:3])) for b in range(1,m.nbody)),default=0.0)
            cv=getattr(self,'cmd_v',_np.zeros(6))
            self.viewer.set_texts([                                # 좌상=시간 우상=외력 좌하=명령/실제
                (_mj.mjtFont.mjFONT_BIG,_mj.mjtGridPos.mjGRID_TOPLEFT,'sim time','%.2f s'%d.time),
                (_mj.mjtFont.mjFONT_BIG,_mj.mjtGridPos.mjGRID_TOPRIGHT,'ext force','%.0f N'%_fext),
                (_mj.mjtFont.mjFONT_BIG,_mj.mjtGridPos.mjGRID_BOTTOMLEFT,'cmd vx/vy/wz\nactual vx','%+.2f %+.2f %+.2f\n%+.2f m/s'%(cv[0],cv[1],cv[5],_vact))])
            self.viewer.sync()
            import time as _t; _t.sleep(self.m.opt.timestep)   # 실시간 페이싱
    def changeCamera(self,*a,**k): pass
    def showQuadrupedFeet(self,*a,**k): pass
    def moveQuadrupedFeet(self,*a,**k): pass
from simple_mpc import (
    RobotModelHandler,
    RobotDataHandler,
    KinodynamicsOCP,
    MPC,
    Interpolator,
    KinodynamicsID,
    KinodynamicsIDSettings,
)
import os as _os, pinocchio as _pin
class _ERD:
    PKG=_os.path.join(_os.environ["CONDA_PREFIX"],"share")            # package:// 루트
    SHARE=_os.path.join(PKG,"example-robot-data/robots")              # robots 디렉토리
    def load(self,name):
        rw=_pin.RobotWrapper.BuildFromURDF(self.SHARE+"/go2_description/urdf/go2.urdf",self.PKG,_pin.JointModelFreeFlyer())
        _pin.loadReferenceConfigurations(rw.model,self.SHARE+"/go2_description/srdf/go2.srdf",False)  # "standing" 자세
        return rw
    def getModelPath(self,sub):
        return self.SHARE
erd=_ERD()
import time
import copy

# ####### CONFIGURATION  ############
# Load robot
URDF = "/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf"  # ★17-DOF(허리+전발목)
base_joint_name = "root_joint"
_M = _pin.buildModelFromUrdf(URDF, _pin.JointModelFreeFlyer())
# ★17-DOF sphere발 접촉프레임: sphere=contact_link 원점(pos 0 0 0)에 반경 0.025 → sole_off=[0,0,-r]
_R = 0.025
_sole={_L:[0.0,0.0,-_R] for _L in ['FL','FR','HL','HR']}
for _L in ['FL','FR','HL','HR']:                          # 접촉프레임 {L}_foot = contact_link + sole_off
    _fr=_M.frames[_M.getFrameId(_L+"_foot_contact_link")]
    _pl=_fr.placement*_pin.SE3(np.eye(3), np.array(_sole[_L]))
    _M.addFrame(_pin.Frame(_L+"_foot", _fr.parentJoint, _fr.parentFrame, _pl, _pin.FrameType.OP_FRAME))
# ★17-DOF standing pose = pin IK(발을 힙 아래 지면, base 0.50, 허리 0). 손하드코드 대신 IK로 견고.
_data = _M.createData(); _q = _pin.neutral(_M); _q[2] = 0.50
_fid = {_L: _M.getFrameId(_L+"_foot") for _L in ['FL','FR','HL','HR']}
_ftgt = {'FL':[0.30,0.16,0.0],'FR':[0.30,-0.16,0.0],'HL':[-0.30,0.16,0.0],'HR':[-0.30,-0.16,0.0]}
for _it in range(400):
    _pin.forwardKinematics(_M,_data,_q); _pin.updateFramePlacements(_M,_data); _pin.computeJointJacobians(_M,_data,_q)
    _err=np.zeros(12); _J=np.zeros((12,_M.nv))
    for _i,_L in enumerate(['FL','FR','HL','HR']):
        _err[3*_i:3*_i+3]=np.array(_ftgt[_L])-_data.oMf[_fid[_L]].translation
        _J[3*_i:3*_i+3]=_pin.getFrameJacobian(_M,_data,_fid[_L],_pin.LOCAL_WORLD_ALIGNED)[:3]
    _J[:,:6]=0.0
    if np.linalg.norm(_err)<1e-5: break
    _q=_pin.integrate(_M,_q,0.5*np.linalg.lstsq(_J,_err,rcond=None)[0])
_qstand = _q.copy(); _M.referenceConfigurations["standing"] = _qstand
print("[17DOF] standing IK base_z=%.3f 발오차=%.4f nq=%d nv=%d"%(_qstand[2],np.linalg.norm(_err),_M.nq,_M.nv),flush=True)
_lms_pin = float(_os.environ.get('LEG_MASS_SCALE','1.0'))   # ★다리무게 가설: 모델(OCP+TSID) 다리링크 관성 스케일
if _lms_pin != 1.0:                                          # joint 0=universe,1=root_joint(base) 제외, 2..=다리링크
    for _ji in range(2, _M.njoints):
        _I = _M.inertias[_ji]
        _M.inertias[_ji] = _pin.Inertia(_I.mass*_lms_pin, _I.lever, _I.inertia*_lms_pin)
    print('[LEG_MASS-PIN] 다리링크 ×%.2f → pin 총질량 %.1fkg'
          % (_lms_pin, sum(_M.inertias[_j].mass for _j in range(1,_M.njoints))), flush=True)
_bad_pin = float(_os.environ.get('BODY_ADD','0'))   # ★바디무게 추가(OCP+TSID 모델, pinocchio base=joint1)
if _bad_pin != 0.0:
    _Ib = _M.inertias[1]; _mnb = _Ib.mass + _bad_pin
    _M.inertias[1] = _pin.Inertia(_mnb, _Ib.lever, _Ib.inertia * (_mnb/_Ib.mass))
    print('[BODY_ADD-PIN] base +%.1fkg → %.2fkg'%(_bad_pin, _mnb), flush=True)
model_handler = RobotModelHandler(_M, "standing", base_joint_name)
model_handler.addPointFoot("FL_foot", base_joint_name)
model_handler.addPointFoot("FR_foot", base_joint_name)
model_handler.addPointFoot("HL_foot", base_joint_name)
model_handler.addPointFoot("HR_foot", base_joint_name)
data_handler = RobotDataHandler(model_handler)

nq = model_handler.getModel().nq
nv = model_handler.getModel().nv
nu = nv - 6
nf = 12
force_size = 3
nk = model_handler.getFeetNb()
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -model_handler.getMass() / nk * gravity[2]
u0 = np.concatenate((fref, fref, fref, fref, np.zeros(model_handler.getModel().nv - 6)))
dt_mpc = 0.01

_wbp = float(_os.environ.get("WBPOS", "0"))   # ★base x,y 위치 가중(기본0=자유=보행용). STAND/드리프트엔 >0로 앵커
_wbz = float(_os.environ.get("WBZ", "100"))   # base z 가중(FullDynamics=0=발프레임에 위임)
w_basepos = [_wbp, _wbp, _wbz, float(_os.environ.get("WBORI","200")), float(_os.environ.get("WBORI","200")), 0]
w_legpos = [1, 1, 1, 1]

w_basevel = [float(_os.environ.get("WBVX","60")), float(_os.environ.get("WBVY","10")), 10, 10, 10, float(_os.environ.get("WBWZ","10"))]  # ★측방/yaw 가중 env(드리프트 억제)
w_legvel = [0.1, 0.1, 0.1, 0.1]
# ★FullDynamics 참조: 뒷발목(pin idx 9=HL_foot,13=HR_foot)은 point-foot서 floppy → posture/vel 강하게 핀고정
_ankw = float(_os.environ.get("ANKLE_W", "50")); _ankdw = float(_os.environ.get("ANKLE_DW", "5"))
_wlp = [1.0]*nu; _wlv = [0.1]*nu
for _ia in (4, 8, 12, 16):   # ★17-DOF: 전 4발목(FL4·FR8·HL12·HR16) point-foot floppy → 강핀
    _wlp[_ia] = _ankw; _wlv[_ia] = _ankdw
_waistw = float(_os.environ.get("WAIST_W", "200"))   # ★허리(pin nu idx0) 큰 몸통DOF → 강 posture 홀드(A의 WAIST_KP 등가)
_wlp[0] = _waistw; _wlv[0] = _waistw*0.05
w_x = np.array(w_basepos + _wlp + w_basevel + _wlv)   # _9: nu=14 비균일
w_x = np.diag(w_x)
w_linforce = np.array([0.01, 0.01, 0.01])
w_u = np.concatenate(
    (
        w_linforce,
        w_linforce,
        w_linforce,
        w_linforce,
        np.ones(model_handler.getModel().nv - 6) * 1e-5,
    )
)
w_u = np.diag(w_u)
w_LFRF = 2000
# ★STEP3: 발을 OCP가 더 자유롭게 최적화(footstep adaptation). w_frame↓ → 발 참조구속 완화 → OCP가 최적 foothold 선택.
#   발은 이미 q통한 결정변수이고 reachability는 kinematics_limits(qmin/qmax)가 제공. FOOT_DECISION=0=baseline(w_frame full).
_fd = float(_os.environ.get("FOOT_DECISION","0"))
if _fd > 0: w_LFRF = w_LFRF / (1.0 + 9.0 * min(_fd, 1.0))   # FD=1 → w_frame 10%(거의 자유)
if _fd > 0: print("[FOOT_DECISION] w_frame %d→%.0f (발 자유최적화)" % (2000, w_LFRF), flush=True)
_wcap = float(_os.environ.get("WCENT_ANG_P","0.1"))   # ★pitch/roll 각운동량 가중(02_Leg 다리79%=pitch각모멘텀 폭증, 0.1로 못잡음)
_wcdp = float(_os.environ.get("WCENTDER_ANG_P","0.1"))
w_cent_lin = np.array([0.0, 0.0, 1])
w_cent_ang = np.array([_wcap, _wcap, 10])
w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
w_centder_lin = np.ones(3) * 0.0
w_centder_ang = np.array([_wcdp, _wcdp, 0.1])
w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

problem_conf = dict(
    timestep=dt_mpc,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    w_centder=w_centder,
    gravity=gravity,
    force_size=3,
    w_frame=np.eye(3) * w_LFRF,
    qmin=model_handler.getModel().lowerPositionLimit[7:],
    qmax=model_handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    kinematics_limits=True,
    force_cone=_os.environ.get("FCONE","0")!="0",    # FullDynamics는 ON
    land_cstr=_os.environ.get("LAND","0")!="0",       # FullDynamics는 ON
)
T = int(_os.environ.get("HORIZON","50"))

dynproblem = KinodynamicsOCP(problem_conf, model_handler)
dynproblem.createProblem(
    model_handler.getReferenceState(), T, force_size, gravity[2], False
)

T_ds = int(_os.environ.get("TDS","10"))   # ★cadence env(빠른보폭=고속): 더블서포트/스윙 노드수
T_ss = int(_os.environ.get("TSS","30"))

mpc_conf = dict(
    support_force=-model_handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=float(_os.environ.get("MU_INIT", "1e-8")),     # 정규화(↑=안정·보수). 02_Leg 발산 완화
    max_iters=int(_os.environ.get("MAXITER", "1")),        # RTI 반복(↑=수렴↑·느림)
    num_threads=int(_os.environ.get("NTH","8")),
    swing_apex=float(_os.environ.get("APEX","0.15")),   # ★step height env(고속 gait 검토)
    T_fly=T_ss,
    T_contact=T_ds,
    timestep=dt_mpc,
    capture_gain=float(_os.environ.get("KCAP","0")), alip_gain=float(_os.environ.get("ALIP","0")),  # ★반응형 발배치
    predict_foot=float(_os.environ.get("PREDFOOT","0")),   # ★OCP 예측 발배치
    w_foot_ref=float(_os.environ.get("W_FOOT_REF","0")),   # ★STEP2: capture-point cost 참조(측정CoM)
)

mpc = MPC(mpc_conf, dynproblem)

""" Define contact sequence throughout horizon"""
contact_phase_quadru = {
    "FL_foot": True,
    "FR_foot": True,
    "HL_foot": True,
    "HR_foot": True,
}
contact_phase_lift_FL = {
    "FL_foot": False,
    "FR_foot": True,
    "HL_foot": True,
    "HR_foot": False,
}
contact_phase_lift_FR = {
    "FL_foot": True,
    "FR_foot": False,
    "HL_foot": False,
    "HR_foot": True,
}
contact_phase_lift = {
    "FL_foot": False,
    "FR_foot": False,
    "HL_foot": False,
    "HR_foot": False,
}
if _os.environ.get("STAND"):                              # ★서있기: 전 스탠스(stepping 없음) — 지지 격리·튜닝용
    contact_phases = [contact_phase_quadru] * (2 * (T_ds + T_ss))
else:
    contact_phases = [contact_phase_quadru] * T_ds
    contact_phases += [contact_phase_lift_FL] * T_ss
    contact_phases += [contact_phase_quadru] * T_ds
    contact_phases += [contact_phase_lift_FR] * T_ss
mpc.generateCycleHorizon(contact_phases)

""" Interpolation """
N_simu = 10  # Number of substep the simulation does between two MPC computation
dt_simu = dt_mpc / N_simu
interpolator = Interpolator(model_handler.getModel())

""" Inverse Dynamics """
kino_ID_settings = KinodynamicsIDSettings()
kino_ID_settings.kp_base = float(_os.environ.get("KP_BASE","7.0"))
kino_ID_settings.kp_posture = float(_os.environ.get("KP_POSTURE","10.0"))
kino_ID_settings.kp_contact = float(_os.environ.get("KP_CONTACT","10.0"))
kino_ID_settings.w_base = float(_os.environ.get("W_BASE","100.0"))
kino_ID_settings.w_posture = float(_os.environ.get("W_POSTURE","1.0"))
kino_ID_settings.w_contact_force = float(_os.environ.get("W_CFORCE","1.0"))
kino_ID_settings.w_contact_motion = float(_os.environ.get("W_CMOTION","1.0"))   # ★발 고정(미끄럼방지). ↑=firm
# ★TSID 제약 완화 실험: 마찰콘·발고정등식을 풀어 실현 자유도↑ (사용자 "제약 프리하게")
kino_ID_settings.friction_coefficient = float(_os.environ.get("FRICOEF","0.8"))   # ↑=마찰콘 넓힘(전단력 자유)
if _os.environ.get("CME") is not None:
    kino_ID_settings.contact_motion_equality = _os.environ.get("CME") != "0"       # 0=발고정 부등식/soft(slip 허용)

kino_ID = KinodynamicsID(model_handler, dt_simu, kino_ID_settings)


""" Initialize simulation"""
device = MujocoRobot(
    model_handler.getReferenceState()[: model_handler.getModel().nq],
    dt_simu,
    view=bool(int(_os.environ.get("VIEW","0"))),
)

device.initializeJoints(
    model_handler.getReferenceState()[: model_handler.getModel().nq]
)
device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])

# ★① 지형인지 발판: 전역 heightmap을 mj_ray로 생성 → MPC 주입. 발판 z=지형높이, Raibert 발판이 갭(무효)이면
#   반경내 최근접 유효셀로 xy 이동(갭 회피) → OCP가 base 협조 최적화. HEIGHTMAP=1로 켬(미설정=평지 baseline).
if _os.environ.get("HEIGHTMAP","0") != "0":
    _hres = float(_os.environ.get("HM_RES","0.05")); _hox = -0.6; _hoy = -1.0; _hnx = 180; _hny = 40
    _hm = np.full((_hnx, _hny), np.nan, dtype=np.float32)
    _gg = np.array([0,0,1,0,0,0], dtype=np.uint8)   # group2(지형)만 — terrain_z와 동일 마스킹
    _vec = np.array([0.0,0.0,-1.0]); _gid = np.zeros(1, dtype=np.int32); _nval = 0
    for _i in range(_hnx):
        for _j in range(_hny):
            _pnt = np.array([_hox+_i*_hres, _hoy+_j*_hres, 2.0])
            _dist = _mj.mj_ray(device.m, device.d, _pnt, _vec, _gg, 1, -1, _gid)
            if _dist >= 0 and _gid[0] >= 0:
                _hm[_i,_j] = 2.0 - _dist; _nval += 1
    mpc.setHeightmap(_hm, _hres, _hox, _hoy)
    if _os.environ.get("HM_SEARCH"): mpc.hm_search = float(_os.environ.get("HM_SEARCH"))
    print("[HEIGHTMAP] %dx%d res%.2f valid=%d/%d elev %.2f~%.2f 주입"
          % (_hnx,_hny,_hres,_nval,_hnx*_hny, np.nanmin(_hm) if _nval else 0, np.nanmax(_hm) if _nval else 0), flush=True)

q_meas, v_meas = device.measureState()
x_measured = np.concatenate([q_meas, v_meas])

device.showQuadrupedFeet(
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("FL_foot")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("FR_foot")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("HL_foot")),
    mpc.getDataHandler().getFootPose(mpc.getModelHandler().getFootNb("HR_foot")),
)

force_FL = []
force_FR = []
force_RL = []
force_RR = []
FL_measured = []
FR_measured = []
RL_measured = []
RR_measured = []
FL_references = []
FR_references = []
RL_references = []
RR_references = []
x_multibody = []
u_multibody = []
u_riccati = []
com_measured = []
solve_time = []
L_measured = []

v = np.zeros(6); v[0]=float(_os.environ.get("VX","0.0"))  # 전진속도 명령(env)
v[0] = float(_os.environ.get("VX","0.2"))
mpc.velocity_base = v
device.cmd_v = v                      # 뷰어 오버레이(cmd vx/vy/wz) 표시용
import numpy as _npd
_fell=False
print("[MJ] velocity_base 명령 =", list(v))
_CMDFILE=_os.environ.get("CMDFILE")      # GUI(teleop_gui) JSON 명령 채널 → velocity_base (A/C와 동일 연동)
_STATE_PUB=_os.environ.get("STATE_PUB")  # 상태 발행 → GUI IMU·actuator 모니터 패널
_JN=[_mj.mj_id2name(device.m,_mj.mjtObj.mjOBJ_JOINT,_j).replace('_joint','')
     for _j in range(device.m.njnt) if device.m.jnt_type[_j]!=_mj.mjtJoint.mjJNT_FREE]
_DECIM=int(_os.environ.get("MPC_DECIM","1")); _pk=0   # ★비동기 재계획: DECIM틱마다만 OCP solve, 사이엔 stale plan advance(실효 throughput ×DECIM)
_LOG=bool(_os.environ.get("LOG_TRAJ"))                # 진단 궤적 로깅(무거운 C++ getter ×10/스텝) — 기본 off=속도
for step in range(int(_os.environ.get("STEPS","300"))):
    if _CMDFILE and step % 5 == 0:        # GUI JSON 채널 소비(20Hz) → v/vy/w 직접 반영(OCP가 내부적으로 부드럽게 추종)
        try:
            import json as _json
            with open(_CMDFILE) as _f: _cj=_json.load(_f)
            v[0]=float(_cj.get('v',v[0])); v[1]=float(_cj.get('vy',0.0)); v[5]=float(_cj.get('w',0.0))
        except Exception: pass
    mpc.velocity_base = v; device.cmd_v = v
    if step % 30 == 0 or step == 299:
        _z=device.d.qpos[2]; _x=device.d.qpos[0]; _y=device.d.qpos[1]
        _t=_npd.degrees(_npd.arccos(_npd.clip(1-2*(device.d.qpos[4]**2+device.d.qpos[5]**2),-1,1)))
        print("[MJ] step=%3d t=%.2f base_z=%.3f x=%+.3f y=%+.3f tilt=%.1f"%(step,step*0.01,_z,_x,_y,_t),flush=True)
        _nq=model_handler.getModel().nq
        _ocpvx=mpc.xs[1][_nq] if len(mpc.xs)>1 else 0.0   # OCP 계획 base 전진속도(pin local)
        _measvx=v_meas[0]                                  # 측정 base 전진속도(pin local)
        print("    OCP계획vx=%.3f 측정vx=%.3f (명령 %.2f)"%(_ocpvx,_measvx,v[0]),flush=True)
        if _os.environ.get("DIAG"):
            def _rp(qw,qx,qy,qz):   # roll,pitch [deg]
                r=_npd.degrees(_npd.arctan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)))
                p=_npd.degrees(_npd.arcsin(_npd.clip(2*(qw*qy-qz*qx),-1,1)))
                return r,p
            _mw,_mx,_my,_mz=device.d.qpos[3:7]          # 측정 quat [w,x,y,z]
            _mr,_mp=_rp(_mw,_mx,_my,_mz)
            _xs0=mpc.xs[0]; _r0,_p0=_rp(_xs0[6],_xs0[3],_xs0[4],_xs0[5])         # 초기(=측정) pin quat[x,y,z,w]
            _xsT=mpc.xs[-1]; _rT,_pT=_rp(_xsT[6],_xsT[3],_xsT[4],_xsT[5])         # ★terminal(OCP가 가려는 목표)
            _nqv=model_handler.getModel().nq
            _vx0=_xs0[_nqv]; _vxT=_xsT[_nqv]; _wy0=_xs0[_nqv+4]; _wyT=_xsT[_nqv+4]  # base vx, pitch각속도(local)
            print("    [DIAG] 초기 pitch=%+.1f roll=%+.1f vx=%+.2f | ★terminal pitch=%+.1f roll=%+.1f vx=%+.2f wy=%+.2f"
                  %(_p0,_r0,_vx0,_pT,_rT,_vxT,_wyT),flush=True)
        if _z<0.15:
            print("[MJ] ❌ 전복 @%.2fs"%(step*0.01)); _fell=True; break
    # print("Time " + str(step))
    if step % _DECIM == 0:                  # ★비동기 재계획: DECIM틱마다만 OCP solve
        start = time.time(); mpc.iterate(x_measured); solve_time.append(time.time()-start); _pk = 0
    _pkc = min(_pk, T - 2)                   # stale plan advance 인덱스(재사용)

    if _LOG:                                 # 진단 궤적(무거운 C++ getter ×10/스텝) — 기본 off
        force_FL.append(mpc.us[_pkc][:3]); force_FR.append(mpc.us[_pkc][3:6])
        force_RL.append(mpc.us[_pkc][6:9]); force_RR.append(mpc.us[_pkc][9:12])
        _gd=mpc.getDataHandler(); _gm=mpc.getModelHandler()
        FL_measured.append(_gd.getFootPose(_gm.getFootNb("FL_foot")).translation)
        FR_measured.append(_gd.getFootPose(_gm.getFootNb("FR_foot")).translation)
        RL_measured.append(_gd.getFootPose(_gm.getFootNb("HL_foot")).translation)
        RR_measured.append(_gd.getFootPose(_gm.getFootNb("HR_foot")).translation)
        FL_references.append(mpc.getReferencePose(_pkc,"FL_foot").translation); FR_references.append(mpc.getReferencePose(_pkc,"FR_foot").translation)
        RL_references.append(mpc.getReferencePose(_pkc,"HL_foot").translation); RR_references.append(mpc.getReferencePose(_pkc,"HR_foot").translation)
        com_measured.append(_gd.getData().com[0].copy()); L_measured.append(_gd.getData().hg.angular.copy())

    a0 = mpc.getStateDerivative(_pkc)[nv:].copy()
    a1 = mpc.getStateDerivative(_pkc + 1)[nv:].copy()

    a0[6:] = mpc.us[_pkc][nk * force_size :]
    a1[6:] = mpc.us[_pkc + 1][nk * force_size :]
    forces0 = mpc.us[_pkc][: nk * force_size]
    forces1 = mpc.us[_pkc + 1][: nk * force_size]
    contact_states = mpc.ocp_handler.getContactState(_pkc)

    forces = [forces0, forces1]
    ddqs = [a0, a1]
    xss = [mpc.xs[_pkc], mpc.xs[_pkc + 1]]
    uss = [mpc.us[_pkc], mpc.us[_pkc + 1]]

    for sub_step in range(N_simu):
        t = step * dt_mpc + sub_step * dt_simu

        delay = sub_step / float(N_simu) * dt_mpc
        xs_interp = interpolator.interpolateState(delay, dt_mpc, xss)
        acc_interp = interpolator.interpolateLinear(delay, dt_mpc, ddqs)
        force_interp = interpolator.interpolateLinear(delay, dt_mpc, forces).reshape(
            (4, 3)
        )

        q_interp = xs_interp[: mpc.getModelHandler().getModel().nq]
        v_interp = xs_interp[mpc.getModelHandler().getModel().nq :]
        force_interp = [force_interp[i, :] for i in range(4)]

        q_meas, v_meas = device.measureState()
        x_measured = np.concatenate([q_meas, v_meas])

        kino_ID.setTarget(q_interp, v_interp, acc_interp, contact_states, force_interp)
        tau_cmd = kino_ID.solve(t, q_meas, v_meas)

        device.execute(tau_cmd)
        u_multibody.append(copy.deepcopy(tau_cmd))
        x_multibody.append(x_measured)
    if _STATE_PUB and step % 3 == 0:      # 상태 발행(~33Hz) → GUI IMU·actuator 패널 (C와 동일 스키마)
        try:
            import json as _j2
            _dd=device.d
            _qw2,_qx2,_qy2,_qz2=_dd.qpos[3],_dd.qpos[4],_dd.qpos[5],_dd.qpos[6]
            _roll=_npd.degrees(_npd.arctan2(2*(_qw2*_qx2+_qy2*_qz2),1-2*(_qx2*_qx2+_qy2*_qy2)))
            _pitch=_npd.degrees(_npd.arcsin(_npd.clip(2*(_qw2*_qy2-_qz2*_qx2),-1,1)))
            _yaw2=_npd.degrees(_npd.arctan2(2*(_qw2*_qz2+_qx2*_qy2),1-2*(_qy2*_qy2+_qz2*_qz2)))
            _st={'mode':'move','base_z':float(_dd.qpos[2]),'t':step*dt_mpc,
                 'rpy':[float(_roll),float(_pitch),float(_yaw2)],'gyro':[float(_g) for _g in _dd.qvel[3:6]],
                 'names':_JN,'q':[float(_x) for _x in _dd.qpos[7:7+device.nu]],
                 'dq':[float(_x) for _x in _dd.qvel[6:6+device.nu]],'tau':[float(_x) for _x in _dd.ctrl[:device.nu]],
                 'cmd':[float(v[0]),float(v[1]),float(v[5])]}
            _tmp2=_STATE_PUB+'.tmp'
            with open(_tmp2,'w') as _f2: _j2.dump(_st,_f2)
            _os.replace(_tmp2,_STATE_PUB)
        except Exception: pass
    _pk += 1                                 # ★stale plan 인덱스 advance(다음 재계획까지)


force_FL = np.array(force_FL)
force_FR = np.array(force_FR)
force_RL = np.array(force_RL)
force_RR = np.array(force_RR)
solve_time = np.array(solve_time)
if len(solve_time):
    print("[KINO_TIMING] mpc.iterate 평균=%.2fms 최대=%.2fms (%.0fHz 가능)"
          % (solve_time.mean()*1000, solve_time.max()*1000, 1000.0/(solve_time.mean()*1000)), flush=True)
FL_measured = np.array(FL_measured)
FR_measured = np.array(FR_measured)
RL_measured = np.array(RL_measured)
RR_measured = np.array(RR_measured)
FL_references = np.array(FL_references)
FR_references = np.array(FR_references)
RL_references = np.array(RL_references)
RR_references = np.array(RR_references)
com_measured = np.array(com_measured)
L_measured = np.array(L_measured)

""" save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "kinodynamics") """
