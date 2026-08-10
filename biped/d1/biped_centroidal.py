"""biped D1 컨트롤러 — simple-mpc KinodynamicsOCP(centroidal MPC) + KinodynamicsID(TSID/WBC).

★quad_centroidal_17dof.py(작동 버전)를 base로 이식(부서진 quad_centroidal.py 아님). biped 차이:
  - 모델=biped_gen.urdf(2다리 HL/HR, 허리X, nu=8). pin순=mjcf순(HL,HR) → 재정렬 불필요.
  - ★평발=addQuadFoot(발4모서리=CoP 폴리곤·force_size=6=6D wrench) → 단일지지 pitch 권한
    (우리 crocoddyl 진단의 online 해법: SRBD 점발이 못하던 pitch를 centroidal+CoP가 계획).
  - 게이트=양발DS ↔ 단일SS 교대(biped).

실행(pixi env):
  PYTHONPATH=/home/jsh/simple-mpc/build/bindings VX=0.1 STEPS=300 \
    /home/jsh/simple-mpc/.pixi/envs/default/bin/python biped_centroidal.py
"""
import os as _os, time, copy
import numpy as np
import mujoco as _mj
import pinocchio as _pin
from simple_mpc import (RobotModelHandler, RobotDataHandler, KinodynamicsOCP,
                        MPC, Interpolator, KinodynamicsID, KinodynamicsIDSettings)

import os as _os
MJCF = _os.environ.get("MJCF", "/home/jsh/문서/jsh/simulation/biped/biped_flatfoot.mjcf")  # ★BOXFOOT=1로 박스발 검증
if _os.environ.get("BOXFOOT"): MJCF = "/home/jsh/문서/jsh/simulation/biped/biped_boxfoot.mjcf"
URDF = "/home/jsh/문서/jsh/simulation/biped/ocp/biped_gen.urdf"
LEGS = ["HL", "HR"]
Q_FLAT = np.array([0.0, 0.25, -0.50, -1.14626])
FOOT_R = 0.036
LFOOT = float(_os.environ.get("LFOOT", "0.08")); WFOOT = float(_os.environ.get("WFOOT", "0.02"))  # 박스발=WFOOT0.045
GEAR = [7., 7., 10.5, 8.]; I_ROTOR = 1e-4; J_DAMP = 0.1; J_FRIC = 0.5


class MujocoRobot:
    """simple-mpc device 인터페이스(MuJoCo). biped=관절순서 pin=mjcf 동일(재정렬X). quat wxyz↔xyzw, lin world→local."""
    def __init__(self, q0, dt_simu, view=False):
        self.m = _mj.MjModel.from_xml_path(MJCF); self.m.opt.timestep = dt_simu
        for j in range(self.m.nu):                    # GEARBOX(반사관성) — 수치안정
            dof = 6 + j; N = GEAR[j % 4]
            self.m.dof_armature[dof] = I_ROTOR*N*N; self.m.dof_damping[dof] = J_DAMP; self.m.dof_frictionloss[dof] = J_FRIC
        _lms = float(_os.environ.get('LEG_MASS_SCALE', '1.0'))   # ★다리 질량/관성 스케일(sim)
        if _lms != 1.0:
            for _b in range(self.m.nbody):
                _bn = _mj.mj_id2name(self.m, _mj.mjtObj.mjOBJ_BODY, _b) or ''
                if any(_s in _bn for _s in ('hip', 'thigh', 'calf', 'foot')):
                    self.m.body_mass[_b] *= _lms; self.m.body_inertia[_b] *= _lms
            _mj.mj_setConst(self.m, _mj.MjData(self.m))
        _bad = float(_os.environ.get('BODY_ADD', '0'))   # ★base(torso) 질량 추가(sim): 역전분포 교정
        if _bad != 0.0:
            _bb = _mj.mj_name2id(self.m, _mj.mjtObj.mjOBJ_BODY, 'torso')
            _m0 = self.m.body_mass[_bb]; _mn = _m0 + _bad
            self.m.body_inertia[_bb] *= (_mn/_m0); self.m.body_mass[_bb] = _mn
            _mj.mj_setConst(self.m, _mj.MjData(self.m))
        self.d = _mj.MjData(self.m); self.nu = self.m.nu; self.viewer = None
        self._set(q0)
        if view:
            import mujoco.viewer as _v; self.viewer = _v.launch_passive(self.m, self.d)
    def _set(self, q):
        self.d.qpos[0:3] = q[0:3]; x, y, z, w = q[3:7]; self.d.qpos[3:7] = [w, x, y, z]
        self.d.qpos[7:7+self.nu] = q[7:7+self.nu]; self.d.qvel[:] = 0.0; _mj.mj_forward(self.m, self.d)
    def initializeJoints(self, q0): self._set(q0)
    def measureState(self):
        d = self.d; qp = np.zeros(self.m.nq); vp = np.zeros(self.m.nv)
        qp[0:3] = d.qpos[0:3]; w, x, y, z = d.qpos[3:7]; qp[3:7] = [x, y, z, w]
        R = np.zeros(9); _mj.mju_quat2Mat(R, d.qpos[3:7]); R = R.reshape(3, 3)
        vp[0:3] = R.T @ d.qvel[0:3]; vp[3:6] = d.qvel[3:6]
        qp[7:] = d.qpos[7:7+self.nu]; vp[6:] = d.qvel[6:6+self.nu]
        return qp, vp
    def execute(self, tau):
        self.d.ctrl[:] = np.asarray(tau).ravel()[:self.nu]; _mj.mj_step(self.m, self.d)
        if self.viewer: self.viewer.sync()
    def changeCamera(self, *a, **k): pass


def build_model():
    M = _pin.buildModelFromUrdf(URDF, _pin.JointModelFreeFlyer())
    q = _pin.neutral(M); q[2] = 0.45; q[7:11] = Q_FLAT; q[11:15] = Q_FLAT
    d = M.createData(); _pin.forwardKinematics(M, d, q); _pin.updateFramePlacements(M, d)
    zmin = min(min(d.oMf[M.getFrameId(f"{L}_foot_link")].translation[2],
                   d.oMf[M.getFrameId(f"{L}_foot_contact_link")].translation[2]) for L in LEGS)
    q[2] += FOOT_R - zmin
    _pin.forwardKinematics(M, d, q); _pin.updateFramePlacements(M, d)
    # ★★크라우치(CoM 낮춤): base를 CROUCH(m) 낮추고 다리 굽혀 발 접지 유지 → 도립진자 발산율 ω=√(g/z_c)↓.
    _cr = float(_os.environ.get("CROUCH", "0.0"))
    if _cr != 0.0:
        dC = M.createData(); _pin.forwardKinematics(M, dC, q); _pin.updateFramePlacements(M, dC)
        ftgt = {L: dC.oMf[M.getFrameId(f"{L}_foot_link")].translation.copy() for L in LEGS}
        for L in LEGS: ftgt[L][2] = FOOT_R
        q[2] -= _cr
        for _ in range(300):
            _pin.forwardKinematics(M, dC, q); _pin.updateFramePlacements(M, dC); _pin.computeJointJacobians(M, dC, q)
            err = np.zeros(6); J = np.zeros((6, M.nv))
            for i, L in enumerate(LEGS):
                fid = M.getFrameId(f"{L}_foot_link")
                err[3*i:3*i+3] = ftgt[L] - dC.oMf[fid].translation
                J[3*i:3*i+3] = _pin.getFrameJacobian(M, dC, fid, _pin.LOCAL_WORLD_ALIGNED)[:3]
            J[:, :6] = 0.0
            if np.linalg.norm(err) < 1e-5: break
            q = _pin.integrate(M, q, 0.5*np.linalg.lstsq(J, err, rcond=None)[0])
        _pin.forwardKinematics(M, d, q); _pin.updateFramePlacements(M, d)
        print(f"[CROUCH] base -{_cr:.3f}m → base_z={q[2]:.3f}", flush=True)
    # ★★발 x 오프셋(엇갈림 stance): HL 발 앞(+s)·HR 발 뒤(-s)로 IK → 양발지지에 전후 지지폭 생성(pitch 안식처).
    #   양발 同x(전후 지지폭=발1개)가 단일지지 pitch 벽의 근본원인 → 엇갈림으로 직접 제거.
    _stag = float(_os.environ.get("STAGGER", "0.0"))
    if _stag != 0.0:
        dS = M.createData(); _pin.forwardKinematics(M, dS, q); _pin.updateFramePlacements(M, dS)
        ftgt = {L: dS.oMf[M.getFrameId(f"{L}_foot_link")].translation.copy() for L in LEGS}
        ftgt["HL"][0] += _stag; ftgt["HR"][0] -= _stag   # HL 앞·HR 뒤
        for L in LEGS: ftgt[L][2] = FOOT_R
        for _ in range(400):
            _pin.forwardKinematics(M, dS, q); _pin.updateFramePlacements(M, dS); _pin.computeJointJacobians(M, dS, q)
            err = np.zeros(6); J = np.zeros((6, M.nv))
            for i, L in enumerate(LEGS):
                fid = M.getFrameId(f"{L}_foot_link")
                err[3*i:3*i+3] = ftgt[L] - dS.oMf[fid].translation
                J[3*i:3*i+3] = _pin.getFrameJacobian(M, dS, fid, _pin.LOCAL_WORLD_ALIGNED)[:3]
            J[:, :6] = 0.0
            if np.linalg.norm(err) < 1e-5: break
            q = _pin.integrate(M, q, 0.5*np.linalg.lstsq(J, err, rcond=None)[0])
        _pin.forwardKinematics(M, d, q); _pin.updateFramePlacements(M, d)
        print(f"[STAGGER] HL 발 +{_stag:.3f}m·HR 발 -{_stag:.3f}m → 전후 지지폭 {2*_stag*100:.0f}cm", flush=True)
    # ★sole 프레임 위치 = heel + SOLE_F·(toe-heel). SOLE_F=0.5=중심, <0.5=heel쪽(파란구 기준·heel main).
    #   heel쪽이면 CoP 폴리곤이 heel~toe 전방 → CoP가 heel쪽 치중(뒤로발라당 저항+전진push, 사용자 통찰).
    _solef = float(_os.environ.get("SOLE_F", "0.5"))
    corners = {}
    for L in LEGS:
        hM = d.oMf[M.getFrameId(f"{L}_foot_link")]; tM = d.oMf[M.getFrameId(f"{L}_foot_contact_link")]
        L2 = np.linalg.norm(tM.translation - hM.translation)   # heel~toe 거리 ~16cm
        sole_p = hM.translation + _solef*(tM.translation - hM.translation); sole_p[2] = FOOT_R
        pj = M.frames[M.getFrameId(f"{L}_foot_link")].parentJoint
        pf = M.frames[M.getFrameId(f"{L}_foot_link")].parentFrame
        sole_local = d.oMi[pj].inverse() * _pin.SE3(np.eye(3), sole_p)
        M.addFrame(_pin.Frame(f"{L}_foot", pj, pf, sole_local, _pin.FrameType.OP_FRAME))
        _lb = float(_os.environ.get("LEXT_B", "0.0"))   # ★heel 뒤 확장(발목 뒤 발=backward pitch CoP 마진, 핵심)
        _lf = float(_os.environ.get("LEXT_F", "0.0"))   # ★toe 앞 확장
        xf = (1-_solef)*L2 + _lf; xb = -_solef*L2 - _lb   # 발 앞(toe)·뒤(heel) 경계(sole 프레임 기준)
        corners[L] = np.array([[xf, WFOOT, 0], [xf, -WFOOT, 0], [xb, WFOOT, 0], [xb, -WFOOT, 0]])
    # ★발목 하한 여유(WBC viability): CoP용 plantarflex가 물리한계(-1.396)에 붙어 WBC가 교란되지 않게
    _am = float(_os.environ.get("ANKLE_MARGIN", "0.15"))
    for _qi in (10, 14):   # HL_foot, HR_foot q index
        M.lowerPositionLimit[_qi] -= _am
    # ★★전방 lean 바이어스(사용자 아이디어): backward 낙하성향을 forward로 상쇄 → 제어된 전방낙하 보행.
    #   base를 PITCH_BIAS(rad) 만큼 앞으로 기울인 참조자세로 재-IK(발은 지면 유지). +값=nose-down 전방.
    _pb = float(_os.environ.get("PITCH_BIAS", "0.0"))
    if _pb != 0.0:
        q[3:7] = [0.0, np.sin(_pb/2), 0.0, np.cos(_pb/2)]   # base pitch tilt (xyzw)
        # 발을 다시 지면(z=FOOT_R)·전방오프셋 유지하도록 다리 IK(base 고정)
        d2 = M.createData()
        ftgt = {}
        _pin.forwardKinematics(M, d2, q); _pin.updateFramePlacements(M, d2)
        for L in LEGS:  # 현재 발 xy 유지, z=FOOT_R 목표
            p = d2.oMf[M.getFrameId(f"{L}_foot")].translation.copy(); p[2] = FOOT_R; ftgt[L] = p
        for _ in range(200):
            _pin.forwardKinematics(M, d2, q); _pin.updateFramePlacements(M, d2); _pin.computeJointJacobians(M, d2, q)
            err = np.zeros(6); J = np.zeros((6, M.nv))
            for i, L in enumerate(LEGS):
                err[3*i:3*i+3] = ftgt[L] - d2.oMf[M.getFrameId(f"{L}_foot")].translation
                J[3*i:3*i+3] = _pin.getFrameJacobian(M, d2, M.getFrameId(f"{L}_foot"), _pin.LOCAL_WORLD_ALIGNED)[:3]
            J[:, :6] = 0.0
            if np.linalg.norm(err) < 1e-5: break
            q = _pin.integrate(M, q, 0.5*np.linalg.lstsq(J, err, rcond=None)[0])
            q[3:7] = [0.0, np.sin(_pb/2), 0.0, np.cos(_pb/2)]   # base 고정
        print(f"[LEAN] PITCH_BIAS={_pb:.3f}rad({np.degrees(_pb):.1f}°) 전방 lean 참조", flush=True)
    _lms = float(_os.environ.get('LEG_MASS_SCALE', '1.0'))   # ★다리 질량/관성 스케일(OCP 모델, joint2..=다리링크)
    if _lms != 1.0:
        for _ji in range(2, M.njoints):
            _I = M.inertias[_ji]
            M.inertias[_ji] = _pin.Inertia(_I.mass*_lms, _I.lever, _I.inertia*_lms)
        print("[LEG_MASS] 다리링크 ×%.2f → 총질량 %.1fkg"
              % (_lms, sum(M.inertias[_j].mass for _j in range(1, M.njoints))), flush=True)
    _bad = float(_os.environ.get('BODY_ADD', '0'))   # ★base(joint1=torso) 질량 추가(OCP 모델)
    if _bad != 0.0:
        _Ib = M.inertias[1]; _mn = _Ib.mass + _bad
        M.inertias[1] = _pin.Inertia(_mn, _Ib.lever, _Ib.inertia*(_mn/_Ib.mass))
        print("[BODY_ADD] base +%.1fkg → 총질량 %.1fkg"
              % (_bad, sum(M.inertias[_j].mass for _j in range(1, M.njoints))), flush=True)
    M.referenceConfigurations["standing"] = q
    return M, q, corners


def main():
    M, qstand, corners = build_model()
    nq, nv = M.nq, M.nv; nu = nv - 6
    mh = RobotModelHandler(M, "standing", "root_joint")
    for L in LEGS:
        mh.addQuadFoot(f"{L}_foot", "root_joint", corners[L])
    dh = RobotDataHandler(mh)
    nk = mh.getFeetNb(); force_size = 6; gravity = np.array([0, 0, -9.81])
    print(f"[biped-D1] nq={nq} nv={nv} nu={nu} 발={nk}(평발6D) 질량={mh.getMass():.2f} base_z={qstand[2]:.3f}", flush=True)

    dt_mpc = 0.01
    # 가중(17dof 구조 · biped 발목 pin idx 3=HL_foot,7=HR_foot 강핀)
    _wbp = float(_os.environ.get("WB_PITCH", "200"))   # ★base pitch 위치추종 가중
    _wbvp = float(_os.environ.get("WBV_PITCH", "10"))   # ★base pitch RATE 댐핑 가중(발산 rate 억제)
    w_basepos = [0, 0, 100, 200, _wbp, 0]
    w_basevel = [float(_os.environ.get("WBVX", "60")), 30, 10, 10, _wbvp, 10]
    _ankw = float(_os.environ.get("ANKLE_W", "50")); _ankdw = float(_os.environ.get("ANKLE_DW", "5"))
    _wlp = [1.0]*nu; _wlv = [0.1]*nu
    for _ia in (3, 7): _wlp[_ia] = _ankw; _wlv[_ia] = _ankdw
    w_x = np.diag(np.array(w_basepos + _wlp + w_basevel + _wlv))
    w_lf = np.array([0.01]*force_size)
    w_u = np.diag(np.concatenate([w_lf]*nk + [np.ones(nu)*1e-5]))
    _wcap = float(_os.environ.get("WCENT_ANG_P", "0.1"))
    w_cent = np.diag(np.array([0., 0., 1, _wcap, _wcap, 10]))
    w_centder = np.diag(np.array([0., 0., 0., _wcap, _wcap, 0.1]))
    _wfr = float(_os.environ.get("W_FRAME", "2000"))   # ★swing 발 추종 가중(강제로 발 들게)
    problem_conf = dict(timestep=dt_mpc, w_x=w_x, w_u=w_u, w_cent=w_cent, w_centder=w_centder,
                        gravity=gravity, force_size=force_size, w_frame=np.eye(6)*_wfr,
                        qmin=M.lowerPositionLimit[7:], qmax=M.upperPositionLimit[7:],
                        mu=0.8, Lfoot=LFOOT, Wfoot=WFOOT, kinematics_limits=True,
                        force_cone=True, land_cstr=False)
    T = int(_os.environ.get("HORIZON", "40"))
    ocp = KinodynamicsOCP(problem_conf, mh)
    ocp.createProblem(mh.getReferenceState(), T, force_size, gravity[2], False)

    T_ds = int(_os.environ.get("TDS", "8")); T_ss = int(_os.environ.get("TSS", "16"))
    mpc_conf = dict(support_force=-mh.getMass()*gravity[2], TOL=1e-4,
                    mu_init=float(_os.environ.get("MU_INIT", "1e-8")), max_iters=int(_os.environ.get("MAXITER", "1")),
                    num_threads=8, swing_apex=float(_os.environ.get("APEX", "0.05")),
                    T_fly=T_ss, T_contact=T_ds, timestep=dt_mpc,
                    # ★★반응형 발배치(step planner): capture-point로 swing발 이동=넘어지는 CoM 잡기.
                    #   biped 단일지지엔 필수(0이면 발이 못잡아 낙상). KCAP=capture게인·ALIP·PREDFOOT.
                    capture_gain=float(_os.environ.get("KCAP", "0.1")),
                    alip_gain=float(_os.environ.get("ALIP", "0.0")),
                    predict_foot=float(_os.environ.get("PREDFOOT", "0.0")),
                    w_foot_ref=float(_os.environ.get("W_FOOT_REF", "0.0")))
    mpc = MPC(mpc_conf, ocp)
    both = {"HL_foot": True, "HR_foot": True}
    swHL = {"HL_foot": False, "HR_foot": True}; swHR = {"HL_foot": True, "HR_foot": False}
    if _os.environ.get("STAND"):
        phases = [both]*(2*(T_ds+T_ss))
    else:
        phases = [both]*T_ds + [swHL]*T_ss + [both]*T_ds + [swHR]*T_ss
    mpc.generateCycleHorizon(phases)

    N_simu = 10; dt_simu = dt_mpc / N_simu
    interpolator = Interpolator(mh.getModel())
    st = KinodynamicsIDSettings()
    st.kp_base = 7.0; st.kp_posture = 10.0; st.kp_contact = 10.0
    st.w_base = 100.0; st.w_posture = 1.0; st.w_contact_force = 1.0; st.w_contact_motion = 1.0
    st.friction_coefficient = 0.8
    kino_ID = KinodynamicsID(mh, dt_simu, st)

    device = MujocoRobot(mh.getReferenceState()[:nq], dt_simu, view=bool(int(_os.environ.get("VIEW", "0"))))
    device.initializeJoints(mh.getReferenceState()[:nq])
    q_meas, v_meas = device.measureState(); x_measured = np.concatenate([q_meas, v_meas])

    v = np.zeros(6); v[0] = float(_os.environ.get("VX", "0.1")); mpc.velocity_base = v
    if not _os.environ.get("STAND"):
        mpc.switchToWalk(v)   # ★★스윙 사이클 활성화(recedeWithCycle: now_==WALKING이라야 발 apex 참조 생성). 이거 없으면 발 안뜸.
    print(f"[biped-D1] velocity_base={list(v)} STAND={_os.environ.get('STAND','off')} switchToWalk={'on' if not _os.environ.get('STAND') else 'off'}", flush=True)
    solve_time = []
    import mujoco as _mjm
    _bidHL = _mjm.mj_name2id(device.m, _mjm.mjtObj.mjOBJ_BODY, 'HL_foot')
    _bidHR = _mjm.mj_name2id(device.m, _mjm.mjtObj.mjOBJ_BODY, 'HR_foot')
    _z0HL = device.d.xpos[_bidHL][2]; _z0HR = device.d.xpos[_bidHR][2]   # 발 정지 높이
    _liftHL = 0.0; _liftHR = 0.0     # 각 발 최대 지면클리어런스
    _foothist = []
    for step in range(int(_os.environ.get("STEPS", "300"))):
        mpc.velocity_base = v
        start = time.time(); mpc.iterate(x_measured); solve_time.append(time.time()-start)
        _lh = device.d.xpos[_bidHL][2]-_z0HL; _rh = device.d.xpos[_bidHR][2]-_z0HR
        _liftHL = max(_liftHL, _lh); _liftHR = max(_liftHR, _rh)
        _qw,_qx,_qy,_qz = device.d.qpos[3:7]
        _roll = np.degrees(np.arctan2(2*(_qw*_qx+_qy*_qz), 1-2*(_qx*_qx+_qy*_qy)))
        _foothist.append((step, _lh*1000, _rh*1000, device.d.qpos[2],
                          np.degrees(np.arcsin(np.clip(2*(_qw*_qy-_qz*_qx),-1,1))), _roll))
        if step % 30 == 0:
            z = device.d.qpos[2]; x = device.d.qpos[0]
            qw, qx, qy, qz = device.d.qpos[3:7]
            pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy-qz*qx), -1, 1)))
            ocpvx = mpc.xs[1][nq] if len(mpc.xs) > 1 else 0.0   # OCP 계획 base 전진속도(pin local)
            print(f"[MJ] step={step:3d} base_z={z:.3f} x={x:+.3f} pitch={pitch:+.1f} vx_meas={v_meas[0]:+.2f} footlift(HL,HR)=({_lh*1000:+.0f},{_rh*1000:+.0f})mm", flush=True)
        a0 = mpc.getStateDerivative(0)[nv:].copy(); a1 = mpc.getStateDerivative(1)[nv:].copy()
        a0[6:] = mpc.us[0][nk*force_size:]; a1[6:] = mpc.us[1][nk*force_size:]
        forces0 = mpc.us[0][:nk*force_size]; forces1 = mpc.us[1][:nk*force_size]
        contact_states = mpc.ocp_handler.getContactState(0)
        if _os.environ.get("DIAG") and step == int(_os.environ.get("DIAG", "20")):
            dD = M.createData()
            fidHL = M.getFrameId("HL_foot"); fidHR = M.getFrameId("HR_foot")
            print(f"[DIAG s{step}] contact_states(k0)={contact_states}", flush=True)
            print(f"[DIAG] OCP 계획 발 z (horizon knot: HL,HR mm, 접촉스케줄):", flush=True)
            for k in range(0, len(mpc.xs), 3):
                qk = mpc.xs[k][:nq]
                _pin.forwardKinematics(M, dD, qk); _pin.updateFramePlacements(M, dD)
                zhl = dD.oMf[fidHL].translation[2]*1000; zhr = dD.oMf[fidHR].translation[2]*1000
                try:
                    rhl = mpc.getReferencePose(k, "HL_foot").translation[2]*1000
                    rhr = mpc.getReferencePose(k, "HR_foot").translation[2]*1000
                except Exception as e:
                    rhl = rhr = -1
                print(f"    k{k:2d}: 계획HL={zhl:6.1f} HR={zhr:6.1f} | 참조HL={rhl:6.1f} HR={rhr:6.1f}", flush=True)
        forces = [forces0, forces1]; ddqs = [a0, a1]
        xss = [mpc.xs[0], mpc.xs[1]]
        for sub in range(N_simu):
            t = step*dt_mpc + sub*dt_simu; delay = sub/float(N_simu)*dt_mpc
            xs_i = interpolator.interpolateState(delay, dt_mpc, xss)
            acc_i = interpolator.interpolateLinear(delay, dt_mpc, ddqs)
            frc_i = interpolator.interpolateLinear(delay, dt_mpc, forces).reshape((nk, force_size))
            q_i = xs_i[:nq]; v_i = xs_i[nq:]; frc_i = [frc_i[i, :] for i in range(nk)]
            q_meas, v_meas = device.measureState(); x_measured = np.concatenate([q_meas, v_meas])
            kino_ID.setTarget(q_i, v_i, acc_i, contact_states, frc_i)
            tau = kino_ID.solve(t, q_meas, v_meas)
            device.execute(tau)
        if device.d.qpos[2] < 0.25:
            print(f"[MJ] 낙상 @step{step} base_z={device.d.qpos[2]:.3f}", flush=True); break
    st_arr = np.array(solve_time)
    print(f"\n최종 base_x={device.d.qpos[0]:+.3f} base_z={device.d.qpos[2]:.3f} 최대발클리어런스(HL,HR)=({_liftHL*1000:.0f},{_liftHR*1000:.0f})mm", flush=True)
    if _os.environ.get("FOOTCSV"):
        with open(_os.environ["FOOTCSV"], "w") as _f:
            _f.write("step,HL_mm,HR_mm,base_z,pitch_deg,roll_deg\n")
            for _r in _foothist: _f.write("%d,%.1f,%.1f,%.4f,%.2f,%.2f\n" % _r)
        print(f"[FOOTCSV] {len(_foothist)}행 저장 → {_os.environ['FOOTCSV']}", flush=True)
    if len(st_arr): print(f"[TIMING] mpc.iterate 평균={st_arr.mean()*1000:.1f}ms ({1000/(st_arr.mean()*1000):.0f}Hz)", flush=True)


if __name__ == "__main__":
    main()
