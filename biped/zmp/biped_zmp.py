#!/usr/bin/env python3
"""biped ZMP 보행기 — DCM(Divergent Component of Motion) 기반 LIPM 워커.

D1(centroidal MPC)는 불안정한 단일지지 스텝을 "회피"해 발을 안 들었음.
ZMP 워커는 footstep을 강제 커밋하고 ZMP-일관 CoM 궤적을 생성 → 발을 실제로 뗌.

파이프라인:
  1) footstep 계획(전진, 좌우 교대) → 각 스텝의 지지발 ZMP
  2) DCM 역recursion으로 ξ(발산성분) 계획 → CoM 전방적분(ZMP-일관)
  3) swing 발 Bezier(apex)
  4) 매 틱 전신 IK(base=CoM·양발=목표) → 관절각
  5) PD 토크 추종 + base-tilt 안정화(발목 ZMP 피드백)

모델: pinocchio(biped_gen.urdf)=IK · MuJoCo(biped_boxfoot.mjcf)=sim(박스발 CoP).
실행: DISPLAY=:0 VIEW=1 python biped_zmp.py   /  헤드리스는 VIEW=0
"""
import os as _os, numpy as np, pinocchio as pin, mujoco as mj

URDF = "/home/jsh/문서/jsh/simulation/biped/ocp/biped_gen.urdf"
MJCF = _os.environ.get("MJCF", "/home/jsh/문서/jsh/simulation/biped/biped_boxfoot.mjcf")
LEGS = ["HL", "HR"]
FOOT_R = 0.036

# ---------------- 파라미터 ----------------
g = 9.81
Z_C     = float(_os.environ.get("ZC", "0.45"))     # CoM 높이(LIPM)
T_SS    = float(_os.environ.get("TSS", "0.7"))     # 단일지지 시간
T_DS    = float(_os.environ.get("TDS", "0.15"))    # 양발지지 시간
STEP    = float(_os.environ.get("STEP", "0.06"))   # 전진 보폭
HSTANCE = float(_os.environ.get("HSTANCE", "0.115"))  # 발 좌우 오프셋(hip y)
N_STEP  = int(_os.environ.get("NSTEP", "12"))
APEX    = float(_os.environ.get("APEX", "0.05"))   # swing 발 높이
DT      = 0.001                                     # sim dt
W_LIP   = np.sqrt(g / Z_C)                          # LIPM 고유주파수 ω

# 안정화 게인(발목 ZMP/tilt 피드백)
KP_PITCH = float(_os.environ.get("KP_PITCH", "0.0"))  # base pitch→ZMP shift
KP_ROLL  = float(_os.environ.get("KP_ROLL",  "0.0"))


# ---------------- pinocchio 모델(IK) ----------------
def build_pin():
    M = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    q = pin.neutral(M); q[2] = 0.45
    q[7:11] = [0.0, 0.25, -0.50, -1.14626]; q[11:15] = [0.0, 0.25, -0.50, -1.14626]
    d = M.createData(); pin.forwardKinematics(M, d, q); pin.updateFramePlacements(M, d)
    # 발 sole 프레임 = heel~toe 중점(접지면 중심)
    for L in LEGS:
        h = M.getFrameId(f"{L}_foot_link"); t = M.getFrameId(f"{L}_foot_contact_link")
        hp = d.oMf[h].translation; tp = d.oMf[t].translation
        mid = 0.5 * (hp + tp)
        pj = M.frames[h].parentJoint; pf = M.frames[h].parentFrame
        sole_local = d.oMi[pj].inverse() * pin.SE3(np.eye(3), mid)
        M.addFrame(pin.Frame(f"{L}_sole", pj, pf, sole_local, pin.FrameType.OP_FRAME))
    d = M.createData()   # 프레임 추가 후 data 재생성
    # 발이 지면(FOOT_R)에 오도록 base z 보정
    pin.forwardKinematics(M, d, q); pin.updateFramePlacements(M, d)
    zmin = min(d.oMf[M.getFrameId(f"{L}_sole")].translation[2] for L in LEGS)
    q[2] += FOOT_R - zmin
    return M, q


_Q_NOM = np.array([0.0, 0.25, -0.50, -1.14626, 0.0, 0.25, -0.50, -1.14626])  # 자세 정규화 기준

def ik_legs(M, d, q, base_pos, base_quat_xyzw, foot_tgt):
    """base(측정 고정)와 양발 sole 위치 목표 IK. 감쇠최소자승(DLS)+자세정규화로 null space 고정→매끄러운 유일해."""
    q = q.copy()
    q[0:3] = base_pos; q[3:7] = base_quat_xyzw
    R_flat = np.eye(3)
    W_ORI = float(_os.environ.get("W_ORI", "0.0"))
    LAM = float(_os.environ.get("IK_LAM", "0.05"))    # DLS 감쇠
    KPOS = float(_os.environ.get("IK_POST", "0.02"))  # 자세정규화(null space)
    for _ in range(60):
        pin.forwardKinematics(M, d, q); pin.updateFramePlacements(M, d); pin.computeJointJacobians(M, d, q)
        rows = 6 if W_ORI > 0 else 3
        err = np.zeros(2*rows); J = np.zeros((2*rows, 8))
        for i, L in enumerate(LEGS):
            fid = M.getFrameId(f"{L}_sole"); oMf = d.oMf[fid]
            err[rows*i:rows*i+3] = foot_tgt[L] - oMf.translation
            Jf = pin.getFrameJacobian(M, d, fid, pin.LOCAL_WORLD_ALIGNED)[:, 6:]  # 다리 8열
            J[rows*i:rows*i+3] = Jf[:3]
            if W_ORI > 0:
                err[rows*i+3:rows*i+6] = W_ORI * pin.log3(R_flat @ oMf.rotation.T)
                J[rows*i+3:rows*i+6] = W_ORI * Jf[3:]
        if np.linalg.norm(err) < 1e-7: break
        # DLS: dq = Jᵀ(JJᵀ+λ²I)⁻¹ err + null-space 자세정규화
        JJt = J @ J.T + (LAM**2) * np.eye(2*rows)
        dq_task = J.T @ np.linalg.solve(JJt, err)
        # null space에서 q_nom 쪽으로(자세 고정)
        Jpinv = J.T @ np.linalg.inv(JJt)
        Nproj = np.eye(8) - Jpinv @ J
        dq_post = Nproj @ (KPOS * (_Q_NOM - q[7:15]))
        dq = np.zeros(M.nv); dq[6:] = dq_task + dq_post
        q = pin.integrate(M, q, dq)
        q[0:3] = base_pos; q[3:7] = base_quat_xyzw
    return q


T_INIT = float(_os.environ.get("TINIT", "0.8"))   # 초기 체중이동(중앙→첫 지지발) 구간
T_END  = float(_os.environ.get("TEND", "1.0"))    # 종료 정착 구간

# ---------------- phase 계획(footstep+ZMP) ----------------
def plan_phases(foot0):
    """phase 목록 생성. 각 phase=dict(zmp,T,swing,sw_from,sw_to,ss_frac).
       ss_frac=단일지지 비율(1.0=전부SS·0=전부DS)."""
    foot = {L: foot0[L].copy() for L in LEGS}
    phases = []
    swing = "HL"
    first_stance = "HR"
    # 0) 초기: 중앙→첫 지지발(HR) 위로 체중이동. 양발 접지(swing 없음).
    phases.append(dict(zmp=foot[first_stance][:2].copy(), T=T_INIT, swing=None,
                       sw_from=None, sw_to=None))
    # 1..N) 각 스텝: ZMP=지지발, swing 전진
    for k in range(N_STEP):
        stance = "HR" if swing == "HL" else "HL"
        sw_from = foot[swing].copy()
        sw_to = foot[swing].copy(); sw_to[0] += STEP
        if k == N_STEP - 1:
            sw_to[0] = foot[stance][0]        # 마지막=지지발 옆에 모으기
        phases.append(dict(zmp=foot[stance][:2].copy(), T=T_SS+T_DS, swing=swing,
                           sw_from=sw_from, sw_to=sw_to, n_ss=int(round(T_SS/DT))))
        foot[swing] = sw_to
        swing = stance
    # 종료: 양발 중앙 정착
    center = 0.5*(foot["HL"][:2] + foot["HR"][:2])
    phases.append(dict(zmp=center, T=T_END, swing=None, sw_from=None, sw_to=None))
    return phases


def plan_dcm(phases):
    """phase별 ξ_ini 역recursion. ξ_eos(마지막)=마지막 ZMP."""
    n = len(phases)
    xi_ini = [None]*n
    xi_eos = phases[-1]["zmp"].copy()
    for i in range(n-1, -1, -1):
        p = phases[i]["zmp"]; T = phases[i]["T"]
        xi_ini[i] = p + (xi_eos - p) * np.exp(-W_LIP * T)
        xi_eos = xi_ini[i]
    return xi_ini


# ---------------- 궤적 생성(CoM·swing 발) ----------------
def bezier_swing(p0, p1, s, apex):
    """s∈[0,1] swing 보간, 중점 apex 높이."""
    p = (1-s)*p0 + s*p1
    p = p.copy()
    p[2] = FOOT_R + apex * (1 - (2*s - 1)**2)   # 포물선(0→apex→0)
    return p


def generate():
    M, q0 = build_pin()
    # 초기 발 sole 위치 = q0 FK (하드코딩 아님)
    d = M.createData(); pin.forwardKinematics(M, d, q0); pin.updateFramePlacements(M, d)
    foot0 = {L: d.oMf[M.getFrameId(f"{L}_sole")].translation.copy() for L in LEGS}
    for L in LEGS: foot0[L][2] = FOOT_R
    phases = plan_phases(foot0)
    xi_ini = plan_dcm(phases)
    # CoM 초기 = 양발 중점
    com = 0.5*(foot0["HL"][:2] + foot0["HR"][:2])
    traj = []
    foot = {L: foot0[L].copy() for L in LEGS}
    for i, ph in enumerate(phases):
        p = ph["zmp"]; xi0 = xi_ini[i]; T = ph["T"]; sw = ph["swing"]
        n_ph = int(round(T / DT))
        n_ss = ph.get("n_ss", 0)
        for k in range(n_ph):
            t = k * DT
            xi = p + (xi0 - p) * np.exp(W_LIP * t)
            com = com + DT * W_LIP * (xi - com)
            ss = False
            if sw is not None:
                if k < n_ss:
                    sfrac = k / max(1, n_ss - 1)
                    foot[sw] = bezier_swing(ph["sw_from"], ph["sw_to"], sfrac, APEX)
                    ss = True
                else:
                    foot[sw] = ph["sw_to"].copy(); foot[sw][2] = FOOT_R
            traj.append(dict(com=com.copy(), HL=foot["HL"].copy(), HR=foot["HR"].copy(),
                             zmp=p.copy(), swing=(sw or "-"), ss=ss))
    return M, q0, traj


# ---------------- MuJoCo 실행 ----------------
def main():
    M, q0, traj = generate()
    d_pin = M.createData()
    print(f"[ZMP] ω={W_LIP:.2f} T_SS={T_SS} T_DS={T_DS} STEP={STEP} steps={N_STEP} 틱={len(traj)}", flush=True)

    m = mj.MjModel.from_xml_path(MJCF); m.opt.timestep = DT
    dat = mj.MjData(m)
    # 다리질량 스케일(옵션)
    _lms = float(_os.environ.get("LEG_MASS_SCALE", "1.0"))
    if _lms != 1.0:
        for b in range(m.nbody):
            bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) or ''
            if any(s in bn for s in ('hip', 'thigh', 'calf', 'foot')):
                m.body_mass[b] *= _lms; m.body_inertia[b] *= _lms
        mj.mj_setConst(m, mj.MjData(m))

    # 초기 자세 = q0 (base + 관절). mujoco 관절순 = HL,HR (pin과 동일)
    def set_q(qp):
        dat.qpos[0:3] = qp[0:3]
        x, y, z, w = qp[3:7]; dat.qpos[3:7] = [w, x, y, z]
        dat.qpos[7:15] = qp[7:15]
    set_q(q0); mj.mj_forward(m, dat)
    z_base0 = q0[2]

    nu = m.nu
    KP = float(_os.environ.get("KP", "400")); KD = float(_os.environ.get("KD", "20"))
    # ★위치 서보(암시적=수치안정): force=KP(ctrl-q)-KD·qvel. 부유 base를 stiff 다리서보+planted발로 구동.
    m.actuator_gaintype[:] = int(mj.mjtGain.mjGAIN_FIXED); m.actuator_gainprm[:, 0] = KP
    m.actuator_biastype[:] = int(mj.mjtBias.mjBIAS_AFFINE)
    m.actuator_biasprm[:, 1] = -KP; m.actuator_biasprm[:, 2] = -KD
    viewer = None
    if int(_os.environ.get("VIEW", "0")):
        import mujoco.viewer as mv; viewer = mv.launch_passive(m, dat)

    q_ik = q0.copy()
    STEPS = len(traj)
    fell = -1
    for k in range(STEPS):
        tr = traj[k]
        # ★base를 계획 CoM에 명령(z=z_base0·수평). 위치서보가 다리를 몰아 base를 계획 위치로.
        base_pos = np.array([tr["com"][0], tr["com"][1], z_base0])
        foot_tgt = {"HL": tr["HL"].copy(), "HR": tr["HR"].copy()}
        q_ik = ik_legs(M, d_pin, q_ik, base_pos, np.array([0, 0, 0, 1.0]), foot_tgt)
        dat.ctrl[:] = q_ik[7:15]   # 위치 서보 목표
        mj.mj_step(m, dat)
        w_, x_, y_, z_ = dat.qpos[3:7]
        pitch = np.degrees(np.arcsin(np.clip(2*(w_*y_-z_*x_), -1, 1)))
        roll = np.degrees(np.arctan2(2*(w_*x_+y_*z_), 1-2*(x_*x_+y_*y_)))
        if viewer is not None and k % 8 == 0:
            viewer.sync()
            import time as _t; _t.sleep(DT*8)
        if dat.qpos[2] < 0.25:
            fell = k; break
        if k % 200 == 0:
            print(f"[ZMP] t={k*DT:.2f}s base=({dat.qpos[0]:+.3f},{dat.qpos[1]:+.3f},{dat.qpos[2]:.3f}) "
                  f"pitch={pitch:+.1f} roll={roll:+.1f} swing={tr['swing']} ss={tr['ss']}", flush=True)
    if fell >= 0:
        print(f"[ZMP] ★낙상 @t={fell*DT:.2f}s base_z={dat.qpos[2]:.3f}", flush=True)
    else:
        print(f"[ZMP] ✅ 완주 {STEPS*DT:.1f}s · 전진 base_x={dat.qpos[0]:+.3f}m", flush=True)


if __name__ == "__main__":
    main()
