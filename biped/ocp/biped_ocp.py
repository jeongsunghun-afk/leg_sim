"""Biped whole-body walking OCP (Crocoddyl) — 2점 평발 보행. [2026-07-21]

★★★ 핵심 검증 성공(open-loop): 호라이즌 OCP가 flat 단일지지 backward-pitch를 잡는다.
  ✅ actuation: pinocchio가 URDF(mjcf_to_urdf.py=질량13.571·FK 3e-7 일치) 로드
          → crocoddyl 내장 ActuationModelFloatingBase nu=8 정상(MJCF 직접로드는 오인식 nu=13).
  ✅ ★★근본 버그였던 것 = **stance 접촉을 2×ContactModel3D(heel+toe)로 한 것** → ffeas 안닫힘
          (앞서 "연구급 난제/접촉스위칭"이라 한 건 오진). **정석 = ContactModel6D(sole 1개,
          6-DOF wrench=CoP)** → 2점버그 회피 + 평발 CoP로 pitch 제어. (진단: 1점 toe만도 ffeas=0.)
  ✅ **open-loop 단발 solve 성공: ffeas=0.00·base +0.117전진·max|pitch|=2.3°(수평)·tau126.**
          = 우리 순간 WBIC가 못잡던 flat 단일지지 pitch를 호라이즌 OCP가 잡음 = 검증 완료.
  ⚠️ closed-loop MPC(biped_mpc.py): 피드백게인 K 추가로 falls 6→3·전진 있으나 아직 불안정
          (z 바운스·pitch spike). 원인 = **OCP의 6D 접촉(발 완전용접) vs sim의 2-구 접촉 모델 갭**.
  ★다음: ①OCP 접촉모델을 sim에 맞추기(구 접촉/컴플라이언트) 또는 ②WBC 층 추가(OCP=계획,
          WBC=sim 접촉으로 실현, D1/simple-mpc 구조) ③MPC 튜닝(호라이즌·replan·비용). = 바운드된 튜닝.
  ★교훈: 2×3D 접촉이 FDDP 수렴 깨는 버그였음(같은 강체 발). 평발=ContactModel6D+WrenchCone 정석.



목적: 우리 순간 WBIC가 못 잡는 flat 단일지지 backward-pitch 토플을,
**호라이즌 전신 OCP**(각운동량·CoP를 미래까지 계획)가 푸는지 검증한다.

구조는 ci_mpc/ocp_fixed.py(사족 trot OCP)를 biped로 옮긴 것:
  - 모델 = biped_flatfoot.mjcf를 pinocchio가 직접 로드(MJCF↔sim 완전 일치, URDF 불필요).
  - flat 발 = heel(foot_link) + toe(foot_contact_link) 두 점을 각각 ContactModel3D로 지지
    → OCP가 heel/toe 힘분배(=전후 CoP)를 호라이즌으로 계획 → pitch 협조(우리 WBIC엔 없던 것).
  - 게이트 = 교대 단일지지(HL stance/HR swing ↔ 반대) + 짧은 양발지지(DS) 전환.
  - swing 발은 heel·toe 목표를 함께 추종(평발 수평 착지 유지).
  - Box-FDDP(토크 한계 준수).
"""
import numpy as np
import pinocchio as pin
import crocoddyl
import warnings
warnings.filterwarnings("ignore")

MJCF = "/home/jsh/문서/jsh/simulation/biped/biped_flatfoot.mjcf"
URDF = "/home/jsh/문서/jsh/simulation/biped/ocp/biped_gen.urdf"   # mjcf_to_urdf.py 생성(질량/FK 일치)
LEGS = ["HL", "HR"]
HEEL = {L: f"{L}_foot_link" for L in LEGS}          # 발목/heel 접촉구 프레임
TOE  = {L: f"{L}_foot_contact_link" for L in LEGS}   # toe 접촉구 프레임
Q_FLAT = np.array([0.0, 0.25, -0.50, -1.14626])      # 평발 home(발목 눕힘), leg당
FOOT_R = 0.036                                        # 접촉구 반지름(지면 접촉 시 프레임 z)
TAU_LIM = np.tile([84.0, 84.0, 126.0, 168.0], 2)     # hip/thigh/calf/foot peak, leg당 ×2


def load_model():
    # ★URDF 경로(crocoddyl 내장 actuation 정상 인식). MJCF 직접로드는 free-flyer 오인식.
    return pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())


class FloatingBaseActuation(crocoddyl.ActuationModelAbstract):  # (미사용, 참고용 보존)
    """floating base 언더액추에이션 [0_6; I_{nv-6}]. crocoddyl 기본 FloatingBase가
    MJCF-로드 free-flyer를 오인식(nu=nv-1)하는 문제 우회 — 직접 구현."""
    def __init__(self, state):
        super().__init__(state, state.nv - 6)
    def calc(self, data, x, u):
        data.tau[:] = 0.0
        data.tau[6:] = u
    def calcDiff(self, data, x, u):
        data.dtau_du[:] = 0.0
        for i in range(self.nu):
            data.dtau_du[6 + i, i] = 1.0
        data.dtau_dx[:] = 0.0        # 상태 독립(FDDP 미분 정합 위해 명시)
    def commands(self, data, x, tau):
        data.u[:] = np.asarray(tau).ravel()[6:6 + self.nu]
    def torqueTransform(self, data, x, tau):
        data.Mtau[:] = 0.0
        for i in range(self.nu):
            data.Mtau[i, 6 + i] = 1.0


class BipedWalkOCP:
    def __init__(self, dt=2.0e-2, foot_r=FOOT_R):
        self.model = load_model()
        self.dt = dt
        self.foot_r = foot_r
        self.w_cone = 0.5                            # 마찰추/wrench콘 가중
        self.two_point = True                        # 3D 접촉 시 heel+toe / False=toe만(진단)
        self.use_6d = True                           # ★stance=ContactModel6D(평발 CoP, 2점버그 회피)
        self.mu = 0.7
        self.heel_fid = {L: self.model.getFrameId(HEEL[L]) for L in LEGS}
        self.toe_fid = {L: self.model.getFrameId(TOE[L]) for L in LEGS}
        self.data = self.model.createData()
        self.q_stand = self._standing_q()
        # ★발중심 sole 프레임 추가(평발시 world-aligned) → ContactModel6D의 접촉/CoP 기준
        pin.forwardKinematics(self.model, self.data, self.q_stand)
        pin.updateFramePlacements(self.model, self.data)
        self.sole_fid = {}
        for L in LEGS:
            hM, tM = self.data.oMf[self.heel_fid[L]], self.data.oMf[self.toe_fid[L]]
            center = 0.5*(hM.translation + tM.translation); center[2] = foot_r
            pj = self.model.frames[self.heel_fid[L]].parentJoint
            sole_local = self.data.oMi[pj].inverse() * pin.SE3(np.eye(3), center)
            self.sole_fid[L] = self.model.addFrame(pin.Frame(
                f"{L}_sole", pj, self.model.frames[self.heel_fid[L]].parentFrame,
                sole_local, pin.FrameType.OP_FRAME))
        self.data = self.model.createData()          # 프레임 추가 후 재생성
        self.state = crocoddyl.StateMultibody(self.model)
        self.actu = crocoddyl.ActuationModelFloatingBase(self.state)   # ★내장(URDF서 nu=8 정상)
        self.nu = self.actu.nu                       # 8
        self.x_stand = np.concatenate([self.q_stand, np.zeros(self.model.nv)])
        pin.forwardKinematics(self.model, self.data, self.q_stand)
        pin.updateFramePlacements(self.model, self.data)
        self.heel0 = {L: self.data.oMf[self.heel_fid[L]].translation.copy() for L in LEGS}
        self.toe0 = {L: self.data.oMf[self.toe_fid[L]].translation.copy() for L in LEGS}
        base_xy0 = self.q_stand[0:2]
        self.heel_off = {L: self.heel0[L][0:2] - base_xy0 for L in LEGS}
        self.toe_off = {L: self.toe0[L][0:2] - base_xy0 for L in LEGS}
        # 상태 정규화 가중(2*nv): base x,y 자유(전진)·z 추종·자세 추종·관절·속도 추종
        nv = self.model.nv
        self.wx = np.array([0., 0., 50.] + [80.]*3 + [1.]*self.nu
                           + [10.]*3 + [10.]*3 + [1.]*self.nu)
        assert len(self.wx) == 2*nv

    def _standing_q(self):
        """평발 home(Q_FLAT)에서 heel·toe가 지면(z=foot_r)에 닿도록 base_z 조정."""
        q = pin.neutral(self.model)
        q[7:11] = Q_FLAT; q[11:15] = Q_FLAT
        q[2] = 0.5
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        zmin = min(min(self.data.oMf[self.heel_fid[L]].translation[2],
                       self.data.oMf[self.toe_fid[L]].translation[2]) for L in LEGS)
        q[2] += self.foot_r - zmin          # 최저 접촉구가 지면에 앉도록
        return q

    def _contact_and_cone(self, contacts, costs, L):
        """stance 다리 L 접촉. use_6d=ContactModel6D(sole, 전체wrench=CoP)+WrenchCone(발폴리곤)."""
        if self.use_6d:
            fid = self.sole_fid[L]
            ci = crocoddyl.ContactModel6D(self.state, fid, pin.SE3.Identity(),
                                          pin.LOCAL_WORLD_ALIGNED, self.nu, np.array([0., 50.]))
            contacts.addContact(f"{L}_c", ci)
            # WrenchCone: 마찰 + CoP를 발 폴리곤 내(전후 ±8cm, 측방 ±2cm=선발이라 좁게)
            cone = crocoddyl.WrenchCone(np.eye(3), self.mu, np.array([0.08, 0.02]))
            act = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub))
            res = crocoddyl.ResidualModelContactWrenchCone(self.state, fid, cone, self.nu)
            costs.addCost(f"{L}_cone", crocoddyl.CostModelResidual(self.state, act, res), self.w_cone)
            return
        pts = ((self.heel_fid[L], "heel"), (self.toe_fid[L], "toe")) if self.two_point \
              else ((self.toe_fid[L], "toe"),)
        for fid, tag in pts:
            ci = crocoddyl.ContactModel3D(self.state, fid, np.zeros(3),
                                          pin.LOCAL_WORLD_ALIGNED, self.nu, np.array([0., 50.]))
            contacts.addContact(f"{L}_{tag}_c", ci)
            cone = crocoddyl.FrictionCone(np.eye(3), self.mu, 4, False)
            act = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub))
            res = crocoddyl.ResidualModelContactFrictionCone(self.state, fid, cone, self.nu)
            costs.addCost(f"{L}_{tag}_cone", crocoddyl.CostModelResidual(self.state, act, res), self.w_cone)

    def _swing_cost(self, costs, L, heel_tgt, toe_tgt):
        """swing 발 heel·toe 위치 추종(평발 수평 착지)."""
        for fid, tgt, tag in ((self.heel_fid[L], heel_tgt, "heel"), (self.toe_fid[L], toe_tgt, "toe")):
            res = crocoddyl.ResidualModelFrameTranslation(self.state, fid, tgt, self.nu)
            costs.addCost(f"{L}_{tag}_track", crocoddyl.CostModelResidual(self.state, res), 1e5)

    def _make_node(self, stance, swing_targets, v_cmd, is_terminal=False):
        contacts = crocoddyl.ContactModelMultiple(self.state, self.nu)
        costs = crocoddyl.CostModelSum(self.state, self.nu)
        for L in stance:
            self._contact_and_cone(contacts, costs, L)
        for L, (h, t) in swing_targets.items():
            self._swing_cost(costs, L, h, t)
        # base 속도 명령 = 이동하는 상태 목표로 정규화
        x_ref = self.x_stand.copy()
        x_ref[self.model.nq + 0] = v_cmd[0]
        x_ref[self.model.nq + 1] = v_cmd[1]
        x_ref[self.model.nq + 5] = v_cmd[2]
        sref = crocoddyl.ResidualModelState(self.state, x_ref, self.nu)
        act = crocoddyl.ActivationModelWeightedQuad(self.wx**2)
        costs.addCost("xreg", crocoddyl.CostModelResidual(self.state, act, sref),
                      0.2 if not is_terminal else 10.0)
        if not is_terminal:
            costs.addCost("ureg", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelControl(self.state, self.nu)), 1e-3)
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actu, contacts, costs, 0., True)
        iam = crocoddyl.IntegratedActionModelEuler(dam, 0. if is_terminal else self.dt)
        if not is_terminal:
            iam.u_lb = -TAU_LIM; iam.u_ub = TAU_LIM
        return iam

    # ---- biped 게이트: SS_HL → DS → SS_HR → DS ... ----
    def _phase(self, k, T_ss, T_ds):
        """전역 노드 k에서 (stance legs, swing legs). 한 주기 = SS+DS+SS+DS."""
        period = 2 * (T_ss + T_ds)
        p = k % period
        if p < T_ss:               return ["HR"], ["HL"]     # HL swing
        elif p < T_ss + T_ds:      return ["HL", "HR"], []    # DS
        elif p < 2*T_ss + T_ds:    return ["HL"], ["HR"]     # HR swing
        else:                      return ["HL", "HR"], []    # DS

    def _swing_frac(self, k, T_ss, T_ds):
        period = 2 * (T_ss + T_ds); p = k % period
        if p < T_ss:            return (p + 0.5) / T_ss
        elif p < 2*T_ss + T_ds: return (p - T_ss - T_ds + 0.5) / T_ss
        return 0.0

    def _targets(self, L, frac, step_len, step_h, base_xy):
        """swing 발 heel·toe 목표(현 base 앵커, 반보 전진 착지, 아치 리프트)."""
        def tgt(off):
            px = base_xy[0] + off[0] + step_len * (frac - 0.5) + 0.5 * step_len
            py = base_xy[1] + off[1]
            pz = self.foot_r + step_h * np.sin(np.pi * frac)
            return np.array([px, py, pz])
        return tgt(self.heel_off[L]), tgt(self.toe_off[L])

    # ---- kinematic gait warm-start (stance 발 planted + swing arc, base 전진, per-node IK) ----
    def _ik_node(self, base_x, base_z, targets, q_init):
        """base 고정((base_x,0,base_z),수평)에서 8 leg DOF로 heel·toe 목표 도달(damped LS)."""
        m, d = self.model, self.data
        q = q_init.copy()
        q[0], q[1], q[2] = base_x, 0.0, base_z; q[3:7] = [0, 0, 0, 1]
        for _ in range(80):
            pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
            pin.computeJointJacobians(m, d, q)
            err = np.zeros(12); J = np.zeros((12, m.nv)); i = 0
            for L in LEGS:
                for fr, tag in ((self.heel_fid[L], "heel"), (self.toe_fid[L], "toe")):
                    err[3*i:3*i+3] = targets[f"{L}_{tag}"] - d.oMf[fr].translation
                    J[3*i:3*i+3] = pin.getFrameJacobian(m, d, fr, pin.LOCAL_WORLD_ALIGNED)[:3]
                    i += 1
            J[:, :6] = 0.0                       # base 고정
            if np.linalg.norm(err) < 1e-5:
                break
            q = pin.integrate(m, q, 0.5 * np.linalg.lstsq(J, err, rcond=None)[0])
            q[0], q[1], q[2] = base_x, 0.0, base_z; q[3:7] = [0, 0, 0, 1]
        return q

    def warm_start(self, v_cmd, N, T_ss, step_h):
        """보행 모양 초기 궤적 xs: base 전진·stance 발 planted(월드고정)·swing 발 arc."""
        v = float(v_cmd[0]); step_len = v * (T_ss * self.dt)
        planted = {L: {"heel": self.heel0[L].copy(), "toe": self.toe0[L].copy()} for L in LEGS}
        xs = []; q = self.q_stand.copy()
        for k in range(N + 1):
            p = k % (2 * T_ss)
            swing = "HL" if p < T_ss else "HR"; stance = "HR" if swing == "HL" else "HL"
            frac = ((k % T_ss) + 0.5) / T_ss
            base_x = self.q_stand[0] + v * k * self.dt
            tg = {f"{stance}_heel": planted[stance]["heel"], f"{stance}_toe": planted[stance]["toe"]}
            h, t = self._targets(swing, frac, step_len, step_h, np.array([base_x, 0.0]))
            tg[f"{swing}_heel"], tg[f"{swing}_toe"] = h, t
            q = self._ik_node(base_x, self.q_stand[2], tg, q)
            x = np.concatenate([q, np.zeros(self.model.nv)]); x[self.model.nq] = v
            xs.append(x)
            if k % T_ss == T_ss - 1:              # 스텝 끝: 착지한 발을 planted로 갱신
                hl, tl = self._targets(swing, 1.0, step_len, step_h, np.array([base_x, 0.0]))
                hl[2] = tl[2] = self.foot_r
                planted[swing]["heel"], planted[swing]["toe"] = hl, tl
        # ★속도를 위치 차분으로 채움(동역학 consistent=ffeas↓)
        for k in range(N):
            xs[k][self.model.nq:] = pin.difference(self.model, xs[k][:self.model.nq],
                                                    xs[k+1][:self.model.nq]) / self.dt
        xs[N][self.model.nq:] = xs[N-1][self.model.nq:]
        return xs

    def create_problem(self, x0, v_cmd=(0.15, 0., 0.), N=40, T_ss=12, T_ds=4, step_h=0.05, k0=0):
        v_cmd = np.asarray(v_cmd, float)
        step_len = float(v_cmd[0]) * ((T_ss + T_ds) * self.dt)
        base_xy0 = np.asarray(x0[0:2], float)
        models = []
        for k in range(N):
            g = k0 + k
            stance, swing = self._phase(g, T_ss, T_ds)
            frac = self._swing_frac(g, T_ss, T_ds)
            base_xy = base_xy0 + np.array([v_cmd[0] * k * self.dt, 0.0])
            swing_targets = {L: self._targets(L, frac, step_len, step_h, base_xy) for L in swing}
            models.append(self._make_node(stance, swing_targets, v_cmd))
        stance, swing = self._phase(k0 + N, T_ss, T_ds)
        term = self._make_node(stance, {}, v_cmd, is_terminal=True)
        return crocoddyl.ShootingProblem(x0, models, term)


def _selftest():
    ocp = BipedWalkOCP()
    m = ocp.model
    print(f"biped OCP: nq={m.nq} nv={m.nv} nu={ocp.nu} mass={pin.computeTotalMass(m):.3f}kg")
    print(f"q_stand base_z={ocp.q_stand[2]:.3f}  heel0 HL z={ocp.heel0['HL'][2]:.3f} toe0 HL z={ocp.toe0['HL'][2]:.3f}")
    prob = ocp.create_problem(ocp.x_stand, v_cmd=(0.15, 0., 0.), N=40)
    solver = crocoddyl.SolverBoxFDDP(prob)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    xs = [ocp.x_stand] * (prob.T + 1)
    us = prob.quasiStatic([ocp.x_stand] * prob.T)
    ok = solver.solve(xs, us, 150)
    xN = solver.xs[-1]
    # base pitch 추출(quat xyzw)
    def pitch(x):
        qx, qy, qz, qw = x[3:7]
        return np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1)))
    print(f"\nsolved={ok} iters={solver.iter} cost={solver.cost:.3f}")
    print(f"base x drift = {xN[0]-ocp.x_stand[0]:+.3f} m (기대 ~{0.15*prob.T*ocp.dt:+.3f})")
    print(f"base z 시작={ocp.x_stand[2]:.3f} 끝={xN[2]:.3f}")
    print(f"base pitch 시작={pitch(ocp.x_stand):+.1f}° 끝={pitch(xN):+.1f}°  (최대|pitch|={max(abs(pitch(x)) for x in solver.xs):.1f}°)")
    print(f"max |tau| = {max(np.abs(u).max() for u in solver.us):.1f} N·m")


if __name__ == "__main__":
    _selftest()
