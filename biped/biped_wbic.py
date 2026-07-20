"""biped WBIC 균형 (B1) — 성숙 quad_control.hpp wbic_stance 를 8-DOF biped(HL/HR)로 이식.

정식화 (quad_control.hpp:314 wbic_stance 동일):
  변수 z = [ q̈(nv=14) ; λ(3·K), K=2 ]
  min  1·‖Jc q̈ − a_com‖²  +  5·‖자세 roll/pitch/yaw 레벨링‖²  +  Σ w·‖posture‖²  + reg
  s.t. 부동베이스 6행:  M[0:6] q̈ − Σ Jsᵀ λ = −h[0:6]
       접촉 3K:         Js q̈ = −STANCE_KD·(Js q̇)      (baumgarte, 터치다운 잔류속도→0)
       마찰추(피라미드) |λx|,|λy| ≤ μλz ,  λz ≥ LAMZ_MIN
  τ = M[6:] q̈ + h[6:] − Σ Jsᵀλ  → clip(±τ_peak)

옛 wbic_balance.py 대비 개선(quad 교훈): 현재-yaw 프레임 roll/pitch 레벨링(yaw 안 되당김)·STANCE_KD·발목 posture 가중·Peak토크.
헤드리스: python biped_wbic.py  (3s 균형, 드리프트 리포트) · 뷰어: VIEW=1 python biped_wbic.py
"""
from __future__ import annotations
import os, time, numpy as np, mujoco, mujoco.viewer
from qpsolvers import solve_qp

MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'biped_from_quad.mjcf')

# ── 게인/파라미터 (quad_control.hpp 기본값) ──
STANCE_KD = 20.0
W_ORI     = 5.0          # 자세 레벨링 task 가중
W_POST    = 1.0          # 관절 posture 가중(기본)
W_ANKLE   = 20.0         # 발목(foot) posture 가중 ↑ (whip 억제)
MU, MU_MARGIN = 0.8, 0.707     # μ_eff = 0.566 (덜 보수적, 뷰어 피드백. MJCF 물리 1.6)
LAMZ_MIN  = 1.0
# ★mature 동일 환경(params.html): peak hip/thigh 84·calf 126·foot 96(8:1 재기어)
TAU_PEAK  = np.array([84, 84, 126, 96, 84, 84, 126, 96.0])     # HL/HR × (hip,thigh,calf,foot)
# GEARBOX(반사관성) — quad_control.hpp:92-103 동일. armature=Irot·N² + damping + frictionloss
GEAR    = np.array([7.0, 7.0, 10.5, 8.0])   # hip,thigh,calf,foot (배포=foot 14→8:1 재기어)
ROTOR_I = 1e-4
JDAMP, JFRIC = 0.1, 0.5

# home posture — ★body 낮춤(mild crouch, 뷰어 피드백): 무릎 굽혀 IK 특이점 회피(legh~0.5, base 0.48).
# ★얕은 crouch가 sweet spot(12.8s): deep crouch(legh0.44)=과함4.6s·taller(0.58)=4.8s. hip=0.
Q_HOME = np.array([0.0, 0.05, -0.2, 0.0,  0.0, 0.05, -0.2, 0.0])
# ★평발(flat-foot) home: 발목 -1.146으로 눕혀 heel(foot_joint)+toe 밑창을 지면과 수평.
# thigh0.25·calf-0.50로 발을 뒤로 보내 CoM이 밑창중심(x0.053)에 정렬(앞뒤 마진 ±8cm). base~0.43.
Q_HOME_FLAT = np.array([0.0, 0.25, -0.50, -1.14626,  0.0, 0.25, -0.50, -1.14626])
ANKLE_IDX = [3, 7]       # HL_foot, HR_foot


class BipedWBIC:
    def __init__(self, mjcf=MJCF):
        self.m = mujoco.MjModel.from_xml_path(mjcf)
        self.d = mujoco.MjData(self.m)
        self.nv, self.nu = self.m.nv, self.m.nu          # 14, 8
        self.K = 2
        self.sph = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in ['HL_sphere', 'HR_sphere']]
        self.fbody = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
                      for b in ['HL_foot_contact_link', 'HR_foot_contact_link']]
        # ★다접촉(line/flat foot): 발당 접촉구 리스트 = tip + 추가구(sphere2/3/4…). 없으면 1점(하위호환).
        self.foot_spheres = [[self.sph[k]] for k in range(2)]
        for suf in ['2', '3', '4']:
            ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f'{L}_sphere{suf}') for L in ['HL', 'HR']]
            if all(i >= 0 for i in ids):
                for k in range(2): self.foot_spheres[k].append(ids[k])
        self.two_contact = len(self.foot_spheres[0]) > 1
        self.sph2 = [fs[1] for fs in self.foot_spheres] if self.two_contact else None
        # ★heel 구 보유 = 통합 모델(발목 세우면 toe만=점발, 눕히면 heel+toe=평발). 접촉모드는 런타임 자세로 결정.
        self.has_heel = self.two_contact and any(
            self.m.geom_bodyid[fs[1]] != self.fbody[k] for k, fs in enumerate(self.foot_spheres))
        # 런타임 접촉모드: 통합 모델은 평발(정적 rest) 기본, 아니면 점발. q_home/높이 목표를 결정.
        self.contact_mode = '2pt' if self.has_heel else '1pt'
        self.q_home = Q_HOME_FLAT if self.contact_mode == '2pt' else Q_HOME   # ★자세 task 기준(램프됨)
        self.qmin = self.m.jnt_range[1:, 0].copy()       # 관절 하한(freejoint 제외)
        self.qmax = self.m.jnt_range[1:, 1].copy()
        self.com_ref = None
        self.setup_gearbox()

    def setup_gearbox(self):
        """반사관성(armature=Irot·N²) + 점성감쇠 + 마찰. mature와 동일(GEARBOX ON). 다리 flail 억제."""
        m = self.m
        for j in range(self.nu):                          # 액추에이터 관절 = dof 6+j (freejoint 뒤 hinge)
            N = GEAR[j % 4]
            dof = 6 + j
            m.dof_armature[dof] = ROTOR_I * N * N
            m.dof_damping[dof] = JDAMP
            m.dof_frictionloss[dof] = JFRIC

    # ── 초기화: home pose 스폰 + 발 착지 높이 + com_ref = 지지중심 ──
    def reset_stand(self):
        d, m = self.d, self.m
        d.qpos[:] = 0; d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[:] = 0; d.qacc[:] = 0                  # ★런타임 재정착(모드전환) 시 잔류속도 제거
        d.qpos[7:] = Q_HOME_FLAT if self.contact_mode == '2pt' else Q_HOME   # ★2점=눕힌 발목 home
        d.qpos[2] = 0.7
        mujoco.mj_forward(m, d)
        _allsph = [g for fs in self.foot_spheres for g in fs]
        zmin = min(d.geom_xpos[s][2] - m.geom_size[s][0] for s in _allsph)   # 모든 접촉구 바닥 z=0
        d.qpos[2] -= zmin
        mujoco.mj_forward(m, d)
        fp = np.array([self.foot_center(k) for k in range(2)])               # 발 중심(2구 평균)
        self.com_ref = np.array([fp[:, 0].mean(), fp[:, 1].mean(), d.subtree_com[0][2]])  # 지지중심 xy + 현 CoM z
        # ★평발 swing 발 수평 목표 = home(밑창 수평)에서의 발 world quat (yaw=0 기준). swing 중 이 자세로 유지.
        self.foot_home_quat = [d.xquat[self.fbody[k]].copy() for k in range(2)]
        self.q_home = (Q_HOME_FLAT if self.contact_mode == '2pt' else Q_HOME).copy()   # 스폰 자세 = q_home

    def foot_jac(self, k):
        jacp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jacp, None, self.d.geom_xpos[self.sph[k]], self.fbody[k])
        return jacp

    # ── 2점 접촉 헬퍼 ──
    def foot_jac_at(self, geom, body):
        jacp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jacp, None, self.d.geom_xpos[geom], body)
        return jacp

    def contact_pts(self, stance):
        """stance 발 → 접촉점 [(geom, body, foot), ...]. ★실제 지면 근처 구만(적응: 발 기울면 뜬 구 제외).
        둘 다 닿음=2점(line-foot 이득) / 하나만 닿음=1점(점발처럼, 과구속 회피)."""
        pts = []
        for f in stance:
            insph = [g for g in self.foot_spheres[f]
                     if self.d.geom_xpos[g][2] < self.m.geom_size[g][0] + 0.006]   # 바닥 6mm 이내=접촉
            if not insph:
                insph = [self.foot_spheres[f][0]]   # 다 뜸(이례적)=tip 폴백
            for g in insph:
                pts.append((g, int(self.m.geom_bodyid[g]), f))   # ★구별 실제 body(heel=foot_link, toe=contact_link)
        return pts

    def _ref_sph(self, leg):
        """발 기준 구 = 접촉모드별. 점발=tip 하나(heel은 공중), 평발=heel+toe 전부(밑창중점)."""
        return self.foot_spheres[leg] if getattr(self, 'contact_mode', '1pt') == '2pt' else [self.foot_spheres[leg][0]]

    def foot_center(self, leg):
        """발 기준점(world) = 모드별 기준구 평균(점발=tip 위치, 평발=밑창중점)."""
        return np.mean([self.d.geom_xpos[g] for g in self._ref_sph(leg)], axis=0)

    def foot_jac_center(self, leg):
        """발 중심 자코비안(swing task용) = 모드별 기준구 자코비안 평균. ★구별 실제 body."""
        return np.mean([self.foot_jac_at(g, int(self.m.geom_bodyid[g])) for g in self._ref_sph(leg)], axis=0)

    # ── WBIC stance QP (1틱) ──
    def wbic_stance(self):
        d, m, nv, nu = self.d, self.m, self.nv, self.nu
        cpts = self.contact_pts([0, 1])            # ★양발 stance 접촉점(2접촉=4점)
        K = len(cpts); nz = nv + 3 * K
        M = np.zeros((nv, nv)); mujoco.mj_fullM(m, M, d.qM)
        h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        Js = [self.foot_jac_at(g, b) for (g, b, f) in cpts]
        Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        com = d.subtree_com[0].copy()

        P = np.zeros((nz, nz)); g = np.zeros(nz)
        # CoM task (weight 1)
        a_com = np.array([120, 120, 200]) * (self.com_ref - com) - np.array([20, 20, 25]) * (Jc @ qv)
        P[:nv, :nv] += Jc.T @ Jc; g[:nv] -= Jc.T @ a_com
        # 자세 레벨링: 현재-yaw 프레임서 roll/pitch/yaw (yaw 0으로 안 되당김)
        qc = d.qpos[3:7]
        yaw = np.arctan2(2 * (qc[0]*qc[3] + qc[1]*qc[2]), 1 - 2 * (qc[2]**2 + qc[3]**2))
        qlev = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        oerr = np.zeros(3); mujoco.mju_subQuat(oerr, qc, qlev)
        for j in range(3):
            a = 150 * (-oerr[j]) - 20 * qv[3 + j]
            P[3 + j, 3 + j] += W_ORI; g[3 + j] -= W_ORI * a
        # 관절 posture (nullspace)
        for j in range(nu):
            a = 60 * (self.q_home[j] - d.qpos[7 + j]) - 5 * qv[6 + j]
            w = W_ANKLE if j in ANKLE_IDX else W_POST
            P[6 + j, 6 + j] += w; g[6 + j] -= w * a
        # 정칙화
        P[:nv, :nv] += 1e-4 * np.eye(nv)
        for k in range(K):
            P[nv + 3*k:nv + 3*k + 3, nv + 3*k:nv + 3*k + 3] += 1e-3 * np.eye(3)

        # 등식: 부동베이스 6 + 접촉 3K
        A = np.zeros((6 + 3 * K, nz)); b = np.zeros(6 + 3 * K)
        A[:6, :nv] = M[:6, :]; b[:6] = -h[:6]
        for k in range(K):
            A[:6, nv + 3*k:nv + 3*k + 3] = -Js[k][:, :6].T
            A[6 + 3*k:6 + 3*k + 3, :nv] = Js[k]
            b[6 + 3*k:6 + 3*k + 3] = -STANCE_KD * (Js[k] @ qv)
        # 부등식: 마찰추 4 + λz≥min 1, per foot
        G = np.zeros((5 * K, nz)); hh = np.zeros(5 * K)
        mu = MU * MU_MARGIN; sgn = [(1, 0), (-1, 0), (0, 1), (0, -1)]; r = 0
        for k in range(K):
            o = nv + 3 * k
            for sx, sy in sgn:
                G[r, o] = sx; G[r, o + 1] = sy; G[r, o + 2] = -mu; r += 1
            G[r, o + 2] = -1.0; hh[r] = -LAMZ_MIN; r += 1

        P = 0.5 * (P + P.T) + 1e-8 * np.eye(nz)
        # ★실제 발당 2점 접지 시만 rank-deficient → proxqp. 점발(발당1점)=quadprog(강건·결정적).
        _fc = {}
        for (_g, _b, _f) in cpts: _fc[_f] = _fc.get(_f, 0) + 1
        x = solve_qp(P, g, G, hh, A, b, solver='proxqp' if any(n > 1 for n in _fc.values()) else 'quadprog')
        if x is None:
            return False
        qdd = x[:nv]
        tau = M[6:, :] @ qdd + h[6:]
        for k in range(K):
            tau -= Js[k][:, 6:].T @ x[nv + 3*k:nv + 3*k + 3]
        d.ctrl[:] = np.clip(tau, -TAU_PEAK, TAU_PEAK)
        return True


def base_rpy(qc):
    r = np.arctan2(2*(qc[0]*qc[1] + qc[2]*qc[3]), 1 - 2*(qc[1]**2 + qc[2]**2))
    p = np.arcsin(np.clip(2*(qc[0]*qc[2] - qc[3]*qc[1]), -1, 1))
    y = np.arctan2(2*(qc[0]*qc[3] + qc[1]*qc[2]), 1 - 2*(qc[2]**2 + qc[3]**2))
    return np.degrees([r, p, y])


def main():
    c = BipedWBIC()
    c.reset_stand()
    m, d = c.m, c.d
    print(f"모델 nv={c.nv} nu={c.nu} · com_ref={np.round(c.com_ref,3)} · 초기 base z={d.qpos[2]:.3f}")
    T = float(os.environ.get('T', 3.0))
    steps = int(T / m.opt.timestep)
    fails = 0
    view = os.environ.get('VIEW', '0') == '1'
    viewer = mujoco.viewer.launch_passive(m, d) if view else None
    z0 = d.qpos[2]
    for i in range(steps):
        if not c.wbic_stance():
            fails += 1
        mujoco.mj_step(m, d)
        if viewer is not None and i % 10 == 0:
            viewer.sync()
        if d.qpos[2] < 0.2:                     # 낙상
            print(f"❌ 낙상 @ t={i*m.opt.timestep:.2f}s (base z={d.qpos[2]:.3f})")
            break
    rpy = base_rpy(d.qpos[3:7])
    print(f"t={min(i*m.opt.timestep, T):.2f}s 종료 · QP실패 {fails}회")
    print(f"base pos={np.round(d.qpos[:3],4)}  (z 드리프트 {d.qpos[2]-z0:+.4f})")
    print(f"base rpy(deg)={np.round(rpy,2)}  · tilt={np.hypot(rpy[0],rpy[1]):.2f}°")
    print("✅ 균형 유지" if d.qpos[2] > 0.2 and np.hypot(rpy[0], rpy[1]) < 15 else "⚠️ 불안정")
    if viewer is not None:
        while viewer.is_running():
            c.wbic_stance(); mujoco.mj_step(m, d); viewer.sync(); time.sleep(m.opt.timestep)


if __name__ == '__main__':
    main()
