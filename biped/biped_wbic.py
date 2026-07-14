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
        d.qpos[7:] = Q_HOME
        d.qpos[2] = 0.7
        mujoco.mj_forward(m, d)
        zmin = min(d.geom_xpos[s][2] - m.geom_size[s][0] for s in self.sph)
        d.qpos[2] -= zmin                                # 발 바닥 z=0
        mujoco.mj_forward(m, d)
        fp = np.array([d.geom_xpos[s] for s in self.sph])
        self.com_ref = np.array([fp[:, 0].mean(), fp[:, 1].mean(), d.subtree_com[0][2]])  # 지지중심 xy + 현 CoM z

    def foot_jac(self, k):
        jacp = np.zeros((3, self.nv))
        mujoco.mj_jac(self.m, self.d, jacp, None, self.d.geom_xpos[self.sph[k]], self.fbody[k])
        return jacp

    # ── WBIC stance QP (1틱) ──
    def wbic_stance(self):
        d, m, nv, nu, K = self.d, self.m, self.nv, self.nu, self.K
        nz = nv + 3 * K
        M = np.zeros((nv, nv)); mujoco.mj_fullM(m, M, d.qM)
        h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        Js = [self.foot_jac(k) for k in range(K)]
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
            a = 60 * (Q_HOME[j] - d.qpos[7 + j]) - 5 * qv[6 + j]
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
        x = solve_qp(P, g, G, hh, A, b, solver='quadprog')
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
