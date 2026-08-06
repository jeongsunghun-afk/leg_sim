"""biped 제자리 capture-point 스텝핑 (B2) — 점 발 동적 균형.

점 발은 정적 균형 불가(pivot 모멘트=0) → 넘어지는 CoM을 스윙발을 capture point에 놓아 잡는다.
성숙 quad_control.hpp wbic_track(스윙추종+stance GRF) 이식 + 2-phase 게이트 + capture point.

capture point:  ξ = x_com + v_com / ω ,  ω = √(g/z_com)   (착지 시 CoM 속도 소거)
게이트: 반주기 T_step = [더블서포트 ds] + [단일지지: 한 발 stance, 다른 발 swing→ξ 착지] 교대.
헤드리스: python biped_step.py  ·  뷰어: VIEW=1 python biped_step.py
"""
from __future__ import annotations
import os, time, numpy as np, mujoco, mujoco.viewer
from qpsolvers import solve_qp
import biped_wbic as BW
from biped_wbic import (STANCE_KD, W_ORI, W_POST, W_ANKLE, MU, MU_MARGIN,
                        LAMZ_MIN, TAU_PEAK, Q_HOME, ANKLE_IDX, base_rpy)

# ── 게이트/스텝 파라미터 ──
# ★교훈: 점 발 biped는 느린 게이트가 오히려 위험(두 발 같은 x=sagittal 제어불가라 대기시간만↑).
#   빠른 스텝(진자 시상수 0.23s 이하)이 안전. 안전게이트(0.40/0.55) 시도=0.83s로 악화 확인.
# ★T_STEP 0.24 → 0.38 (2026-08-06, C++ 기준으로 통일. cpp/src/biped_control.hpp:22-34)
#   실측 ROTOR_I(7.4e-4 = 구 placeholder 의 7.4배)를 넣으면 반사관성이 7.4배가 되어
#   0.24s 스텝의 스윙 가속에 필요한 토크가 tau_peak 을 넘어 QP 가 포화 → **2.18s 낙상**.
#   필요가속도 ∝ 1/T² 이므로 스텝을 늦추는 것이 해법이다. C++ 실측 스윕:
#     ROTOR_I 1e-4/2e-4/4e-4/5e-4 = 15s 무낙상 · 6e-4 = 9.4s · 7.4e-4 = 2.18s 낙상
#     7.4e-4 + T_STEP 0.32 = 15s 무낙상 tilt 2.7°(최량) · 0.40/0.50 = 낙상
#   ⚠ 스윙게인 상향은 역효과였다(SW_KP 800→1600/3200/5920 = 1.16/1.10/0.65s 낙상)
#     — 대역 부족이 아니라 **토크 포화**이기 때문.
#   이후 leg-odom 야코비안 편향 제거 후 재스윕 → 0.32→0.38 (cpp/STABILITY_MAP.md).
#   ⚠ 종전 주석("빠른 스텝=진자 시상수 0.23s 이하가 최적")은 armature 가 7.4배 과소하던
#     모델에서만 성립하던 결론이다. 실측 관성에서는 그 대역을 낼 토크가 없다.
T_STEP  = 0.38          # C++ 기준(실측 armature 하 재스윕값). 되돌리지 말 것
DS_FRAC = 0.10          # 짧은 더블서포트
STEP_H  = 0.06          # 스윙 발 들림 (★강건성 스윕: 0.05→0.06=발스침 여유. base0.50과 조합서 yaw 안정)
K_CAP   = 1.0           # capture 이론값
CAP_CLAMP = 0.22        # capture가 앞으로 뻗어 낙하 잡도록 완화
SW_KP, SW_KD = 800.0, 60.0    # 스윙 추종(biped 무거운 다리용 완화, quad 2400/110)
# ★K_RETURN 0.45 → 0.15 (2026-08-06, C++ 기준으로 통일. cpp/src/biped_control.hpp:30-35)
#   leg-odom 야코비안 편향(구중심 vs 접촉점)을 제거하자 기존 튜닝이 성립하지 않았다 —
#   0.45 는 **그 편향을 전제로** 맞춰진 값이었다. 실측 센서노이즈까지 넣고 재스윕한 결과가
#   T_STEP 0.38 + K_RETURN 0.15 다. 상세: cpp/STABILITY_MAP.md
K_RETURN = 0.15        # C++ 기준. sagittal return(원점복귀)
K_RET_LAT = 0.0        # 측방 crab: 튜닝 시도 모두 악화(점발 측방 근본한계). RL 몫. 0=로버스트 유지
K_LAT    = 0.5         # 측방 capture 감쇠(backward 안정)
SPREAD   = 1.0         # ★측방 벌림 전체(수렴방지=다리 안모임). in-place/전진 벌림 28/19cm 유지
GAP_MIN  = 0.14        # ★스윙-스탠스 최소 측방간격(꼬임 차단, 소프트)
GAP_MAX  = 0.34
W_LAM   = 0.1           # stance GRF 힌트(중력보상) 가중(약)
GVEC    = 9.81
# ── event-based gait (측방 DCM 동기): 착지 타이밍을 CoM이 swing측 넘어올 때로 ──
SS_NOMINAL = 0.16       # 스윙 궤적 nominal 시간(빠른 swing)
SS_MIN  = 0.10          # 최소 단일지지(발 들 시간)
SS_MAX  = 0.45          # 최대 단일지지(안전 강제전환)
TRIG_Y  = 0.03         # 측방 DCM이 midline서 swing측으로 이만큼 넘으면 착지


class BipedStep(BW.BipedWBIC):
    def __init__(self, mjcf=BW.MJCF):
        super().__init__(mjcf)
        self.t = 0.0
        self.nominal = None        # 발별 기준 xy(hip 아래)
        self.liftoff = [None, None]

    def reset(self):
        self.reset_stand()
        d = self.d
        self.nominal = np.array([self.foot_center(k)[:2].copy() for k in range(2)])  # 착지 기준(발 중심)
        com0 = d.subtree_com[0][:2].copy()
        self.com0 = com0                                                        # ★원점(복귀 목표 CoM xy)
        self.nominal_off = np.array([self.foot_center(k)[:2] - com0 for k in range(2)])  # ★몸(CoM) 상대 발 중심 오프셋
        self.t = 0.0
        self.stance = 1; self.swing = 0; self.t_ss = 0.0   # event-based gait 상태(HR stance·HL swing 시작)

    # ── event-based gait: 스윙궤적=SS_NOMINAL, 착지=측방 DCM이 swing측 넘어올 때 ──
    def step_gait(self, dt):
        d = self.d
        com = d.subtree_com[0]
        vcom = (self.Jc_cache @ d.qvel)[:2] if hasattr(self, 'Jc_cache') else np.zeros(2)
        z = max(com[2] - min(d.geom_xpos[sp][2] for sp in self.sph), 0.15)
        w = np.sqrt(GVEC / z)
        xi_y = com[1] + vcom[1] / w                        # 측방 DCM(world)
        _fc0, _fc1 = self.foot_center(0), self.foot_center(1)   # 발 중심(2접촉=2구 평균)
        mid_y = 0.5 * (_fc0[1] + _fc1[1])
        # ★측방 DCM을 body-frame으로(yaw 나도 올바른 측방=보행 강건). 발중점 기준 DCM벡터를 body-y축에 투영.
        qc = d.qpos[3:7]
        yaw_act = np.arctan2(2*(qc[0]*qc[3]+qc[1]*qc[2]), 1-2*(qc[2]**2+qc[3]**2))
        cya, sya = np.cos(yaw_act), np.sin(yaw_act)
        midx = 0.5 * (_fc0[0] + _fc1[0])
        dcm_by = -sya*(com[0]+vcom[0]/w-midx) + cya*(xi_y-mid_y)   # body-y (직진 yaw=0시 =xi_y-mid_y)
        sy = 1.0 if self.swing == 0 else -1.0              # swing측 부호(HL좌=+)
        s = np.clip(self.t_ss / SS_NOMINAL, 0.0, 1.0)
        mode = os.environ.get('SWMODE', 'dcm')
        if mode == 'com':          # CoM이 중선을 swing측으로 넘음(대칭 rock)
            committed = sy * (com[1] - mid_y) > 0.0
        elif mode == 'orbital':    # LIPM orbital energy: swing 발목표 기준 E≥0 (그쪽 apex 도달 보장)
            p_sw_y = mid_y + self.nominal_off[self.swing][1]     # swing 발이 갈 측방
            E = 0.5 * vcom[1]**2 - 0.5 * w**2 * (com[1] - p_sw_y)**2
            committed = (sy * vcom[1] > 0) and (E >= -TRIG_Y)    # swing측 이동 + 궤도에너지
        else:                      # 기본: body-frame 측방 DCM 임계
            committed = sy * dcm_by > TRIG_Y
        if self.t_ss > SS_MIN and (committed or self.t_ss > SS_MAX):   # 착지 이벤트
            self.stance, self.swing = self.swing, self.stance
            self.t_ss = 0.0
            self.liftoff[self.swing] = self.foot_center(self.swing).copy()
            return [self.stance], self.swing, 0.0
        self.t_ss += dt
        return [self.stance], self.swing, s

    # ── 게이트: (stance list, swing leg or None, 스윙위상 s∈[0,1]) ──
    def gait(self):
        p = (self.t % (2 * T_STEP)) / T_STEP     # [0,2)
        half = int(p)                            # 0=오른발 stance, 1=왼발 stance
        ph = p - half                            # [0,1) 반주기 내
        stance_leg = 1 if half == 0 else 0       # 0=HL(좌),1=HR(우)
        swing_leg = 1 - stance_leg
        if ph < DS_FRAC:                         # 더블서포트
            return [0, 1], None, 0.0
        s = (ph - DS_FRAC) / (1 - DS_FRAC)       # 스윙 위상 0→1
        return [stance_leg], swing_leg, s

    def dcm_target(self, swing_leg, s):
        """★base(body) 프레임 발배치: 발오프셋·capture·간격을 base 기준 계산 후 world 변환.
        base가 yaw로 돌면 발도 함께 회전 → 좌/우발 대응 유지(world고정 시 꼬임). 선회에도 필수."""
        d = self.d
        com = d.subtree_com[0]
        vcom_w = (self.Jc_cache @ d.qvel)[:2] if hasattr(self, 'Jc_cache') else np.zeros(2)
        z = max(com[2] - min(d.geom_xpos[sp][2] for sp in self.sph), 0.15)
        w = np.sqrt(GVEC / z)
        qc = d.qpos[3:7]                                    # base yaw → world↔body 회전
        yaw_a = np.arctan2(2 * (qc[0]*qc[3] + qc[1]*qc[2]), 1 - 2 * (qc[2]**2 + qc[3]**2))
        yaw = getattr(self, 'yaw_des', yaw_a)              # ★목표 heading 프레임(발이 선회 선도). 없으면 현재
        cy, sy = np.cos(yaw), np.sin(yaw)
        to_body  = lambda v: np.array([cy*v[0] + sy*v[1], -sy*v[0] + cy*v[1]])
        to_world = lambda v: np.array([cy*v[0] - sy*v[1],  sy*v[0] + cy*v[1]])
        v_b   = to_body(vcom_w)                             # 속도(body)
        err_b = to_body(com[:2] - self.com0)               # 원점복귀 오차(body)
        off   = self.nominal_off[swing_leg]                # 발 오프셋(body, reset yaw=0)
        lat   = 1.0 if off[1] > 0 else -1.0                # HL(좌)=+, HR(우)=−
        rel_fwd = off[0] + K_CAP * v_b[0] / w + K_RETURN * err_b[0]         # 전방: capture+원점복귀
        rel_fwd = np.clip(rel_fwd, off[0] - CAP_CLAMP, off[0] + CAP_CLAMP)
        rel_lat = SPREAD * off[1] + K_LAT * (v_b[1] / w) + K_RET_LAT * err_b[1]  # 측방: 벌림+capture+측방복귀(crab추종)
        st_b = to_body(self.foot_center(1 - swing_leg)[:2] - com[:2])  # 스탠스발 중심(body 상대)
        gap = np.clip(lat * (rel_lat - st_b[1]), GAP_MIN, GAP_MAX)          # 스탠스 반대쪽 최소간격(body)
        rel_lat = st_b[1] + lat * gap
        return com[:2] + to_world(np.array([rel_fwd, rel_lat]))             # → world

    def swing_traj(self, leg, s):
        """liftoff→capture target, apex STEP_H. 5차 위치 + 속도."""
        p0 = self.liftoff[leg]
        tgt = self.dcm_target(leg, s)
        gsz = self.m.geom_size[self.sph[leg]]
        clr = gsz[2] if self.m.geom_type[self.sph[leg]] == mujoco.mjtGeom.mjGEOM_BOX else gsz[0]  # box=z반·sphere=r
        gz = min(self.d.geom_xpos[sp][2] for sp in self.sph) + clr
        p1 = np.array([tgt[0], tgt[1], gz])
        ss = np.clip(s, 0, 1)
        smooth = 10*ss**3 - 15*ss**4 + 6*ss**5           # smoothstep(위치)
        dsmooth = (30*ss**2 - 60*ss**3 + 30*ss**4) / max(1e-6, (1-DS_FRAC)*T_STEP)
        p = p0 + (p1 - p0) * smooth
        zlift = 4 * STEP_H * ss * (1 - ss)               # 포물선 들림
        p[2] = p0[2] + (p1[2]-p0[2]) * smooth + zlift
        v = (p1 - p0) * dsmooth
        v[2] = (p1[2]-p0[2]) * dsmooth + 4*STEP_H*(1-2*ss)*dsmooth
        return p, v

    # ── wbic_track (quad_control.hpp:359 이식): 스윙추종 + stance GRF + 자세/CoM_z/posture ──
    def wbic_track(self, contacts, swing):
        d, m, nv, nu = self.d, self.m, self.nv, self.nu
        Kc = len(contacts); nz = nv + 3 * Kc
        sl = lambda k: nv + 3 * k
        cjac = [self.foot_jac(c) for c in contacts]
        M = np.zeros((nv, nv)); mujoco.mj_fullM(m, M, d.qM)
        h = d.qfrc_bias.copy(); qv = d.qvel.copy()
        P = np.zeros((nz, nz)); g = np.zeros(nz)
        sw_vidx = set()
        # 스윙 추종
        for leg, (ptgt, vtgt) in swing.items():
            J = self.foot_jac(leg)
            accel = SW_KP * (ptgt - d.geom_xpos[self.sph[leg]]) + SW_KD * (vtgt - J @ qv)
            P[:nv, :nv] += 90.0 * (J.T @ J); g[:nv] -= 90.0 * (J.T @ accel)
            for t in range(4): sw_vidx.add(6 + leg * 4 + t)
        Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
        self.Jc_cache = Jc
        # 자세 레벨링(현재 yaw)
        qc = d.qpos[3:7]
        yaw = np.arctan2(2*(qc[0]*qc[3]+qc[1]*qc[2]), 1-2*(qc[2]**2+qc[3]**2))
        qlev = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        oerr = np.zeros(3); mujoco.mju_subQuat(oerr, qc, qlev)
        for j in range(2):
            a = 150*(-oerr[j]) - 20*qv[3+j]; P[3+j, 3+j] += W_ORI; g[3+j] -= W_ORI*a
        a_yaw = 150*(-oerr[2]) - 20*qv[5]; P[5, 5] += 1.0; g[5] -= 1.0*a_yaw
        # CoM 높이
        zref = self.com_ref[2]; a_z = 200*(zref - d.subtree_com[0][2]) - 25*(Jc @ qv)[2]
        P[:nv, :nv] += 150.0*np.outer(Jc[2], Jc[2]); g[:nv] -= 150.0*a_z*Jc[2]
        # 약한 CoM 수평(드리프트 억제, 스텝핑이 주 균형)
        for j in range(2):
            a_h = 30*(self.nominal.mean(0)[j] - d.subtree_com[0][j]) - 8*(Jc @ qv)[j]
            P[:nv, :nv] += 3.0*np.outer(Jc[j], Jc[j]); g[:nv] -= 3.0*a_h*Jc[j]
        # posture
        for j in range(nu):
            a = 60*(Q_HOME[j] - d.qpos[7+j]) - 5*qv[6+j]
            w = W_ANKLE if j in ANKLE_IDX else (5.0 if (6+j) in sw_vidx else W_POST)
            P[6+j, 6+j] += w; g[6+j] -= w*a
        P[:nv, :nv] += 1e-3*np.eye(nv)
        # stance GRF 힌트(중력보상 수직)
        wsupp = 13.571 * GVEC / max(1, Kc)
        for k in range(Kc):
            P[sl(k):sl(k)+3, sl(k):sl(k)+3] += W_LAM*np.eye(3)
            g[sl(k)+2] -= W_LAM*wsupp
        # 등식: base6 + 접촉(STANCE_KD)
        neq = 6 + 3*Kc; A = np.zeros((neq, nz)); b = np.zeros(neq)
        A[:6, :nv] = M[:6, :]; b[:6] = -h[:6]
        for k in range(Kc):
            A[:6, sl(k):sl(k)+3] = -cjac[k][:, :6].T
            A[6+3*k:6+3*k+3, :nv] = cjac[k]
            b[6+3*k:6+3*k+3] = -STANCE_KD*(cjac[k] @ qv)
        # 부등식: 마찰추 + λz≥min + 토크한계
        rows = []; hh = []
        mu = MU*MU_MARGIN; sgn = [(1,0),(-1,0),(0,1),(0,-1)]
        for k in range(Kc):
            o = sl(k)
            for sx, sy in sgn:
                r = np.zeros(nz); r[o]=sx; r[o+1]=sy; r[o+2]=-mu; rows.append(r); hh.append(0.0)
            r = np.zeros(nz); r[o+2]=-1; rows.append(r); hh.append(-LAMZ_MIN)
        # 토크 한계: -τpk ≤ M[6:]q̈+h[6:]-ΣJᵀλ ≤ τpk
        Tm = np.zeros((nu, nz)); Tm[:, :nv] = M[6:, :]
        for k in range(Kc): Tm[:, sl(k):sl(k)+3] = -cjac[k][:, 6:].T
        h_act = h[6:]
        for i in range(nu):
            rows.append(Tm[i]);  hh.append(TAU_PEAK[i]-h_act[i])
            rows.append(-Tm[i]); hh.append(TAU_PEAK[i]+h_act[i])
        G = np.array(rows); hh = np.array(hh)
        P = 0.5*(P+P.T) + 1e-8*np.eye(nz)
        x = solve_qp(P, g, G, hh, A, b, solver='quadprog')
        if x is None:
            return self.wbic_stance()      # 폴백
        qdd = x[:nv]; tau = M[6:, :] @ qdd + h[6:]
        for k in range(Kc): tau -= cjac[k][:, 6:].T @ x[sl(k):sl(k)+3]
        d.ctrl[:] = np.clip(tau, -TAU_PEAK, TAU_PEAK)
        return True

    def control(self, dt):
        stance, swing_leg, s = self.gait()
        # 스윙 시작 시 liftoff 기록
        if swing_leg is not None and s < 1e-3:
            self.liftoff[swing_leg] = self.d.geom_xpos[self.sph[swing_leg]].copy()
        if swing_leg is None:
            self.wbic_track([0, 1], {})            # 더블서포트
        else:
            if self.liftoff[swing_leg] is None:
                self.liftoff[swing_leg] = self.d.geom_xpos[self.sph[swing_leg]].copy()
            p, v = self.swing_traj(swing_leg, s)
            self.wbic_track(stance, {swing_leg: (p, v)})
        self.t += dt


def main():
    c = BipedStep(); c.reset()
    m, d = c.m, c.d
    print(f"제자리 capture-point 스텝핑 · T_step={T_STEP} DS={DS_FRAC} STEP_H={STEP_H} · com_ref={np.round(c.com_ref,3)}")
    T = float(os.environ.get('T', 5.0)); dt = m.opt.timestep
    view = os.environ.get('VIEW', '0') == '1'
    viewer = mujoco.viewer.launch_passive(m, d) if view else None
    z0 = d.qpos[2]; steps = int(T/dt); fell = None; maxtilt = 0.0
    for i in range(steps):
        c.control(dt); mujoco.mj_step(m, d)
        rpy = base_rpy(d.qpos[3:7]); maxtilt = max(maxtilt, np.hypot(rpy[0], rpy[1]))
        if viewer is not None and i % 10 == 0: viewer.sync()
        if d.qpos[2] < 0.2:
            fell = i*dt; break
    rpy = base_rpy(d.qpos[3:7])
    tend = min(i*dt, T)
    if fell: print(f"❌ 낙상 @ {fell:.2f}s")
    print(f"t={tend:.2f}s · base z={d.qpos[2]:.3f}(Δ{d.qpos[2]-z0:+.3f}) xy드리프트=({d.qpos[0]:+.3f},{d.qpos[1]:+.3f}) tilt now={np.hypot(rpy[0],rpy[1]):.1f}° max={maxtilt:.1f}°")
    ok = fell is None and maxtilt < 25 and d.qpos[2] > 0.4
    print("✅ 스텝핑으로 균형 유지" if ok else "⚠️ 아직 불안정(튜닝 필요)")
    if viewer is not None:
        while viewer.is_running():
            c.control(dt); mujoco.mj_step(m, d); viewer.sync(); time.sleep(dt)


if __name__ == '__main__':
    main()
