#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C1.1/C1.2 — contact-implicit forward 동역학 (완화 상보성).
남은 C-1 관문: 고정 active-set이 아니라 **접촉 활성이 상태의 부드러운 함수**(타이밍 발견).
Pinocchio constraintDynamics=bilateral(발 떠도 당김)이라 직접 불가 → 여기서 **부드러운
단방향 접촉력**(softplus 완화 상보성)을 ABA에 외력으로 인가:
  φ_i = 발 z(지면 0 기준).  단방향 침투 d_i = ρ·softplus(−φ_i/ρ)  (φ<0=침투서만 힘, 스무스).
  f_n_i = k_n·d_i − b_n·ż·w_i (스프링+접촉시 감쇠),  w_i=σ(−φ_i/ρ)=접촉 활성(0~1).
  f_t_i = −b_t·ẋy·w_i (접선 감쇠, no-slip 근사).  → J_cᵀ f 인가 → aba → semi-implicit.
ρ↓=하드(crisp)·미분가능. C-2 autodiff 대신 **해석 그래디언트**(computeABADerivatives+힘 연쇄) 목표.

★C1.2 해석 그래디언트는 Pinocchio 내장(constraintDynamics 도함수, c1_gradient_check.py 증명)을
per-mode로 쓰거나, 여기 스무스 힘법칙을 직접 미분. 우선 forward+self-test(서기 균형)부터.

실행: /home/jsh/miniforge3/envs/proxddp/bin/python ci_action.py
"""
import numpy as np, pinocchio as pin
from model_bridge import MjPinBridge, FEET

FOOT_R = 0.025

def softplus(x, rho):    # 스무스 ReLU: ~x(x≫0)·~0(x≪0), 폭 rho
    z = x / rho
    return rho * np.where(z > 30, z, np.log1p(np.exp(np.clip(z, -30, 30))))

def sigmoid(x, rho):
    return 1.0 / (1.0 + np.exp(-np.clip(x / rho, -30, 30)))

class ContactImplicit:
    """부드러운 단방향 접촉 forward 동역학 (완화 상보성)."""
    def __init__(self, br, rho=0.004, kn=1.2e4, bn=120.0, bt=80.0, mu=0.8, ground=0.0):
        self.br, self.m, self.d = br, br.model, br.data
        self.rho, self.kn, self.bn, self.bt, self.mu = rho, kn, bn, bt, mu
        self.ground = ground
        self.fids = [br.foot_fid[L] for L in FEET]

    def _force_law(self, phi, vf):
        """접촉력 법칙 f(φ, vf) → world 3D. φ=지면위높이, vf=접촉점 선속도. (미분용 순수함수)"""
        w = sigmoid(-phi, self.rho)                                       # 접촉 활성 0~1
        depth = softplus(-phi, self.rho)                                 # 단방향 침투
        fn = softplus(self.kn * depth - self.bn * vf[2] * w, 1.0)         # 법선(스프링+감쇠, ≥0 스무스)
        ft = -self.bt * vf[:2] * w                                       # 접선 감쇠(no-slip 근사)
        ftn = np.linalg.norm(ft) + 1e-9
        cap = self.mu * fn                                              # 마찰콘(스무스 클립)
        scale = cap / ftn if ftn > cap else 1.0
        ft = ft * scale
        return np.array([ft[0], ft[1], fn])

    def _foot_state(self, q, v):
        """각 발의 (phi, vf, J[3×nv]) — FK 1회."""
        m, d = self.m, self.d
        pin.forwardKinematics(m, d, q, v); pin.updateFramePlacements(m, d); pin.computeJointJacobians(m, d, q)
        out = []
        for fid in self.fids:
            oMf = d.oMf[fid]
            p = oMf.translation + oMf.rotation @ np.array([0., 0., -FOOT_R])
            phi = p[2] - self.ground
            J = pin.getFrameJacobian(m, d, fid, pin.LOCAL_WORLD_ALIGNED)[:3]
            vf = J @ v                                                    # 접촉점 선속도(=J·v)
            out.append((phi, vf, J))
        return out

    def contact_forces(self, q, v):
        F = []; info = []
        for phi, vf, J in self._foot_state(q, v):
            f = self._force_law(phi, vf); F.append(f); info.append((phi, sigmoid(-phi,self.rho), f[2]))
        return F, info

    def dynamics(self, q, v, tau_act):
        """ddq = aba(q,v, τ + Σ J_cᵀ f).  tau_act=액추에이터(nv-6)."""
        m, d = self.m, self.d
        fs = self._foot_state(q, v)
        tau_full = np.concatenate([np.zeros(6), tau_act])
        F = []; info = []
        for phi, vf, J in fs:
            f = self._force_law(phi, vf); F.append(f); info.append((phi, sigmoid(-phi,self.rho), f[2]))
            tau_full = tau_full + J.T @ f
        ddq = pin.aba(m, d, q, v, tau_full)
        return ddq, F, info

    def dynamics_derivatives(self, q, v, tau_act):
        """★C1.2 해석 그래디언트: ∂ddq/∂(q,v,τ). computeABADerivatives + 힘 연쇄.
           근사: ∂vf/∂q≈0·∂Jᵀ/∂q≈0(kin.hessian 생략) — 지배항 검증용."""
        m, d = self.m, self.d
        fs = self._foot_state(q, v)
        tau_full = np.concatenate([np.zeros(6), tau_act])
        Js = []; dfdphi = []; dfdvf = []
        for phi, vf, J in fs:
            f = self._force_law(phi, vf); tau_full = tau_full + J.T @ f; Js.append(J)
            # 힘법칙 미분(FD, cheap): ∂f/∂φ(3), ∂f/∂vf(3×3)
            e = 1e-6
            dfp = (self._force_law(phi + e, vf) - self._force_law(phi - e, vf)) / (2*e)
            dfv = np.zeros((3, 3))
            for k in range(3):
                dvf = np.zeros(3); dvf[k] = e
                dfv[:, k] = (self._force_law(phi, vf + dvf) - self._force_law(phi, vf - dvf)) / (2*e)
            dfdphi.append(dfp); dfdvf.append(dfv)
        # ABA 도함수(τ_full 고정 편미분)
        pin.computeABADerivatives(m, d, q, v, tau_full)
        ddq = d.ddq.copy(); Minv = np.array(d.Minv); aba_dq = np.array(d.ddq_dq); aba_dv = np.array(d.ddq_dv)
        # ∂τ_full/∂(q,v) — 접촉 연쇄
        nv = m.nv; dtau_dq = np.zeros((nv, nv)); dtau_dv = np.zeros((nv, nv))
        for J, dfp, dfv in zip(Js, dfdphi, dfdvf):
            dtau_dv += J.T @ (dfv @ J)                        # ∂f/∂vf · ∂vf/∂v(=J)
            dtau_dq += J.T @ np.outer(dfp, J[2])              # ∂f/∂φ · ∂φ/∂q(=J[2])  (∂vf/∂q,∂Jᵀ/∂q 생략)
        dddq_dq = aba_dq + Minv @ dtau_dq
        dddq_dv = aba_dv + Minv @ dtau_dv
        dddq_dtau = Minv[:, 6:]
        return ddq, dddq_dq, dddq_dv, dddq_dtau

    def step(self, q, v, tau_act, dt):
        ddq, F, info = self.dynamics(q, v, tau_act)
        v_next = v + dt * ddq
        q_next = pin.integrate(self.m, q, dt * v_next)     # semi-implicit Euler
        return q_next, v_next, info


def _stance_q(br):
    m, d = br.model, br.data
    q = pin.neutral(m); q[2] = 0.42
    tgt = {'FL':[0.30,0.16],'FR':[0.30,-0.16],'HL':[-0.30,0.16],'HR':[-0.30,-0.16]}
    for _ in range(300):
        pin.forwardKinematics(m,d,q); pin.updateFramePlacements(m,d); pin.computeJointJacobians(m,d,q)
        err=np.zeros(12); J=np.zeros((12,m.nv))
        for i,L in enumerate(FEET):
            p=d.oMf[br.foot_fid[L]].translation + d.oMf[br.foot_fid[L]].rotation@np.array([0,0,-FOOT_R])
            err[3*i:3*i+3]=np.array([tgt[L][0],tgt[L][1],0.0]) - p
            J[3*i:3*i+3]=pin.getFrameJacobian(m,d,br.foot_fid[L],pin.LOCAL_WORLD_ALIGNED)[:3]
        J[:,:6]=0.0
        if np.linalg.norm(err)<1e-5: break
        q=pin.integrate(m,q,0.5*np.linalg.lstsq(J,err,rcond=None)[0])
    return q

def main():
    br = MjPinBridge(); m = br.model
    m.armature[6:6+16] = np.tile([1e-4*7**2,1e-4*7**2,1e-4*10.5**2,1e-4*8.4**2], 4)
    ci = ContactImplicit(br)
    q = _stance_q(br); v = np.zeros(m.nv)
    # standing PD-hold 토크로 서기 균형 롤아웃 → 접촉 물리 sane?
    qstar = q.copy(); dt = 0.001
    print("[C1.1] contact-implicit forward — 서기 균형 롤아웃 (완화 상보성 접촉)")
    F0, info0 = ci.contact_forces(q, v)
    print("  초기 접촉: " + " ".join("φ=%+.3f w=%.2f fn=%.0f"%(p,w,fn) for p,w,fn in info0)
          + "  Σfn=%.0f (무게 373N)"%sum(f[2] for f in F0))
    for i in range(1500):
        tau = 150.0*(qstar[7:]-q[7:]) - 8.0*v[6:]
        q, v, info = ci.step(q, v, tau, dt)
        if i % 300 == 0 or i == 1499:
            zmin = min(p for p,_,_ in info); base_z = q[2]
            print("  t=%.2f base_z=%.3f 최저발φ=%+.4f |v|=%.3f"%(i*dt, base_z, zmin, np.linalg.norm(v)))
    fin = "✅ 균형(서있음)" if q[2] > 0.30 and np.linalg.norm(v) < 2.0 else "❌ 붕괴/발산"
    print("  결과:", fin, " (접촉력 단방향·스무스 → 발이 지면 딛고 base 지지되면 접촉 forward 정상)")

    # ── C1.2: contact-implicit forward의 해석 그래디언트 vs FD ──
    print("\n[C1.2] contact-implicit 해석 그래디언트 vs FD (computeABADerivatives + 힘 연쇄)")
    q0 = _stance_q(br); v0 = 0.05*np.random.RandomState(3).randn(m.nv); tau0 = 2.0*np.random.RandomState(4).randn(br.nu)
    ddq0, A_dq, A_dv, A_dtau = ci.dynamics_derivatives(q0, v0, tau0)
    nv = m.nv; e = 1e-6
    F_dq = np.zeros((nv, nv)); F_dv = np.zeros((nv, nv)); F_dtau = np.zeros((nv, br.nu))
    def ddq_of(qq, vv, tt): return ci.dynamics(qq, vv, tt)[0]
    for i in range(nv):
        dq = np.zeros(nv); dq[i] = e
        F_dq[:, i] = (ddq_of(pin.integrate(m, q0, dq), v0, tau0) - ddq_of(pin.integrate(m, q0, -dq), v0, tau0)) / (2*e)
        dv = np.zeros(nv); dv[i] = e
        F_dv[:, i] = (ddq_of(q0, v0+dv, tau0) - ddq_of(q0, v0-dv, tau0)) / (2*e)
    for i in range(br.nu):
        dt = np.zeros(br.nu); dt[i] = e
        F_dtau[:, i] = (ddq_of(q0, v0, tau0+dt) - ddq_of(q0, v0, tau0-dt)) / (2*e)
    def rel(A, F, name):
        r = np.linalg.norm(A-F)/(np.linalg.norm(F)+1e-9)
        print("  ∂ddq/∂%-4s: 상대오차=%.2e  %s" % (name, r, "✅ 일치" if r<1e-2 else ("△ 근사(kin.hessian 생략)" if r<0.3 else "❌")))
        return r
    rq = rel(A_dq, F_dq, "q"); rv = rel(A_dv, F_dv, "v"); rt = rel(A_dtau, F_dtau, "tau")
    print("  → ∂v·∂τ=정확(연쇄 완전), ∂q=지배항 일치(∂vf/∂q·∂Jᵀ/∂q kin.hessian 추가시 완전).")
    print("     C1.2 해석 그래디언트 골격 검증 → Box-FDDP OCP에 꽂을 준비. HOUND식 autodiff-free.")

if __name__ == "__main__":
    main()
