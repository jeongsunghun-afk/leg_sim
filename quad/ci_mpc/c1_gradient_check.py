#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C1.2 de-risk 증명 — Pinocchio 접촉 해석 그래디언트 vs 유한차분(FD).
C-1의 최고난도(C1.2)=하드접촉 동역학의 해석 그래디언트 ∂ddq/∂(q,v,τ). HOUND는 이걸
손유도(닫힌형식 ∂λ/∂q)했다. ★Pinocchio 4.0의 computeConstraintDynamicsDerivatives가
이걸 내장 → 여기서 해석≈FD 검증되면 **C1.2를 Pinocchio 내장으로 풀 수 있음이 증명**.

실행: /home/jsh/miniforge3/envs/proxddp/bin/python c1_gradient_check.py
"""
import numpy as np, pinocchio as pin
from model_bridge import MjPinBridge, FEET

FOOT_R = 0.025

def stance_q(br):
    """발을 힙 아래 지면(z=FOOT_R)에 두는 standing pin q (간이 IK)."""
    m, d = br.model, br.data
    q = pin.neutral(m); q[2] = 0.42
    tgt = {'FL':[0.30,0.16], 'FR':[0.30,-0.16], 'HL':[-0.30,0.16], 'HR':[-0.30,-0.16]}
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
    br = MjPinBridge(); m, d = br.model, br.data
    m.armature[6:6+16] = np.tile([1e-4*7**2,1e-4*7**2,1e-4*10.5**2,1e-4*8.4**2], 4)   # 반사관성
    q = stance_q(br); nv = m.nv
    v = np.zeros(nv) + 0.05*np.random.RandomState(0).randn(nv)   # 비영 속도(그래디언트 일반성)
    tau = np.zeros(nv); tau[6:] = 2.0*np.random.RandomState(1).randn(nv-6)

    # 접촉모델(스탠스 4발, 3D)
    cms = pin.StdVec_RigidConstraintModel()
    for L in FEET:
        fr = m.frames[br.foot_fid[L]]
        pl = fr.placement * pin.SE3(np.eye(3), np.array([0.,0.,-FOOT_R]))
        cms.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, m, fr.parentJoint, pl,
                                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED))
    cds = pin.StdVec_RigidConstraintData()
    for cm in cms: cds.append(cm.createData())
    prox = pin.ProximalSettings(1e-12, 1e-8, 50)
    pin.initConstraintDynamics(m, d, cms, cds)

    def ddq_of(qq, vv, tt):
        return pin.constraintDynamics(m, d, qq, vv, tt, cms, cds, prox).copy()

    # ── 해석 도함수 ──
    ddq0 = ddq_of(q, v, tau)
    pin.computeConstraintDynamicsDerivatives(m, d, cms, cds)
    A_dq = np.array(d.ddq_dq).copy(); A_dv = np.array(d.ddq_dv).copy(); A_dtau = np.array(d.ddq_dtau).copy()

    # ── 유한차분 ──
    eps = 1e-6
    F_dq = np.zeros((nv, nv)); F_dv = np.zeros((nv, nv)); F_dtau = np.zeros((nv, nv))
    for i in range(nv):
        dq = np.zeros(nv); dq[i] = eps
        F_dq[:, i] = (ddq_of(pin.integrate(m, q, dq), v, tau) - ddq0) / eps
        dv = np.zeros(nv); dv[i] = eps
        F_dv[:, i] = (ddq_of(q, v + dv, tau) - ddq0) / eps
        dt = np.zeros(nv); dt[i] = eps
        F_dtau[:, i] = (ddq_of(q, v, tau + dt) - ddq0) / eps

    def rel(A, F, name):
        num = np.linalg.norm(A - F); den = np.linalg.norm(F) + 1e-9
        print("  ∂ddq/∂%-4s : 해석 vs FD  절대차 max=%.2e  상대오차=%.2e  %s"
              % (name, np.abs(A-F).max(), num/den, "✅ 일치" if num/den < 1e-3 else "❌ 불일치"))
        return num/den

    print("[C1.2 de-risk] Pinocchio 접촉 해석 그래디언트 vs 유한차분 (4발 스탠스, v≠0)")
    r1 = rel(A_dq, F_dq, "q"); r2 = rel(A_dv, F_dv, "v"); r3 = rel(A_dtau, F_dtau, "tau")
    ok = max(r1, r2, r3) < 1e-3
    print("\n★결론: %s" % ("Pinocchio 접촉 해석 도함수 = FD 일치 → C-1의 C1.2(해석 그래디언트)가 "
          "손유도 아닌 Pinocchio 내장으로 정확히 풀림. C-1 최대리스크 해소." if ok else
          "불일치 — 접촉모델/설정 재점검 필요."))
    # 접촉력 미분도 존재 확인(dlambda) — 완화상보성 레이어 설계 참고
    print("  (접촉력 미분 dλ/dq,dv,dτ도 내장: shape %s)" % (np.array(d.dlambda_dq).shape,))

if __name__ == "__main__":
    main()
