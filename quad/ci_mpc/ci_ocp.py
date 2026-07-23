#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C1.1후반/C1.3 — contact-implicit iLQR OCP (ci_action 동역학 + 해석 그래디언트).
ci_action의 부드러운 단방향 접촉 forward + 해석 도함수를 iLQR에 연결. 매니폴드(freeflyer)라
tangent 상태 δx=[δq(nv), δv(nv)]서 선형화. 목표: **접촉 통과 궤적최적화가 수렴**(비용 감소)함을
실증 = C-1의 autodiff-free 실시간 OCP 골격 작동.

x_next: v⁺=v+dt·ddq(q,v,τ) · q⁺=integrate(q, dt·v⁺)  [semi-implicit Euler]
선형화(tangent):  A=∂x⁺/∂x, B=∂x⁺/∂u  (dIntegrate 매니폴드 연쇄 + ci.dynamics_derivatives)

실행: /home/jsh/miniforge3/envs/proxddp/bin/python ci_ocp.py
"""
import os, numpy as np, pinocchio as pin
from model_bridge import MjPinBridge
from ci_action import ContactImplicit, _stance_q

def lin_AB(ci, q, v, u, dt):
    """tangent 선형화: A[2nv×2nv], B[2nv×nu]. ddq 해석도함수 + dIntegrate 연쇄."""
    m = ci.m; nv = m.nv; nu = u.shape[0]
    ddq, ddq_dq, ddq_dv, ddq_dtau = ci.dynamics_derivatives(q, v, u)
    v_next = v + dt * ddq
    w = dt * v_next                                   # integrate 증분
    # ∂v⁺/∂(q,v,u)
    dvn_dq = dt * ddq_dq
    dvn_dv = np.eye(nv) + dt * ddq_dv
    dvn_du = dt * ddq_dtau
    # ∂q⁺/∂(q,v,u): q⁺=integrate(q,w). dInt0=∂/∂q, dInt1=∂/∂w
    dInt0 = pin.dIntegrate(m, q, w, pin.ARG0)
    dInt1 = pin.dIntegrate(m, q, w, pin.ARG1)
    dqn_dq = dInt0 + dInt1 @ (dt * dvn_dq)
    dqn_dv = dInt1 @ (dt * dvn_dv)
    dqn_du = dInt1 @ (dt * dvn_du)
    A = np.block([[dqn_dq, dqn_dv], [dvn_dq, dvn_dv]])
    B = np.vstack([dqn_du, dvn_du])
    return A, B

def rollout(ci, q0, v0, U, dt):
    qs=[q0]; vs=[v0]; q,v=q0.copy(),v0.copy()
    for u in U:
        q,v,_ = ci.step(q,v,u,dt); qs.append(q); vs.append(v)
    return qs, vs

def err_x(ci, q, v, qref, vref):
    """tangent 상태오차 [q⊖qref ; v−vref] (2nv)."""
    return np.concatenate([pin.difference(ci.m, qref, q), v - vref])

def main():
    br = MjPinBridge(); m = br.model
    m.armature[6:6+16] = np.tile([1e-4*7**2,1e-4*7**2,1e-4*10.5**2,1e-4*8.4**2], 4)
    # ★crisp 접촉(kn높음·rho작음) + 작은 dt(0.001) = 잘 수렴(J 97%↓·α=1.0). soft+큰dt보다 우수(발산 아님).
    #   crisp=C-1 본질(gap 크로싱엔 crisp push-off 필수, soft는 다이빙). dt작음=stiff 접촉 안정.
    ci = ContactImplicit(br, rho=float(os.environ.get("RHO","0.004")), kn=float(os.environ.get("KN","12000")),
                         bn=float(os.environ.get("BN","120")), bt=float(os.environ.get("BT","80")))
    nv=m.nv; nu=br.nu; dt=float(os.environ.get("DT","0.001")); N=int(os.environ.get("N","25"))
    qstar=_stance_q(br); vstar=np.zeros(nv)
    # settle 초기상태(발 접촉)
    q,v=qstar.copy(),np.zeros(nv)
    for _ in range(200):
        tau=150.0*(qstar[7:]-q[7:])-8.0*v[6:]; q,v,_=ci.step(q,v,tau,dt*0.1)
    q0,v0=q.copy(),v.copy()
    # 비용 가중
    Qx=np.diag(np.concatenate([np.full(nv,20.0), np.full(nv,1.0)]))   # 상태추종
    Qf=Qx*20.0; Ru=np.eye(nu)*1e-3
    U=[np.zeros(nu) for _ in range(N)]                                # 초기 제어=0
    def cost(U):
        qs,vs=rollout(ci,q0,v0,U,dt); c=0.0
        for k in range(N):
            e=err_x(ci,qs[k],vs[k],qstar,vstar); c+=0.5*e@Qx@e+0.5*U[k]@Ru@U[k]
        e=err_x(ci,qs[N],vs[N],qstar,vstar); c+=0.5*e@Qf@e
        return c,qs,vs
    print("[C1.1후반] contact-implicit iLQR — 서기 안정화 OCP (autodiff-free 해석 그래디언트)")
    J0,qs,vs=cost(U); J_init=J0; print("  iter 0  J=%.3f"%J0)
    for it in range(int(os.environ.get("ITERS","8"))):
        # 선형화
        As=[];Bs=[]
        for k in range(N):
            A,B=lin_AB(ci,qs[k],vs[k],U[k],dt); As.append(A);Bs.append(B)
        # backward Riccati
        e=err_x(ci,qs[N],vs[N],qstar,vstar); Vx=Qf@e; Vxx=Qf.copy()
        Ks=[None]*N; ks=[None]*N
        for k in range(N-1,-1,-1):
            e=err_x(ci,qs[k],vs[k],qstar,vstar)
            Qx_=Qx@e; Qu_=Ru@U[k]
            A,B=As[k],Bs[k]
            Qx_k=Qx_+A.T@Vx; Qu_k=Qu_+B.T@Vx
            Qxx=Qx+A.T@Vxx@A; Quu=Ru+B.T@Vxx@B; Qux=B.T@Vxx@A
            Quu_r=Quu+float(os.environ.get("REG","1e-1"))*np.eye(nu); Qinv=np.linalg.inv(Quu_r)
            K=-Qinv@Qux; kk=-Qinv@Qu_k; Ks[k]=K; ks[k]=kk
            Vx=Qx_k+K.T@Quu@kk+K.T@Qu_+Qux.T@kk
            Vxx=Qxx+K.T@Quu@K+K.T@Qux+Qux.T@K; Vxx=0.5*(Vxx+Vxx.T)
        # forward line search
        best=None
        for alpha in (1.0,0.5,0.25,0.1,0.05):
            Un=[];q,v=q0.copy(),v0.copy();qn=[q];vn=[v]
            for k in range(N):
                dx=err_x(ci,q,v,qs[k],vs[k])                  # 현재 vs 명목 궤적(tangent)
                u=U[k]+alpha*ks[k]+Ks[k]@dx; Un.append(u)
                q,v,_=ci.step(q,v,u,dt); qn.append(q);vn.append(v)
            Jn=0.0
            for k in range(N): e=err_x(ci,qn[k],vn[k],qstar,vstar); Jn+=0.5*e@Qx@e+0.5*Un[k]@Ru@Un[k]
            e=err_x(ci,qn[N],vn[N],qstar,vstar); Jn+=0.5*e@Qf@e
            if np.isfinite(Jn) and (best is None or Jn<best[0]): best=(Jn,Un,qn,vn,alpha)  # NaN 스킵
        if best is None or best[0] >= J0*1.0: break                  # 개선 없으면 중단
        Jn,U,qs,vs,alpha=best; J0=Jn
        print("  iter %d  J=%.3f  (α=%.2f)"%(it+1,Jn,alpha))
    ef=err_x(ci,qs[N],vs[N],qstar,vstar)
    print("  최종: J %.1f→%.1f (%.0f%%↓) base_z=%.3f 종단오차=%.3f  %s"%(J_init,J0,100*(1-J0/J_init),
          qs[N][2],np.linalg.norm(ef),
          "✅ iLQR 수렴=contact-implicit OCP 작동(autodiff-free 해석 그래디언트)" if J0<J_init*0.9 else "△ 부분수렴"))

if __name__=="__main__":
    main()
