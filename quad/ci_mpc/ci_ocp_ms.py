#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C1.3 — multiple-shooting FDDP (Mastalli 2020, "Crocoddyl") on contact-implicit dynamics.
single-shooting iLQR(ci_ocp.py)은 긴 horizon서 open-loop nominal 발산으로 ~200ms서 깨짐.
FDDP는 각 노드 상태 x_k를 결정변수로 두고 **gap(dynamics defect)이 물리 불일치를 흡수** →
nominal이 물리적으로 안정할 필요 없음(모든 노드=서기로 초기화 가능). gap을 (1-α)로 점진 폐쇄.

핵심(vs ci_ocp 단일슈팅):
  gap  f̄_{k+1} = step(x_k,u_k) ⊖ x_{k+1}                       (dynamics defect)
  backward: V_x⁺ = V_x + V_xx·f̄  (gap 주입)
  forward:  x̂_{k+1} = step(x̂_k,û_k) ⊖ (1-α)·f̄_{k+1}          (feasibility-driven, gap 수축)

접촉=forward stiff(물리 crisp) + backward 완화(KN_G, well-conditioned) [ci_action relax].
실행: WARM 무관. env: DT NSUB N KN KN_G RHO_G BN_G BT_G REG ITERS
"""
import os, numpy as np, pinocchio as pin
from model_bridge import MjPinBridge
from ci_action import ContactImplicit, _stance_q
from ci_ocp import lin_AB, err_x

def main():
    br = MjPinBridge(); m = br.model
    m.armature[6:6+16] = np.tile([1e-4*7**2,1e-4*7**2,1e-4*10.5**2,1e-4*8.4**2], 4)
    ci = ContactImplicit(br, rho=float(os.environ.get("RHO","0.004")), kn=float(os.environ.get("KN","12000")),
                         bn=float(os.environ.get("BN","120")), bt=float(os.environ.get("BT","80")))
    nv=m.nv; nu=br.nu; dt=float(os.environ.get("DT","0.01")); N=int(os.environ.get("N","50"))
    nsub=int(os.environ.get("NSUB","10")); REG=float(os.environ.get("REG","1e-1"))
    qstar=_stance_q(br); vstar=np.zeros(nv)
    # settle → 초기상태 + 지지토크
    q,v=qstar.copy(),np.zeros(nv); tau_hold=np.zeros(nu)
    for _ in range(200):
        tau_hold=150.0*(qstar[7:]-q[7:])-8.0*v[6:]; q,v,_=ci.step(q,v,tau_hold,dt*0.1)
    q0,v0=q.copy(),v.copy()

    # 상태 매니폴드 연산: x=(q,v). tangent 2nv=[dq(nv),dv(nv)]
    def sdiff(a,b):  # b ⊖ a (a→b tangent)
        return np.concatenate([pin.difference(m,a[0],b[0]), b[1]-a[1]])
    def sint(a,t):   # a ⊕ t
        return (pin.integrate(m,a[0],t[:nv]), a[1]+t[nv:])

    Qx=np.diag(np.concatenate([np.full(nv,20.0), np.full(nv,1.0)]))
    Qf=Qx*20.0; Ru=np.eye(nu)*1e-3
    xstar=(qstar,vstar)
    # ★nominal 초기화: 모든 노드=서기(발산 불가, gap이 불일치 흡수) · U=지지토크
    X=[(q0.copy(),v0.copy())]+[(qstar.copy(),vstar.copy()) for _ in range(N)]
    X[0]=(q0.copy(),v0.copy())
    U=[tau_hold.copy() for _ in range(N)]

    GAP_W=float(os.environ.get("GAP_W","50"))    # merit gap 벌점(feasibility 구동)
    def eval_traj(X,U):
        """gap f̄_{k+1}, 비용 J, merit=J+GAP_W·Σ|gap|(feasibility 포함)."""
        gaps=[None]*(N+1); J=0.0; gsum=0.0
        for k in range(N):
            qs,vs,_=ci.step(X[k][0],X[k][1],U[k],dt,nsub)
            gaps[k+1]=sdiff(X[k+1],(qs,vs))          # step ⊖ x_{k+1}
            gsum+=np.linalg.norm(gaps[k+1])
            e=sdiff(xstar,X[k]); J+=0.5*e@Qx@e+0.5*U[k]@Ru@U[k]
        e=sdiff(xstar,X[N]); J+=0.5*e@Qf@e
        return gaps,J,J+GAP_W*gsum

    print("[C1.3] multiple-shooting FDDP — contact-implicit (gap 주입 + feasibility-driven)")
    gaps,J,M=eval_traj(X,U); J_init=J
    print("  iter 0  J=%.3f  merit=%.3f |gap|max=%.3f"%(J, M, max(np.linalg.norm(g) for g in gaps[1:])))
    for it in range(int(os.environ.get("ITERS","30"))):
        # 선형화
        As=[];Bs=[]
        for k in range(N):
            A,B=lin_AB(ci,X[k][0],X[k][1],U[k],dt,nsub); As.append(A);Bs.append(B)
        # backward (gap 주입)
        e=sdiff(xstar,X[N]); Vx=Qf@e; Vxx=Qf.copy()
        Ks=[None]*N; ks=[None]*N
        for k in range(N-1,-1,-1):
            Vxp=Vx+Vxx@gaps[k+1]                     # ★gap 주입 V_x⁺
            e=sdiff(xstar,X[k]); lx=Qx@e; lu=Ru@U[k]
            A,B=As[k],Bs[k]
            Qx_=lx+A.T@Vxp; Qu_=lu+B.T@Vxp
            Qxx=Qx+A.T@Vxx@A; Quu=Ru+B.T@Vxx@B; Qux=B.T@Vxx@A
            Quu_r=Quu+REG*np.eye(nu); Qinv=np.linalg.inv(Quu_r)
            K=-Qinv@Qux; kk=-Qinv@Qu_; Ks[k]=K; ks[k]=kk
            Vx=Qx_+K.T@Quu@kk+K.T@Qu_+Qux.T@kk
            Vxx=Qxx+K.T@Quu@K+K.T@Qux+Qux.T@K; Vxx=0.5*(Vxx+Vxx.T)
        # forward line search (feasibility-driven, gap 수축). 수용=merit(비용+gap 벌점) 감소
        best=None
        for alpha in (1.0,0.5,0.25,0.1,0.05,0.02,0.01):
            Xn=[X[0]]; Un=[]; ok=True
            for k in range(N):
                dx=sdiff(X[k],Xn[k])                 # x̂_k ⊖ x_k
                u=U[k]+alpha*ks[k]+Ks[k]@dx; Un.append(u)
                qs,vs,_=ci.step(Xn[k][0],Xn[k][1],u,dt,nsub)
                if not np.all(np.isfinite(qs)): ok=False; break
                xn1=sint((qs,vs), -(1.0-alpha)*gaps[k+1])   # ★gap 수축
                Xn.append(xn1)
            if not ok or not np.all(np.isfinite(Xn[-1][0])): continue
            _,Jn,Mn=eval_traj(Xn,Un)                 # merit(gap 재계산 포함)
            if np.isfinite(Mn) and (best is None or Mn<best[0]): best=(Mn,Jn,Xn,Un,alpha)
        if best is None or best[0]>=M*0.999999: break
        Mn,Jn,X,U,alpha=best
        gaps,J,M=eval_traj(X,U)
        gmax=max(np.linalg.norm(g) for g in gaps[1:])
        print("  iter %d  J=%.3f  merit=%.3f (α=%.2f) |gap|max=%.4f"%(it+1,J,M,alpha,gmax))
    gmax=max(np.linalg.norm(g) for g in gaps[1:]); g0=7.594  # 초기 gap
    ef=sdiff(xstar,X[N])
    print("  최종: J=%.1f base_z=%.3f 종단오차=%.3f |gap|max=%.4f(초기%.1f)  %s"%(
          J,X[N][0][2],np.linalg.norm(ef),gmax,g0,
          "✅ FDDP 수렴(gap 폐쇄=feasible 긴 horizon OCP)" if gmax<0.05 else
          "△ 부분 폐쇄(%.0f%%)"%(100*(1-gmax/g0)) if gmax<g0*0.9 else "✗ 미폐쇄"))

if __name__=="__main__":
    main()
