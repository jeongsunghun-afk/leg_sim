#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C1.3 — receding-horizon MPC (HOUND 40Hz 형태). 걸음은 단일 OCP가 아니라 **폐루프서 창발**.
짧은 horizon FDDP를 매 제어스텝 재풀이 → 첫 제어 적용 → sim 한 스텝 → 재계획(warm-start).
base는 현재 위치서 전진 속도 참조 + foot-slip cost(발 낮으면 no-slip) → 전진하려면 발 들수밖에→걸음.

solver=multiple-shooting FDDP(gap 주입·feasibility-driven·merit), 접촉=forward stiff+backward 완화,
cost=regulating(몸통SE3 속도) + foot-slip/clearance(eq22) + air-time(φ²). 발궤적 처방 없음.
실행: /home/jsh/miniforge3/envs/proxddp/bin/python ci_mpc_walk.py
env: VX MPC_STEPS N DT NSUB SOLVE_ITERS CF AIR_W W_BASE VXVEL
"""
import os, numpy as np, pinocchio as pin
from model_bridge import MjPinBridge
from ci_action import ContactImplicit, _stance_q, FEET, FOOT_R
from ci_ocp import lin_AB, lin_AB_kkt

def main():
    br=MjPinBridge(); m=br.model; dd=br.data; nv=m.nv; nu=br.nu
    m.armature[6:6+16]=np.tile([1e-4*7**2,1e-4*7**2,1e-4*10.5**2,1e-4*8.4**2],4)
    ci=ContactImplicit(br, rho=0.004, kn=12000, bn=120, bt=80)   # forward stiff / backward=완화(env KN_G 등)
    DT=float(os.environ.get("DT","0.02")); N=int(os.environ.get("N","15"))
    NSUB=int(os.environ.get("NSUB","5")); ITERS=int(os.environ.get("SOLVE_ITERS","5"))
    SIM_NSUB=int(os.environ.get("SIM_NSUB","20"))   # ★적용(sim) 스텝은 finer substep=접촉 적분 안정(0.001급)
    # ★★HOUND §6.3 아키텍처: sim=step_kkt(안정 hard 접촉)·저수준=PD+FF fine rate로 MPC 계획 추종.
    #   planner=soft(빠름)·sim=hard·PD+FF가 다리(제어 유지간격 짧아야 안정=CTRL_DT≤0.001).
    HARD=int(os.environ.get("HARD","1"))            # 1=step_kkt hard sim, 0=soft sim(구)
    CTRL_DT=float(os.environ.get("CTRL_DT","0.001"))# 저수준 제어 갱신간격(≤0.001=안정)
    KKT_NSUB=int(os.environ.get("KKT_NSUB","2"))    # step_kkt substep(CTRL_DT/KKT_NSUB=sim h≈0.0005)
    KP_T=float(os.environ.get("KP_T","150")); KD_T=float(os.environ.get("KD_T","12"))  # 추종 PD 게인
    # ★★논문 핵심(§5.3 eq26): optimizer forward=hard impulse(step_kkt) + backward=**λ(접촉임펄스)의
    #   해석 그래디언트** ∂λ/∂(q,v,u)가 ∂ddq에 흘러듦. Pinocchio computeConstraintDynamicsDerivatives
    #   가 제공(C1.0 FD검증). 기존 lin_AB는 soft force 그래디언트라 hard impulse ∂λ 아님 → KKT 그래디언트로.
    HARD_PLAN=int(os.environ.get("HARD_PLAN","0")); PLAN_NSUB=int(os.environ.get("PLAN_NSUB","10"))
    MPC_STEPS=int(os.environ.get("MPC_STEPS","120")); VX=float(os.environ.get("VX","0.3"))
    CF=float(os.environ.get("CF","2000")); AIR_W=float(os.environ.get("AIR_W","100")); C1S=-30.0
    W_BASE=float(os.environ.get("W_BASE","40")); VXVEL=float(os.environ.get("VXVEL","80"))
    REG=float(os.environ.get("REG","1e-1")); GAP_W=float(os.environ.get("GAP_W","100"))
    qstar=_stance_q(br); vstar=np.zeros(nv)
    # settle → 초기상태·지지토크
    q,v=qstar.copy(),np.zeros(nv); tau_hold=np.zeros(nu)
    for _ in range(200): tau_hold=150.0*(qstar[7:]-q[7:])-8.0*v[6:]; q,v,_=ci.step(q,v,tau_hold,DT*0.05)
    q0,v0=q.copy(),v.copy()

    def sdiff(a,b): return np.concatenate([pin.difference(m,a[0],b[0]), b[1]-a[1]])
    def sint(a,t): return (pin.integrate(m,a[0],t[:nv]), a[1]+t[nv:])

    # 비용 가중: 관절/base. base는 z·자세 강하게(균형)·x는 속도추종
    W_BVEL=float(os.environ.get("W_BVEL","30"))    # ★base 속도 감쇠(vz·각속도)=스텝 중 crash 억제
    Qx=np.diag(np.concatenate([np.full(nv,20.0), np.full(nv,1.0)]))
    Qx[0]*=0.1; Qx[nv]*=VXVEL; Qx[2]*=W_BASE; Qx[3]*=W_BASE; Qx[4]*=W_BASE
    Qx[nv+2]*=W_BVEL; Qx[nv+3]*=W_BVEL; Qx[nv+4]*=W_BVEL; Qx[nv+5]*=W_BVEL   # base vz·roll·pitch·yaw rate 감쇠
    Qf=Qx*10.0; Ru=np.eye(nu)*float(os.environ.get("RU_W","1e-3"))   # 제어 정규화(↑=부드러움·과보정 억제)

    fids=[br.foot_fid[L] for L in FEET]
    def _foot_kin(q,v):
        """각 발 (phi, J[3×nv], vf). FK 1회."""
        pin.forwardKinematics(m,dd,q,v); pin.updateFramePlacements(m,dd); pin.computeJointJacobians(m,dd,q)
        out=[]
        for fid in fids:
            oMf=dd.oMf[fid]
            phi=(oMf.translation+oMf.rotation@np.array([0.,0.,-FOOT_R]))[2]
            J=pin.getFrameJacobian(m,dd,fid,pin.LOCAL_WORLD_ALIGNED)[:3].copy()
            out.append((phi,J,J@v))
        return out
    def foot_val(q,v):
        """foot-slip/clearance(eq22)+air-time(φ²) 값만(싸게, line search용)."""
        c=0.0
        for phi,J,vf in _foot_kin(q,v):
            w2=vf[:2]@vf[:2]; S=1.0/(1.0+np.exp(-C1S*phi)); c+=CF*S*w2
            if AIR_W>0 and phi>0: c+=AIR_W*phi*phi
        return c
    def foot_grad(q,v):
        """c,g[2nv],H — ★∂vt/∂q를 FD로 정확히(getFrameVelocityDerivatives convention 회피). backward 전용."""
        base=_foot_kin(q,v)
        e=1e-6; dvf=[np.zeros((3,nv)) for _ in fids]                  # ∂vf/∂q FD 1스윕(전 발)
        for j in range(nv):
            dq=np.zeros(nv); dq[j]=e
            for sgn,qq in ((1.0,pin.integrate(m,q,dq)),(-1.0,pin.integrate(m,q,-dq))):
                for i,(_,_,vfp) in enumerate(_foot_kin(qq,v)): dvf[i][:,j]+=sgn*vfp/(2*e)
        c=0.0; g=np.zeros(2*nv); H=np.zeros((2*nv,2*nv))
        for (phi,J,vf),dvq in zip(base,dvf):
            vt=vf[:2]; w2=vt@vt; Jz=J[2]; S=1.0/(1.0+np.exp(-C1S*phi)); Sp=S*(1.0-S)
            c+=CF*S*w2
            g[:nv]+=CF*(Sp*C1S*Jz*w2 + S*2.0*(vt[0]*dvq[0]+vt[1]*dvq[1]))  # ★∂vt/∂q 포함(정확)
            g[nv:]+=CF*S*2.0*(vt[0]*J[0]+vt[1]*J[1]); Jt=J[:2]; H[nv:,nv:]+=CF*2.0*S*(Jt.T@Jt)
            if AIR_W>0 and phi>0: c+=AIR_W*phi*phi; g[:nv]+=AIR_W*2*phi*Jz; H[:nv,:nv]+=AIR_W*2*np.outer(Jz,Jz)
        return c,g,H

    HARD_FWD=int(os.environ.get("HARD_FWD","0"))   # forward: 1=step_kkt(fine dt 필요), 0=soft(빠름·안정)
    def fwd(q,v,u):                                                  # ★optimizer forward
        if HARD_FWD: return ci.step_kkt(q,v,u,DT,PLAN_NSUB)[:2]
        return ci.step(q,v,u,DT,NSUB)[:2]
    def linAB(q,v,u):                                               # ★backward: HARD_PLAN=1→exact λ그래디언트
        if HARD_PLAN: return lin_AB_kkt(ci,q,v,u,DT)
        return lin_AB(ci,q,v,u,DT,NSUB)
    def solve(x0, Xref, U, X):
        """짧은 horizon FDDP 몇 iter(warm-start). return X,U."""
        X=[x0]+list(X[1:])                                           # 현재 실제상태로 앵커
        def evalt(X,U):
            gaps=[None]*(N+1); J=0.0; gs=0.0
            for k in range(N):
                qs,vs=fwd(X[k][0],X[k][1],U[k])
                gaps[k+1]=sdiff(X[k+1],(qs,vs)); gs+=np.linalg.norm(gaps[k+1])
                e=sdiff(Xref[k],X[k]); J+=0.5*e@Qx@e+0.5*U[k]@Ru@U[k]+foot_val(X[k][0],X[k][1])
            e=sdiff(Xref[N],X[N]); J+=0.5*e@Qf@e
            return gaps,J,J+GAP_W*gs
        gaps,J,M=evalt(X,U)
        for _ in range(ITERS):
            As=[];Bs=[]
            for k in range(N): A,B=linAB(X[k][0],X[k][1],U[k]); As.append(A);Bs.append(B)
            e=sdiff(Xref[N],X[N]); Vx=Qf@e; Vxx=Qf.copy(); Ks=[None]*N; ks=[None]*N
            for k in range(N-1,-1,-1):
                Vxp=Vx+Vxx@gaps[k+1]; e=sdiff(Xref[k],X[k]); lx=Qx@e; lu=Ru@U[k]
                _,fg,fH=foot_grad(X[k][0],X[k][1]); lx=lx+fg; Qxx_k=Qx+fH
                A,B=As[k],Bs[k]; Qx_=lx+A.T@Vxp; Qu_=lu+B.T@Vxp
                Qxx=Qxx_k+A.T@Vxx@A; Quu=Ru+B.T@Vxx@B; Qux=B.T@Vxx@A
                try: Qinv=np.linalg.inv(Quu+REG*np.eye(nu))       # 발산 시 특이행렬 → pinv 폴백
                except np.linalg.LinAlgError: Qinv=np.linalg.pinv(Quu+REG*np.eye(nu))
                K=-Qinv@Qux; kk=-Qinv@Qu_; Ks[k]=K; ks[k]=kk
                Vx=Qx_+K.T@Quu@kk+K.T@Qu_+Qux.T@kk; Vxx=Qxx+K.T@Quu@K+K.T@Qux+Qux.T@K; Vxx=0.5*(Vxx+Vxx.T)
            best=None
            for alpha in (1.0,0.5,0.25,0.1,0.05):
                Xn=[X[0]]; Un=[]; ok=True
                for k in range(N):
                    dx=sdiff(X[k],Xn[k]); u=U[k]+alpha*ks[k]+Ks[k]@dx; Un.append(u)
                    qs,vs=fwd(Xn[k][0],Xn[k][1],u)
                    if not np.all(np.isfinite(qs)): ok=False; break
                    Xn.append(sint((qs,vs),-(1.0-alpha)*gaps[k+1]))
                if not ok or not np.all(np.isfinite(Xn[-1][0])): continue
                _,Jn,Mn=evalt(Xn,Un)
                if np.isfinite(Mn) and (best is None or Mn<best[0]): best=(Mn,Xn,Un)
            if best is None or best[0]>=M*0.999999: break
            Mn,X,U=best; gaps,J,M=evalt(X,U)
        return X,U

    # ===== MPC 폐루프 =====
    x=(q0.copy(),v0.copy()); U=[tau_hold.copy() for _ in range(N)]
    X=[x]+[(qstar.copy(),vstar.copy()) for _ in range(N)]
    print("[C1.3] receding-horizon MPC — 걸음 창발 (N=%d·DT=%.3f·%dHz재풀이·VX=%.2f)"%(N,DT,int(1/DT),VX))
    x0_base=x[0][0]; hist_z=[]; hist_x=[]
    for s in range(MPC_STEPS):
        bx=x[0][0]                                                   # 현재 base_x서 전진 참조
        Xref=[]
        for k in range(N+1):
            qk=qstar.copy(); qk[0]=bx+VX*k*DT; vk=vstar.copy(); vk[0]=VX; Xref.append((qk,vk))
        X,U=solve(x,Xref,U,X)
        if HARD:                                                    # ★hard sim + PD+FF fine-rate 계획 추종
            ci.margin=0.004
            q_tgt,v_tgt=X[1]; u_ff=U[0]; q_c,v_c=x[0].copy(),x[1].copy()
            for _ in range(int(round(DT/CTRL_DT))):                 # 저수준 제어 갱신(fine rate)
                tau=u_ff+KP_T*(q_tgt[7:]-q_c[7:])+KD_T*(v_tgt[6:]-v_c[6:])   # FF+PD 계획 추종
                q_c,v_c,_=ci.step_kkt(q_c,v_c,tau,CTRL_DT,nsub=KKT_NSUB)
                if not np.all(np.isfinite(q_c)): break
            x=(q_c,v_c)
        else:
            x=ci.step(x[0],x[1],U[0],DT,SIM_NSUB)                   # (구) soft sim
        hist_z.append(x[0][2]); hist_x.append(x[0][0]-x0_base)
        U=U[1:]+[U[-1].copy()]; X=X[1:]+[X[-1]]                     # warm-start shift
        if (s+1)%15==0 or not np.isfinite(x[0][2]):
            print("  step %3d  t=%.2fs  전진=%.3fm  base_z=%.3f  vx=%.2f"%(s+1,(s+1)*DT,x[0][0]-x0_base,x[0][2],x[1][0]))
        if not np.all(np.isfinite(x[0])): print("  ✗ 발산"); break
    fwd=x[0][0]-x0_base; T=(s+1)*DT
    zmin=min(hist_z) if hist_z else 0
    print("  최종: %.2fs동안 전진 %.3fm (평균 %.2f m/s, 목표 %.2f) base_z 최저 %.3f  %s"%(
        T,fwd,fwd/T,VX,zmin,
        "✅ 걸음(전진+균형유지)" if fwd>0.15 and zmin>0.30 else
        "△ 전진하나 균형약함" if fwd>0.15 else "✗ 전진 부족"))

if __name__=="__main__":
    main()
