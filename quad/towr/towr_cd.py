#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TOWR-in-CasADi — 모델기반 지형 궤적최적화(오프라인).
TOWR(Winkler 2018) 알고리즘을 CasADi+IPOPT로 우리 스택(MuJoCo-native)에 재구현.
C++ TOWR(ifopt/ROS/catkin) 빌드 없이, casadi 3.7 번들 IPOPT 사용.

핵심(A/B/D1이 못하는 것):
  ▸ footholds(발 착지점)를 지형 위 결정변수로 최적화 → 갭/계단 회피·정밀 배치
  ▸ 스윙 발이 지형 위로 클리어런스
  ▸ SRBD(단일강체) 동역학 + 마찰콘 + ROM(도달성) 제약
  ▸ base 높이가 지형 따라 상승(TAMOLS 등가, 여기선 최적화가 자동 결정)
출력: base pos/vel/ori, 발 위치, 접촉력 궤적 → MuJoCo WBIC/PD 추종(Phase1).

기반: proxddp/simple_mpc env (casadi 3.7 + IPOPT + pinocchio).
실행:  /home/jsh/simple-mpc/.pixi/envs/default/bin/python towr_cd.py
"""
import numpy as np, casadi as ca, os, json

# ───────── 로봇 SRBD 파라미터 (pinocchio URDF 추출값, [[b-elevation-tamols-towr-track]]) ─────────
MASS = 38.016                                   # kg (총질량)
G    = 9.81
INER = np.array([0.941, 2.521, 2.236])          # centroidal 회전관성 diag Ixx,Iyy,Izz
BASE_H = 0.50                                    # 명목 base 높이
MU   = 0.6                                       # 마찰계수
# 공칭 발위치(base 로컬, 서있는 자세) — ROM 박스 중심
FEET = ['FL','FR','HL','HR']
P_NOM = {'FL':np.array([ 0.30, 0.16]), 'FR':np.array([ 0.30,-0.16]),
         'HL':np.array([-0.30, 0.16]), 'HR':np.array([-0.30,-0.16])}
ROM_DXY = np.array([0.13, 0.09])                # 발 xy가 공칭±ROM_DXY 내(도달성)
FOOT_Z_LO, FOOT_Z_HI = -0.56, -0.40            # 발 z(base 로컬) 범위(다리 길이)
F_MAX = 2.0*MASS*G                              # 접촉력 상한(수직)


def terrain_height(x, y, kind='flat', **kw):
    """지형 높이 h(x,y). CasADi 심볼릭·numpy 양쪽 동작(smooth). kind별."""
    if kind == 'flat':
        return 0.0*x
    if kind == 'step':                          # x=x0에서 h만큼 올라가는 계단(smooth)
        x0 = kw.get('x0', 1.0); h = kw.get('h', 0.10); s = kw.get('sharp', 40.0)
        return h*0.5*(1+ca.tanh(s*(x-x0))) if isinstance(x, (ca.SX,ca.MX,ca.DM)) \
               else h*0.5*(1+np.tanh(s*(x-x0)))
    if kind == 'gap':                           # [x0,x1] 구간이 깊게 파인 갭(smooth 우물)
        x0=kw.get('x0',0.9); x1=kw.get('x1',1.2); d=kw.get('depth',0.30); s=kw.get('sharp',60.0)
        th = ca.tanh if isinstance(x,(ca.SX,ca.MX,ca.DM)) else np.tanh
        return -d*0.5*(th(s*(x-x0)) - th(s*(x-x1)))
    if kind == 'platgap':                       # 높은 플랫폼(plat) + 갭 슬롯(floor 노출)
        x0=kw.get('x0',1.0); x1=kw.get('x1',1.3); plat=kw.get('plat',0.20); s=kw.get('sharp',60.0)
        th = ca.tanh if isinstance(x,(ca.SX,ca.MX,ca.DM)) else np.tanh
        return plat - plat*0.5*(th(s*(x-x0)) - th(s*(x-x1)))
    return 0.0*x


def build_and_solve(kind='flat', tkw=None, N=40, dt=0.02, Tg=0.40, duty=0.5,
                    x_goal=0.8, vx_des=None, verbose=True, gait='trot', phase_off=0.0):
    """
    phase-based TO. footholds=지형 위 변수.
      gait: 'trot'(대각쌍 FL·HR/FR·HL, 동적) | 'crawl'(한발씩 스윙, 3발지지 정적안정)
      N/dt: 노드·간격 → 총시간 T=N·dt ; Tg: gait 주기, duty: stance 비율
      x_goal: 전진 목표(m)
    반환: dict(궤적) 또는 None(실패).
    """
    tkw = tkw or {}
    T = N*dt
    if vx_des is None: vx_des = x_goal/T
    th = lambda x,y: terrain_height(x,y,kind,**tkw)
    # base 높이 참조: 갭 위에서 base가 갭 바닥으로 빠지면 안 됨 → 지지면(플랫폼) 레벨 사용.
    #   platgap=플랫폼 상수레벨, 그 외=지형 추종.
    _bl = tkw.get('plat', 0.0) if kind=='platgap' else None
    bhref = (lambda x,y: _bl) if _bl is not None else th

    # ── 접촉 스케줄: 각 발이 각 노드에서 stance인지(1/0) ──
    _CRAWL_ORDER = ['FL','HR','FR','HL']         # crawl 스윙 순서(지지삼각형 유지)
    def in_stance(foot, k):
        t = k*dt; ph = ((t + phase_off*Tg) % Tg)/Tg   # 0~1 (phase_off=전역 위상 오프셋)
        if gait == 'crawl':                       # 4구간, 각 구간서 1발만 스윙(swing=duty만큼)
            idx = _CRAWL_ORDER.index(foot)
            win_lo = idx*0.25; sw = 0.25*duty     # 이 발의 스윙 창
            return not (win_lo <= ph < win_lo+sw)  # 스윙 창 밖이면 stance
        off = 0.0 if foot in ('FL','HR') else 0.5  # trot: 반주기 오프셋
        return ((ph - off) % 1.0) < duty
    contact = {f:[in_stance(f,k) for k in range(N+1)] for f in FEET}

    opti = ca.Opti()
    # ── 결정변수 ──
    P  = opti.variable(3, N+1)                  # base pos (x,y,z)
    Th = opti.variable(3, N+1)                  # base ori (roll,pitch,yaw) 소각
    Ft = {f: opti.variable(3, N+1) for f in FEET}   # 발 위치(world)
    Fr = {f: opti.variable(3, N+1) for f in FEET}   # 접촉력(world)

    g_vec = ca.DM([0,0,-G])
    J = 0
    # ── 초기조건 (시작점 지형 위 BASE_H) ──
    _z0 = BASE_H + float(th(0.0, 0.0))
    opti.subject_to(P[:,0] == ca.DM([0,0,_z0]))
    opti.subject_to(Th[:,0] == ca.DM([0,0,0]))
    for f in FEET:
        opti.subject_to(Ft[f][0,0] == P_NOM[f][0])
        opti.subject_to(Ft[f][1,0] == P_NOM[f][1])
        opti.subject_to(Ft[f][2,0] == th(P_NOM[f][0], P_NOM[f][1]))

    for k in range(N+1):
        pk = P[:,k]; thk = Th[:,k]
        # 총 접촉력·모멘트
        Fsum = ca.DM([0,0,0]); Msum = ca.DM([0,0,0])
        for f in FEET:
            fk = Fr[f][:,k]; rk = Ft[f][:,k]
            Fsum = Fsum + fk
            Msum = Msum + ca.cross(rk - pk, fk)
            if contact[f][k]:
                # 마찰콘(평지 법선 z; 지형법선은 Phase0b): fz>=0, |fxy|<=mu fz
                opti.subject_to(fk[2] >= 0)
                opti.subject_to(fk[2] <= F_MAX)
                opti.subject_to(fk[0] <= MU*fk[2]); opti.subject_to(-fk[0] <= MU*fk[2])
                opti.subject_to(fk[1] <= MU*fk[2]); opti.subject_to(-fk[1] <= MU*fk[2])
                # 지지발: 지형 위 + 정지(발 xy 고정=이전노드와 동일)
                opti.subject_to(rk[2] == th(rk[0], rk[1]))
                if k>0 and contact[f][k-1]:
                    opti.subject_to(Ft[f][0,k] == Ft[f][0,k-1])
                    opti.subject_to(Ft[f][1,k] == Ft[f][1,k-1])
            else:
                opti.subject_to(fk == ca.DM([0,0,0]))      # 스윙: 힘 0
                # 스윙: 지형 위 클리어런스(중간 노드에서 apex)
                opti.subject_to(rk[2] >= th(rk[0], rk[1]) + 0.0)
            # ROM: 발이 base 로컬 공칭±ROM 내 (yaw 소각→월드≈로컬+base_xy)
            dx = rk[0] - (pk[0] + P_NOM[f][0]); dy = rk[1] - (pk[1] + P_NOM[f][1])
            opti.subject_to(dx <=  ROM_DXY[0]); opti.subject_to(-dx <= ROM_DXY[0])
            opti.subject_to(dy <=  ROM_DXY[1]); opti.subject_to(-dy <= ROM_DXY[1])
            opti.subject_to(rk[2]-pk[2] >= FOOT_Z_LO); opti.subject_to(rk[2]-pk[2] <= FOOT_Z_HI)

        # ── SRBD 동역학(중앙차분 가속) ──
        if 0 < k < N:
            acc = (P[:,k+1] - 2*P[:,k] + P[:,k-1])/dt**2
            opti.subject_to(MASS*acc == Fsum + MASS*g_vec)         # 선형
            angacc = (Th[:,k+1] - 2*Th[:,k] + Th[:,k-1])/dt**2
            for i in range(3):
                opti.subject_to(INER[i]*angacc[i] == Msum[i])      # 회전(소각 근사)
        # ── 비용: 전진속도 추종 + 자세/힘 정규화 ──
        if k>0:
            vx = (P[0,k]-P[0,k-1])/dt
            J += 5.0*(vx - vx_des)**2
        J += 50.0*(thk[0]**2 + thk[1]**2 + thk[2]**2)              # roll/pitch/yaw 레벨(강)
        J += 20.0*(P[1,k]**2)                                       # y 드리프트 억제(강)
        # base 높이 = 지지면(플랫폼) 레벨 위 명목(갭에서 안 빠지게 bhref 사용)
        J += 80.0*(P[2,k] - (bhref(P[0,k],P[1,k]) + BASE_H))**2
        # 평활: base 선속도/각속도 최소화(매끄러운 추종 참조)
        if 0<k<N:
            J += 2.0*ca.sumsqr((P[:,k+1]-P[:,k-1])/(2*dt) - ca.DM([vx_des,0,0]))
            J += 5.0*ca.sumsqr((Th[:,k+1]-Th[:,k-1])/(2*dt))
        for f in FEET:
            J += 1e-4*ca.sumsqr(Fr[f][:,k])

    opti.minimize(J)
    # ── 초기추정 ──
    for k in range(N+1):
        xk = x_goal*k/N
        opti.set_initial(P[:,k], [xk, 0, BASE_H + float(bhref(xk,0))])
        for f in FEET:
            fx=xk+P_NOM[f][0]
            fz=float(th(fx,P_NOM[f][1]))
            if _bl is not None and fz < _bl-1e-3: fz=_bl   # 갭 위 init발=플랫폼 레벨(갭바닥 아님)
            opti.set_initial(Ft[f][:,k], [fx, P_NOM[f][1], fz])
            if contact[f][k]:
                opti.set_initial(Fr[f][:,k], [0,0,MASS*G/2])

    opti.solver('ipopt', {'print_time':0}, {'print_level':5 if verbose else 0,
                 'max_iter':800, 'tol':1e-4, 'acceptable_tol':1e-3})
    try:
        sol = opti.solve()
    except Exception as e:
        print("[TOWR] solve 실패:", str(e)[:200])
        try:
            P_=opti.debug.value(P); print("  마지막 x_end=%.3f z범위 %.3f~%.3f"%(P_[0,-1],P_[2].min(),P_[2].max()))
        except Exception: pass
        return None

    out = {'dt':dt,'N':N,'kind':kind,'tkw':tkw,'Tg':Tg,'duty':duty,'gait':gait,
           'P':sol.value(P).tolist(), 'Th':sol.value(Th).tolist(),
           'Ft':{f:sol.value(Ft[f]).tolist() for f in FEET},
           'Fr':{f:sol.value(Fr[f]).tolist() for f in FEET},
           'contact':{f:contact[f] for f in FEET}}
    Pv = sol.value(P)
    print("[TOWR] ✅ solve 성공  x:%.3f→%.3f  z:%.3f~%.3f  총시간%.2fs  vx_avg=%.3f"
          % (Pv[0,0],Pv[0,-1],Pv[2].min(),Pv[2].max(),T,(Pv[0,-1]-Pv[0,0])/T))
    return out


if __name__ == '__main__':
    kind = os.environ.get('TERRAIN','flat')
    gait = os.environ.get('GAIT','trot')
    N    = int(os.environ.get('N','40')); dt=float(os.environ.get('DT','0.02'))
    xg   = float(os.environ.get('XGOAL','0.8'))
    Tg   = float(os.environ.get('TG','0.80' if gait=='crawl' else '0.40'))
    duty = float(os.environ.get('DUTY','0.8' if gait=='crawl' else '0.5'))
    tkw  = {}
    if kind=='step': tkw={'x0':float(os.environ.get('X0','0.6')),'h':float(os.environ.get('H','0.10'))}
    if kind=='gap':  tkw={'x0':float(os.environ.get('X0','0.6')),'x1':float(os.environ.get('X1','0.85')),'depth':float(os.environ.get('DEPTH','0.30'))}
    if kind=='platgap': tkw={'x0':float(os.environ.get('X0','1.0')),'x1':float(os.environ.get('X1','1.3')),'plat':float(os.environ.get('PLAT','0.20'))}
    r = build_and_solve(kind=kind, tkw=tkw, N=N, dt=dt, x_goal=xg, Tg=Tg, duty=duty,
                        gait=gait, verbose=bool(os.environ.get('VERBOSE')))
    if r is not None:
        outf = os.environ.get('OUT','/home/jsh/문서/jsh/simulation/quad/towr/traj_%s.json'%kind)
        with open(outf,'w') as f: json.dump(r,f)
        print("[TOWR] 궤적 저장:", outf)
