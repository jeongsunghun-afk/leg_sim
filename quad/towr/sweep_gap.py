#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase0c-1: 전역 gait 위상 오프셋 스윕 → 고정 timing으로 GAP 크로싱 feasible 정렬 탐색.
crawl이 사이클당 ~vx·Tg 전진하므로, 스윙이 갭을 걸치도록 위상만 맞으면 고정timing도 가능.
feasible φ가 있으면 최소변경 해결, 없으면 per-foot 변동timing 필요 확정.
"""
import numpy as np, os, json
from towr_cd import build_and_solve

X0=float(os.environ.get('X0','1.0')); X1=float(os.environ.get('X1','1.20'))
PLAT=float(os.environ.get('PLAT','0.20'))
N=int(os.environ.get('N','120')); Tg=float(os.environ.get('TG','0.80'))
DUTY=float(os.environ.get('DUTY','0.8')); XGOAL=float(os.environ.get('XGOAL','1.7'))
tkw={'x0':X0,'x1':X1,'plat':PLAT}
NOFF=int(os.environ.get('NOFF','10'))

print("=== 위상 오프셋 스윕: 갭[%.2f,%.2f] w=%.2f, %d개 φ ==="%(X0,X1,X1-X0,NOFF))
best=None
for i in range(NOFF):
    ph=i/NOFF
    r=build_and_solve(kind='platgap',tkw=tkw,N=N,dt=0.02,Tg=Tg,duty=DUTY,
                      x_goal=XGOAL,gait='crawl',phase_off=ph,verbose=False)
    if r is None:
        print("  φ=%.2f  ❌ infeasible"%ph); continue
    # 갭내 지지발 착지 검사
    ing=0
    for f in ['FL','FR','HL','HR']:
        Ft=np.array(r['Ft'][f]); con=r['contact'][f]
        for k in range(len(con)):
            if con[k] and X0<Ft[0,k]<X1 and Ft[2,k]<PLAT-0.05: ing+=1
    P=np.array(r['P'])
    ok = (ing==0)
    print("  φ=%.2f  ✅ solve  갭내착지=%d  x_end=%.2f  %s"%(ph,ing,P[0,-1],"★크로싱OK" if ok else "갭침범"))
    if ok and (best is None):
        best=(ph,r)
if best:
    ph,r=best
    outf=os.environ.get('OUT','/home/jsh/문서/jsh/simulation/quad/towr/traj_crawl_platgap.json')
    with open(outf,'w') as f: json.dump(r,f)
    print("\n★ feasible 위상 φ=%.2f → 저장 %s (고정timing으로 GAP 크로싱 가능!)"%(ph,outf))
else:
    print("\n어떤 위상도 갭 회피 실패 → per-foot 변동 phase timing 필요(Phase0c-2)")
