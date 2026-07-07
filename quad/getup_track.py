import os, sys, numpy as np, mujoco
sys.path.insert(0,'/home/jsh/문서/jsh/simulation/quad'); sys.argv=[sys.argv[0]]
import quad_mpc_wbic_17dof as C
C._ROBOT='ours_17dof_waist_sphere'
q=C.QuadSim()
m,d=q.m,q.d
print('legs 순서:', q.legs)          # 어느 인덱스가 앞발(FL/FR)인지
front=[i for i,L in enumerate(q.legs) if L in ('FL','FR')]
allc=(0,1,2,3)
T=np.load('/tmp/getup_stand.npz', allow_pickle=True)
Q=T['q']; DQ=T['dq']; base_z=T['base_z']; sched=[str(s) for s in T['sched']]
com=T['com_ref']; comv=T['comv_ref']; acom=T['acom_ref']; dt=float(T['dt'])
# 액추에이터→qpos 순서 permutation
aq=m.actuator_trnid[:,0]; perm=m.jnt_qposadr[aq]-7
Qqp=np.zeros_like(Q); Qqp[:,perm]=Q; DQqp=np.zeros_like(DQ); DQqp[:,perm]=DQ
# 시작=정착 sit
qs=np.array([float(x) for x in open('/tmp/q_sit.txt').read().split()])
d.qpos[:]=qs; d.qvel[:]=0; mujoco.mj_forward(m,d)
sub=max(1,round(dt/m.opt.timestep)); N=len(Q)
print('추종 시작 sit z=%.3f, N=%d sub=%d front=%s'%(d.qpos[2],N,sub,front))
qpfail=0
for k in range(N):
    cts = tuple(front) if sched[k]=='A' else allc
    for _ in range(sub):
        r=q.wbic_jump(com[k],comv[k],acom[k],Qqp[k],DQqp[k],cts)
        if r[0] is None: qpfail+=1; d.ctrl[:]=np.clip(d.qfrc_bias[6:6+m.nu],-q._tau_peak,q._tau_peak)
        mujoco.mj_step(m,d)
    if k%20==0 or k==N-1:
        w,x,y,z=d.qpos[3:7]; tilt=np.degrees(np.arccos(max(-1,min(1,1-2*(x*x+y*y)))))
        print('  k=%3d z=%.3f tilt=%4.1f x=%+.2f (QP실패%d)'%(k,d.qpos[2],tilt,d.qpos[0],qpfail))
# 끝 홀드 0.8s = wbic_stance
for _ in range(800):
    q.com_ref=d.subtree_com[0].copy(); q.wbic_stance(); mujoco.mj_step(m,d)
w,x,y,z=d.qpos[3:7]; tilt=np.degrees(np.arccos(max(-1,min(1,1-2*(x*x+y*y)))))
print('=== 종료 z=%.3f tilt=%.1f° (성공=z>0.45 tilt<15) ==='%(d.qpos[2],tilt))
