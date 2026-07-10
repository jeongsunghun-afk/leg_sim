# 하이브리드 추종: Phase A(A1/A2)=관절PD+중력보상(물리 haunch가 받침), Phase B=WBC(4발 기립)
import os, sys, numpy as np, mujoco
sys.path.insert(0,'/home/jsh/문서/jsh/simulation/quad'); sys.argv=[sys.argv[0]]
import quad_mpc_wbic_17dof as C
C._ROBOT='ours_17dof_waist_sphere'; q=C.QuadSim(); m,d=q.m,q.d
T=np.load('/tmp/getup_stand.npz',allow_pickle=True)
Q=T['q']; DQ=T['dq']; sched=[str(s) for s in T['sched']]
com=T['com_ref']; comv=T['comv_ref']; acom=T['acom_ref']; dt=float(T['dt'])
aq=m.actuator_trnid[:,0]; perm=m.jnt_qposadr[aq]-7
Qqp=np.zeros_like(Q); Qqp[:,perm]=Q; DQqp=np.zeros_like(DQ); DQqp[:,perm]=DQ
qs=np.array([float(x) for x in open('/tmp/q_sit.txt').read().split()])
d.qpos[:]=qs; d.qvel[:]=0; mujoco.mj_forward(m,d)
sub=max(1,round(dt/m.opt.timestep)); N=len(Q)
KP=float(os.environ.get('KP','120')); KD=float(os.environ.get('KD','4'))
allc=(0,1,2,3)
for k in range(N):
    for _ in range(sub):
        if sched[k]=='B':   # 4발 기립=WBC(base GRF로 닫음)
            r=q.wbic_jump(com[k],comv[k],acom[k],Qqp[k],DQqp[k],allc,w_ori=float(os.environ.get('WORI','40')))
            if r[0] is None: d.ctrl[:]=np.clip(d.qfrc_bias[6:6+m.nu],-q._tau_peak,q._tau_peak)
        else:               # Phase A=관절PD+중력보상(물리 haunch 접촉이 받침)
            tau=d.qfrc_bias[6:6+m.nu]+KP*(Q[k]-d.qpos[7:7+m.nu])+KD*(DQ[k]-d.qvel[6:6+m.nu])
            d.ctrl[:]=np.clip(tau,-q._tau_peak,q._tau_peak)
        mujoco.mj_step(m,d)
    if k%20==0 or k==N-1:
        w,x,y,z=d.qpos[3:7]; tilt=np.degrees(np.arccos(max(-1,min(1,1-2*(x*x+y*y)))))
        print('  k=%3d %3s z=%.3f tilt=%4.1f x=%+.2f'%(k,sched[k],d.qpos[2],tilt,d.qpos[0]))
for _ in range(600):
    r=q.wbic_jump(com[-1],comv[-1]*0,acom[-1]*0,Qqp[-1],DQqp[-1]*0,allc,w_ori=float(os.environ.get('WORI','40')))
    if r[0] is None: d.ctrl[:]=np.clip(d.qfrc_bias[6:6+m.nu],-q._tau_peak,q._tau_peak)
    mujoco.mj_step(m,d)
w,x,y,z=d.qpos[3:7]; tilt=np.degrees(np.arccos(max(-1,min(1,1-2*(x*x+y*y)))))
print('=== 종료 z=%.3f tilt=%.1f° (성공=z>0.45 tilt<15) ==='%(d.qpos[2],tilt))
