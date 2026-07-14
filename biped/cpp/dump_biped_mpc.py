"""biped MPC 기준값 덤프 → C++ 포팅 파리티 검증용.
   실제 보행 몇 스텝 후 mpc_qp_plan 입력(x0·cs·fp·x_ref)+출력(lam)+cfg를 텍스트로 저장.
   실행: python dump_biped_mpc.py > /tmp/biped_mpc_dump.txt
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import biped_mpc_wbic as BM

c = BM.BipedMPCWBIC(); c.reset(); c.vx_cmd = 0.15; c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
for _ in range(80):                       # 현실적 상태 확보
    c.control(dt); mujoco.mj_step(m, d)

# ── mpc_grf 입력 재구성 (biped_mpc_wbic.mpc_grf 동일) ──
stance = [c.stance]
x0, yaw0 = c.body_x0()
com = d.subtree_com[0]
frel = np.array([d.geom_xpos[c.sph[i]] - com for i in range(2)])   # 2×3 (tiled over N)
cur = np.array([1 if i in stance else 0 for i in range(2)])        # 접촉 (tiled)
cyd, syd = np.cos(c.yaw_des), np.sin(c.yaw_des)
vx_w = cyd*c.vx_cmd - syd*c.vy_cmd; vy_w = syd*c.vx_cmd + cyd*c.vy_cmd
x_ref = np.array([0,0,c.yaw_des, com[0],com[1],c.com_ref[2], 0,0,c.wz_cmd, vx_w,vy_w,0, -9.81])
csN = [cur.tolist()] * BM.MPC_N
fpN = [[frel[0], frel[1]]] * BM.MPC_N
lam = c.mpc_qp_plan(x0, csN, fpN, x_ref)

def row(a): return ' '.join(f'{v:.12e}' for v in np.asarray(a).ravel())
P = print
P("N", BM.MPC_N); P("DT", BM.MPC_DT); P("MASS", c.mass); P("MU", c.mu)
P("LAMZ_MIN", BM.LAMZ_MIN); P("LAMZ_MAX", c.lamz_max)
P("I_BODY", row(c.I_body))
P("QDIAG", row(BM.Q_DIAG)); P("RDIAG", row(BM.R_DIAG))
P("X0", row(x0)); P("XREF", row(x_ref))
P("CS", row(cur)); P("FP", row(frel))
P("LAM", row(lam))
