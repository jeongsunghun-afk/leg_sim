"""biped WBIC 기준값 덤프 → C++ 파리티 검증용.
   보행 몇 스텝 후 wbic_track 입력(M·h·jac·swing·lam·게인)+출력(tau) 저장.
   실행: python dump_biped_wbic.py > /tmp/biped_wbic_dump.txt
"""
import os, sys, numpy as np, mujoco
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import biped_mpc_wbic as BM
from biped_step import (SW_KP, SW_KD, W_ORI, W_ANKLE, W_POST, STANCE_KD,
                        MU, MU_MARGIN, LAMZ_MIN, DRV_PEAK, Q_HOME, ANKLE_IDX)
from biped_mpc_wbic import W_LAM

c = BM.BipedMPCWBIC(); c.reset(); c.vx_cmd = 0.15; c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
for _ in range(80):
    c.control(dt); mujoco.mj_step(m, d)

# 다음 스텝의 wbic_track 입력 (control() 재현, 단 상태는 안 바뀜)
stance, swing_leg, s = c.step_gait(dt)
if c._k % c.mpc_decim == 0: c.lam = c.mpc_grf(stance)
if c.liftoff[swing_leg] is None: c.liftoff[swing_leg] = d.geom_xpos[c.sph[swing_leg]].copy()
p, v = c.swing_traj(swing_leg, s)

nv, nu = c.nv, c.nu; Kc = len(stance)
M = np.zeros((nv, nv)); mujoco.mj_fullM(m, M, d.qM)
cjac = [c.foot_jac(cc) for cc in stance]
Jc = np.zeros((3, nv)); mujoco.mj_jacSubtreeCom(m, d, Jc, 0)
Jsw = c.foot_jac(swing_leg)

def row(a): return ' '.join(f'{x:.12e}' for x in np.asarray(a).ravel())
P = print
P("nv", nv); P("nu", nu); P("Kc", Kc)
P("M", row(M)); P("h", row(d.qfrc_bias)); P("qv", row(d.qvel))
P("q", row(d.qpos[7:])); P("qc", row(d.qpos[3:7]))
P("com", row(d.subtree_com[0])); P("zref", c.com_ref[2])
P("Jc", row(Jc))
P("contacts", ' '.join(str(x) for x in stance))
for k in range(Kc): P(f"cjac{k}", row(cjac[k]))
for k in range(Kc): P(f"lam{k}", row(c.lam[stance[k]]))
P("swing_leg", swing_leg)
P("Jsw", row(Jsw)); P("sw_pos", row(d.geom_xpos[c.sph[swing_leg]]))
P("sw_ptgt", row(p)); P("sw_vtgt", row(v))
P("Qhome", row(Q_HOME)); P("drv_peak", row(c.drv_peak))   # ★MJCF 파생값(모듈상수 아님). 2026-08-13 개명
P("ankle", ' '.join(str(x) for x in ANKLE_IDX))
P("gains", f"{SW_KP} {SW_KD} {W_ORI} {W_ANKLE} {W_POST} {W_LAM} {STANCE_KD} {MU*MU_MARGIN} {LAMZ_MIN}")

c.wbic_track(stance, {swing_leg: (p, v)}, c.lam)   # → d.ctrl
# ★2026-08-13 키 개명 tau → u_drive. `d.ctrl` 은 이제 **관절토크가 아니라 드라이브 토크**다
#   (발목 액추에이터 tendon 이전). 이름을 그대로 두면 C++ 쪽이 관절토크와 비교해 조용히
#   어긋난다 — 실제로 어긋났고, 파리티가 그걸 잡았다(calf 만 foot 값만큼 차이).
#   ⇒ **하드웨어에 실제로 나가는 양**으로 비교한다. 그게 파리티가 지켜야 할 것이다.
P("u_drive", row(d.ctrl))
