"""control/model_ctrl.py — 모델기반(MPC+WBIC) 래퍼. stand/walk. ★jog 검증 후 사용.

biped_mpc_wbic 컨트롤러 + leg-odometry 추정기(deploy/robot_interface.StateEstimator)를 감싼다.
  측정상태(관절 q/dq + IMU quat/gyro + 접촉) 주입 → 추정 base → c.control → 관절 토크(Nm).
biped_deploy.py 의 제어 루프 본체를 그대로 포팅(플랜트만 실HW). 무거운 의존(mujoco·qpsolvers)은
이 모듈 import 시에만 로드 → app 은 stand/walk 진입 때만 lazy import(jog 단계는 numpy만).
"""
from __future__ import annotations
import os, sys
import numpy as np


class ModelController:
    def __init__(self, biped_dir: str, deploy_dir: str, tau_max_frac: float = 0.6):
        for d in (biped_dir, deploy_dir):
            if d not in sys.path:
                sys.path.insert(0, d)
        import mujoco
        import biped_mpc_wbic as BM
        from robot_interface import StateEstimator, FOOT_NAMES
        self.mj = mujoco
        self.c = BM.BipedMPCWBIC(); self.c.reset(); self.c.setup_mpc()
        self.m, self.d = self.c.m, self.c.d
        self.dt = float(self.m.opt.timestep)
        self.nu = int(self.m.nu)
        sph = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
        rad = [float(self.m.geom_size[g][0]) for g in sph]
        self.est = StateEstimator(self.m, sph, rad, self.dt,
                                  dwell_steps=15, k_anchor=0.05, contact_height=True)
        self.est.reset(self.d.qpos[0:3])
        # 안전 토크 클램프 = actuatorfrcrange · frac
        self.tau_lim = np.abs(self.m.actuator_forcerange[:, 1]) * float(tau_max_frac)
        if not np.any(self.tau_lim):                       # frcrange 미설정 모델 대비
            self.tau_lim = np.full(self.nu, 80.0)

    def reset(self, body_h: float):
        self.c.reset(); self.c.setup_mpc()
        self.c.com_ref[2] = body_h; self.c._k = 0
        self.est.reset(self.d.qpos[0:3])

    def set_cmd(self, v, vy, w, body_h, walking: bool):
        self.c.vx_cmd = float(v) if walking else 0.0
        self.c.wz_cmd = float(w) if walking else 0.0
        self.c.vy_cmd = float(vy) if walking else 0.0
        self.c.com_ref[2] = float(body_h)

    def step(self, q_rad, dq_rad, quat, gyro, acc, contact) -> np.ndarray:
        """측정상태 → 토크[Nm] (biped 관절 순서 = actuator 순서)."""
        self.est.estimate(q_rad, dq_rad, quat, gyro, contact, acc=acc)
        d, m, mj = self.d, self.m, self.mj
        d.qpos[7:] = q_rad;  d.qvel[6:] = dq_rad
        d.qpos[3:7] = quat;  d.qvel[3:6] = gyro
        d.qpos[0:3] = self.est.p; d.qvel[0:3] = self.est.v
        mj.mj_forward(m, d)
        self.c.control(self.dt)
        tau = np.asarray(d.ctrl[:self.nu], float).copy()
        return np.clip(tau, -self.tau_lim, self.tau_lim)   # 안전 클램프

    @property
    def est_state(self):
        return self.est.p.copy(), self.est.v.copy()
