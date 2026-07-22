#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
C1.0 · sim2sim 갭 프로브 — C-1 착수 가치 게이트.
동일 (q, v, τ)에서 MuJoCo(STIFF soft-sphere 접촉) qacc vs Pinocchio 강체 접촉
constraintDynamics ddq 를 비교해 **접촉모델 sim2sim 갭**을 정량화.

의미: C-1은 Pinocchio 강체 접촉 + 해석 도함수(computeConstraintDynamicsDerivatives)로
계획을 세우고 MuJoCo(배포 sim, soft)서 실행. 갭이 작으면 C-1의 강체계획이 전이=착수가치↑.
크면 C-2/RL(MuJoCo-native)이 유리 → C-1 재고.

실행: /home/jsh/miniforge3/envs/proxddp/bin/python sim2sim_probe.py

★상태(2026-07-22): 순간 qacc 비교는 **STIFF 접촉 강성응답이 지배해 오염**(|qacc|~1e4,
비물리적). 깨끗한 갭 측정 = **궤적기반 비교**(짧은 스탠스 push를 양쪽서 적분→base 발산)로
재설계 필요. ★단 C1.0 핵심 성과 = Pinocchio가 constraintDynamics(강체접촉 forward) +
computeConstraintDynamicsDerivatives(**해석 도함수**)를 내장 확인 → C-1의 C1.2(해석
그래디언트, 최고난도)가 손코딩 아닌 Pinocchio 내장으로 대폭 de-risk. C1_ROADMAP.md 참조.
"""
import numpy as np, mujoco, os
import pinocchio as pin
from model_bridge import MJCF, MjPinBridge, apply_gearbox, set_foot_sphere, strip_mesh_collision, FEET, FOOT_FRAME, MJ2PIN_LEG

FOOT_R = float(os.environ.get("FOOT_R", "0.025"))
STIFF  = float(os.environ.get("STIFF", "0.002"))   # MuJoCo 접촉 강성(작을수록 강체에 근접)

def main():
    mm = mujoco.MjModel.from_xml_path(MJCF)
    apply_gearbox(mm); set_foot_sphere(mm, FOOT_R); strip_mesh_collision(mm)
    mm.geom_solref[:, 0] = STIFF; mm.geom_solref[:, 1] = 1.0          # 강체 매칭
    md = mujoco.MjData(mm)
    br = MjPinBridge()
    m, d = br.model, br.data

    # ── 스탠딩 자세로 정착(강한 PD로 평형 근접) ──
    q_stand_mj = br.pin_to_mj_qpos(_stand_pin(br))
    md.qpos[:] = q_stand_mj; md.qvel[:] = 0.0
    for _ in range(3000):                                            # 충분히 오래 → 평형(qacc→0)
        mujoco.mj_forward(mm, md)
        md.ctrl[:] = 300.0 * (q_stand_mj[7:] - md.qpos[7:]) - 15.0 * md.qvel[6:]
        mujoco.mj_step(mm, md)
    mujoco.mj_forward(mm, md)
    _rest = np.linalg.norm(md.qacc)
    print(f"[probe] 정착 base_z={md.qpos[2]:.3f} 접촉수={md.ncon} |qacc_rest|={_rest:.2f} (작을수록 평형)")

    # ── Pinocchio 접촉모델(스탠스 4발, 3D point) ──
    cms = pin.StdVec_RigidConstraintModel()
    for L in FEET:
        fid = br.foot_fid[L]
        fr = m.frames[fid]
        jid = fr.parentJoint
        # 접촉점 = foot frame + sole offset(반경) → 지면 접점
        pl = fr.placement * pin.SE3(np.eye(3), np.array([0., 0., -FOOT_R]))
        cm = pin.RigidConstraintModel(pin.ContactType.CONTACT_3D, m, jid, pl,
                                      pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        cms.append(cm)
    cds = pin.StdVec_RigidConstraintData()
    for cm in cms:
        cds.append(cm.createData())
    prox = pin.ProximalSettings(1e-12, 1e-10, 20)
    pin.initConstraintDynamics(m, d, cms, cds)

    # ── 비교: 정착 PD-홀드 토크(양쪽 동일) + 섭동. 평형이면 둘 다 ddq≈0 → 일치 ──
    q_mj = md.qpos.copy(); v_mj = md.qvel.copy()
    tau_hold = (300.0 * (q_stand_mj[7:] - q_mj[7:]) - 15.0 * v_mj[6:])   # MuJoCo 홀드 토크(17)
    tau_hold_pin = tau_hold[MJ2PIN_LEG]                                   # → pin leg 16(waist 제외)
    q = br.mj_to_pin_q(q_mj); v = br.mj_to_pin_v(v_mj, q_mj)
    print("\n case            | ddq base(6) 갭 | ddq leg(16) 갭 | |ddq_mj| | |ddq_pin| | rel")
    for name, dtau in [("hold", 0.0), ("perturb+", 15.0), ("perturb-", -15.0)]:
        tau_pin = tau_hold_pin + dtau
        tau_mj = tau_hold + dtau
        # Pinocchio 강체 접촉 ddq
        ddq_pin = pin.constraintDynamics(m, d, q, v, np.concatenate([np.zeros(6), tau_pin]), cms, cds, prox)
        # MuJoCo qacc (동일 τ, forward만)
        md.ctrl[:] = tau_mj; mujoco.mj_forward(mm, md)
        qacc_pin_order = br.mj_to_pin_v(md.qacc.copy(), q_mj)
        gb = np.linalg.norm(ddq_pin[:6] - qacc_pin_order[:6])
        gl = np.linalg.norm(ddq_pin[6:] - qacc_pin_order[6:])
        nmj = np.linalg.norm(qacc_pin_order); npn = np.linalg.norm(ddq_pin)
        rel = (gb + gl) / (nmj + npn + 1e-6)
        print(f" {name:15s} |  {gb:10.3f}  |  {gl:10.3f}  | {nmj:7.2f} | {npn:8.2f} | {rel:.3f}")

    print("\n[게이트] base+leg ddq 갭이 작으면(rel<~0.1) Pinocchio 강체≈MuJoCo STIFF → C-1 전이가치↑.")
    print("         크면 soft/rigid 접촉 불일치 커 C-1 배포 재고(C-2/RL이 MuJoCo-native라 유리).")

def _stand_pin(br):
    """발을 힙 아래 지면(z=FOOT_R)에 두는 standing pin q (간이 IK)."""
    m, d = br.model, br.data
    q = pin.neutral(m); q[2] = 0.42
    tgt = {'FL':[0.30,0.16], 'FR':[0.30,-0.16], 'HL':[-0.30,0.16], 'HR':[-0.30,-0.16]}
    for _ in range(300):
        pin.forwardKinematics(m, d, q); pin.updateFramePlacements(m, d); pin.computeJointJacobians(m, d, q)
        err = np.zeros(12); J = np.zeros((12, m.nv))
        for i, L in enumerate(FEET):
            p = d.oMf[br.foot_fid[L]].translation + d.oMf[br.foot_fid[L]].rotation @ np.array([0,0,-FOOT_R])
            err[3*i:3*i+3] = np.array([tgt[L][0], tgt[L][1], 0.0]) - p
            J[3*i:3*i+3] = pin.getFrameJacobian(m, d, br.foot_fid[L], pin.LOCAL_WORLD_ALIGNED)[:3]
        J[:, :6] = 0.0
        if np.linalg.norm(err) < 1e-5: break
        q = pin.integrate(m, q, 0.5 * np.linalg.lstsq(J, err, rcond=None)[0])
    return q

if __name__ == "__main__":
    main()
