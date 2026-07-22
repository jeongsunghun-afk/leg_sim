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

★결과(2026-07-22): **궤적기반** 비교(스탠스 홀드 80스텝=0.16s 양쪽 적분→base 발산).
- **sim2sim 갭 ≈ 0.059 m**(MuJoCo 거의정지 vs Pinocchio base 0.047m 침하). 주로 접촉 drift
  (Baumgarte 없는 순진 적분 아티팩트)라 **근본 갭은 더 작음**. 정밀화=Baumgarte 안정화 필요.
- ★핵심 교훈: **Pinocchio 모델에도 armature 넣어야**(없으면 stiff PD가 dt2ms서 발산=NaN,
  [[ci-mpc-track]] 그 교훈). KP150=양쪽 안정(KP300은 pin 발산).
- ★C1.0 de-risk: Pinocchio가 constraintDynamics + computeConstraintDynamicsDerivatives
  (**해석 도함수**) 내장 → C-1 C1.2(해석 그래디언트) 대폭 de-risk. C1_ROADMAP.md 참조.
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
    # ★Pinocchio 모델에도 armature(반사 로터관성) — 없으면 stiff PD가 dt=2ms서 발산([[ci-mpc-track]] 교훈)
    _arm = np.tile([1e-4*7**2, 1e-4*7**2, 1e-4*10.5**2, 1e-4*8.4**2], 4)   # FL,FR,HL,HR × [hip,thigh,calf,foot]
    m.armature[6:6+16] = _arm

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
        try:                                         # Baumgarte(있으면) — 위치 drift 보정
            cm.corrector.Kp = np.full(3, 50.0); cm.corrector.Kd = np.full(3, 14.0)
        except AttributeError:
            pass
        cms.append(cm)
    cds = pin.StdVec_RigidConstraintData()
    for cm in cms:
        cds.append(cm.createData())
    prox = pin.ProximalSettings(1e-10, 1e-6, 20)     # mu=1e-6(KKT 정칙화, 특이방지)
    pin.initConstraintDynamics(m, d, cms, cds)

    # ── 궤적기반 sim2sim 비교: 같은 τ열을 양쪽서 N스텝 적분 → base 궤적 발산 ──
    #   순간 qacc는 STIFF 강성응답이 지배해 오염 → 적분이 그걸 평활(접촉이 스텝간 평형화).
    q0_mj = md.qpos.copy(); v0_mj = md.qvel.copy()
    N = int(os.environ.get("NSTEP", "100")); dt = mm.opt.timestep
    def tau_of(qpos17, qvel16):   # PD-홀드 토크(17, MuJoCo순) — 양쪽 동일 입력(KP150=양쪽 안정)
        return 150.0 * (q_stand_mj[7:] - qpos17) - 8.0 * qvel16
    # MuJoCo rollout
    mj_base = []
    for _ in range(N):
        md.ctrl[:] = tau_of(md.qpos[7:], md.qvel[6:])
        mujoco.mj_step(mm, md); mj_base.append(md.qpos[:3].copy())
    mj_base = np.array(mj_base)
    # Pinocchio 강체접촉 rollout(semi-implicit Euler, 같은 τ)
    q = br.mj_to_pin_q(q0_mj); v = br.mj_to_pin_v(v0_mj, q0_mj); pin_base = []
    for _ in range(N):
        # τ: MuJoCo순 PD(현재 pin자세를 mj순으로 근사 매핑) — leg만
        qmj7 = np.zeros(17); qmj7[MJ2PIN_LEG] = q[7:]        # pin leg → mj 17(waist=0)
        vmj6 = np.zeros(17); vmj6[MJ2PIN_LEG] = v[6:]
        tau17 = tau_of(qmj7, vmj6); tau_pin = tau17[MJ2PIN_LEG]
        ddq = pin.constraintDynamics(m, d, q, v, np.concatenate([np.zeros(6), tau_pin]), cms, cds, prox)
        v = v + dt * ddq; q = pin.integrate(m, q, dt * v)   # semi-implicit
        pin_base.append(q[:3].copy())
    pin_base = np.array(pin_base)
    # 발산(base xyz)
    div = np.linalg.norm(mj_base - pin_base, axis=1)
    print("\n[궤적 sim2sim 갭] %d스텝(%.2fs) base xyz 발산:"%(N, N*dt))
    print("  MuJoCo   base 이동: dx=%+.4f dz=%+.4f"%(mj_base[-1,0]-mj_base[0,0], mj_base[-1,2]-mj_base[0,2]))
    print("  Pinocchio base 이동: dx=%+.4f dz=%+.4f"%(pin_base[-1,0]-pin_base[0,0], pin_base[-1,2]-pin_base[0,2]))
    print("  발산 |Δbase|: 최종=%.4f m, 최대=%.4f m (작을수록 sim2sim 갭↓=C-1 전이↑)"%(div[-1], div.max()))

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
