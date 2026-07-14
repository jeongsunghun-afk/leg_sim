"""biped 레퍼런스 궤적 생성기 (B4 / RL 레퍼런스화) — MPC+WBIC 컨트롤러를 DTC 교사로.

현 biped_mpc_wbic(제자리 균형+0.4m/s 보행)를 돌려 **레퍼런스 궤적 라이브러리**를 뽑는다.
RL(hind_leg)이 이를 추종(DTC 패턴 A): RL이 반응형 균형을 학습해 모델기반 ~8s 한계를 넘어선다.

저장 포맷(프레임별):
  full(env 재생/보상용): t·root_pos(3)·root_quat(4 wxyz)·q(8)·dq(8)·foot_pos(2×3)·contact(2)·tau(8)·lam(2×3)
  ★DTC subset(정책 관측 = "작은 서브셋"): swing_footstep_tgt(2)·ref_q(8)·contact_sched(2)  (report §3 A)
사용: python biped_ref_export.py        (기본 라이브러리: vx=0/0.2/0.3)
"""
from __future__ import annotations
import os, numpy as np, mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref_lib')
REC_HZ = 50          # 레퍼런스 저장 주기(정책 50Hz와 일치)


def gen_reference(vx, dur=6.0, tilt_lim=20.0):
    """컨트롤러 1회 실행 → 레퍼런스 궤적(안정 구간만). fall/과tilt 전까지."""
    c = BM.BipedMPCWBIC(); c.reset(); c.vx_cmd = vx; c.setup_mpc()
    m, d = c.m, c.d; dt = m.opt.timestep
    decim = max(1, int(round((1.0 / REC_HZ) / dt)))
    rec = {k: [] for k in ('t','root_pos','root_quat','q','dq','foot','contact',
                           'tau','lam','swing_tgt','ref_q','cs')}
    i = 0
    while i * dt < dur:
        stance, swing_leg, s = c.step_gait(dt)
        if c._k % c.mpc_decim == 0:
            c.lam = c.mpc_grf(stance)
        c._k += 1
        c.com0[0] += vx * dt
        if c.liftoff[swing_leg] is None:
            c.liftoff[swing_leg] = d.geom_xpos[c.sph[swing_leg]].copy()
        p, v = c.swing_traj(swing_leg, s)
        c.wbic_track(stance, {swing_leg: (p, v)}, c.lam)
        mujoco.mj_step(m, d)
        tilt = np.hypot(*base_rpy(d.qpos[3:7])[:2])
        if d.qpos[2] < 0.3 or tilt > tilt_lim:      # 안정 구간 종료
            break
        if i % decim == 0:                          # 50Hz 저장
            cs = [1 if k in stance else 0 for k in range(2)]
            rec['t'].append(i * dt)
            rec['root_pos'].append(d.qpos[:3].copy())
            rec['root_quat'].append(d.qpos[3:7].copy())
            rec['q'].append(d.qpos[7:].copy())
            rec['dq'].append(d.qvel[6:].copy())
            rec['foot'].append(np.concatenate([d.geom_xpos[sp].copy() for sp in c.sph]))
            rec['contact'].append(np.array(cs, float))
            rec['tau'].append(d.ctrl.copy())
            rec['lam'].append(c.lam.flatten().copy())
            rec['swing_tgt'].append(p[:2].copy())    # ★DTC: 스윙 발디딤 2D 목표
            rec['ref_q'].append(d.qpos[7:].copy())    # ★DTC: 목표 관절각(=컨트롤러 q)
            rec['cs'].append(np.array(cs, float))     # ★DTC: 접촉 스케줄
        i += 1
    return {k: np.array(v) for k, v in rec.items()}, i * dt


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    lib = [('inplace', 0.0), ('walk02', 0.2), ('walk03', 0.3)]
    print(f"레퍼런스 라이브러리 생성 → {OUTDIR}/ (50Hz, DTC subset 포함)")
    for name, vx in lib:
        ref, dur = gen_reference(vx)
        path = os.path.join(OUTDIR, f'biped_ref_{name}.npz')
        np.savez(path, vx=vx, dt=1.0/REC_HZ, **ref)
        n = len(ref['t'])
        print(f"  {name:8s} vx={vx} · {n} 프레임 ({dur:.1f}s 안정) · dist={ref['root_pos'][-1,0]:+.2f}m → {os.path.basename(path)}")
    print("★DTC 서브셋 키: swing_tgt(발디딤2D)·ref_q(목표관절)·cs(접촉스케줄) = RL 정책 관측용")
    print("★full 키: root_pos/quat·q·dq·foot·contact·tau·lam = env 재생·tracking 보상용")


if __name__ == '__main__':
    main()
