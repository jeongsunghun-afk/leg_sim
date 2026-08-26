#!/usr/bin/env python3
"""push 스윕 리허설 시뮬레이션 — 로봇 시간을 쓰기 전에 실험 설계를 검증한다.

배포기의 push 제어법칙을 그대로 재현한다:
    τ_cmd = g*축별 · G_model(q_meas)  +  Jz(q_meas, 미는점) · (−F)     (kp=0, kd=FLOAT_KD)
여기에 실물에만 있는 것들을 얹는다:
    τ_applied = r축별 ∘ τ_cmd            (경로 전달비 — 저울 실측 ~0.80)
    관절 쿨롱마찰 τ_c (조립 실측 ~0.7 Nm) · 관절 점성(kd_raw×FLOAT_KD 근사)
베이스는 배포기와 같이 고정(운동학적 핀). 바닥면 = 저울. 접촉 수직력 합 = 저울 눈금.

시나리오:
  A. 평발 · 발끝밀기 · r 공통 0.80        — ② 관측(뒤꿈치 떠오름)이 나오나?
  B. 평발 · 발끝밀기 · r_foot 만 0.66     — 떠오름이 r_foot 결손으로 재현되나?
  C. 평발 · 뒤꿈치밀기(E3) · r 공통 0.80  — 접촉이 뒤꿈치로 수렴하고 T=0.80 나오나?
  D. C 에서 r_foot=0.66                   — E3 가 foot 에 정말 무감한가? (설계 목적)
  E. 1점 Qhome8 · 발끝밀기(E4) · r 공통   — 실측자세 가중·T 확인

사용: python3 tools/push_sim_rehearsal.py            (전 시나리오)
"""
import io, os
import numpy as np, mujoco as mj

GSTAR = np.array([1.20, 1.10, 1.22, 1.00, 1.18, 1.10, 1.22, 1.00])
FLOAT_KD = 0.30
KD_RAW = np.array([6.0, 4.0, 7.88, 2.88])          # kd_ch·gear_k²  (관절 점성 근사)
TAUC   = 0.70                                       # 조립 쿨롱마찰 [Nm] (전축 근사)
PUSH_RATE, F_MAX = 5.0, 30.0                        # 배포기와 동일 램프
QFLAT4 = np.array([0, 0.064256, -0.416657, -1.043858])
QHOME4 = np.deg2rad([0, 11.634, -38.454, 0])

def load_fixed(mjcf, q4, preload_mm=0.5):
    """freejoint 를 제거한 고정베이스 모델을 만들어 로드한다.

    ★좌표 핀(매 스텝 qpos 리셋)은 안 된다 — 스텝 내부에서 베이스가 자유낙하해
    다리가 순간 무중력이 되고, 중력보상 ff 가 그대로 폭주 토크가 된다(실측 재현됨).
    베이스 높이는 목표자세에서 발끝 구 바닥이 지면에 preload 만큼 실리게 잡는다.
    """
    src = io.open(mjcf, encoding='utf-8').read()
    src = src.replace('<freejoint name="root" />', '<!-- 리허설: 베이스 고정 -->')
    tmp = os.path.join(os.path.dirname(os.path.abspath(mjcf)) or '.', '_rehearsal_tmp.mjcf')
    io.open(tmp, 'w', encoding='utf-8').write(src)
    m = mj.MjModel.from_xml_path(tmp)
    d = mj.MjData(m)
    d.qpos[:8] = np.concatenate([q4, q4]); mj.mj_forward(m, d)
    zmin = min(d.geom_xpos[g][2] - m.geom_size[g, 0]
               for g in range(m.ngeom) if m.geom_type[g] == mj.mjtGeom.mjGEOM_SPHERE)
    src2 = src.replace('<body name="torso" pos="0 0 0.5257">',
                       f'<body name="torso" pos="0 0 {0.5257 - zmin - preload_mm/1000:.6f}">')
    io.open(tmp, 'w', encoding='utf-8').write(src2)
    m = mj.MjModel.from_xml_path(tmp)
    os.remove(tmp)
    for i in range(m.nv):                      # 실물화 주입(관절 8dof 뿐)
        m.dof_frictionloss[i] = TAUC
        m.dof_damping[i] = KD_RAW[i % 4] * FLOAT_KD
    return m

def gid(m, name):
    g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, name)
    assert g >= 0, name
    return g

def run(mjcf, q4, leg, point, r4, label, t_end=16.0):
    m = load_fixed(mjcf, q4)
    d = mj.MjData(m)
    md = mj.MjData(m)                                # 제어기용 forward (배포기 재현)
    off = 0 if leg == 'HL' else 4
    gt = gid(m, f"{leg}_sphere")
    try: gh = gid(m, f"{leg}_sphere2")
    except AssertionError: gh = -1
    gpush = gt if point == 'toe' else gh
    floor = 0                                        # 첫 geom = 바닥(plane) 가정
    assert m.geom_type[floor] == mj.mjtGeom.mjGEOM_PLANE
    d.qpos[:8] = np.concatenate([q4, q4]); d.qvel[:] = 0

    r8 = np.ones(8); r8[off:off+4] = r4
    dt = m.opt.timestep
    log = []
    F = 0.0
    for step in range(int(t_end / dt)):
        t = step * dt
        F = min(F_MAX, PUSH_RATE * max(0.0, t - 2.0))    # 2 s 정착 후 램프
        # ── 제어기 재현: 측정 q 로 forward → g*·G + Jᵀ(0,0,−F) ──
        md.qpos[:8] = d.qpos[:8]; md.qvel[:] = 0
        mj.mj_forward(m, md)
        tau_cmd = GSTAR * md.qfrc_bias[:8]
        jac = np.zeros((3, m.nv)); mj.mj_jacGeom(m, md, jac, None, gpush)
        tau_cmd += jac[2, :8] * (-F)
        d.qfrc_applied[:] = 0
        d.qfrc_applied[:8] = r8 * tau_cmd
        mj.mj_step(m, d)
        if step % int(0.25 / dt) == 0:
            ft = fh = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                pair = {c.geom1, c.geom2}
                if floor not in pair: continue
                other = (pair - {floor}).pop()
                f6 = np.zeros(6); mj.mj_contactForce(m, d, ci, f6)
                if other == gt: ft += f6[0]
                elif other == gh: fh += f6[0]
            q = np.rad2deg(d.qpos[off:4+off])
            dz = (d.geom_xpos[gh][2] - d.geom_xpos[gt][2]) * 1000 if gh >= 0 else 0.0
            log.append((t, F, ft, fh, q.copy(), dz))
    # ── 요약 ──
    print(f"\n■ {label}")
    print("   t     F    toe[N]  heel[N]  합/F    q(hip,thigh,calf,foot)         heel-toe[mm]")
    for (t, F, ft, fh, q, dz) in log[::8]:
        tot = (ft + fh) / F if F > 1 else float('nan')
        print(f"  {t:5.1f} {F:5.1f}  {ft:6.1f}  {fh:6.1f}  {tot:5.2f}  [{','.join(f'{v:7.2f}' for v in q)}]  {dz:+7.1f}")
    t, F, ft, fh, q, dz = log[-1]
    print(f"  최종: T={(ft+fh)/F:.3f}  toe {ft:.1f} N / heel {fh:.1f} N  발목각 {q[3]:+.2f}°  heel-toe {dz:+.1f} mm")
    return log

if __name__ == '__main__':
    flat, quad = 'biped_flatfoot.mjcf', 'biped_from_quad.mjcf'
    r = lambda f=0.80: np.array([0.80, 0.80, 0.80, f])
    run(flat, QFLAT4, 'HL', 'toe',  r(),     "A. 평발·발끝밀기·r공통 0.80  (② 재현 검사)")
    run(flat, QFLAT4, 'HL', 'toe',  r(0.66), "B. 평발·발끝밀기·r_foot 0.66 (떠오름 원인 가설)")
    run(flat, QFLAT4, 'HL', 'heel', r(),     "C. 평발·뒤꿈치밀기(E3)·r공통 0.80")
    run(flat, QFLAT4, 'HL', 'heel', r(0.66), "D. E3·r_foot 0.66 — foot 무감 검사")
    run(quad, QHOME4, 'HL', 'toe',  r(),     "E. 1점 Qhome8·발끝밀기(E4)·r공통 0.80")
