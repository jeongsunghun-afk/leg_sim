#!/usr/bin/env python3
"""발끝 무게추 변형 MJCF 생성 — foot(+calf) 경로를 float 브래킷으로 재기 위한 준비.

원리: float 브래킷이 calf/foot 을 못 쟀던 이유는 중력토크 ≪ 마찰이었다.
알고 있는 질량(주방저울 실측)을 발에 달면 foot 중력토크가 마찰을 넘고,
그 질량은 CAD 가 아니라 실측이므로 G비≈1 → 브래킷 1/g* ≈ r_foot 단독이 나온다.

사용 (Pi 에서도 됨 — mujoco 불필요, 순수 텍스트 가공):
  python3 tools/make_weighted_mjcf.py --leg HL --mass-g 1500 --at toe --x 0 --y 0 --z -40
    → biped_from_quad_wHL.mjcf 생성
  --at toe   : 오프셋 기준 = 발끝 구 중심(HL_foot_contact_link 원점)
  --at ankle : 오프셋 기준 = 발목 관절(HL_foot_link 원점)
  --x/--y/--z: 그 기준에서 mm (발 링크 좌표계)
  --base flat: 평발 모델 기반 (기본 1점 from_quad)

⚠추는 실험하는 다리에만 단다 — 모델도 그 다리에만 넣는다(반대다리 중력 ff 왜곡 방지).
⚠노트북(mujoco 있음)에서 돌리면 자세별 신호/마찰 타당성 표까지 출력한다.
"""
import argparse, io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(HERE)
TAU_C_FOOT = 0.639          # tendon 쿨롱마찰 [Nm] — 신호 타당성 판정 기준

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leg', choices=['HL', 'HR'], required=True)
    ap.add_argument('--mass-g', type=float, required=True, help='추+걸이 총질량 [g] 실측값')
    ap.add_argument('--at', choices=['toe', 'ankle'], default='toe')
    ap.add_argument('--x', type=float, default=0.0, help='기준점에서 x 오프셋 [mm]')
    ap.add_argument('--y', type=float, default=0.0)
    ap.add_argument('--z', type=float, default=0.0)
    ap.add_argument('--base', default='quad', help="quad|flat|MJCF경로")
    a = ap.parse_args()

    base = {'quad': 'biped_from_quad.mjcf', 'flat': 'biped_flatfoot.mjcf'}.get(a.base, a.base)
    src_path = base if os.path.isabs(base) else os.path.join(BIPED, base)
    s = io.open(src_path, encoding='utf-8').read()

    m_kg = a.mass_g / 1000.0
    # 부착점을 foot_contact_link(=발끝 구) 좌표로 통일. ankle 기준이면 링크 원점 오프셋을 더한다.
    # foot_contact_link 은 foot_link 안에서 pos="0.025353 0 -0.14378" (양다리 동일).
    off = [a.x / 1000.0, a.y / 1000.0, a.z / 1000.0]
    if a.at == 'ankle':
        off = [off[0] - 0.025353, off[1], off[2] + 0.14378]

    parent = f'<body name="{a.leg}_foot_contact_link" pos="0.025353 0 -0.14378">'
    assert s.count(parent) == 1, f'{parent} 를 찾지 못함 ({src_path})'
    # 점질량 body 삽입 — 회전관성은 점질량 근사(작은 값. 0 은 컴파일러가 거부할 수 있다)
    inject = (parent +
              f'\n                  <body name="{a.leg}_weight" pos="{off[0]:.6f} {off[1]:.6f} {off[2]:.6f}">'
              f'\n                    <inertial pos="0 0 0" mass="{m_kg:.4f}" diaginertia="2e-5 2e-5 2e-5" />'
              f'\n                  </body>'
              f'\n                  <!-- ★무게추 {a.mass_g:.0f} g @ {a.at}+({a.x},{a.y},{a.z})mm — make_weighted_mjcf.py 생성. 실험 후 이 파일 삭제 -->')
    s = s.replace(parent, inject, 1)

    stem = os.path.splitext(os.path.basename(src_path))[0]
    out = os.path.join(BIPED, f'{stem}_w{a.leg}.mjcf')
    io.open(out, 'w', encoding='utf-8').write(s)
    print(f'생성: {out}  (추 {a.mass_g:.0f} g · {a.leg} · {a.at} 기준 +({a.x},{a.y},{a.z}) mm)')

    # ── 타당성 표 (mujoco 있으면 — 노트북) ──
    try:
        import numpy as np, mujoco as mj
    except ImportError:
        print('(mujoco 없음 — 자세별 타당성 표는 노트북에서 같은 명령으로 확인)')
        return
    m = mj.MjModel.from_xml_path(out); d = mj.MjData(m)
    free = m.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
    qoff, voff = (7, 6) if free else (0, 0)
    jidx = {'HL': (0, 3), 'HR': (4, 7)}[a.leg]
    # 조건 A: foot 관절각 0° 창(E4/Qhome8 의심 구간) — home = Qhome8 그대로
    # 조건 B: foot 관절각 −50° 창(평발 구간) — home 을 [0,30,−20,−50] 로 임시 변경
    poses = [('0° (영점자세) — 참고용, 측정불가', [0, 0, 0, 0]),
             ('A: Qhome8 (foot관절 0° 창)', [0, 11.634, -38.454, 0]),
             ('B: [0,30,-20,-50] (foot관절 -50° 창)', [0, 30, -20, -50])]
    print(f'\n자세별 foot 축 신호 (추 기여 중력토크 vs tendon 마찰 {TAU_C_FOOT} Nm):')
    print('  자세                                G_foot계 [Nm]   판정 (>1.3 Nm 권장)')
    for name, q4 in poses:
        d.qpos[:] = 0
        if free: d.qpos[3] = 1.0
        q8 = q4 + q4
        d.qpos[qoff:qoff+8] = np.deg2rad(q8); d.qvel[:] = 0
        mj.mj_forward(m, d)
        gfoot = d.qfrc_bias[voff + jidx[1]]
        verdict = '✅ 측정 가능' if abs(gfoot) > 2 * TAU_C_FOOT else ('△ 밴드 넓음' if abs(gfoot) > TAU_C_FOOT else '❌ 마찰에 묻힘')
        print(f'  {name:34s} {gfoot:+8.2f}        {verdict}')
    print('\n다음: run_deploy_hw.sh 에 이 MJCF 경로를 넘겨 기동 → float_gstar --axis %s_foot' % a.leg)

if __name__ == '__main__':
    main()
