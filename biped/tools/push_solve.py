#!/usr/bin/env python3
"""push 스윕 전체 → 경로 전달비 r=(hip,thigh,calf,foot) 연립 해석.

각 스윕에서:
  · 미는 점 자동판별 — 기록된 τ_cmd 의 dτ/dF 를 발끝/뒤꿈치 자코비안과 대조
    (도구가 PUSH_POINT 를 기록하지 않으므로 데이터가 스스로 말하게 한다)
  · 상승가지 후반 기울기 T 적합 (시작 후 1.2 kgf 마찰전이 제외 · 30 N 초과점 제외
    — 50 N 스윕의 고력 처짐 구간 배제 · 자세붕괴(축사망) 점 자동 마스킹)
  · 실측-q 가중 w_j = Jz_j²/ΣJz² (미는 점 기준 · 사용점 평균)
다리별로  Σ_스윕 [ w·r = T ]  가중 최소자승 → r ± σ, 특이값(식별성).

사용:
  python3 tools/push_solve.py                  # data/push/*.json 전부
  python3 tools/push_solve.py --selftest       # 가상 E3·E4 로 복원 검증
"""
import argparse, glob, json, os, sys
import numpy as np
import mujoco as mj

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(HERE)
AX = ['hip', 'thigh', 'calf', 'foot']

_models = {}
def model(flat):
    key = 'biped_flatfoot.mjcf' if flat else 'biped_from_quad.mjcf'
    if key not in _models:
        m = mj.MjModel.from_xml_path(os.path.join(BIPED, key))
        _models[key] = (m, mj.MjData(m))
    return _models[key]

def jz(m, d, q8deg, geom, off_v):
    d.qpos[:] = 0
    if m.jnt_type[0] == mj.mjtJoint.mjJNT_FREE:
        d.qpos[3] = 1.0
    qoff = 7 if m.jnt_type[0] == mj.mjtJoint.mjJNT_FREE else 0
    d.qpos[qoff:qoff+8] = np.deg2rad(q8deg); d.qvel[:] = 0
    mj.mj_forward(m, d)
    g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, geom)
    if g < 0: return None
    jac = np.zeros((3, m.nv)); mj.mj_jacGeom(m, d, jac, None, g)
    return jac[2, off_v:off_v+4].copy()

def analyze_sweep(rows, leg):
    """→ dict(point, T, sigT, w, n_used, flags) 또는 None(사용불가)."""
    q0 = np.array(rows[0]['q_leg_deg'])
    q4 = q0[:4] if leg == 'HL' else q0[4:]
    flat = q4[3] < -30
    m, d = model(flat)
    voff = 6 if m.jnt_type[0] == mj.mjtJoint.mjJNT_FREE else 0
    off_v = voff + (0 if leg == 'HL' else 4)
    sl = slice(0, 4) if leg == 'HL' else slice(4, 8)

    F = np.array([r['F_cmd'] for r in rows])
    S = np.array([r['scale_g'] for r in rows]) / 1000.0
    Q = np.array([r['q_leg_deg'] for r in rows])
    TAU = np.array([r['tau_cmd_nm'] for r in rows])[:, sl]

    # ── 미는 점 자동판별: dτ/dF (상승가지) vs −Jz(toe/heel) ──
    ap = int(np.argmax(F))
    up = np.arange(len(F)) <= ap
    cand = {}
    for name, geom in [('toe', f'{leg}_sphere'), ('heel', f'{leg}_sphere2')]:
        Js = [jz(m, d, Q[i], geom, off_v) for i in range(len(F)) if up[i]]
        if Js[0] is None: continue
        cand[name] = np.mean(Js, axis=0)
    slope = np.array([np.polyfit(F[up], TAU[up, j], 1)[0] for j in range(4)])
    point = min(cand, key=lambda k: np.linalg.norm(slope + cand[k]))
    resid = {k: np.linalg.norm(slope + v) for k, v in cand.items()}
    if resid[point] > 0.03 * 4:            # 축당 3 cm 이상 어긋나면 신뢰 불가
        return None

    # ── 오염 마스킹: 자세가 상승가지 궤적에서 3° 이상 튄 점(축사망) 제외 ──
    qref = np.median(Q[up], axis=0)
    bad = np.max(np.abs(Q - qref), axis=1) > 8.0
    # ── 상승가지 후반만: 시작 마찰전이 1.2 kgf 제외 · 30 N 초과(고력 처짐) 제외 ──
    use = up & ~bad & (F / G >= 1.2) & (F <= 30.0 + 1e-9)
    if use.sum() < 3:
        return None
    a, b = np.polyfit(F[use] / G, S[use], 1)
    res = S[use] - (a * F[use] / G + b)
    sig = max(np.sqrt((res**2).mean()) / (np.ptp(F[use]) / G) * 2, 0.01)
    geom = f'{leg}_sphere' if point == 'toe' else f'{leg}_sphere2'
    W = np.array([(lambda j: j**2 / (j @ j))(jz(m, d, Q[i], geom, off_v))
                  for i in np.where(use)[0]])
    return dict(point=point, pose=('평발' if flat else '0°/Qhome8'), T=a, sigT=sig,
                w=W.mean(axis=0), n=int(use.sum()), n_bad=int(bad.sum()))

def solve_leg(entries):
    Wm = np.array([e['w'] for e in entries])
    T = np.array([e['T'] for e in entries])
    sig = np.array([e['sigT'] for e in entries])
    A = Wm / sig[:, None]; y = T / sig
    r, *_ = np.linalg.lstsq(A, y, rcond=None)
    U, sv, Vt = np.linalg.svd(A)               # Vt 4×4 전체 — 영공간까지
    k = int(np.sum(sv > max(sv[0], 1.0) * 1e-3))
    Vr, Vn = Vt[:k], Vt[k:]
    cov = Vr.T @ np.diag(1.0 / sv[:k]**2) @ Vr
    err = np.sqrt(np.diag(cov))
    # ★영공간 성분이 큰 축은 "미식별" — 최소노름 해가 임의로 0 을 박은 방향이라
    #   숫자·오차 모두 무의미하다. lstsq 의 겉보기 정밀도에 속지 말 것.
    nullfrac = np.linalg.norm(Vn, axis=0) if k < 4 else np.zeros(4)
    pred = Wm @ r
    return r, err, sv[:k], nullfrac, pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--dir', default=os.path.join(BIPED, 'data', 'push'))
    args = ap.parse_args()

    per_leg = {'HL': [], 'HR': []}
    for f in sorted(glob.glob(os.path.join(args.dir, 'push_scale_*.json'))):
        D = json.load(open(f)); leg = D['leg']
        r = analyze_sweep(D['rows'], leg)
        tag = os.path.basename(f)[11:-5]
        if r is None:
            print(f"  ✗ {tag}: 사용불가(점 부족/판별 실패)")
            continue
        r['tag'] = tag
        per_leg[leg].append(r)
        print(f"  {tag:24s} {leg} {r['pose']:6s} {r['point']:4s}밀기  T={r['T']:.3f}±{r['sigT']:.3f} "
              f"({r['n']}점{'·오염'+str(r['n_bad']) if r['n_bad'] else ''})  w={np.round(r['w'],2)}")

    if args.selftest:
        rt = np.array([0.84, 0.78, 0.82, 0.80])
        m, d = model(True)
        voff = 6 if m.jnt_type[0] == mj.mjtJoint.mjJNT_FREE else 0
        qf = [0, 3.68, -23.87, -59.81]
        wE3 = (lambda j: j**2/(j@j))(jz(m, d, qf*2, 'HL_sphere2', voff))
        m2, d2 = model(False)
        voff2 = 6 if m2.jnt_type[0] == mj.mjtJoint.mjJNT_FREE else 0
        qh = [0, 11.634, -38.454, 0]
        wE4 = (lambda j: j**2/(j@j))(jz(m2, d2, qh*2, 'HL_sphere', voff2))
        for w, nm in [(wE3, 'E3(가상)'), (wE4, 'E4(가상)')]:
            per_leg['HL'].append(dict(tag=nm, point='-', pose='-', T=float(w @ rt),
                                      sigT=0.01, w=w, n=0, n_bad=0))
            print(f"  {nm:24s} HL 합성            T={w@rt:.3f}         w={np.round(w,2)}")
        print(f"  (참값 r = {rt})")

    for leg, entries in per_leg.items():
        if len(entries) < 2: continue
        r, err, sv, nullfrac, pred = solve_leg(entries)
        print(f"\n■ {leg}: 스윕 {len(entries)}개 연립  (유효특이값 {np.round(sv, 2)})")
        for a, v, e, nf in zip(AX, r, err, nullfrac):
            if nf > 0.5:
                print(f"  r_{a:5s} = 미식별 (영공간 {nf:.2f} — 이 축을 보는 스윕이 없음)")
            else:
                print(f"  r_{a:5s} = {v:.3f} ± {e:.3f}" + ("  ⚠약식별" if e > 0.15 else ""))
        for ent, p in zip(entries, pred):
            print(f"    {ent['tag']:24s} T측정 {ent['T']:.3f} vs 예측 {p:.3f}  ({(ent['T']-p)*1000:+.0f}×10⁻³)")

if __name__ == '__main__':
    main()
