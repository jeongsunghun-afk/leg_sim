#!/usr/bin/env python3
"""extract_ilink.py — MJCF 에서 관절축 링크관성 `I_link` 를 뽑는다 (spec.yaml 채우기용).

왜 필요한가
-----------
PACE 는 관절축 등가관성 `I_total` 을 측정하고, 거기서 로터관성을 분리한다:

    ROTOR_I = (I_total − I_link) / N²          (act_identify_pace.py:20)

`I_link` 는 **PACE 의 출력이 아니라 입력**이다. 다리 미장착 시절엔 출력축에 링크가
없어서 `I_link = 0` 이 맞았지만(spec.yaml 주석), 다리를 조립한 뒤로는 0 이 아니다.
0 으로 두고 돌리면 **다리 관성이 통째로 로터관성으로 둔갑**한다.

무엇을 계산하나
---------------
관절 i 의 축 â 둘레로, **그 관절보다 아래(subtree)에 달린 모든 링크**의 관성:

    I_link(i) = âᵀ · [ Σ_{b∈subtree}  R_b I_b R_bᵀ + m_b (|r_b|² E − r_b r_bᵀ) ] · â
                                                       r_b = com(b) − anchor(i)

이는 관절공간 질량행렬의 대각성분 M[i][i] 와 같다(단, armature 제외). 그래서 두 경로로
각각 구해 **교차검증**한다:
  (A) mujoco `mj_fullM` 대각성분  — armature 를 0 으로 두고 뽑는다.
      ★`dof_armature` 는 M 의 대각에 그대로 더해진다. 안 지우면 로터 반사관성이
        섞여 들어와 I_link 가 부풀고, 그걸 다시 ROTOR_I 분리에 쓰면 순환논법이 된다.
  (B) 위 식을 직접 합산 — CRBA 와 다른 코드경로라 (A) 의 오용을 잡아낸다.

자세 의존성
-----------
I_link 는 **자세의 함수**다. hip roll 축에서 보는 관성은 무릎이 펴져 있으면 크고
접혀 있으면 작다. 그래서 홈자세(q=0) 값과 함께 각 축의 시험범위에서의 변동폭을 같이
보고한다. PACE 시험은 관절을 흔들면서 재므로, 변동폭이 크면 "단일 I_link" 자체가
근사임을 알고 써야 한다.

실행:  ~/.venvs/mj/bin/python extract_ilink.py
       ~/.venvs/mj/bin/python extract_ilink.py --emit-yaml     # spec.yaml 붙여넣기용
"""
from __future__ import annotations

import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(os.path.dirname(HERE))          # simulation/biped
# ★배포 모델 = **점발(1pt)** biped_from_quad.mjcf (2026-08-06 확인).
#   ⚠biped_flatfoot.mjcf(평발 2pt)로 뽑아도 **I_link 는 완전히 동일**하다 —
#     두 모델의 차이는 heel 접촉구(sphere2) 뿐이고 그건 무질량 geom 이라
#     링크 관성·총질량(13.571 kg)에 기여하지 않는다. 검증 완료(전 축 유효숫자 7자리 일치).
#     그래도 기본값은 배포 모델로 둔다 — 나중에 두 모델이 갈릴 때 조용히 틀리지 않도록.
MJCF_DEFAULT = os.path.join(BIPED, "biped_from_quad.mjcf")

# spec.yaml 의 ch 순서 ↔ MJCF 관절명
CH2JOINT = [
    (0, "HL_hip",   "HL_hip_joint"),
    (1, "HL_thigh", "HL_thigh_joint"),
    (2, "HL_calf",  "HL_calf_joint"),
    (3, "HL_foot",  "HL_foot_joint"),
    (4, "HR_hip",   "HR_hip_joint"),
    (5, "HR_thigh", "HR_thigh_joint"),
    (6, "HR_calf",  "HR_calf_joint"),
    (7, "HR_foot",  "HR_foot_joint"),
]


def subtree_bodies(m, root_body: int) -> list[int]:
    """root_body 와 그 모든 자손 body id."""
    out, stack = [], [root_body]
    while stack:
        b = stack.pop()
        out.append(b)
        stack.extend(int(c) for c in range(m.nbody) if m.body_parentid[c] == b and c != b)
    return out


def ilink_direct(m, d, jid: int) -> float:
    """(B) 복합강체 관성을 직접 합산해 관절축에 투영. mj_fullM 과 독립된 경로."""
    import mujoco
    axis = d.xaxis[jid].copy()                       # 월드 좌표 관절축(단위벡터)
    axis = axis / np.linalg.norm(axis)
    anchor = d.xanchor[jid].copy()                   # 월드 좌표 관절 원점
    body = int(m.jnt_bodyid[jid])

    I_tot = np.zeros((3, 3))
    for b in subtree_bodies(m, body):
        mass = float(m.body_mass[b])
        if mass <= 0:
            continue
        R = d.ximat[b].reshape(3, 3)                 # 관성주축 프레임 → 월드
        I_com = R @ np.diag(m.body_inertia[b]) @ R.T # 무게중심 둘레 관성(월드)
        r = d.xipos[b] - anchor                      # 관절원점 → 무게중심
        I_tot += I_com + mass * (float(r @ r) * np.eye(3) - np.outer(r, r))
    return float(axis @ I_tot @ axis)


def ilink_fullM(m, d, jid: int) -> float:
    """(A) mj_fullM 대각성분. armature 는 호출 전에 0 으로 지워둘 것."""
    import mujoco
    M = np.zeros((m.nv, m.nv))
    # ★MuJoCo 3.11 에서 시그니처가 바뀌었다: mj_fullM(m, d.qM, dst) → mj_fullM(m, d, dst).
    #   구버전에서도 돌게 둘 다 시도한다(이 스크립트는 노트북/Pi 를 오간다).
    try:
        mujoco.mj_fullM(m, d, M)
    except (TypeError, ValueError):
        mujoco.mj_fullM(m, M, d.qM)
    return float(M[m.jnt_dofadr[jid], m.jnt_dofadr[jid]])


def evaluate(m, d, qpos_leg: dict | None = None):
    """주어진 자세에서 8축 I_link 를 두 경로로 계산해 반환."""
    import mujoco
    mujoco.mj_resetData(m, d)
    if qpos_leg:
        for jname, val in qpos_leg.items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            d.qpos[m.jnt_qposadr[jid]] = val
    mujoco.mj_forward(m, d)

    rows = []
    for ch, name, jname in CH2JOINT:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise SystemExit(f"MJCF 에 관절 {jname} 가 없다")
        rows.append((ch, name, jname, jid, ilink_fullM(m, d, jid), ilink_direct(m, d, jid)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default=MJCF_DEFAULT)
    ap.add_argument("--emit-yaml", action="store_true", help="spec.yaml 붙여넣기용 출력")
    a = ap.parse_args()

    import mujoco
    m = mujoco.MjModel.from_xml_path(a.mjcf)
    d = mujoco.MjData(m)
    arm0 = m.dof_armature.copy()
    # ★armature 제거 — 안 지우면 로터 반사관성이 M 대각에 섞여 I_link 가 부푼다.
    m.dof_armature[:] = 0.0

    print(f"MJCF: {a.mjcf}")
    print(f"  nq={m.nq} nv={m.nv} nu={m.nu}  (freejoint 6 + 관절 8)")
    print(f"  총질량 {float(np.sum(m.body_mass)):.3f} kg   "
          f"armature 원본 max={float(np.max(arm0)):.3e} → 계산 시 0 으로 지움\n")

    rows = evaluate(m, d, None)                       # 홈자세 q=0

    print("=== I_link @ 홈자세(q=0) — 두 경로 교차검증 ===")
    print(f"{'ch':>2} {'축':9} {'mj_fullM':>12} {'직접합산':>12} {'상대차':>9}")
    ok = True
    for ch, name, _jn, _jid, a_val, b_val in rows:
        rel = abs(a_val - b_val) / max(abs(a_val), 1e-12)
        ok &= rel < 1e-9
        print(f"{ch:2d} {name:9} {a_val:12.6e} {b_val:12.6e} {rel:9.1e}")
    print(f"\n교차검증: {'일치 ✓' if ok else '불일치 ✗ — 어느 한쪽이 틀렸다'}")

    # ── 자세 의존성: 각 축을 시험범위로 흔들 때 I_link 가 얼마나 변하나 ──
    #   ★근위축(hip)의 I_link 는 원위축(무릎 등) 자세에 크게 좌우된다.
    print("\n=== 자세 의존성 — 무릎(calf) 각도에 따른 I_link 변화 ===")
    print(f"{'calf[deg]':>10} " + " ".join(f"{n:>11}" for _, n, _, _, _, _ in rows[:4]))
    for calf_deg in (0, -20, -40, -55):
        q = {"HL_calf_joint": np.deg2rad(calf_deg), "HR_calf_joint": np.deg2rad(calf_deg)}
        r = evaluate(m, d, q)
        print(f"{calf_deg:10d} " + " ".join(f"{v[4]:11.4e}" for v in r[:4]))

    # ── PACE 예측: 다리 장착 상태에서 I_total 이 얼마로 나와야 하는가 ──
    #   I_total = I_link + ROTOR_I·N²   (ROTOR_I = PACE 실측 7.4e-4, 전축 동일 모터)
    ROTOR_I = 7.4e-4
    GEAR = {0: 7.0, 1: 7.0, 2: 10.5, 3: 8.4, 4: 7.0, 5: 7.0, 6: 10.5, 7: 8.4}
    print(f"\n=== PACE 예측 (ROTOR_I={ROTOR_I:.2e} 전축 동일 가정) ===")
    print(f"{'ch':>2} {'축':9} {'I_link':>12} {'ROTOR_I·N²':>12} {'예상 I_total':>13} {'링크비중':>8}")
    for ch, name, _jn, _jid, il, _b in rows:
        N = GEAR[ch]
        refl = ROTOR_I * N * N
        tot = il + refl
        print(f"{ch:2d} {name:9} {il:12.4e} {refl:12.4e} {tot:13.4e} {il/tot*100:7.1f}%")

    if a.emit_yaml:
        print("\n=== spec.yaml 붙여넣기용 (홈자세 q=0 기준) ===")
        for ch, name, _jn, _jid, il, _b in rows:
            print(f"  # ch{ch} {name}: I_link={il:.6e}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
