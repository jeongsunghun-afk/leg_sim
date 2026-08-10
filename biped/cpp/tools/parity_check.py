#!/usr/bin/env python3
"""parity_check.py — JointMap Python ↔ C++ 패리티 검사.

★왜 있는가 (2026-08-10): Python 에 ±180 포화를 넣고 C++ 미러를 빠뜨렸는데, 당시
  "패리티 확인" 이 **기준자세 한 점**만 봐서 통과했다. 실제로는 관절한계 꼭짓점에서
  60.09° 어긋나 있었다. **한 점 검증은 패리티 검증이 아니다.**
  ⇒ build/biped_parity 가 한계 박스 전 꼭짓점(2^8=256) + 무작위 내부점을 뱉고,
    여기서 같은 입력을 Python JointMap 에 넣어 비교한다.

사용법:  python3 cpp/tools/parity_check.py [--tol 2e-5]
"""
from __future__ import annotations
import argparse, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CPP  = os.path.dirname(HERE)
ROOT = os.path.dirname(CPP)
CFG  = os.path.join(ROOT, "emb", "config", "biped_emb.yaml")
BIN  = os.path.join(CPP, "build", "biped_parity")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=2e-5, help="허용 최대차[deg]. float32 왕복 수준")
    ap.add_argument("--config", default=CFG)
    a = ap.parse_args()
    if not os.path.exists(BIN):
        print(f"✗ {BIN} 가 없다 — cmake --build build 먼저"); return 2
    sys.path.insert(0, os.path.join(ROOT, "emb", "interface"))
    import yaml
    from joint_map import JointMap
    jm = JointMap(yaml.safe_load(open(a.config)))

    r = subprocess.run([BIN, a.config], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"✗ biped_parity 실패: {r.stderr}"); return 1
    rows = [l.split() for l in r.stdout.splitlines() if l.strip()]
    print(f"  표본 {len(rows)}개 (꼭짓점 {sum(1 for x in rows if x[0]=='V')} + 무작위 {sum(1 for x in rows if x[0]=='R')})")
    print(f"  C++ 포화 통계: {r.stderr.strip()}")

    # ★전 경로를 본다 — 위치만 보다가 q_ctrl_to_ch 의 포화 누락(54.9° 차)을 놓쳤다.
    D2R = np.pi / 180.0
    PATHS = ["q_joint_to_ch", "q_ctrl_to_ch", "dq_ctrl_to_ch", "tau_ctrl_to_ch", "ch_to_q_joint(왕복)"]
    worst = {k: (0.0, None) for k in PATHS}
    for row in rows:
        segs, cur = [], []
        for tok in row[1:]:
            if tok == "|": segs.append(cur); cur = []
            else: cur.append(float(tok))
        segs.append(cur)
        q = np.array(segs[0])
        py = [
            jm.q_joint_to_ch(q)[jm.ch],
            jm.q_ctrl_to_ch(q * D2R)[jm.ch],
            jm.dq_ctrl_to_ch(q * D2R)[jm.ch],
            jm.tau_ctrl_to_ch(q)[jm.ch],
            jm.ch_to_q_joint(jm.q_joint_to_ch(q)),
        ]
        for k, name in enumerate(PATHS):
            cpp = np.array(segs[k + 1])
            d = float(np.max(np.abs(py[k] - cpp)))
            if d > worst[name][0]:
                worst[name] = (d, (row[0], q, py[k], cpp))
    print()
    bad = False
    for name in PATHS:
        d, _ = worst[name]
        ok = d <= a.tol
        bad |= not ok
        print(f"  {name:22} 최대차 {d:.3e}   {'✅' if ok else '❌'}")
    print(f"\n  허용 {a.tol:.0e}")
    if bad:
        for name in PATHS:
            d, w = worst[name]
            if d <= a.tol or w is None: continue
            tag, q, py_v, cpp = w
            i = int(np.argmax(np.abs(py_v - cpp)))
            print(f"\n  ❌ {name} — 유형 {tag} · 축 {jm.names[i]}")
            print(f"     q      = {np.round(q,2).tolist()}")
            print(f"     py/cpp = {py_v[i]:+.4f} / {cpp[i]:+.4f}")
        return 1
    print(f"  ✅ 통과 — 한계 박스 전 영역·전 변환경로에서 Python↔C++ 일치")
    return 0

if __name__ == "__main__":
    sys.exit(main())
