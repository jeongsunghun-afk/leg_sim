#!/usr/bin/env python3
"""scale_check.py — 축별 **보고각 vs 실제각 배율**을 재는 도우미.

왜 필요한가 (2026-08-10):
  JOG 검증 중 "calf 가 뷰어에서 실제보다 훨씬 많이 움직인다" 는 관찰이 나왔다.
  Emb 는 스케일 변환을 하지 않는다(fDir=±1 만 곱하고 드라이버 값을 그대로 통과,
  halGait.cpp:477-497). 따라서 배율 오차가 있다면 **드라이버/MCU** 쪽이다.

가설 (배율로 갈린다):
  1.50배 → 드라이버가 전 축을 7:1 로 가정. calf 실제 10.5 → 10.5/7 = 1.50
           (같은 논리면 foot 은 8.4/7 = 1.20배)
  그 외   → 무릎 링키지(4절 등)로 모터각 ≠ 링크각. 이 경우 **자세에 따라 배율이 변한다**
           → 여러 각도에서 재보면 구분된다.
  1.00배 → 보고값은 맞고 뷰어(MJCF 축/범위)만 틀린 것.

사용법:
  1) 제어기(app/biped_emb.py)와 GUI 를 띄운 상태에서 실행
  2) 이 스크립트가 축을 고르라고 하면 축 번호 입력
  3) GUI JOG 슬라이더로 그 축을 움직이거나, off 상태에서 손으로 돌린다
  4) 실제 회전각을 각도기/눈금으로 재서 입력 → 배율이 나온다

  python3 diag/scale_check.py            # 대화형
  python3 diag/scale_check.py --watch 2  # ch2 만 실시간 감시(측정 준비용)

⚠ 이 스크립트는 **읽기 전용**이다. 모터에 아무것도 쓰지 않는다(state.json 만 읽음).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")
GEAR = {0: 7.0, 1: 7.0, 2: 10.5, 3: 8.4, 4: 7.0, 5: 7.0, 6: 10.5, 7: 8.4}


def read_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return None


def names(st):
    n = len(st.get("q_leg_deg", []))
    return st.get("joint_names") or [f"ch{i}" for i in range(n)]


def watch(idx: int, secs: float = 1e9):
    """축 하나를 실시간 감시하며 min/max/누적변화를 보여준다."""
    st = read_state()
    if st is None:
        sys.exit(f"✗ 상태파일을 못 읽는다: {STATE}\n  제어기(app/biped_emb.py)가 떠 있는지 확인할 것.")
    q0 = st["q_leg_deg"][idx]
    lo = hi = q0
    t0 = time.time()
    print(f"[watch] ch{idx} 감시 시작 (시작각 {q0:+.2f}°). Ctrl-C 로 종료.\n")
    try:
        while time.time() - t0 < secs:
            st = read_state()
            if st:
                q = st["q_leg_deg"][idx]
                lo = min(lo, q); hi = max(hi, q)
                print(f"\r  현재 {q:+8.2f}°   범위 [{lo:+.2f}, {hi:+.2f}]   "
                      f"총 변화 {hi-lo:6.2f}°   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    print(f"\n\n[watch] ch{idx} 보고 총 변화량 = {hi-lo:.2f}°")
    return hi - lo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, help="이 채널만 실시간 감시(측정 준비용)")
    a = ap.parse_args()

    st = read_state()
    if st is None:
        print(f"✗ 상태파일을 못 읽는다: {STATE}")
        print("  제어기를 먼저 띄울 것:  cd emb && python3 app/biped_emb.py")
        return 1
    nm = names(st)

    if a.watch is not None:
        watch(a.watch)
        return 0

    print("=== 축별 현재 보고각 ===")
    for i, q in enumerate(st["q_leg_deg"]):
        print(f"  {i}: {nm[i]:10} {q:+8.2f}°   (감속비 {GEAR.get(i, '?')})")
    print()
    try:
        idx = int(input("측정할 축 번호: ").strip())
    except (ValueError, EOFError):
        return 1
    if not 0 <= idx < len(st["q_leg_deg"]):
        print("✗ 범위 밖"); return 1

    print(f"\n이제 ch{idx}({nm[idx]}) 를 움직이세요.")
    print("  · JOG 모드에서 슬라이더로 움직이거나")
    print("  · off(limp) 상태에서 손으로 돌려도 된다")
    print("  움직임이 끝나면 Ctrl-C 를 누르세요.\n")
    reported = watch(idx)

    try:
        actual = float(input("\n실제 회전각[deg] (각도기/눈금으로 측정): ").strip())
    except (ValueError, EOFError):
        return 1
    if abs(actual) < 1e-6:
        print("✗ 실제각이 0 이면 배율을 못 구한다"); return 1

    ratio = reported / actual
    g = GEAR.get(idx)
    print(f"\n{'='*56}")
    print(f"  보고 변화량 : {reported:8.2f}°")
    print(f"  실제 변화량 : {actual:8.2f}°")
    print(f"  ★배율       : {ratio:8.3f} 배")
    print(f"{'='*56}")
    if g:
        print(f"\n  이 축 감속비 {g} · 기준축(hip/thigh) 7.0 → 예상 배율 {g/7.0:.2f}")
        if abs(ratio - g / 7.0) < 0.08:
            print(f"  ⇒ **드라이버가 전 축을 7:1 로 가정**하고 있을 가능성이 높다.")
            print(f"     같은 논리면 foot 은 {8.4/7:.2f}배, calf 는 {10.5/7:.2f}배 여야 한다 — 다른 축도 재서 확인할 것.")
        elif abs(ratio - 1.0) < 0.05:
            print(f"  ⇒ 보고값은 정확하다. 뷰어(MJCF 축/범위) 쪽을 의심할 것.")
        else:
            print(f"  ⇒ 감속비 가설({g/7.0:.2f})과 안 맞는다. 링키지(모터각≠링크각) 가능성.")
            print(f"     ★그 경우 **자세에 따라 배율이 변한다** — 다른 각도에서 한 번 더 재서 확인할 것.")
    print("\n⚠ 배율이 1.0 이 아니면 그 축의 모든 각도 기반 값이 틀어진다:")
    print("   jog 한계·home 목표·PACE I_link·MJCF 대조 전부. 확정되면 반드시 config 에 반영할 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
