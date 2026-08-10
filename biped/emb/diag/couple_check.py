#!/usr/bin/env python3
"""couple_check.py — calf ↔ foot **기구 커플링** 계수 측정.

관찰(2026-08-10, 사용자): calf 를 움직이면 foot 각도가 따라 움직인다.
  발목이 무릎/대퇴에서 링키지(또는 벨트)로 구동되는 구조면 흔한 현상이다.
  MJCF 는 4관절을 **독립**으로 모델링하고 있고 코드에도 보정이 없다 — 완전한 공백이다.

★먼저 갈라야 하는 것 — **엔코더가 커플링을 보는가?**
  (A) 엔코더가 foot **모터축**에 있다  → 무릎이 움직여도 foot 채널각은 안 변한다.
      그런데 실제 발목은 기울어진다. ⇒ 눈으로만 보이고 숫자로는 안 보인다.
      이 경우 보정은 **명령 쪽에만** 필요하다(측정은 이미 모터각이라 맞다).
  (B) 엔코더가 커플링 **뒤**를 본다      → 무릎을 움직이면 foot 채널각도 같이 변한다.
      ⇒ 측정·명령 **양쪽** 보정이 필요하다.
  이 스크립트는 채널각(=raw 보고각)끼리 회귀해서 (A)/(B) 를 가른다.
  ★회귀를 **채널각 공간**에서 하는 이유: 우리 변환(sign·k·offset)을 거치면 그 가정이
    결과에 섞여 든다. raw 끼리 재야 하드웨어가 실제로 뭘 하는지가 나온다.

측정 절차 (안전 — 아무것도 쓰지 않는다):
  1) 제어기를 띄우고 GUI 에서 **off(무여자)** 로 둔다.
  2) 이 스크립트를 실행한다.
  3) **무릎만** 손으로 천천히 전 구간 왕복시킨다. 발목은 잡지 말 것
     (발목을 같이 잡으면 커플링과 사람 손이 섞여 계수가 오염된다).
  4) Ctrl-C → 계수 c, 결정계수 R², 자세의존성 판정이 나온다.

  ⚠off 상태면 발목도 중력으로 흔들린다. 그 성분이 섞이므로 **천천히** 움직이고,
    R² 가 낮으면(<0.9) 결과를 믿지 말 것 — 그때는 hold 로 발목을 잡고 무릎만
    JOG 로 돌리는 편이 깨끗하다(--expect-hold 로 경고를 끈다).

사용법:
  python3 diag/couple_check.py                 # 양쪽 다리
  python3 diag/couple_check.py --leg HL        # 한쪽만
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")
EMB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGS = {"HL": (2, 3), "HR": (6, 7)}          # (calf, foot) 관절 인덱스


def cfg_idx():
    import yaml
    c = yaml.safe_load(open(os.path.join(EMB, "config", "biped_emb.yaml")))
    names = [j["name"] for j in c["joints"]]
    out = {}
    for leg in LEGS:
        out[leg] = (names.index(f"{leg}_calf"), names.index(f"{leg}_foot"))
    return c, out


def fit(x, y):
    """최소제곱 y = a + c·x. (c, a, R²) 반환."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    sxy = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    if sxx < 1e-12:
        return None, None, None
    c = sxy / sxx
    a = my - c * mx
    ss_res = sum((y[i] - (a + c * x[i])) ** 2 for i in range(n))
    ss_tot = sum((v - my) ** 2 for v in y)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return c, a, r2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=list(LEGS), help="한쪽만 측정")
    ap.add_argument("--expect-hold", action="store_true", help="hold 로 발목을 잡고 잰다(off 경고 끔)")
    a = ap.parse_args()
    cfg, idx = cfg_idx()
    legs = [a.leg] if a.leg else list(LEGS)

    try:
        st0 = json.load(open(STATE))
    except Exception as e:
        print(f"✗ 상태파일을 못 읽는다: {STATE} ({e})\n  제어기를 먼저 띄울 것."); return 1
    if "q_ch_deg" not in st0:
        print("✗ state 에 q_ch_deg 가 없다 — 제어기를 재시작할 것(구버전)."); return 1
    if st0.get("mode") != "off" and not a.expect_hold:
        print(f"  ⚠ 모드가 off 가 아니다({st0.get('mode')}). 손으로 무릎을 돌리려면 off 가 맞다.")

    print(f"\n  무릎(calf)만 천천히 전 구간 왕복시키세요. 발목은 잡지 마세요.")
    print(f"  Ctrl-C 로 종료하면 계수가 나옵니다.\n")
    rec = {l: [] for l in legs}
    try:
        while True:
            st = json.load(open(STATE))
            line = []
            for l in legs:
                ci, fi = idx[l]
                rec[l].append((st["q_ch_deg"][ci], st["q_ch_deg"][fi],
                               st["q_leg_deg"][ci], st["q_leg_deg"][fi]))
                span = max(r[0] for r in rec[l]) - min(r[0] for r in rec[l])
                line.append(f"{l} calf_ch {st['q_ch_deg'][ci]:+7.2f} foot_ch {st['q_ch_deg'][fi]:+7.2f} "
                            f"(무릎 범위 {span:5.1f}°)")
            print("\r  " + " | ".join(line) + "   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n")

    rc = 0
    for l in legs:
        d = rec[l]
        cch = [r[0] for r in d]; fch = [r[1] for r in d]
        cm  = [r[2] for r in d]; fm  = [r[3] for r in d]
        span = max(cch) - min(cch)
        print("=" * 70)
        print(f"  {l}   샘플 {len(d)}개 · 무릎 채널각 범위 {span:.1f}°")
        print("=" * 70)
        if span < 10.0:
            print(f"  ✗ 무릎을 {span:.1f}° 밖에 안 움직였다 — 최소 10°, 가능하면 30° 이상 움직일 것.\n")
            rc = 1; continue
        c_ch, _, r2_ch = fit(cch, fch)
        c_m,  _, r2_m  = fit(cm,  fm)
        print(f"  채널각 회귀 : foot_ch = {c_ch:+.4f} · calf_ch + const     R² = {r2_ch:.4f}")
        print(f"  모델각 회귀 : foot    = {c_m:+.4f} · calf    + const     R² = {r2_m:.4f}")
        print()
        if abs(c_ch) < 0.02:
            print(f"  ⇒ **(A) 엔코더는 커플링을 보지 못한다** (기울기 ≈ 0).")
            print(f"     엔코더가 foot 모터축에 있다는 뜻이다. 측정은 그대로 두고")
            print(f"     **명령 쪽에만** 보정이 필요하다 — 계수는 눈/각도기로 따로 재야 한다.")
        else:
            print(f"  ⇒ **(B) 엔코더가 커플링을 본다.** 무릎 1° 당 발목 채널각이 {c_ch:+.3f}° 움직인다.")
            print(f"     측정·명령 양쪽을 보정해야 한다.")
            print(f"     모델각 기준 커플링 계수 c = {c_m:+.4f}  ← 이 값을 config 에 넣는다")
        if r2_ch is not None and r2_ch < 0.90:
            print(f"\n  ⚠ R² {r2_ch:.3f} 로 낮다 — 선형이 아니거나 잡음(발목이 중력으로 흔들림)이 섞였다.")
            print(f"     hold 로 발목을 잡고 무릎만 JOG 로 돌려 다시 잴 것(--expect-hold).")
        # 자세 의존성: 앞/뒤 절반을 따로 적합해 비교
        h = len(d) // 2
        c1, _, _ = fit(cch[:h], fch[:h]); c2, _, _ = fit(cch[h:], fch[h:])
        if c1 is not None and c2 is not None and abs(c_ch) > 0.02:
            dev = abs(c1 - c2) / max(abs(c_ch), 1e-9) * 100
            print(f"\n  자세 의존성: 전반 {c1:+.4f} · 후반 {c2:+.4f}  (차이 {dev:.0f}%)")
            print("  " + ("⇒ 상수로 봐도 된다 — 선형 보정으로 충분하다." if dev < 15 else
                          "⇒ ★자세에 따라 변한다 — 상수 c 로는 부족하다. 링키지 기구학이 필요하다."))
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
