#!/usr/bin/env python3
"""chain_check.py — GUI 명령 → 실제 관절 → 뷰어 사슬을 확인하는 **관찰 전용** 도구.

    GUI 명령(모델각) ──sign·offset──> 채널각 ──모터──> [실제 관절각] ──엔코더──>
    채널각 ──sign·offset──> 발행 q_leg_deg(모델각) ──> 뷰어·GUI 표시

★★GUI 표시와 뷰어 표시가 일치한다고 각도가 맞는 게 아니다 — 반드시 읽을 것:
  명령과 피드백이 **같은 변환을 거치므로** 둘은 하드웨어가 어떻든 항상 서로 일치한다.
  위 그림에서 [실제 관절각] 만 진짜 오차를 드러낸다.
  ⇒ 이 도구는 (a) 제어 추종(명령 vs 피드백) (b) 부호 를 본다.
    각도 **크기**는 --physical 로 각도기/수평계 측정값을 넣어야 검증된다.

⚠알려진 미해결: calf·foot 은 드라이버 감속비 오설정으로 채널각이 실제의 1.5/1.2 배로
  추정된다(소프트 보정 안 함 — config/biped_emb.yaml 사유 참조). --physical 로 그 배율을
  실측해 RGA 에 전달할 수 있다.

사용법 (GUI 로 조작하면서 이 창을 함께 본다. 명령 파일을 쓰지 않으므로 GUI 와 충돌 없음):
  python3 diag/chain_check.py                 # 8축 실시간 관찰
  python3 diag/chain_check.py --axis 2        # 한 축만 크게
  python3 diag/chain_check.py --axis 2 --physical   # 실제각 입력 → scale 검증

⚠ 읽기 전용이다. 모터에도 명령 채널에도 아무것도 쓰지 않는다.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

CMD   = os.environ.get("QUAD_CMD",   "/tmp/biped_cmd.json")
STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def cfg_names():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        import yaml
        c = yaml.safe_load(open(os.path.join(here, "config", "biped_emb.yaml")))
        return ([j["name"] for j in c["joints"]],
                [1.0] * len(c["joints"]),          # scale 은 규약에서 제거됨(항상 1)
                [int(j["sign"]) for j in c["joints"]])
    except Exception:
        n = 8
        return [f"ch{i}" for i in range(n)], [1.0]*n, [1]*n


def watch(axis=None):
    names, scale, sign = cfg_names()
    base_cmd = base_meas = None
    print("GUI 로 축을 움직이세요. Ctrl-C 로 종료.")
    print("  cmd  = GUI 가 보낸 목표(모델각)   ·   meas = 제어기가 발행한 실측(모델각, 뷰어와 동일)")
    print("  Δ    = 시작 시점 대비 변화량 — **이 두 Δ 가 같아야 추종이 맞는 것**\n")
    try:
        while True:
            c, s = load(CMD), load(STATE)
            if not c or not s:
                print("\r  (제어기/GUI 대기중…)   ", end="", flush=True); time.sleep(0.2); continue
            cmd = c.get("jog_deg") or [0.0]*len(names)
            meas = s.get("q_leg_deg") or [0.0]*len(names)
            if base_cmd is None:
                base_cmd, base_meas = list(cmd), list(meas)
            idxs = [axis] if axis is not None else range(min(len(names), len(meas)))
            lines = []
            for i in idxs:
                dc = cmd[i] - base_cmd[i]
                dm = meas[i] - base_meas[i]
                err = dm - dc
                flag = ""
                if abs(dc) > 1.0:
                    if abs(err) > max(1.0, abs(dc) * 0.15):
                        flag = "  ★추종오차"
                    if dc * dm < 0:
                        flag = "  ★★부호반대"
                lines.append(f"  {names[i]:9} cmd {cmd[i]:+7.2f} (Δ{dc:+6.2f}) | "
                             f"meas {meas[i]:+7.2f} (Δ{dm:+6.2f}) | 오차 {err:+6.2f}{flag}")
            out = f"mode={s.get('mode','-'):5} " + f"scale={scale[axis] if axis is not None else ''}\n" + "\n".join(lines)
            print("\033[2J\033[H" + out, flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\n")
        return base_cmd, base_meas, cmd, meas, names, scale, sign


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", type=int, help="관찰할 축 번호(0~7). 생략하면 전 축")
    ap.add_argument("--physical", action="store_true",
                    help="종료 시 실제 관절각 변화를 입력받아 scale 을 검증")
    a = ap.parse_args()

    if load(STATE) is None:
        print(f"✗ 상태파일이 없다: {STATE}\n  제어기를 먼저 띄울 것: cd emb && python3 app/biped_emb.py")
        return 1

    r = watch(a.axis)
    if not a.physical or a.axis is None or r is None:
        return 0
    base_cmd, base_meas, cmd, meas, names, scale, sign = r
    i = a.axis
    dc, dm = cmd[i] - base_cmd[i], meas[i] - base_meas[i]
    print(f"  {names[i]}:  명령 Δ{dc:+.2f}°  ·  발행(뷰어) Δ{dm:+.2f}°")
    if abs(dm) < 0.5:
        print("  ✗ 움직임이 너무 작다(0.5° 미만) — 다시 재라."); return 1
    try:
        actual = float(input("  실제 관절 회전각[deg] (각도기/수평계로 측정, 부호 포함): ").strip())
    except (ValueError, EOFError):
        return 1
    if abs(actual) < 1e-6:
        print("  ✗ 0 이면 배율을 못 구한다"); return 1

    ratio = dm / actual
    print(f"\n{'='*58}")
    print(f"  ★보고(뷰어) / 실제  =  {dm:+.2f} / {actual:+.2f}  =  {ratio:6.3f} 배")
    print(f"{'='*58}")
    if abs(ratio - 1.0) < 0.05:
        print("  ⇒ ✅ 각도가 맞다. 이 축은 드라이버 감속비가 올바르게 설정돼 있다.")
    else:
        print(f"  ⇒ ❌ 드라이버가 {ratio:.3f} 배로 과대/과소 보고한다.")
        for g, nm in ((7.0, "hip/thigh 7:1"), (10.5, "calf 10.5:1"), (8.4, "foot 8.4:1")):
            if abs(ratio - g / 7.0) < 0.06:
                print(f"     → {g}/7 = {g/7:.2f} 와 일치. **드라이버가 이 축을 7:1 로 가정**하고 있다.")
                print(f"        RGA 에 '{nm} 로 설정 필요' 로 전달할 것.")
        print("     ⚠부호가 반대로 나오면 감속비가 아니라 sign 문제다.")
        print("     ⚠소프트로 보정하지 않는 이유는 config/biped_emb.yaml 의 사유 참조.")
    print("\n  ⚠자세를 바꿔 한 번 더 재라 — 배율이 자세에 따라 변하면 감속비가 아니라 링키지다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
