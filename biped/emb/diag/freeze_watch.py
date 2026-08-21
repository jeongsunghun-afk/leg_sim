#!/usr/bin/env python3
"""freeze_watch.py — 채널별 **갱신 여부**를 실시간으로 본다. 흔들기 시험용.

═══ 왜 필요한가 ═══
왼다리 FDCAN 동결이 2026-08-20 하루에 **5회** 났다. 소프트웨어로는 더 못 좁혔다:
  · 하중 때문이 아니다 — 마지막 건 stand 진입 46ms 만에, 왼다리 토크 2.5Nm 에서 났다
  · 명령 방식 때문도 아니다 — write_pos 와 write_mit 은 같은 함수(ucMode=1 동일)를 쓰고
    그 시점 fVelocity·fTorque 는 둘 다 ~0 이었다
  · health/n_fault 로는 안 보인다 — MCU 가 마지막 값을 계속 올려 **ok 로 위장**한다

남은 건 물리 점검이다. 끊기는 지점이 매번 다르다:
    ch0 ch1 ch2 ch3   → hip 앞
        ch1 ch2 ch3   → hip 과 thigh 사이
            ch2 ch3   → thigh 와 calf 사이
⇒ 공통 구간(몸통→다리)이 흔들리거나 여러 커넥터가 한계다.

═══ 쓰는 법 ═══
  ① 제어기를 **off 모드**로 띄운다(무여자 — 로봇이 안 움직여야 손으로 만진다)
       cd ~/simulation/biped/cpp && ./build/biped_deploy --mjcf ../biped_flatfoot.mjcf --start-mode off
  ② 이 스크립트를 다른 터미널에서 돌린다
       python3 emb/diag/freeze_watch.py
  ③ 하네스를 **구간별로** 손으로 흔든다. 커넥터·굽는 지점·클램프 순서로.
     얼어붙는 순간 화면이 빨갛게 바뀌고 **어느 채널부터** 죽었는지 남는다.

⚠읽기 전용이다 — 아무것도 쓰지 않는다. 제어기와 같이 돌려도 안전하다.
⚠실기 엔코더는 정지 중에도 미세하게 떨린다. 그 떨림이 이 검사의 근거다.
  0.4초 동안 (q,dq,tau) 가 셋 다 그대로면 동결로 본다.
"""
import json
import os
import sys
import time

STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")
NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot",
         "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
HOLD_S = 0.4          # 이 시간 동안 무변화면 동결
POLL_S = 0.05

R, G, Y, X = "\033[91m", "\033[92m", "\033[93m", "\033[0m"


def main():
    prev = None
    still = [0.0] * 8          # 채널별 무변화 누적시간
    worst = {}                 # 이번 세션에서 동결된 적 있는 채널 → 최장 시간
    t_last = time.time()
    print(__doc__.split("═══ 쓰는 법 ═══")[0].strip())
    print(f"\n  상태파일 {STATE} · 동결판정 {HOLD_S}s\n")
    while True:
        try:
            s = json.load(open(STATE))
        except Exception as e:
            print(f"  {Y}상태파일을 못 읽는다{X}: {e}", end="\r")
            time.sleep(0.5)
            continue
        q, dq, tau = s.get("q_leg_deg"), s.get("dq_leg_dps"), s.get("tau_leg_nm")
        if not (q and dq and tau):
            time.sleep(POLL_S)
            continue
        cur = [(q[i], dq[i], tau[i]) for i in range(8)]
        now = time.time()
        dt, t_last = now - t_last, now
        if prev:
            for i in range(8):
                still[i] = still[i] + dt if cur[i] == prev[i] else 0.0
                if still[i] >= HOLD_S:
                    worst[i] = max(worst.get(i, 0), still[i])
        prev = cur

        cells = []
        for i in range(8):
            if still[i] >= HOLD_S:
                cells.append(f"{R}{NAMES[i]}✗{X}")
            elif i in worst:
                cells.append(f"{Y}{NAMES[i]}·{X}")
            else:
                cells.append(f"{G}{NAMES[i]}✓{X}")
        dead = [NAMES[i] for i in range(8) if still[i] >= HOLD_S]
        tail = f"  {R}동결 {len(dead)}축: {' '.join(dead)}{X}" if dead else "  전축 정상"
        print("  " + " ".join(cells) + tail + " " * 8, end="\r")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  종료")
        sys.exit(0)
