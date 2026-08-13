#!/usr/bin/env python3
"""ch_probe.py — 채널 상태를 **읽기만** 하며 실시간으로 보여준다. 모터에 아무것도 안 쓴다.

★왜 필요한가 (2026-08-12)
  다축 처프에서 축이 "파워단 사망" 으로 판정됐는데, 작업자가 **그 축만 힘이 들어가
  있다**고 관찰했다. 죽은 파워단은 힘을 낼 수 없다 — 둘 중 하나가 틀렸다.
  ⇒ 손으로 움직여 보면서 전류·토크가 따라 움직이는지 **직접** 본다.

판별법
  · 손으로 돌릴 때 **전류가 0 근처로 가만히** 있다 → 모터는 꺼져 있다.
    (뻑뻑한 건 감속기다. calf 는 실제 10.5:1 · foot 8.4:1 라 무여자여도 잘 안 돌아간다.)
  · 손으로 돌릴 때 **전류가 따라 뛴다** → 모터가 버티고 있다. 파워단은 살아 있고
    명령 경로가 문제다(옛 setpoint 를 물고 있거나 모드가 다르다).
  · 값이 **아예 안 변한다**(전 채널) → SHM 동결. RUNBOOK 의 동결 판별 참조.

사용:
    python3 diag/ch_probe.py            # 전 채널
    python3 diag/ch_probe.py 2          # ch2 만 크게
"""
from __future__ import annotations

import ctypes as C
import os
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.dirname(HERE)
SPEC = os.path.join(EMB, "pace", "spec.yaml")


def main() -> int:
    only = int(sys.argv[1]) if len(sys.argv) > 1 else None
    sp = yaml.safe_load(open(SPEC, encoding="utf-8"))
    n = int(sp["shm"]["n_channel"])
    lib = C.CDLL(sp["shm"]["lib"])
    F, I = C.POINTER(C.c_float), C.POINTER(C.c_int)
    lib.bridge_init.argtypes = [C.c_int]
    lib.bridge_read.argtypes = [F] * 7 + [I, I]
    z = lambda k=n: np.zeros(k, np.float32)
    q, dq, tau, cur = z(), z(), z(), z()
    rpy, acc, gyr = z(3), z(3), z(3)
    conn, stt = np.zeros(n, np.int32), np.zeros(n, np.int32)
    p = lambda a: a.ctypes.data_as(F)
    ip = lambda a: a.ctypes.data_as(I)
    if lib.bridge_init(int(sp["shm"]["recv_wait_ms"])) != 0:
        print("✗ bridge_init 실패 — Emb 가 떠 있는지 볼 것")
        return 1

    names = {int(j["ch"]): j["name"] for j in sp["joints"]}
    chs = [only] if only is not None else sorted(names)
    print("■ 읽기 전용 — 모터에 아무것도 쓰지 않는다. Ctrl+C 로 종료")
    print("  ★해당 축을 **손으로 천천히 돌려 보라.**")
    print("    전류가 0 근처로 가만히 있으면 → 모터는 꺼져 있다(뻑뻑한 건 감속기다)")
    print("    전류가 따라 뛰면            → 모터가 버티고 있다(파워단은 살아 있다)\n")
    print("    " + "".join(f"{names.get(c, c):>22}" for c in chs))
    # 손으로 돌린 범위를 누적해 보여준다 — '움직이긴 하는가' 를 눈으로 확인
    lo = np.full(n, +1e9)
    hi = np.full(n, -1e9)
    imax = np.zeros(n)
    try:
        while True:
            lib.bridge_read(p(q), p(dq), p(tau), p(cur), p(rpy), p(acc), p(gyr),
                            ip(conn), ip(stt))
            lo = np.minimum(lo, q[:n])
            hi = np.maximum(hi, q[:n])
            imax = np.maximum(imax, np.abs(cur[:n]))
            row = "".join(f"{q[c]:>8.2f}{cur[c]:>7.2f}A{tau[c]:>7.2f}" for c in chs)
            print("\r    " + row, end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n")
        print(f"    {'축':<10}{'움직인 폭':>10}{'최대 |전류|':>12}{'stt':>6}{'conn':>6}   판정")
        for c in chs:
            span = hi[c] - lo[c] if hi[c] > lo[c] else 0.0
            if span < 0.5:
                v = "안 움직였다 — 더 세게 돌려 볼 것"
            elif imax[c] < 0.15:
                v = "**모터 꺼짐** (뻑뻑한 건 감속기다)"
            else:
                v = "**모터가 버틴다** — 파워단은 살아 있다"
            print(f"    {names.get(c, c):<10}{span:>9.2f}°{imax[c]:>11.2f}A"
                  f"{int(stt[c]):>6}{int(conn[c]):>6}   {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
