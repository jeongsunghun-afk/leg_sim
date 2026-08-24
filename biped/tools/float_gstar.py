#!/usr/bin/env python3
"""float_gstar.py — 무중력 중립점 g* 를 **8축 동시에** 잰다.

★원리
    무중력에서 명령 토크는  τ_cmd = g · G_model(q)  이고, 드라이버가 α 를 곱한다.
        α·g·G_CAD  <  G_real   →  그 축은 **처진다**
        α·g·G_CAD  >  G_real   →  그 축은 **뜬다**
    ⇒ g 를 훑으면 각 축의 표류가 자기 중립점 g* 에서 **부호를 바꾼다.**
      전 축이 동시에 떠 있으므로 **한 번의 스윕으로 8개가 다 나온다.**

    g* 의 뜻:   1/g* = α · (G_CAD / G_real)
      · 8축이 다 비슷하면      → 공통 α (구동계). 저울로 게이지를 깨야 한다
      · 축마다 다르면          → **CAD 질량·CoM 오차**. 모델을 고쳐야 한다
      · 좌우만 다르면          → 좌우 비대칭(배선·조립)

★마찰이 데드밴드를 만든다
    |α·g·G_CAD − G_real| ≤ τ_c 인 동안은 **안 움직인다.** 그래서 부호가 바뀌는 두 점
    사이를 선형보간하면 그 구간의 중점이 나오고, 그게 마찰이 소거된 g* 다.
    구간 폭 자체도 유용하다 — (g⁺−g⁻)/2·G_CAD ≈ τ_c/α 로 마찰을 역산할 수 있다.

★쓰기 전에
    · 로봇은 **크레인에 매달려** 있어야 한다. 접지 중이면 제어기가 무중력을 거부한다.
    · `biped_deploy` 가 이미 떠 있어야 한다(이 스크립트는 명령파일로만 조종한다).
    · 손을 대지 말 것 — 표류만 봐야 한다.

사용:
    python3 tools/float_gstar.py                    # 0.85~1.25, 0.05 간격
    python3 tools/float_gstar.py --lo 0.9 --hi 1.3 --step 0.05
    python3 tools/float_gstar.py --dwell 4 --settle 6
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

CMD = "/tmp/biped_cmd.json"
STT = "/tmp/biped_state.json"
NJ = 8
NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot",
         "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]

_seq = [0]


def send(**kw):
    """명령 1건. ⚠seq 를 매번 올린다 — 워치독이 **내용 변화**로 생존을 판단한다."""
    _seq[0] += 1
    c = {"v": 0.0, "vy": 0.0, "w": 0.0, "body_h": 0.38,
         "jog_deg": [0.0] * NJ, "pos_kp_scale": 1.0, "seq": _seq[0]}
    c.update(kw)
    tmp = CMD + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f)
    os.replace(tmp, CMD)


def state():
    try:
        with open(STT) as f:
            return json.load(f)
    except Exception:
        return {}


def q():
    s = state()
    for k in ("q_leg_deg", "q_deg", "q_leg"):
        v = s.get(k)
        if isinstance(v, list) and len(v) >= NJ:
            return [float(x) for x in v[:NJ]]
    return None


def hold(mode, secs, hz=20, **kw):
    """모드를 유지하며 워치독을 먹인다. 중간에 상태를 계속 읽어 반환한다."""
    t0 = time.time()
    last = None
    while time.time() - t0 < secs:
        send(mode=mode, **kw)
        time.sleep(1.0 / hz)
        last = q() or last
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.85)
    ap.add_argument("--hi", type=float, default=1.25)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--dwell", type=float, default=3.0, help="각 배율에서 표류를 보는 시간[s]")
    ap.add_argument("--settle", type=float, default=5.0, help="매 점 전에 home 으로 되돌리는 시간[s]")
    ap.add_argument("--abort-deg", type=float, default=25.0,
                    help="한 축이 이만큼 표류하면 그 점을 중단하고 home 으로")
    a = ap.parse_args()

    if q() is None:
        print("✗ /tmp/biped_state.json 에서 q_leg_deg 를 못 읽는다.")
        print("  biped_deploy 가 떠 있는지 확인할 것.")
        return 1

    grid = []
    g = a.lo
    while g <= a.hi + 1e-9:
        grid.append(round(g, 3))
        g += a.step

    print("■ 무중력 중립점 g* 측정 — 전 축 동시")
    print(f"  배율 {grid[0]:.2f}~{grid[-1]:.2f} ({len(grid)}점) · 각 점 {a.dwell:.0f}s · "
          f"매번 home {a.settle:.0f}s 복귀")
    print("  ⚠크레인에 매달린 상태여야 한다. 손을 대지 말 것.\n")

    rows = []          # [(g, [drift × 8])]
    try:
        for gi, gv in enumerate(grid):
            hold("home", a.settle)
            q0 = q()
            if q0 is None:
                print("✗ 상태를 못 읽는다 — 중단"); return 1

            # float 로 dwell 동안. 중간에 과표류면 조기 종료.
            t0 = time.time()
            q1 = q0
            while time.time() - t0 < a.dwell:
                send(mode="float", grav_scale=gv)
                time.sleep(0.05)
                cur = q()
                if cur:
                    q1 = cur
                    if max(abs(b - c) for b, c in zip(q0, q1)) > a.abort_deg:
                        break
            d = [b - c for c, b in zip(q0, q1)]
            rows.append((gv, d))
            mx = max(range(NJ), key=lambda i: abs(d[i]))
            print(f"  ×{gv:.2f}  최대 {NAMES[mx]:9s}{d[mx]:+7.2f}°   "
                  + " ".join(f"{v:+6.2f}" for v in d), flush=True)
    finally:
        hold("home", 2.0)
        send(mode="hold")

    # ── 축별 영점교차 ────────────────────────────────────────────────────
    print("\n■ 결과 — 축별 중립점\n")
    print(f"  {'축':10s}{'g*':>8s}{'1/g*':>8s}{'데드밴드':>12s}   판정")
    out = {}
    for i, n in enumerate(NAMES):
        series = [(gv, d[i]) for gv, d in rows]
        gs = None; band = None
        for (g1, d1), (g2, d2) in zip(series, series[1:]):
            if d1 == 0.0:
                gs = g1; band = 0.0; break
            if d1 * d2 < 0:
                gs = g1 + (g2 - g1) * abs(d1) / (abs(d1) + abs(d2))
                band = g2 - g1
                break
        if gs is None:
            # 부호가 안 바뀜 = 구간 밖. 어느 쪽인지 알려 준다.
            sgn = series[0][1]
            side = "경계 밖(더 낮게)" if sgn < 0 else "경계 밖(더 높게)"
            print(f"  {n:10s}{'—':>8s}{'—':>8s}{'—':>12s}   {side}")
            out[n] = None
        else:
            print(f"  {n:10s}{gs:>8.3f}{1/gs:>8.3f}{band:>11.2f}   "
                  + ("부족" if gs > 1.02 else "과다" if gs < 0.98 else "맞음"))
            out[n] = round(gs, 3)

    print("\n■ 그대로 붙여 쓸 수 있는 축별 배율")
    print("  GRAV_SCALE_JOINT=\"" +
          ",".join(f"{out[n]:.3f}" if out[n] else "1.000" for n in NAMES) + "\"")

    ts = time.strftime("%Y%m%d-%H%M%S")
    path = f"/tmp/float_gstar_{ts}.json"
    with open(path, "w") as f:
        json.dump({"grid": grid, "rows": [[g, d] for g, d in rows], "gstar": out}, f, indent=1)
    print(f"\n  원자료 → {path}   (이 파일을 그대로 전달하면 된다)")
    return 0


sys.exit(main())
