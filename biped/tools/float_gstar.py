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
import argparse, glob, json, math, os, subprocess, sys, time

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
    # ★임시파일 이름에 **PID** 를 넣는다. GUI(teleop_gui_biped.py:128)도 `CMD + ".tmp"` 를
    #   쓰기 때문에, 같은 이름이면 GUI 의 os.replace 가 우리 tmp 를 먼저 가져가고
    #   우리 os.replace 는 FileNotFoundError 로 죽는다(2026-08-24 실기에서 실제로 그랬다).
    tmp = "%s.%d.tmp" % (CMD, os.getpid())
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
    ap.add_argument("--min-amp", type=float, default=0.10,
                    help="배율 전 구간의 표류율 폭[deg/s] 문턱. 이보다 작으면 마찰 데드밴드로 본다")
    a = ap.parse_args()

    if q() is None:
        print("✗ /tmp/biped_state.json 에서 q_leg_deg 를 못 읽는다.")
        print("  biped_deploy 가 떠 있는지 확인할 것.")
        return 1

    # ★★**경쟁 발행자 검사** — GUI 가 떠 있으면 20ms 마다 자기 모드로 덮어쓴다.
    #   그러면 float 와 hold 가 번갈아 들어가 측정이 통째로 무의미해진다(그리고 로봇이 떤다).
    try:
        r = subprocess.run(["pgrep", "-af", "teleop_gui_biped"],
                           capture_output=True, text=True, timeout=3)
        others = [l for l in r.stdout.splitlines() if "pgrep" not in l]
    except Exception:
        others = []
    if others:
        print("✗ **teleop GUI 가 떠 있다.** GUI 도 20ms 마다 같은 명령파일을 쓰므로")
        print("  이 스크립트와 모드를 번갈아 덮어쓴다 — 측정이 성립하지 않는다.")
        for l in others:
            print("    " + l[:100])
        print("  → GUI 를 닫고 다시 실행할 것. (뷰어·모니터는 무관하다)")
        return 1

    # ★E-stop 래치를 먼저 푼다. 래치 중이면 어떤 모드도 안 먹고 로봇이 limp 라
    #   표류가 전부 0 으로 나와 "중립점을 찾았다" 처럼 보인다 — 최악의 오독이다.
    send(mode="off"); time.sleep(0.4)
    hold("off", 0.6)

    grid = []
    g = a.lo
    while g <= a.hi + 1e-9:
        grid.append(round(g, 3))
        g += a.step

    print("■ 무중력 중립점 g* 측정 — 전 축 동시")
    print(f"  배율 {grid[0]:.2f}~{grid[-1]:.2f} ({len(grid)}점) · 각 점 {a.dwell:.0f}s · "
          f"매번 home {a.settle:.0f}s 복귀")
    print("  ⚠크레인에 매달린 상태여야 한다. 손을 대지 말 것.\n")

    # ★생존 확인 — home 을 한 번 돌려 로봇이 **실제로 움직이는지** 본다.
    #   안 움직이면 E-stop 래치이거나 통신 두절이다. 그 상태로 스윕하면 전 축 표류 0 이
    #   나오고, 그건 "완벽한 중립" 이 아니라 **아무 데이터도 없는 것**이다.
    qa = q(); hold("home", max(3.0, a.settle)); qb = q()
    moved = max(abs(x - y) for x, y in zip(qa, qb)) if (qa and qb) else 0.0
    print(f"  생존 확인 — home 으로 최대 {moved:.2f}° 이동")
    if moved < 0.5:
        print("✗ **로봇이 안 움직인다.** E-stop 래치이거나 통신이 끊긴 상태다.")
        print("  이대로 재면 전 축 표류가 0 으로 나와 '완벽한 중립' 처럼 보인다 — 측정 아님.")
        print("  → 제어기 로그에서 E-STOP/동결을 확인하고 복구한 뒤 다시 실행할 것.")
        return 1
    print()

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
            el = max(1e-3, time.time() - t0)
            # ★★**표류율**(deg/s)로 기록한다 — 표류량이 아니라.
            #   한 축이 abort 에 걸리면 그 점이 **조기 종료**돼 나머지 축의 관측창이 짧아진다.
            #   그러면 표류량이 작게 나오고, 그게 "배율이 맞아간다" 로 오독된다.
            #   실기 2차(2026-08-24)에서 실제로 그랬다: HR_hip 이 ×1.30 부터 포화해
            #   HL_thigh 가 1.16 → 0.76 으로 줄었는데 **창이 짧아진 것**과 구분이 안 됐다.
            #   ⇒ 경과시간으로 나누면 점끼리 비교가 성립한다.
            d = [(b - c) / el for c, b in zip(q0, q1)]
            sat = [abs(b - c) >= a.abort_deg * 0.98 for c, b in zip(q0, q1)]
            rows.append((gv, d, sat))
            mx = max(range(NJ), key=lambda i: abs(d[i]))
            print(f"  ×{gv:.2f} [{el:.1f}s] 최대 {NAMES[mx]:9s}{d[mx]:+7.2f}°/s"
                  + ("★포화" if any(sat) else "    ") + " "
                  + " ".join((f"{v:+6.2f}" + ("*" if sat[i] else " ")) for i, v in enumerate(d)),
                  flush=True)
    except KeyboardInterrupt:
        print("\n  (중단됨 — 지금까지 모은 점으로 계산한다)")
    finally:
        try:
            hold("home", 1.5); send(mode="hold")
        except Exception as e:
            print(f"  ⚠종료 정리 실패({type(e).__name__}) — GUI 로 직접 hold 를 눌러 둘 것")

    # ── 축별 영점교차 ────────────────────────────────────────────────────
    print("\n■ 결과 — 축별 중립점\n")
    print(f"  {'축':10s}{'g*':>8s}{'1/g*':>8s}{'데드밴드':>12s}   판정")
    #   ★교차로 인정하려면 **의미 있는 진폭**이 있어야 한다. 안 그러면 −0.02 → +0.00 같은
    #     양자화 잡음이 "완벽한 중립" 으로 둔갑한다(2026-08-24 실기에서 HL_calf/foot 이 그랬다).
    #   ★그리고 배율을 바꿔도 표류가 **안 변하는** 축은 마찰 데드밴드에 묻힌 것이다.
    #     calf(중력/마찰 0.47)·foot(0.30) 이 그렇다 — 이 방법으로는 원리적으로 못 잰다.
    MINAMP = a.min_amp
    out = {}
    for i, n in enumerate(NAMES):
        series = [(gv, d[i]) for gv, d, _ in rows if not _[i]]     # 포화점 제외
        if len(series) < 2:
            print(f"  {n:10s}{'—':>8s}{'—':>8s}{'—':>12s}   유효점 부족(포화)")
            out[n] = None; continue
        vals = [v for _, v in series]
        span = max(vals) - min(vals)
        if span < MINAMP:
            print(f"  {n:10s}{'—':>8s}{'—':>8s}{'—':>12s}   "
                  f"**측정 불가** — 배율에 반응 안 함(폭 {span:.2f}°/s < {MINAMP:.2f}). 마찰 데드밴드")
            out[n] = None; continue
        #   ⚠점별 진폭 조건은 뺐다 — 실기 2차에서 HR_hip 이 +0.16 → −0.30 으로 교차했는데
        #     `max(|d1|,|d2|) >= 0.30` 이 부동소수 경계에서 걸려 **교차를 놓쳤다.**
        #     잡음 방어는 위의 span 검사가 이미 한다(HL_calf 폭 0.06° 는 거기서 걸린다).
        gs = None; band = None
        for (g1, d1), (g2, d2) in zip(series, series[1:]):
            if d1 * d2 < 0:
                gs = g1 + (g2 - g1) * abs(d1) / (abs(d1) + abs(d2))
                band = g2 - g1
                break
        if gs is None:
            last = series[-1]
            side = ("경계 밖 — **더 높게**" if last[1] * (1 if vals[0] > 0 else -1) > 0 else
                    "경계 밖 — **더 낮게**")
            print(f"  {n:10s}{'—':>8s}{'—':>8s}{'—':>12s}   {side} "
                  f"(×{last[0]:.2f} 에서 {last[1]:+.2f}°/s)")
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
        json.dump({"grid": grid, "names": NAMES,
                   "rows": [[g, d, s] for g, d, s in rows], "gstar": out}, f, indent=1)
    print(f"\n  원자료 → {path}   (이 파일을 그대로 전달하면 된다)")
    return 0


sys.exit(main())
