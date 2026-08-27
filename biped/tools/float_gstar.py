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
_bias = [None]          # 축별 기준배율(2차 패스에서 심는다)


def send(**kw):
    """명령 1건. ⚠seq 를 매번 올린다 — 워치독이 **내용 변화**로 생존을 판단한다."""
    _seq[0] += 1
    c = {"v": 0.0, "vy": 0.0, "w": 0.0, "body_h": 0.38,
         "jog_deg": [0.0] * NJ, "pos_kp_scale": 1.0, "seq": _seq[0]}
    if _bias[0] is not None:
        c["grav_scale_joint"] = list(_bias[0])
    c.update(kw)
    # ★공통배율(grav_scale)은 축별배율이 설정된 배포기(run_deploy_hw.sh 의
    #   GRAV_SCALE_JOINT env)에서 **무시**된다 — dispatch 가 grav_axis[j]>=0 이면
    #   공통값을 안 본다. 공통 스윕이 조용히 죽지 않도록 축별 미지정 시
    #   공통값을 8축 배열로도 함께 발행한다. (08-27 검토에서 발견)
    if "grav_scale" in c and "grav_scale_joint" not in c:
        c["grav_scale_joint"] = [float(c["grav_scale"])] * NJ
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


def estopped():
    """★E-stop 래치 여부. 래치되면 **전 축 무여자**라 그 뒤 모든 점이 쓰레기다.

    왜 필요한가 (2026-08-24 실기에서 당함) — 스윕 첫 점에서 속도트립이 걸려 래치되면
    나머지 구간이 통째로 limp 다. 무여자면 중력보상이 0 이니 표류가 **배율과 무관하게
    일정**해진다. 실측 HL_thigh 가 0.60~2.00 전 구간 +0.34°/s 로 붙어 있었는데,
    그게 "마찰 밴드" 처럼 보여서 "g* > 2.0" 이라는 허구의 결론이 나왔다.
    ⇒ 이건 데이터가 없는 것이지 넓은 밴드가 아니다. **구분해서 중단해야 한다.**
    """
    try:
        st = json.load(open(STT))       # ★버그픽스(08-27): STATE(미정의) NameError 가
    except Exception:                    #   bare except 에 삼켜져 E-stop 감시가 사문화됐었다
        return None
    for k in ("estop", "estop_latched"):
        if st.get(k):
            return st.get("estop_reason") or k
    return None


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
    ap.add_argument("--fine-step", type=float, default=0.02,
                    help="2차 패스의 공통배수 간격")
    ap.add_argument("--fine-n", type=int, default=5,
                    help="2차 패스 점수(±n). 기본 ±5 = 0.90~1.10")
    # ★★브래킷 판독의 문턱 (2026-08-24). 이 값보다 느린 표류는 **안 움직인 것**으로 본다.
    #   왜 필요한가 — 영교차 보간은 **잡음에서도 교차를 만들어낸다.** 실측 HR_thigh 는
    #   1.00~1.15 구간이 +0.03/+0.03/+0.02/−0.01 인데, 이건 3초 동안 0.06~0.09° 다.
    #   엔코더 잡음·크리프와 구분이 안 된다. 그런데 보간은 거기서 g*=1.136 을 뽑았다.
    #   ⇒ 문턱을 넘는 점만 "실제로 움직였다" 로 치고, 그 사이는 **마찰 밴드**로 남긴다.
    # ★★축별 스윕 (2026-08-24). 왜 필요한가 — **전 축 동시 스윕은 가장 약한 축에 끌려간다.**
    #   실측: 배율 1.35 에서 HR_thigh 가 −11.72°/s 로 폭주해 관측창이 8s→2.1s 로 무너졌고,
    #   그 위 배율은 통째로 버려졌다. 그런데 HL_thigh 의 답은 **바로 그 위**에 있었다.
    #   ⇒ 다른 축은 자기 중립점(--hold)에 묶어 두고 **한 축만** 훑는다. 아무도 안 폭주한다.
    #   ⚠--hold 는 축별로 줘야 의미가 있다. 미지정 축은 1.0 이다.
    ap.add_argument("--axis", default=None,
                    help="이 축만 훑는다(예 HL_thigh). 나머지는 --hold 값에 고정")
    ap.add_argument("--hold", default=None,
                    help="--axis 사용 시 나머지 축의 배율 8개(콤마). 생략하면 전부 1.0")
    ap.add_argument("--v-move", type=float, default=0.10,
                    help="이 속도[°/s] 이상이라야 '움직였다'로 친다(브래킷 판독 문턱)")
    ap.add_argument("--min-amp", type=float, default=0.10,
                    help="배율 전 구간의 표류율 폭[deg/s] 문턱. 이보다 작으면 마찰 데드밴드로 본다")
    a = ap.parse_args()

    # ★인자 검증은 **로봇을 건드리기 전에** 한다. 종전엔 생존확인(home 5s) 뒤에야
    #   축 이름 오타가 드러났다 — 오타 하나에 다리를 한 번 움직이게 된다.
    if a.axis and a.axis not in NAMES:
        print(f"✗ 축 이름이 틀렸다: {a.axis}\n  가능: {', '.join(NAMES)}")
        return 1
    if a.hold:
        _t = [x for x in a.hold.split(",") if x.strip()]
        if len(_t) != NJ:
            print(f"✗ --hold 는 {NJ}개여야 한다(받은 것 {len(_t)}개)"); return 1
        # ★숫자 변환까지 여기서 해 본다 (2026-08-25 실기에서 당함) — 개수만 보고 통과시켰더니
        #   '<HL의 g*>' 같은 플레이스홀더가 **생존확인(home 47° 이동) 뒤에야** float() 에서
        #   터졌다. 로봇을 움직인 다음에 죽는 검증은 검증이 아니다.
        try:
            [float(x) for x in _t]
        except ValueError as e:
            print(f"✗ --hold 에 숫자가 아닌 값이 있다: {e}")
            print(f"   받은 것: {a.hold}")
            return 1
    if a.hold and not a.axis:
        print("✗ --hold 는 --axis 와 함께 써야 한다(단독으로는 효과가 없다)"); return 1

    if q() is None:
        print("✗ /tmp/biped_state.json 에서 q_leg_deg 를 못 읽는다.")
        print("  biped_deploy 가 떠 있는지 확인할 것.")
        return 1
    q_start = q()   # ★시작(수동 매달림) 자세 — 종료 때 여기로 서행 복귀하면 낙차 0

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

    def sweep(grid, tag, bias=None, mult=False):
        """한 번의 스윕. bias 가 있으면 축별로 심고 grid 는 **공통 배수**가 된다."""
        _bias[0] = bias
        out_rows = []
        print(f"\n■ {tag}")
        for gv in grid:
            if bias is not None:
                _bias[0] = [b * gv for b in bias]     # 축별 기준 × 공통배수
            hold("home", a.settle)
            q0 = q()
            if q0 is None:
                print("✗ 상태를 못 읽는다 — 중단"); return out_rows
            t0 = time.time(); q1 = q0
            while time.time() - t0 < a.dwell:
                send(mode="float", grav_scale=(1.0 if bias is not None else gv))
                time.sleep(0.05)
                cur = q()
                if cur:
                    q1 = cur
                    if max(abs(x - y) for x, y in zip(q0, q1)) > a.abort_deg:
                        break
            el = max(1e-3, time.time() - t0)
            d = [(y - x) / el for x, y in zip(q0, q1)]
            sat = [abs(y - x) >= a.abort_deg * 0.98 for x, y in zip(q0, q1)]
            es = estopped()
            if es:
                print(f"\n  ⛔ **E-stop 래치** ({es}) — 스윕 중단.")
                print("     래치되면 전 축 무여자다. 이 뒤 점들은 표류가 배율과 무관해져")
                print("     **넓은 마찰 밴드처럼 보인다** — 데이터가 아니다.")
                return out_rows
            out_rows.append((gv, d, sat))
            mx = max(range(NJ), key=lambda i: abs(d[i]))
            print(f"  ×{gv:.3f} [{el:.1f}s] 최대 {NAMES[mx]:9s}{d[mx]:+7.2f}°/s"
                  + ("★포화" if any(sat) else "    ") + " "
                  + " ".join((f"{v:+6.2f}" + ("*" if sat[i] else " ")) for i, v in enumerate(d)),
                  flush=True)
            # ⚠창이 무너지면 그 위는 전부 무의미하다 — 조기 종료한다
            if el < a.dwell * 0.35:
                print(f"  ⚠관측창이 {el:.1f}s 로 무너졌다({a.dwell:.0f}s 목표의 "
                      f"{el/a.dwell*100:.0f}%) — 이 위 배율은 **다른 축까지 오염**시킨다. 스윕 중단.")
                break
        return out_rows

    def crossings(rows, label=""):
        """축별 영점교차. 포화점 제외 + 전 구간 폭 문턱으로 잡음을 거른다."""
        res = {}
        for i, n in enumerate(NAMES):
            ser = [(gv, d[i]) for gv, d, st in rows if not st[i]]
            if len(ser) < 2:
                res[n] = (None, "유효점 부족(포화)"); continue
            vals = [v for _, v in ser]
            if max(vals) - min(vals) < a.min_amp:
                res[n] = (None, f"**측정 불가** — 배율 무반응(폭 {max(vals)-min(vals):.2f}°/s). 마찰 데드밴드")
                continue
            gs = None
            for (g1, d1), (g2, d2) in zip(ser, ser[1:]):
                if d1 * d2 < 0:
                    gs = g1 + (g2 - g1) * abs(d1) / (abs(d1) + abs(d2)); break
            if gs is None:
                last = ser[-1]
                res[n] = (None, f"경계 밖 (×{last[0]:.3f} 에서 {last[1]:+.2f}°/s)")
            else:
                res[n] = (gs, "부족" if gs > 1.02 else "과다" if gs < 0.98 else "맞음")
        return res

    def bracket(rows):
        """★마찰 브래킷 판독 — 영교차 보간 대신 **밴드의 두 끝**을 읽는다.

        물리: 낙하는 G − α·g·G_CAD > τ_c 일 때, 상승은 그 반대일 때만 일어난다.
            g_lo = (G−τ_c)/(α·G_CAD)   ← 마지막으로 **지고 있던** 배율
            g_hi = (G+τ_c)/(α·G_CAD)   ← 처음으로 **뜨는** 배율
            g*   = (g_lo+g_hi)/2       ← 마찰이 소거된다
            밴드 = g_hi − g_lo = 2τ_c/(α·G_CAD)   ← **τ_c/α 를 덤으로 준다**
        ⇒ 올라가는 한 번의 스윕에 두 경계가 **이미 들어 있다.** 양방향 스윕이 필요 없다.
          종전 판독(영교차 보간)은 이 밴드 정보를 버리고, 잡음에서도 교차를 만들어냈다.

        ★낙하방향은 **최저 배율의 부호**로 자동 판정한다 — 거기가 중력이 가장 우세한 점이라
          축별 부호규약을 손으로 적을 필요가 없다(적으면 반드시 어긋난다).
        """
        res = {}
        for i, n in enumerate(NAMES):
            # ★포화점을 **버리지 않는다** (2026-08-24 실기에서 당함). HR_thigh 가 ×1.50 에서
            #   −15.4°/s 로 명백히 떴는데, 포화 제외 때문에 g_hi 를 못 잡고 "상한 밖" 이 나왔다.
            #   포화는 **크기**가 잘린 것이지 **방향**은 유효하다 — 문턱 판정은 방향만 쓴다.
            #   (영교차 보간은 크기를 쓰므로 거기선 계속 제외한다.)
            ser = [(gv, d[i]) for gv, d, st in rows]
            if len(ser) < 3:
                res[n] = (None, None, None, None, "유효점 부족"); continue
            sgn = 1.0 if ser[0][1] >= 0 else -1.0          # 낙하방향
            g_lo = g_hi = None
            for gv, v in ser:
                if v * sgn >= a.v_move:
                    g_lo = gv                              # 계속 갱신 → 마지막 낙하점
            for gv, v in ser:
                if v * sgn <= -a.v_move:
                    g_hi = gv; break                       # 첫 상승점
            if g_lo is not None and g_hi is not None:
                res[n] = ((g_lo + g_hi) / 2, g_hi - g_lo, g_lo, g_hi, "브래킷")
            elif g_lo is not None:
                res[n] = (None, None, g_lo, None, f"g* > {g_lo:.3f} — **상한 밖**(--hi 를 올릴 것)")
            elif g_hi is not None:
                res[n] = (None, None, None, g_hi, f"g* < {g_hi:.3f} — **하한 밖**(--lo 를 내릴 것)")
            else:
                res[n] = (None, None, None, None,
                          f"전 구간 정지(<{a.v_move:.2f}°/s) — 마찰 밴드가 스윕보다 넓다")
        return res

    grid1 = grid
    rows1, rows2, res2 = [], [], None

    # ── ★축별 스윕 경로 — 여기서 끝낸다(2차 패스 없음) ────────────────────
    if a.axis:
        ax = NAMES.index(a.axis)                      # 이름 검증은 위에서 이미 했다
        hold_v = [float(x) for x in a.hold.split(",")] if a.hold else [1.0] * NJ
        print(f"■ ★축별 스윕 — **{a.axis}** 만 {grid[0]:.2f}~{grid[-1]:.2f} 로 훑는다")
        print("  나머지 축 고정배율: " + " ".join(f"{n.split('_')[1][:2]}{v:.2f}"
                                             for n, v in zip(NAMES, hold_v)))
        print("  ⇒ 다른 축이 자기 중립점에 있으면 폭주하지 않는다 = 관측창이 안 무너진다\n")
        rows = []
        try:
            for gv in grid:
                _bias[0] = list(hold_v); _bias[0][ax] = gv
                hold("home", a.settle)
                q0 = q()
                if q0 is None:
                    print("✗ 상태를 못 읽는다 — 중단"); break
                t0 = time.time(); q1 = q0
                while time.time() - t0 < a.dwell:
                    send(mode="float", grav_scale=1.0)
                    time.sleep(0.05)
                    cur = q()
                    if cur:
                        q1 = cur
                        if abs(q1[ax] - q0[ax]) > a.abort_deg:
                            break
                el = max(1e-3, time.time() - t0)
                d = [(y - x) / el for x, y in zip(q0, q1)]
                sat = [abs(y - x) >= a.abort_deg * 0.98 for x, y in zip(q0, q1)]
                es = estopped()
                if es:
                    print(f"\n  ⛔ **E-stop 래치** ({es}) — 여기서 중단한다.")
                    print("     래치되면 전 축이 무여자라 이 뒤 모든 점이 쓰레기다.")
                    print("     표류가 배율과 무관하게 일정해져서 **넓은 마찰 밴드처럼 보인다.**")
                    print("     → 제어기 로그에서 트립 축·값을 확인하고, 스윕 상한을 낮춰 다시 할 것.")
                    break
                rows.append((gv, d, sat))
                print(f"  ×{gv:.3f} [{el:.1f}s] {a.axis:9s}{d[ax]:+7.2f}°/s"
                      + ("★포화" if sat[ax] else "     ")
                      + "  (그 외 최대 " + f"{max(abs(v) for i, v in enumerate(d) if i != ax):.2f})",
                      flush=True)
        except KeyboardInterrupt:
            print("\n  (중단됨 — 지금까지 모은 점으로 계산한다)")
        finally:
            _bias[0] = None
            try: hold("home", 1.5); send(mode="hold")
            except Exception: pass
        br = bracket(rows)
        gs, bd, lo, hi, why = br[a.axis]
        print(f"\n■ 결과 — {a.axis}\n")
        f = lambda v: f"{v:.3f}" if v is not None else "—"
        print(f"  g_lo {f(lo)} · g_hi {f(hi)}")
        if gs:
            print(f"  ★g* = **{gs:.3f}**   1/g* = **{1/gs:.3f}**   밴드 {bd:.3f}")
            print(f"    밴드×G_CAD/2 = τ_c/α  (G_CAD 는 MJCF 에서 뽑을 것)")
        else:
            print(f"  {why}")
        # ★원자료 저장 (2026-08-27) — 축별 모드가 저장 없이 반환해 세 스윕의 원자료가
        #   화면에만 남았던 것을 수정. rows 의 d[8] 로 유지축 표류(드룹/커플링)까지 검증 가능.
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = f"/tmp/float_gstar_{a.axis}_{ts}.json"
        with open(path, "w") as fjs:
            json.dump({"axis": a.axis, "names": NAMES, "hold": a.hold,
                       "rows": [[g, list(d), list(st)] for g, d, st in rows],
                       "g_lo": lo, "g_hi": hi, "gstar": gs, "band": (bd if gs else None)}, fjs, indent=1)
        print(f"\n  원자료 → {path}   (전 8축 표류 포함 — 이 파일을 그대로 전달하면 된다)")
        return 0

    try:
        rows1 = sweep(grid1, f"1차 — 공통 배율 {grid1[0]:.2f}~{grid1[-1]:.2f} (축별 대략값을 잡는다)")
        res1 = crossings(rows1)

        # ★★2차 — 1차 값을 **축별로 심고** 공통 배수만 훑는다.
        #   그러면 모든 축이 자기 중립점 근처에 있어 **어느 축도 폭주하지 않는다** ⇒ 관측창이 안 무너진다.
        #   1차에서 못 잡은 축은 마지막 배율(또는 1.0)로 심어 둔다 — 최소한 폭주는 막는다.
        bias = []
        for i, n in enumerate(NAMES):
            g0 = res1[n][0]
            if g0 is None:
                ser = [(gv, d[i]) for gv, d, st in rows1 if not st[i]]
                g0 = ser[-1][0] if ser else 1.0
            bias.append(round(g0, 3))
        print("\n■ 1차 결과를 축별로 심는다")
        print("  " + " ".join(f"{n.split('_')[1][:2]}{b:.2f}" for n, b in zip(NAMES, bias)))
        fine = [round(1.0 + k * a.fine_step, 4)
                for k in range(-a.fine_n, a.fine_n + 1)]
        rows2 = sweep(fine, f"2차 — 축별 기준 × 공통배수 {fine[0]:.3f}~{fine[-1]:.3f}", bias=bias)
        res2 = crossings(rows2)
    except KeyboardInterrupt:
        print("\n  (중단됨 — 지금까지 모은 점으로 계산한다)")
    finally:
        _bias[0] = None
        try:
            # ★안전 종료 v3 (2026-08-27): 배율/게인 램프다운은 두 번 다 실패했다 —
            #   중력토크는 상수인데 붙잡는 힘만 줄이면 말미에 반드시 "버티다 놓침"이 된다.
            #   올바른 종료 = **낙차가 0 인 자세로 이동한 뒤 끄는 것**: 도구 시작 시의
            #   수동 매달림 자세(q_start)로 jog(20dps 위치제어) 서행 복귀 → 수동 평형이므로
            #   off 로 바꿔도 움직일 것이 없다. (hold() 20Hz 가 워치독을 먹인다)
            if q_start:
                print("\n  ■ 안전 종료 — 시작 매달림 자세로 서행 복귀(jog) 후 무여자.")
                t0 = time.time()
                while time.time() - t0 < 25.0:
                    cur = hold("jog", 0.5, jog_deg=list(q_start))
                    if cur and max(abs(x - y) for x, y in zip(cur, q_start)) < 1.5:
                        break
                hold("jog", 1.5, jog_deg=list(q_start))     # 정착
            send(mode="off")
            print("  ✅ 안전 종료 완료(무여자·수동 평형 자세) — 배포기·Emb 를 꺼도 된다.")
        except Exception as e:
            print(f"  ⚠종료 정리 실패({type(e).__name__}) — GUI [무중력]에서 배율을 서서히 0 으로 내릴 것")

    # ── ★브래킷 판독 (2026-08-24) — 이쪽이 물리적으로 옳다 ────────────────
    #   ⚠아래 '영교차' 표와 **나란히** 낸다. 종전 표를 지우지 않는 이유:
    #     지금까지 쌓인 기록이 그 판독 기준이라, 갑자기 바꾸면 과거와 비교가 끊긴다.
    try:
        br = bracket(rows1)
        print("\n■ ★브래킷 판독 — 마찰 밴드의 두 끝 (1차 스윕)\n")
        print(f"  문턱 |v| ≥ {a.v_move:.2f}°/s 이상이라야 '움직였다'로 친다\n")
        print(f"  {'축':10s}{'g_lo':>8s}{'g_hi':>8s}{'g*':>9s}{'1/g*':>8s}{'밴드':>8s}   근거")
        for n in NAMES:
            gs, bd, lo, hi, why = br[n]
            f = lambda v: f"{v:.3f}" if v is not None else "—"
            print(f"  {n:10s}{f(lo):>8s}{f(hi):>8s}"
                  f"{(f'{gs:.3f}' if gs else '—'):>9s}"
                  f"{(f'{1/gs:.3f}' if gs else '—'):>8s}"
                  f"{(f'{bd:.3f}' if bd else '—'):>8s}   {why}")
        print("\n  밴드 = 2·τ_c/(α·G_CAD) — 여기에 G_CAD/2 를 곱하면 **τ_c/α** 가 나온다.")
        print("  g* 가 '—' 인데 g_lo 만 있으면 스윕 상한을 올릴 것(--hi).")
    except Exception as e:
        print(f"  ⚠브래킷 판독 실패({type(e).__name__}: {e})")

    # ── 최종 ────────────────────────────────────────────────────────────
    print("\n■ 결과 — 축별 중립점(영교차 보간 · 종전 기준)\n")
    print(f"  {'축':10s}{'g*':>9s}{'1/g*':>8s}   판정 / 근거")
    out = {}
    for i, n in enumerate(NAMES):
        gs, why = (res2 or res1)[n]
        if gs is not None and res2:
            gs = bias[i] * gs                       # 2차는 **공통배수**라 기준을 곱해 되돌린다
        if gs is None:
            gs, why = res1[n]
            why = (why + " (1차)") if gs is None else (why + " ← 1차만")
        if gs is None:
            print(f"  {n:10s}{'—':>9s}{'—':>8s}   {why}")
            out[n] = None
        else:
            print(f"  {n:10s}{gs:>9.3f}{1/gs:>8.3f}   {why}")
            out[n] = round(gs, 3)

    print("\n■ 그대로 붙여 쓸 수 있는 축별 배율")
    print("  GRAV_SCALE_JOINT=\"" +
          ",".join(f"{out[n]:.3f}" if out[n] else "1.000" for n in NAMES) + "\"")

    ts = time.strftime("%Y%m%d-%H%M%S")
    path = f"/tmp/float_gstar_{ts}.json"
    with open(path, "w") as f:
        json.dump({"names": NAMES, "grid1": grid1,
                   "rows1": [[g, d, st] for g, d, st in rows1],
                   "rows2": [[g, d, st] for g, d, st in rows2],
                   "bias": bias if rows2 else None, "gstar": out}, f, indent=1)
    print(f"\n  원자료 → {path}   (이 파일을 그대로 전달하면 된다)")
    return 0


sys.exit(main())
