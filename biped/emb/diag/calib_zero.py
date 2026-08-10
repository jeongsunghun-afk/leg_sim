#!/usr/bin/env python3
"""calib_zero.py — 영점(offset_deg) 산정기.

무엇을 하는가:
  로봇을 **기준자세**에 놓은 상태에서 드라이버 보고각(채널각)을 읽어, 그 자세가
  의도한 모델각으로 읽히도록 offset_deg 를 계산한다.

식 (interface/joint_map.py 와 동일):
      모델각 = (채널각 − offset) / sign
  기준자세에서 모델각이 φ_ref 로 읽히길 원하므로
      φ_ref = (ch_ref − offset)/sign   ⇒   ★ offset = ch_ref − φ_ref · sign

  ⇒ φ_ref 가 전부 0 이면 offset 은 그냥 **그때의 채널각 그대로**다.
    0 이 아닌 축(예: calf −55°)만 sign 을 곱해 빼면 된다.

사용법:
  1) 제어기를 띄운다:            cd emb && python3 app/biped_emb.py
  2) 모드를 **off(limp)** 로 두고 로봇을 기준자세로 물리적으로 맞춘다(지그/수평계).
  3) python3 diag/calib_zero.py              # 계산만 (config 안 건드림)
     python3 diag/calib_zero.py --apply      # config/biped_emb.yaml 의 offset_deg 갱신
     python3 diag/calib_zero.py --ref 0,0,-55,0,0,0,-55,0    # 기준자세 직접 지정

기준자세 기본값은 config 의 `calib.ref_joint_deg`.

⚠ 이 스크립트는 모터에 아무것도 쓰지 않는다. --apply 는 config 파일만 고친다.
⚠ calf·foot 은 드라이버 감속비 오설정(7:1 가정)으로 보고각이 실제의 1.5/1.2 배다.
  offset 은 **기준자세 그 한 점에서만** 정확해지고, 거기서 멀어질수록 그 배율만큼
  틀어진다. 근본 해결은 드라이버 설정이다 — 아래 경고 참조.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time

EMB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(EMB, "interface"))
CFG_PATH = os.path.join(EMB, "config", "biped_emb.yaml")
STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")

# 드라이버가 전 축을 7:1 로 가정하고 있다는 가설에서 나오는 보고 배율(미해결 이슈)
GEAR_TRUE = {"hip": 7.0, "thigh": 7.0, "calf": 10.5, "foot": 8.4}
GEAR_ASSUMED = 7.0


def axis_kind(name: str) -> str:
    for k in GEAR_TRUE:
        if name.endswith(k):
            return k
    return "hip"


def load_cfg():
    import yaml
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def read_state():
    try:
        with open(STATE) as f:
            st = json.load(f)
        st["_age"] = time.time() - os.path.getmtime(STATE)
        return st
    except Exception as e:
        sys.exit(f"✗ 상태파일을 못 읽는다: {STATE}  ({e})\n"
                 f"  제어기를 먼저 띄울 것:  cd {EMB} && python3 app/biped_emb.py")


def channel_angles(st, joints):
    """기준자세에서의 **채널각**을 얻는다. 없으면 모델각에서 역산."""
    if "q_ch_deg" in st:
        return [float(x) for x in st["q_ch_deg"]], "state.q_ch_deg (raw)"
    # 폴백: 모델각 → 채널각 역산. 현재 config 의 sign/offset 이 그 값을 만든 것과
    # 같아야 정확하다(제어기 기동 후 config 를 안 고쳤다면 참).
    q = st.get("q_leg_deg")
    if not q:
        sys.exit("✗ state 에 q_ch_deg 도 q_leg_deg 도 없다.")
    ch = [float(q[i]) * float(j["sign"]) + float(j["offset_deg"])
          for i, j in enumerate(joints)]
    return ch, "q_leg_deg 에서 역산 (⚠제어기 재시작 필요 — q_ch_deg 필드가 없는 구버전)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="기준자세 모델각[deg], 콤마구분 8개. 생략시 config calib.ref_joint_deg")
    ap.add_argument("--apply", action="store_true", help="config/biped_emb.yaml 의 offset_deg 를 실제로 갱신")
    ap.add_argument("--settle-s", type=float, default=8.0,
                    help="이 시간 동안 자세가 멈춰 있어야 채취한다[s]. 0 이면 검사 생략")
    ap.add_argument("--settle-tol", type=float, default=0.2,
                    help="정지 판정 허용 변동폭[deg]")
    a = ap.parse_args()

    cfg = load_cfg()
    joints = cfg["joints"]
    n = len(joints)
    names = [j["name"] for j in joints]
    sign = [float(j["sign"]) for j in joints]
    old = [float(j["offset_deg"]) for j in joints]

    if a.ref:
        ref = [float(x) for x in a.ref.split(",")]
    else:
        ref = cfg.get("calib", {}).get("ref_joint_deg")
        if ref is None:
            sys.exit("✗ 기준자세가 없다. --ref 로 주거나 config 에 calib.ref_joint_deg 를 넣을 것.")
        ref = [float(x) for x in ref]
    if len(ref) != n:
        sys.exit(f"✗ 기준자세 길이 {len(ref)} ≠ 관절수 {n}")

    # ── ★정지 게이트 (2026-08-10 추가) ──────────────────────────────────────
    #   왜 필요한가 — 실제로 당한 실수다: 로봇이 공중에 매달린 채 limp 이면 hip·thigh·foot
    #   은 **자유롭게 흔들리는 진자**다. 짧은 창(4.8초)을 우연히 잔잔한 순간에 잡으면
    #   변동폭 0.02° 로 "안정" 처럼 보이고, 그 순간의 임의 자세로 영점이 박힌다.
    #   그렇게 잡은 offset 은 재시작 직후 이미 5~13° 어긋나 있었다.
    #   반대로 calf 는 구조적 한계에 닿아 있어 0.01° 로 반복된다 — 그 차이를 여기서 가른다.
    #   ⇒ 기계적으로 구속되지 않은 축은 **지그로 잡거나 사람이 붙들고** 채취해야 한다.
    if a.settle_s > 0:
        print(f"  정지 확인 중… {a.settle_s:.0f}초 (허용 {a.settle_tol:.2f}°)", flush=True)
        buf, t_end = [], time.time() + a.settle_s
        while time.time() < t_end:
            buf.append(read_state().get("q_leg_deg") or [])
            time.sleep(0.2)
        buf = [b for b in buf if len(b) == n]
        if len(buf) < 5:
            sys.exit("✗ 상태 샘플이 부족하다 — 제어기가 살아있는지 확인할 것.")
        span = [max(s[i] for s in buf) - min(s[i] for s in buf) for i in range(n)]
        moving = [(names[i], span[i]) for i in range(n) if span[i] > a.settle_tol]
        if moving:
            print(f"\n  ❌ 자세가 멈춰 있지 않다 — 채취를 중단한다.\n")
            for nm_, sp in moving:
                print(f"     {nm_:10} {a.settle_s:.0f}초간 {sp:6.2f}° 움직임")
            print(f"\n     고정된 축: {[names[i] for i in range(n) if span[i] <= a.settle_tol] or '없음'}")
            print(f"\n  ▸ 이 축들은 기계적으로 구속돼 있지 않다(매달린 상태 + limp = 자유 진자).")
            print(f"    지그로 고정하거나 사람이 기준자세로 붙든 채 다시 실행할 것.")
            print(f"    구속 없이 잡은 영점은 다음 순간 이미 틀어진다 — 그게 이 게이트를 만든 이유다.")
            print(f"    (정말 이대로 강행하려면 --settle-s 0)")
            return 1
        print(f"  ✅ 전 축 정지 (최대 변동 {max(span):.3f}°)\n")

    st = read_state()
    ch, src = channel_angles(st, joints)

    print("=" * 78)
    print("  영점 캘리브레이션 — offset = 채널각(기준자세) − 모델각(기준자세) · sign")
    print("=" * 78)
    print(f"  채널각 출처 : {src}")
    print(f"  상태 나이   : {st['_age']:.2f}s      모드: {st.get('mode','?')}")
    if st.get("mode") not in ("off", None):
        print(f"  ⚠ 모드가 off 가 아니다({st.get('mode')}). 모터가 구동 중이면 기준자세가")
        print(f"     '사람이 맞춘 자세'가 아니라 '제어기가 끌고 간 자세'다. off 로 두고 다시 할 것.")
    if st["_age"] > 1.0:
        print(f"  ⚠ 상태가 {st['_age']:.1f}s 나 묵었다 — 제어기가 살아있는지 확인할 것.")
    print()

    new = [ch[i] - ref[i] * sign[i] for i in range(n)]

    print(f"  {'축':10} {'sign':>5} {'채널각':>9} {'기준모델각':>11} "
          f"{'현 offset':>10} {'→ 새 offset':>12} {'변화':>8}")
    print("  " + "-" * 74)
    for i in range(n):
        print(f"  {names[i]:10} {sign[i]:+5.0f} {ch[i]:+9.2f} {ref[i]:+11.2f} "
              f"{old[i]:+10.2f} {new[i]:+12.2f} {new[i]-old[i]:+8.2f}")
    print()

    # ── 검증: 새 offset 으로 기준자세가 정말 ref 로 읽히는가 (실제 JointMap 으로) ──
    import copy
    import numpy as np
    from joint_map import JointMap
    c2 = copy.deepcopy(cfg)
    for i, j in enumerate(c2["joints"]):
        j["offset_deg"] = float(new[i])
    jm = JointMap(c2)
    q_ch_full = np.zeros(jm.n_channel)
    for i, c in enumerate(jm.ch):
        q_ch_full[c] = ch[i]
    back = jm.ch_to_q_joint(q_ch_full)
    err = float(np.max(np.abs(back - np.array(ref))))
    print(f"  ▸ 왕복검증: 새 offset 적용 시 기준자세가 {list(np.round(back,3))} 로 읽힌다")
    print(f"    최대오차 {err:.6f}°  →  {'✅ 통과' if err < 1e-6 else '❌ 실패'}")
    if err >= 1e-6:
        print("    ✗ 식이 안 맞는다. 적용하지 말 것."); return 1
    print()

    # ── 재현성 검사 ────────────────────────────────────────────────────────
    #   ★정지 게이트만으로는 부족하다 — **다 흔들리고 나서 멈춘 진자도 "정지" 로 통과한다.**
    #     그 정지점은 중력 평형점이지 기준자세가 아니다. 구분하는 유일한 방법은
    #     "지난번에 잡은 영점과 같은 값이 나오는가" 다. 안 나오면 그 자세는 재현되지 않는 것.
    redo = [(names[i], old[i], new[i]) for i in range(n)
            if abs(old[i]) > 1e-9 and abs(new[i] - old[i]) > 1.0]
    if redo:
        print("  ── ❌ 재현성 실패 — 이 축들은 지난 영점과 다른 값이 나온다 ──")
        for nm_, o, nw in redo:
            print(f"     {nm_:10} 지난 {o:+8.2f}  →  이번 {nw:+8.2f}   ({nw-o:+.2f}° 차)")
        print("     이 자세는 재현되지 않는다 = 그 축이 기계적으로 구속돼 있지 않다는 뜻이다.")
        print("     (매달린 상태 + limp 이면 hip·thigh·foot 은 자유 진자다. calf 는 구조적")
        print("      한계에 닿아 있어 재현된다 — 그 차이가 그대로 여기 드러난다.)")
        print("     ⇒ 지그/사람이 기준자세로 붙든 채 다시 잴 것. 그 전에는 --apply 하지 말 것.\n")

    # ── 안전 경고 ──────────────────────────────────────────────────────────
    warn = []
    for i in range(n):
        lo, hi = float(joints[i]["min_deg"]), float(joints[i]["max_deg"])
        jlo, jhi = jm.jog_min[i], jm.jog_max[i]
        if not (lo - 1e-9 <= ref[i] <= hi + 1e-9):
            warn.append(f"  ❌ {names[i]}: 기준자세 {ref[i]:+.1f}° 가 **관절한계 [{lo:+.0f},{hi:+.0f}] 밖**이다. "
                        f"한계나 기준자세 중 하나가 틀렸다.")
        elif not (jlo - 1e-9 <= ref[i] <= jhi + 1e-9):
            d = ref[i] - (jlo if ref[i] < jlo else jhi)
            warn.append(f"  ⚠ {names[i]}: 기준자세 {ref[i]:+.1f}° 가 jog 한계 [{jlo:+.1f},{jhi:+.1f}] 밖 "
                        f"({abs(d):.1f}° 초과). JOG 진입 시 한계까지 {abs(d):.1f}° 이동한다"
                        f"(램프 {cfg['jog']['max_speed_dps']:.0f}dps → 약 {abs(d)/cfg['jog']['max_speed_dps']:.1f}s).")
        k = axis_kind(names[i])
        g = GEAR_TRUE[k]
        if abs(g - GEAR_ASSUMED) > 1e-9 and abs(ref[i]) > 1e-9:
            r = g / GEAR_ASSUMED
            warn.append(f"  ⚠ {names[i]}: 감속비 오설정 축(보고 {r:.2f}배)인데 기준자세가 0 이 아니다"
                        f"({ref[i]:+.0f}°). 이 offset 은 **그 자세에서만** 맞고, 모델 0° 부근에선 "
                        f"약 {abs(ref[i])*(r-1):.1f}° 틀어진다. 드라이버 감속비 수정 후 반드시 재측정.")
    home = cfg.get("home", {}).get("q_deg")
    if home:
        for i in range(min(n, len(home))):
            d = abs(float(home[i]) - ref[i])
            if d > 20.0:
                warn.append(f"  ⚠ {names[i]}: HOME 목표 {float(home[i]):+.0f}° 와 기준자세 {ref[i]:+.0f}° 가 "
                            f"{d:.0f}° 차이다 — 영점 적용 후 HOME 을 누르면 그만큼 움직인다.")
    if warn:
        print("  ── 경고 ──")
        for w in warn:
            print(w)
        print()

    # ── YAML 조각 출력 ─────────────────────────────────────────────────────
    print("  ── config/biped_emb.yaml 에 넣을 값 ──")
    for i in range(n):
        print(f"     {names[i]:10} offset_deg: {new[i]:.2f}")
    print()

    if not a.apply:
        print("  (계산만 함. 실제로 반영하려면 --apply)")
        return 0

    # ── in-place 적용: 주석 보존 위해 라인단위 정규식 치환 ──────────────────
    with open(CFG_PATH) as f:
        lines = f.readlines()
    done = {}
    for li, line in enumerate(lines):
        m = re.search(r"name:\s*([A-Za-z0-9_]+)\b", line)
        if not m or m.group(1) not in names or "offset_deg" not in line:
            continue
        i = names.index(m.group(1))
        new_line, cnt = re.subn(r"(offset_deg:\s*)[-+0-9.eE]+", rf"\g<1>{new[i]:.2f}", line, count=1)
        if cnt:
            lines[li] = new_line
            done[m.group(1)] = True
    missing = [nm for nm in names if nm not in done]
    if missing:
        print(f"  ✗ 이 축들의 offset_deg 라인을 못 찾았다: {missing} — 적용 취소")
        return 1
    bak = CFG_PATH + ".bak"
    os.replace(CFG_PATH, bak)
    with open(CFG_PATH, "w") as f:
        f.writelines(lines)
    # 되읽어 검증
    try:
        c3 = load_cfg()
        got = [float(j["offset_deg"]) for j in c3["joints"]]
        assert max(abs(got[i] - round(new[i], 2)) for i in range(n)) < 1e-9
    except Exception as e:
        os.replace(bak, CFG_PATH)
        print(f"  ✗ 적용 후 검증 실패 → 원복했다: {e}")
        return 1
    print(f"  ✅ 적용 완료. 백업: {bak}")
    print(f"  ⚠ 제어기와 뷰어를 **모두 재시작**해야 반영된다(둘 다 기동 시 config 를 읽는다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
