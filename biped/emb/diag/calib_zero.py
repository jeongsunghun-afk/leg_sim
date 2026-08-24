#!/usr/bin/env python3
"""calib_zero.py — 영점(offset_deg) 산정기.

무엇을 하는가:
  로봇을 **기준자세**에 놓은 상태에서 드라이버 보고각(채널각)을 읽어, 그 자세가
  의도한 모델각으로 읽히도록 offset_deg 를 계산한다.

식 — ★수식을 여기 복사하지 말 것. **JointMap 을 그대로 쓴다.**
  (2026-08-10: 이 파일이 `offset = ch − ref·sign` 복사본을 갖고 있었는데 gear_k 와
   커플링이 추가되자 조용히 틀린 값을 내게 됐다. gen_emb_init_pose.py 도 같은 실수를
   했었다 — 변환 수식의 복사본은 예외 없이 stale 이 된다.)

      q_raw   = (q_ch − offset) / (sign·k)
      q_joint = q_raw − coef · q_joint_src            (커플링 있는 축만)
  기준자세에서 q_joint 가 ref 로 읽히길 원하므로, raw_ref = ref + coef·ref_src 에 대해
      ★ offset = q_ch(기준자세) − raw_ref · sign · k

  ⇒ 커플링·감속비가 없는 축(k=1, coef 없음)이고 ref=0 이면 offset 은 **그때의 채널각 그대로**다.

사용법:
  1) 제어기를 띄운다:            cd emb && python3 app/biped_emb.py
  2) 모드를 **off(limp)** 로 두고 로봇을 기준자세로 물리적으로 맞춘다(지그/수평계).
  3) python3 diag/calib_zero.py              # 계산만 (config 안 건드림)
     python3 diag/calib_zero.py --apply      # config/biped_emb.yaml 의 offset_deg 갱신
     python3 diag/calib_zero.py --ref 0,0,-55,0,0,0,-55,0    # 기준자세 직접 지정

기준자세 기본값은 config 의 `calib.ref_joint_deg`.

⚠ 이 스크립트는 모터에 아무것도 쓰지 않는다. --apply 는 config 파일만 고친다.
⚠ calf·foot 은 드라이버 감속비 오설정(7:1 가정)으로 채널각이 실제의 1.5/1.2 배다.
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
    import numpy as np
    from joint_map import JointMap
    _jm0 = JointMap({**cfg_all, "joints": joints}) if False else None
    ch = [float(q[i]) * float(j["sign"]) * float(j.get("gear_k", 1.0)) + float(j["offset_deg"])
          for i, j in enumerate(joints)]   # ⚠커플링 미반영 폴백 — q_ch_deg 를 쓰는 게 정답
    return ch, "q_leg_deg 에서 역산 (⚠제어기 재시작 필요 — q_ch_deg 필드가 없는 구버전)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="기준자세 모델각[deg], 콤마구분 8개. 생략시 config calib.ref_joint_deg")
    ap.add_argument("--apply", action="store_true", help="config/biped_emb.yaml 의 offset_deg 를 실제로 갱신")
    ap.add_argument("--settle-s", type=float, default=8.0,
                    help="이 시간 동안 자세가 멈춰 있어야 채취한다[s]. 0 이면 검사 생략")
    ap.add_argument("--max-age", type=float, default=3.0,
                    help="상태파일 최대 허용 나이[s]. 넘으면 중단(옛 값·mock 값 방지)")
    ap.add_argument("--force", action="store_true",
                    help="변화량 게이트를 무시하고 적용(원인을 확인한 뒤에만)")
    ap.add_argument("--allow-powered", action="store_true",
                    help="모터가 여자된 상태에서도 강행(권장하지 않음 — 위 주석 참조)")
    ap.add_argument("--max-shift", type=float, default=3.0,
                    help="새 offset 이 이만큼[deg] 넘게 바뀌면 확인을 요구한다")
    ap.add_argument("--only", help="이 축만 갱신(콤마구분 이름). 나머지는 현행 offset 유지. "
                                   "예: --only HL_hip,HR_hip")
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

    # ── ★★게이트 0: 모터가 **놓여 있어야** 한다 (2026-08-11 추가) ──────────
    #   왜 필요한가 — 실제로 당했다. 제어기가 HOME 을 잡고 있으면 그 자세는
    #   **"제어기가 생각하는 홈"** 이지 기준자세가 아니다. 홈복귀가 calf 를 2° 못
    #   맞추고 끝나면 그 2° 가 그대로 offset 으로 박힌다.
    #   ⇒ 커플링 때문에 **foot 은 그보다 더 나쁘다**: foot 채널각은 물리적 foot+calf 를
    #     반영하므로 calf 오차가 foot offset 에도 실린다(2026-08-11 HR 에서 잔차 0.06° 로
    #     확인된 경로다).
    #   ★영점은 **기구(지그)** 가 정의해야 한다. 모터가 아니라.
    #     그래서 모터가 여자돼 있으면 중단한다 — 지그를 물리고 limp 로 두고 잴 것.
    st0 = read_state()
    # ── 게이트 0-a: 상태가 **신선한가** ────────────────────────────────────
    #   _age 를 계산해 놓고 **쓰지 않았다**. 제어기가 안 떠 있으면 몇 시간 전 값이나
    #   mock 실행이 남긴 값을 그대로 읽고 offset 을 계산한다 — 2026-08-11 실제로
    #   mock 상태파일을 읽고 "HL 이 91° 변한다" 는 허구의 표가 나왔다.
    if st0.get("_age", 1e9) > a.max_age:
        print(f"\n  ❌ 상태파일이 **{st0['_age']:.0f}초 전** 값이다(허용 {a.max_age:.0f}초)"
              f" — 채취를 중단한다.\n")
        print( "     제어기가 떠 있지 않으면 남아 있던 옛 값(또는 mock 값)을 그대로 읽는다.")
        print(f"     ▸ 제어기를 **off(limp) 모드로** 띄운 뒤 다시 실행할 것:")
        print(f"       cd {EMB} && python3 app/biped_emb.py --start-mode off\n")
        return 1

    # ── 게이트 0-b: 제어기가 **아무 축도 붙들고 있지 않아야** 한다 ──────────
    #   ⚠motors_on 은 **전축 공통 플래그**다. "한쪽만 지그로 고정하고 나머지는 제어기가
    #     잡고 있는" 상태를 못 가른다 — 2026-08-11 실제로 그렇게 잡혔다:
    #       HL 은 지그가 정의(신뢰 가능) · **HR 은 biped_emb 홀드 위치가 그대로 박힘**
    #     그 흔적이 HR_foot +2.13° · HR_calf +1.19° 로 다른 축보다 크게 남았다.
    #   ⇒ mode 가 off(limp) 가 아니면 중단한다. limp 여야 **기구만이** 자세를 정한다.
    if st0.get("mode") not in ("off", None) or st0.get("motors_on"):
        print(f"\n  ❌ 제어기가 **축을 붙들고 있다**(mode={st0.get('mode')}, "
              f"motors_on={st0.get('motors_on')}) — 채취를 중단한다.\n")
        print( "     제어기가 붙들고 있는 자세는 '제어기가 생각하는 홈' 이지 기준자세가 아니다.")
        print( "     홈복귀가 2° 못 맞추고 끝나면 그 2° 가 그대로 영점에 박힌다.")
        print( "     커플링 때문에 foot 은 calf 오차까지 함께 뒤집어쓴다.")
        print( "     ⚠일부 축만 지그로 고정해도 **나머지 축은 제어기 위치가 박힌다** —")
        print( "       motors_on 은 전축 공통 플래그라 그 상황을 구분하지 못한다.")
        print( "\n  ▸ **전 축을 지그로 고정**하고 제어기를 off(limp) 로 둔 뒤 다시 실행할 것:")
        print(f"       python3 app/biped_emb.py --start-mode off")
        print( "    (기구가 자세를 정의해야 한다 — 모터가 아니라)")
        print( "    정말 강행하려면 --allow-powered")
        if not a.allow_powered:
            return 1
        print( "     ⚠--allow-powered 로 강행한다. 이 영점은 제어기 오차를 포함한다.\n")

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
    print("  영점 캘리브레이션 — offset = 채널각(기준자세) − raw각(기준자세)·sign·k")
    print("=" * 78)
    print(f"  채널각 출처 : {src}")
    print(f"  상태 나이   : {st['_age']:.2f}s      모드: {st.get('mode','?')}")
    if st.get("mode") not in ("off", None):
        print(f"  ⚠ 모드가 off 가 아니다({st.get('mode')}). 모터가 구동 중이면 기준자세가")
        print(f"     '사람이 맞춘 자세'가 아니라 '제어기가 끌고 간 자세'다. off 로 두고 다시 할 것.")
    if st["_age"] > 1.0:
        print(f"  ⚠ 상태가 {st['_age']:.1f}s 나 묵었다 — 제어기가 살아있는지 확인할 것.")
    print()

    # ★JointMap 으로 raw 를 만든다 — sign·k·커플링을 여기서 다시 쓰지 않는다.
    import numpy as np
    from joint_map import JointMap
    _jm = JointMap(cfg)
    _raw_ref = np.asarray(ref, float).copy()
    for _d, _s, _c in zip(_jm.cpl_dst, _jm.cpl_src, _jm.cpl_coef):
        _raw_ref[_d] = ref[_d] + _c * ref[_s]
    new = [ch[i] - float(_raw_ref[i]) * float(_jm.sk[i]) for i in range(n)]

    # ── ★★--only: 일부 축만 갱신한다 (2026-08-24) ──────────────────────────
    #   왜 필요한가 — **재현되는 축과 안 되는 축이 한 판에 섞여 나온다.** 매달린 채
    #   limp 이면 thigh·foot 은 자유 진자라 잴 때마다 다르고(이번 37°·48°),
    #   hip 은 기구가 잡아 줘서 재현된다. 전 축 --apply 는 그 둘을 구분하지 못해
    #   **믿을 수 있는 축까지 못 고치게** 만든다("나머지가 못 미더우니 통째로 보류").
    #   ⇒ 믿는 축만 골라서 반영한다. 나머지는 old 를 그대로 둔다(계산값을 버린다).
    #   ⚠골라낸 축이 정말 재현되는지는 **사람이 판단한다.** 이 옵션은 그 판단을
    #     실행할 수단일 뿐, 재현성을 보증하지 않는다 — 아래 재현성 표는 그대로 나온다.
    only = None
    if a.only:
        want = [w.strip() for w in a.only.split(",") if w.strip()]
        bad = [w for w in want if w not in names]
        if bad:
            sys.exit(f"✗ --only 에 없는 축: {bad}\n  가능: {', '.join(names)}")
        only = set(want)
        held = [names[i] for i in range(n) if names[i] not in only]
        for i in range(n):
            if names[i] not in only:
                new[i] = old[i]          # ★계산값을 버리고 현행 유지
        print(f"  ▸ --only {', '.join(want)} — 이 축만 갱신한다.")
        print(f"    나머지 {len(held)}축은 **현행 유지**: {', '.join(held)}\n")

    print(f"  {'축':10} {'sign·k':>7} {'채널각':>9} {'기준모델각':>11} "
          f"{'현 offset':>10} {'→ 새 offset':>12} {'변화':>8}")
    print("  " + "-" * 74)
    for i in range(n):
        tag = "" if only is None else ("  ← 갱신" if names[i] in only else "  (유지)")
        print(f"  {names[i]:10} {float(_jm.sk[i]):+7.2f} {ch[i]:+9.2f} {ref[i]:+11.2f} "
              f"{old[i]:+10.2f} {new[i]:+12.2f} {new[i]-old[i]:+8.2f}{tag}")
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
    # ★검증 범위는 **갱신하는 축**뿐이다 (2026-08-24, --only 도입과 함께).
    #   왜: 이 검사는 "offset 식이 맞는가" 를 보는 것이지 "로봇이 기준자세인가" 가 아니다.
    #   --only 에서 유지축은 옛 offset 을 그대로 쓰므로 지금 자세가 ref 로 안 읽히는 게
    #   **정상**이다(자유 진자라 어디에 있든 상관없다). 전 축으로 재면 그게 통째로 '실패' 로
    #   잡혀 갱신을 막는다 — 실제로 그렇게 막혔다(최대오차 58.5° = HL_calf 의 현재 자세).
    idx = [i for i in range(n) if (only is None or names[i] in only)]
    err = float(np.max(np.abs(back[idx] - np.array(ref)[idx]))) if idx else 0.0
    scope = "전 축" if only is None else f"갱신축({', '.join(names[i] for i in idx)})"
    print(f"  ▸ 왕복검증({scope}): 새 offset 적용 시 기준자세가 "
          f"{[round(float(back[i]),3) for i in idx]} 로 읽힌다")
    print(f"    최대오차 {err:.6f}°  →  {'✅ 통과' if err < 1e-6 else '❌ 실패'}")
    if err >= 1e-6:
        print("    ✗ 식이 안 맞는다. 적용하지 말 것."); return 1
    if only is not None:
        # 유지축이 **지금** 어떻게 읽히는지 같이 보여준다 — 크면 그 축은 기준자세가 아니다.
        hold_i = [i for i in range(n) if names[i] not in only]
        worst = max(hold_i, key=lambda i: abs(back[i]-ref[i])) if hold_i else None
        if worst is not None:
            print(f"    (유지축은 지금 자세로 " +
                  " · ".join(f"{names[i]} {back[i]:+.1f}°" for i in hold_i) + " 로 읽힌다 —")
            print(f"     기준자세가 아니라는 뜻이고, 그래서 갱신하지 않는 것이다. "
                  f"최대 {names[worst]} {back[worst]-ref[worst]:+.1f}°)")
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

    # ── ★커플링 몫 분리 (2026-08-11 추가) ──────────────────────────────────
    #   foot 채널각은 **물리적 foot+calf** 를 반영한다. 그래서 calf 가 기준자세에서
    #   벗어나 있으면 그 오차가 foot offset 에도 그대로 실린다.
    #   ⇒ foot 의 offset 변화 중 **calf 로 설명되는 몫**을 갈라 보여준다. 남는 몫이
    #     크면 그건 커플링이 아니라 **기구 변경이나 자세 오차**다.
    #   실측 근거(2026-08-11): HR 은 calf Δ−1.09° → foot 예측 −1.09° · 실측 −1.15°
    #     (잔차 0.06°)로 정확히 맞았고, HL 은 9.30° 가 남아 **풀리 재조임**으로 밝혀졌다.
    cpl_note = []
    for _d, _s, _c in zip(_jm.cpl_dst, _jm.cpl_src, _jm.cpl_coef):
        dq_src = (new[_s] - old[_s]) / float(_jm.sk[_s])       # 소스축 변화(모델각 등가)
        dq_dst = (new[_d] - old[_d]) / float(_jm.sk[_d])
        expl = _c * dq_src
        cpl_note.append((names[_d], names[_s], dq_dst, expl, dq_dst - expl))
    if cpl_note and any(abs(o[2]) > 0.3 for o in cpl_note):
        print("  ── 커플링 몫 분리 (모델각 등가) ──")
        print(f"  {'축':10} {'전체변화':>9} {'커플링설명':>11} {'남는몫':>9}  판정")
        for nm_, src_, tot, expl, res in cpl_note:
            tag = "✓ 커플링으로 설명됨" if abs(res) < 0.5 else "★설명 안 됨 — 기구변경/자세오차 의심"
            print(f"  {nm_:10} {tot:+9.2f} {expl:+11.2f} {res:+9.2f}  {tag}")
        print(f"     (커플링설명 = {src_} 변화 × coef. 남는몫이 크면 그 축을 직접 확인할 것)\n")

    # ── ★변화량 게이트 (2026-08-11) — 큰 이동은 **확인 없이 적용 못 한다** ──
    #   재현성 검사는 경고만 하고 --apply 를 막지 않았다. 그래서 자세가 틀어진 채
    #   재교정하면 그대로 박혔다(2026-08-11 HL_foot 14.06° 이동).
    big = [(names[i], old[i], new[i]) for i in range(n)
           if abs(new[i] - old[i]) > a.max_shift]
    if big and a.apply and not a.force:
        print(f"  ── ❌ offset 변화가 {a.max_shift:.1f}° 를 넘는 축이 있다 — **적용 중단** ──")
        for nm_, o, nw in big:
            print(f"     {nm_:10} {o:+8.2f} → {nw:+8.2f}   ({nw-o:+.2f}°)")
        print( "     큰 이동은 셋 중 하나다: ①기구 변경(풀리·벨트) ②자세가 기준이 아님")
        print( "                              ③지난 영점이 틀렸음")
        print( "     ②라면 지금 적용하면 오차가 영점으로 박힌다. 원인을 확인할 것.")
        print(f"     확인했고 그래도 적용하려면 --force (또는 --max-shift 로 문턱 조정)\n")
        return 1

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
