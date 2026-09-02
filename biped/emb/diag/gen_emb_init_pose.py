#!/usr/bin/env python3
"""gen_emb_init_pose.py — Emb 기동 램프의 목표자세를 **우리 영점 기준**으로 생성/패치.

════════════════════════════════════════════════════════════════════════════════
⚠⚠ 2026-08-26 현재 — 이 도구는 **아무 효과가 없다**. 죽은 경로다. ⚠⚠
════════════════════════════════════════════════════════════════════════════════
  --patch 자체는 지금도 멀쩡히 동작한다. m_fGaitCmd_PositionInit[] 배열에 값이 정확히
  들어가고 되읽기 검증도 통과한다. 그런데 **그 배열이 매 부팅 통째로 덮어써진다**:

      halGait.cpp:586   ← 2026-08-10 '기동 램프 = 제자리 유지' 패치
          m_fGaitCmd_PositionInit[unMotID] = m_fGaitStt_Position[unMotID];

  램프가 시작되기 직전(상태수신 100틱 구간)에 목표를 **측정각**으로 덮어쓴다.
  그래서 Befo == Curr 이 되고, half-sine 보간 결과가 상수 = **현재 자세**다.
  ⇒ 램프는 4.5초 동안 돌지만 로봇은 있던 자리를 잡고만 있다.
    **기동할 때 로봇이 안 움직이는 것이 정상이다.**
  ⇒ 이 스크립트가 배열에 무슨 값을 써 넣든 부팅 100틱째에 지워진다. 지금 이 도구를
    돌리는 것은 **로봇 거동에 아무 영향이 없다.**

  ★다시 유효해지는 조건: halGait.cpp:586 **그 한 줄만 지우고** 다시 빌드하면 된다.
    그 순간 배열이 되살아나고 — 이 도구도, 아래 '문제:' 절도 다시 참이 된다.
    ⚠램프 블록 자체를 지우거나 주석처리하면 안 된다. 같은 경로 끝(:691)에서
      m_ucIsGaitInitialized = 1 을 세우므로, 지우면 **SHM 명령이 영원히 수용되지
      않는다.** 08-10 패치도 그래서 초기화 시퀀스는 남기고 목표만 없앤 것이다.
  ★그래서 이 도구를 지우지 않고 남겨 둔다. :627 을 원복하는 날 반드시 다시 필요하고,
    그때 손으로 값을 베끼면 아래 compute() 주석의 사고를 또 되풀이하게 된다.

  아래 '문제:' 절은 **:627 을 원복한 경우**를 전제한 서술이다(= 이 도구의 존재 이유).
────────────────────────────────────────────────────────────────────────────────

문제(:627 원복 시):
  RobotEmbedded 는 기동할 때 4.5초 half-sine 램프로 전 관절을 m_fGaitCmd_PositionInit
  으로 보낸다(halGait.cpp:597-647). 벤더 기본값은 전부 0 인데 그 0 은 **채널각 0** 이고,
  영점을 잡고 나면 우리 홈자세와 전혀 다른 자세다. 실측 예(2026-08-10 영점 기준):
      채널 0 = 모델각 [−1.4, +31.0, −36.6, −25.2, +0.9, +34.4, −47.2, −27.7]
      우리 홈 = 모델각 [ 0.0, +35.0, −60.0, +60.0,  0.0, +35.0, −60.0, +60.0]
  발목이 약 86° 어긋난다. 즉 **Emb 를 재기동할 때마다 로봇이 홈이 아닌 자세로 끌려간다.**
  그 4.5초 동안 우리 SHM 명령은 무시되므로(halGait_IsInitialized 게이트) 소프트로 막을
  방법이 없다 — 목표값 자체를 고쳐야 한다.
  ★이 게이트 서술만은 :627 패치와 **무관하게 지금도 참이다**: 상태수신 100틱 + 램프
    4500틱 @1kHz ≈ 4.6초 동안 m_ucIsGaitInitialized = 0 이라 SHM 명령이 무시된다.
    (지금은 그동안 제자리를 유지할 뿐이다.) Emb 기동 후 5초는 기다린 뒤 명령할 것.

좌표계가 같다는 근거 (이게 성립해야 그대로 넣을 수 있다):
  · m_fGaitCfg_Dir 8축 전부 1.0            (halGait.cpp:127-134)
  · m_fGaitOfs_Position 은 0 으로 강제      (halGait.cpp:536)
  · 상태:  m_fGaitStt_Position = fDir·(get − ofs) = get        (:552)
  · 명령:  m_fGaitSet_Position = fDir·target + ofs  = target   (:770-783)
  · SHM 위치명령은 m_fGaitCmd_Position 으로 들어가고(:308), 램프는 같은 배열에
    m_fGaitCmd_PositionInit 을 넣는다(:650,660,677)
  ⇒ m_fGaitCmd_PositionInit 의 단위 = 우리가 SHM 에 쓰는 **채널각**이다.
    따라서 JointMap.q_joint_to_ch(홈자세) 결과를 그대로 넣으면 된다
    (sign·gear_k·offset·커플링·±180 포화가 전부 반영된 값).
  ⚠줄번호는 **2026-08-26 판 halGait.cpp 기준**이다. 벤더 트리는 수시로 밀린다 —
    이 절엔 원래 :481/:497/:265/:665-674 로 적혀 있었는데 08-20 개정에서 통째로
    어긋났다. 안 맞으면 줄번호를 믿지 말고 심볼명으로 grep 할 것.

★손으로 베끼지 말 것. config 의 sign/offset 이 바뀌면 이 값도 같이 바뀌어야 하는데,
  수동 동기화는 반드시 어긋난다(같은 실수를 GUI 의 jog 한계 복제에서 이미 했다).
  영점을 다시 잡을 때마다 이 스크립트를 --patch 로 재실행할 것
  — 단 이것도 :627 을 원복한 뒤에나 의미가 있다. 지금은 돌려도 그만 안 돌려도 그만이다.

사용법:
  python3 diag/gen_emb_init_pose.py            # 값만 출력(대조용) — 지금은 이쪽만 쓸모 있다
  python3 diag/gen_emb_init_pose.py --patch    # halGait.cpp 를 실제로 수정(.bak 백업)

⚠--patch 후 **RobotEmbedded 를 다시 빌드**해야 반영된다:
    cd ~/ZSource/RobotEmbedded/build && make -j4
⚠벤더 트리(~/ZSource)는 git 관리가 아니다(2026-08-26 확인). .bak 백업만이 되돌릴 수단이다.
  별도로 관리되는 트리이므로 이 스크립트의 --patch 말고는 손대지 말 것.
⚠--patch 만으로는 로봇 거동이 바뀌지 않는다. 맨 위에 적었듯 halGait.cpp:586 을 함께
  지워야 비로소 이 값이 실제 램프 목표가 된다. 둘은 **세트로 해야 한다.**
"""
from __future__ import annotations
import argparse
import os
import re
import shutil
import sys

EMB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(EMB, "config", "biped_emb.yaml")
HAL = os.path.expanduser("~/ZSource/RobotEmbedded/modules/ctrlGait/halGait.cpp")

# ENUM_Gait_JointID 순서(= 우리 channel 0~7). defineConfigMotor.h:104-111
EMB_NAMES = ["LtR", "LtP", "LkP", "LaP", "RtR", "RtP", "RkP", "RaP"]
EMB_DESC = ["Left  thighs   Roll", "Left  thighs   Pitch", "Left  Knee     Pitch",
            "Left  Ankle    Pitch", "Right thighs   Roll", "Right thighs   Pitch",
            "Right Knee     Pitch", "Right Ankle    Pitch"]


def compute():
    """★변환은 반드시 JointMap 을 통해서 한다 — 수식을 여기 복사하지 말 것.

    2026-08-10 실제로 당했다: 이 파일이 `ch = q·sign + offset` 복사본을 갖고 있었는데
    gear_k 와 커플링이 추가되자 조용히 틀린 값을 뽑아 벤더 파일에 패치까지 했다
    (HL_calf −6.62 를 냈지만 정답은 +23.38). 이 스크립트의 docstring 이 바로 그
    "손으로 베끼지 말 것" 을 경고하고 있었는데 정작 자신이 베끼고 있었다.
    """
    import yaml
    sys.path.insert(0, os.path.join(EMB, "interface"))
    from joint_map import JointMap
    cfg = yaml.safe_load(open(CFG))
    js = cfg["joints"]
    ref = cfg["calib"]["ref_joint_deg"]
    home = cfg.get("home", {}).get("q_deg")
    if home is not None and [float(x) for x in home] != [float(x) for x in ref]:
        print("  ⚠ home.q_deg 와 calib.ref_joint_deg 가 다르다. **home 기준**으로 생성한다.")
        print(f"      home {home}\n      ref  {ref}")
        ref = home
    jm = JointMap(cfg)
    ch_full = jm.q_joint_to_ch([float(x) for x in ref])      # sign·k·offset·커플링 전부 반영
    ch = {int(j["channel"]): float(ch_full[int(j["channel"])]) for j in js}
    if sorted(ch) != list(range(8)):
        sys.exit(f"✗ 채널이 0~7 이 아니다: {sorted(ch)}")
    return cfg, js, ref, [ch[c] for c in range(8)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", action="store_true", help="halGait.cpp 를 실제로 수정")
    ap.add_argument("--hal", default=HAL)
    a = ap.parse_args()

    cfg, js, ref, ch = compute()
    names = [j["name"] for j in js]

    # ★배너 — docstring 을 안 읽고 바로 돌리는 사람이 반드시 있다(내가 그랬다).
    print("\n  ⚠⚠ 2026-08-26 현재 이 도구는 로봇 거동에 **아무 효과가 없다**.")
    print("     halGait.cpp:586 이 매 부팅 m_fGaitCmd_PositionInit[] 을 측정각으로 덮어쓴다")
    print("     (2026-08-10 '기동 램프 = 제자리 유지' 패치). 아래 값은 그대로 지워진다.")
    print("     :627 한 줄을 지워 원복해야 이 값이 실제 램프 목표가 된다 — 자세한 건 이 파일의")
    print("     모듈 docstring 을 읽을 것.\n")

    print("  Emb 기동 램프 목표 = 우리 홈자세  (JointMap.q_joint_to_ch 로 산출)\n")
    print(f"  {'Emb':5} {'우리 축':10} {'모델각':>9} {'sign':>5} {'offset':>9} {'→ 채널각':>10}")
    for i in range(8):
        j = next(x for x in js if int(x["channel"]) == i)
        k = names.index(j["name"])
        print(f"  {EMB_NAMES[i]:5} {j['name']:10} {float(ref[k]):+9.2f} "
              f"{float(j['sign']):+5.0f} {float(j['offset_deg']):+9.2f} {ch[i]:+10.2f}")
    over = [(EMB_NAMES[i], ch[i]) for i in range(8) if abs(ch[i]) > 180.0]
    # ★2026-08-26 정정: 옛 주석은 "Emb 가 ±180 으로 **래핑**한다(:666-671)" 였고,
    #   08-07 판(.bak)에선 실제로 while(fPosition>180) fPosition-=360 이 살아 있었다.
    #   지금 그 while 은 주석처리됐고(:773-778) 대신 m_fGaitCfg_Min/Max(=±180) 클램프가
    #   산다(:780-781). 래핑이면 −190 이 +170 으로 돌지만 포화면 −180 에 멈춘다 —
    #   거동이 전혀 다르다. 어느 쪽이든 ±180 을 넘기면 패치하면 안 된다는 결론은 같다.
    print(f"\n  ±180 초과: {over if over else '없음 ✅'}"
          f"    (Emb 는 fDir·target 을 ±180 으로 **포화**시킨다 — halGait.cpp:716-717)")
    if over:
        print("  ❌ 초과 축이 있으면 패치하지 말 것. offset 을 다시 잡아야 한다.")
        return 1

    if not a.patch:
        print("\n  (출력만 함. 실제 반영은 --patch)")
        return 0

    src = open(a.hal).read()
    m = re.search(r"(static\s+float\s+m_fGaitCmd_PositionInit\s*\[[^\]]*\]\s*=\s*\{)(.*?)(\n\};)",
                  src, re.S)
    if not m:
        sys.exit("✗ m_fGaitCmd_PositionInit 배열을 못 찾았다 — 벤더 코드 구조가 바뀌었다.")
    head, body, tail = m.groups()
    # 주석 블록(/* ... */ = 팔 관절)은 보존하고, 그 뒤 실제 8줄만 교체
    cm = re.search(r"^(.*?\*/\s*\n)(.*)$", body, re.S)
    keep, legs = (cm.group(1), cm.group(2)) if cm else ("", body)
    n_old = len(re.findall(r"^\s*[-+0-9.]+\s*,", legs, re.M))
    if n_old != 8:
        sys.exit(f"✗ 다리 항목이 8개가 아니라 {n_old}개다 — 수동 확인 필요.")
    new_legs = "".join(
        f"\t{ch[i]:+9.2f}, //ENUM_Gait_JointID_{EMB_NAMES[i]} - {EMB_DESC[i]}"
        f"{'   ★biped 영점 기준(emb/diag/gen_emb_init_pose.py 생성)' if i == 0 else ''}\n"
        for i in range(8))
    new = src[:m.start()] + head + keep + new_legs + tail + src[m.end():]

    bak = a.hal + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(a.hal, bak)
        print(f"\n  백업 생성: {bak}")
    else:
        print(f"\n  백업 이미 있음(유지): {bak}")
    open(a.hal, "w").write(new)

    # 되읽어 검증
    #   ★주석 블록(/* 팔 관절 8줄 */)을 반드시 먼저 잘라낼 것 — 그 안의 `0.0,` 도
    #     정규식에 걸려서 16개가 잡힌다(실제로 걸려서 원복됐다).
    chk = re.search(r"m_fGaitCmd_PositionInit\s*\[[^\]]*\]\s*=\s*\{(.*?)\n\};", open(a.hal).read(), re.S)
    cbody = chk.group(1)
    cm2 = re.search(r"^.*?\*/\s*\n(.*)$", cbody, re.S)
    if cm2:
        cbody = cm2.group(1)
    got = [float(x) for x in re.findall(r"^\s*([-+0-9.]+)\s*,", cbody, re.M)]
    if len(got) != 8 or max(abs(got[i] - round(ch[i], 2)) for i in range(8)) > 1e-9:
        shutil.copy2(bak, a.hal)
        sys.exit(f"✗ 패치 후 검증 실패 → 원복했다. got={got}")
    print(f"  ✅ 패치 완료 · 검증 통과 {got}")
    print(f"\n  ⚠ 반드시 다시 빌드할 것:  cd ~/ZSource/RobotEmbedded/build && make -j4")
    # ★2026-08-26 정정: 예전엔 여기서 무조건 "다리를 받쳐 둘 것" 이라고 경고했다.
    #   지금은 :627 이 목표를 측정각으로 덮어쓰므로 램프가 무동작이다 — 그 경고를 그대로
    #   두면 **거짓 경고**가 되고, 거짓 경고는 진짜 경고까지 같이 못 믿게 만든다.
    #   그래서 조건부로 갈라 적는다.
    print(f"  ⚠ 이 값이 실제로 쓰이는지는 halGait.cpp:586 을 지웠는지에 달려 있다:")
    print(f"      · :627 을 지워 원복했다면 → 빌드 후 첫 기동 때 **다리를 받쳐 둘 것.**")
    print(f"        램프가 4.5초간 이 자세로 끌고 간다(현재 자세와 멀면 그만큼 크게 움직인다).")
    print(f"      · :627 을 그대로 뒀다면(=지금 기본) → 부팅 때 이 배열이 측정각으로")
    print(f"        덮어써진다. 로봇은 제자리를 유지한다. **안 움직이는 게 정상이다.**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
