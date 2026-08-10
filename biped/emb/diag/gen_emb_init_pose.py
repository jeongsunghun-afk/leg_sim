#!/usr/bin/env python3
"""gen_emb_init_pose.py — Emb 기동 램프의 목표자세를 **우리 영점 기준**으로 생성/패치.

문제:
  RobotEmbedded 는 기동할 때 4.5초 half-sine 램프로 전 관절을 m_fGaitCmd_PositionInit
  (기본 전부 0) 으로 보낸다(halGait.cpp:522-580). 그 0 은 **채널각 0** 이고, 영점을
  잡고 나면 우리 홈자세와 전혀 다른 자세다. 실측 예(2026-08-10 영점 기준):
      채널 0 = 모델각 [−1.4, +31.0, −36.6, −25.2, +0.9, +34.4, −47.2, −27.7]
      우리 홈 = 모델각 [ 0.0, +35.0, −60.0, +60.0,  0.0, +35.0, −60.0, +60.0]
  발목이 약 86° 어긋난다. 즉 **Emb 를 재기동할 때마다 로봇이 홈이 아닌 자세로 끌려간다.**
  그 4.5초 동안 우리 SHM 명령은 무시되므로(halGait_IsInitialized 게이트) 소프트로 막을
  방법이 없다 — 목표값 자체를 고쳐야 한다.

좌표계가 같다는 근거 (이게 성립해야 그대로 넣을 수 있다):
  · m_fGaitCfg_Dir 8축 전부 1.0            (halGait.cpp:151-158)
  · m_fGaitOfs_Position 은 0 으로 강제      (halGait.cpp:481)
  · 상태:  m_fGaitStt_Position = fDir·(get − ofs) = get        (:497)
  · 명령:  m_fGaitSet_Position = fDir·target + ofs  = target   (:665-674)
  · SHM 위치명령은 m_fGaitCmd_Position 으로 들어가고(:265), 램프는 같은 배열에
    m_fGaitCmd_PositionInit 을 넣는다(:547,557,574)
  ⇒ m_fGaitCmd_PositionInit 의 단위 = 우리가 SHM 에 쓰는 **채널각**이다.
    따라서 JointMap.q_joint_to_ch(홈자세) 결과를 그대로 넣으면 된다
    (sign·gear_k·offset·커플링·±180 포화가 전부 반영된 값).

★손으로 베끼지 말 것. config 의 sign/offset 이 바뀌면 이 값도 같이 바뀌어야 하는데,
  수동 동기화는 반드시 어긋난다(같은 실수를 GUI 의 jog 한계 복제에서 이미 했다).
  영점을 다시 잡을 때마다 이 스크립트를 --patch 로 재실행할 것.

사용법:
  python3 diag/gen_emb_init_pose.py            # 값만 출력(대조용)
  python3 diag/gen_emb_init_pose.py --patch    # halGait.cpp 를 실제로 수정(.bak 백업)

⚠--patch 후 **RobotEmbedded 를 다시 빌드**해야 반영된다:
    cd ~/ZSource/RobotEmbedded/build && make -j4
⚠벤더 트리(~/ZSource)는 git 관리가 아니다. .bak 백업만이 되돌릴 수단이다.
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

    print("  Emb 기동 램프 목표 = 우리 홈자세  (JointMap.q_joint_to_ch 로 산출)\n")
    print(f"  {'Emb':5} {'우리 축':10} {'모델각':>9} {'sign':>5} {'offset':>9} {'→ 채널각':>10}")
    for i in range(8):
        j = next(x for x in js if int(x["channel"]) == i)
        k = names.index(j["name"])
        print(f"  {EMB_NAMES[i]:5} {j['name']:10} {float(ref[k]):+9.2f} "
              f"{float(j['sign']):+5.0f} {float(j['offset_deg']):+9.2f} {ch[i]:+10.2f}")
    over = [(EMB_NAMES[i], ch[i]) for i in range(8) if abs(ch[i]) > 180.0]
    print(f"\n  ±180 초과: {over if over else '없음 ✅'}"
          f"    (Emb 는 fDir·target 을 ±180 으로 래핑한다 — halGait.cpp:666-671)")
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
    print(f"  ⚠ 빌드 후 첫 기동 때 다리를 받쳐 둘 것 — 램프가 4.5초간 이 자세로 끌고 간다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
