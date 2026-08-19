#!/usr/bin/env bash
# run_leg_L.sh — 왼다리(HL) 4축 마찰 측정. **오른다리는 작업자가 손으로 잡는다.**
#
# ★--solo (2026-08-12, 사용자 결정)
#   측정축 **하나만** 제어한다. 나머지 7축은 kp=kd=0 으로 완전히 놓는다.
#     · 홈복귀도 측정축만 움직인다 — 손으로 잡은 축은 그대로 둔다
#     · 트립 검사도 측정축만 본다 — 손 위치는 오차가 크게 나는 게 정상이고
#       그걸로 시험이 꺼지면 안 된다
#     · 홀드축이 없으므로 "홀드축이 밀렸다"·"스톨" 트립이 원천적으로 안 난다
#   ⚠하위 관절이 무여자라 I_link 강체가정이 깨진다 — **inertia·pace 에는 쓰지 말 것.**
#     마찰·기동은 ±방향 차로 중력이 상쇄되므로 영향이 작다.
#
# ★왜 이 방식인가 — 2026-08-12 에 모터로 잡는 방식이 반복 실패했다:
#     두 다리가 안쪽으로 처져 **발끼리 부딪히고**(늘어진 자세에서 −27mm 침투),
#     그 상태로 모터가 밀어 스톨 → 과전류 → **드라이버 파워단 5회 사망**
#     (ch7 · ch4 2회 · ch0 · ch2). 손으로 잡으면 그 경로가 통째로 사라진다.
#
# ★순서 — foot → thigh → calf → hip (사용자 지정)
#   측정축이 바뀔 때마다 홈복귀가 그 축만 움직이므로, 작업자는 **그 축을 놓고
#   나머지를 잡으면** 된다.
#
# ★시험 종류 — 마찰(기본)과 토크를 **같은 스크립트**로 돈다 (2026-08-12, 사용자 요청).
#   파일을 나누지 않는다: 오늘 `$1` → `$*` 버그를 두 파일에 각각 고쳐야 했다.
#   같은 절차(제어기 종료·축별 안내·hip 경고·실패 시 계속 여부)를 복사하면 반드시 갈라진다.
#
# 사용:  bash run_leg_L.sh                 # 마찰, 4축 전부
#        bash run_leg_L.sh 3 1             # 마찰, 특정 축만
#        bash run_leg_L.sh --torque        # **토크(무여자 램프)**, 4축 전부
#        bash run_leg_L.sh --torque 3 1    # 토크, 특정 축만
set -u
cd "$(dirname "$0")"

TESTS=friction
if [ "${1:-}" = "--torque" ]; then TESTS=torque; shift; fi

# ★`$1` 이 아니라 `$*` 다 (2026-08-12). `bash run_leg_L.sh 3 1 2` 를 실행했더니
#   **ch3 만 돌고 "왼다리 완료" 로 끝났다** — $1 은 "3" 하나뿐이다.
#   조용히 두 축을 건너뛰고 성공한 척 끝나는 게 제일 나쁜 실패다.
CHS=${*:-"3 1 2 0"}          # HL_foot · HL_thigh · HL_calf · HL_hip
NAME=(HL_hip HL_thigh HL_calf HL_foot HR_hip HR_thigh HR_calf HR_foot)

echo "════════════════════════════════════════════════════════════════"
_TN=$([ "$TESTS" = torque ] && echo "토크(무여자 램프)" || echo "마찰")
echo " 왼다리(HL) $_TN 측정 — **오른다리를 손으로 잡아 주세요**"
echo "════════════════════════════════════════════════════════════════"
echo " ⚠ 제어기(biped_emb.py)를 먼저 끕니다 — writer 는 하나여야 합니다."
pkill -f 'app/biped_emb.py' 2>/dev/null && sleep 1

for ch in $CHS; do
  echo
  echo "────────────────────────────────────────────────────────────────"
  echo " ch$ch  ${NAME[$ch]}  측정"
  echo "   · 이 축은 **놓아** 주세요 (모터가 잡습니다)"
  echo "   · 나머지 축은 **잡아** 주세요 (무여자입니다)"
  echo "   · 힘으로 붙들지 말고 **가려는 자리에서 받쳐** 주세요"
  if [ "$ch" = "0" ] || [ "$ch" = "4" ]; then
    echo "   ⚠ hip 은 **오늘 드라이버를 잃고 EtherCAT 이 2회 얼어붙은 축**입니다."
    echo "     · 가동폭이 ±2.5° 뿐이라 눈에 거의 안 보입니다 — 정상입니다"
    echo "     · **소리를 들으세요.** 끙끙대는 소리가 나면 즉시 Ctrl+C"
    echo "     · 스톨 감지가 중력+2Nm·300ms 에서 끊습니다(τ_trip 10Nm 보다 훨씬 먼저)"
  fi
  echo "────────────────────────────────────────────────────────────────"
  python3 - "$ch" "$TESTS" <<'EOF'
import sys, yaml
sys.path[:0] = ["tests"]
ch, tests = int(sys.argv[1]), sys.argv[2]
sp_all = yaml.safe_load(open("spec.yaml"))
if tests == "torque":
    # ★토크 시험은 **각도를 안 흔든다.** 무여자(kp=kd=0)로 토크만 램프해 기동을 본다.
    #   그래서 작업자에게 알려야 할 것이 다르다: 얼마나 세게 미는가(swing)와,
    #   **움직이면 복원력이 0 이라 계속 흘러간다**는 점이다.
    tm = sp_all["torque_mode"]
    sw = float((tm.get("swing_by_ch") or {}).get(ch, tm.get("tau_max_nm", 1.4)))
    print(f"   토크 램프: 그 자리의 중력에서 **±{sw:g} Nm** 까지 "
          f"{tm.get('ramp_nm_per_s', 0.25)} Nm/s 로 올린다")
    print(f"   ⚠**무여자다(kp=kd=0).** 기동하면 복원력이 없어 그대로 흘러간다 —")
    print(f"     드리프트 감시가 잡지만, 손이 가까이 있으면 안 된다.")
else:
    from act_measure_friction import swing_str
    fr = sp_all["friction"]
    m = {k: (dict(v) if isinstance(v, dict) else v) for k, v in fr.items()}
    for sec, kv in (m.pop("by_ch", None) or {}).get(ch, {}).items():
        m.setdefault(sec, {}).update(kv)
    st, spd = m["sweep"]["stroke_deg"], m["sweep"]["speeds_dps"]
    print(f"   흔드는 폭·빠르기: ±{st/2:g}° 를 "
          + " · ".join(swing_str(st, float(v)).split("·")[1] for v in spd))
    print(f"   기동푸시: {m['breakaway']['max_push_deg']:g}° 까지 "
          f"{m['breakaway']['ramp_dps']:g}deg/s 로 밀어 봄 "
          f"(초과토크 상한 {m['breakaway'].get('tau_cap_nm', '없음')}Nm)")
EOF
  read -r -p "   준비되면 Enter (건너뛰려면 s + Enter): " ans
  [ "$ans" = "s" ] && { echo "   건너뜀"; continue; }

  python3 actuator_test.py --ch "$ch" --tests "$TESTS" --solo
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "   ✗ ch$ch 실패(rc=$rc)."
    read -r -p "   계속할까요? (y/N): " go
    [ "$go" = "y" ] || { echo "   중단."; exit $rc; }
  fi
done

echo
echo "════════════════════════════════════════════════════════════════"
echo " 왼다리 $_TN 완료. 리포트: results/output.html"
echo " ⚠ 각 실행이 report 를 덮어씁니다 — 값은 터미널 로그에서 확인할 것."
echo "════════════════════════════════════════════════════════════════"
