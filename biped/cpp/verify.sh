#!/usr/bin/env bash
# ── biped 컨트롤러 품질 게이트 (헤드리스 회귀 배터리) ───────────────────────
#   빌드 → 2점 평발 stand + 1점 점발 보행(정지~0.20) + 선회 + 측방 + 배포경로(EST_CTRL).
#   판정: 낙상 0 (하드) · tilt ≤ 한계 · 전진거리 x ∈ [min,max] · **base z ≥ 하한**.
#   사용: ./verify.sh            (~100초. 시뮬이 대략 실시간이라 시간이 걸린다)
#
# ★base z 하한이 왜 있나 (2026-08-13): 낙상만 보면 **조용한 품질 저하를 놓친다.**
#   실제로 condim 4 실험이 "무낙상" 인데 tilt 0.2→2.0° · base z 0.438→0.403(3.5cm 침하)
#   이었다. 낙상 게이트만으로는 통과한다. 침하는 stance 힘이 모자란다는 신호다.
#
# ★평발 stand 가 첫 항목인 이유 — **실기에서 제일 먼저 할 동작**이라서다.
#   그리고 이 항목이 없어서 2026-08-12 Qflat8 회귀(1.29s 낙상)를 아무도 못 잡았다.
#
#   ★임계값은 2026-08-13 기준선에 여유를 준 값이다(회귀=낙상/tilt급증/침하만 잡는다).
#     물리 캡이 아니므로 여기를 올려 통과시키지 말 것 — 올릴 일이 생기면 그게 회귀다.
set -uo pipefail
cd "$(dirname "$0")"                       # simulation/biped/cpp
PT=../biped_from_quad.mjcf                 # 1점 점발 (보행 배포 모델)
FF=../biped_flatfoot.mjcf                  # 2점 평발 (정적 자세유지 · 실기 첫 목표)
DUR=${DUR:-15}

echo "▶ 빌드(biped_sim)…"
cmake --build build --target biped_sim -j >/dev/null 2>&1 || { echo "❌ BUILD FAIL"; exit 1; }

pass=0; fail=0
_num(){ echo "$1" | grep -oE "$2" | grep -oE '[+-]?[0-9.]+' | head -1; }
run(){ # name model envs vx T tilt_max min_x max_x min_z
  local name=$1 model=$2 envs=$3 vx=$4 T=$5 tmax=$6 minx=$7 maxx=$8 minz=$9
  local o=$(env $envs timeout 300 ./build/biped_sim "$model" "$vx" "$T" 2>/dev/null | tail -n 1)
  local t=$(_num "$o" 'tilt=[0-9.]+') x z ok=1 src
  if grep -q 'EST_CTRL' <<<"$o"; then                    # 배포경로: falls= 로 판정, GT 좌표
    [ "$(_num "$o" 'falls=[0-9]+')" = "0" ] || ok=0
    src=$(sed -n 's/.*GT=(\(.*\)).*/\1/p' <<<"$o")
  else                                                   # 단독: (무낙상) 여부로 판정
    grep -q '(무낙상)' <<<"$o" || ok=0
    src=$(sed -n 's/.*base=(\(.*\)).*/\1/p' <<<"$o")
  fi
  x=$(cut -d, -f1 <<<"$src"); z=$(cut -d, -f3 <<<"$src")
  awk "BEGIN{exit !(${t:-99} <= $tmax)}"   || ok=0
  awk "BEGIN{exit !(${x:-0} >= $minx)}"    || ok=0
  awk "BEGIN{exit !(${x:-0} <= $maxx)}"    || ok=0
  awk "BEGIN{exit !(${z:-0} >= $minz)}"    || ok=0
  if [ $ok = 1 ]; then printf "  ✅ %-20s tilt=%5s° x=%7s z=%6s\n" "$name" "$t" "$x" "$z"; pass=$((pass+1))
  else printf "  ❌ %-20s tilt=%5s° x=%7s z=%6s  (한계 tilt≤%s · x∈[%s,%s] · z≥%s)\n" \
         "$name" "$t" "$x" "$z" "$tmax" "$minx" "$maxx" "$minz"; fail=$((fail+1)); fi
}

echo "▶ 2점 평발 자세유지 (실기 첫 목표)"
#   기준선 tilt 0.2° · base(−0.010, 0.001, 0.438). 드리프트 ±5cm · 침하 2cm 까지 허용.
#
# ★★2026-08-14 **DUR 15 → 60 으로 늘렸다. 15초가 실패를 가리고 있었다.**
#   PACE 축별 마찰(fit_v2 이후) 에서 이 stand 는 **~20.6초에 넘어진다** — 15초 창 밖이라
#   그동안 ✅ 로 통과했다. v6 도 20.73s 로 같다(즉 v6 회귀가 아니라 v2 부터다).
#     구 스칼라 JDAMP0.09/JFRIC0.38 : 60s 무낙상 tilt 0.1°
#     fit_v2                        : 20.60s 낙상
#     fit_v6                        : 20.73s 낙상
#   ⚠튜닝으로 맞출 문제가 아니다 — `JFRIC.calf` 를 **0.05%**(0.5717→0.5720) 만 바꿔도
#     무낙상 tilt 0.5° 가 낙상으로 뒤집힌다. 8점 스윕에서 2점만 생존했다.
#     낙상 방향도 판마다 반대다(base x −0.52 / +0.59). **여유가 없다는 뜻이다.**
#   ⇒ 이 줄은 지금 **의도적으로 실패한다.** 고쳐질 때까지 red 로 두는 게 맞다.
#     원인 후보: WBIC 는 접촉을 **3-DOF 점힘**으로 푸는데 평발은 면접촉이다
#     (condim 3→6 이 이 stand 를 가장 먼저 깨뜨린 것과 같은 뿌리로 보인다).
run "flat stand 60s" $FF "CONTACT=1" 0.00 60 2.0 -0.05 0.05 0.42
echo "▶ 1점 점발 보행"
run "walk v0.05"   $PT ""          0.05 $DUR 5   0.60  99   0.50
run "walk v0.15"   $PT ""          0.15 $DUR 5   1.80  99   0.50
run "walk v0.20"   $PT ""          0.20 $DUR 5   2.40  99   0.50
echo "▶ 선회·측방  (접촉모델을 건드리면 여기가 먼저 드러난다)"
run "turn wz0.3"   $PT "WZ=0.3"    0.10 12   5   0.90  99   0.50
run "side vy0.1"   $PT "VY=0.10"   0.00 10   5  -9.9   99   0.50
echo "▶ 배포경로 (추정 폐루프 EST_CTRL)"
#   ★2026-08-14 실기 기본값을 그대로 건다 — 구동지연 8.4ms(실측) + 운동학 지연보상.
#     종전엔 지연도 보상도 없이 돌아서, **실기가 실제로 겪는 조건을 한 번도 안 봤다.**
#     (지연보상은 그때 실기 바이너리에 아예 없었다. biped_deploy.cpp 주석 참조.)
#     보상 없이 이 조건을 걸면 vx 0.15/0.20 에서 낙상한다 — 그게 이 줄의 존재 이유다.
run "EST v0.15"    $PT "EST_CTRL=1 ACT_LAT_MS=8.4 LAT_COMP_MS=8.4 LAT_COMP_KIN=1" 0.15 $DUR 5  1.80  99   0.50
run "EST v0.20"    $PT "EST_CTRL=1 ACT_LAT_MS=8.4 LAT_COMP_MS=8.4 LAT_COMP_KIN=1" 0.20 $DUR 5  2.40  99   0.50

echo "──────────── PASS=$pass  FAIL=$fail ────────────"
[ $fail = 0 ] && { echo "✅ ALL PASS — 회귀 없음"; exit 0; } || { echo "❌ REGRESSION — 위 실패 항목 확인"; exit 1; }
