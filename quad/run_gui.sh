#!/usr/bin/env bash
# 02_Leg 17-DOF GUI 텔레옵 원샷 런처 (C++ 뷰어 + dearpygui GUI)
#   사용: bash run_gui.sh [map]        map = course(기본)|flat|stairs|rough|friction|gap|stepping|soft
#   기본 맵 = 종합코스(마찰→험지→계단, perceptive 자동 ON)
#
# ★견고화(한 번에 확실히 실행): ①이전 인스턴스 SIGTERM 후 "실제 종료까지 폴링"(SIGKILL은 GL/X
#   리소스를 안 풀어 다음 실행이 컨텍스트를 못 얻는 간헐 크래시 유발 → 부드럽게 종료+대기).
#   ②뷰어는 "state 파일이 실제로 갱신(렌더 루프 생존)"될 때까지 검증하고, 실패 시 자동 재시도.
#   ③뷰어가 확인된 뒤에만 GUI 실행. ④pgrep은 bash wrapper까지 잡으므로 실제 바이너리/렌더로 판정.
# ★CMDFILE/STATE_PUB 필수(누락 시 GUI 명령 무시하고 저절로 전진) — 이 스크립트가 항상 붙임.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # simulation/quad
CPP="$HERE/cpp"
PXI=/home/jsh/miniforge3/envs/proxddp/bin/python
export DISPLAY="${DISPLAY:-:0}"
CMD=/tmp/quad_cmd.json; STATE=/tmp/quad_state.json

case "${1:-course}" in
  course)   MJCF=mjcf/quad_terrain_course.mjcf ;;
  flat)     MJCF=mjcf/quad_real_17dof_waist_sphere.mjcf ;;
  stairs)   MJCF=mjcf/quad_terrain_stairs.mjcf ;;
  rough)    MJCF=mjcf/quad_terrain_rough.mjcf ;;
  friction) MJCF=mjcf/quad_terrain_friction.mjcf ;;
  gap)      MJCF=mjcf/quad_terrain_gap.mjcf ;;
  stepping) MJCF=mjcf/quad_terrain_stepping.mjcf ;;
  soft)     MJCF=mjcf/quad_terrain_soft.mjcf ;;
  *)        MJCF="$1" ;;                                       # 임의 mjcf 경로 허용(mjcf/ 포함해 전달)
esac

# ── ① 이전 인스턴스 부드럽게 종료 후 실제 종료까지 폴링(최대 5초, 안 죽으면 SIGKILL) ──
pkill -TERM -f 'build/trot_view'   2>/dev/null
pkill -TERM -f teleop_gui_17dof.py 2>/dev/null
for _ in $(seq 1 25); do
  pgrep -f 'build/trot_view' >/dev/null || pgrep -f teleop_gui_17dof.py >/dev/null || break
  sleep 0.2
done
pkill -9 -f 'build/trot_view'   2>/dev/null   # 잔존 시에만
pkill -9 -f teleop_gui_17dof.py 2>/dev/null
sleep 0.5; rm -f "$CMD" "$STATE"

# 실제 뷰어 바이너리 실행 중인가(bash wrapper 제외)
viewer_alive(){ pgrep -f 'build/trot_view ' >/dev/null; }
# 렌더 루프 생존(state mtime 갱신) 확인
viewer_rendering(){
  local a b; a=$(stat -c %Y.%N "$STATE" 2>/dev/null) || return 1
  sleep 0.8; b=$(stat -c %Y.%N "$STATE" 2>/dev/null) || return 1
  [ "$a" != "$b" ]
}

# ── ② 뷰어: 렌더 확인될 때까지 최대 4회 재시도 ──
VOK=0
for try in 1 2 3 4; do
  rm -f "$STATE"
  setsid bash -c "cd '$CPP'; env RATE=1.0 CMDFILE='$CMD' STATE_PUB='$STATE' \
    ./build/trot_view '../$MJCF' > /tmp/trot_view.log 2>&1" </dev/null &
  # ★렌더(STATE 갱신) 최대 ~11초 폴링(구 3초 단발은 warmup+GL init 느린 머신서 오탐 킬)
  for w in $(seq 1 6); do
    sleep 1
    if ! viewer_alive; then echo "viewer 조기종료(try $try) — 로그:"; tail -4 /tmp/trot_view.log 2>/dev/null; break; fi
    if viewer_rendering; then VOK=1; break; fi
  done
  [ $VOK = 1 ] && { echo "viewer RUNNING (try $try)"; break; }
  echo "viewer 재시도 $try…"; pkill -9 -f 'build/trot_view' 2>/dev/null; sleep 1
done
if [ $VOK = 0 ]; then echo "viewer DEAD (4회 실패)"; tail -5 /tmp/trot_view.log 2>/dev/null; exit 1; fi

# ── ③ 뷰어 확인 후 GUI ──
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' '$PXI' teleop_gui_17dof.py > /tmp/teleop_gui.log 2>&1" </dev/null &
sleep 4

echo "map=$MJCF"
pgrep -f teleop_gui_17dof.py >/dev/null && echo "gui RUNNING" || { echo "gui DEAD"; tail -5 /tmp/teleop_gui.log; }
echo "✅ 준비 완료 — 뷰어+GUI 실행 중"
