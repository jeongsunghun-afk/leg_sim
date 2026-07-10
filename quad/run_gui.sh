#!/usr/bin/env bash
# 02_Leg 17-DOF GUI 텔레옵 원샷 런처 (C++ 뷰어 + dearpygui GUI)
#   사용: bash run_gui.sh [map]        map = course(기본)|flat|stairs|rough|friction|gap|stepping
#   기본 맵 = 종합코스(마찰→험지→계단, perceptive 자동 ON)
# ★뷰어에 CMDFILE/STATE_PUB 필수(누락 시 GUI 명령 무시하고 저절로 전진) — 이 스크립트가 항상 붙임.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # simulation/quad
CPP="$HERE/cpp"
PXI=/home/jsh/miniforge3/envs/proxddp/bin/python
export DISPLAY="${DISPLAY:-:0}"
CMD=/tmp/quad_cmd.json; STATE=/tmp/quad_state.json

case "${1:-course}" in
  course)   MJCF=quad_terrain_course.mjcf ;;
  flat)     MJCF=quad_real_17dof_waist_sphere.mjcf ;;
  stairs)   MJCF=quad_terrain_stairs.mjcf ;;
  rough)    MJCF=quad_terrain_rough.mjcf ;;
  friction) MJCF=quad_terrain_friction.mjcf ;;
  gap)      MJCF=quad_terrain_gap.mjcf ;;
  stepping) MJCF=quad_terrain_stepping.mjcf ;;
  *)        MJCF="$1" ;;                                       # 임의 mjcf 경로 허용
esac

pkill -f trot_view 2>/dev/null; pkill -f teleop_gui_17dof 2>/dev/null
sleep 1.5; rm -f "$CMD" "$STATE"

# 터미널1: C++ 뷰어(1kHz) — setsid=SIGURG 회피, CMDFILE/STATE_PUB=GUI 연동
setsid bash -c "cd '$CPP'; env GEAR_FOOT=0.5714 RATE=1.0 CMDFILE='$CMD' STATE_PUB='$STATE' \
  ./build/trot_view '../$MJCF' > /tmp/trot_view.log 2>&1" </dev/null &
sleep 2
# 터미널2: dearpygui GUI(proxddp env)
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' '$PXI' teleop_gui_17dof.py > /tmp/teleop_gui.log 2>&1" </dev/null &
sleep 4

echo "map=$MJCF"
pgrep -f trot_view >/dev/null && echo "viewer RUNNING" || { echo "viewer DEAD"; tail -5 /tmp/trot_view.log; }
pgrep -f teleop_gui_17dof >/dev/null && echo "gui RUNNING" || { echo "gui DEAD"; tail -5 /tmp/teleop_gui.log; }
