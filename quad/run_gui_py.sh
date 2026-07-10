#!/usr/bin/env bash
# 02_Leg 17-DOF GUI 텔레옵 원샷 런처 — ★Python 백엔드(quad_mpc_wbic_17dof 자체 뷰어 + GUI)
#   사용: bash run_gui_py.sh [map]     map = course(기본)|flat|stairs|rough|friction|gap|stepping|soft
#   C++ 실시간 배포는 run_gui.sh 사용. 이건 Python 레퍼런스(연구/디버그, 느림).
#   기본 맵 = 종합코스(3레인: 마찰→소프트 / 험지→계단 / 갭→스테핑, perceptive 자동 ON)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # simulation/quad
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
  soft)     MJCF=quad_terrain_soft.mjcf ;;
  *)        MJCF="$1" ;;                                       # 임의 mjcf 경로 허용
esac

pkill -f trot_view 2>/dev/null; pkill -f teleop_gui_17dof 2>/dev/null; pkill -f quad_mpc_wbic_17dof 2>/dev/null
sleep 1.5; rm -f "$CMD" "$STATE"

# 터미널1: dearpygui GUI
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' QUAD_CMD='$CMD' '$PXI' teleop_gui_17dof.py > /tmp/teleop_gui.log 2>&1" </dev/null &
sleep 3
# 터미널2: Python 컨트롤러(자체 MuJoCo 뷰어) — MJCF=지형씬·CMDFILE/STATE_PUB=GUI 연동·GEAR_FOOT=발목 8:1
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' MJCF='$MJCF' GEAR_FOOT=0.5714 CMDFILE='$CMD' STATE_PUB='$STATE' \
  '$PXI' quad_mpc_wbic_17dof.py --robot ours_17dof_waist_sphere --mode trot > /tmp/quad_py.log 2>&1" </dev/null &
sleep 5

echo "map=$MJCF (Python 백엔드)"
pgrep -f quad_mpc_wbic_17dof >/dev/null && echo "controller RUNNING" || { echo "controller DEAD"; tail -8 /tmp/quad_py.log; }
pgrep -f teleop_gui_17dof >/dev/null && echo "gui RUNNING" || { echo "gui DEAD"; tail -5 /tmp/teleop_gui.log; }
