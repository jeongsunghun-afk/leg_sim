#!/usr/bin/env bash
# D1(OCS2 NMPC+WBC) GUI 텔레옵 원샷 — teleop_gui(dearpygui) + D1 뷰어(VIEW=1) 동시 기동.
#   GUI의 v/vy/w 명령을 CMDFILE(/tmp/quad_cmd.json)로 D1이 50Hz 소비. 좌스틱=전진·측방, 우스틱=선회.
# 사용(어디서든):  bash /home/jsh/문서/jsh/simulation/quad/ocs2_02leg/run_gui_d1.sh [course|flat|rough|slope|<절대경로.mjcf>] [gait]
set -e
HERE=/home/jsh/문서/jsh/simulation/quad
PXI=/home/jsh/miniforge3/envs/proxddp/bin/python     # ★dearpygui 설치된 env(기본 python엔 없음)
CMD=/tmp/quad_cmd.json
case "${1:-course}" in
  course) MJCF=$HERE/mjcf/quad_terrain_course.mjcf ;;
  flat)   MJCF=$HERE/mjcf/quad_real_17dof_waist_sphere.mjcf ;;
  rough)  MJCF=$HERE/mjcf/quad_terrain_rough.mjcf ;;
  slope)  MJCF=$HERE/mjcf/quad_terrain_slope.mjcf ;;
  *)      MJCF="$1" ;;
esac
GAIT="${2:-trot}"
pkill -TERM -f teleop_gui_17dof.py 2>/dev/null || true
pkill -TERM -f test02legMujoco   2>/dev/null || true
sleep 0.5
# GUI 백그라운드(proxddp python=dearpygui). QUAD_CMD=D1이 읽는 채널과 동일.
setsid bash -c "cd '$HERE'; DISPLAY='${DISPLAY:-:0}' QUAD_CMD='$CMD' '$PXI' teleop_gui_17dof.py > /tmp/teleop_gui_d1.log 2>&1" </dev/null &
sleep 1.2
pgrep -f teleop_gui_17dof.py >/dev/null && echo "GUI RUNNING" || { echo "GUI DEAD — 로그:"; tail -6 /tmp/teleop_gui_d1.log; }
# D1 뷰어 포그라운드
source /opt/ros/humble/setup.bash 2>/dev/null || true
source "$HERE/ocs2_ws/install/setup.bash" 2>/dev/null || true
EXE="$HERE/ocs2_ws/install/ocs2_legged_robot/lib/ocs2_legged_robot/test02legMujoco"
CFG="$HERE/ocs2_02leg/config/task.info $HERE/ocs2_02leg/urdf/02leg_ocs2.urdf $HERE/ocs2_02leg/config/reference.info"
echo "▶ D1 GUI 구동 | $(basename "$MJCF") gait=$GAIT | 좌스틱=전진·우스틱=선회 (GUI에서 Walk 먼저 누르기)"
env WBC=1 VIEW=1 WBC_LEGGED=1 PERCEPTIVE=1 PLACEMENT=1 TERRAIN_Z=1 SMOOTH_W=0.25 MPC_HZ=100 CMDFILE="$CMD" \
    "$EXE" $CFG "$MJCF" "$GAIT" 100000
