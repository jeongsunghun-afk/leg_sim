#!/usr/bin/env bash
# D1(OCS2 NMPC+WBC) GUI 구동 — teleop_gui_17dof가 /tmp/quad_cmd.json에 발행한 v/vy/w를 D1이 소비.
# 사용:
#   터미널①  cd simulation/quad && python teleop_gui_17dof.py
#   터미널②  cd simulation/quad && ocs2_02leg/run_gui_d1.sh [mjcf] [gait]
#   좌스틱=전후·측방, 우스틱=선회. (D1은 trot 보행+선회 지원. 종합코스 기본.)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # = simulation/quad
MJCF="${1:-$HERE/mjcf/quad_terrain_course.mjcf}"
GAIT="${2:-trot}"
CMD="${QUAD_CMD:-/tmp/quad_cmd.json}"
source /opt/ros/humble/setup.bash 2>/dev/null || true
source "$HERE/ocs2_ws/install/setup.bash" 2>/dev/null || true
EXE="$HERE/ocs2_ws/install/ocs2_legged_robot/lib/ocs2_legged_robot/test02legMujoco"
CFG="$HERE/ocs2_02leg/config/task.info $HERE/ocs2_02leg/urdf/02leg_ocs2.urdf $HERE/ocs2_02leg/config/reference.info"
echo "▶ D1 GUI 구동 | mjcf=$(basename $MJCF) gait=$GAIT cmd=$CMD"
echo "  (좌스틱=전진/측방, 우스틱=선회. 먼저 GUI에서 Walk 누르고 조이스틱)"
env WBC=1 VIEW=1 WBC_LEGGED=1 W_BASE=50 PERCEPTIVE=1 PLACEMENT=1 TERRAIN_Z=1 \
    SMOOTH_W=0.25 MPC_HZ=100 CMDFILE="$CMD" \
    "$EXE" $CFG "$MJCF" "$GAIT" 100000
