#!/usr/bin/env bash
# 02_Leg OCS2 NMPC+WBC 뷰어. 사용: ./run_view.sh [gait] [vx]
#   gait=stance|trot|standing_trot|static_walk (기본 stance) · vx=전진 m/s (기본 0)
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # = simulation/quad
GAIT="${1:-stance}"; VXX="${2:-0}"
source /opt/ros/humble/setup.bash 2>/dev/null || true
source "$HERE/ocs2_ws/install/setup.bash" 2>/dev/null || true
EXE="$HERE/ocs2_ws/install/ocs2_legged_robot/lib/ocs2_legged_robot/test02legMujoco"
CFG="$HERE/ocs2_02leg/config/task.info $HERE/ocs2_02leg/urdf/02leg_ocs2.urdf $HERE/ocs2_02leg/config/reference.info"
MJCF="$HERE/mjcf/quad_real_17dof_waist_sphere.mjcf"
# ★legged식 WBC(WBC_LEGGED) 사용. 동적 gait은 base 가중↑(W_BASE 50)로 안정 trot.
#   1ms/1kHz·바닥강성 0.005·허리 단단 홀드는 바이너리 기본값(A 제어기와 동일 세팅).
# ★W_BASE는 바이너리가 게이트별로 설정(trot=150 고속회복·bound/walk=50). 강제 안 함(env W_BASE로만 override).
WFLAGS="WBC_LEGGED=1"
echo "▶ gait=$GAIT vx=$VXX  (마우스=카메라, 창 닫기=종료)"
env WBC=1 VIEW=1 VX="$VXX" $WFLAGS "$EXE" $CFG "$MJCF" "$GAIT" 100000
