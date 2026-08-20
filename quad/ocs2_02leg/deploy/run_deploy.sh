#!/usr/bin/env bash
# D1 배포 골격 실행(MuJoCo-as-model 백엔드) — HAL 경계 뒤 OCS2 제어. 헤드리스 검증.
# 사용(어디서든): bash .../deploy/run_deploy.sh [flat|slope|<절대.mjcf>] [gait] [simTime]
#   env: VX(고정 전진)·PERCEPTIVE 등은 아래서 지형별 자동세팅(override 가능).
set -e
HERE=/home/jsh/문서/jsh/simulation/quad
case "${1:-slope}" in
  flat)  MJ=$HERE/mjcf/quad_real_17dof_waist_sphere.mjcf; PERC=0 ;;
  slope) MJ=$HERE/mjcf/quad_terrain_slope8.mjcf;          PERC=1 ;;
  *)     MJ="$1";                                         PERC=1 ;;
esac
GAIT="${2:-trot}"; T="${3:-18}"
PERCENV=""; [ "$PERC" = 1 ] && PERCENV="PERCEPTIVE=1 PLACEMENT=1 TERRAIN_Z=1 W_BASE=50"
source /opt/ros/humble/setup.bash 2>/dev/null || true
source "$HERE/ocs2_ws/install/setup.bash" 2>/dev/null || true
EXE="$HERE/ocs2_ws/install/ocs2_legged_robot/lib/ocs2_legged_robot/d1_deploy"
CFG="$HERE/ocs2_02leg/config/task.info $HERE/ocs2_02leg/urdf/02leg_ocs2.urdf $HERE/ocs2_02leg/config/reference.info"
echo "▶ d1_deploy | $(basename "$MJ") gait=$GAIT | perceptive=$PERC | HAL 경계 뒤 OCS2 제어"
exec env OMP_NUM_THREADS=1 $PERCENV VX="${VX:-0.25}" "$EXE" $CFG "$MJ" "$GAIT" "$T"
