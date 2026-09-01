#!/usr/bin/env bash
# D1(OCS2 NMPC+WBC) GUI 텔레옵 원샷 — teleop_gui(dearpygui) + D1 뷰어(VIEW=1) 동시 기동.
#   GUI의 v/vy/w 명령을 CMDFILE(/tmp/quad_cmd.json)로 D1이 50Hz 소비. 좌스틱=전진·측방, 우스틱=선회.
#   ★D1이 flat 포함 전 지형 담당(A제어기와 독립 런처). Walk/Ready/게이트 버튼=보행 중 라이브 전환.
# 사용(어디서든):  bash /home/jsh/문서/jsh/simulation/quad/ocs2_02leg/run_gui_d1.sh [course|flat|rough|slope|<절대경로.mjcf>] [gait]
set -e
HERE=/home/jsh/문서/jsh/simulation/quad
PXI=/home/jsh/miniforge3/envs/proxddp/bin/python     # ★dearpygui 설치된 env(기본 python엔 없음)
CMD=/tmp/quad_cmd.json
# ★D1이 flat 포함 전 지형 담당. A제어기(run_gui.sh)와 독립 런처로 별도 관리.
case "${1:-course}" in
  course) MJCF=$HERE/mjcf/quad_terrain_course.mjcf; PERC=1 ;;
  flat)   MJCF=$HERE/mjcf/quad_real_17dof_waist_sphere.mjcf; PERC=0 ;;   # ★평지=perceptive OFF(지형인지 불필요=오버헤드). perceptive는 명령에 취약X(과거 붕괴는 참조버그, 68feab0서 수정)
  rough)  MJCF=$HERE/mjcf/quad_terrain_rough.mjcf; PERC=1 ;;
  slope)  MJCF=$HERE/mjcf/quad_terrain_slope.mjcf; PERC=1 ;;
  *)      MJCF="$1"; PERC=1 ;;
esac
# ★지형만 perceptive(발판배치+지형적응 base높이). 평지는 끔(지형인지 불필요). CMD_TAU=명령 슬루 시정수.
# ★KP_F/KD_F=swing 발끝 추종게인 상향(기본350→700, ~10-15% 조임·falls0). 잔여오차=전방swing 지연(속도의존, 게인한계).
PERCENV=""; [ "$PERC" = 1 ] && PERCENV="PERCEPTIVE=1 PLACEMENT=1 TERRAIN_Z=1 SMOOTH_W=0.25"
GAIT="${2:-trot}"
pkill -TERM -f teleop_gui_d1.py 2>/dev/null || true
pkill -TERM -f test02legMujoco   2>/dev/null || true
sleep 0.5
# GUI 백그라운드(proxddp python=dearpygui). QUAD_CMD=D1이 읽는 채널과 동일.
setsid bash -c "cd '$HERE'; DISPLAY='${DISPLAY:-:0}' QUAD_CMD='$CMD' '$PXI' teleop_gui_d1.py > /tmp/teleop_gui_d1.log 2>&1" </dev/null &
sleep 1.2
pgrep -f teleop_gui_d1.py >/dev/null && echo "GUI RUNNING" || { echo "GUI DEAD — 로그:"; tail -6 /tmp/teleop_gui_d1.log; }
# D1 뷰어 포그라운드
source /opt/ros/humble/setup.bash 2>/dev/null || true
source "$HERE/ocs2_ws/install/setup.bash" 2>/dev/null || true
EXE="$HERE/ocs2_ws/install/ocs2_legged_robot/lib/ocs2_legged_robot/test02legMujoco"
CFG="$HERE/ocs2_02leg/config/task.info $HERE/ocs2_02leg/urdf/02leg_ocs2.urdf $HERE/ocs2_02leg/config/reference.info"
echo "▶ D1 GUI 구동 | $(basename "$MJCF") gait=$GAIT | perceptive=$PERC(0=평지 순수속도추종) | 명령슬루 τ=0.30 | 좌스틱=전진·우스틱=선회 (GUI에서 Walk 먼저 누르기)"
env OMP_NUM_THREADS=1 WBC=1 VIEW=1 WBC_LEGGED=1 $PERCENV MPC_HZ=100 CMD_TAU=0.30 KP_F=700 KD_F=50 JKP=20 JKD=1 CMDFILE="$CMD" \
    "$EXE" $CFG "$MJCF" "$GAIT" 100000
