#!/bin/bash
# run_view_robot.sh — 로봇(Pi, aarch64)에서 biped C++ 뷰어 실행.
#
# run_gui_cpp.sh 의 로봇판. 차이:
#   - 노트북 conda(/home/jsh/miniforge3/...) 대신 시스템 python + ~/mujoco 사용
#   - GUI(teleop_gui_biped, dearpygui)는 선택 — 미설치면 뷰어만 띄운다
#     (명령은 CMDFILE JSON 으로 직접 넣을 수 있다)
#
# ★순수 시뮬레이션이다. SHM·실모터와 무관하므로 Emb 가 돌고 있어도 안전하다.
#
# 사용:
#   ./run_view_robot.sh              # 추정 폐루프(배포 경로)
#   GT=1 ./run_view_robot.sh         # GT 제어(디버그)
#   DISPLAY=:1 ./run_view_robot.sh   # 디스플레이 지정
# 종료: pkill -f biped_view
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
MJ="${MUJOCO_DIR:-$HOME/mujoco}"
CMD="${CMDFILE:-/tmp/biped_cmd.json}"
MODEL="${MODEL:-$HERE/biped_flatfoot.mjcf}"
export DISPLAY="${DISPLAY:-:0}"

# ★X 인증 쿠키 자동탐색 — 이 데스크톱은 Wayland(mutter)라 X 앱은 Xwayland 를 거친다.
#   ssh·비대화 셸에는 XAUTHORITY 가 없어 "Authorization required, but no authorization
#   protocol specified" 로 glfw init 이 실패한다. 세션의 Xwayland 쿠키를 직접 지정한다.
if [ -z "${XAUTHORITY:-}" ]; then
  for c in /run/user/$(id -u)/.mutter-Xwaylandauth.* "$HOME/.Xauthority"; do
    [ -r "$c" ] && { export XAUTHORITY="$c"; break; }
  done
fi
echo "XAUTHORITY=${XAUTHORITY:-<없음>}"

[ -x "$HERE/cpp/build/biped_view" ] || {
  echo "❌ biped_view 없음. 먼저 빌드:"
  echo "   cd $HERE/cpp && CONDA_PREFIX=/opt/ros/jazzy cmake -B build -S . \\"
  echo "     -DEIGEN_INCLUDE=/usr/include/eigen3 -DEIQP_INCLUDE=/opt/ros/jazzy/include \\"
  echo "     -DEIQP_LIB=/opt/ros/jazzy/lib/aarch64-linux-gnu/libeiquadprog.so \\"
  echo "     -DMUJOCO_INCLUDE=\$HOME/mujoco/include -DMUJOCO_LIB=\$HOME/mujoco/lib/libmujoco.so"
  echo "   cmake --build build --target biped_view -j4"
  echo "   (GLFW 필요: sudo apt install -y libglfw3-dev — 없으면 뷰어 타겟이 건너뛰어진다)"
  exit 1; }

pkill -f biped_view 2>/dev/null; sleep 1

# 시작 자세: 2점 평발 정적 rest (GUI 없이도 이 파일을 고치면 명령이 바뀐다)
[ -f "$CMD" ] || echo '{"v":0.0,"vy":0.0,"w":0.0,"body_h":0.38,"mode":"stand","contact":"2pt"}' > "$CMD"

MODEDESC="추정 폐루프(배포)"; [ "${GT:-0}" = "1" ] && MODEDESC="GT 제어(디버그)"
echo "모드: $MODEDESC · DISPLAY=$DISPLAY · 모델=$(basename "$MODEL")"

setsid bash -c "cd '$HERE/cpp'; CMDFILE='$CMD' GT='${GT:-0}' \
  DISPLAY='$DISPLAY' XAUTHORITY='${XAUTHORITY:-}' LD_LIBRARY_PATH='$MJ/lib' \
  ./build/biped_view '$MODEL' > /tmp/biped_view.log 2>&1" </dev/null &
sleep 3

if pgrep -f biped_view >/dev/null; then
  echo "✅ 뷰어 RUNNING (로그: /tmp/biped_view.log)"
  echo "   명령 바꾸기: echo '{\"v\":0.15,\"body_h\":0.38,\"mode\":\"walk\",\"contact\":\"2pt\"}' > $CMD"
  echo "   종료: pkill -f biped_view"
else
  echo "❌ 뷰어 DEAD — 로그:"; tail -n 15 /tmp/biped_view.log
fi
