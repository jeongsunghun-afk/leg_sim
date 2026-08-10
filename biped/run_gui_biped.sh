#!/bin/bash
# biped GUI 원샷 런처 — 컨트롤러+뷰어(biped_run) + 슬림 GUI(teleop_gui_biped) 동시 실행.
# JSON 채널(/tmp/biped_cmd.json)로 디커플. proxddp env(dearpygui+mujoco+qpsolvers).
# 사용: ./run_gui_biped.sh   (뷰어 창 + GUI 창이 뜸. 자기 터미널에서 실행해야 디스플레이 연결)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jsh/miniforge3/envs/proxddp/bin/python
CMD=/tmp/biped_cmd.json
export DISPLAY="${DISPLAY:-:0}"

pkill -f biped_run.py 2>/dev/null || true
pkill -f teleop_gui_biped 2>/dev/null || true
sleep 1

# 초기 명령(제자리)
echo '{"v":0.0,"body_h":0.50,"mode":"stand","contact":"1pt"}' > "$CMD"

# ① 컨트롤러 + 뷰어
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' QUAD_CMD='$CMD' BIPED_MJCF='${BIPED_MJCF:-}' '$PY' biped_run.py > /tmp/biped_run.log 2>&1" </dev/null &
sleep 2
# ② 슬림 GUI (quad/ 에 위치)
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' QUAD_CMD='$CMD' '$PY' teleop_gui_biped.py > /tmp/teleop_gui_biped.log 2>&1" </dev/null &
sleep 2

pgrep -f biped_run.py       >/dev/null && echo "✅ controller+viewer RUNNING" || { echo "❌ controller DEAD"; tail -5 /tmp/biped_run.log; }
pgrep -f teleop_gui_biped   >/dev/null && echo "✅ gui RUNNING"              || { echo "❌ gui DEAD";        tail -5 /tmp/teleop_gui_biped.log; }
echo "명령채널=$CMD · 종료: pkill -f biped_run.py; pkill -f teleop_gui_biped"
