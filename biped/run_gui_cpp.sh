#!/bin/bash
# biped C++ GUI 런처 — C++ 뷰어(biped_view: 컨트롤러+렌더+JSON연동) + 슬림 GUI 동시.
# Python(run_gui_biped.sh)과 동일 GUI·JSON채널, 컨트롤러만 C++. ★자기 터미널에서 실행.
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jsh/miniforge3/envs/proxddp/bin/python
ENV=/home/jsh/miniforge3/envs/proxddp
CMD=/tmp/biped_cmd.json
export DISPLAY="${DISPLAY:-:0}"

if [ ! -x "$HERE/cpp/build/biped_view" ]; then
  echo "biped_view 미빌드 → 빌드 중…"
  ( cd "$HERE/cpp"; CONDA_PREFIX=$ENV cmake -B build -S . >/dev/null 2>&1 && CONDA_PREFIX=$ENV cmake --build build -j4 >/dev/null 2>&1 )
fi
pkill -f biped_view 2>/dev/null; pkill -f teleop_gui_biped 2>/dev/null; sleep 1
echo '{"v":0.0,"vy":0.0,"w":0.0,"body_h":0.483,"mode":"stand"}' > "$CMD"

# ① C++ 뷰어 (컨트롤러+렌더+명령소비+상태발행)
setsid bash -c "cd '$HERE/cpp'; CMDFILE='$CMD' DISPLAY='$DISPLAY' LD_LIBRARY_PATH='$ENV/lib' ./build/biped_view ../biped_from_quad.mjcf > /tmp/biped_view_cpp.log 2>&1" </dev/null &
sleep 2
# ② 슬림 GUI (Python dearpygui)
setsid bash -c "cd '$HERE/../quad'; QUAD_CMD='$CMD' DISPLAY='$DISPLAY' '$PY' teleop_gui_biped.py > /tmp/teleop_gui_biped.log 2>&1" </dev/null &
sleep 2

pgrep -f biped_view     >/dev/null && echo "✅ C++ 뷰어 RUNNING" || { echo "❌ 뷰어 DEAD"; tail -4 /tmp/biped_view_cpp.log; }
pgrep -f teleop_gui_biped >/dev/null && echo "✅ GUI RUNNING"    || { echo "❌ GUI DEAD"; tail -4 /tmp/teleop_gui_biped.log; }
echo "종료: pkill -f biped_view; pkill -f teleop_gui_biped"
