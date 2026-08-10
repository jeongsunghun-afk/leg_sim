#!/bin/bash
# biped GUI 원샷 런처 — 컨트롤러+뷰어(biped_run) + 슬림 GUI(teleop_gui_biped) 동시 실행.
# JSON 채널(/tmp/biped_cmd.json)로 디커플.
# 사용: ./run_gui_biped.sh   (뷰어 창 + GUI 창이 뜸. 자기 터미널에서 실행해야 디스플레이 연결)
#
# ★2026-08-07 수정 — 두 군데가 stale 이라 **GUI 가 아예 안 떴다**:
#   (a) GUI 를 `cd ../quad` 에서 실행했다. 커밋 7953c5c 가 파일을 quad/ → biped/ 로 옮겼는데
#       (R085) 런처를 안 고쳐서 없는 경로를 가리키고 있었다.
#   (b) PY 가 노트북 절대경로(/home/jsh/miniforge3/...)로 하드코딩돼 Pi 에선 무조건 실패.
#   ⇒ 경로를 $HERE 로 바로잡고, python 은 자동탐색 + PY env 오버라이드로 바꿨다.
#
# ⚠ 이 런처는 **sim 용**이다(biped_run.py = MuJoCo 컨트롤러+뷰어).
#   실기는 emb/app/biped_emb.py(off·jog·home·hold) 또는 cpp/build/biped_deploy(stand·walk)를
#   직접 띄우고, GUI 만 따로 실행할 것 → `run_gui_only.sh` 참조.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CMD="${QUAD_CMD:-/tmp/biped_cmd.json}"
source "$HERE/lib_display.sh"
setup_display || exit 1

# ★python 자동탐색 — **컨트롤러와 GUI 는 서로 다른 인터프리터일 수 있다.**
#   Pi 에서는 실제로 갈려 있다: ~/.venvs/mj = mujoco·qpsolvers(sim) / ~/.venvs/gui = dearpygui.
#   그래서 "실행 가능한 첫 python" 을 고르면 안 되고 **필요한 모듈이 있는지로** 골라야 한다.
pick_py() {                       # $1=검사할 모듈. set -e 대비로 if 사용
  local mod="$1" p
  for p in "$PY" "$HOME/.venvs/mj/bin/python" "$HOME/.venvs/gui/bin/python" \
           /home/jsh/miniforge3/envs/proxddp/bin/python "$(command -v python3)"; do
    if [ -n "$p" ] && [ -x "$p" ] && "$p" -c "import $mod" 2>/dev/null; then echo "$p"; return; fi
  done
  echo ""; }
PY_RUN="$(pick_py mujoco)"        # 컨트롤러+뷰어: mujoco 필요
PY_GUI="$(pick_py dearpygui)"     # GUI: dearpygui 필요
[ -z "$PY_RUN" ] && { echo "❌ mujoco 가 있는 python 이 없다 (sim 컨트롤러 실행 불가)."; \
  echo "   Pi: ~/.venvs/mj 확인 · 노트북: proxddp env 확인 · 또는 PY=/경로/python"; exit 1; }
[ -z "$PY_GUI" ] && { echo "❌ dearpygui 가 있는 python 이 없다."; \
  echo "   설치: python3 -m venv --system-site-packages ~/.venvs/gui && ~/.venvs/gui/bin/pip install dearpygui"; exit 1; }
# ★sim 뷰어가 뜰 수 있는 디스플레이인지 미리 본다(경고만 — 0x0 은 모니터 절전 등
#   일시적일 수 있어 막지 않는다). 안 그러면 컨트롤러만 조용히 죽어 원인을 못 찾는다.
warn_if_no_viewer
echo "controller=$PY_RUN"
echo "gui=$PY_GUI"

pkill -f biped_run.py 2>/dev/null || true
pkill -f teleop_gui_biped 2>/dev/null || true
sleep 1

# 초기 명령 — ★mode 는 반드시 'off'. 'stand' 로 두면 컨트롤러가 기동 즉시 무장한다
#   (사람이 버튼을 누르기도 전에 모터에 전류가 흐르는 것. teleop_gui_biped 의 Pub 도 같은 이유로 off 시작)
echo '{"v":0.0,"vy":0.0,"w":0.0,"body_h":0.50,"mode":"off","contact":"1pt","seq":0}' > "$CMD"

# ① 컨트롤러 + 뷰어
# ★BIPED_MJCF 는 **값이 있을 때만** 넘긴다. 빈 문자열을 넘기면
#   biped_run.py 의 os.environ.get('BIPED_MJCF', UNIFIED) 가 '' 를 "설정됨"으로 받아
#   기본 MJCF 를 무시하고 빈 경로를 연다 → ParseXML: Error opening file ''
MJENV=""; [ -n "${BIPED_MJCF:-}" ] && MJENV="BIPED_MJCF='$BIPED_MJCF'"
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' QUAD_CMD='$CMD' $MJENV '$PY_RUN' biped_run.py > /tmp/biped_run.log 2>&1" </dev/null &
sleep 2
# ② 슬림 GUI (★biped/ 에 위치. 7953c5c 에서 quad/ 로부터 이동)
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' QUAD_CMD='$CMD' '$PY_GUI' teleop_gui_biped.py > /tmp/teleop_gui_biped.log 2>&1" </dev/null &
sleep 2

pgrep -f biped_run.py       >/dev/null && echo "✅ controller+viewer RUNNING" || { echo "❌ controller DEAD"; tail -5 /tmp/biped_run.log; }
pgrep -f teleop_gui_biped   >/dev/null && echo "✅ gui RUNNING"              || { echo "❌ gui DEAD";        tail -5 /tmp/teleop_gui_biped.log; }
echo "명령채널=$CMD · 종료: pkill -f biped_run.py; pkill -f teleop_gui_biped"
