#!/bin/bash
# GUI 만 띄운다 (컨트롤러는 따로 실행) — ★실기 JOG 검증용.
#
# 실기 절차:
#   ① 모터전원 ON → Emb 기동 → **5초 대기**
#   ② 터미널 A:  cd ~/simulation/biped/emb && python3 app/biped_emb.py
#   ③ 터미널 B:  cd ~/simulation/biped && ./run_gui_only.sh
#   ④ GUI 에서 Off → JOG 검증 → Home 복귀 (emb/NEXT_HW.md §2~4)
#
# ⚠ **모터 명령 writer 는 한 번에 하나만.** biped_emb.py 와 cpp/build/biped_deploy 를
#   동시에 띄우지 말 것. GUI 는 writer 가 아니라 명령 채널에 JSON 을 쓸 뿐이라 무관하다.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CMD="${QUAD_CMD:-/tmp/biped_cmd.json}"
STATE="${QUAD_STATE:-/tmp/biped_state.json}"
export DISPLAY="${DISPLAY:-:0}"

# dearpygui 가 있는 인터프리터를 고른다(시스템 python 엔 보통 없다)
PY_GUI=""
for p in "$PY" "$HOME/.venvs/gui/bin/python" /home/jsh/miniforge3/envs/proxddp/bin/python \
         "$(command -v python3)"; do
  if [ -n "$p" ] && [ -x "$p" ] && "$p" -c "import dearpygui" 2>/dev/null; then PY_GUI="$p"; break; fi
done
[ -z "$PY_GUI" ] && {
  echo "❌ dearpygui 가 있는 python 을 못 찾았다."
  echo "   설치: python3 -m venv --system-site-packages ~/.venvs/gui && ~/.venvs/gui/bin/pip install dearpygui"
  exit 1; }
echo "gui=$PY_GUI · CMD=$CMD · STATE=$STATE"

pkill -f teleop_gui_biped 2>/dev/null || true
sleep 0.5
# ★명령파일이 없으면 off 로 만들어 둔다. 컨트롤러가 먼저 떠 있어도 무장되지 않게.
[ -f "$CMD" ] || echo '{"v":0.0,"vy":0.0,"w":0.0,"body_h":0.50,"mode":"off","seq":0}' > "$CMD"

cd "$HERE"
QUAD_CMD="$CMD" QUAD_STATE="$STATE" exec "$PY_GUI" teleop_gui_biped.py
