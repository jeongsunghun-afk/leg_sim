#!/bin/bash
# run_hw.sh — ★실기 세션 런처. GUI(입력) + 모니터 뷰어(표시) 를 띄운다.
#
#   ┌ GUI ──cmd.json──> [제어기] ──SHM──> 모터        ← 제어기는 **따로** 띄운다
#   └ 모니터 뷰어 <──state.json── [제어기]            ← 엔코더 표시 전용
#
#   ★sim 런처(run_gui_biped/run_gui_cpp)와 근본적으로 다르다:
#     sim  = 뷰어가 **제어+물리(mj_step)** 를 돌리는 폐루프
#     실기 = 뷰어는 **아무것도 제어하지 않는다**(mj_forward 로 기구학만). 물리는 진짜 로봇이 한다.
#
# 사용 순서:
#   ① 모터전원 ON → Emb 기동 → **5초 대기**
#        cd ~/ZSource/RobotEmbedded/build && ./src/RobotEmbedded
#        ★sudo 없이 실행할 것 (2026-08-11). setcap 이 적용돼 있다
#        (cap_net_admin,cap_net_raw=eip · SHM perms 666 이라 비루트 attach 도 된다).
#        ⚠sudo 로 띄우면 래퍼 2개가 끼어 **Ctrl+C 가 실제 프로세스에 도달하지 않는다**
#          — 죽은 줄 알고 다시 띄우면 중복 writer 가 된다. 게다가 root 소유라
#          일반 사용자가 kill 도 못 한다(실측: sudo=신호무시 / 직접실행=0.11초 종료).
#   ② 제어기 (별도 터미널)   cd ~/simulation/biped/emb && python3 app/biped_emb.py
#   ③ 이 스크립트            cd ~/simulation/biped && ./run_hw.sh
#
# ⚠ **모터 명령 writer 는 한 번에 하나만** — app/biped_emb.py 와 cpp/build/biped_deploy 를
#   동시에 띄우지 말 것. GUI 와 모니터 뷰어는 writer 가 아니므로 무관하다.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
CMD="${QUAD_CMD:-/tmp/biped_cmd.json}"
STATE="${QUAD_STATE:-/tmp/biped_state.json}"
MJ="${BIPED_MJCF:-$HERE/biped_from_quad.mjcf}"      # 배포=점발(§8-g)
CFG="${BIPED_CFG:-$HERE/emb/config/biped_emb.yaml}"

source "$HERE/lib_display.sh"
setup_display || exit 1

# GUI 는 dearpygui 가 있는 인터프리터로
PY_GUI=""
for p in "$PY" "$HOME/.venvs/gui/bin/python" /home/jsh/miniforge3/envs/proxddp/bin/python "$(command -v python3)"; do
  if [ -n "$p" ] && [ -x "$p" ] && "$p" -c "import dearpygui" 2>/dev/null; then PY_GUI="$p"; break; fi
done
[ -z "$PY_GUI" ] && { echo "❌ dearpygui 가 있는 python 이 없다."; \
  echo "   설치: python3 -m venv --system-site-packages ~/.venvs/gui && ~/.venvs/gui/bin/pip install dearpygui"; exit 1; }

VIEW=1
if [ ! -x "$HERE/cpp/build/biped_monitor" ]; then
  echo "⚠ cpp/build/biped_monitor 가 없다 → GUI 만 띄운다."
  echo "   빌드: cd cpp/build && cmake --build . --target biped_monitor -j4"
  VIEW=0
fi
[ "${NOVIEW:-0}" = "1" ] && VIEW=0

echo "gui=$PY_GUI"
echo "모델=$MJ · config=$CFG"
if ! pgrep -f "biped_emb.py|biped_deploy" >/dev/null; then
  echo "⚠ 제어기가 안 보인다 — GUI 를 조작해도 로봇은 안 움직인다."
  echo "   별도 터미널에서:  cd $HERE/emb && python3 app/biped_emb.py"
fi

# ★재시작 대상은 **이 스크립트가 띄우는 것 전부**여야 한다 (2026-08-20 수정).
#   종전엔 monitor_plot.py 가 목록에서 빠져 있었다 — 종료 안내(아래)에는 적혀 있는데
#   정작 여기서 안 죽여서, 다시 띄울 때마다 **그래프 모니터만 중복으로 쌓였다.**
#   중복이 쌓이면 CPU 를 먹고(실측: 뷰어+모니터가 27%) 500Hz 제어루프가 밀린다.
#   ⚠목록을 늘릴 때 이 세 곳을 같이 고칠 것: ①여기 ②MON=graph 종료안내 ③MON=* 종료안내
for _p in teleop_gui_biped biped_monitor monitor_state.py monitor_plot.py; do
  pkill -f "$_p" 2>/dev/null || true
done
sleep 0.5
# ★명령파일이 없으면 off 로 만들어 둔다(제어기가 먼저 떠 있어도 무장되지 않게)
[ -f "$CMD" ] || echo '{"v":0.0,"vy":0.0,"w":0.0,"body_h":0.50,"mode":"off","seq":0}' > "$CMD"

if [ "$VIEW" = "1" ]; then
  setsid bash -c "cd '$HERE/cpp'; STATE='$STATE' CONFIG='$CFG' DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' \
    LD_LIBRARY_PATH='$HOME/mujoco/lib' ./build/biped_monitor '$MJ' > /tmp/biped_monitor.log 2>&1" </dev/null &
  sleep 2
fi
setsid bash -c "cd '$HERE'; QUAD_CMD='$CMD' QUAD_STATE='$STATE' DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' \
  '$PY_GUI' teleop_gui_biped.py > /tmp/teleop_gui_biped.log 2>&1" </dev/null &
sleep 2

[ "$VIEW" = "1" ] && { pgrep -f biped_monitor >/dev/null && echo "✅ 자세 뷰어 RUNNING (3D, 표시 전용)" \
                       || { echo "❌ 자세 뷰어 DEAD"; tail -4 /tmp/biped_monitor.log; }; }
pgrep -f teleop_gui_biped >/dev/null && echo "✅ GUI RUNNING" || { echo "❌ GUI DEAD"; tail -4 /tmp/teleop_gui_biped.log; }

# ── 값 모니터 (측정 vs 명령: 위치·속도·토크) ────────────────────────────────
#   ★뷰어(biped_monitor)는 **자세만** 보여준다. 명령을 얼마나 못 따라오는지,
#     토크가 어디서 커지는지는 안 보인다 — stand/토크제어에선 그게 판단의 핵심이다.
#
#   MON=1|graph (기본) 그래프 창 · MON=text 표(터미널, SSH 용) · MON=0 끔
#   MON_ARGS 로 옵션 — 그래프: "--win 20 --hz 20" · 표: "--ch --hz 20"
#
#   ⚠**x-terminal-emulator 로 창을 띄우지 않는다** (2026-08-13). 그렇게 했다가 안 떴다:
#     ① 이 기기의 그것은 gnome-terminal.wrapper 인데 코드가 `$TE = "gnome-terminal"`
#        문자열만 비교해 폐기된 `-e` 분기로 갔고 ② gnome-terminal 런처는 서버에 붙은 채
#        **반환하지 않으며** ③ 에러를 /dev/null 로 버려 ①②가 조용히 실패했다.
#     에뮬레이터마다 인자 규약이 달라 그 분기는 계속 깨진다 ⇒ 의존을 없앴다.
#     그래프 창은 **teleop GUI 와 같은 경로**(setsid + 명시적 DISPLAY)로 띄운다 — 검증된 방식이다.
case "${MON:-1}" in
  1|graph)   # ★기본 — **그래프 창**. 축을 고르면 그 축의 위치·속도·토크만 그린다.
             #   GUI 와 **같은 방식**으로 띄운다(setsid + 명시적 DISPLAY). 그 경로는 검증돼 있다.
    setsid bash -c "cd '$HERE'; QUAD_STATE='$STATE' DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' \
      '$PY_GUI' monitor_plot.py ${MON_ARGS:-} > /tmp/biped_monitor_plot.log 2>&1" </dev/null &
    sleep 2
    pgrep -f monitor_plot.py >/dev/null && echo "✅ 값 모니터(그래프) RUNNING" \
      || { echo "❌ 값 모니터 DEAD"; tail -6 /tmp/biped_monitor_plot.log; }
    echo "종료: pkill -f teleop_gui_biped; pkill -f biped_monitor; pkill -f monitor_plot.py"
    ;;
  text)      # 표 형태(터미널). SSH 전용이거나 GUI 를 더 못 얹을 때.
    echo ""
    echo " 값 모니터(표) 시작 — Ctrl-C 로 모니터만 종료(뷰어·GUI 는 계속 돈다)"
    sleep 1
    cd "$HERE" && QUAD_STATE="$STATE" exec python3 monitor_state.py ${MON_ARGS:-}
    ;;
  *)
    echo "종료: pkill -f teleop_gui_biped; pkill -f biped_monitor; pkill -f monitor_state.py"
    ;;
esac
