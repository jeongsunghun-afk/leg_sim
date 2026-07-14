#!/bin/bash
# biped 데모 녹화 — GUI(뷰어+조종) + OBS Studio 동시 실행.
# OBS에서 '창 캡처'로 biped 뷰어 창을 선택해 녹화. OBS_AUTOREC=1 이면 자동 녹화 시작.
# ★사용자 자기 터미널에서 실행해야 창이 뜸(디스플레이 연결).
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
export DISPLAY="${DISPLAY:-:0}"

# ① GUI + 뷰어
"$HERE/run_gui_biped.sh"
sleep 2

# ② OBS Studio
pkill -x obs 2>/dev/null; sleep 1
setsid bash -c "DISPLAY='$DISPLAY' obs ${OBS_AUTOREC:+--startrecording --minimize-to-tray}" </dev/null > /tmp/obs.log 2>&1 &
sleep 3
if pgrep -x obs >/dev/null; then
  echo "✅ OBS 실행됨"
  echo "   → OBS에서 [소스 +] → '창 캡처(Window Capture)' → biped 뷰어 창 선택 → 녹화 시작"
  echo "   → 자동 녹화: OBS_AUTOREC=1 ./record_biped.sh (사전 씬 설정 필요)"
else
  echo "❌ OBS 미실행:"; tail -3 /tmp/obs.log
fi
echo "종료: pkill -f biped_run.py; pkill -f teleop_gui_biped; pkill -x obs"
