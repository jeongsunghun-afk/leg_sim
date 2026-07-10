#!/bin/bash
# ============================================================
# 뷰어 + GUI 데모를 NVENC로 화면 녹화 → mp4
#   ★반드시 "본인 터미널"에서 실행 (Claude 도구 세션에선 프로세스가 회수됨)
#   사용:  bash record_demo.sh [map]      map=course(기본)|flat|stairs|rough|friction|gap|stepping|soft
#   종료:  이 터미널에서 [Enter] → 녹화 종료 + mp4 저장
#
# ★견고화(2026-07-10):
#   ①뷰어+GUI 기동을 run_gui.sh 재사용 = 렌더검증+자동재시도(간헐 GL크래시 방지, 단일 소스).
#   ②SIZE 자동감지(xdpyinfo 현재 해상도) — 지정 안 해도 화면 전체 캡처. SIZE=... 로 오버라이드.
# ============================================================
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # simulation/quad
export DISPLAY="${DISPLAY:-:0}"
OUT_DIR=${OUT_DIR:-$HOME/Videos/Screencasts}
MAP=${1:-course}                                          # run_gui.sh 맵 인자
# 캡처 해상도: 미지정 시 현재 화면 전체(xdpyinfo) 자동감지 → 실패 시 1920x1080
SIZE=${SIZE:-$(DISPLAY="$DISPLAY" xdpyinfo 2>/dev/null | awk '/dimensions:/{print $2; exit}')}
SIZE=${SIZE:-1920x1080}

mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MKV="$OUT_DIR/quad_$STAMP.mkv"          # 녹화 원본(크래시 안전)
MP4="$OUT_DIR/quad_$STAMP.mp4"          # 배포용

FF=""
cleanup(){
  [ -n "$FF" ] && kill "$FF" 2>/dev/null
  pkill -TERM -f 'build/trot_view'   2>/dev/null
  pkill -TERM -f teleop_gui_17dof.py 2>/dev/null
}
trap cleanup EXIT

# 1) 뷰어 + GUI = 견고화된 run_gui.sh 재사용(렌더 검증 + 실패 시 자동 재시도)
echo "▶ 뷰어+GUI 기동 (run_gui.sh $MAP)…"
if ! bash "$HERE/run_gui.sh" "$MAP"; then
  echo "❌ 뷰어/GUI 기동 실패 — 중단"; exit 1
fi

echo "============================================================"
echo " ▶ SIZE=$SIZE · 3초 후 녹화 시작. GUI에서 Walk/trot·선회·앉기 등 조작하세요."
echo "   녹화 종료 = 이 터미널에서 [Enter]"
echo "============================================================"
sleep 3

# 2) NVENC 화면 녹화 (전체화면). mkv = 도중 종료돼도 안 깨짐
ffmpeg -y -f x11grab -framerate 30 -video_size "$SIZE" -i :0.0 \
       -c:v h264_nvenc -preset p5 -cq 20 -pix_fmt yuv420p "$MKV" </dev/null &
FF=$!

read -r _         # Enter 누르면 아래로
kill -INT "$FF" 2>/dev/null; wait "$FF" 2>/dev/null || true
FF=""

# 3) mkv → mp4 remux (재인코딩 없음, 빠름·무손실)
ffmpeg -y -i "$MKV" -c copy "$MP4" </dev/null && rm -f "$MKV"
echo ""
echo "✅ 저장 완료: $MP4"
echo "배속 예) ffmpeg -i \"$MP4\" -filter:v \"setpts=0.5*PTS\" \"${MP4%.mp4}_2x.mp4\""
