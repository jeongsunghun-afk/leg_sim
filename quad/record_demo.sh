#!/bin/bash
# ============================================================
# 뷰어 + GUI 데모를 NVENC로 화면 녹화 → mkv → mp4
#   ★반드시 "본인 터미널"에서 실행 (Claude 도구 세션에선 프로세스가 회수됨)
#   사용:  bash record_demo.sh
#   종료:  이 터미널에서 [Enter] 를 누르면 녹화 종료 + mp4 생성
# ============================================================
set -e

PY=/home/jsh/miniforge3/envs/proxddp/bin/python
QUAD=/home/jsh/문서/jsh/simulation/quad
OUT_DIR=${OUT_DIR:-$HOME/Videos/Screencasts}
SIZE=${SIZE:-1920x1200}                 # 전체화면 해상도 (xdpyinfo 로 확인)
GAINS="GEARBOX=1 GEAR_FOOT=0.5714"   # ★17dof 게인(W_ORI20·W_AM12·KD_AM24·발목)은 이제 자동감지 기본값 → env 불요
# ★GEARBOX=1 GEAR_FOOT=0.5714 = 발목 반사관성(8:1)→현실적 발끝(이상화 발목 flail 억제). 이상화로 보려면 이 둘 제거.

mkdir -p "$OUT_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MKV="$OUT_DIR/quad_$STAMP.mkv"          # 녹화 원본(크래시 안전)
MP4="$OUT_DIR/quad_$STAMP.mp4"          # 배포용

cleanup(){ kill "$GUI" "$VIEW" "$FF" 2>/dev/null || true; }
trap cleanup EXIT

# 1) GUI + C++ 뷰어 실행 (본인 세션이라 유지됨)
pkill -f trot_view 2>/dev/null || true; pkill -f teleop_gui_17dof 2>/dev/null || true; sleep 1   # 기존 인스턴스 정리(★|| true: 매칭없을때 pkill exit1로 set -e 조기종료 방지)
rm -f /tmp/quad_cmd.json
( cd "$QUAD"      && DISPLAY=:0 QUAD_CMD=/tmp/quad_cmd.json "$PY" teleop_gui_17dof.py ) &
GUI=$!
sleep 3
( cd "$QUAD/cpp" && env DISPLAY=:0 $GAINS RATE=1.0 CMDFILE=/tmp/quad_cmd.json \
    ./build/trot_view ../quad_real_17dof_waist_sphere.mjcf ) &   # ★허리 능동모델(조향스파인). 고정허리는 quad_real_17dof_sphere.mjcf
VIEW=$!
sleep 2

echo "============================================================"
echo " ▶ 3초 후 녹화 시작. GUI에서 Walk/trot·선회·허리조향 슬라이더 등을 조작하세요."
echo "   녹화 종료 = 이 터미널에서 [Enter]"
echo "============================================================"
sleep 3

# 2) NVENC 화면 녹화 (전체화면). mkv = 도중 종료돼도 안 깨짐
ffmpeg -y -f x11grab -framerate 30 -video_size "$SIZE" -i :0.0 \
       -c:v h264_nvenc -preset p5 -cq 20 -pix_fmt yuv420p "$MKV" </dev/null &
FF=$!

read -r _         # Enter 누르면 아래로
kill -INT "$FF" 2>/dev/null; wait "$FF" 2>/dev/null || true

# 3) mkv → mp4 remux (재인코딩 없음, 빠름·무손실)
ffmpeg -y -i "$MKV" -c copy "$MP4" </dev/null
echo ""
echo "✅ 저장 완료:"
echo "   원본 : $MKV"
echo "   배포 : $MP4"
echo ""
echo "배속 예) ffmpeg -i \"$MP4\" -filter:v \"setpts=0.5*PTS\" \"${MP4%.mp4}_2x.mp4\""
