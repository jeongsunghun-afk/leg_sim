#!/bin/bash
# biped C++ GUI 런처 — C++ 뷰어(biped_view: 컨트롤러+렌더+JSON연동) + 슬림 GUI 동시.
# Python(run_gui_biped.sh)과 동일 GUI·JSON채널, 컨트롤러만 C++. ★자기 터미널에서 실행.
#   기본 = 추정 폐루프(배포경로: leg-odom+접촉높이+지연보상). GT=1 이면 GT 제어(디버그).
#   실측지연 재현: SENSE_LAT_MS/ACT_LAT_MS/LAT_COMP_MS 로 실행.
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=/home/jsh/miniforge3/envs/proxddp/bin/python
ENV=/home/jsh/miniforge3/envs/proxddp
CMD=/tmp/biped_cmd.json
export DISPLAY="${DISPLAY:-:0}"

# ★항상 최신 빌드(추정기·DeployLoop 반영)
echo "빌드 확인 중…"
( cd "$HERE/cpp"; CONDA_PREFIX=$ENV cmake -B build -S . >/dev/null 2>&1; CONDA_PREFIX=$ENV cmake --build build -j4 >/dev/null 2>&1 ) \
  && echo "✅ 빌드 OK" || { echo "❌ 빌드 실패"; exit 1; }

# 모드: 기본=추정 폐루프(배포, 바이너리 기본 ON). GT=1 → GT 제어.
if [ "${GT:-0}" = "1" ]; then EST="GT=1"; MODEDESC="GT 제어(디버그)"; else EST="EST_CTRL=1"; MODEDESC="추정 폐루프(배포: leg-odom+접촉높이+지연보상)"; fi
LATENV="SENSE_LAT_MS=${SENSE_LAT_MS:-0} ACT_LAT_MS=${ACT_LAT_MS:-0} LAT_COMP_MS=${LAT_COMP_MS:-0} NOISE=${NOISE:-0}"
# ★2점 평발 보행(실험 WIP): WALK2=1 반응형·WALK2=zmp event-DCM프리뷰(측방안정). 기본=2점 서기전용.
W2ENV=""; case "${WALK2:-}" in 1) W2ENV="FLAT_WALK=1"; echo "★2점 보행=반응형(WIP·~2.5s 전진 후 리셋)";;
  zmp) W2ENV="FLAT_WALK=1 ZMP_WALK=1"; echo "★2점 보행=ZMP프리뷰(측방안정·전후WIP)";; esac
echo "모드: $MODEDESC"

pkill -f biped_view 2>/dev/null; pkill -f teleop_gui_biped 2>/dev/null; sleep 1
# ★통합모델(biped_flatfoot): 시작=2점 평발 정적 rest. GUI 1점/2점 버튼으로 전환.
echo '{"v":0.0,"vy":0.0,"w":0.0,"body_h":0.42,"mode":"stand","contact":"2pt"}' > "$CMD"

# ① C++ 뷰어 (컨트롤러+렌더+명령소비+상태발행) — 통합모델 biped_flatfoot(heel+toe)
setsid bash -c "cd '$HERE/cpp'; CMDFILE='$CMD' $EST $W2ENV $LATENV DISPLAY='$DISPLAY' LD_LIBRARY_PATH='$ENV/lib' ./build/biped_view ../biped_flatfoot.mjcf > /tmp/biped_view_cpp.log 2>&1" </dev/null &
sleep 2
# ② 슬림 GUI (Python dearpygui)
setsid bash -c "cd '$HERE/../quad'; QUAD_CMD='$CMD' DISPLAY='$DISPLAY' '$PY' teleop_gui_biped.py > /tmp/teleop_gui_biped.log 2>&1" </dev/null &
sleep 2

pgrep -f biped_view     >/dev/null && echo "✅ C++ 뷰어 RUNNING ($MODEDESC)" || { echo "❌ 뷰어 DEAD"; tail -6 /tmp/biped_view_cpp.log; }
pgrep -f teleop_gui_biped >/dev/null && echo "✅ GUI RUNNING"    || { echo "❌ GUI DEAD"; tail -4 /tmp/teleop_gui_biped.log; }
echo "토글: GT=1 ./run_gui_cpp.sh (GT 제어) · 종료: pkill -f biped_view; pkill -f teleop_gui_biped"
