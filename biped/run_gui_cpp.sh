#!/bin/bash
# biped C++ GUI 런처 — C++ 뷰어(biped_view: 컨트롤러+렌더+JSON연동) + 슬림 GUI 동시.
# Python(run_gui_biped.sh)과 동일 GUI·JSON채널, 컨트롤러만 C++. ★자기 터미널에서 실행.
#   기본 = 추정 폐루프(배포경로: leg-odom+접촉높이+지연보상). GT=1 이면 GT 제어(디버그).
#   실측지연 재현: SENSE_LAT_MS/ACT_LAT_MS/LAT_COMP_MS 로 실행.
set +e
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV="${CONDA_PREFIX:-/home/jsh/miniforge3/envs/proxddp}"   # 노트북 conda env(빌드용). Pi 는 아래 폴백
CMD="${QUAD_CMD:-/tmp/biped_cmd.json}"
# ★2026-08-07: DISPLAY 강제(:0) 제거 → 자동감지. Wayland 는 mutter 전용 쿠키가 필요하고
#   그냥 :0 을 주면 X11 인증 실패로 죽는다(run_gui_biped.sh 와 같은 수정).
source "$HERE/lib_display.sh"
setup_display || exit 1
warn_if_no_viewer
# GUI 는 dearpygui 가 있는 인터프리터로 (Pi 는 ~/.venvs/gui, 노트북은 proxddp)
PY=""
for p in "$PYBIN" "$HOME/.venvs/gui/bin/python" "$ENV/bin/python" "$(command -v python3)"; do
  if [ -n "$p" ] && [ -x "$p" ] && "$p" -c "import dearpygui" 2>/dev/null; then PY="$p"; break; fi
done
[ -z "$PY" ] && { echo "❌ dearpygui 가 있는 python 이 없다."; \
  echo "   설치: python3 -m venv --system-site-packages ~/.venvs/gui && ~/.venvs/gui/bin/pip install dearpygui"; exit 1; }
echo "gui=$PY"

# ★항상 최신 빌드(추정기·DeployLoop 반영)
echo "빌드 확인 중…"
# ★Pi 는 conda env 가 없다 — MuJoCo 를 ~/mujoco 에서 찾는다. 기존 build 캐시가 있으면 그대로 쓴다.
MJLIB="$HOME/mujoco/lib"
( cd "$HERE/cpp"; CONDA_PREFIX=$ENV cmake --build build -j4 >/dev/null 2>&1 ) \
  && echo "✅ 빌드 OK" || { echo "⚠ 빌드 실패/생략 — 기존 바이너리로 진행"; }
[ -x "$HERE/cpp/build/biped_view" ] || { echo "❌ cpp/build/biped_view 가 없다. 먼저 빌드할 것."; exit 1; }

# 모드: 기본=추정 폐루프(배포, 바이너리 기본 ON). GT=1 → GT 제어.
if [ "${GT:-0}" = "1" ]; then EST="GT=1"; MODEDESC="GT 제어(디버그)"; else EST="EST_CTRL=1"; MODEDESC="추정 폐루프(배포: leg-odom+접촉높이+지연보상)"; fi
LATENV="SENSE_LAT_MS=${SENSE_LAT_MS:-0} ACT_LAT_MS=${ACT_LAT_MS:-0} LAT_COMP_MS=${LAT_COMP_MS:-0} NOISE=${NOISE:-0}"
# ★2점 평발 보행(실험 WIP): WALK2=1 반응형·WALK2=zmp event-DCM프리뷰(측방안정). 기본=2점 서기전용.
W2ENV=""; case "${WALK2:-}" in 1) W2ENV="FLAT_WALK=1"; echo "★2점 보행=반응형(WIP·~2.5s 전진 후 리셋)";;
  zmp) W2ENV="FLAT_WALK=1 ZMP_WALK=1"; echo "★2점 보행=ZMP프리뷰(측방안정·전후WIP)";; esac
echo "모드: $MODEDESC"

pkill -f biped_view 2>/dev/null; pkill -f teleop_gui_biped 2>/dev/null; sleep 1
# ★통합모델(biped_flatfoot): 시작=2점 평발 정적 rest. GUI 1점/2점 버튼으로 전환.
# ★mode 는 off 로 시작한다. stand 면 뷰어가 기동 즉시 무장한다(run_gui_biped.sh 와 동일 이유).
echo '{"v":0.0,"vy":0.0,"w":0.0,"body_h":0.50,"mode":"off","contact":"1pt","seq":0}' > "$CMD"

# ① C++ 뷰어 (컨트롤러+렌더+명령소비+상태발행) — 통합모델 biped_flatfoot(heel+toe)
# ★MJCF: 기본을 **점발 biped_from_quad(=배포 모델)** 로 바꿨다. 평발은 MJCF=flat 로.
#   종전 기본이 biped_flatfoot 이라 배포와 다른 모델로 보고 있었다(§8-g).
MJ="${BIPED_MJCF:-../biped_from_quad.mjcf}"; [ "${MJCF:-}" = "flat" ] && MJ=../biped_flatfoot.mjcf
echo "모델: $MJ"
setsid bash -c "cd '$HERE/cpp'; CMDFILE='$CMD' $EST $W2ENV $LATENV DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' LD_LIBRARY_PATH='${MJLIB}:$ENV/lib' ./build/biped_view $MJ > /tmp/biped_view_cpp.log 2>&1" </dev/null &
sleep 2
# ② 슬림 GUI (Python dearpygui)
# ★GUI 는 biped/ 에 있다(커밋 7953c5c 에서 quad/ 로부터 이동). 종전 ../quad 는 죽은 경로였다.
setsid bash -c "cd '$HERE'; QUAD_CMD='$CMD' DISPLAY='$DISPLAY' XAUTHORITY='$XAUTHORITY' '$PY' teleop_gui_biped.py > /tmp/teleop_gui_biped.log 2>&1" </dev/null &
sleep 2

pgrep -f biped_view     >/dev/null && echo "✅ C++ 뷰어 RUNNING ($MODEDESC)" || { echo "❌ 뷰어 DEAD"; tail -6 /tmp/biped_view_cpp.log; }
pgrep -f teleop_gui_biped >/dev/null && echo "✅ GUI RUNNING"    || { echo "❌ GUI DEAD"; tail -4 /tmp/teleop_gui_biped.log; }
echo "토글: GT=1 ./run_gui_cpp.sh (GT 제어) · 종료: pkill -f biped_view; pkill -f teleop_gui_biped"
