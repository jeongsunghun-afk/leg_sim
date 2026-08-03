#!/usr/bin/env bash
# full-TAMOLS injection 뷰어 + GUI 런처 (run_gui.sh 기반 + 인젝션 config 주입).
#   사용: bash run_tamols_gui.sh [map]   map=flat(기본)|course|stairs|gap|stepping|rough
#   GUI에서 'TAMOLS 인젝션' 버튼 → mode=tamols → full-TAMOLS injection 보행(평지 V≤0.4).
#   'Ready 서기'=정지 · 'Run 이동'=A 보행(비교) 토글. Walk Speed 게이지로 속도(≤0.4).
#   지형(gap/stepping/계단)=미성숙(A우세), TERRAIN=1 하면 TAMOLS_TERRAIN+FOOT_SNAP 켬.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; CPP="$HERE/cpp"
PXI=/home/jsh/miniforge3/envs/proxddp/bin/python
export DISPLAY="${DISPLAY:-:0}"
CMD=/tmp/quad_cmd.json; STATE=/tmp/quad_state.json

case "${1:-flat}" in
  flat)     MJCF=mjcf/quad_real_17dof_waist_sphere.mjcf ;;
  course)   MJCF=mjcf/quad_terrain_course.mjcf ;;
  stairs)   MJCF=mjcf/quad_terrain_stairs.mjcf ;;
  gap)      MJCF=mjcf/quad_terrain_gap.mjcf ;;
  stepping) MJCF=mjcf/quad_terrain_stepping.mjcf ;;
  rough)    MJCF=mjcf/quad_terrain_rough.mjcf ;;
  *)        MJCF="$1" ;;
esac

# ── 인젝션 config(검증된 평지 V≤0.4 최적) — trot_view launch env ──
INJ="OMP_NUM_THREADS=1 RSL_ONLINE=1 HQP=1 GIAC_FIX=1 TAM_MPC=1 TAM_CLEANV=1 REPLAN_DT=0.02 \
PHASE_DUR=0.08 SW_DUR=0.06 STEP_H=0.08 HQP_BASE_L1=1 \
ORI_KP=650 ORI_KD=70 KP_Z=350 KD_Z=35 W_BASE_XY=200 KP_BASE=120 \
TAM_KCAP=0.6 TAM_RAI=0.8 TAM_TST=0.8 SW_KP=700 SW_KD=35 KD_BASE=80"
[ -n "${TERRAIN:-}" ] && INJ="$INJ TAMOLS_TERRAIN=1 FOOT_SNAP=1"

# ── ① 이전 인스턴스 부드럽게 종료 ──
pkill -TERM -f 'build/trot_view' 2>/dev/null; pkill -TERM -f teleop_gui_17dof.py 2>/dev/null
for _ in $(seq 1 25); do pgrep -f 'build/trot_view' >/dev/null || pgrep -f teleop_gui_17dof.py >/dev/null || break; sleep 0.2; done
pkill -9 -f 'build/trot_view' 2>/dev/null; pkill -9 -f teleop_gui_17dof.py 2>/dev/null
sleep 0.5; rm -f "$CMD" "$STATE"

viewer_alive(){ pgrep -f 'build/trot_view ' >/dev/null; }
viewer_rendering(){ local a b; a=$(stat -c %Y.%N "$STATE" 2>/dev/null) || return 1; sleep 0.8; b=$(stat -c %Y.%N "$STATE" 2>/dev/null) || return 1; [ "$a" != "$b" ]; }

# ── ② 뷰어: 인젝션 config로 실행, 렌더 확인될 때까지 재시도 ──
VOK=0
for try in 1 2 3 4; do
  rm -f "$STATE"
  setsid bash -c "cd '$CPP'; env $INJ RATE=1.0 CMDFILE='$CMD' STATE_PUB='$STATE' \
    ./build/trot_view '../$MJCF' > /tmp/trot_view.log 2>&1" </dev/null &
  for w in $(seq 1 8); do sleep 1
    if ! viewer_alive; then echo "viewer 조기종료(try $try):"; tail -4 /tmp/trot_view.log 2>/dev/null; break; fi
    if viewer_rendering; then VOK=1; break; fi
  done
  [ $VOK = 1 ] && { echo "viewer RUNNING (try $try)"; break; }
  echo "viewer 재시도 $try…"; pkill -9 -f 'build/trot_view' 2>/dev/null; sleep 1
done
[ $VOK = 0 ] && { echo "viewer DEAD"; tail -5 /tmp/trot_view.log 2>/dev/null; exit 1; }

# ── ③ GUI ──
setsid bash -c "cd '$HERE'; DISPLAY='$DISPLAY' '$PXI' teleop_gui_17dof.py > /tmp/teleop_gui.log 2>&1" </dev/null &
sleep 4
echo "map=$MJCF"
pgrep -f teleop_gui_17dof.py >/dev/null && echo "gui RUNNING" || { echo "gui DEAD"; tail -5 /tmp/teleop_gui.log; }
echo "✅ 준비 완료 — GUI에서 'TAMOLS 인젝션' 버튼 → Walk Speed로 전진(≤0.4)"
