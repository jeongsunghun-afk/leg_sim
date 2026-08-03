#!/bin/bash
# full-TAMOLS injection 뷰어 + GUI 배선 실행.
#   뷰어(trot_view)=인젝션 config로 실행, GUI(teleop_gui_17dof.py)가 /tmp/quad_cmd.json으로 구동.
#   GUI에서 'TAMOLS 인젝션' 버튼 → mode=tamols → full-TAMOLS injection 보행(평지 V≤0.4).
#   'Ready 서기'=정지·'Run 이동'=A 보행(비교용)과 토글 가능.
set -e
QUAD=/home/jsh/문서/jsh/simulation/quad
CPP=$QUAD/cpp
MJCF=${MJCF:-$QUAD/mjcf/quad_real_17dof_waist_sphere.mjcf}   # 기본=평지. MJCF=... 로 지형 지정 가능

# ── 인젝션 config(검증된 평지 V≤0.4 최적) ──
export OMP_NUM_THREADS=1
export HQP=1 GIAC_FIX=1 TAM_MPC=1 TAM_CLEANV=1 REPLAN_DT=0.02   # strict HQP + GIAC수정 + MPC 예측GRF + 50Hz 재풀이
export PHASE_DUR=0.08 SW_DUR=0.06 STEP_H=0.08                    # 빠른 cadence(발 따라잡기)
export HQP_BASE_L1=1 ORI_KP=650 ORI_KD=70 KP_Z=350 KD_Z=35 W_BASE_XY=200 KP_BASE=120
export TAM_KCAP=0.6 TAM_RAI=0.8 TAM_TST=0.8 SW_KP=700 SW_KD=35 KD_BASE=80   # 발배치 보폭(TST=고속안정 핵심)
export CMDFILE=/tmp/quad_cmd.json STATE_PUB=/tmp/quad_state.json
# 지형 검증 시(선택): TERRAIN=1 ./run_tamols_gui.sh  → TAMOLS_TERRAIN+FOOT_SNAP 켬(미성숙, 험지=A우세)
[ -n "$TERRAIN" ] && export TAMOLS_TERRAIN=1 FOOT_SNAP=1

echo "[run] 뷰어(trot_view) 실행 — 인젝션 config. GUI에서 'TAMOLS 인젝션' 버튼 눌러 구동."
cd "$CPP"
./build/trot_view "$MJCF" &
VIEW_PID=$!
sleep 1.5
echo "[run] GUI(teleop_gui_17dof.py) 실행."
cd "$QUAD"
python3 teleop_gui_17dof.py || true
kill $VIEW_PID 2>/dev/null || true
