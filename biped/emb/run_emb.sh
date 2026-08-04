#!/bin/bash
# run_emb.sh — biped 실기(Emb) 배포 런처. app(RT 루프) + GUI(teleop_emb) 동시 실행.
#   ★작업디렉토리 무관: 어디서든 절대경로로 호출 가능.
#     /home/jsh/문서/jsh/simulation/biped/emb/run_emb.sh
#
#   MOCK=1 [MOCK_CONNECTED="1,2"] <위 경로>   # 데스크톱 데모(SHM 없이 GUI 검증, 부분연결 시뮬)
#   <위 경로>                                  # 실기(Pi): ShmBackend(libbipedshm.so). 없으면 자동 mock 폴백.
set +e
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"     # 스크립트(emb/) 절대경로
PY="${PY:-/home/jsh/miniforge3/envs/proxddp/bin/python}"
CMD="${QUAD_CMD:-/tmp/biped_cmd.json}"
STATE="${QUAD_STATE:-/tmp/biped_state.json}"
export DISPLAY="${DISPLAY:-:0}"

# 실행 옵션 → 인라인 env 문자열(app 자식 프로세스에 명시 전달)
ENVSTR="QUAD_CMD='$CMD' QUAD_STATE='$STATE'"
[ "${MOCK:-0}" = "1" ] && ENVSTR="$ENVSTR MOCK=1" && echo "★MOCK 모드(데스크톱 데모, SHM 없음)"
[ -n "${MOCK_CONNECTED:-}" ] && ENVSTR="$ENVSTR MOCK_CONNECTED='$MOCK_CONNECTED'" \
    && echo "★부분연결 시뮬: 채널 [$MOCK_CONNECTED] 만 통신연결(LED 초록)"

pkill -f "app/biped_emb.py" 2>/dev/null; pkill -f "teleop_gui_biped.py" 2>/dev/null; sleep 1
echo '{"mode":"off","jog_deg":[0,0,0,0,0,0,0,0],"v":0,"vy":0,"w":0,"body_h":0.42}' > "$CMD"

# ① app (RT 루프: read→매핑→모드→write, 상태발행)
setsid bash -c "cd '$HERE'; $ENVSTR '$PY' app/biped_emb.py > /tmp/biped_emb_app.log 2>&1" </dev/null &
sleep 2
# ② GUI (통합 biped GUI: 조이스틱 + 각축 JOG 슬라이더 + 통신 LED). biped/teleop_gui_biped.py 사용.
setsid bash -c "cd '$HERE'; QUAD_CMD='$CMD' QUAD_STATE='$STATE' DISPLAY='$DISPLAY' '$PY' ../teleop_gui_biped.py > /tmp/biped_emb_gui.log 2>&1" </dev/null &
sleep 2

pgrep -f "app/biped_emb.py"    >/dev/null && echo "✅ app RUNNING"  || { echo "❌ app DEAD";  tail -8 /tmp/biped_emb_app.log; }
pgrep -f "teleop_gui_biped.py" >/dev/null && echo "✅ GUI RUNNING"  || { echo "❌ GUI DEAD";  tail -8 /tmp/biped_emb_gui.log; }
echo "명령채널=$CMD · 로그=/tmp/biped_emb_app.log,/tmp/biped_emb_gui.log"
echo "종료: pkill -f app/biped_emb.py; pkill -f teleop_gui_biped.py"
