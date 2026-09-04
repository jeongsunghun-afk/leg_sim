#!/usr/bin/env bash
# run_hw.sh — biped 모드 전환 + 상태 관찰 헬퍼.
#   run_deploy_hw.sh(=producer 소비자, RT 루프)가 **떠 있는 상태**에서 쓴다.
#   cmd/state JSON 을 손으로 안 만지게 감싼다.
#
#   사용:
#     ./run_hw.sh hold                # 현자세 유지(IMU-free, 크레인 내려도 됨)
#     ./run_hw.sh stand               # ★base 피드백 균형 — 크레인 남긴 채!
#     ./run_hw.sh off                 # limp 정지
#     ./run_hw.sh home|float|jog|push|walk
#     ./run_hw.sh gui                 # ★teleop GUI (모드버튼·중력지지 슬라이더·LED) — 테스트는 이걸로
#     ./run_hw.sh watch [N]           # 상태 N회(기본30) 관찰(mode·rpy·tilt·estop)
#     ./run_hw.sh status              # 1회 스냅샷
#
#   ⚠ 순서: 터미널A RobotEmbedded → 터미널B run_deploy_hw.sh → (여기) run_hw.sh
#   ⚠ stand/walk 는 **크레인** 받친 채. 불안정하면 즉시 `./run_hw.sh hold` 또는 off.
set -u
CMD="${QUAD_CMD:-/tmp/biped_cmd.json}"
STATE="${QUAD_STATE:-/tmp/biped_state.json}"

usage(){ echo "사용: $0 {off|hold|stand|home|float|jog|push|walk|gui|watch [N]|status}"; exit 1; }
[ $# -ge 1 ] || usage
HERE="$(cd "$(dirname "$0")" && pwd)"

set_mode(){
  printf '{"mode":"%s","jog_deg":[0,0,0,0,0,0,0,0],"v":0,"vy":0,"w":0,"body_h":0.42}\n' "$1" > "$CMD"
  pgrep -f build/biped_deploy >/dev/null || echo "  ⚠ biped_deploy 안 떠 있음 — run_deploy_hw.sh 먼저"
  echo "→ mode=$1  ($CMD)"
}

case "$1" in
  watch)
    python3 - "$STATE" "${2:-30}" <<'PY' 2>/dev/null
import json,sys,time
path,n=sys.argv[1],int(sys.argv[2])
for _ in range(n):
    try:
        d=json.load(open(path)); r=d['rpy_deg']
        print("mode %-5s  roll %+5.1f pitch %+5.1f tilt %4.1f  estop %s  tiltOK %s"
              %(d['mode'], r[0], r[1], d['tilt_deg'], d['estop'], d['tilt_estop_ok']))
    except Exception as e:
        print("state 못읽음(deploy 떠 있나?):", e)
    time.sleep(0.4)
PY
    ;;
  status)
    python3 - "$STATE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); r=d['rpy_deg']
print("mode=%s  rpy=%s  tilt=%.1f  estop=%s  tiltOK(imu)=%s  loop_hz=%s"
      %(d['mode'],[round(x,1) for x in r],d['tilt_deg'],d['estop'],d['tilt_estop_ok'],d.get('loop_hz')))
PY
    ;;
  gui)
    # teleop GUI — 모드버튼·중력지지 슬라이더·통신 LED. deploy 와 같은 cmd/state 를 공유.
    #   GUI 는 cmd 만 쓴다(모터 writer 아님) → biped_deploy 와 동시 실행 안전.
    #   ⚠ PY 가 GUI 라이브러리를 가진 인터프리터여야 함(run_emb.sh 와 동일). 필요시 PY 지정.
    pgrep -f build/biped_deploy >/dev/null || echo "  ⚠ biped_deploy 안 떠 있음 — run_deploy_hw.sh 먼저"
    PY="${PY:-python3}"
    echo "→ GUI 기동 ($PY · cmd=$CMD state=$STATE)"
    exec env DISPLAY="${DISPLAY:-:0}" QUAD_CMD="$CMD" QUAD_STATE="$STATE" "$PY" "$HERE/teleop_gui_biped.py" ;;
  off|hold|stand|home|float|jog|push|walk)
    case "$1" in stand|walk) echo "⚠ $1 — 크레인 받친 채인지 확인!";; esac
    set_mode "$1" ;;
  *) usage ;;
esac
