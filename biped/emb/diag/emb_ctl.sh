#!/usr/bin/env bash
# emb_ctl.sh — RobotEmbedded(Emb) 기동/종료/로그 헬퍼. 진단 절차를 한 곳에 고정.
#   Emb 는 root 필요(EtherCAT raw socket / SPI). sudoers 미설치면 암호를 물어본다.
#
#   ./emb_ctl.sh start   # 기동(로그: /tmp/emb.log) 후 halGait 초기화(≈5s) + 신선도 확인
#   ./emb_ctl.sh stop    # 종료
#   ./emb_ctl.sh log     # EtherCAT 요약(슬레이브/OP/WKC) + 마지막 로그
#   ./emb_ctl.sh status  # 프로세스·SHM·프레임 유통 상태
#
# ★Emb 는 EtherCAT 이 OP 를 잃으면 스스로 복구하지 못한다(commEtherCATm.cpp:520 이 조기 return
#   하여 복구 경로인 CheckState 에 영영 도달하지 못함). 유일한 복구는 이 스크립트의 stop→start.
set -u

EMB_DIR=/home/rpetubt/ZSource/RobotEmbedded/build
EMB_BIN=$EMB_DIR/src/RobotEmbedded
LOG=/tmp/emb.log
DIAG=$(cd "$(dirname "$0")" && pwd)

running(){ pgrep -x RobotEmbedded >/dev/null 2>&1; }

# ★중복 기동은 EtherCAT 버스를 깬다. 어떤 명령이든 먼저 알린다.
_n=$(pgrep -cx RobotEmbedded 2>/dev/null || echo 0)
if [ "$_n" -gt 1 ] 2>/dev/null; then
    echo "⚠⚠ RobotEmbedded 가 **$_n 개** 떠 있다 — EtherCAT 마스터가 여럿이면 버스가 깨진다."
    echo "   pid: $(pgrep -x RobotEmbedded | tr '\n' ' ')"
    echo "   'diag/emb_ctl.sh stop' 으로 전부 정리한 뒤 다시 시작할 것."
fi

case "${1:-status}" in
start)
    if running; then echo "이미 실행 중 (pid $(pgrep -x RobotEmbedded | tr '\n' ' ')) — 모터 명령 writer 는 하나만."; exit 1; fi
    if pgrep -f "biped_emb.py|RobotTestGait|mot_test" >/dev/null 2>&1; then
        echo "다른 writer(app/RobotTestGait/mot_test)가 실행 중 — 먼저 종료할 것."; exit 1
    fi
    echo "[emb_ctl] 기동 → $LOG"
    echo "⚠ 기동 직후 Emb 가 자체적으로 전 관절을 4.5초에 걸쳐 0°로 램프한다(Kp=20/Kd=5). 주변 확인!"
    # ★sudoers 규칙은 "정확히 이 바이너리 경로"만 NOPASSWD 이므로 sh -c 로 감싸지 말 것.
    #   리다이렉션은 호출측(비root) 셸이 수행 → 로그 파일 소유자는 rpetubt.
    ( cd "$EMB_DIR" && sudo "$EMB_BIN" > "$LOG" 2>&1 & )
    sleep 2
    if ! running; then echo "✗ 기동 실패:"; tail -30 "$LOG" 2>/dev/null; exit 1; fi
    echo "[emb_ctl] pid $(pgrep -x RobotEmbedded | tr '\n' ' ') — halGait 초기화 대기(≈5s)"
    # ★플래그만으로는 부족하다. stt_probe 의 "값이 갱신됨" 판정(신선도)까지 확인한다.
    for i in $(seq 1 20); do
        sleep 1
        if "$DIAG/stt_probe" 6 2>/dev/null | grep -q "값이 갱신됨"; then
            echo "✓ MotorStatus16 신선 — EtherCAT 생존 + Emb 가 SHM 명령을 읽는 상태 (${i}s)"; exit 0
        fi
    done
    # ★실패했으면 **띄운 것을 반드시 정리한다** (2026-08-12).
    #   종전엔 그냥 exit 2 라 죽은 Emb 가 계속 떠 있었다. 사용자가 수동으로 다시 띄우면
    #   **EtherCAT 마스터가 둘**이 되어 버스를 서로 물어뜯는다 — 실제로 4개까지 늘어나
    #   전 채널이 얼어붙었다. 신선하지 않은 Emb 는 남겨둘 가치가 없다(모터 명령은 계속
    #   재전송하면서 상태는 못 읽는 상태다).
    echo "✗ 20s 내 신선한 MotorStatus16 미수신."
    echo "  → 띄운 Emb 를 정리한다(그냥 두면 재시도 시 EtherCAT 마스터가 둘이 된다)."
    sudo pkill -x RobotEmbedded 2>/dev/null; sleep 1
    n=$(pgrep -cx RobotEmbedded 2>/dev/null || echo 0)
    echo "  남은 RobotEmbedded: $n"
    "$0" log
    echo
    echo "  다음 순서로 복구할 것:"
    echo "    ① 모터 전원 OFF → 3초 → ON   (EtherCAT 슬레이브 초기화. Emb 재기동만으론 부족)"
    echo "    ② diag/emb_ctl.sh start"
    echo "  ⚠수동으로 RobotEmbedded 를 따로 띄우지 말 것 — 중복 기동이 버스를 깬다."
    exit 2
    ;;
stop)
    sudo pkill -x RobotEmbedded 2>/dev/null
    sleep 1
    running && { echo "강제 종료"; sudo pkill -9 -x RobotEmbedded; }
    echo "[emb_ctl] 종료됨"
    ;;
log)
    echo "=== EtherCAT 요약 ==="
    grep -aE "EtherCAT|slave|Slave|WKC|OP|SAFEOP|mismatch|Failed|failure|lost|recover" "$LOG" 2>/dev/null | head -40
    echo "=== 마지막 15줄 ==="
    tail -15 "$LOG" 2>/dev/null
    ;;
status)
    running && echo "Emb: 실행 중 (pid $(pgrep -x RobotEmbedded | tr '\n' ' '))" || echo "Emb: 정지"
    pgrep -af "biped_emb.py|RobotTestGait|mot_test" || echo "다른 writer: 없음"
    ipcs -m | awk 'NR<4 || /0x000004d2/'
    # EtherCAT 프레임이 실제로 오가는지 — OP 이탈 시 TX 가 0 증가로 멈춘다.
    A=$(cat /sys/class/net/eth0/statistics/tx_packets 2>/dev/null || echo 0)
    sleep 1
    B=$(cat /sys/class/net/eth0/statistics/tx_packets 2>/dev/null || echo 0)
    echo "eth0 tx_packets 1초 증가: $((B-A))  (0 이면 EtherCAT 정지 → Emb 재기동 필요)"
    ;;
*)
    sed -n '2,10p' "$0"; exit 1;;
esac
