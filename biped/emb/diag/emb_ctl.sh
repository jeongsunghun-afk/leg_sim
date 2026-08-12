#!/usr/bin/env bash
# emb_ctl.sh — RobotEmbedded(Emb) 기동/종료/로그 헬퍼. 진단 절차를 한 곳에 고정.
#   Emb 는 root 필요(EtherCAT raw socket / SPI). sudoers 미설치면 암호를 물어본다.
#
#   ./emb_ctl.sh start   # 기동(로그: /tmp/emb.log) 후 halGait 초기화(≈5s) + 신선도 확인
#   ./emb_ctl.sh stop    # 종료
#   ./emb_ctl.sh log     # EtherCAT 요약(슬레이브/OP/WKC) + 마지막 로그
#   ./emb_ctl.sh reset   # **싹 종료** — writer·Emb·되살리는 래퍼 셸까지 전부
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
_n=$(pgrep -x RobotEmbedded 2>/dev/null | wc -l)
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
    # ★**sudo 없이 먼저 시도한다** (2026-08-12).
    #   바이너리에 capabilities 가 박혀 있어 root 가 필요 없다:
    #       getcap → cap_net_admin,cap_net_raw=eip   (EtherCAT 원시소켓)
    #       /dev/spidev0.0,0.1 은 dialout 그룹 rw, 사용자가 dialout 소속
    #   ⇒ sudo 를 쓰면 얻는 게 없고 잃는 게 크다: root 셸이 고아로 남아 Emb 를 되살리고
    #     EtherCAT 마스터가 중복된다(2026-08-12 4개까지 늘어 전 채널이 얼어붙었다).
    #   ⚠/dev/gpiomem*·/dev/mem 은 여전히 root 전용이다. 그걸 쓴다면 무권한 기동이
    #     실패하므로 **그때만** sudo 로 재시도한다.
    #   ★sudoers 규칙은 "정확히 이 바이너리 경로"만 NOPASSWD 이므로 sh -c 로 감싸지 말 것.
    #     리다이렉션은 호출측(비root) 셸이 수행 → 로그 파일 소유자는 rpetubt.
    # ★stdbuf -oL — **줄단위 버퍼링 강제** (2026-08-12).
    #   터미널로 직접 띄우면 로그가 콸콸 올라오는데 파일로 넘기면 0바이트였다.
    #   stdout 이 파이프·파일이면 libc 가 **블록 버퍼링**(4~8KB)으로 바꾸고,
    #   Emb 는 영원히 도니 버퍼가 flush 되지 않는다. pkill 로 죽이면 **통째로 버려진다.**
    #   ⇒ "RobotEmbedded 는 stdout 에 안 쓴다" 는 오판이었다. 실은 쓰는데 안 보였다.
    #     그 오판 때문에 EtherCAT 동결 원인 규명이 하루 막혔다.
    ( cd "$EMB_DIR" && stdbuf -oL -eL "$EMB_BIN" > "$LOG" 2>&1 & )
    sleep 2
    if ! running; then
        echo "  무권한 기동 실패 → sudo 로 재시도(gpiomem/mem 접근이 필요한 듯):"
        tail -5 "$LOG" 2>/dev/null | sed 's/^/    /'
        # ⚠sudo 경로는 stdbuf 로 감쌀 수 없다 — sudoers 규칙이 "정확히 이 바이너리
        #   경로"만 NOPASSWD 라 stdbuf 를 끼우면 암호를 묻는다. 이 경로로 떨어지면
        #   로그가 블록 버퍼링된다(위 주석). 무권한 기동이 되는 한 이 경로는 안 쓴다.
        ( cd "$EMB_DIR" && sudo "$EMB_BIN" > "$LOG" 2>&1 & )
        sleep 2
    fi
    if ! running; then echo "✗ 기동 실패:"; tail -30 "$LOG" 2>/dev/null; exit 1; fi
    echo "[emb_ctl] 권한: $(ps -o user= -p "$(pgrep -x RobotEmbedded | head -1)" 2>/dev/null)"
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
    n=$(pgrep -x RobotEmbedded 2>/dev/null | wc -l)
    echo "  남은 RobotEmbedded: $n"
    "$0" log
    echo
    echo "  다음 순서로 복구할 것:"
    echo "    ① 모터 전원 OFF → 3초 → ON   (EtherCAT 슬레이브 초기화. Emb 재기동만으론 부족)"
    echo "    ② diag/emb_ctl.sh start"
    echo "  ⚠수동으로 RobotEmbedded 를 따로 띄우지 말 것 — 중복 기동이 버스를 깬다."
    exit 2
    ;;
reset)
    # ★"싹 종료" — 모터를 만질 수 있는 모든 프로세스를 없앤다 (2026-08-12).
    #   ⚠stop 만으로는 부족했다. Emb 를 죽여도 **그걸 띄운 래퍼 셸이 살아남아** 다음
    #     명령에서 다시 띄운다. 실기에서 계보가 이랬다:
    #         RobotEmbedded ← sudo ← sudo ← bash(PPID=1, 고아)
    #     emb_ctl.sh 를 sudo 로 실행해서 sudo 가 두 겹이 됐고, 터미널을 닫아도
    #     그 bash 가 살아 EtherCAT 마스터가 4개까지 늘었다.
    #   ⇒ writer → Emb → 래퍼 셸 순으로 지우고, 마지막에 남은 게 없는지 확인한다.
    #   ★자기 자신과 조상은 절대 죽이지 않는다(스크립트가 중간에 죽으면 정리가 안 끝난다).
    echo "[emb_ctl] reset — 모터를 만질 수 있는 프로세스를 전부 정리한다"
    _keep=" $$ $PPID "
    _anc=$PPID; for _ in 1 2 3 4 5; do
        _anc=$(ps -o ppid= -p "$_anc" 2>/dev/null | tr -d ' '); [ -z "$_anc" ] && break
        _keep="$_keep$_anc "; [ "$_anc" = "1" ] && break
    done

    # ★pkill -f 를 쓰지 않는다. **자기 자신을 죽인다.**
    #   2026-08-12: 이 스크립트를 시험하던 셸의 argv 에 "actuator_test.py" 라는 문자열이
    #   들어 있었더니 pkill -f 가 그 셸을 죽여 reset 이 ①에서 끊겼다(exit 144).
    #   패턴이 인자·히스토리·에디터 명령줄에 우연히 들어가는 건 흔한 일이다.
    #   ⇒ PID 를 모아 조상 제외 후 하나씩 죽인다. _kill_matching 로 ①③ 공통 처리.
    _kill_matching() {   # $1 = ERE 패턴, $2 = 라벨
        for _p in $(ps -eo pid=,args= | grep -E "$1" | grep -vE "grep -E|ps -eo" \
                    | awk '{print $1}'); do
            case "$_keep" in *" $_p "*) continue;; esac
            _a=$(ps -o args= -p "$_p" 2>/dev/null | cut -c1-70)
            [ -z "$_a" ] && continue
            echo "     kill $_p  $_a"
            kill -9 "$_p" 2>/dev/null || sudo kill -9 "$_p" 2>/dev/null
        done
    }
    echo "  ① writer 종료 (biped_emb.py · actuator_test.py · RobotTestGait · mot_test)"
    _kill_matching "biped_emb\.py|actuator_test\.py|collect_multichirp\.py|RobotTestGait|mot_test"
    sleep 1

    echo "  ② Emb 종료"
    sudo pkill -x RobotEmbedded 2>/dev/null; sleep 1
    if pgrep -x RobotEmbedded >/dev/null 2>&1; then
        echo "     TERM 으로 안 죽는다 → KILL"
        sudo pkill -9 -x RobotEmbedded 2>/dev/null; sleep 1
    fi

    echo "  ③ 되살리는 래퍼 셸·sudo 정리"
    _kill_matching "RobotEmbedded|emb_ctl\.sh"
    sleep 1

    # ⚠pgrep -c 는 "0" 을 **찍으면서 exit 1** 이다. `|| echo 0` 을 붙이면
    #   출력이 "0\n0" 이 되어 문자열 비교가 깨진다(2026-08-12 실제로 실패분기로 빠졌다).
    _n=$(pgrep -x RobotEmbedded 2>/dev/null | wc -l)
    _w=0
    for _p in $(ps -eo pid=,args= | grep -E "biped_emb\.py|actuator_test\.py" \
                | grep -vE "grep -E|ps -eo" | awk '{print $1}'); do
        case "$_keep" in *" $_p "*) continue;; esac
        _w=$((_w+1))
    done
    echo
    echo "  남은 것 — Emb $_n 개 · writer $_w 개"
    if [ "$_n" != "0" ] || [ "$_w" != "0" ]; then
        echo "  ✗ 아직 남았다:"; ps -eo pid,ppid,etime,args | grep -E "RobotEmbedded|biped_emb\\.py|actuator_test\\.py" | grep -vE "grep -E|ps -eo"   # ★\.py 를 붙인다 — 뷰어의 biped_emb.**yaml** 경로에 걸렸었다
        exit 1
    fi
    echo "  ✓ 전부 정리됨"
    echo
    echo "  다음: ① 모터 전원 OFF → 3초 → ON   ② diag/emb_ctl.sh start"
    echo "  ⚠ emb_ctl.sh 에 sudo 를 붙이지 말 것 — 스크립트가 안에서 쓴다. 밖에서 한 겹 더"
    echo "    씌우면 sudo 가 이중이 되고 고아 셸이 남는다(이번 사고의 원인)."
    exit 0
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
