/* stt_probe.cpp — 무(無)모션 상태 진단.
 *   Emb 가 EtherCAT 에서 채운 MotorStatus16 을 읽어 체인 생존/드라이버 폴트 판별.
 *   명령을 전혀 쓰지 않으므로 안전(모터 무동작).
 *
 *   ★플래그만 보면 속는다: Emb 는 EtherCAT 이 OP 를 잃어도 마지막 버퍼를 계속 재발행하며
 *     updated flag 를 0xff 로 세운다(commEtherCATm.cpp:520 조기 return). 따라서 "값이 실제로
 *     변하는지" 를 봐야 한다. 아래 요약에서 변화량(diff)을 함께 보고한다.
 *
 *   빌드: g++ -O2 -std=c++17 -I/usr/include stt_probe.cpp -o stt_probe -lRobotSharedMem -lrt
 *   실행: (Emb 기동 후) ./stt_probe [iterations]
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "RobotSharedMem.h"

static void sleep_ms(int ms){ struct timespec ts{ ms/1000, (long)(ms%1000)*1000000L }; nanosleep(&ts,nullptr); }

int main(int argc, char** argv){
    int iters = (argc > 1) ? atoi(argv[1]) : 40;   // 기본 40회 (~2s @ 50ms)
    if (RobotMemGait_InitComm() != 0){ printf("InitComm FAIL\n"); return 1; }
    printf("InitComm OK. 채널 0..7 상태 폴링 (%d회)\n", iters);

    int   rx_any = 0;
    int   changed[8] = {0};
    float prev[8][3] = {{0}};
    int   have_prev = 0;

    for (int k=0;k<iters;k++){
        unsigned char upd = RobotMemGait_IsUpdatedMotorStatus16();
        unsigned long flag = RobotMemGait_GetUpdatedFlag_MotorStatus16();
        printf("[%3d] IsUpdated=%u  Flag=0x%02lx  |", k, upd, flag & 0xff);
        for (int i=0;i<8;i++){
            MotorParam16_t st; memset(&st,0,sizeof(st));
            RobotMemGait_GetMotorStatus16(&st, i);
            const float p = (float)st.fPosition, v = (float)st.fVelocity, t = (float)st.fTorque;
            if (have_prev && (p != prev[i][0] || v != prev[i][1] || t != prev[i][2])) changed[i]++;
            prev[i][0] = p; prev[i][1] = v; prev[i][2] = t;
            // 설치된 축(0,4)만 상세, 나머지는 생략
            if (i==0 || i==4){
                printf(" [%d]md=%u stt=0x%02x pos=%7.2f tau=%6.2f cur=%6.2f",
                       i, st.ucMode, st.ucStatus, p, t, (float)st.fCurrent);
            }
        }
        have_prev = 1;
        printf("\n");
        if (upd) rx_any = 1;
        sleep_ms(50);
    }

    printf("\n요약: MotorStatus 플래그 수신=%s\n", rx_any ? "YES" : "NO(Emb 미기동 or 초기화 미완료)");
    printf("      값 변화 횟수(신선도): ");
    for (int i=0;i<8;i++) printf("[%d]%d ", i, changed[i]);
    printf("\n");
    if (rx_any && changed[0] == 0 && changed[4] == 0)
        printf("★ 플래그는 오는데 값이 전혀 안 변함 = 정지 데이터. EtherCAT OP 이탈 의심 → Emb 재기동 필요.\n");
    else if (rx_any)
        printf("✓ 값이 갱신됨 = EtherCAT 체인 생존.\n");
    return 0;
}
