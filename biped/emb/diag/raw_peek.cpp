/* raw_peek.cpp — SHM 헤더를 라이브러리 없이 읽기전용으로 확인한다(스핀락 회피).
 *
 *   ★배경: libRobotSharedMem 의 Get/Set 은 스핀락을 쓴다.
 *       Get 계열: while(ucIsBusy_Comm2Mem == 1);  … 그 뒤 ucIsBusy_Mem2Ctrl = 1 → 0
 *       Set 계열: while(ucIsBusy_Mem2Ctrl == 1);  … 그 뒤 ucIsBusy_Comm2Mem = 1 → 0
 *     writer 가 임계구역(busy=1) 안에서 죽으면 플래그가 1 로 남고, Emb 는 Get 진입점에서
 *     영원히 스핀한다(CPU 100%, EtherCAT 프레임 정지). SHM 세그먼트는 재부팅 전까지 남으므로
 *     Emb 를 재시작해도 똑같이 다시 멈춘다 → 핸드오프의 "재시작 후 멈춤" 증상과 일치.
 *
 *   이 도구는 lib 함수를 호출하지 않고 SHM_RDONLY 로 붙어 헤더 바이트만 본다. 쓰지 않으므로
 *   절대 멈추지 않고, 다른 프로세스에 영향도 주지 않는다.
 *
 *   빌드: g++ -O2 -std=c++17 raw_peek.cpp -o raw_peek
 *   실행: ./raw_peek [반복횟수]      (기본 3회 — 순간적인 1 과 고착된 1 을 구분)
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <sys/ipc.h>
#include <sys/shm.h>

#define SHARED_KEY 1234

/* RobotSharedMem_Gait.cpp 의 RobotGaitSharedMem_t 헤더 오프셋(aarch64, 자연 정렬).
 *   ucReserved[4] 0..3 | ucIsUpdated_* 4..13 | (pad) | ulFlagUpdated_* 16,24,32,40 | ucIsBusy_* 48..51 */
#define OFF_UPD_CMD        4
#define OFF_UPD_MOTCMD16  11
#define OFF_UPD_MOTSTT16  13
#define OFF_FLAG_CMD16    24
#define OFF_FLAG_STT16    40
#define OFF_BUSY_COMM2MEM 48
#define OFF_BUSY_MEM2CTRL 49
#define OFF_BUSY_MEM2COMM 50
#define OFF_BUSY_CTRL2MEM 51

static void sleep_ms(int ms){ struct timespec ts{ ms/1000, (long)(ms%1000)*1000000L }; nanosleep(&ts,nullptr); }

int main(int argc, char** argv){
    const int iters = (argc > 1) ? atoi(argv[1]) : 3;

    const int id = shmget(SHARED_KEY, 0, 0666);            // size 0 = 기존 세그먼트 그대로
    if (id < 0){ perror("shmget"); return 1; }
    const unsigned char* p = (const unsigned char*)shmat(id, nullptr, SHM_RDONLY);
    if (p == (const unsigned char*)-1){ perror("shmat"); return 1; }
    printf("shmid=%d (읽기전용 부착)\n\n", id);

    int stuck_all = 1;
    for (int k=0;k<iters;k++){
        unsigned long f_cmd16, f_stt16;
        std::memcpy(&f_cmd16, p + OFF_FLAG_CMD16, sizeof(f_cmd16));
        std::memcpy(&f_stt16, p + OFF_FLAG_STT16, sizeof(f_stt16));

        const unsigned char b1 = p[OFF_BUSY_COMM2MEM], b2 = p[OFF_BUSY_MEM2CTRL];
        const unsigned char b3 = p[OFF_BUSY_MEM2COMM], b4 = p[OFF_BUSY_CTRL2MEM];

        printf("[%d] IsUpdated Cmd=%u MotCmd16=%u MotStt16=%u | Flag Cmd16=0x%02lx Stt16=0x%02lx\n",
               k, p[OFF_UPD_CMD], p[OFF_UPD_MOTCMD16], p[OFF_UPD_MOTSTT16],
               f_cmd16 & 0xff, f_stt16 & 0xff);
        printf("    Busy Comm2Mem=%u Mem2Ctrl=%u Mem2Comm=%u Ctrl2Mem=%u\n", b1, b2, b3, b4);

        if (!(b1 || b2 || b3 || b4)) stuck_all = 0;
        if (k+1 < iters) sleep_ms(300);
    }

    if (stuck_all){
        printf("\n★ busy 플래그가 %d회 연속 세워져 있음 = 고착.\n", iters);
        printf("  writer 가 임계구역에서 죽어 Emb 가 스핀 중일 가능성이 높다.\n");
        printf("  복구: Emb 종료 → SHM 세그먼트 제거(root) → Emb 재기동.\n");
        printf("        sudo pkill -x RobotEmbedded && sudo ipcrm -M 1234 && sudo ipcrm -M 1235\n");
        printf("  ※ 세그먼트를 지우지 않으면 재기동해도 같은 지점에서 다시 멈춘다\n");
        printf("    (InitComm 은 '새로 생성했을 때만' 버퍼를 초기화하므로 기존 세그먼트는 그대로 재사용).\n");
        return 2;
    }
    printf("\n busy 플래그 정상(스핀락 고착 아님).\n");
    return 0;
}
