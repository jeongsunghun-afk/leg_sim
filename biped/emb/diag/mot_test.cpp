/* mot_test.cpp — SHM MotorCommand16 직접 구동 판별 테스트(소진폭·안전종료).
 *
 *   목적: "Emb 는 살아있는데 SHM 명령에 모터가 반응 안 함" 이슈의 최소 재현/판별.
 *   RobotTestGait(±90°) 대신 ±수° 만 흔들어 같은 경로(MotorCommand16, MIT/mode=1)를 검증한다.
 *
 *   명령 파리티는 RobotTestGait/src/main.cpp 와 바이트 단위로 동일하게 맞춤:
 *     ucDevID=ch, ucMode=1, ucCommand=0, ucStatus=0, vel/acc/tau/ki/cur=0, Kp/Kd 지정.
 *
 *   ★게인 단위(2026-08-05 실측): tau[Nm] = Kp·err[rad] + Kd·derr[rad/s].
 *     명령은 deg 인데 게인은 rad 기준이라 Kp 1 당 0.0175 Nm/deg 뿐.
 *     Kp=10 이면 ±5° 명령에 실제 ±0.7° 만 움직여 "무반응" 으로 오인하기 쉽다. Kp 40~60 권장.
 *
 *   단계:
 *     [0] 관측 1s — 쓰기 없음. MotorStatus16 수신 + 신선도 확인.
 *     [1] 홀드 1s — 측정 위치를 그대로 목표로, Kp/Kd 인가. tau 가 생기면 드라이버 살아있음.
 *     [2] 정현파 N s — 홀드 위치 ±AMP°. 실제 위치가 따라오는지 확인.
 *     [3] 복귀 후 limp(Kp=Kd=0) 로 종료. Ctrl-C / 이상 시에도 동일하게 limp 로 빠짐.
 *
 *   ⚠ Emb 기동 후 최소 5초 지난 뒤 실행할 것(halGait 초기화 게이트 = 100 + 4500 tick @1kHz).
 *   ⚠ 모터 명령 writer 는 한 번에 하나만. app/RobotTestGait 와 동시 실행 금지.
 *
 *   빌드: g++ -O2 -std=c++17 mot_test.cpp -o mot_test -lRobotSharedMem -lrt
 *   실행: ./mot_test [ch=0] [amp_deg=5] [kp=40] [kd=2] [sec=6]
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <csignal>
#include <ctime>
#include "RobotSharedMem.h"

#define NCH        10          // MAX_GaitMOT_CHAN — Emb 가 읽는 채널 수
#define RATE_HZ    200
#define DT_MS      (1000 / RATE_HZ)
#define POS_LIMIT  30.0f       // |위치| 이 값을 넘으면 즉시 limp 후 중단

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int){ g_stop = 1; }

static void sleep_ms(int ms){ struct timespec ts{ ms/1000, (long)(ms%1000)*1000000L }; nanosleep(&ts,nullptr); }

static float g_pos[NCH] = {0}, g_tau[NCH] = {0}, g_vel[NCH] = {0};
static int   g_stt_seen = 0;
static int   g_same[NCH] = {0};        // 직전과 완전히 동일한 값이 반복된 횟수

/* ★stale(정지 데이터) 검출 — 반드시 필요한 이유:
 *   Emb 는 EtherCAT 이 OP 를 잃으면 commEtherCATm.cpp:520 에서 조기 return 하여 프레임 교환을
 *   완전히 멈추지만, 제어루프는 계속 돌며 "마지막 버퍼" 를 SHM 에 재발행하고 updated flag(0xff)
 *   까지 세운다. 즉 플래그는 신선도를 증명하지 못한다.
 *   위치가 얼어붙으면 POS_LIMIT 검사는 "편차 0" 으로 영원히 통과하는데, 실제 관절이 어디 있는지는
 *   알 수 없다 → 눈먼 채로 계속 명령하는 위험. 살아있는 축은 tau/vel 이 항상 미세하게 흔들리므로
 *   (±0.1Nm 관측) (pos,vel,tau) 가 비트 단위로 동일하게 반복되면 정지로 판정한다. */
#define STALE_LIMIT (RATE_HZ)          // 1초(=200샘플) 연속 무변화 → 정지

static void read_status(void){
    if (RobotMemGait_IsUpdatedMotorStatus16() != 1) return;
    unsigned long flag = RobotMemGait_GetUpdatedFlag_MotorStatus16();
    for (int i=0;i<8;i++){
        if (!(flag & (((unsigned long)1) << i))) continue;
        MotorParam16_t s; memset(&s,0,sizeof(s));
        if (RobotMemGait_GetMotorStatus16(&s, i) != 0) continue;
        const float p = (float)s.fPosition, v = (float)s.fVelocity, t = (float)s.fTorque;
        if (g_stt_seen && p == g_pos[i] && v == g_vel[i] && t == g_tau[i]) g_same[i]++;
        else                                                              g_same[i] = 0;
        g_pos[i] = p; g_vel[i] = v; g_tau[i] = t;
        g_stt_seen = 1;
    }
}

static int stale(int ch){ return g_same[ch] >= STALE_LIMIT; }

/* 전 채널 쓰기 — target 채널만 (pos,kp,kd), 나머지는 limp(Kp=Kd=0). RobotTestGait 와 동일 형식. */
static void write_all(int ch, float pos_deg, float kp, float kd){
    MotorParam16_t c;
    for (int i=0;i<NCH;i++){
        memset(&c, 0, sizeof(c));
        c.ucDevID = (unsigned char)(i & 0xff);
        c.ucMode  = 1;                       // MIT/임피던스
        c.ucCommand = 0;
        c.fPosition = (_Float16)((i == ch) ? pos_deg : 0.0f);
        c.fGainKp   = (_Float16)((i == ch) ? kp : 0.0f);
        c.fGainKd   = (_Float16)((i == ch) ? kd : 0.0f);
        RobotMemGait_SetMotorCommand16(&c, i);
    }
}

static void limp_all(void){
    MotorParam16_t c;
    for (int i=0;i<NCH;i++){
        memset(&c, 0, sizeof(c));
        c.ucDevID = (unsigned char)(i & 0xff);
        c.ucMode  = 1;
        RobotMemGait_SetMotorCommand16(&c, i);   // Kp=Kd=tau=0 → 무여자(limp)
    }
}

/* 이상 종료 공통 경로 — 반드시 limp 로 빠진다. */
static int bail(const char* why, int rc){
    printf("\n✗ %s → limp 후 중단\n", why);
    for (int k=0;k<20;k++){ limp_all(); sleep_ms(DT_MS); }
    return rc;
}

int main(int argc, char** argv){
    int   ch  = (argc > 1) ? atoi(argv[1]) : 0;
    float amp = (argc > 2) ? (float)atof(argv[2]) : 5.0f;
    float kp  = (argc > 3) ? (float)atof(argv[3]) : 40.0f;
    float kd  = (argc > 4) ? (float)atof(argv[4]) : 2.0f;
    int   sec = (argc > 5) ? atoi(argv[5]) : 6;

    if (ch < 0 || ch >= 8){ printf("ch 는 0..7\n"); return 1; }
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);

    if (RobotMemGait_InitComm() != 0){ printf("InitComm FAIL\n"); return 1; }
    printf("ch=%d amp=%.1fdeg kp=%.1f kd=%.1f sec=%d rate=%dHz\n", ch, amp, kp, kd, sec, RATE_HZ);
    printf("(예상 최대 토크 ≈ %.2f Nm = kp·amp·π/180)\n", kp * amp * 3.14159f / 180.0f);

    // [0] 관측 — 쓰기 없음.
    printf("\n[0] 관측 1s (쓰기 없음) — MotorStatus16 수신·신선도 확인\n");
    for (int k=0;k<RATE_HZ && !g_stop;k++){ read_status(); sleep_ms(DT_MS); }
    if (!g_stt_seen){
        printf("  ✗ MotorStatus16 미수신 → Emb 미기동이거나 halGait 초기화 미완료(기동 후 5s 대기 필요).\n");
        printf("    이 상태에서는 Emb 가 SHM 명령 자체를 읽지 않는다(engRobot_Command 의 IsInitialized 게이트).\n");
        return 2;
    }
    if (stale(ch)){
        printf("  ✗ ch%d 텔레메트리 정지 — pos/vel/tau 가 1초간 완전히 동일(플래그만 갱신됨).\n", ch);
        printf("    EtherCAT 이 OP 를 잃었을 가능성. Emb 는 스스로 복구하지 못하므로 재기동 필요:\n");
        printf("      sudo pkill -x RobotEmbedded ; cd ~/ZSource/RobotEmbedded/build && sudo ./src/RobotEmbedded\n");
        return 4;                            // 명령을 쓰지 않고 종료(아직 아무것도 안 썼다)
    }
    printf("  ✓ 수신 OK(신선). ch%d pos=%.3f vel=%.3f tau=%.3f\n", ch, g_pos[ch], g_vel[ch], g_tau[ch]);

    const float hold = g_pos[ch];

    // [1] 홀드 — 현재 위치를 목표로 게인 인가.
    printf("\n[1] 홀드 1s @ %.3fdeg (kp=%.1f kd=%.1f) — tau 발생 여부\n", hold, kp, kd);
    for (int k=0;k<RATE_HZ && !g_stop;k++){
        write_all(ch, hold, kp, kd);
        read_status();
        if (stale(ch)) return bail("홀드 중 텔레메트리 정지", 4);
        if (k % 40 == 0) printf("  t=%4dms pos=%8.3f tau=%7.3f\n", k*DT_MS, g_pos[ch], g_tau[ch]);
        sleep_ms(DT_MS);
    }

    // [2] 정현파 — hold ± amp.
    printf("\n[2] 정현파 %ds: %.3f ± %.1fdeg (0.5Hz)\n", sec, hold, amp);
    const int n2 = RATE_HZ * sec;
    for (int k=0;k<n2 && !g_stop;k++){
        const float t   = (float)k / (float)RATE_HZ;
        const float tgt = hold + amp * sinf(2.0f * (float)M_PI * 0.5f * t);
        write_all(ch, tgt, kp, kd);
        read_status();
        if (stale(ch))                                return bail("텔레메트리 정지(위치 미상)", 4);
        if (fabsf(g_pos[ch] - hold) > POS_LIMIT)      return bail("위치 이탈", 3);
        if (k % 20 == 0)
            printf("  t=%5.2fs cmd=%8.3f pos=%8.3f err=%7.3f vel=%8.2f tau=%7.3f\n",
                   t, tgt, g_pos[ch], tgt - g_pos[ch], g_vel[ch], g_tau[ch]);
        sleep_ms(DT_MS);
    }

    // [3] 복귀 후 limp.
    printf("\n[3] 복귀 → limp\n");
    for (int k=0;k<RATE_HZ/2;k++){ write_all(ch, hold, kp, kd); read_status(); sleep_ms(DT_MS); }
    for (int k=0;k<20;k++){ limp_all(); sleep_ms(DT_MS); }
    printf("  종료. ch%d pos=%.3f tau=%.3f (limp)\n", ch, g_pos[ch], g_tau[ch]);
    return 0;
}
