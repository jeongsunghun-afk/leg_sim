/* shm_dump.cpp — SHM(Gait) 부착·크기 검증 + 현재 내용 덤프. 읽기 전용, 모터 무동작.
 *   - RobotMemGait_InitComm() 로그로 shmget 성공/실패(=구조체 크기 불일치) 판별.
 *   - MotorCommand16 / MotorStatus16 을 **전 필드** 덤프한다(미사용 필드 포함).
 *
 *   ★미사용 필드를 굳이 찍는 이유 (2026-08-05 실측):
 *     20B PDO(MotorParam16_t) 중 실제로 쓰이는 건 pos/vel/tau 뿐이다.
 *       fAccelrationOrTemperture = +1.000 고정 상수 (미사용)
 *       fGainKp/Kd/Ki            = 0 (상태측 미사용)
 *       fCurrent                 = fTorque 와 완전 동일(중복)
 *     액추에이터에는 입력축·출력축 엔코더가 2개 있지만 PDO 에는 위치가 하나뿐이라
 *     백래시를 관측할 수 없다. → MCU 펌웨어에 "비어 있는 fAccelrationOrTemperture 에
 *     출력축 엔코더 각도를 실어달라" 고 요청해 둔 상태다(PDO 크기·Emb·SHM ABI 무변경).
 *     펌웨어가 값을 싣기 시작하면 이 덤프에서 그 필드가 상수 1.0 이 아니게 되므로
 *     **여기서 바로 확인**할 수 있다.
 *
 *   ⚠ GetMotorCommand16/GetMotorStatus16 은 updated 플래그를 소비한다(RobotSharedMem_Gait.cpp:873).
 *     Emb 구동 중 반복 실행하면 Emb 가 볼 명령 플래그를 훔칠 수 있으니 기본 1회만 읽는다.
 *   빌드: g++ -O2 -std=c++17 shm_dump.cpp -o shm_dump -lRobotSharedMem -lrt
 *   실행: ./shm_dump [iterations]
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include "RobotSharedMem.h"

static void sleep_ms(int ms){ struct timespec ts{ ms/1000, (long)(ms%1000)*1000000L }; nanosleep(&ts,nullptr); }

static void dump(const char* tag, const MotorParam16_t& p, int i){
    printf("  %s[%d] dev=%u mode=%u cmd=%u stt=0x%02x | pos=%8.3f vel=%8.3f "
           "acc/tmp=%7.3f tau=%7.3f kp=%6.2f kd=%6.2f ki=%5.2f cur=%7.3f\n",
           tag, i, p.ucDevID, p.ucMode, p.ucCommand, p.ucStatus,
           (float)p.fPosition, (float)p.fVelocity,
           (float)p.fAccelrationOrTemperture,          // ★미사용 필드 — 2nd 엔코더 후보
           (float)p.fTorque,
           (float)p.fGainKp, (float)p.fGainKd, (float)p.fGainKi, (float)p.fCurrent);
}

int main(int argc, char** argv){
    int iters = (argc > 1) ? atoi(argv[1]) : 1;

    printf("=== RobotMemGait_InitComm ===\n");
    unsigned int rc = RobotMemGait_InitComm();
    printf("=== InitComm rc=%u (0=SUCCESS) ===\n\n", rc);
    if (rc != 0) { printf("SHM 부착 실패 — 세그먼트 크기/권한 불일치 의심.\n"); return 1; }

    float acc_min = 1e9f, acc_max = -1e9f;

    for (int k=0;k<iters;k++){
        printf("--- iter %d ---\n", k);
        printf("IsUpdated: Cmd=%u MotCmd16=%u MotStt16=%u Pos=%u IMU=%u\n",
               RobotMemGait_IsUpdatedCommand(),
               RobotMemGait_IsUpdatedMotorCommand16(),
               RobotMemGait_IsUpdatedMotorStatus16(),
               RobotMemGait_IsUpdatedPosition(),
               RobotMemGait_IsUpdatedIMU());
        printf("Flags: Cmd16=0x%02lx Stt16=0x%02lx\n",
               RobotMemGait_GetUpdatedFlag_MotorCommand16() & 0xff,
               RobotMemGait_GetUpdatedFlag_MotorStatus16() & 0xff);
        for (int i=0;i<8;i++){
            MotorParam16_t c; memset(&c,0,sizeof(c));
            MotorParam16_t s; memset(&s,0,sizeof(s));
            RobotMemGait_GetMotorCommand16(&c, i);
            RobotMemGait_GetMotorStatus16(&s, i);
            dump("CMD", c, i);
            dump("STT", s, i);
            if (i==0 || i==4){                          // 실장 축만 추적
                const float a = (float)s.fAccelrationOrTemperture;
                if (a < acc_min) acc_min = a;
                if (a > acc_max) acc_max = a;
            }
        }
        if (k+1 < iters) sleep_ms(200);
    }

    printf("\n미사용 필드 acc/tmp (ch0·ch4) 범위: %.4f ~ %.4f\n", acc_min, acc_max);
    if (acc_max - acc_min < 1e-3f)
        printf("  → 상수. 아직 2nd 엔코더 미탑재(펌웨어 요청 대기 중).\n");
    else
        printf("  ★ 값이 변한다! 2nd 엔코더가 실렸을 가능성 — 단위·부호·영점 규약을 확인할 것.\n");
    return 0;
}
