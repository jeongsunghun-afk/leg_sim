/* slot_dump.cpp — MotorStatus16 의 **12필드 전부**를 떠서 "어느 칸에 살아있는 값이 오는가" 를 본다.
 *
 *   ★왜 (2026-08-28): 우리 브리지는 fPosition/fVelocity/fTorque/fCurrent/ucStatus **5개만** 읽는다.
 *     나머지 칸(fAccelrationOrTemperture · fGainKp/Kd/Ki · ucMode/ucCommand)에 무엇이 오는지
 *     아무도 본 적이 없다. MD80 레거시 응답(24B)에는 **출력축 엔코더 pos/vel 이 이미 들어 있으므로**,
 *     Emb/MCU 가 그걸 남는 칸에 이미 넣어두었을 가능성을 배제해야 한다.
 *       · 값이 온도(20~60)면 fAccelrationOrTemperture = MOTOR TEMPERATURE
 *       · 값이 관절각과 상관되게 움직이면 → **출력축 엔코더일 수 있다**(대박)
 *       · 전 채널 0 이면 안 쓰는 칸 = fCurrent 재활용 제안의 근거 보강
 *
 *   ★같이 판정한다 — fCurrent 가 토크 파생값인가:
 *     MD80 문서상 응답에 전류 필드가 없고 "토크는 상전류로 추정" 이므로 둘은 같은 정보다.
 *     비(fCurrent/fTorque)가 **상수면 파생 확정** · 흔들리면 독립 측정이 어딘가 있다는 뜻.
 *
 *   명령을 전혀 쓰지 않는다 → 모터 무동작. 읽기만 하므로 배포기와 동시 실행해도 안전하다.
 *   (단 배포기가 돌고 있으면 값이 움직이는 상태로 보인다 — 그게 오히려 판별에 좋다)
 *
 *   빌드: g++ -O2 -std=c++17 -I/usr/include -I$HOME/ZSource/RobotTestGait/inc \
 *         slot_dump.cpp -o slot_dump -lRobotSharedMem -lrt
 *   (RobotTestGait/inc 는 defineGeneral.h(ENUM_RESULT_*)용 — 원 빌드줄엔 빠져 있었다)
 *   ★32비트 경로도 같이 본다 (2026-08-28): 헤더에 MotorParam32_t + Set/GetMotorStatus32 가
 *     이미 있다. Emb 이 그쪽도 채우고 있다면 **float32(사실상 무손실)** 로 받을 수 있어,
 *     float16 의 각도 해상도 0.11°(2~3 rad 부근) 제약이 사라진다. 16/32 를 나란히 찍어
 *     "32비트 영역이 살아 있는가" 를 판정한다.
 *
 *   실행: ./slot_dump [iterations]      기본 200회 (~4s @ 20ms)
 */
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include "define/defineGeneral.h"   // ENUM_RESULT_* (RobotTestGait/inc)
#include "RobotSharedMem.h"

#ifndef MAX_CH
#define MAX_CH 10
#endif

static void sleep_ms(int ms){ struct timespec ts{ ms/1000, (long)(ms%1000)*1000000L }; nanosleep(&ts,nullptr); }

struct Stat {
    double mn = 1e30, mx = -1e30, sum = 0; int n = 0;
    void add(double v){ if(v<mn) mn=v; if(v>mx) mx=v; sum+=v; n++; }
    double avg() const { return n ? sum/n : 0.0; }
    double span() const { return n ? mx-mn : 0.0; }
};

int main(int argc, char** argv){
    const int iters = (argc > 1) ? atoi(argv[1]) : 200;
    if (RobotMemGait_InitComm() != 0){ printf("InitComm FAIL — Emb 이 떠 있나?\n"); return 1; }

    const char* FN[8] = {"fPosition","fVelocity","fAccelOrTemp","fTorque",
                         "fGainKp","fGainKd","fGainKi","fCurrent"};
    Stat st_f[MAX_CH][8];
    Stat st_g[MAX_CH][8];                     // 32비트 경로
    int n32_ok = 0;
    Stat st_ratio[MAX_CH];                    // fCurrent / fTorque
    unsigned char uc_seen[MAX_CH][4] = {};    // ucDevID/Mode/Command/Status 마지막값
    int nupd = 0;

    for (int k = 0; k < iters; k++){
        if (RobotMemGait_IsUpdatedMotorStatus16() == 1) nupd++;
        for (int i = 0; i < MAX_CH; i++){
            MotorParam16_t s;
            if (RobotMemGait_GetMotorStatus16(&s, i) != ENUM_RESULT_SUCCESS) continue;
            const double f[8] = {(double)(float)s.fPosition, (double)(float)s.fVelocity,
                                 (double)(float)s.fAccelrationOrTemperture, (double)(float)s.fTorque,
                                 (double)(float)s.fGainKp, (double)(float)s.fGainKd,
                                 (double)(float)s.fGainKi, (double)(float)s.fCurrent};
            for (int j = 0; j < 8; j++) st_f[i][j].add(f[j]);
            if (std::fabs(f[3]) > 0.05) st_ratio[i].add(f[7] / f[3]);
            uc_seen[i][0]=s.ucDevID; uc_seen[i][1]=s.ucMode;
            uc_seen[i][2]=s.ucCommand; uc_seen[i][3]=s.ucStatus;
            MotorParam32_t g;
            if (RobotMemGait_GetMotorStatus32(&g, i) == ENUM_RESULT_SUCCESS){
                const double h[8] = {g.fPosition, g.fVelocity, g.fAccelrationOrTemperture,
                                     g.fTorque, g.fGainKp, g.fGainKd, g.fGainKi, g.fCurrent};
                for (int j = 0; j < 8; j++) st_g[i][j].add(h[j]);
                if (i == 0) n32_ok++;
            }
        }
        sleep_ms(20);
    }

    printf("\n■ MotorStatus16 전 필드 덤프 — %d 회 샘플 (갱신플래그 %d/%d)\n\n", iters, nupd, iters);
    printf("  %-4s %-13s %10s %10s %10s %8s   판정\n", "ch", "필드", "평균", "최소", "최대", "변동폭");
    for (int i = 0; i < MAX_CH; i++){
        if (st_f[i][0].n == 0) continue;
        printf("  --- ch%d  (ucDevID %u · ucMode %u · ucCommand %u · ucStatus 0x%02X) ---\n",
               i, uc_seen[i][0], uc_seen[i][1], uc_seen[i][2], uc_seen[i][3]);
        for (int j = 0; j < 8; j++){
            const Stat& s = st_f[i][j];
            const char* verdict = "";
            if (s.mx == 0.0 && s.mn == 0.0)            verdict = "0 고정 — 안 쓰는 칸";
            else if (s.span() < 1e-6)                  verdict = "상수";
            else                                       verdict = "★살아있는 값(변함)";
            printf("  %-4s %-13s %10.4f %10.4f %10.4f %8.4f   %s\n",
                   "", FN[j], s.avg(), s.mn, s.mx, s.span(), verdict);
        }
        // 32비트 경로 요약 — 살아 있으면 우선 사용 후보
        bool any32 = false;
        for (int j = 0; j < 8; j++) if (st_g[i][j].n && (st_g[i][j].mn != 0.0 || st_g[i][j].mx != 0.0)) any32 = true;
        if (st_g[i][0].n == 0)      printf("  %-4s [32비트] GetMotorStatus32 호출 실패 — 그 경로 없음\n", "");
        else if (!any32)            printf("  %-4s [32비트] 전 필드 0 — 영역은 있으나 **Emb 이 안 채운다**\n", "");
        else {
            printf("  %-4s [32비트] ★살아있다 — pos %.4f  vel %.4f  tau %.4f  (f32 = 각도 무손실)\n", "",
                   st_g[i][0].avg(), st_g[i][1].avg(), st_g[i][3].avg());
        }
        if (st_ratio[i].n > 10){
            const Stat& r = st_ratio[i];
            printf("  %-4s fCurrent/fTorque  평균 %.4f  변동 %.4f (%d 표본)   %s\n", "",
                   r.avg(), r.span(), r.n,
                   r.span() < 0.01 ? "→ 비가 상수 = **토크 파생값 확정**"
                                   : "→ 비가 흔들림 = 독립 측정 가능성");
        }
    }
    printf("\n  32비트 상태영역 %s\n", n32_ok ? "읽기 성공(위 [32비트] 줄 참조)" : "읽기 실패");
    printf("\n  판독 요령:\n"
           "   · fAccelOrTemp 가 20~60 근처면 = MOTOR TEMPERATURE(문서상 응답 B3)\n"
           "   · 남는 칸(fGainKp/Kd/Ki)이 **관절 움직임과 함께 변하면** 출력축 엔코더가\n"
           "     이미 실려 오는 것이다 — 그러면 요청 없이 바로 쓸 수 있다\n"
           "   · 전부 0 이면 빈 칸 확정 → fCurrent 재활용 제안의 근거\n"
           "   · 배포기를 돌리며(jog 등) 같이 재면 '움직임과 함께 변하는가' 가 잘 보인다\n"
           "   · [32비트] 가 살아 있으면 출력축을 **float32 로** 요청하는 게 낫다\n"
           "     (float16 은 2~3 rad 부근 0.11° — 백래쉬 7° 엔 충분하나 미세 유격엔 거칠다)\n");
    return 0;
}
