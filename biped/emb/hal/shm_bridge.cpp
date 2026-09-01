/* shm_bridge.cpp — RobotSharedMem(Gait) 위 C ABI 구현. RobotTestGait/src/main.cpp 의
 *   SHM 사용 패턴(상태 read + MIT command write)을 라이브러리로 추출.
 *   ★Pi에서만 컴파일(RobotSharedMem.h·defineConfigMotor.h 필요). hal/build_bridge.sh 참조.
 */
#include "shm_bridge.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "define/defineGeneral.h"        // RobotTestGait/inc — ENUM_RESULT_*
#include "define/defineConfigMotor.h"    //   MotGeneral_t, MAX_GaitMOT_CHAN, ENUM_Gait_JointID_NUM
#include "RobotSharedMem.h"              // /usr/include — RobotMemGait_* (Pi 전용)

// RobotSharedMem.h 가 제공하는 IMU 인덱스. 심볼명이 다르면 여기만 조정.
//   RobotTestGait 사용: IDX_OF_IMU_ForeC_START, LEN_OF_IMU_DATA, IDX_OF_IMU_ARPY, IDX_OF_IMU_ACCL
// ★2026-08-05 수정. 여기 있던 IDX_OF_IMU_AVEL 은 **SDK 어디에도 없는 심볼**이었다.
//   ZSource 전체·/usr/include/RobotSharedMem.h 전수 grep 0건. 실명은 IDX_OF_IMU_GYRO(=7).
//   그 결과 아래 #ifndef 폴백이 항상 발동해 -1 로 정의되고, 사용처의 `if (... >= 0)` 이
//   **컴파일 타임에 접혀** 자이로가 무조건 0 으로 채워졌다. 경고 하나 없이 조용히 통과한다.
//   (배포된 libbipedshm.so 역어셈블로 확증: rpy/acc 는 ldr 로 읽는데 gyro 자리는 movi #0x0 상수.)
//   ⚠ 이걸 고쳐도 **지금은 값이 안 들어온다** — Emb 쪽 결함으로 SHM 슬롯 자체가 0 이다.
//     자세한 건 emb/IMU_RECOVERY.md. 여기서는 브리지를 올바른 심볼로 맞춰만 둔다.
#ifndef IDX_OF_IMU_GYRO
#error "IDX_OF_IMU_GYRO 없음 — RobotSharedMem.h 버전 확인 필요(폴백으로 조용히 0 채우지 말 것)"
#endif

static const int NCH = (int)MAX_GaitMOT_CHAN;
static int   g_enabled = 0;
static float g_last_q[16] = {0};         // limp/hold 기준(마지막 측정 위치)
static int   g_conn_ctr[16] = {0};       // 채널별 통신연결 카운터(상태 수신 시 리셋, 미수신 시 감쇠)
static int   g_status[16] = {0};         // 채널별 마지막 ucStatus(모터/펌웨어 보고)
static const int CONN_WINDOW = 250;      // 이 콜 수(≈0.5s@500Hz) 동안 미수신이면 연결끊김(LED off)

// ── 2026-09-01 (RGA 08/31 펌웨어 대응) ──────────────────────────────────────
// ★출력축(aux) 엔코더 — ucMode=0x5A 로 보내면 MCU 가 상태의 fGainKp/fGainKd 슬롯에
//   출력축 pos[deg]/vel[deg/s] 를 싣는다(modLeg.c ParseStatusEach). 아니면 그 슬롯 0.
//   기본 **꺼짐**(ucMode=1 유지) — 0x5A 가 MCU→MD80 명령 프레임까지 바꾸는지 RGA 미확인.
//   확인 전엔 매달린 상태에서만 AUX_MODE=1 로 시험할 것.
// ★ACK 카운터 — ucCommand 상위 니블(&0xF0)은 halGait 패스스루가 **보존**하고
//   하위 니블엔 Emb 가 자기 카운트를 넣는다. MCU 는 받은 ucCommand 를 상태에 에코(08/31).
//   ⇒ 상위 니블에 4비트 카운터를 실으면 Pi↔MCU 왕복을 직접 셀 수 있다 — 동결 포렌식이
//     carrier 로 추정하던 것을 대체한다. 지금까지도 ucCommand 는 Emb 카운트가 실려
//     매 틱 변하는 값이었으므로(우리가 0 을 보내도) 새 값이 가는 것 자체는 새 위험이 아니다.
static unsigned char g_mode = 1;         // ucMode: 1=MIT/임피던스 · 0x5A=+출력축 엔코더 응답
static int   g_ack_on = 1;               // ACK_CTR=0 으로 끔
static unsigned g_tick = 0;              // write 틱마다 +1 (전 채널 공통 — 상관 가능하게)
static float g_aux_pos[16] = {0};        // 마지막 aux pos[deg] (fGainKp 슬롯)
static float g_aux_vel[16] = {0};        //          vel[deg/s] (fGainKd 슬롯)
static int   g_echo_nib[16];             // 마지막 에코 상위 니블(-1=미수신)
static int   g_echo_stale[16] = {0};     // 에코 니블 무변화 연속 read 수
static int   g_ack_lag[16] = {0};        // (송신 − 에코) & 0xF [write 틱]

int bridge_n_channel(void){ return NCH; }

static void sleep_ms(int ms){ struct timespec ts{ ms/1000, (long)(ms%1000)*1000000L }; nanosleep(&ts, nullptr); }

int bridge_init(int recv_wait_ms){
    // env 는 여기서 한 번만 읽는다(운전 중 바뀌면 계단이 되므로 재읽기 금지)
    { const char* am = getenv("AUX_MODE");
      g_mode = (am && atoi(am) != 0) ? (unsigned char)0x5A : (unsigned char)1;
      const char* ac = getenv("ACK_CTR");
      g_ack_on = (ac && atoi(ac) == 0) ? 0 : 1;
      for (int i = 0; i < 16; i++) g_echo_nib[i] = -1;
      if (g_mode == 0x5A)
          std::printf("[shm_bridge] ★AUX_MODE — ucMode=0x5A: 상태 GainKp/Kd 슬롯 = 출력축 pos/vel\n"
                      "             ⚠MD80 명령 프레임 영향 미확인 — 매달린 상태에서만 시험할 것\n");
      if (!g_ack_on)
          std::printf("[shm_bridge] ACK 카운터 꺼짐(ACK_CTR=0) — ucCommand 상위 니블 0 고정\n");
    }
    if (RobotMemGait_InitComm() != ENUM_RESULT_SUCCESS) return -1;
    // RobotEmbedded 기동 확인: 모터 상태 수신될 때까지 대기(명령 전 안전 핸드셰이크).
    int waited = 0;
    while (waited < recv_wait_ms){
        if (RobotMemGait_IsUpdatedMotorStatus16() == 1) { g_enabled = 0; return 0; }
        sleep_ms(5); waited += 5;
    }
    return -1;   // 미수신 = 임베디드 모터 컨트롤러 미기동
}

int bridge_read(float* q_deg, float* dq_dps, float* tau_nm, float* cur_a,
                float* imu_rpy_deg, float* imu_acc, float* imu_gyro,
                int* connected, int* status){
    int mask = 0;
    // ── 채널별 모터 상태(MotorStatus16) = 값 + 통신연결 + ucStatus. RobotTestGait 활성 경로와 동일. ──
    //    ★만약 임베디드가 bulk GetPosition 경로만 채운다면 여기를 GetPosition 으로 교체(README TODO).
    unsigned long flag = 0;
    if (RobotMemGait_IsUpdatedMotorStatus16() == 1)
        flag = RobotMemGait_GetUpdatedFlag_MotorStatus16();     // 채널별 상태 수신 비트마스크
    MotGeneral_t st;
    for (int i = 0; i < NCH; i++){
        unsigned long bit = (((unsigned long)1) << i);
        if (flag & bit){                                        // 이 채널 상태 수신 → 값 갱신·연결 리셋
            if (RobotMemGait_GetMotorStatus16((MotorParam16_t*)&st, i) == ENUM_RESULT_SUCCESS){
                if (q_deg)  q_deg[i]  = (float)st.fPosition;
                if (dq_dps) dq_dps[i] = (float)st.fVelocity;
                if (tau_nm) tau_nm[i] = (float)st.fTorque;
                if (cur_a)  cur_a[i]  = (float)st.fCurrent;
                g_last_q[i]   = (float)st.fPosition;
                g_status[i]   = (int)st.ucStatus;               // 모터/펌웨어 보고 상태
                g_conn_ctr[i] = CONN_WINDOW;
                // aux 엔코더(AUX_MODE 시 MCU 가 채움 — 아니면 0 이 온다. 그대로 저장)
                g_aux_pos[i] = (float)st.fGainKp;
                g_aux_vel[i] = (float)st.fGainKd;
                // ACK 에코 추적 — 상위 니블만 우리 몫(하위는 Emb 카운트)
                { const int nib = ((int)st.ucCommand >> 4) & 0x0F;
                  if (nib == g_echo_nib[i]) { if (g_echo_stale[i] < 1000000) g_echo_stale[i]++; }
                  else                      { g_echo_nib[i] = nib; g_echo_stale[i] = 0; }
                  g_ack_lag[i] = (int)((g_tick - (unsigned)nib) & 0x0F); }
            }
            mask |= 0x01;
        } else if (g_conn_ctr[i] > 0){
            g_conn_ctr[i]--;                                    // 미수신 → 연결 카운터 감쇠
        }
        if (connected) connected[i] = (g_conn_ctr[i] > 0) ? 1 : 0;
        if (status)    status[i]    = g_status[i];
    }
    // ── IMU (별도 채널) ──
    if ((imu_rpy_deg || imu_acc || imu_gyro) && RobotMemGait_IsUpdatedIMU()){
        float buf[LEN_OF_IMU_DATA] = {0};
        if (RobotMemGait_GetIMU(buf, IDX_OF_IMU_ForeC_START, LEN_OF_IMU_DATA) == ENUM_RESULT_SUCCESS){
            if (imu_rpy_deg) for (int i=0;i<3;i++) imu_rpy_deg[i] = buf[IDX_OF_IMU_ARPY + i];
            if (imu_acc)     for (int i=0;i<3;i++) imu_acc[i]     = buf[IDX_OF_IMU_ACCL + i];
            if (imu_gyro)    for (int i=0;i<3;i++) imu_gyro[i]    = buf[IDX_OF_IMU_GYRO + i];
            // ★"신선한 0" 방어. RobotMemGait_IsUpdatedIMU() 는 **내용을 검증하지 않는다** —
            //   RobotSharedMem_Gait.cpp 의 SetIMU 가 검증 루프를 빈 채로 두고 memcpy 직후
            //   무조건 ucIsUpdated_IMU=1 을 세운다. 그래서 Emb 이 0 배열을 써도 "신선함"으로 온다.
            //   값이 0 인 것보다 이게 더 위험하다: freshness 검사로 못 걸러지므로 하류
            //   tilt E-stop 이 "유효한 수평 자세"로 오해한다(tilt≡0 → 임계 영원히 미도달).
            //   정상 IMU 는 정지 중에도 가속도계에 중력 ~9.81 m/s^2 가 반드시 잡히므로
            //   가속도 3축 크기가 사실상 0 이면 센서가 죽은 것이다. 이때는 mask 를 세우지 않아
            //   **상류가 IMU 없음을 인지**하게 한다(0 을 유효값으로 흘려보내지 않는다).
            const float ax=buf[IDX_OF_IMU_ACCL+0], ay=buf[IDX_OF_IMU_ACCL+1], az=buf[IDX_OF_IMU_ACCL+2];
            if (ax*ax + ay*ay + az*az > 0.25f) mask |= 0x10;   // |a| > 0.5 m/s^2 이어야 유효
        }
    }
    return mask;
}

// 공통 MIT 쓰기. tau_ff/dq_des = nullptr 이면 0.
static int write_mit_impl(const float* q_des, const float* dq_des, const float* tau_ff,
                          const float* kp, const float* kd, int n){
    if (n > NCH) n = NCH;
    g_tick++;                                              // write 틱(전 채널 공통 ACK 카운터)
    MotGeneral_t cmd;
    for (int i=0;i<n;i++){
        std::memset(&cmd, 0, sizeof(cmd));
        cmd.ucDevID  = (unsigned char)(i & 0xff);
        cmd.ucMode   = g_mode;                             // 1=MIT/임피던스 · 0x5A=+출력축 응답
        // 상위 니블 = 우리 ACK 카운터. 하위 니블은 halGait 가 자기 카운트로 덮는다(&0xF0 보존).
        cmd.ucCommand = g_ack_on ? (unsigned char)((g_tick & 0x0F) << 4) : (unsigned char)0;
        cmd.fPosition = (float16)(g_enabled ? q_des[i] : g_last_q[i]);
        cmd.fVelocity = (float16)(g_enabled && dq_des ? dq_des[i] : 0.0f);
        cmd.fTorque   = (float16)(g_enabled && tau_ff ? tau_ff[i] : 0.0f);
        cmd.fGainKp   = (float16)(g_enabled ? kp[i] : 0.0f);   // 전원 off = kp/kd/tau 0 = limp
        cmd.fGainKd   = (float16)(g_enabled ? kd[i] : 0.0f);
        cmd.fGainKi   = (float16)0.0f;
        if (RobotMemGait_SetMotorCommand16((MotorParam16_t*)&cmd, i) != ENUM_RESULT_SUCCESS) return -1;
    }
    return 0;
}

int bridge_write_pos(const float* q_des_deg, const float* kp, const float* kd, int n){
    return write_mit_impl(q_des_deg, nullptr, nullptr, kp, kd, n);
}
int bridge_write_mit(const float* q_des_deg, const float* dq_des_dps, const float* tau_ff_nm,
                     const float* kp, const float* kd, int n){
    return write_mit_impl(q_des_deg, dq_des_dps, tau_ff_nm, kp, kd, n);
}

int bridge_enable(int on){ g_enabled = on ? 1 : 0; return 0; }

int bridge_aux(float* pos_deg, float* vel_dps){
    for (int i = 0; i < NCH; i++){
        if (pos_deg) pos_deg[i] = g_aux_pos[i];
        if (vel_dps) vel_dps[i] = g_aux_vel[i];
    }
    return (g_mode == 0x5A) ? 1 : 0;
}

int bridge_ack(int* lag, int* stale){
    for (int i = 0; i < NCH; i++){
        if (lag)   lag[i]   = g_ack_lag[i];
        if (stale) stale[i] = (g_echo_nib[i] < 0) ? -1 : g_echo_stale[i];   // -1 = 에코 미수신
    }
    return 0;
}
