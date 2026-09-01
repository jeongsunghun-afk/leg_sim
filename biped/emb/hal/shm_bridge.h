/* shm_bridge.h — RobotSharedMem(Gait) 위 얇은 C ABI. Python(ctypes)이 로드.
 *   단위: 위치=deg, 속도=deg/s, 토크=Nm, IMU RPY=deg, gyro=deg/s(또는 rad/s, imu_deg로 처리).
 *   C 의존(RobotSharedMem.h)을 이 파일 하나에 격리 → 나머지 배포 로직은 순수 Python.
 *   ★Pi에서 hal/build_bridge.sh 로 컴파일(RobotSharedMem.h·RobotTestGait/inc 필요). */
#ifndef BIPED_SHM_BRIDGE_H
#define BIPED_SHM_BRIDGE_H
#ifdef __cplusplus
extern "C" {
#endif

/* 채널 수(Gait). 성공=0. */
int  bridge_n_channel(void);

/* RobotMemGait_InitComm + 모터 상태 첫 수신 대기(RobotEmbedded 기동 확인).
 *   recv_wait_ms 안에 상태 미수신 시 -1. 성공=0. */
int  bridge_init(int recv_wait_ms);

/* 센서 스냅샷. 각 배열 길이 = n_channel(위치/속도/토크/전류·connected·status), imu=3.
 *   connected[i] = 채널 i 통신연결(최근 상태 수신). 0/1.
 *   status[i]    = 채널 i 모터 보고 상태(MotGeneral_t.ucStatus, 임베디드/펌웨어 정의. 0=정상 가정).
 *   nullptr 전달 시 해당 항목 스킵.  반환: 갱신 비트마스크(1=모터상태 16=IMU). 실패=-1. */
int  bridge_read(float* q_deg, float* dq_dps, float* tau_nm, float* cur_a,
                 float* imu_rpy_deg, float* imu_acc, float* imu_gyro,
                 int* connected, int* status);

/* 위치+임피던스 명령(jog·hold): 모터가 tau = kp·(q_des−q) + kd·(0−dq) 실행. 성공=0. */
int  bridge_write_pos(const float* q_des_deg, const float* kp, const float* kd, int n);

/* 풀 MIT 명령(모델기반): tau = kp·(q_des−q) + kd·(dq_des−dq) + tau_ff. 성공=0. */
int  bridge_write_mit(const float* q_des_deg, const float* dq_des_dps, const float* tau_ff_nm,
                      const float* kp, const float* kd, int n);

/* 모터 전원 enable/disable. off(0)=토크 0 명령(limp). 성공=0. */
int  bridge_enable(int on);

/* ── 2026-09-01 신설 (RGA 08/31 펌웨어 대응) — 구 .so 에는 없다: dlsym 실패 허용할 것 ── */

/* 출력축(aux) 엔코더 스냅샷. env AUX_MODE=1 로 기동했을 때만 값이 온다(ucMode=0x5A).
 *   MCU 가 상태의 fGainKp/fGainKd 슬롯에 출력축 pos[deg]/vel[deg/s] 를 실어 보낸다
 *   (modLeg.c ParseStatusEach: ucMode==0x5A 분기. 아니면 그 슬롯은 0).
 *   반환: 1=AUX_MODE 켜짐(값 유효) · 0=꺼짐(배열은 0 채움). 배열 길이 = n_channel. */
int  bridge_aux(float* pos_deg, float* vel_dps);

/* 통신 ACK 왕복 감시. 우리가 ucCommand 상위 니블에 4비트 카운터를 실어 보내면
 *   Emb 는 상위 니블을 보존한 채 하위 니블에 자기 카운트를 넣고(halGait: &0xF0 | cnt&0x0F),
 *   MCU 가 받은 그대로 상태에 에코한다(08/31 신설). 그 에코로 두 값을 계산한다:
 *     lag[i]   = (마지막 송신 카운터 − 에코 카운터) & 0xF   — 왕복 지연[write 틱]
 *     stale[i] = 에코 니블이 변하지 않은 연속 read 횟수      — 통신 두절 감지(랩 없음)
 *   ⚠구 MCU 펌웨어는 에코가 없다 → stale 만 계속 자란다(그 자체가 "구 펌웨어" 판정).
 *   env ACK_CTR=0 으로 카운터 송신을 끌 수 있다(기본 1). 반환 0. */
int  bridge_ack(int* lag, int* stale);

#ifdef __cplusplus
}
#endif
#endif /* BIPED_SHM_BRIDGE_H */
