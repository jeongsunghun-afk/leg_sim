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

#ifdef __cplusplus
}
#endif
#endif /* BIPED_SHM_BRIDGE_H */
