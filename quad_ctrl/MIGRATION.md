# quad_ctrl 이관 명세 (quad/cpp → quad_ctrl)

README의 3단계 이관을 **실행 가능한 수준**으로 구체화. 핵심 = 급조 금지·항상 sim 초록불.

## ★핵심 발견 (2026-08-05): 목표 아키텍처가 이미 `trot_sim` 안에 있다
`quad/cpp/src/trot_sim.cpp`의 **`EST_CTRL` 경로**(env `EST_CTRL=1`)가 정확히 quad_ctrl 목표구조의 **원형**:
```
실기형 센서(d→noise+지연 링)  →  KF 추정(estimate_kf)  →  d_est(base=추정·자세/gyro/관절=측정)  →  컨트롤러 계산  →  토크(지연 링)  →  실 d 적용
   = HAL.read (sensor)            = Estimator.update       = State                          = Controller.step        = HAL.write
```
즉 "센서 in → 추정 → **컨트롤러는 추정상태(d_est)만 보고 계산** → 토크 out"이 **이미 검증되어 동작**(반복기립 튕김버그도 이 경로로 해결, [[sim2real-gt-est-policy]]). **이관 = 이 EST_CTRL 루프를 모듈 경계로 정식화하는 것**이지 새 알고리즘이 아니다.

## 모델 소유(m,d) — 이관의 진짜 난점과 해법
- 현재: `QuadControl`이 `mjModel* m·mjData* d`를 소유하고 모든 수식이 `q.d`를 직접 씀(`mj_fullM`·`mj_jacBody`·`d->qpos`).
- **오해 금지**: MuJoCo는 sim에서만 쓰는 게 아니라 **강체 동역학 계산기**(M·h·Jacobian)로 real에서도 필요.
- **해법(EST_CTRL이 이미 하는 것)**: `d`를 두 역할로 분리 —
  - **물리 스텝용 `d_phys`** (sim에서만; HAL이 `mj_step`) ← real에선 없음(실 로봇이 물리).
  - **동역학 계산용 `d_est`** (컨트롤러 소유; `mj_forward`만, `mj_step` 안 함) ← real qpos/qvel 주입.
- 컨트롤러는 `d_est`만 본다 → sim/real 무관 동일. HAL은 `d_phys`(sim) 또는 모터드라이버(real)만 다룬다.

## 모듈 매핑 (README 표 + 구체)
| quad/cpp | → quad_ctrl | EST_CTRL 대응 |
|---|---|---|
| sim 물리 루프(`mj_step`·xfrc·센서읽기·지연 링) | `hal/mujoco_hal` | 센서 링 sring·구동 링 cring |
| `estimate_kf`(선형 접촉 KF) | `estimator/ekf_estimator` | KF 추정 → d_est base |
| `quad_control.hpp` wbic_*·mpc_grf(d_est 계산) | `control/wbic`·`control/mpc` | 컨트롤러가 d_est만 봄 |
| `trot_controller.hpp` gait/Raibert/mode/getup | `control/gait`·`mode_fsm`·`controller`·`planner/getup_traj` | ctrl.control() 디스패치 |
| env(HAUNCH_*·GEAR_*·SIT_*·게인) | `config/*.yaml` | (EST_CTRL env도 config로) |

## 순서 (각 단계 sim 초록불 검증)
1. **1단계 — HAL+State 정식화(코드 이동 최소)**: `MujocoHal` 구현(trot_sim의 mj 루프+센서 링 이동). `SimEstimator`=GT(d_est=d). `controller`가 기존 `TrotCtrl`을 **얇게 래핑**(d_est 주입). 검증: `sim_main` == `trot_sim`(GT) falls=0. **CMake에 MuJoCo+eiquadprog 링크**(현 TODO).
2. **2단계 — EKF(sim서)**: `estimate_kf`를 `estimator/ekf_estimator`로. 노이즈·지연 주입(sim2real-checklist C). 검증: `sim_main EST_CTRL` == `trot_sim EST_CTRL` falls=0(반복기립 포함).
3. **3단계 — real_hal + robot_main** (★참조=`~/문서/jsh/RobotTestGait`, RGA 실기 C++ 인터페이스): `#include "/usr/include/RobotSharedMem.h"`(Pi 전용). `hal/real_hal.hpp`가 `RobotInterface` 구현 —
   - **read()**: SHM Pos/Vel/Tor/Cur/IMU(update flag `UPT_POS/VEL/TOR/CUR/IMU`=0x01~0x10) → `LowState`. **단위 deg→rad**, IMU `m_fIMUStt_Body`→quat/gyro.
   - **write()**: `LowCmd`(q_des/dq_des/tau_ff/kp/kd) → `MotGeneral_t`(채널별: `ucDevID·ucMode=1·fPosition·fVelocity·fTorque·fGainKp·fGainKd·fGainKi`, **float16**) → `WRITE_SHMMotorCommand`. **rad→deg**, tau_ff→fTorque, kp/kd→fGainKp/Kd.
   - **관절맵/한계**: `MotGeneral` 채널 ↔ 17-DOF 순서(부호·오프셋·`fPosZero/fPosMin/fPosMax/fVelMax` deg)=config. RobotTestGait `m_fGaitMotorRange` 형식.
   - **RT 루프**(`app/robot_main.cpp`): `timer_t`+`ftimerEvent`(RobotTestGait 골격), read→est→ctrl.step→write. 안전(한계·통신두절·enable).
   - **검증 순서**(biped/emb 실증대로): 축별 JOG(부호·오프셋 확정) → stand → walk. Mock(데스크톱)로 먼저.
   - 컨트롤러·estimator **불변**(read/write만 real). biped/emb([[biped-emb-deploy-interface]])가 같은 SHM을 Python+C브리지로 이미 검증=역이식 직관적.

## 1단계 진행 (★GT wrap 달성 2026-08-05)
- [x] CMake: MuJoCo+eiquadprog 링크·`quad/cpp/src` include (`sim_bridge` 타겟, ENV_PREFIX=proxddp).
- [x] `hal/mujoco_hal.hpp`: `MujocoHal`이 `::QuadControl` 소유(load+apply_env_gains+crouch_home+LUT+setup_mpc)·read(d→LowState)·write(LowCmd.tau_ff→d->ctrl+mj_step).
- [x] `control/trot_bridge.hpp`: `TrotBridge`가 `::TrotCtrl` wrap(재작성X). set_command(HighCmd→mode/V/gait)·step(control()→LowCmd.tau_ff).
- [x] `app/sim_bridge.cpp`: MujocoHal+TrotBridge 루프(MODE/GAIT/TROT_V/BODY_H → HighCmd).
- [x] **검증: `sim_bridge` == `trot_sim`(NO_JUMP_WARMUP) V=0/0.5/1.0 bit-동등**(x·z·tilt·falls 모두 일치). 실행 `quad_ctrl/build/sim_bridge <mjcf>`.
- [ ] (다음) estimator 정식화: 현재 컨트롤러가 `q.d` 직접(=GT, d_est=d). `estimator/sim_estimator`(GT)+`ekf_estimator`로 d_est 분리(trot_sim `EST_CTRL`·`estimate_kf` 이식) → 2단계.
- [ ] (다음) wbic/mpc/gait를 `control/` 순수모듈로 쪼개기(현재는 quad/cpp TrotCtrl을 wrap; 점진 분해).

> ★핵심 달성: **배포급 A 컨트롤러가 재작성 없이 quad_ctrl HAL 경계 뒤에서 동작·sim 초록불**. real_hal은 `read/write`만 실센서/모터로 교체(컨트롤러·나머지 불변)=3단계 직결.

> 현 배포 작동본은 `quad/cpp/`. 이관 완료 전까지 그게 진실. big-bang 재작성 금지.
