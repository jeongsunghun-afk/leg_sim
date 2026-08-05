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
3. **3단계 — real_hal + robot_main**: [[biped-emb-deploy-interface]] 패턴 역이식(RobotSharedMem Gait/Pi 브리지+Mock+관절맵config+축별JOG→stand→walk). 컨트롤러·estimator 불변. 안전·RT 루프.

## 1단계 착수 체크리스트 (다음 세션)
- [ ] CMake: MuJoCo(`$CONDA_PREFIX/lib`)+eiquadprog 링크, `quad/cpp/src` include.
- [ ] `hal/mujoco_hal.hpp/.cpp`: load(재기어/GEARBOX/솔버=quad/cpp `load`+`apply_env_gains` 재사용)·read(d→LowState)·write(LowCmd→d->ctrl+mj_step).
- [ ] `estimator/sim_estimator`: GT(d_est ← d 직접).
- [ ] `control/controller`: 기존 `TrotCtrl` 인스턴스 보유, `step(State, HighCmd)`에서 d_est 세팅→`ctrl.control()`→LowCmd 추출.
- [ ] `app/sim_main`: MujocoHal+SimEstimator+Controller 루프, `MODE`/`TROT_V` 등 HighCmd로.
- [ ] 검증: `sim_main` 결과가 `trot_sim`과 동등(x·z·falls). **초록불 아니면 다음 단계 금지**.

> 현 배포 작동본은 `quad/cpp/`. 이관 완료 전까지 그게 진실. big-bang 재작성 금지.
