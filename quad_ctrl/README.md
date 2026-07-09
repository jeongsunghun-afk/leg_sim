# quad_ctrl — 02_Leg 제어기 구조 스켈레톤

WBIC + 반응형 MPC 사족 제어기를 **sim↔real 이식 가능한** 모듈 구조로 재편하기 위한 골격.
현 `simulation/quad/cpp/`(monolithic)에서 이 구조로 **점진 이관**한다(big-bang 재작성 금지).

## 핵심 원칙 (3)
1. **State가 컨트롤러의 유일한 입력** — WBIC/MPC는 `State`만 보고 동작 → sim/real 무관 동일 코드.
2. **HAL이 유일한 부작용 경계** — 제어 코드는 `mj_step`/`d->qpos`를 직접 만지지 않는다(→ real 이식 시 제어 코드 불변).
3. **파라미터는 config(yaml)** — env-var 소거(HAUNCH_*·GEAR_*·SIT_KP…) = 재현성·튜닝·sim2real 정합.

데이터 흐름: `HAL.read → LowState → Estimator → State → Controller → LowCmd → HAL.write`

## 디렉토리
```
quad_ctrl/
├── hal/        robot_interface(LowState/LowCmd)·mujoco_hal(sim)·real_hal(나중)·mock_hal(데모)
├── estimator/  state(컨트롤러 입력)·sim_estimator(GT)·ekf_estimator(real, 나중)
├── control/    wbic·mpc·gait·mode_fsm·controller(조율)
├── planner/    getup_traj(기립 궤적 추종)  ← foot_placement 등 추가
├── command/    sport_client(HighCmd: Move/Sit/StandUp)
├── common/     types(Eigen·Leg)  ← math·params 추가
├── app/        sim_main(현 trot_view) · robot_main(real RT, 나중)
└── config/     *.yaml (env 대체)
```

## 이관 매핑 (현 코드 → 새 모듈)
| 현재 (simulation/quad/) | → 새 위치 | 비고 |
|---|---|---|
| `cpp/src/quad_control.hpp` wbic_stance/track/jump | `control/wbic.*` | QP·부동베이스 동역학 |
| `cpp/src/quad_control.hpp` crouch_home/haunch_sit_home | `control/wbic.*` 또는 `planner/` | 자세 참조 IK |
| `cpp/src/mpc.hpp` | `control/mpc.*` | Di Carlo SRBD. ★Rz(yaw)ᵀ 정합 유지 |
| `cpp/src/trot_controller.hpp` set_gait/Raibert/whip | `control/gait.*` | 게이트·발배치 |
| `cpp/src/trot_controller.hpp` mode dispatch | `control/mode_fsm.*` + `controller.*` | off/sit/getup/walk |
| `cpp/src/trot_controller.hpp` getup 블록 | `planner/getup_traj.*` | gather 궤적 추종(속도FF 필수) |
| `cpp/src/trot_sim.cpp`·`trot_view.cpp` mj 루프 | `hal/mujoco_hal.*` + `app/sim_main.cpp` | 물리·렌더 격리 |
| `teleop_gui_17dof.py` + CMDFILE | `command/sport_client.*` | 고수준 명령 |
| env-var(HAUNCH_*·GEAR_*…) | `config/*.yaml` | 파라미터화 |

## 순서 (점진, 항상 sim 초록불)
- **1단계(지금)**: HAL·State 인터페이스 확정 → 제어 코드를 `control/`로 쪼개 **MujocoHal + SimEstimator(GT)** 뒤에 배치. `sim_main`이 현 trot_view와 동등 동작할 때까지.
- **2단계**: EKF estimator를 **sim에서** 먼저(노이즈·지연 주입=sim2real-checklist C) → 동일 State 인터페이스로 검증.
- **3단계**: `real_hal`(모터/IMU)·안전·RT 루프(`robot_main`) 연결. 컨트롤러 불변.

## 빌드 (골격 데모 — MuJoCo 불필요)
```
cd quad_ctrl && cmake -S . -B build && cmake --build build -j
./build/sim_main      # → [quad_ctrl] 스켈레톤 루프 10틱 OK ...
```
2단계부터 MuJoCo/eiquadprog 링크(CMake TODO 참조).

> 현 배포(작동본)는 `simulation/quad/cpp/`에 그대로 유지. 이관 완료 전까지 그게 진실.
