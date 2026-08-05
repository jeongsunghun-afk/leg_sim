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

## Emb(실기) 배포 경로 — 3단계 = `real_hal` + `robot_main`
배포급 컨트롤러 = **A(MPC+WBIC)**, 현재 `quad/cpp`(monolithic·검증됨). Emb 올리기 = 이 골격으로 이관 완료 후 HAL만 교체:
```
LowState(센서)          →  Estimator → State → QuadController → LowCmd(tau/PD)
   ▲ hal/real_hal.read                                              │ hal/real_hal.write ▼
RobotSharedMem(모터·IMU, RGA)  ←────────────────────────────────  RobotSharedMem(MIT)
```
- **컨트롤러 불변**(3원칙: State만 입력). sim(`mujoco_hal`)→real(`real_hal`)은 HAL 교체만.
- **모델 동역학**: MuJoCo를 *모델*로 사용(real qpos/qvel 주입→`mj_forward`→M·h·Jac). 물리엔진이 아니라 강체계산기.
- **real_hal 패턴 = 검증된 `biped/emb`를 그대로**: RobotSharedMem(Gait/Pi) 얇은 C ABI 브리지 + Mock(데스크톱) + 관절맵(부호·오프셋·한계 config) + 축별 JOG로 각축 검증 → stand → walk. (biped/emb가 `quad_ctrl` 원칙을 미러링했으므로 역이식 직관적.)
- **남은 sim2real**(배포 전 필수, [[sim2real-checklist-17dof]]): ①액추에이터 물리 실측(로터관성·기어마찰·감쇠=#1 갭) ②EKF estimator(2단계, sim서 노이즈·지연 주입 검증) ③접촉판정·지연·부호/오프셋 실측.

## SLAM/Navigation 통합 경계 — 별도 스택, 두 지점에 플러그인
nav는 `quad_ctrl` **안에 넣지 않는다**(제어 스택은 제어만). 이미 노출된 깨끗한 경계 2개에 꽂는다:
- **입력** `command/sport_client`(HighCmd: Move vx/vy/wz·steer) ← nav가 **cmd_vel** 공급
- **출력** `estimator/state`(base pose·vel) → nav가 **odom** 소비
```
[별도 nav_stack]  SLAM(VoxelSLAM live) → odom→base → planner → cmd_vel
        odom ▲                                              │ cmd_vel(HighCmd)
             └──────── estimator/state ◄── quad_ctrl ──► command/sport_client ◄┘
```
근거·설계=[[fullstack-integration-separate-project]]·[[odom-base-source-decision]]·[[voxelslam-live-hub-architecture-A]]. 즉 "아래 붙이기"가 아니라 **cmd_vel-in/odom-out 경계에 별도 프로젝트로 결합**.

## 현 상태 (2026-08-05) + 다음 트랙
- ✅ **구조·경계 설계 완료**: HAL(LowState/LowCmd)·State·command(HighCmd=cmd_vel)·estimator(odom)·config 3원칙. 골격 sim_main 빌드 OK.
- ⏳ **1단계 이관 미완**: 실제 A 로직(`quad/cpp` 수천 줄 wbic/mpc/gait/getup)이 `control/` 모듈로 아직 안 옮겨짐(현 모듈=인터페이스+TODO). = **배포의 최대 남은 작업(multi-session, 점진·항상 sim 초록불)**.
- **순서**: 1단계 이관(quad/cpp→control/, sim_main==trot_sim) → 2단계 EKF(sim) → 3단계 real_hal(biped/emb 패턴)+안전+RT. **배포급=A, 목표구조=quad_ctrl, 실기브리지=biped/emb 검증완료.**
