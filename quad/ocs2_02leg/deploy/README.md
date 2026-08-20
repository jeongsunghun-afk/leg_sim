# D1(OCS2 NMPC+WBC) 배포 골격

목표: 검증된 D1 제어(sim `test02legMujoco.cpp`)를 **재작성 없이 HAL 경계 뒤로** 옮겨,
sim(MuJoCo-as-model)과 real(실모터/IMU)에서 동일 제어가 돌게 한다. quad_ctrl 규약 재사용.

## 아키텍처 (quad_ctrl 규약 + D1 전용 TerrainProvider)
```
  RobotInterface.read → LowState(센서)  ─┐
                                          ├─ Estimator → State(컨트롤러 유일입력)
  TerrainProvider(지형 heightmap/SDF)  ──┘        │
                                                   ▼
                                          D1Controller.update(State, Terrain, cmd_vel)
                                                   │  = OCS2 MPC+WBC (test02legMujoco 제어 추출)
                                                   ▼
                                          LowCmd(tau_ff=관절토크) → RobotInterface.write
```
- **State**(quad_ctrl `estimator/state.hpp`): base pose/twist·q/dq·contact = 컨트롤러 유일입력.
- **LowState/LowCmd**(quad_ctrl `hal/robot_interface.hpp`): 센서/임피던스명령. D1은 kp=kd=0·tau_ff=WBC토크.
- **TerrainProvider**(D1 신규): perceptive용 지형질의(height·gradient). sim=mj_ray·real=인지 heightmap.
- **D1Controller**(D1 신규): OCS2 interface/mpc/mrt/wbc + SDF/region 소유. MuJoCo 비의존.

## 파일
- `terrain_provider.hpp` — 추상 지형 인터페이스(+MjTerrainSdf 어댑터).
- `d1_controller.hpp` — 제어 코어(OCS2 MPC+WBC), State+Terrain+cmd → tau. **test02legMujoco 추출**.
- `mujoco_backend.hpp` — D1MujocoHal(RobotInterface) + MuJoCo TerrainProvider. 데스크톱 검증.
- `real_hal_stub.hpp` — 실모터/IMU 자리(미구현, 배선 지점 표시).
- `d1_deploy.cpp` — 메인 루프(sim_main 미러). VIEW/헤드리스.
- `CMakeLists` 통합(ocs2_ws), `run_deploy.sh`.

## 단계
1. **[착수] 골격 + MuJoCo 백엔드**: 제어를 D1Controller로 추출, MuJoCo HAL로 sim서 재현
   (경사 등반 falls=0 = test02legMujoco와 동일 검증). ← 지금.
2. 상태추정기: sim=GT passthrough / real=InEKF(IMU+leg-odom+contact).
3. TerrainProvider real 구현(인지 heightmap). 4. real_hal(실모터/IMU 배선).

## 정직한 경계
- 지금 단계는 **sim 재현으로 HAL 경계 검증**이 목표. 실기 배선(real_hal·EKF·실 heightmap)은 미구현.
- D1은 OCS2/ROS2 스택 의존 → 임베디드 실시간 포팅은 별도 큰 작업(quad_ctrl A와 달리 heavy).

## 검증 (2026-08-20)
- **HAL 경계 작동**: d1_deploy가 State/Terrain/Command만으로 OCS2 제어 구동(MuJoCo 비의존).
- **평지**: 안정 보행 falls=0(base_z 0.53·tilt<1.6°).
- **경사(slope8 perceptive)**: 등반 falls=0, base_z 0.53→0.704. test02legMujoco(0.804)와 차이는
  로봇-상대 참조(배포=항상 명령구동) vs 절대 램프 차이(측정 0.705≈)와 정확히 일치 → 정상.
- 남은 실기화: ①상태추정기(sim=GT / real=InEKF) ②TerrainProvider real(인지 heightmap) ③real_hal(실모터/IMU).
