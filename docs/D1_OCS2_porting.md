# D1 포팅 — OCS2 통합 Perceptive NMPC → 02_Leg

> 작성 2026-07-29. injection(짜깁기·마진 얇음) 대신 **강건 모델기반 = OCS2 통합 폐루프 perceptive NMPC**(Grandia/legged_control 계열)를 02_Leg에 포팅. B승격 아님(B aligator=20Hz 실시간 열세). DTC와 별개(배포용 모델기반 제어기).

## 0. 왜 OCS2 포팅인가

- **injection 한계 규명**: A gait클럭+TAMOLS 주입=지형 크로싱 되나 짜깁기라 **마진 얇음**(연속~0.24m·이산 형상민감·"불안불안").
- **Grandia perceptive NMPC(D1/ANYmal 실기)** = **하나의 폐루프 통합 NMPC**가 base+발판+힘 지형-aware co-optimize → 강건. legged_control(OCS2)이 그 구현.
- **OCS2 = 성숙 실시간**(SQP/DDP·auto-diff·HPIPM류). B(aligator 20Hz)를 실시간化하는 것보다 OCS2 재사용이 효율.

## 1. 환경 (확인됨 2026-07-29)

- **ROS2 Humble** 설치 (`/opt/ros/humble`, colcon). OCS2 **ros2 브랜치 존재**(leggedrobotics/ocs2, refs/heads/ros2).
- **pinocchio** (proxddp conda env), **boost_system**(시스템). hpp-fcl·eigen 확인 필요.
- 클론 위치: `~/문서/jsh/ocs2_ws/src/ocs2` (colcon 워크스페이스). towr_ext 패턴 참조(standalone 우선, ROS/Gazebo 최소).

## 2. 계획

### Phase 0 — OCS2 빌드 (환경 관문)
1. OCS2 ros2 클론(진행 중). 의존성(pinocchio·hpp-fcl·boost·eigen·raisim optional) colcon rosdep 해결.
2. **핵심 패키지만 빌드**: ocs2_core·ocs2_ddp·ocs2_sqp·ocs2_centroidal_model·ocs2_pinocchio·ocs2_legged_robot. (raisim/rviz 등 시각화·하드웨어 제외 가능).
3. 검증: ocs2_legged_robot **dummy simulation** 실행(ANYmal 기본, MPC 돌아가는지). 함정=ROS2 msg·rosdep·conda pinocchio 충돌 주의.

### Phase 1 — 02_Leg 모델 포팅 ✅ **완료(2026-07-29)**
1. **URDF**: `quad/ocs2_02leg/urdf/02leg_ocs2.urdf` = 17-DOF에서 **발목(foot_joint)×4 + FB_waist를 fixed로 잠근 12-DOF point-foot**. pinocchio 관절순=[FL,FR,HL,HR]×(hip,thigh,calf). 총질량 38.02kg 검증.
2. **OCS2 config** (`quad/ocs2_02leg/config/`):
   - `task.info`: **jointNames(12)·contactNames3DoF(4)를 task.info서 로드**하도록 OCS2 `ModelSettings.cpp` 확장(`loadStdVector`, 하위호환=ANYmal 기본값 유지). contactNames3DoF=[FL,FR,HL,HR]_foot_contact_link가 OCS2 `enum ModeNumber{LF,RF,LH,RH}`와 정확히 정렬 → gait.info 무수정.
   - `initialState`/`reference.info`: **base_z=0.45**(발목잠금 보정, 무릎 16~19° 조건화)·standing 관절각=pinocchio IK 산출.
3. **검증** (`test/test02legLoad.cpp`, ocs2_legged_robot에 add_executable):
   - 모델 로드: robotMass 38.02kg·stateDim24·inputDim24·contacts4·genCoord18 ✅
   - **STANCE MPC**: 총 수직력 373.9N ≈ mg 372.9N(힘균형) ✅ · SQP ~4.5ms/iter(실시간) ✅
   - **TROT MPC**: 대각쌍(FL+HR↔FR+HL) 교대 유각(Fz→0), 전환 t=0.35 정확, base_x 0.22m/s 전진 ✅

**Phase 1 결론**: OCS2 통합 NMPC가 02_Leg 12-DOF로 물리적으로 유효한 trot 계획을 실시간 생성. 발목·허리 잠금(point-foot 단순화)이 유일한 근사(sim2real 갭 항목). 실행:
`ocs2_ws/install/.../test02legLoad task.info urdf reference.info [gait.info]`

### Phase 2 — MuJoCo 브리지 (우리 sim서 검증) 🔶 **부분완료(2026-07-29)**
1. **폐루프 브리지 구축** (`ocs2_legged_robot/test/test02legMujoco.cpp`, MuJoCo를 conda서 링크·conda libstdc++로 ABI정합):
   - MuJoCo 상태 → OCS2 centroidal state(`computeCentroidalStateFromRbdModel`) → SqpMpc 재계획(MRT, 50~250Hz) → ff토크(`computeRbdTorqueFromCentroidalModel` RBD 역동역학) + 관절 PD → MuJoCo ctrl → mj_step. **발목·허리=0 홀드**(OCS2 point-foot 정합).
   - **버그 2개 잡음**: ①액추에이터명 파생(`_joint` 접미사 제거) ②rbdState 레이아웃=[eulerZYX, position, jointPos, **angVel**, **linVel**, jointVel] (pos↔euler·ang↔lin 순서, diff=0 확인).
2. **검증 결과**:
   - ✅ **정지 STANCE falls=0**: base_z 0.445·tilt 0.5°·완전 안정(3s). **폐루프 전체 정확 확인**(상태변환·MPC 측정상태 재계획·ff토크·base 피드백).
   - 🔶 **동적 TROT 미달**: 전진(base_x 0→2m)하나 자세 발산·낙상. **게인(Kp60~400)·재계획률(50~250Hz) 스윕 전부 실패**. 
   - **진단**: 순수 PD(nominal 홀드)는 falls=0 → 관절매핑·역학 정상. **ff+관절PD가 동적 게이트엔 구조적 불충분**(swing 발 Cartesian 추종·접촉제약·base task를 QP로 안 풀어서). = legged_control이 **WBC 저수준**을 두는 정확한 이유.
3. **Phase 2b — QP WBC 신규 구현** 🔶 **진행중(2026-07-29, 사용자 선택=QP WBC)**: `test/wbc_02leg.hpp` = weighted QP WBC(legged_control식). 결정변수 [q̈(18), f(12)]. hard=floating-base 동역학(6)·stance no-slip(3/발)·swing force0·friction cone. soft=base/swing 발 가속·f_des 추종. τ=[Mq̈+h−Jcᵀf]_actuated. eiquadprog(conda, dynamic-double ABI안전).
   - **검증된 것**: 빌드·QP solve OPTIMAL·**no-slip 제약 만족**(|Jf·q̈+J̇v|=2e-14, 발 고정)·M/h/Jc 조립.
   - **미해결(핵심 난부)**: **base 6-DOF가 발 Jacobian의 null-space라 base-task 가중만이 잡는데, 그 정식화가 아직 정적 STANCE도 안정화 못 함**(W_BASE 1000~2000 스윕에도 base 서서히 이탈·낙상). = pinocchio floating-base q̈가 **local spatial 가속**이라 world PD 목표와의 프레임/Coriolis(ω×v) 정합이 필요(legged_control WBC의 핵심 미묘부). ff+PD 정적(2a)은 됐으나 WBC 정적은 base-task 정식화 완성 필요.
   - **다음**: base-task를 (i)centroidal momentum task or (ii)정확한 base-frame 6D task(spatial→classical 변환·Coriolis항)로 교체 → 정적 검증 후 trot. 대안=우리 WBIC 재사용(보류).

### Phase 3 — Perceptive (지형)
1. 지형 heightmap(mj_ray) → OCS2 footstep **SDF 제약**(Grandia식 edge/gap 회피) + terrain base.
2. 검증: 우리 지형(hsteps·dsteps)서 injection과 비교(마진·강건성).

### Phase 4 — 벤치·판단
1. **D1(OCS2) NMPC vs injection vs A**: 동일 지형·지표(falls/tilt/외란복구/속도). **핵심 가설**: 통합 NMPC가 injection 마진 이김.
2. 실기갭·실시간(OCS2 rate).

### Phase 5 — DTC (별도, 후속)
OCS2/TAMOLS 참조 → RL. NMPC 완성 후 or 병행.

## 3. 리스크 (정직)

- **★OCS2 빌드=큰 관문**: ROS2 의존·rosdep·conda pinocchio 충돌 가능. 수 시간~다세션. (towr_ext도 함정 다수였음).
- **ROS2 vs standalone**: OCS2 core는 ROS 최소나 legged_robot은 config/interface에 ROS 의존. standalone 빌드 가능성 탐색(towr_ext처럼).
- **모델 포팅**: frame 이름·gait·비용 튜닝 = 실측 반복.
- **폴백**: OCS2 빌드/포팅 정체 시 injection(작동함)이 모델기반 baseline 잔존. RL(DTC)이 최종 강건. B(quad_centroidal)도 대안(느리나 우리 것).

## 4. 즉시 다음
**Phase 2** — MuJoCo 폐루프 브리지. OCS2 SqpMpc 출력(base·발·힘·관절 참조)을 MuJoCo(우리 sim, `quad_real_17dof_waist_sphere.mjcf`)로 추종(발목·허리는 0 홀드 or PD). 폐루프=매사이클 측정상태서 재계획(MPC_MRT_Interface). 평지 falls=0·실시간 목표. WBC 선택=OCS2 whole-body vs 우리 WBIC.
- 상태추정=`state_estimator.hpp` KF 재사용. 좌표 브리지=`quad_centroidal.py`(base quat wxyz↔xyzw, pin↔mjcf 리맵) 패턴.
- 발목잠금 point-foot ↔ 실제 sphere발 접촉점 차이가 첫 갭. 필요시 발목을 4번째 관절로 되살려 16-DOF 재포팅.

**~~Phase 0.1~~** (완료) — OCS2 ros2 빌드(15팩키지)·Phase 1 모델포팅·MPC 검증 완료.

## 참조
- Grandia 2023 "Perceptive Locomotion through NMPC"(OCS2). [[d1-navbothub-mpc-amp-analysis]]·[[perceptive-nav-tamols]].
- legged_control(qiayuanl, ROS1 Noetic) — OCS2 quadruped 구현 참조.
- OCS2 leggedrobotics/ocs2 (ros2 브랜치). TAMOLS injection=baseline([[b-elevation-tamols-towr-track]]).
