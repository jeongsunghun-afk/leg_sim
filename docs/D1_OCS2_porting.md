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

### Phase 1 — 02_Leg 모델 포팅
1. **URDF**: `02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf`(17-DOF, pinocchio 로드 검증됨) → OCS2 config.
2. **OCS2 config**(ocs2_legged_robot 참조): frame 이름(4발 contact·base)·관절순·질량/관성·gait 정의(walk/trot)·비용 가중·제약(마찰·kinematic·토크).
3. 검증: OCS2 MPC가 02_Leg로 평지 보행 계획 생성(dummy sim).

### Phase 2 — MuJoCo 브리지 (우리 sim서 검증)
1. OCS2 MPC 출력(base·발·힘·관절 참조) → MuJoCo(우리 sim)로 추종. towr_track_B 패턴(개루프 재생 아니라 **폐루프**: OCS2가 매사이클 측정상태서 재계획).
2. OCS2의 WBC(ocs2_legged_robot의 whole-body) or 우리 WBIC 재사용 선택.
3. 검증: MuJoCo서 평지 falls=0·실시간.

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
**Phase 0.1** — OCS2 클론 완료 확인 → colcon 의존성(rosdep) 파악 → 핵심 패키지 빌드 시도. 함정 기록.

## 참조
- Grandia 2023 "Perceptive Locomotion through NMPC"(OCS2). [[d1-navbothub-mpc-amp-analysis]]·[[perceptive-nav-tamols]].
- legged_control(qiayuanl, ROS1 Noetic) — OCS2 quadruped 구현 참조.
- OCS2 leggedrobotics/ocs2 (ros2 브랜치). TAMOLS injection=baseline([[b-elevation-tamols-towr-track]]).
