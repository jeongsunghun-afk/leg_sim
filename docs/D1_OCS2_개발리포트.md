# D1 개발리포트 — OCS2 통합 Perceptive NMPC → 02_Leg

> **상태: 활성 (2026-07-30).** 모델기반 배포 트랙. OCS2(ros2) 통합 perceptive NMPC를 02_Leg에 포팅.
> **Phase 1~2b 완료(동적 TROT 0.3m/s·13s+·falls=0, 2bb8dd4). ★Phase 2c에서 16-DOF 능동 발목 확장**
> (A정합 자세·GEARBOX·널스페이스 posture, 3751a7a→afa54bc). **★Phase 3a perceptive 발-지형 클리어런스 작동**
> (SDF 제약, gap 전진 x1.16→1.50, 5634041). **★Phase 3b 브리지 통합·검증**(발판배치+지형적응 base높이, a1c1841):
> 평지 falls=0 불변·**15° 램프(연속) 지형추종 등반 실증**(base 0.50→0.82·blind 불가)·**급단차(0.16m 이산)=동적 불안정**(swingHeight↑ 무효)=RL 영역.
> ★결론: D1 모델기반=**연속지형 지형적응 작동**, 이산험지=RL(DTC)로 수렴. 다음=Phase 4 벤치·연속지형 강건화.

## 0. 왜 OCS2 포팅인가

- **injection 한계**: A gait클럭+TAMOLS 주입=지형 크로싱은 되나 짜깁기라 마진 얇음(연속~0.24m·이산 형상민감).
- **Grandia perceptive NMPC(D1/ANYmal 실기)** = 하나의 폐루프 통합 NMPC가 base+발판+힘을 지형-aware co-optimize. legged_control(OCS2)이 그 구현.
- **OCS2 = 성숙 실시간**(SQP/DDP·auto-diff·HPIPM). B(aligator 20Hz)를 실시간化하는 것보다 OCS2 재사용이 효율. B승격 아님, DTC와 별개(배포용 모델기반 제어기).

## 0.1 이전 시도 — B(quad_centroidal) perceptive NMPC 승격 (보류 → 이 트랙으로 이관)

같은 "injection 마진을 통합 폐루프 perceptive NMPC로 근본해결" 목표를 처음엔 **B(`quad_centroidal_17dof.py`, 커밋 bea4a9a)를 Grandia식으로 승격**해 풀려 했다(2026-07-29 계획). B는 이미 kino-dynamic centroidal(다리관성)+발 결정변수+폐루프라 기반이 맞았다.
- **B 자산(유효)**: 17-DOF 안정보행 falls=0·~0.4 m/s(명령 0.5의 94%)·~20Hz(aligator ProxDDP RTI). 안정화 핵심=STIFF(바닥 강체화 접촉정합)·FOOT_DECISION·WBVY·WAIST_LOCK. **연속경사 perceptive**: 15° ramp에서 base z 0.50→0.54 지형추종 등반(tilt<5°, setHeightmap+HM_BASE).
- **왜 보류→D1**: B는 20Hz·0.4 m/s로 A(1kHz·~1.85 m/s) 실시간 열세. B 승격의 유일한 정당성("kino-dynamic이 여는 지형 통합 NMPC")은 **이미 실시간(HPIPM) 검증된 OCS2 계열로 얻는 편이 정공** — simple_mpc를 aligator에서 실시간화하는 건 솔버교체·코드젠·condensing의 큰 투자. → **B 승격 폐기, 통합 NMPC는 이 D1/OCS2 트랙으로 이관.**
- **남는 원리**: "centroidal도 접촉정합하면 반응형 발배치로 제대로 걷는다"(B flat config·연속경사 등반)는 실증은 유효. discrete(gap/stepping)·발판 XY 회피는 B 마진 밖=RL(DTC) 몫으로 확정.

## 1. 환경 (확인 2026-07-29)

- **ROS2 Humble**(`/opt/ros/humble`, colcon). OCS2 **ros2 브랜치**(leggedrobotics/ocs2, refs/heads/ros2).
- **pinocchio**(proxddp conda env), boost_system. MuJoCo·eiquadprog·qpOASES=conda/소스빌드.
- 클론: `quad/ocs2_ws/`(3rd-party 61M, **.gitignore**). 우리 작성 소스·수정노트=`quad/ocs2_02leg/`+`src/BUILD.md`.
- **legged_control 레퍼런스 클론**(대조용): `/home/jsh/legged_control`·`/home/jsh/legged_perceptive`.

## 2. 진행

### Phase 0 — OCS2 빌드 (환경 관문) ✅
- OCS2 ros2 브랜치 15팩키지 빌드(ocs2_legged_robot·python_interface·HPIPM). 검증=ocs2_legged_robot dummy MPC 구동.
- **함정**: `size_t 미선언`(→cmake `-include cstddef/cstdint`)·urdfdom(`apt liburdfdom-dev`)·ROS1 catkin(→ros2 브랜치)·conda pinocchio 충돌(`env -u PYTHONPATH`).

### Phase 1 — 02_Leg 모델 포팅 ✅ (커밋 dcb80b1)
- **reduced URDF** (`quad/ocs2_02leg/urdf/02leg_ocs2.urdf`): 17-DOF에서 **발목(foot_joint)×4 + FB_waist를 fixed로 잠근 12-DOF point-foot**. pin 관절순=[FL,FR,HL,HR]×(hip,thigh,calf). 총질량 38.02kg 검증.
- **OCS2 config** (`quad/ocs2_02leg/config/`):
  - `task.info`: **jointNames(12)·contactNames3DoF(4)를 task.info서 로드**하도록 OCS2 `ModelSettings.cpp` 확장(`loadStdVector`, 하위호환=ANYmal 기본 유지). contactNames3DoF=[FL,FR,HL,HR]_foot_contact_link가 OCS2 `enum ModeNumber{LF,RF,LH,RH}`와 정확 정렬 → gait.info 무수정.
  - `initialState`/`reference.info`: **base_z=0.45**(발목잠금 보정, 무릎 16~19° 조건화)·standing 관절각=pinocchio IK 산출.
- **검증** (`test/test02legLoad.cpp`):
  - 로드: robotMass 38.02kg·stateDim24·inputDim24·contacts4·genCoord18 ✅
  - **STANCE MPC**: 총 수직력 373.9N ≈ mg 372.9N(힘균형) ✅ · SQP ~4.5ms/iter(실시간) ✅
  - **TROT MPC**: 대각쌍(FL+HR↔FR+HL) 교대 유각(Fz→0), 전환 t=0.35 정확, base_x 0.22m/s 전진 ✅
- **결론**: OCS2 통합 NMPC가 02_Leg 12-DOF로 물리 유효한 trot을 실시간 계획. **발목·허리 잠금(point-foot)이 유일 근사**(sim2real 갭 항목).

### Phase 2 — MuJoCo 폐루프 브리지 ✅
`ocs2_legged_robot/test/test02legMujoco.cpp`(MuJoCo를 conda서 링크·conda libstdc++로 ABI정합):
- 루프: MuJoCo 상태 → OCS2 centroidal state(`computeCentroidalStateFromRbdModel`) → SqpMpc 재계획(MRT) → ff토크(`computeRbdTorqueFromCentroidalModel` RBD 역동역학) + 관절 PD → MuJoCo ctrl. 발목·허리=0 홀드.
- **버그 2개**: ①액추에이터명 파생(`_joint` 접미사 제거) ②rbdState 레이아웃=[eulerZYX, position, jointPos, **angVel**, **linVel**, jointVel](pos↔euler·ang↔lin 순서 주의, diff=0 확인).
- **정지 STANCE falls=0**(base_z 0.445·tilt 0.5°): 폐루프 전체 정확(상태변환·측정상태 재계획·ff토크·base 피드백) 확인.
- 초기 동적 TROT 미달 → **ff+관절PD가 동적 게이트엔 구조적 불충분**(swing Cartesian 추종·접촉제약·base task를 QP로 안 풀어서) → **저수준 WBC 필요**(Phase 2b).

### Phase 2b — QP WBC + 동적 보행 ✅ (커밋 6ca4ced→ef72f8e→2bb8dd4)

**WBC 두 구현**:
- **custom weighted QP** (`src/wbc_02leg.hpp`): 결정변수 [q̈(18), f(12)]. hard=floating-base 동역학(6)·stance no-slip·swing force0·friction. soft=base 6D·swing 가속·f_des. τ=[Mq̈+h−Jcᵀf]_actuated. eiquadprog.
- **faithful legged 포트** (`src/wbc_legged.hpp`, 커밋 d59b39e): legged_control WbcBase+WeightedWbc 충실 이식. 결정변수 [q̈(18), f(12), τ(12)]=42, full-EOM(`[M,−Jᵀ,−Sᵀ]x=−nle`)·no-contact-motion·swing zero-force·torque limit·friction pyramid hard, base-accel-FF·swing·force soft. 솔버=legged와 동일한 **qpOASES 3.2**(coin-or 소스빌드 `~/qpOASES`, `WBC_USE_QPOASES`).

**핵심 버그·정식화 (실효 수정)**:
- **pinocchio ABI**: conda/include가 OCS2의 시스템 pinocchio를 shadow(FK가 비대칭 garbage) → CMake `-idirafter ${conda}/include`(conda 검색 맨뒤, 시스템 -isystem이 이김).
- **base 파라미터화**: OCS2 centroidal base=`Composite(Translation+SphericalZYX)`=**euler base(nq=nv=18)**, quaternion(19) 아님 → qPin/vPin을 [pos, eulerZYX, joints]+[linVel_world, eulerZYX_rate, jointVel]로(RbdConversions 규약). 결과=발 위치 정확·대칭, tau_WBC≈tau_ff(~1%, 동역학 정확).
- **useFeedbackPolicy=false**: MPC 피드백 정책이 jumpy 입력보정을 내고 WBC가 증폭 → 개루프 참조(MPC=계획, WBC=피드백 전담)로 전환.
- **nThreads=1**: nThreads=3이 MPC 비결정성(동일명령 |w| 0.34 vs 5.50)+첫 swing 불안정 유발 → 1로 결정적화(첫 swing |w| 5.5→0.34).
- **표준 τ=τ_ff+τ_pd** (Bellicoso 2016 ANYmal WBC, eth-50106-01, 식17-19): 모션task=`r̈_des(FF)+Kp·posErr+Kd·velErr−J̇u`. base task를 FF+PD로(basePd_ 기본 ON), 저수준 τ=τ_wbc(+선택적 관절 PD). A의 WBIC도 동일 구성(사용자 확인). ★단 **관절 kd 댐핑은 이상토크 MuJoCo서 WBC 정확토크와 충돌**(실로봇/Gazebo 액추에이터 모델용 층, 이상 sim 불요) → WBC 출력만 적용.

**★★대반전 — STANCE "붕괴"는 WBC 버그가 아니라 테스트 오염 (커밋 ef72f8e)**:
- 오랜 "legged WBC가 STANCE에서 <1s 붕괴, custom PD-base는 solid" 진단은 **비교 조건이 불공정**했음. 브리지 `vx` 기본값이 **0.3**이라 stance 테스트가 0.3m/s 전진목표로 돌아, 4발 고정 게이트가 못 걷는 목표를 쫓다 전복. 그 "aBaseFF 0.53·20N 전방력·다중원인(FF드리프트·댐핑·QP추격)"이 전부 이것.
- **VX=0으로 legged 원형(순수 FF·PD無·댐핑無) STANCE 완벽 안정**(z 0.440·tilt 0.2°·qpFail 7 동결). ⇒ **WBC 버그 없음. 솔버(eiquadprog↔qpOASES)·순수FF vs PD 논쟁·"로봇/물리 전이 실패"는 전부 오염 조건 위의 오진(정정)**.
- **교훈**: 제어 실패 진단 전에 명령/참조 조건(VX·게이트·타깃)부터 검증. (긴 디버깅 체인=warm-start·SqpSolver trust-region·g_max·relaxed barrier(SOFT_ZF)·DDP 백엔드 등은 이 오염 위에서 돌린 것이라 대부분 무효화됨. 실효로 남은 것=위 버그 수정 + nThreads=1 + τ_ff+τ_pd + useFeedbackPolicy=false.)

**★★★동적 TROT 성공 (커밋 2bb8dd4) — 근본원인=MuJoCo 세팅이 A와 달랐던 것**:
제어기·WBC 문제가 아니라 sim 세팅을 A 배포 제어기와 맞추자 동적 trot이 즉시 안정화됨.
- **timestep 2ms→1ms(1kHz)**: A와 동일(quad_control.hpp:57). 2ms stiff 접촉이 불안정(에너지 주입·QP실패·base pop), 1ms서 ~1.5s에 trot 안정.
- **바닥강성 solref 0.02→0.005**: A와 동일(quad_control.hpp:64). 발 침투 35mm→3mm, 동적 접촉 안정.
- **허리 홀드 단단히 KpW 40→300**: 무거운 몸통분리 DOF wobble이 주 불안정원 → 단단히 잡아 10s+ 안정.
- **base task 가중 W_BASE 1→50**: legged Go1 가중이 3배 중량 02_Leg엔 부족(z 드리프트) → 올려서 nominal 고정.
- τ_ff+τ_pd(FF+PD base) 유지.
- **결과**: VX=0.3 정확 추종(7.1s에 2.16m=0.30m/s), z=0.45 고정, tilt<4°, **qpFail=0 내내, 13s+ falls=0**, sched≈act(게이트 정확 추종).

**결론(Phase 2b)**: 정적 STANCE·동적 TROT 모두 MuJoCo 폐루프서 falls=0. 앞선 "동적 model-based는 접촉전환 벽=RL 영역"은 **오진(테스트 오염+sim 세팅 불일치)이었고, sim을 A와 정합하니 해소**. WBC·MPC standalone 계획 모두 정확 검증됨.

### Phase 2c — 16-DOF 능동 발목 확장 ✅ (커밋 3751a7a→ec8ccbe→afa54bc)

12-DOF point-foot(발목 잠금)의 한계(스윙 버징·발목 흡수 없음) → **발목 4개를 능동 관절로 되살려 16-DOF**(허리는 fixed 유지).
- **모델**: URDF 4개 `*_foot_joint` fixed→revolute(12-DOF 백업=`02leg_ocs2_12dof.urdf`). task.info jointNames 12→16·Q/R/initialState 28차원. WBC torqueLimits perLeg 자동(발목 peak 168Nm). **브리지 관절을 OCS2 jointNames로 동적구성(12/16 자동감지)**, 발목홀드 조건부.
- **★GEARBOX(반사관성) 필수**: A와 동일 이식(`dof_armature=I_rot·N²`+damping+friction, quad_control.hpp:107). GEARBOX off시 발목 flail 폭발(frontJvPk 475·1.8s 낙상), on시 4.9. **작은 발목이 반사관성 없인 flail** — 16-DOF 안정의 핵심.
- **★널스페이스 posture task**(wbc_legged, 기본 wPosture=0.5): 능동 발목 redundancy(4관절 3D발) 표류 제어 = A의 `swing_w_r/f`(whip 억제)와 동일. ~7s 낙상(발목 표류)을 해결.
- **★앞/뒤 발목 nominal 분리**(A 값: FRONT_ANKLE −0.5·REAR_ANKLE −0.3). A 확인=발목 nominal이 flail 직접 좌우(나쁜 각=155% 모터한계). nominal=0(구)이 flail 원인이었음.
- **★base 높이 A 정합**(0.45→0.50): pinocchio IK로 다리자세 재계산(`tools/ik_base.py`). 뒷발 앞으로(+0.10)해 **뒷다리 접힘**(사용자 피드백). swingHeight 0.05→0.08.
- **결과**: STANCE solid(base_z 0.515≈A 0.52·tilt<0.5°). TROT 0.30m/s·base 0.50·tilt<3°·qpFail=0·**버징 26→평균 1~2·10.9s+ 지속**.

### 고속 붕괴 진단 + 벤치마크

- **★고속 붕괴 원인=게이트 주기 미스케일링**: 고정 0.35s phase가 고속엔 너무 길어 발판 과전방 배치→지지부족→**base 침하(sink)**→붕괴(tip 아님). **주기 0.50s로 최대 안정속도 0.3~0.4→0.6~0.7 m/s(~2배)**. A의 속도의존 주기와 동일 원리(완전해결=walk/trot/run 프리셋).
- **벤치마크**: 속도 0.1~0.45 m/s 안정(구 주기)·외란 측방 push 40~80N(0.1s) 복구. 브리지 PUSH env.
- **A vs D1(현재)**: 평지속도 D1 ~0.7 m/s(주기 튜닝 후) vs A ~2.0 m/s. 험지=D1은 perceptive(Phase 3) 전엔 blind → gap 첫 플랫폼(x1.16) 낙상, A는 footScore로 건넘.

### Phase 3 — Perceptive (지형) 🔶 진행 (Phase 3a 작동·3b core)

**★핵심 발견**: `ocs2_perceptive`가 **이미 워크스페이스에 ROS-free 존재**(DistanceTransformInterface·ComputeDistanceTransform Felzenszwalb ESDF·EndEffectorDistanceConstraint) → grid_map/CGAL 우회. FootPlacement/FootCollision 제약은 순수 OCS2, CGAL은 ConvexRegionSelector 2함수에만.

**Phase 3a — 발-지형 클리어런스 SDF 제약 ✅ (커밋 5634041)**:
- `mj_terrain_sdf.hpp`: mj_ray heightmap → 2.5D 높이-SDF(`SDF(p)=z−h(x,y)`·쌍선형·중앙차분, `DistanceTransformInterface` 구현).
- `foot_terrain_clearance.hpp`: 스윙발만 활성(`SwitchedModelRefMgr` 접촉플래그)·`value=SDF(발)−clr`·`dfdx=∇SDF·발Jac`(legged_perceptive `FootCollisionConstraint` 포팅).
- 브리지(PERCEPTIVE=1): EE키네매틱스(CppAd)→StateSoftConstraint 주입(MPC 구성 전)·매틱 `sdf.update`. 인터페이스 `getMutableOptimalControlProblem()` 1줄.
- **검증**: 평지 회귀 OK. gap 지형 blind x1.16→**perceptive x1.50**(첫 플랫폼 진입, 발이 엣지 클리어). ★단 **클리어런스뿐=foothold를 플랫폼에 '놓지' 않음**(발판 여전히 blind Raibert)+지형base 없어 턱 오르다 전복.

**Phase 3b — 발판 배치 + 지형적응 base높이 ✅ 브리지 통합·검증 (커밋 a1c1841, core=79bdab7)**:
- `local_convex_region.hpp`: **CGAL 없는 발판영역 자체생성**(ConvexRegionSelector 대체). `walkable`(경사<tan25°∧|h−hSeed|<0.06=같은 단)·nominal→유효셀 나선스냅·walkable 박스성장→반평면 `A·p+b≥0`(4행)·valid/stanceEnd.
- `foot_terrain_placement.hpp`: 제약(순수 OCS2). **isActive=접촉∧valid∧time≥stanceEnd**(미래 착지 stance만 성형, 상류 `getFootPlacementFlags` 동일).
- **브리지 통합(PLACEMENT·TERRAIN_Z env)**: legged_perceptive verbatim 대조 이식(워크플로 3/3 정밀분석). ①발판배치 4발 soft 주입(`RelaxedBarrier 1e-2/1e-4`, clearance의 1e-3와 구별). ②매틱 nominal 씨앗(`base_xy+vel·Δt+Rz(yaw)·nominalOffset`≈`getNominalFoothold` FK, 평지 pitch≈0)+stanceEnd(`initStandFinalTime`, ConvexRegionSelector 로직 복제)→`region.updateFoot`. ③지형적응 base높이 = `[t,t+H]` 11노드 참조 `z=hS+comH/cos(pitch)`·`pitch=지형법선`. base x,y는 **원본 forward 램프 보존**(modifyReferences가 desired x,y 보존하듯 z/pitch만 덮어씀).
- **★smooth heightmap**(±0.14 box-avg + step=0.3 넓은차분) = legged_perceptive `smooth_planar` 대응. 원 mj_ray 계단이 날카로워 base z 급점프→전복이라, 스무딩으로 계단 앞서 점진 상승.
- **검증(정직)**: ①평지 회귀 PLACEMENT+TERRAIN_Z ON에서도 falls=0(0.30m/s·tilt<1.6°)=베이스라인 불변. ②**★15° 램프(연속지형): BLIND는 base z 0.50 고정→지형 못따라 x≈1.4 전복. Phase 3b는 base 지형추종 0.50→0.82(tilt<18°·qpFail=0)로 x≈2.1 등반(+0.32m)**=지형적응 base높이가 blind 불가 등반을 가능케 함(핵심 능력 실증). 크레스트(급단차)서 전복.
- **★연속지형 벤치(blind vs D1, 첫 tilt>90° base_x / falls, VX=0.2)**: rough(±0.04 굴곡) blind 1.47/7059 → **D1 2.12/994**(7× 적음). slope(15°) blind 1.36/5219 → **D1 2.65/958**(5× 적음, base 0.53→**0.84** 등반). ⇒ D1 지형적응이 연속지형서 일관되게 blind 압도(+0.7~1.3m·낙상 5~7×↓). 전복은 지형 이산난구간(rough 끝·램프 crest)서만.
- **★급단차(계단코스 0.16m) 미해결 — 근본원인 규명(정직)**: 처음엔 swing height(0.08<0.16) 탓으로 봤으나 **swingHeight 0.20으로 올려 재실험→여전히 전복**(hsteps x≈1.2·gap x≈1.9, 평지는 falls=0 유지). ⇒ **발 클리어런스 문제가 아니라 급단차 등판의 동적 불안정**(앞다리 등판시 base pitch 급증·뒷다리 못따라감). 지형변조 swing(SwingTrajectoryPlanner 높이 주입)도 이 문제는 못 고침(클리어런스 아님). **결론: 연속지형(램프/경사)=모델기반 작동 / 급단차·이산=동적 불안정=RL 영역**([[perceptive-nav-tamols]]·[[ci-mpc-track]]의 "이산 험지=RL" 프로젝트 논지와 수렴).
- **자산 재사용**: `getNominalFoothold`(Raibert항은 legged_perceptive서도 주석처리·미사용, 능동경로=desired FK−pitch offset)·`getPolygonConstraint`(순수 Eigen)·mj_terrain_sdf.

### Phase 4 — 교차 벤치 (2026-07-31, 실측)
동일 17-DOF·연속지형·VX=0.2, 첫 tilt>90° base_x / 상태(C++ trot_sim·D1 bridge):
- **A(반응형)**: slope x2.17·z0.838·falls0 완주 / rough x2.14·falls0 완주 = **연속지형 최강건**.
- **injection(TAMOLS 발판주입)**: slope/rough 모두 **x0.6 stall**(TAMOLS 명목발판 fwd0=0→A 전진추진 제거).
- **pure-online RSL**: **z0.16 붕괴**(yaw 0→−50°+횡드리프트=death spiral, TAM_DBG 실측).
- **D1(perceptive NMPC)**: slope x2.65·z0.84·rough x2.12 등반(이후 이산난구간서 전복).
- **결론**: 연속지형=A/D1 강건 / injection·pure-online RSL=미달(재앵커·발판·모멘텀률FF 부재, [[perceptive-nav-tamols]]·TAMOLS §6.1). D1 참조로 RSL 근본해결=사실상 D1 자체(이미 보유).

### Phase 4.5 — 험지 강건화 env 튜닝 캠페인 (2026-08-04, 27 config 실측·헤드리스)
사용자 "험지보행 뚫기" 요청 → D1 험지 env 노브 전면 스윕(SMOOTH_W·W_BASE·swingHeight·VX·gait·GAIT_T·momentum).
- **★속도 정정(중요)**: "D1 ~0.03× 실시간"은 **오측정** — `test02legMujoco.cpp:318 const bool view=getenv("VIEW")`가 값 아닌 **존재만** 체크 → `VIEW=0`도 뷰어 무한루프(simTime 무시). **VIEW를 unset하면 헤드리스 정상·~0.2~0.3× 실시간**(3s sim=14s 벽시계)=벤치/튜닝 실용화. 평지 t=141s·tilt<3° 견고.
- **외란(push)=재현되는 강점**: 측방 60·90N push **falls=0·tilt≤4°**(flat, W_BASE 20~50 무관). Phase 3 "60N marginal chaos"보다 개선(config 안정). D1 disturbance rejection 견고.
- **★crest(15° 램프 볼록엣지 ≈x2.6)=env 저항적(핵심 결과)**: SMOOTH_W(0.14~0.40)·W_BASE(15~100)·swingHeight(0.08/0.14)·VX(0.2~0.5)·gait(trot/static_walk)·GAIT_T·모멘텀 **전부 시험 → 어떤 config도 crest 크로싱 실패**. 패턴: compliant base(W_BASE↓)는 나쁜 참조와 덜 싸워 tumble 지연(더 멀리 등반, tilt 165→105→33°)하나 결국 crest서 tumble 또는 그 앞 stall. ★"W_BASE20 falls=0"(x2.32)은 **sim시간 artifact**(13s에 crest 미도달, 28s 주면 x2.55서 tumble169°). = **env로 안 뚫림**.
- **rough(±0.04)=marginal chaos**: reached_x 1.0~2.6로 config 미세변화가 결과 뒤집음(WB25가 VX0.3선 x2.58·VX0.2선 x1.47), 신뢰 완주 없음. Phase 4 "rough x2.12"는 그 chaotic 범위의 한 샘플.
- **이산(hsteps)=tumble**(x1.03)=RL 벽 재확인.
- **★진단·다음 레버**: 병목 = crest서 base terrain-z/pitch **참조 품질**(TERRAIN_Z가 forward-predict한 base_z가 볼록엣지서 flat-top 높이를 조기 명령 → 물리 지지와 불일치 → tumble; compliant base가 더 버팀=참조와 싸운다는 증거). **env 아닌 코드레벨 참조 셰이핑**(base-z를 도달가능 지지높이로 캡·crest 근처 pitch-rate 제한, `test02legMujoco.cpp:411-436`)이 다음 레버 — 단 OCS2 재빌드 필요·working 등반 깨질 위험. 그마저 볼록엣지=준-이산이라 궁극 험지 크로싱은 RL(DTC)로 수렴. **최적 지형 config(크로싱 아닌 참고)**: W_BASE≈20~25·SMOOTH_W0.25·swingHeight0.14(tumble 최소·최원거리 등반).

**★TAMOLS 교훈 적용 검토 (2026-08-04, 사용자 "TAMOLS서 배운거 적용 가능한지 확인")**:
- ① **base-z 참조 rate-limit(TAMOLS z-band 클램프+slew 교훈)**: D1 `test02legMujoco.cpp:411-436` TERRAIN_Z 참조에 상승률 캡 추가(`ZRATE` env 게이트, 미설정=원본, 51s 재빌드·마스터+워크스페이스 동기). **결과: 램프 tumble 못 막음**(ZRATE 0.12~0.20 전부 tumble). ⇒ **base-z가 병목 아님** — 램프면을 정확 추종(climb_z 0.9=램프높이+comH)하는데도 램프 중간서 넘어짐.
- ② **★진짜 병목 규명=capture-point 부재(결정적 대조)**: **A(배포 trot+perceptive)는 동일 15° 램프를 x3.23·z1.11·tilt1.2°·falls0로 안정 등반**(VX0.4는 top x3.9 끝단 dropoff서만 낙하). **D1은 램프 중간 x2.5서 tumble**(tilt146°+). ⇒ **15° 램프=모델기반 한계 아님(A가 넘음)·D1 특유 갭**. 차이 = **A의 capture-point(속도오차 반응 발배치)를 D1이 못 씀** — NMPC가 발판+GRF+base를 협조계획하므로 WBC 발target shift 이식은 실패(§리스크, "NMPC 내부 foothold cost로 가야=큰 작업"). **= pure-online TAMOLS가 실패하고 injection이 A의 capture-point 빌려 해결한 그 갭과 동일**.
- **결론**: 적용되는 TAMOLS 교훈은 base-z(z침하)가 아니라 **"capture-point=반응층 필수"**. D1 램프 실패의 진짜 레버 = **NMPC 내부 반응형 발배치**(env·base-z 참조론 불가). 이건 [[perceptive-nav-tamols]]·[[full-tamols-modelbased-tracker]]의 "예측 NMPC+반응 발배치 협조" 미해결 과제이자, 궁극 험지=RL(DTC) 논지와 수렴. **A는 왜 되나=loose-coupled(capture 발배치가 독립 반응)·D1은 tight-coupled(발+GRF+base 협조계획이라 반응 발shift가 계획과 싸움)**.

**★D1 경사 envelope 실측 (2026-08-04, 5~15° 램프, 사용자 "경사는?")**: D1 연속경사 등반 한계 ≈ **8~10°**. **5°·8°**=램프 top(x3.9) 넘어 완주(끝단 dropoff서만 낙하)·**10°**=상단 marginal(x3.08)·**12°**(x2.99)·**15°**(x2.5)=연속면 tumble. 가팔수록 낙하점 앞당김=**graded 마진 한계**. cf. **A(반응형)는 15° tilt1.2°·falls0 완주** → 격차(A≥15° vs D1~8-10°)=반응형 균형의 몫.

**★"지형 SDF 넣고 최적화까지 하는데 왜 더 강건 안 되나?" (2026-08-04 사용자 핵심질문)**:
- **SDF/perceptive가 주는 것 = 기하(kinematic) 적응만** — 발판배치·base높이/pitch·swing 클리어런스. **실제로 도움됨**(blind는 경사서 base 고정→x1.4 전복 / perceptive는 지형추종 base로 **8~10° 등반**). 즉 SDF는 "못 오름→오름"으로 robustness를 **올렸다**.
- **SDF가 안 주는 것 = 반응형 동적 균형**. 경사 실패는 기하 아닌 **동적 문제**(경사방향 중력=지속외란 + heavy-leg 모델오차 → 속도오차 누적). 증거: D1 base-z가 경사면을 **완벽 추종**(climb_z=램프높이+comH)하는데도 tumble = **기하는 풀렸고 균형이 안 풀림**.
- **왜 예측 NMPC가 자동으로 robust하지 않나**: ①NMPC=**nominal 모델 최적화**(명시적 disturbance/model-error robustness 없음, robust/stochastic MPC 아님) ②02_Leg **모델오차**(centroidal SRBD가 heavy-leg 모멘텀 미포착)→경사서 nominal-최적 plan이 어긋남 ③재계획 사이엔 WBC가 **stale plan 추종**(반응형 발조정 X). A의 Raibert/capture=**매스텝 속도오차 직접 상쇄 tight feedback**이라 "덜 똑똑해도 더 robust". = **지형 인지 ≠ 동적 robustness**(직교·상보). 둘 다 얻으려면 반응형 발배치를 NMPC 내부(foothold cost)로 = **DTC/Kim2025 방향**(어려움).

**★★capture-point NMPC 주입 실험 (2026-08-04, 사용자 "foothold cost에 넣어 계획+반응 협조 진행")**:
- **구현**: A식 capture(CoM 선속도+각운동량 → 속도오차 만큼 발판 shift)를 NMPC foothold seed(`region->updateFoot`)에 주입(`CAPTURE_K` env). 효과 약해 placement 박스 tight화(`CAP_BOX`)로 강제주입도 추가. 3회 재빌드(각 ~45s, 마스터+워크스페이스 동기).
- **★결과: 협조가 아니라 "싸움"**. ①**soft 주입**(CAPTURE_K, 큰 walkable 박스)=거의 무효(발이 ±12cm 박스 안서 NMPC 원하는 대로, reached_x 2.18~2.53 노이즈). ②**강제 주입**(tight-box ±3~5cm + capture)=**오히려 악화**(reached_x 2.37→**0.48~1.43** 조기 전복). tight-box 단독(capture無)도 악화(x2.04). ⇒ **soft=무효·hard=싸움 → 외부 주입 sweet spot 없음**.
- **★근본 원인=tight-coupling(구조적, 구현 문제 아님)**: D1 NMPC는 발판+GRF+base를 **함께 협조 최적화**하므로, 발판을 **외부서 강제하면 GRF/base 계획과 불일치→붕괴**. 리포트 "capture-point WBC 이식=실패(전복 악화)"를 NMPC-foothold 레벨서도 재확인. **A가 되는 건 loose-coupled**(MPC 힘계획 ↔ Raibert 발배치 ↔ WBIC 독립)라 반응 발배치가 힘계획과 안 싸움.
- **★결론**: 반응형 발배치는 D1 NMPC에 **외부 bolt-on 불가**. 반응 보정이 **NMPC 내부 비용**(속도오차를 발판+GRF로 함께 푸는 재정식화=큰 작업)이거나 **RL**(end-to-end 반응정책 학습)이어야 함 = **DTC 존재이유 실증**.
- **★★사용자 확정 원칙 (2026-08-04): "각 컨트롤러를 자기 표준에 충실히"** — A=레퍼런스(loose-coupled Raibert/capture, 유지) · **D1=D1 표준**(Grandia perceptive NMPC, tight-coupled: 발판=지형 feasibility 제약, 균형=재계획+피드백) · TAMOLS=TAMOLS 표준(GIAC 안정성 cone이 균형 담당+재-solve). **교차 이식(A-capture를 D1/TAMOLS에) 금지**. ⇒ 이 세션서 D1에 실험 추가한 **CAPTURE_K·CAP_BOX·ZRATE를 전부 revert**(A-graft 제거, D1 표준 복원·재빌드·평지/완만경사 회귀 falls=0 확인). 각 균형 메커니즘: A=Raibert/capture(반응 휴리스틱)·TAMOLS=GIAC(계획 안정성제약)·D1=NMPC dynamics+재계획. **경사/외란 반응성=A(또는 TAMOLS injection), D1=전신 perceptive 계획, 강건 험지=RL(DTC)** 역할분담이 아키텍처적으로 정당.

### Phase 5 — DTC (별도, 후속)
OCS2/TAMOLS 참조 → RL. NMPC 완성 후 or 병행.

### 고속 엔벨로프 확장 — base 회복 authority (2026-08-10)
발목 TSID 조사에서 출발(사용자 "TSID라 발목 조절 힘들다→HQP?"). D1은 이미 16-DOF 능동발목(null-space weighted posture 제어). 규명:
- **①비결정성**: run-to-run 편차 주원인=라이브러리 OpenMP 리덕션 순서 → `OMP_NUM_THREADS=1` 바이너리 내장(override 가능). 마진 튜닝 신뢰 회복.
- **②발목은 오답**: POST 튜닝=whack-a-mole(VX0.5 잡으면 0.4 깨짐). 발목 hard-constraint(ANKLE_HARD=TSID 내 발목만 strict="poor-man's HQP", 진짜 HQP계층 아님)도 weighted보다 나쁨(VX0.5 4/6<6/6). 실측 발목강성 유한+백래시(PACE)라 강체 불가→weighted가 sim2real 정합. **TSID 유지가 맞음**(HQP 이식 불요).
- **③진짜 병목=base 회복 authority**(터치다운 coupled base_z sink+tilt 회복 한계, 진단이 정확히 지목). 레버=**W_BASE**(base task 가중, trot 전용 150·bound/walk는 붕괴→50)·**MPC_HZ**(재계획률, 범용, 기본 50→100 승격). KP_B(base PD) 단독은 무효. 저속/stance 무회귀.

- **④[중대 정정] 반사관성 모델 불일치**(사용자 "댐핑 부족"·"MJCF armature 표준?" 지적, 커밋19aaa10): GEARBOX(plant armature)는 있으나 ROTOR_I 기본=**1e-4**(옛 placeholder, PACE 실측 7.4e-4의 1/7.4)·**컨트롤러 모델(pinocchio)엔 로터관성 전무** → plant는 무겁고 컨트롤러는 가벼운 줄 앎. ROTOR_I 작을 땐 불일치 작아 고속됨(비현실). **실측 7.4e-4면 불일치 커져 저-토크→고속 붕괴**. 수정=GEARBOX 기본 PACE화(7.4e-4·JFRIC0.38·JDAMP0.099) + **WBC M 대각에 반사관성 가산**(`M(6+j,6+j)+=Irot·N²`) → plant-컨트롤러 정합.

**결과 — 실측 PACE 물리 위 D1 강건 엔벨로프**(OMP=1, 각 4회 falls=0):

| 게이트 | 실측물리·수정前(불일치) | **실측물리·반사관성 정합 後** |
|---|---|---|
| trot | ≤0.3 (붕괴) | **≤0.8** |
| bound | ≤0.3 (붕괴) | **≤0.8** |
| static_walk | ≤0.3 | **≤0.3~0.4** |

⚠앞서 보고한 "trot≤1.1"은 **비현실 물리(반사관성 1/7.4)의 산물**이라 폐기. 실측-물리 정직 엔벨로프=**trot/bound≤0.8·walk≤0.3**. 여전히 **비배포급**(실시간 0.15~0.3×; A가 배포 본선). 커밋 6ca5050(OMP=1)·c21ab1c(W_BASE)·0e574a3(MPC_HZ)·**19aaa10(반사관성 정합=근본)**. 상세=[[controller-balance-mechanisms]].

### 실접촉 기반 제어(CONTACT_ACTUAL) — MPC-WBC 정합 문제 (2026-08-10, negative·revert)
사용자 요청②(발접촉상태 기반 움직임). WBC에 실접촉 오버라이드 훅(`setActualContact`, `d->ncon`→발별 감지)은 있으나 결함(기본 off). **falling-edge 디바운스·pure-actual 정식(motionC=forceC=touch·swingC=!touch) 둘 다 시도했으나 저속조차 붕괴**(trot0.3 falls>2000). **근본=MPC가 스케줄로 힘·base 피드포워드(aBaseFF)를 계획** → WBC가 실접촉으로 접촉집합을 바꾸면 **늦은착지 구간에 스케줄이 기대한 지지발이 실제 미접촉→지지 부족→base 침하→붕괴**. 스케줄 모드가 작동하는 건 낙관적 "닿았다" 가정+soft-contact가 미세 타이밍 흡수 덕. ⇒ **WBC 토글이 아니라 이벤트-기반 게이트**(실접촉→gait schedule 앞당김→MPC 재계획, OCS2 reference manager 온라인 갱신)가 필요=research-scope. tight-coupled NMPC의 구조적 한계([[controller-balance-mechanisms]] capture 이식 실패와 동형). **실험은 revert**(현 커밋 클린 유지).

### 연속지형 배포급 A·B·C (2026-08-12)
사용자 목표=D1 연속경사 배포급. 험지비교(A제어기 대비)서 D1이 경사서 낙상한 근본=**과속**(VX0.3이 지형서 과속, VX≤0.2면 slope8/12 강건 완주·등반). 3단계:
- **A. 지형 속도정책(TERRAIN_CAP, 커밋4f799f8)** ✅: perceptive 참조빌딩에 국소 경사/거칠기(현재 실제 base 위치서 전방1.0m SDF 샘플) 기반 전진속도 자동캡(경사>3.4°→0.2·거칠기>2cm→0.15). **과속명령 VX=0.3이 slope8/12서 자동감속(vxEff→0.15)→2/2 falls=0 완주·등반**. 평지 무영향. rough는 미해결(bump 시작부터+본질 마진 VX~0.15).
- **B. slope15°** ❌: 전 속도 실패=**균형 한계**(램프 오르며 tilt 성장 후 tips, 속도로 안 고쳐짐). steep-slope CoM/pitch 튜닝 필요·불확실 → **≤12° 수용**.
- **C. 실시간성** ✅(컨트롤러): **측정(i7-13700H, OMP=1) MPC solve 5.85ms<10ms(100Hz)·WBC 0.254ms<1ms(1kHz)=여유롭게 실시간**. "sim 0.15×"는 MuJoCo 1kHz스텝+단일스레드 직렬루프 아티팩트이지 컨트롤러 아님(실기=플랜트가 실로봇). **배포 실시간=타깃HW: x86 RT-PC 가능·Pi 과중(A가 Pi용)**. 옵션=호라이즌 다이어트(dt0.015→0.025, N67→40, ~40%경량).

**⇒ D1 연속경사(≤12°) = 온보드 x86이면 배포급 달성**(강건 envelope + 실시간 컨트롤러). 15°/rough/gap은 D1 실측물리 밖(A 또는 RL).

### [정정+추가진단] 강건 envelope·PACE·조인트한계 (2026-08-12)
- **★envelope 정정**: 반복 검증 결과 **강건 연속경사=slope8(~8°)만**(falls=0 일관). **slope12는 marginal**(초기 3/3은 운 좋은 표본, 이후 VX0.2/0.3서 낙상=D1 변동). "≤12° 강건"은 과대평가였음 → **정직 envelope=~8° 강건·10~12° marginal**.
- **PACE 적용 확인**(사용자 지적): GUI 포함 전 실행서 `GEARBOX=1 ROTOR_I=7.4e-04(PACE실측) JFRIC0.38 JDAMP0.099` 적용됨(startup [GBX] 프린트). run_gui 미오버라이드=기본 적용. → 붕괴는 PACE 미적용 탓 아님.
- **★조인트 각도한계=컨트롤러 부재 확인**(사용자 지적 정확): MJCF엔 17관절 range 있으나 WBC/MPC는 토크한계만·각도한계 없음 → 컨트롤러가 관절을 range 밖으로 몰면 MuJoCo 클램프와 싸워 붕괴(특히 극단 GUI 명령). **WBC에 조인트한계 PD-wall 부등식(jointLimitTask, JLIM=1 opt-in) 추가**. ⚠기본OFF: 이득이 D1 marginal 변동에 묻혀 미검증 + 슬로프 등반(한계근처 자세) 과잉제한 위험(marg/kp 튜닝 필요). WBC solve 0.254ms<1ms(계측 WBC_TIME).
- **GUI 붕괴 종합**: D1 marginal 지형 envelope(~8°) + 조인트한계 부재 + 과속/과조작 명령 복합. 속도캡(A단계)은 경사 과속만 방지.
- **★JLIM 튜닝 결과=하드벽 부적합 확정(2026-08-12)**: JL_DBG 계측서 **정상 운용도 관절이 MJCF 한계를 넘어감**(slope8: 발목 −0.089·calf −0.031rad 초과, MuJoCo 클램프). 즉 D1 등반 자세가 관절 한계에 걸쳐 있어 **하드 JLIM 벽은 등반을 못 만들게 막아 슬로프 붕괴**(발목 제외해도 calf가 넘어 slope8 falls=3373). 마진/게인/발목제외 튜닝으로 해결 불가. **⇒ WBC 하드벽은 이 로봇 부적합**(관절이 한계서 동작). 정도(正道)=①MJCF range가 실측보다 타이트하면 넓힘(JLIM 불요) 또는 ②MPC/OCP 계획단서 joint position limit 준수(reference가 한계 밖 자세 미명령, WBC 반응형벽보다 근본). JLIM=1 opt-in(발목제외)으로 잔존. 다음=실기 관절 range 스펙 대조.

## 3. 리스크 (정직)
- **OCS2 빌드=관문(해결)**: ROS2 의존·rosdep·conda pinocchio 충돌 → Phase 0에서 처리됨(BUILD.md에 재현 노트).
- **모델 포팅 근사(해소)**: 발목 잠금(point-foot)은 **Phase 2c에서 16-DOF 능동 발목으로 해소**(허리만 fixed 유지). 발목 능동화엔 GEARBOX(반사관성)·널스페이스 posture·앞뒤 발목 nominal이 필수.
- **게이트/속도 범위(배포성 진행, 2026-07-31, eb31e48)**: **GAIT_T 주기 스케일링 구현** → 안정 0.3~0.5(기본 0.70s)·**0.7(GAIT_T=0.5, falls=0)**로 확대(기본 주기선 VX0.7 낙상). ~1.0(GAIT_T0.45)=marginal(z0.48·falls758). **1.4~2.0(A급)은 full 고속 WBC튜닝(SW_KP·재기어·reach) 별도 필요.** 프리셋 권장 walk T0.7@0.3-0.5·trot T0.5@0.6-0.7.
- **외란 강건성(배포성 진단, 정직)**: 측방 push 40N/80N 복구·**60N 위상민감**(경계 근처라 **build/run마다 결과 변동=marginal chaos**, 124/0/837↔504/0/0). ★**W_AM은 A도 안 씀**(17dof서 W_AM=0, "외란복구 무의미·측방 오히려↑"). A의 실제 push복구=**capture-point 발배치(KCAP·(v_meas−v_des))+MPC 예측**. ★**capture-point를 D1 WBC swing target에 이식=실패**(CAPTURE_K 0.12/0.2 전부 전복 악화): NMPC가 발위치X+GRF/base를 협조계획하는데 WBC서 발target을 shift하면 계획-추종 불일치→붕괴(RSL과 동일 교훈=반응층이 예측계획과 싸움). **결론: capture-point는 WBC 애드온이 아니라 NMPC 내부(foothold cost)로 들어가야 계획+반응 협조**(더 큰 작업). GIAC(TAMOLS)와 capture-point는 상보(계획형 제약 vs 반응형 발배치, 같은 CoM-지지 물리).
- **legged 게인은 drop-in 아님**: Go1(12kg/Gazebo)↔02_Leg(37.9kg/MuJoCo)로 재튜닝 필요(W_BASE·허리 홀드·sim 세팅=1ms·solref 0.005·GEARBOX가 그 예).
- **Phase 3b 경량 근사**: CGAL(convex_plane_decomposition) 미설치 → 발판영역을 축정렬 박스(nV=4)로 근사(convex 다각형 대신). 복잡 지형서 정밀도 한계 가능(이후 nV 확장/실 convex화 여지). 지형 base pitch 적응(roll 제외)도 근사.

## 4. 현재 상태 / 다음
- ✅ Phase 0(빌드)·1(모델포팅)·2(브리지)·**2b(STANCE+TROT falls=0)**·**2c(16-DOF 능동발목·A정합 자세)** 완료.
- 🔶 **Phase 3(perceptive)**: 3a(발-지형 클리어런스 SDF)=작동. **3b(발판배치+지형적응 base높이)=브리지 통합·검증(a1c1841)** — 평지 falls=0·**15° 램프(연속) 지형추종 등반 실증(blind 불가)**·**급단차(0.16m 이산)=동적 불안정 미해결(swingHeight↑ 실험 무효)=RL 영역**.
- **★Phase 3 결론**: 통합 perceptive NMPC(D1)는 **연속지형(경사·rough)에서 모델기반 지형적응 작동**. 이산/급단차는 [[perceptive-nav-tamols]]·[[ci-mpc-track]]과 동일하게 **RL(DTC) 영역**으로 수렴. D1의 모델기반 가치=연속지형 강건·실시간 배포.
- **다음(선택)**: ①**Phase 4 벤치**(D1 연속지형 vs injection vs A: falls/tilt/등반각/속도)·②연속지형 강건화(crest·rough·외란)·③이산험지는 DTC(Phase 5). ※급단차 모델기반 추격은 무효 확인(동적한계)이라 중단.
- **후속(대기)**: TAMOLS WBC 전환(D1 WeightedWbc를 TAMOLS 트래커로=TSID 개념정합, OCS2 의존성 분리 필요)·r_r2 torque-min 태스크(Bellicoso 식20).
- **재현/뷰어**: `quad/ocs2_02leg/run_view.sh [gait] [vx]`(GLFW). env 노브: WBC_LEGGED·W_BASE·POST·PERCEPTIVE·CLEARANCE·**PLACEMENT·TERRAIN_Z·SMOOTH_W**·STIFF·WAIST_KP·GEARBOX·JKD·NWSR·VX. 헤드리스=BUILD.md.
  예: `env WBC=1 WBC_LEGGED=1 W_BASE=50 VX=0.15 PERCEPTIVE=1 PLACEMENT=1 TERRAIN_Z=1 <exe> <cfg> mjcf/quad_terrain_slope.mjcf trot 16`
- **12-DOF↔16-DOF**: 16-DOF가 기본(config·URDF). 12-DOF=`02leg_ocs2_12dof.urdf`+git이력 보존. Phase 3 헤더=`src/test/`(mj_terrain_sdf·foot_terrain_clearance·foot_terrain_placement·local_convex_region), 인터페이스/CMake 변경=`src/patch/`.
- **상태추정·좌표 브리지**: 배포 시 `state_estimator.hpp` KF 재사용, base quat wxyz↔xyzw·pin↔mjcf 리맵은 `quad_centroidal.py` 패턴.

## 참조
- Grandia 2023 "Perceptive Locomotion through NMPC"(OCS2).
- legged_control(qiayuanl)·legged_perceptive — OCS2 quadruped WBC/perceptive 구현 참조(로컬 클론).
- OCS2 leggedrobotics/ocs2 (ros2 브랜치). TAMOLS injection=baseline.
- 관련 메모리: [[perceptive-nav-tamols]]·[[b-elevation-tamols-towr-track]]·[[d1-navbothub-mpc-amp-analysis]].
