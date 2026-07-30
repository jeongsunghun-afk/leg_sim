# D1 개발리포트 — OCS2 통합 Perceptive NMPC → 02_Leg

> **상태: 활성 (2026-07-30).** 모델기반 배포 트랙. OCS2(ros2) 통합 perceptive NMPC를 02_Leg에 포팅.
> **Phase 1~2b 완료(동적 TROT 0.3m/s·13s+·falls=0, 2bb8dd4). ★Phase 2c에서 16-DOF 능동 발목 확장**
> (A정합 자세·GEARBOX·널스페이스 posture, 3751a7a→afa54bc). **★Phase 3a perceptive 발-지형 클리어런스 작동**
> (SDF 제약, gap 전진 x1.16→1.50, 5634041). **★Phase 3b 브리지 통합·검증**(발판배치+지형적응 base높이, a1c1841):
> 평지 falls=0 불변·**15° 램프 지형추종 등반 실증**(base 0.50→0.82·blind 불가)·급단차(0.16m>스윙0.08)는 미해결.
> 다음=스윙높이 증가·급단차 안정화·진짜 홀 지형 검증 → Phase 4 벤치(vs A/injection).

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
- **검증(정직)**: ①평지 회귀 PLACEMENT+TERRAIN_Z ON에서도 falls=0(0.30m/s·tilt<1.6°)=베이스라인 불변. ②**★15° 램프: BLIND는 base z 0.50 고정→지형 못따라 x≈1.4 전복. Phase 3b는 base 지형추종 0.50→0.82(tilt<18°·qpFail=0)로 x≈2.1 등반(+0.32m)**=지형적응 base높이가 blind 불가 등반을 가능케 함(핵심 능력 실증). 크레스트(급단차)서 전복. ③**한계: 계단코스(0.16m 급단차)=swing height 0.08 초과라 발이 못 올라감→전복**(스윙높이 증가 별도 필요). **즉 연속지형(램프)=작동, 급단차=미해결.**
- **자산 재사용**: `getNominalFoothold`(Raibert항은 legged_perceptive서도 주석처리·미사용, 능동경로=desired FK−pitch offset)·`getPolygonConstraint`(순수 Eigen)·mj_terrain_sdf.

### Phase 4 — 벤치·판단 — 예정
- **D1(OCS2) NMPC vs injection vs A**: 동일 지형·지표(falls/tilt/외란복구/속도). 핵심 가설=통합 NMPC가 injection 마진 이김. 실기갭·실시간(OCS2 rate).

### Phase 5 — DTC (별도, 후속)
OCS2/TAMOLS 참조 → RL. NMPC 완성 후 or 병행.

## 3. 리스크 (정직)
- **OCS2 빌드=관문(해결)**: ROS2 의존·rosdep·conda pinocchio 충돌 → Phase 0에서 처리됨(BUILD.md에 재현 노트).
- **모델 포팅 근사(해소)**: 발목 잠금(point-foot)은 **Phase 2c에서 16-DOF 능동 발목으로 해소**(허리만 fixed 유지). 발목 능동화엔 GEARBOX(반사관성)·널스페이스 posture·앞뒤 발목 nominal이 필수.
- **게이트/속도 범위**: trot 0.3~0.45(구 주기)·0.6~0.7(짧은 주기) 검증. 고속=주기 스케일링(walk/trot/run 프리셋) 필요. crawl/walk 게이트·고속 외란은 미검증.
- **legged 게인은 drop-in 아님**: Go1(12kg/Gazebo)↔02_Leg(37.9kg/MuJoCo)로 재튜닝 필요(W_BASE·허리 홀드·sim 세팅=1ms·solref 0.005·GEARBOX가 그 예).
- **Phase 3b 경량 근사**: CGAL(convex_plane_decomposition) 미설치 → 발판영역을 축정렬 박스(nV=4)로 근사(convex 다각형 대신). 복잡 지형서 정밀도 한계 가능(이후 nV 확장/실 convex화 여지). 지형 base pitch 적응(roll 제외)도 근사.

## 4. 현재 상태 / 다음
- ✅ Phase 0(빌드)·1(모델포팅)·2(브리지)·**2b(STANCE+TROT falls=0)**·**2c(16-DOF 능동발목·A정합 자세)** 완료.
- 🔶 **Phase 3(perceptive)**: 3a(발-지형 클리어런스 SDF)=작동. **3b(발판배치+지형적응 base높이)=브리지 통합·검증(a1c1841)** — 평지 회귀 falls=0·**15° 램프 지형추종 등반 실증(blind 불가)**·급단차(0.16m>스윙0.08)는 미해결.
- **다음(Phase 3b 마무리)**: ①**스윙높이 증가**(급단차 클리어=현 0.08→계단높이 초과 필요, config swingHeight 또는 gait별)·②크레스트/하강 급단차 안정화·③실제 갭(무한바닥 없는 지형=진짜 홀) 검증용 지형 필요(현 지형들은 base 모델 무한 floor plane 포함→계단코스, 홀 아님). 그 후 Phase 4 벤치(vs A/injection).
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
