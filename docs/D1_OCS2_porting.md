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
3. **Phase 2b — QP WBC 신규 구현** 🔶 **진행중(2026-07-29, 사용자 선택=QP WBC)**: `test/wbc_02leg.hpp` = weighted QP WBC(legged_control식). 결정변수 [q̈(18), f(12)]. hard=floating-base 동역학(6)·stance no-slip(3/발)·swing force0·friction cone. soft=base 6D 가속·swing 발 가속·f_des 추종. τ=[Mq̈+h−Jcᵀf]_actuated. eiquadprog(conda, dynamic-double ABI안전).
   - **★버그 2개 잡음(핵심)**: ①**pinocchio ABI**: conda/include가 OCS2의 시스템 pinocchio를 shadow(FK가 비대칭 garbage) → CMake `-idirafter ${conda}/include`로 conda를 검색 맨뒤로(시스템 pinocchio -isystem이 이김). ②**base 파라미터화**: OCS2 centroidal 모델 base=`JointModelComposite(Translation+SphericalZYX)`=**euler base(nq=nv=18)**, quaternion(19) 아님 → qPin/vPin을 [pos(3),eulerZYX(3),joints] + [linVel_world(3),eulerZYX_rate(3),jointVel]로 재작성(RbdConversions 규약, `getEulerAnglesZyxDerivativesFromGlobalAngularVelocity`).
   - **결과**: 두 수정 후 **발 위치 정확·대칭**(FL[0.349,0.138,0] 등)·**WBC 토크가 ff 토크와 ~1% 일치**(tau_WBC≈tau_ff, 동역학 정확 검증).
   - **★정적 STANCE solid 달성(핵심 3번째 발견)**: 초기 marginal(tilt 느린 드리프트)의 주범은 **MPC 피드백 정책**(`useFeedbackPolicy=true`)이 jumpy한 입력보정을 내고 WBC가 증폭한 것. 진단=재계획 억제(0.3Hz)=안정 / 매스텝 재계획=발산 → **`useFeedbackPolicy=false`(개루프 참조: MPC=계획, WBC=피드백 전담)로 STANCE 5s falls=0·tilt 0.8°·base_z 0.450 완전 고정**. 이게 올바른 MPC-WBC 역할분담. (hard base·posture task·joint PD 첨가·게인↑은 모두 악화 확인.)
   - **🔶 TROT — trot의 open-loop 불안정을 피드백이 못 잡음(정밀 규명, 앞 "MPC가 회전계획"은 착시 정정)**:
     - QP 항상 solve(실패0)·**swing 발 완벽 추종**(des z≈act z). swing task 문제 아님.
     - **★★★핵심: standalone MPC(깨끗한 상태 단발 solve)는 완벽한 level trot 계획**(eulZYX pitch<1°·baseZ 0.45 유지·전진 매끄러움). **재계획 억제(clean 계획 추종)로 실행하면 |w|des(계획)≈0.05 인데 |w|act(실제)=5~16 폭주** → **계획은 level, 실제 로봇이 회전**. 즉 **trot swing(2발 대각지지)의 대각축 회전이 open-loop 불안정**(2점 지지=대각선 축 모멘트 0=underactuated)하고, **MPC 재계획(1-iter)+base task 피드백이 이를 못 잡음**. (앞서 폐루프서 |w|des=|w|act로 보인 건 계획이 매순간 측정상태서 시작하는 착시.)
     - **피드백 강화 시도**: SQP iteration 1→8 하면 **첫 swing 극적 개선(|w| 0.3~0.5)**. 그러나 ①**대각 SWAP(trot은 겹침없는 즉시 교체)서 스파이크** ②SQP8+500Hz재계획+standing_trot는 **MPC 수치발산**(|w|des→7521). 즉 강피드백이 방향이나 안정성·비용 관문.
   - **부분개선(적용)**: gait 파라미터 02_Leg 적응(swingHeight0.1→0.05·liftOff→0.1·touchDown→−0.2, ANYmal 과격) → 계획 여유↑.
   - **★★foot-lift 외란 = swing task 규명(추가 정밀진단)**: trot뿐 아니라 **crawl(3발지지)·준정적(VX=0.03)도 첫 발-lift서 즉시 |w|=4-5 폭주** → dynamic 균형이 아니라 **foot-lift 처리 자체** 문제로 좁힘. **swing task 끄니(W_SW=0) 첫 lift |w| 4.52→0.86 급감** = **swing 발 추종(kpF=400 고게인)이 다리를 급가속→base 반작용 외란**이 주범(WBC 전신동역학이 이를 완전 상쇄 못함, MPC fDes는 SRBD라 미반영). 단 gentle swing 게인(kpF 50~150)만으론 후속 mode 전환서 재발(force 재분배·이중 접촉전환). FullCentroidal(다리관성)도 무효=SRBD 문제 아님.
   - **★★★가장 깊은 병목 = 접촉스케줄 전환 warm-start(추가 규명)**: SETTLE 실험(1s STANCE 후 gait 개시)로 규명. **깨끗한 stance settle 중엔 |w|→0.02 완전안정·MPC 계획 정상**. 그러나 **gait 개시 순간(STANCE→swing 전환)에 MPC 해가 garbage(fDes −1060~+40484 N·|w|des 9~123·baseZ→1.6)** → 로봇 |w|=22 폭발. = **접촉스케줄 변경 시 warm-start(이전 stance 해)가 새 스케줄과 불일치, 1-iter SQP가 회복 못함**. SQP=8은 trot 첫swing엔 도움되나 **crawl에선 오히려 수치발산 심화**(fDes 40484) → gait별 solver 취약성 상이. **폐루프 MPC-WBC 캐스케이드**: 나쁜 WBC가중→실행악화→측정상태 이탈→MPC 재계획 garbage.
     - swing task도 부차 원인(W_SW=0시 첫lift |w| 4.52→0.86)이나 **주범은 전환 warm-start/solver conditioning**.
   - **★★★★메커니즘 수준 확정(병렬조사+계측, 2026-07-29 심화)**: OCS2 solver 병렬조사 + Fix A/B 계측으로 근본원인을 **정확히** 규명:
     - **계측 disambiguation**: ①전환서 `advanceMpc` **THROW 안 함**(MPC FAIL 로그 없음)=솔버 실패 아님. ②`coldStart=true`여도 |w|des 여전히 증가=seed 오염 아님. → **원인 확정=단일 SQP 스텝 overshoot**.
     - **정확한 메커니즘**: `SqpSolver`는 **trust-region/Levenberg 정규화가 없고 filter line-search만** 있음(`SqpSolver.cpp`). 필터의 `g_max`가 우리 task.info선 **1e-2**(코드기본 1e6)라, 전환 시 constraint violation g>1e-2면 **constraint-reduction 모드**(cost 무시, g만 줄이면 스텝 수용). stance seed 선형화점서 swing 동역학+마찰콘+zero-force가 갑자기 활성→ 선형모델상 g를 줄이는 **물리적 garbage 스텝(fDes 40484N·baseZ 1.6)**을 필터가 수용, 재선형화(추가 iter) 없어 교정 불가.
     - **config 레버 전부 소진**: SQP iter(3/5/8)·R정규화(1e-3/5e-3)·g_max(1/100/1e6)·coldStart·WBC가중 모두 테스트 → **어느 것도 robust하게 못 고침**. SQP=5가 crawl 첫전환 |w| 13→3 억제하나 여전 drift(baseZ 0.66), SQP=8은 crawl 발산. trot·crawl 병목 상이(단일 config 불가).
   - **★★전문가가이드 후속 검증(2026-07-30)**: 사용자 전문가 시퀀스(1 mode-consistent seed→2 relaxed barrier→3 DDP→4 SQP패치) 실행:
     - **#1 seed 검증=범인 아님**: `LeggedRobotInitializer::compute`가 `getContactFlags(time)`로 stance발에만 힘분배(swing zero-force) + warm-start도 `trajectorySpread`+`initializeStateInputTrajectories`가 mode-consistent 이전해 보간. **seed는 이미 mode-consistent** → 문제는 전환 순간 **state 자체가 "feet down인데 mode swing"**이라 단일스텝 overshoot.
     - **#3 DDP(SLQ) 백엔드=실패**: `GaussNewtonDDP_MPC` 토글(브리지 DDP=1) 추가. 전환서 **"DDP controller does not generate a stable rollout!" throw→latch**. **trust-region 있는 solver도 전환서 choke** → SQP trust-region이 유일원인 아님, 전환의 **hard 제약 스위치 자체가 문제**(#2 relaxed barrier가 더 유망).
     - **★nThreads 발견**: nThreads=3(SQP·rollout)이 **MPC 비결정성**(동일명령 |w| 0.34 vs 5.50) + 첫swing 불안정 유발. **nThreads=1로 결정적+첫swing |w| 5.5→0.34 안정**(채택). 단 trot은 여전 ~1s서 실패(스레딩은 한 기여요인).
   - **stance pose(사용자관찰 "뒷다리 splay")**: 발은 hip 바로아래(FL Δ+0.05·HL/HR Δ-0.03)=**balance 정상**. splay 외관=locked ankle(0) vs 실제굽힘(-0.3)로 뒷발이 뒤로뻗음(point-foot 근사 artifact). posture task로 관절규제 시도=stance 불안정화(wPost>0 falls)라 불가. 실 발목각 baking하면 자연스러우나 별도작업.
   - **#2 relaxed barrier=구현·실패**: 제약구조 확인(LeggedRobotInterface:180-190)=마찰콘은 이미soft(useHardFrictionCone=false), 그러나 **zeroForce·zeroVelocity·normalVelocity는 HARD equality**(g에 들어감). `SOFT_ZF` env로 zeroForce+normalVelocity를 QuadraticPenalty soft로 전환(코드추가). **결과=crawl 여전 실패**(|w| 25.76). 남은 zeroVelocity·다중 hard제약 스위치라 부분soft로 부족.
   - **★전환 불안정=MPC+WBC 복합(추가규명)**: 일부 전환서 **|w|act(13.4) >> |w|des(4.95)** = **WBC도 전환서 추종실패**(MPC 단독 아님). WBC측 원인=contact-mode 스위치 순간 발이 물리적으론 지면인데 mode가 swing이라 force=0 처리→그쪽 지지 상실→base 회전. 즉 **①MPC 단일스텝 overshoot + ②hard제약 스위치 + ③WBC contact-mode 불일치 + ④폐루프 오차누적** 복합.
   - **★★★legged_control 레퍼런스 대조(2026-07-30, 사용자 지시로 클론 `/home/jsh/legged_control`)**: 검증된 OCS2+WBC+물리 구현과 병렬 심층분석(워크플로 5에이전트). **★근본원인 코드확증**: 내 커스텀 WBC의 **base task=순수 PD**(kpB=kpO=100, w=10)인데 legged_control은 **base task=MPC 계획 모멘텀률 feedforward**(`WbcBase::formulateBaseAccelTask`: `b=Ab⁻¹·(m·정규화모멘텀률(uDes)−Ȧ·vDes−Aj·q̈)`, **PD 0**, w=1). 전환 임펄스로 측정 base ω 튐→내 kpO=100 PD가 증폭→폐루프 발산. legged는 FF라 이 되먹임 없음. (부차: 관절 kd=3 상시댐핑 부재, 가중비 swing:base:force 100:1:0.01 vs 내 20:10:1).
   - **★FF 포팅 시도·아키텍처 갭 규명**: legged식 momentum-rate FF를 브리지에 구현(`FF_BASE` 토글, `getNormalizedCentroidalMomentumRate`·`dccrba`·`AbInv`, updateDesired식 FK 순서). **t=0선 aBaseFF≈0(정상)** 이나 **순수/주로-FF가 정적도 미달**(드리프트 falls). 원인=**legged_control은 MPC를 별도 스레드서 연속 재계획**해 FF에 피드백 공급하는데, 내 **동기 50Hz 브리지**엔 그 구조 없음→FF가 피드백 없이 표류. 즉 **legged 동적 강건성=WBC 정식화(FF)뿐 아니라 전체 아키텍처(스레드 연속 MPC+FF WBC+ros_control)의 산물**. 단일 WBC 정식화 이식으론 부족.
   - **★스레드 연속 MPC 구현·검증(2026-07-30, option 1)**: legged_control LeggedController 패턴 이식 — MPC를 별도 std::thread서 `advanceMpc()` 연속호출, 제어루프는 매스텝 `setCurrentObservation`+`updatePolicy`(스왑)+`evaluatePolicy`, 실시간 페이싱(`MPC_THREAD=1` 활성, 기본=동기 빠른헤드리스). **스레딩 정상**(stance falls=0). **그러나 스레드+FF·스레드+PD 모두 trot 여전 실패**(|w|=12 @1s). = **legged 아키텍처(스레드MPC)+정식화(FF) 다 붙여도 02_Leg+MuJoCo서 trot 미달** → 남은갭=**워크플로 반증 예측대로 로봇(02_Leg 37.9kg·다리관성 vs Go1)·물리(MuJoCo 접촉 vs Gazebo ODE) 차이**. legged 게인·weight은 Go1+Gazebo 튜닝값이라 drop-in 아님. (부차: 내 FF에 nominal서도 잔여 pitch가속~0.28=미세 계산편차, SRBD com-to-contact 위치추정 관련 의심).
   - **★★★근본 구조차 확정(2026-07-30)**: legged 정확 config(FF·weight100:1:0.01·스레드·kd3) 전부 적용해도 **내 WBC는 stance조차 실패**(PD로는 solid). 원인=**내 커스텀 WBC가 legged와 근본적으로 다른 정식화**: legged는 **결정변수[q̈,f,τ]**(토크 명시변수)+**전체18행 EOM equality**(`Mq̈−Jᵀf−Sᵀτ=−nle`)+torque limits+friction+no-contact-motion가 hard, base FF·swing·force가 soft cost(qpOASES). 내 것은 **[q̈,f]**만+base 6행EOM+no-slip hard, 토크 사후계산=**다른(단순) WBC**. FF는 legged의 [q̈,f,τ]+full-EOM 구조용이라 내 구조엔 비호환. **→faithful 이식(legged WbcBase+WeightedWbc ~500줄 통째)이 정답**, 패치 불가.
   - **★legged_wbc faithful 이식 착수(`test/wbc_legged.hpp` 신규)**: legged WbcBase+WeightedWbc 충실 이식 — 결정변수[q̈(18),f(12),τ(12)]=42, full EOM(`[M,-jᵀ,-Sᵀ]x=-nle`)·no-contact-motion·swing zero-force(eq)·torque limit(84/84/126)·friction pyramid(ineq) hard, swing(Cartesian kp350)·base FF(모멘텀률)·force(fDes) soft. updateMeasured(crba·nle·j_·djv)·updateDesired(FF·swing목표) 원본대로. `WBC_LEGGED=1` 토글. **구조 검증=t=0 QP OPTIMAL(status=0)**. 그러나 **eiquadprog(Goldfarb-Idnani)가 상태이탈시 UNBOUNDED(status=2)/실패** — legged는 **qpOASES**(MPC용 active-set, semi-def H 허용) 사용. eiquadprog는 이 큰 제약문제(neq30·nineq44)에 robust 부족(reg 1e-2로 t=0은 풀리나 이탈시 붕괴). **남은 관문=qpOASES 솔버**(conda/system 부재, legged qpoases_catkin=소스빌드) + 02_Leg 재튜닝.
   - **★qpOASES 빌드·swap 완료(2026-07-30)**: legged의 정확한 솔버(qpOASES 3.2, coin-or 소스빌드 `~/qpOASES`, `setToMPC`·`enableEqualities`) 를 wbc_legged.hpp에 `WBC_USE_QPOASES` 매크로로 연결(H=AᵀA weighted·제약 lbA≤A·x≤ubA, legged WeightedWbc 정식 그대로). 링크·RUNPATH 확인.
   - **★★★결정적 음성결과**: **faithful legged 포트 + legged의 정확한 qpOASES 솔버로도 02_Leg가 STANCE에서 <1s에 붕괴**(z 0.45→0.58 상승 후 t≈1s tilt 147° 전복). base-FF on(W_BASE=1)·off(W_BASE=0)·여러 가중치(W_F 50/100) 전부 동일. 반면 **내 custom pure-PD-base WBC는 STANCE solid(falls=0)**. ⇒ 병목은 **솔버(eiquadprog↔qpOASES)도, base-task 단일선택(PD↔FF)도 아님**. legged WBC가 **02_Leg/MuJoCo로 그대로 이식되지 않음**(Go1 12kg/Gazebo → 02_Leg 37.9kg/MuJoCo: 관성·접촉솔버·액추에이터 모델 상이). eiquadprog↔qpOASES가 서로 다른 해를 고름(부족구속 nullspace, stance서 swing 0행·비용 미약)=cost가 42 결정변수를 충분히 앵커 못함. 실제 이식엔 base-accel-FF 프레임 재유도 + 3배질량 재튜닝 + 스레드MPC가 **동시** 필요(단발로 안 됨).
   - **★★종합 결론(확정)**: 동적 model-based(OCS2 NMPC+legged-WBC) 이식은 **정적 STANCE는 solid(custom WBC)·MPC standalone 정확**하나, **동적 보행용 WBC는 02_Leg에 전이 실패**. 솔버·단일 base-task가 아니라 **로봇/물리 전이 + Go1-특화 튜닝**이 벽. 추가 투자(FF 프레임 재유도·전면 재튜닝)의 산출물(모델기반 동적보행)은 **RL 트랙(타처 학습중)이 이미 커버** → 여기서 정지, 음성결과로 확정. 자산=faithful 포트(wbc_legged.hpp, 구조·솔버 완비, 재튜닝 시 출발점)·qpOASES 빌드·정적자세/MPC 파이프라인. **정직한 다음 옵션**: (a)**브리지를 스레드 연속 MPC로 재구성**(legged 아키텍처, MPC 별도스레드+WBC가 최신정책 스왑) → FF가 제대로 작동, (b)legged_control 전체 스택 이식(ROS1/Gazebo), (c)메모리 확정방향=RL(DTC) 피벗. **정적 STANCE solid·MPC standalone 계획 정확은 불변**. 실험토글=FF_BASE·SOFT_ZF·DDP·SWING_JOINT·SETTLE 전부 기본 off면 정적 solid.
   - **★뷰어**: `quad/ocs2_02leg/run_view.sh [gait] [vx]` (GLFW, 마우스 카메라). 정적 STANCE solid를 눈으로 확인 가능.

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
