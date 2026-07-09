[MPC]
ANYmal: SLIP-Convex MPC + WBC // Spring-Loaded inverted pendulum 기반(몸체는 하나의 점질량, 다리는 탄성이 있는 가상의 스프링)
: 몸체를 띄우고 튕겨내기 위한 가상의 스프링 힘 계산
Cheetah3: Linear Convex MPC + WBIC  // Single Rigid Body Dynamics 기반(몸체는 단단한 하나의 상자, 다리무게는 0이다.)
: 미래 경로를 따라갈 때 필요한 실제 지면 반력을 직접 계산
: 평소 주행은 Linear Convex MPC, 고기동 시 NMPC로 전환하는 전략. // "Real-time Optimal Landing Control of the MIT Mini Cheetah", 2021
: 현재 기개발된 코드 Di Carlo 2018 Linear Convex MPC (MIT Cheetah) 참조
: 그런데 우리의 하드웨어는 cheetah3와 달리 다리의 무게가 상당함. SRBD모델을 적용불가.
(다리를 들었을때, 무게중심이 삼각형 안에 들어와야함!!!(짤 확인) -> 다리가 엄청 가볍거나, 왼쪽 오른쪽 발의 간격이 넓어야함)

[NMPC]
- ANYmal: OCS2기반 Hierarchical MPC(SLQ) + WBC // Centroidal Dynamics 기반
: 여기까지는 MPC, WBC분리구조로 감.(SLQ: 연속 시간 제약 DDP), https://github.com/qiayuanl/legged_control/
- FDDP(수학적 오차가 조금 있더라도 일단 빠르게 궤적을 생성) // Full body dynamics 기반
: DDP/iLQR계열은 부등식 제약조건을 다루기 힘들어 SOCP풀기 어려움. 따라서 FDDP는 따로 WBC를 두어 SOCP문제를 풀도록하거나, 그냥 pyramid 모델로 풀려있음.  
- ProxDDP / Aligator : FDDP의 SOCP 한계를 극복한 ProxQLM 이나 ProxDDP의 차세대 솔버 개발(2025)
===============================================================================================================================================================================================================
[접촉모델] 물리엔진(mujoco, elliptic cone 지원), 제어기(A.다면 피라미드, 2.진짜 SOC제약)
- 대부분의 보행제어는 A.다면 피라미드(8~16면 피라미드)로 충분
- 현재 MPC의 접촉모델은 마찰 피라미드 (선형 4면, |Fx|,|Fy| ≤ μFz). 진짜 2차 콘 아님 → QP 유지용. μ=0.7 사용. 진짜 콘 √(Fx²+Fy²) ≤ μFz을 넣으면 SOCP(2차 원뿔 계획)가 되어 QP로 풀 수 없게 됨.(SOCP솔버필요)
- Mujoco의 접촉모델은 역시 pyramidal 콘, μ=0.8. 제어기(0.7)가 물리(0.8)보다 보수적이라 안전 측 
- 현재 NMPC의 접촉모델은 nmpc_fric_nf: int = 4, pyramid sides (4/8) (config.py:115) 파라미터가 있어 8면 피라미드를 지원함.
- 현재 WBIC·MPC는 선형 box pyramid + μ/√2 내접 마진(보수적 근사). 물리엔진은 elliptic cone.
- 정밀 일치하려면 제어기를 SOCP(2차 원뿔)로 정식화 필요 → QP 솔버(quadprog) 대신 ECOS/SCS 등 SOCP 솔버 도입. 실익 대비 우선순위 낮으나, 고마찰·고기동 보행 시 정확도 향상.
- proxDDP는 SOCP 솔버가 내부에 있음
==================================================================================================================================================================================================================================
[하체모델 MPC(Biped)]
- stance 결과분석
기본 홈자세는 다리가 자코비안 특이점 근처라 WBIC가 발산함. 좀더 안정적인 자세로 수정
biped 형태는 발바닥이 아니면 앞뒤로 쓰러짐. -> stepping 제어(lipm, march)로 잡아줘야함. -> 실패함, 몸통 pitch를 제어할 수 있도록 평발필요? 그리고 현재 점접촉이라 접촉 모멘트도 0임, 단일 접촉점이 모멘트를 못 만듦.
- lipm 결과분석
다리 떨림현상이있음. -> 게인, IK문제아님, swing 궤적 자체 문제아님?, step plan 문제, 마찰콘/mujoco 물리엔진 접촉모델 문제(점-접촉 트렁크문제)있음.
-> base 좌표계의 변화량이 최소가 되도록하는 step plan 필요.
-> 점-접촉 트렁크 문제는 발바닥이 변형되지 않는 이상 해결하기 힘듦.

[4족모델 MPC(Quadruped)]
- 기본 home이 q=0(다리 완전 직립) 이라, 발을 접지시켜도(ncon=12) 중력보상·강한 PD 둘 다 붕괴. 이건 biped와 동일한 직립-다리 특이점 문제 — 무릎을 편 상태론 못 섬
목표 CoM 변경으로 stance(MPC+WBIC) 성공
- walk 결과분석
제 walk가 실패한 큰 이유: CoM 위치를 고정하려 해서 동적 게이트와 충돌했습니다. baseline은 위치를 안 박고 속도+자세+높이만 추종하며, 균형은 MPC의 예측이 담당합니다.
자세 기울기를 못잡음 -> MPC_Q에서 roll, pitch의 각속도 가중치가 0임. 각속도 감소추가, 게인 올리고 테스트. -> 못잡음.

- 리팩토링 전 후 비교(참조소스코드 변경 후)
- 현재 mpc(50Hz)+wbic(500Hz) 제어기 qp구조 한계 레포팅 // inear(SRBD) Convex MPC + Pyramid 접촉모델로 구현되어있음.
- detection_contact -> schedule-primary
- MPC+WBIC(go2 어느정도 걸음, R.pet 정지까지)
- go2 mpc 안정화, go2 모델에서 조인트, 토크, 리미트와 충돌모델이 적용되었니?
- go2 ProxDDP -> 4사이클 보행(o) -> 12사이클 연속보행(x)
- R.pet ProxDDP -> 4사이클 보행(o) -> 12사이클 연속보행(o) -> Horizen는 괜찮음. 그러나 연속보행x -> RTI/MPC로 전환

[ FDDP 제어 루프 (Crocoddyl) ]
┌────────────────────────────────────────────────────────┐
│ 1. 입력: 목표 경로 (Desired Trajectory)                │
│ 2. 전신 역학 연산 (via Pinocchio Engine)              │
│    - Mass Matrix, Jacobian 계산                        │
│ 3. 다중 슈팅 (Multiple Shooting) 노드 배치            │
│ 4. 제약 조건 처리 (마찰원뿔 등)                       │
│    - 원형 원뿔(SOCP)을 사각뿔 형태로 다각형 선형화      │
└──────────────────────────┬─────────────────────────────┘
                           │ (선형화된 행렬 데이터 전달)
                           ▼
     ┌──────────────────────────────────────────┐
     │  하위 외부 수학 솔버 (ProxQP / OSQP 등)   │
     │  - 사각뿔 제약 조건 하에서 QP 최적화 수행  │
     └─────────────────────┬────────────────────┘
                           │ (최적 지면반력 및 상태 반환)
                           ▼
┌────────────────────────────────────────────────────────┐
│ 5. 최종 모터 토크 (τ) 및 피드백 게인 (K) 생성          │
└────────────────────────────────────────────────────────┘

[ ProxDDP 제어 루프 (Aligator) ]
┌────────────────────────────────────────────────────────┐
│ 1. 입력: 목표 경로 (Desired Trajectory)                │
│ 2. 전신 역학 연산 (via Pinocchio Engine)              │
│    - 대규모 전신 자유도 행렬 계산                      │
│ 3. 다중 슈팅 (Multiple Shooting) 노드 배치            │
└──────────────────────────┬─────────────────────────────┘
                           │ (전신 역학 데이터를 내부 연산기로 토스)
                           ▼
     ┌──────────────────────────────────────────┐
     │  알고리즘 내장 근접 연산기 (Proximal Loop) │
     │  - 외부 솔버 호출 없음 (No External Solver)│
     │  - 증강 라그랑주법 (Augmented Lagrangian) │
     │  - 원형 마찰원뿔(SOCP)을 그대로 원뿔에   │
     │    다이렉트 투영 (Cone Projection)       │
     └─────────────────────┬────────────────────┘
                           │ (자체 수렴 완료)
                           ▼
┌────────────────────────────────────────────────────────┐
│ 4. 완벽한 물리 법칙을 만족하는 최종 모터 토크 (τ) 출력 │
\(\tau (t)=u_{0}+K_{0}\cdot (x(t)\ominus x_{ocp}(t))\)u₀ (Feed-forward Torque): ProxDDP가 전신 역학을 고려해 계산한 기본 관절 토크입니다.
K₀ (Riccati Feedback Gain): ProxDDP가 계산해 준 최적 반사 신경(리카티 게인)입니다.
\(x(t) \ominus x_{ocp}(t)\) (State Error): 로봇의 현재 실제 관절 위치/속도와 MPC가 계획한 목표 위치/속도의 차이(에러)입니다.
└────────────────────────────────────────────────────────┘

                     [원조 DDP (1966년)]── (2차 미분이 너무 무겁고 복잡함)
                               │
                               ▼
           ┌─── [iLQR (2004년)] ─────────┐
           │            (1차 미분만 사용)            │
           │                                        │
    [연속시간(Continuous) 가지]              [이산시간(Discrete) 가지]
           │                                        │
           ▼                                        ├───────────┐
  [SLQ (2016년, OCS2)]                              ▼                        ▼
  - iLQR을 미분방정식으로                     [FDDP (2019년, Crocoddyl)]   [ALTRO (2019년, 미국)]
    연속시간화함.                             - 물리 법칙이 일시적으로       - 순정 iLQR 구조 외곽에
  - 다리 관성 무시하고                         깨져도(Infeasible)          '증강 라그랑주' 방패를
    센트로이달 모델 풀 때                      접촉 충격을 버티며          씌워 제약조건(장애물)
    주로 사용.                                 수렴하는 iLQR 업그레이드.    문제를 완벽 극복.
                                                    │
                                                    ▼
                                       [proxDDP (2023년, Aligator)]
                                       - FDDP의 초고속 연산 뼈대에
                                         Proximal 수학을 코어에 내장.
                                       - 이산시간 제약조건 방어력 끝판왕.

OCP(Optimal Control Problem) 계획
TSID-ID 튜닝노브
: kp_base/kp_contact/kp_posture, w_base/w_posture/w_contact_*, friction_coefficient. OCP: w_basevel(속도추종), w_frame(발), mu.

> Cost 튜닝, 접촉 모델 튜닝, 게인 튜닝, step plan 튜닝, MPC 타임스텝 튜닝, MPC 예측 수평선 길이 튜닝 등등
: 제어기, 물리환경 접촉 모델 불일치 // pinocchio(URDF):sphere(점접촉), MuJoCo(MJCF):box(면접촉)
: 빠른 주기의 gait가 더 안정적임.
- proxddp + WBC(TSID-ID)

이산 stitching (현재)
 ├─ 루프: 1사이클 반복 → seam(작은 limp)        ← re-anchor로 위치만 봉합
 └─ 재계획: 주기 완전수렴 → transient(전복)
        ↓ 같은 처방
연속 갱신 (RTI): 매 틱 1뉴턴스텝 → 이음매 없음 → 둘 다 해결

URDF -> SRDF(Semantic Robot Description Format)을 통한 모션플래너
SRDF는 URDF로 표현하지 못하는, 로봇의 의미론적(semantic) 정보를 정의하는 XML 포맷
예를 들어, 어떤 조인트들이 로봇의 "팔"에 해당하는지, 어떤 링크쌍의 충돌은 무시해도 되는지 등의 정보가 포함, MoveIt이나 모션 플래닝에서 많이 사용
ROS + MoveIt! + Gazebo/Webots 구조:
Gazebo(물리 엔진): URDF를 읽어서 로봇의 물리 법칙과 충돌을 시뮬레이션합니다.
MoveIt(모션 플래너): SRDF를 읽어서 "아, Gazebo 안에서 로봇이 움직일 때 이 관절들은 충돌 체크를 안 해도 되는구나" 하고 판단하여 경로를 계산합니다.

simple-mpc의 OCP/TSID는 강체 점접촉을 가정하고 GRF(지면반력)를 계산합니다. 
pybullet은 그 가정에 가까워 잘 맞지만, MuJoCo는 soft 접촉이라 발이 잠기고 실제 GRF가 가정과 달라 → tilt 드리프트·추진 부족. 
(우리 walk_loop은 반대로 MuJoCo-native라 MuJoCo서 잘 됨)
→ "simple-mpc가 MuJoCo서 나쁘다"는 컨트롤러 잘못이 아니라 접촉모델 불일치. MuJoCo 접촉을 강체로 튜닝(solref/solimp)하면 개선됩니다.

===========================================================================================================================================
26.06.15
구조1(proxddp) : horizen 검증, open loop 검증
- open loop에서 좌우비대칭으로 gait하는 limp(주기성)문제 발생 -> 접촉모델 위치 비대칭, foot_home 좌우불일치 
-> 좌우 hip의 관성이 매우 비대칭 -> 일단 같다고 고려하고 진행 -> 좌우비대칭이 문제가 아님 -> course-hold 게인문제아님 -> 해결못함.
- 초기자세를 기준으로 걷지 않는문제...
- 고속보행 진행
- 접촉점 확실한 위치로 생성필요.
PW_ORI=250 PW_BV=5 <- 이건 무슨옵션?
- step 위치 추종 문제(특히 y축) -> 속도가 아닌 위치오차에 의한 제어필요 피드백필요. capture-point IK? K_CAP=0.15(capture)/ 발착지점을 속도오차에 비례해 추가로 이동시키는 양. -> 구현방식문제 RTI로 재접근 
KinodynamicsID(centroidal, CoM+각운동량6D+kinematics)와 pinocchino(full body dynamics)의 차이로 NMPC+TSID가 붙질않음. 그래서 RTI를 검증하지못함. -> 구조1에서 시도, DDP게인으로 적용. -> 효과x -> 위치오차보정으로 변경
(centroidal+접촉력 => 토크)

- vel 0.3 -> 1m/s에 대한 각축 토크,각속도 추출
- sim2sim에서 좌측, 우측 이동 구현, 1m/s

구조2(proxddp+tsid) : horizen 검증, open loop 붕괴
- 다리 무게가 커서 tsid가 무너짐..?
- RTI가 적용안되어서?

[sim2real]
-하드웨어 추상화: 컨트롤러가 센서 read, 액추에이터 write를 인터페이스로만 하게 분리, MuJoCo impl+실로봇 ROS stub
-상태추정: 컨트롤러가 ground-truth 대신 추정상태 사용, 실로봇엔 참값이 없으니 필수
-도메인 랜덤화+센서노이즈/지연: 질량/마찰/지연/관측노이즈 랜덤화로 정책 강건화
-실시간성, 지연, 액추에이터 모델: 제어/센서 지연, 액추에이터 동역학 모델링


---

# 메모 (참조 노트)

참조1 cheetah3구조(quad_mpc_wbic.py) 
: Linear convex MPC(SRBD) + WBIC(full-body)
- mpc 측방보행, 선회 구현 // 보행 1m/s 이상 달성
- mpc gait 뒷다리 th2는 좀더 펴고 foot은 좀더 꺽는 궤적변경

참조2(quad_centroidal.py) 구조2의 C++변환 버전
: simple-mpc Kinodynamics(Centroidal) + TSID
- TSID가 go2에서는 되나 02leg에선 안됨. 다리무게 이슈로 판단

구조3(quad_fulldynamics.py) 
: NMPC(proxddp, full-body dynamics)+RTI, Riccati
- state estimate, SDK, 비동기 구현
- 실로봇 배포(RBQ 저수준 API 매핑) · (선택)속도상한↑·보행중 높이조절

구조2(qud_proxddp_tsid.py)
: NMPC(proxddp), RTI 결합, TSID 결합
- 서있기는 성공, 보행실패

구조1(quad_proxddp.py)
: NMPC(proxddp)
- 서있기는 성공, 보행실패
====================================================================================================
Online MPC  = 짧은 horizon + 무한루프(매 틱 현재상태로 재계획)  ← 무한루프가 online
Offline     = 긴 horizon(동작 전체) 1번 + 재생/추종           ← 재계획 없음

MPC: 궤적생성기(offline or online) - horizon이 있음(horizon을 푸는데 1사이클당 50step 33Hz)
WBIC: 추종기(online) 
SE: 상태추정

1kHz로 데이터는 던져주지만 계산은 33Hz당 한번씩 하게됨.

OCP가 한정된 주기(0.35s)안에 많은 계획을 갱신해야하는데 현재 33Hz로 재계획하고 주기당 12회 갱신+ 1kHz 리카티피드백
Full dynamics OCP는 30ms가 한계
Centroidal dynamics 으로 변환시


핵심: FullDynamics는 TSID가 필요 없음
FullDynamics OCP:  제어변수 = 관절 토크 (us)  →  토크가 OCP에서 직접 나옴
                   ★별도 ID/TSID 층 불필요
Centroidal OCP:    제어변수 = 접촉력 + centroidal  →  관절 토크 안 나옴
                   ★TSID/ID로 토크 매핑 필요
