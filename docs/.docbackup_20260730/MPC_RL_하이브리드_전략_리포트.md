# MPC 개발자의 전략 방향과 RL 하이브리드 로드맵

> R.pet 제어기 개발 맥락에서, MPC 기반 제어기 개발자가 팀 내 RL 담당자와 협업하며 추구할 방향을 정리한다. 하이브리드 아키텍처의 다섯 가지 패턴, 반드시 알아야 할 개념, 단계별 필독 문헌, 그리고 R.pet 적용 로드맵을 담는다. (작성일: 2026-07-09)

> **[현재 상태 연결 · 2026-07-10, v15.0]** 17-DOF 배포 baseline(보행 walk/trot/run/stairs·perceptive·점프 offline OCP→C++추종·자세·3레인 코스)이 완료됨([CHANGELOG](../CHANGELOG.md)). **다음 단계 = crocoddyl C++ 실시간 OCP**(점프를 offline replay가 아닌 C++ live-solve로): 이 문서 **§9(full-dynamics 실시간화 — RTI·호라이즌 다이어트·warm-start·모델 계층화)** 가 그 기술 로드맵이고, 성숙 후 **§8(DDP 스택→RL 레퍼런스 공급)·Phase H1(sit/getup RL 폐루프)** 로 이어진다. 즉 crocoddyl C++ 작업은 이 하이브리드 로드맵의 선행 인프라(궤적 생성기 실시간화)에 해당. ★이 문서는 **추후 TODO/로드맵 참조 문서**(RPET_JUMP_MPC·RPET_HEAD_GAZE_MPC와 동급).

---

## 1. 요약 — 결론 먼저

MPC 개발자로서 추구할 방향은 **"RL과 경쟁하지 않고, RL이 만들 수 없는 것을 공급하는 위치"** 다. 2024–2026년 판도에서 순수 MPC와 순수 RL의 대결 구도는 사실상 끝났고, 실기에 배포되는 최고 성능 시스템들(DTC 등)은 전부 하이브리드다. 이 구도에서 MPC 쪽 사람의 가치는 세 가지로 압축된다.

첫째, **RL의 교사(teacher)이자 데이터 생성기**. RL 정책이 모방하거나 추종할 최적 레퍼런스를 온라인으로 생성하는 능력은 MPC/OCP만 가진다. 둘째, **구조와 보장의 공급자**. 하드 제약, 안전 필터, 해석 가능한 비용 설계, 모델 기반 일반화는 RL이 원리적으로 약한 부분이다. 셋째, **모델의 소유자**. 동역학 모델, 접촉 정식화, 시뮬-실기 인터페이스에 대한 이해는 RL의 sim2real 성패를 좌우하는데, 이건 이미 R.pet URDF/MJCF 작업으로 확보한 자산이다.

따라서 단기 목표는 "내 MPC 스택을 RL 파이프라인의 상류(upstream)에 배치하는 것"이고, 이 문서의 나머지는 그 구체적 방법이다.

---

## 2. 왜 하이브리드인가 — 두 방법의 실증적 상보성

2025년 MuJoCo 기반 벤치마크(Go1, 동일 조건 직선 보행)가 상보성을 정량적으로 보여준다. RL은 외란 처리와 에너지 효율에서 앞서고(회복 시간 0.25~0.33초 단축, CoT 1.23 낮음 — 고주파 제어 입력 덕), MPC는 큰 외란으로부터의 복구와 안정성에서 앞서며 관절 간 제어 노력을 균형 있게 분배한다. 반면 RL은 학습 분포 밖 지형 일반화에서 약점을 드러냈다.

이 상보성의 근원은 구조적이다. MPC는 모델이 맞는 한 최적성·제약 만족·즉시 재계획이 가능하지만 모델 불일치에 취약하고, RL은 domain randomization으로 모델 불일치를 통째로 흡수하지만 태스크당 보상 설계·재학습 비용과 분포 밖 취약성을 안는다. R.pet에서 직접 겪은 사례가 정확히 이 구도다 — crocoddyl OCP는 rump 접촉 미모델(모델 불일치)로 실패했고, contact-implicit MPPI는 발견에 성공했지만 open-loop 강건성 한계(bounce 47°→60° 간극)에 부딪혔다. 그 간극을 메우는 것이 바로 학습된 폐루프 정책의 역할이다.

---

## 3. 하이브리드 아키텍처의 다섯 가지 패턴

문헌 전체를 관통하는 결합 패턴은 다섯 가지로 분류된다. 각 패턴마다 MPC 담당자가 공급하는 것이 다르므로, 어느 패턴을 택하느냐가 곧 역할 분담 설계다.

### 패턴 A — MPC가 교사: 레퍼런스 생성 + RL 추종 (★ 가장 추천)

모델 기반 planner(TO/MPC)가 최적 레퍼런스 모션을 생성하고, RL로 학습된 신경망 정책이 이를 강건하게 추종하는 구조. 대표작이 ETH의 **DTC(Deep Tracking Control, Science Robotics 2024)** 다. 학습 중에 model-based planner가 레퍼런스를 최적화하고 RL 정책이 이를 추종하도록 훈련하여, TO의 정확한 발 배치·계획 능력과 RL의 강건성·반사 행동을 결합했다. 실기 실험에서 미끄러운 지면 생존, 시각 정보가 고유수용감각과 불일치하거나 아예 없는 상황에서의 회복 반사까지 시연했고, 기존 SOTA MPC·RL 베이스라인 대비 우월한 강건성을 보였다.

**DTC 파이프라인 해부 (논문 정독 반영 — H2 설계의 직접 참조):**

```
[온라인 planner: TAMOLS]                    [RL 정책 (PPO, 오프라인 학습)]
지형 elevation map + 명령                    관측 = 고유수용감각 + 지형 샘플
  → 발디딤 + base 포즈 동시 최적화                 + 레퍼런스 "작은 서브셋":
  → touch-down 시점마다 재계획                     · 2D 평면 발디딤 좌표
     (매 틱 아님 — 가변 업데이트)                   · 목표 관절 각도 (IK 산출)
                                                  · 접촉 스케줄
                └──────── 레퍼런스 ────────────────┘
                                             → 관절 명령 (고주파 추종 + 반사)
```

- **레퍼런스는 "작은 서브셋"만 노출**: 정책에 planner의 전체 궤적(무거운 base 수식)을 다 보여주지 않고 발디딤 2D 좌표·IK 관절 각도·접촉 스케줄만 준다. 이 정보 병목이 모델 기반 제어의 취약점(상태추정·비전 오차 가정)을 정책으로부터 가려주어 강건성의 원천이 되고, 동시에 planner 교체에 대한 불변성을 만든다 — 학습에 안 쓴 전혀 다른 kinodynamic NMPC로 zero-shot 교체해도 발디딤 오차 3 cm.
- **가변 업데이트**: planner를 매 스텝이 아니라 touch-down 순간에만 재계산해 연산 절감. 단 ablation에서 업데이트를 50 Hz(MPC식)로 올리면 실패율 7.11%p 추가 감소 — 연산 예산이 되면 빠른 재계획이 이득.
- **IK 관절 각도를 관측에 포함**한 것이 학습 수렴의 결정타 — 신경망이 IK를 스스로 깨우치는 부담 제거 (ablation으로 실증).
- **학습 규모**: 4,096 병렬 로봇 × 2주, 76,800 m² 험지, 누적 23년 분량의 최적화 궤적.
- **성능**: 발디딤 평균 오차 2.3 cm, 흔들리는 널빤지(맵 오차 0.4 m)·젖은 화이트보드·비전 차단 계단에서 생존, 0.6 m 틈새·1.8 m 외나무다리 100%, 기존 RL 대비 380% 높은 0.48 m 클라이밍. 순수 RL은 stepping stone류의 희소 보상 문제로 실패하고, 순수 TO는 모델 가정이 깨지는 순간(보이지 않는 트랩) 고꾸라진다 — 하이브리드가 양쪽 실패 모드를 동시에 막은 실증.

변형으로 **완전 증류(distillation)** 가 있다: NMPC의 expert demonstration으로 단일 신경망 정책을 훈련해 다양한 사족 gait에 일반화시키는 방식(모방학습 계열). 이 경우 배포 시 MPC가 아예 사라지고 정책만 남아 계산이 극도로 가벼워진다.

**MPC 담당자의 공급물**: 레퍼런스 궤적 생성기(= 지금 가진 MPC 스택 그대로), 다양한 조건에서의 대량 최적 궤적 데이터셋, tracking 보상 설계 자문.

### 패턴 B — RL이 상위, MPC가 하위: 계층 구조

RL이 저차원의 고수준 결정(발디딤 위치, gait 파라미터, 접촉 의도)을 내리고, MPC/WBC가 이를 물리적으로 실현하는 구조. 2026년 1월의 계층형 RL–MPC가 최신 사례로, 상위 RL 정책이 접촉 의도(접촉 위치 + 접촉 후 서브골 자세)를 예측하고 하위 contact-implicit MPC가 접촉 모드를 온라인 재계획하며 실행한다 — RL은 기하 추론을, MPC는 고주파 동적 접촉 전략을 담당하는 역할 분담이다. 사족 분야의 고전으로는 RLOC(RL 발디딤 선택 + 최적제어 실행), GLIDE(centroidal 모델 가속도를 RL이 결정 + QP가 토크화)가 있다.

**MPC 담당자의 공급물**: RL의 행동 공간(action space) 자체를 정의하는 하위 제어기. RL 동료의 학습이 쉬워지느냐는 이 인터페이스 설계에 달려 있다 — 저차원·물리적으로 유의미·항상 실행 가능해야 한다.

### 패턴 C — RL 컴포넌트를 MPC 내부에 주입

MPC 정식화의 특정 요소를 학습으로 대체한다. 세 가지 하위 변형:

**C-1. 학습된 가치함수를 terminal cost로.** 짧은 호라이즌 MPC의 근시안을, RL이 학습한 Q함수/가치함수를 꼬리 비용(tail cost)으로 붙여 보완한다. 신경망 Q함수를 tail cost로 쓰는 cost roll-out으로 호라이즌에 지수적으로 증가하는 계산 복잡도를 완화하고, 순수 MPC가 실패하는 짧은 호라이즌에서 안정 보행을 달성한 사족 연구가 이 계열의 직접적 사례다. 이론적 원류는 POLO(Lowrey et al., 2019)의 "계획과 가치학습의 상호 보강" 프레임.

**C-2. 학습된 모델/잔차 모델.** 해석적 모델에 신경망 잔차를 더해 MPC의 모델 불일치를 줄인다(예: 액추에이터 네트워크 — Hwangbo et al. 2019의 핵심 기여이기도 함).

**C-3. RL로 MPC 하이퍼파라미터 튜닝.** 비용 가중치·호라이즌 등을 RL/베이지안 최적화로 자동 튜닝. 진입 장벽이 가장 낮은 협업 지점이다 — 지금 손으로 하는 2D 파라미터 스윕(T×STEP_H)의 자동화·고차원화라고 보면 된다.

**MPC 담당자의 공급물**: 미분 가능/평가 가능한 MPC 정식화, 어떤 요소가 병목인지에 대한 진단(예: "terminal cost가 근시안의 원인").

### 패턴 D — MPC가 안전 계층: safety filter / 잔차 정책

RL 정책의 출력을 MPC가 검증·수정하는 구조. Predictive Safety Filter(Zeilinger 그룹)가 대표 개념으로, RL이 제안한 행동이 제약을 위반할 궤적으로 이어지면 MPC가 최소 수정으로 안전 궤적에 투영한다. CBF(Control Barrier Function)와의 결합도 이 계열이며, CI-MPC 서베이도 CBF를 통한 안전 보장 통합을 유망 방향으로 꼽는다. 반대 방향인 잔차 정책(residual policy learning)은 MPC 출력 위에 RL이 작은 보정 토크를 얹는다 — MPC가 80%를 해결하고 RL이 나머지를 채우므로 학습이 극적으로 빨라진다.

**MPC 담당자의 공급물**: 실기 실험의 보험. RL 정책의 실기 테스트를 안전하게 만들어 팀 전체의 반복 속도를 올린다.

### 패턴 E — 샘플링 MPC + 학습된 prior

MPPI류 샘플링 MPC의 nominal 분포를 학습된 정책으로 warm-start하거나, 학습된 가치함수로 롤아웃을 절단한다. TD-MPC2(모델·가치·정책을 함께 학습하고 온라인에서 짧은 계획)가 학계 대표이고, DIAL-MPC 계열의 annealing과도 자연스럽게 결합된다. sit/getup에서 kinematic gather를 seed로 준 것이 사실 이 패턴의 수동 버전이다 — "좋은 seed가 성패를 가른다"는 교훈을 학습으로 자동화하는 것.

**MPC 담당자의 공급물**: 샘플링 MPC 인프라(이미 contact-implicit MPPI로 확보), 정책 prior 주입 인터페이스.

---

## 4. 반드시 알아야 할 개념 용어집

RL 동료와 같은 언어로 대화하기 위한 최소 개념 세트. RL 이론 전체가 아니라 하이브리드 설계에 직접 쓰이는 것만 추렸다.

| 개념 | 핵심 내용 | 하이브리드에서의 역할 |
|---|---|---|
| Teacher–Student / Privileged Learning | 시뮬에서만 접근 가능한 특권 정보(접촉력, 지형 등)로 교사 정책을 학습 후, 관측 가능한 정보만 쓰는 학생 정책으로 증류 | 패턴 A의 표준 학습 구조. MPC도 "특권 교사"가 될 수 있음 |
| Behavior Cloning / DAgger | 전문가 시연의 지도학습 모방. DAgger는 학생이 방문한 상태에서 전문가를 재질의해 분포 이탈(distribution shift) 해결 | MPC→정책 증류의 기본 도구. MPC는 임의 상태에서 재질의 가능한 이상적 전문가 |
| Domain Randomization (DR) | 질량·마찰·지연 등을 학습 중 무작위화해 sim2real 간극 흡수 | RL 강건성의 원천. MPC 담당자는 "무엇을 얼마나 randomize할지"의 물리적 근거 제공 |
| Asymmetric Actor-Critic | critic은 특권 정보를, actor는 실제 관측만 사용 | 학습 효율 핵심 트릭. 레퍼런스 궤적을 critic에만 주는 설계도 가능 |
| Value Function / Q-function | 상태(·행동)의 장기 기대 수익. Bellman 방정식으로 학습 | 패턴 C-1의 terminal cost 재료. "MPC 호라이즌 밖의 미래를 요약한 함수" |
| Residual Learning | 기존 제어기 출력에 학습된 보정을 가산 | 패턴 D. MPC 스택을 버리지 않고 RL을 얹는 최소 침습 경로 |
| Reward Shaping | 희소 보상을 조밀한 학습 신호로 변환 | MPC 비용 설계 경험이 그대로 이식되는 지점 — 비용 함수와 보상 함수는 부호 반대의 같은 물건 |
| Safety Filter / CBF | 정책 출력을 안전 집합 안으로 투영 | 패턴 D. MPC 담당자가 소유하는 계층 |
| Sim2Real Gap | 시뮬-실기 동역학 차이. 액추에이터 모델, 지연, 접촉 파라미터가 주범 | MuJoCo soft floor가 B안 속도를 왜곡했던 것과 동일 계열의 문제. 모델 담당자의 홈그라운드 |
| Curriculum Learning | 쉬운 태스크부터 점진적으로 난이도 상승 | MPC가 생성한 궤적 난이도로 커리큘럼을 설계할 수 있음 |

---

## 5. 필독 문헌 — 3단계 로드맵

### 1단계: 지형 파악 (RL 쪽 기초 + 대결 구도 이해)

| 문헌 | 왜 읽나 |
|---|---|
| Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots" (Science Robotics 2019) | 사족 RL의 원점. 액추에이터 네트워크(= 학습된 모델, 패턴 C-2)가 sim2real의 열쇠였다는 점이 MPC 개발자에게 특히 중요 |
| Lee et al., "Learning Quadrupedal Locomotion over Challenging Terrain" (Science Robotics 2020) | Teacher–student + privileged learning의 표준 레시피 확립 |
| Miki et al., "Learning Robust Perceptive Locomotion" (Science Robotics 2022) | ANYmal 실전 배포 RL의 완성형. belief state 인코더 |
| Akki et al., "Benchmarking MPC and RL for Legged Locomotion" (2025) | 동일 조건 정량 비교. 상보성 주장의 실증 근거 |

### 2단계: 하이브리드 핵심 (본 문서 패턴의 원전들)

| 문헌 | 패턴 | 왜 읽나 |
|---|---|---|
| **Jenelten et al., "DTC: Deep Tracking Control" (Science Robotics 2024)** | A | **최우선 필독.** TO가 레퍼런스 최적화 + RL이 추종 학습. R.pet에 가장 직접적으로 이식 가능한 청사진 |
| Gangapurwala et al., "RLOC" (T-RO 2022) | B | RL 발디딤 선택 + 최적제어 실행의 정석 |
| Xie et al., "GLIDE" (2023) | B | RL이 centroidal 가속도 결정 + QP 토크화. 단순하고 효과적 |
| Kovalev et al., "Combining MPC and Predictive RL for Quadrupedal Locomotion" (2023) | C-1 | Q함수 tail cost로 짧은 호라이즌 MPC 보강의 직접 사례 |
| Lowrey et al., "POLO" (ICLR 2019) | C-1/E | 계획↔가치학습 상호 보강의 이론적 원류 |
| Wabersich & Zeilinger, "Predictive Safety Filter" (Automatica 2021) | D | RL 실기 실험의 보험 설계 |
| Silver et al., "Residual Policy Learning" (2018) | D | 잔차 정책의 원전. 짧고 아이디어가 명확 |
| Hansen et al., "TD-MPC2" (ICLR 2024) | E | 학습 모델+가치+정책 위의 온라인 계획. 샘플링 MPC와 RL의 수렴점 |

### 3단계: 최전선 (2025–2026, 접촉·다중접촉 특화)

| 문헌 | 왜 읽나 |
|---|---|
| Kim et al., "Contact-Implicit MPC" (IJRR 2025) | 접촉 모드 사전 지정 없는 실시간 다중접촉 발견. aligator Phase 3의 직접 참조 |
| Pan et al., "DIAL-MPC" (ICRA 2025) | diffusion annealing 샘플링 MPC. training-free full-order 제어 |
| Suh, Pang, Tedrake 계열 "smoothing 분석" 논문들 | RL 성공 = 접촉 모드의 샘플링·평균화라는 통일 관점. RL 동료와의 논쟁을 정리해주는 이론 틀 |
| 계층형 RL–MPC contact intention (arXiv 2601.10930, 2026) | 패턴 B의 최신형: RL 접촉 의도 + CI-MPC 실행 |
| Frontiers, "Imitation Learning for Legged Robot Locomotion: A Survey" (2025) | MPC→정책 증류 연구 지형 전체 조망 |

---

## 6. R.pet 적용 로드맵 — 구체적 3단계

### Phase H0 (즉시, 협업 준비): 인터페이스와 자산 정리

지금 가진 자산을 RL 파이프라인이 소비할 수 있는 형태로 정리한다. (1) MJCF 모델 + 접촉 파라미터 문서화 — RL 동료의 시뮬 환경 셋업이 곧 sim2real의 절반이고, MuJoCo soft floor 이슈 같은 함정을 아는 사람은 나다. (2) 기존 컨트롤러(A안 MPC+WBIC, trot/run 프리셋, sit/getup 상태기계)를 "궤적 생성 API"로 래핑 — 임의 초기 상태·명령에서 레퍼런스 궤적을 뽑을 수 있게. (3) 비용 함수 설계 노하우를 보상 설계 초안으로 번역(부호 반전 + 조밀화).

### Phase H0.5 (★ 최우선 실행 과제): 학습된 RL 정책의 MuJoCo GUI 배포

RL 동료가 학습한 정책을 받아 **우리 MuJoCo 환경(C++ 뷰어/GUI)에서 돌리는 것**이 하이브리드 파이프라인의 첫 물리적 연결점이다. 이게 되어야 이후 모든 Phase(레퍼런스 공급→학습→재배포)의 왕복 루프가 닫힌다. 본질은 sim2sim 이식(IsaacLab/IsaacGym → MuJoCo)이며, 함정은 전부 인터페이스 세부에 있다.

체크리스트:

1. **정책 아티팩트 수령 규격 합의** — 파일 포맷(PyTorch `.pt` → ONNX 변환 권장, C++ 뷰어에는 onnxruntime이 최소 침습), 그리고 반드시 함께 받을 메타데이터: 관측 벡터의 **차원·순서·정규화 통계(mean/std)**, action 스케일·의미(보통 `q_target = q_default + a·scale`), 제어 주기(보통 50 Hz)와 decimation, 학습 시 PD 게인(Kp/Kd — 정책은 이 게인 전제로 학습되므로 그대로 써야 함).
2. **관측 구성기(obs builder) 구현** — legged_gym/IsaacLab 표준 관측(base 선속·각속도, projected gravity, 명령, q−q_default, q̇, 이전 action, [지형 샘플])을 MuJoCo `mjData`에서 동일 순서·동일 프레임으로 조립. 함정 3종: (a) quaternion 규약(MuJoCo wxyz vs Isaac xyzw), (b) base 속도의 표현 프레임(body vs world), (c) **관절 순서/부호 매핑** — URDF/MJCF 관절 순서가 학습 환경과 다르면 조용히 미쳐 돌아간다. 매핑 테이블을 단위 테스트로 고정.
3. **최소 루프 먼저**: python `mujoco.viewer` + onnxruntime로 50 Hz 정책 / 1 kHz PD decimation 루프 검증 → 통과 후 C++ 뷰어 통합. GUI에는 명령 슬라이더(vx, vy, wz)와 falls/추종 지표 오버레이.
4. **sim2sim 갭 진단 항목**: soft contact 파라미터, 액추에이터 모델(Isaac의 armature/damping vs MJCF 값), default pose 불일치. 정책이 Isaac에서 되고 MuJoCo에서 안 되면 이 셋이 1차 용의자 — B안 soft floor 전례와 같은 계열.
5. **판정 지표**: 평지 falls=0(20회), 속도 추종 오차, 그리고 **A안(Convex MPC+WBIC)과 동일 조건 나란히 비교** — 이 비교표가 §2 상보성 논의의 우리 로봇 버전 실증 데이터가 된다.

산출물: `rl_policy_runner`(obs builder + onnx 추론 + PD 브리지), 관절 매핑 테이블, RL↔MPC 비교 리포트 1장.

### Phase H1 (단기, 첫 하이브리드): sit/getup의 RL 폐루프화 — 패턴 A

이미 문서에 "완전 smooth는 RL 정책 필요 — 별도 대작업"이라고 적어둔 바로 그 항목이 최적의 첫 협업 과제다. 구도: **gather-seeded MPPI가 생성한 궤적(z=0.51 성공 궤적 + 다양한 초기 조건 변형)을 레퍼런스로, RL이 tracking policy를 학습**한다. DTC 레시피의 축소판이며, open-loop의 bounce 60° 간극(접촉 전환 타이밍·시작 상태 민감성)이 정확히 폐루프 학습이 잘 흡수하는 종류의 문제다. 성공 지표가 명확하고(bounce 각, falls), 실패해도 MPPI 궤적 자산은 남는다.

### Phase H2 (중기, 본류 통합): 보행 스택의 DTC화 — 패턴 A+B

A안(Convex MPC+WBIC, ~1.85 m/s) 또는 run 프리셋(T=0.40/STEP_H=0.08, ~2.18 m/s)을 레퍼런스 생성기로 쓰고, RL tracking policy가 실기 배포를 담당하는 구조로 확장한다. 여기서 WBIC가 하위 계층으로 남을지(패턴 B: RL이 발디딤/gait만 결정) 완전 증류될지(패턴 A: 토크 직출력)는 실기 계산 예산과 안전 요구로 결정한다. 병행하여 CI-OCP(aligator Phase 3)가 성숙하면 belly-flat 눕기 같은 보류 항목의 레퍼런스 생성기로 투입.

DTC 논문에서 그대로 가져올 설계 결정 4가지 (§3 패턴 A의 해부 참조):

1. **레퍼런스 노출은 작은 서브셋으로**: 정책 관측에 MPC 전체 상태 궤적이 아니라 (발디딤 목표 ptgt의 2D 좌표, IK/WBIC 산출 목표 관절각, 접촉 스케줄 cs)만 — 우리 파이프라인 ③(Raibert ptgt)·①(gait cs)의 출력이 정확히 이 서브셋이라 이식이 자연스럽다. 정보 병목이 강건성과 planner 교체 불변성(zero-shot)의 원천.
2. **목표 관절각을 관측에 포함** — 학습 수렴의 결정타(DTC ablation 실증). WBIC의 관절 목표를 그대로 노출.
3. **planner 업데이트는 touch-down 이벤트 기반**으로 시작하되(연산 절감), 예산이 되면 50 Hz까지 — DTC ablation에서 실패율 7.11%p 차이. 우리 MPC는 이미 50 Hz이므로 유리한 출발점.
4. **레퍼런스 다양성 = 일반화**: DTC의 23년치 궤적 규모까지는 아니어도, 지형·명령·초기조건 randomize를 레퍼런스 생성 단계(우리 담당)에서 확보하는 것이 정책 품질을 결정한다.

### Phase H3 (장기, 차별화): 안전 계층 소유 — 패턴 D

RL 정책이 실기에 올라가는 시점부터, MPC 기반 predictive safety filter(관절 한계·자세·접촉력 제약)를 상시 계층으로 배치한다. 이 계층은 팀에서 MPC 담당자만 만들 수 있고, 특허 관점에서도 "학습 정책 + 모델 기반 안전 필터" 조합은 방어 가치가 있는 구조다.

---

## 7. 역할 분담 제안 (한 장 요약)

| | MPC 담당 (나) | RL 담당 (동료) |
|---|---|---|
| 모델 | MJCF/URDF, 접촉 파라미터, 액추에이터 모델, sim2real 진단 | DR 범위 실험 |
| 데이터/레퍼런스 | MPC/MPPI 궤적 생성기, 커리큘럼용 난이도 조절 | 학습 인프라(IsaacLab/MJX), 병렬 시뮬 |
| 학습 | 보상 설계 공동(비용→보상 번역), tracking 오차 정의 | 알고리즘(PPO 등), 하이퍼파라미터, teacher–student |
| 배포 | 하위 WBC/토크 변환, safety filter, 실기 인터페이스(LowCmd) | 정책 추론 최적화 |
| 검증 | 모델 기반 상한/기준선(MPPI, CI-OCP) 제공 | 정책 성능 평가 |

핵심 원칙 하나로 끝맺는다: **"RL이 잘 되면 내 궤적·모델·안전 계층 덕이고, RL이 안 되면 내 MPC가 그대로 백업이다."** 이 비대칭이 MPC 개발자가 하이브리드 구도에서 갖는 구조적 우위다.

---

## 8. DDP 스택 → RL 지원 구조와 스케줄

전제가 되는 핵심 통찰: **RL에게 주는 것은 "실시간 제어기"가 아니라 "궤적과 신호"** 다. 교사 역할은 오프라인이므로 실시간성이 필요 없다. 따라서 aligator 속도 문제와 RL 지원은 디커플링되며, RL 협업은 지금 즉시 시작 가능하다. 그리고 DDP 계열에서 MPPI로의 "전환"은 없다 — DDP/OCP 스택이 주력(backbone)이고, MPPI는 접촉 발견이 필요한 예외적 문제에만 투입하는 보조 도구다(sit/getup에서 이미 실행한 그대로).

### 8.1 구조

```
[DDP 스택]                          [RL 파이프라인 (동료)]

① 궤적 생성기 (오프라인)  ──궤적 라이브러리──▶  모방/추종 학습 (BC, tracking reward)
   - Convex MPC+WBIC (실시간, 보행)
   - crocoddyl/aligator (느려도 OK, 정밀)
   - gather-seeded MPPI (접촉 발견)

② 온라인 레퍼런스 (학습 루프 내)  ──────────▶  DTC 스타일 학습
   - A안 Convex MPC만 사용 (이미 실시간!)

③ 검증 기준선  ────────────────────────────▶  RL 정책 성능의 모델 기반 상한/하한
④ safety filter (장기)  ───────────────────▶  실기 테스트 보험
```

설계 포인트: DTC처럼 학습 루프 안에서 planner를 호출하는 구조(②)에는 이미 실시간인 A안(Convex MPC+WBIC)을 쓴다. 병렬 시뮬 수천 개에서 매 스텝 호출되므로 빨라야 하기 때문이다. aligator full-dynamics는 ①에만 투입 — 배치로 밤새 돌려 정밀 궤적 라이브러리를 만드는 용도라 궤적당 10초가 걸려도 무방하다.

### 8.2 스케줄 (10주+)

| 주차 | DDP 트랙 (나) | RL 접점 |
|---|---|---|
| W1–2 | 인터페이스 동결: 궤적 포맷(q, v, τ, GRF, 접촉 상태, dt) 정의. 기존 컨트롤러를 "임의 초기조건 → 궤적" API로 래핑 | 동료와 포맷 합의, 시뮬 환경(MJCF) 인수인계 |
| W3–6 | **H1 파일럿: sit/getup 궤적 라이브러리** — gather-seeded MPPI + crocoddyl로 초기조건 randomize 궤적 수백 개 생성 | 동료가 tracking policy 학습. 성공 지표: bounce <47°, falls=0 |
| W3–6 병행 | aligator 속도 엔지니어링 (§9) — RL과 무관하게 독립 진행 | — |
| W7–10 | **보행 DTC화**: A안을 학습 루프 내 레퍼런스 생성기로 연결 | 동료가 보행 tracking policy 학습, run 프리셋(~2.18 m/s) 목표 |
| W11+ | aligator 안정화 후 정밀 maneuver(belly-flat 등) 궤적 공급원으로 추가 투입. safety filter 설계 착수 | 실기 이식 준비 |

W3–6에서 두 트랙이 병렬이라는 것이 핵심 — aligator 디버깅이 늦어져도 RL 협업은 멈추지 않는다.

---

## 9. aligator full-dynamics 속도 문제 진단 — 한계인가 엔지니어링인가

결론: **절반은 본질적 한계, 절반은 엔지니어링으로 회수 가능. 그리고 RL 지원의 선결 조건이 아니다.**

### 9.1 본질적 한계인 부분

고속 주행을 full-dynamics OCP로 실시간으로 푸는 것은 2026년 현재도 연구 최전선이며, 표준 관행은 **모델 계층화**다 — 고속은 단순 모델(SRBD/Convex MPC), full dynamics는 저속·접촉 정밀 maneuver 담당. C안이 <0.5 m/s에 머문 것과 속도를 올리면 수렴이 깨지는 현상은 이 판의 일반적 경험과 일치한다. 구조적 이유:

- 속도↑ → 사이클 T↓ → 호라이즌 내 접촉 전환 횟수↑ → 문제가 stiff해지고 최적화 지형에 불연속 급증
- flight phase 등장 → underactuation 구간에서 미분 품질 악화
- R.pet 특유의 **다리 무거운 질량 분포** → 질량 행렬 ill-conditioning(브리핑 문서에서 지목한 함정) → 스텝 크기 붕괴, line search 실패
- 접촉 타이밍 해상도 확보를 위해 dt↓ → 노드 수↑ → 연산량 폭증의 악순환

### 9.2 엔지니어링으로 회수 가능한 부분 (체크리스트)

1. **Real-Time Iteration**: 매 주기 수렴까지 돌리지 말고 1~3 이터레이션 + 이전 해 shift warm-start. full-dynamics MPC 실시간 사례(Crocoddyl 계열 ANYmal/Solo 50~100Hz)의 공통 비결.
2. **호라이즌 다이어트**: 0.5~0.7 s / dt 15~20 ms → 노드 30~45개. 고속일수록 호라이즌을 늘리고 싶어지지만 반대로 가야 한다.
3. **aligator 병렬 Riccati**(멀티스레드) 활성화 여부 확인.
4. **조건수 대응**: 상태·제어 스케일링(무차원화), ProxDDP proximal 파라미터(μ) 증대 — 다리 무거운 질량 분포에는 정칙화가 곧 수렴성이다.
5. **접촉 단순화**: 발끝 point contact만, collision pair 최소화, 접촉 완화 파라미터 조정.
6. **모델 강등**: run 속도 영역은 kinodynamic(B안 정식화)으로 내리고, full dynamics는 저속 정밀 동작 전용으로.

### 9.3 우선순위 판정

aligator 속도를 "먼저" 해결할 필요 없다. (1) RL 지원용 궤적 생성은 오프라인이라 현재 속도로 충분, (2) 온라인 레퍼런스는 A안이 이미 실시간, (3) 고속 full-dynamics 실시간 MPC는 로드맵의 최종 목표(Phase 3)이지 전제 조건이 아니다. aligator는 "저속·접촉 정밀 문제의 오프라인/준실시간 솔버"로 먼저 가치를 내게 하고, 실시간화는 RTI+warm-start부터 순서대로. 단, "run 속도를 full dynamics로 실시간"은 목표 자체를 재고할 것 — 그것은 모델 계층화로 푸는 문제다.

---

## 10. MPPI / ProxDDP로 풀 수 있는 문제 클래스 — 그리고 6-DoF 머리

### 10.1 ProxDDP(aligator): "접촉을 아는" 전신 협조

접촉 스케줄이 주어졌을 때의 정밀 전신 최적화가 강점이므로: 점프(이륙/비행/착지 타이밍 지정), 경사·계단 보행, 웅크려 통과(ducking), 좁은 발디딤 등의 **정밀 전신 궤적 라이브러리**; SRBD가 무시하는 다리 관성을 정확히 반영하는 **관성 커플링 동작**(다리 무거운 R.pet에서 특히 유효 — WBIC 각운동량 보상 task와 같은 물리를 OCP 레벨에서 다루는 것); 접촉 위치를 지정할 수 있는 **loco-manipulation**. 하드 제약(토크·관절범위·마찰원뿔)을 ALM으로 걸 수 있는 것이 Convex MPC 대비 결정적 이점.

### 10.2 MPPI: "접촉을 모르는" 발견

접촉 시퀀스 자체가 미지수인 maneuver — sit/getup(해결 완료), belly-flat 눕기/기립, 넘어짐 복구, 장애물 기어오르기; 몸통·주둥이로 문 밀기 같은 비정형 접촉 상호작용; CI-OCP 검증용 기준선 생성.

### 10.3 6-DoF 머리가 들어가면 — 새로운 문제 클래스 세 가지

머리는 단순 부속이 아니라 전신 최적화가 아니면 못 푸는 문제를 새로 만들어내는 확장이다.

**① 예측형 시선 안정화 (ProxDDP의 킬러 앱).** 특허의 듀얼 IMU 방진 피드백은 반응형(reactive)인데, MPC는 미래를 안다 — gait에서 향후 0.5초간 base 진동의 예측 궤적이 이미 손 안에 있다. 전신 OCP 비용에 "카메라 광축이 목표를 향한다"는 gaze task를 넣으면 보행 진동을 미리 상쇄하는 머리 궤적이 자동으로 나온다. **반응형 IMU 피드백(특허) + 예측형 MPC 피드포워드의 2층 구조**는 기술적으로도 특허 포트폴리오 확장(후속 청구항 또는 별건 출원) 관점에서도 자연스러운 조합.

**② 머리 = 반작용 질량(reaction mass).** 6-DoF 머리는 치타 꼬리처럼 각운동량 조절 장치로 쓸 수 있다. R.pet은 다리가 무거워 swing 다리의 각운동량 외란이 큰데, 머리를 반대 방향으로 흔들어 상쇄하는 전신 협조를 OCP가 스스로 발견한다 — WBIC 각운동량 보상 task의 확장. 급회전, 착지 자세 교정, 복구 동작에서 유효하며, MPPI 쪽 실험 아이템으로도 흥미롭다(복구 maneuver에서 MPPI가 머리 스윙을 자발적으로 쓰는지 관찰).

**③ 시선–균형 상충의 최적 중재.** "목표를 응시하며 험지 보행" 같은 태스크에서 시선 task와 균형 task가 충돌하는데, 이 우선순위 중재가 전신 MPC/WBIC 계열의 본령이다. 몸통이 기울 때 머리가 얼마나 보상할지, 머리 가동범위 소진 시 몸통 yaw를 얼마나 허용할지를 최적화가 결정한다.

### 10.4 주의점

- **차원 증가**: 17+6=23-DoF에서 MPPI raw 샘플링은 더 힘들어진다(고차원 약점). 머리는 저차원 파라미터화(gaze 방향 3-DoF로 축약)로 샘플링하거나, MPPI는 다리만 담당하고 머리는 해석적으로 푸는 분리가 현실적. ProxDDP는 다항 스케일링이라 23-DoF 자체는 문제없으나 §9의 속도 문제에 노드당 연산이 가산되는 것은 사실.
- **질량 분포 악화 가능성**: 머리 질량이 크면 원위(distal) 질량 증가로 ill-conditioning 심화 — 다리 무거운 분포와 같은 계열의 문제가 목에서도 생긴다. 설계 단계에서 머리 질량/관성 예산을 OCP 수렴성 관점에서도 검토할 것.

---

## 11. 기술 레퍼런스: MPPI 정리 — 원리·계보·최신 연구

> 본 리포트에서 반복 참조되는 MPPI의 자체 완결적 기술 정리. (Notion 「MPPI 정리」 페이지와 동일 내용의 통합본)

### 11.1 한 줄 정의

**MPPI = 제어 시퀀스에 노이즈를 얹은 수백~수천 개 롤아웃을 물리 모델로 시뮬레이션하고, 비용의 softmax 가중 평균으로 제어를 업데이트하는 샘플링 기반 MPC.** 미분이 필요 없어 접촉처럼 불연속인 동역학을 그대로 다룬다 — R.pet에서 crocoddyl OCP가 실패한 rump 접촉 전환을 MPPI가 해결한 이유.

### 11.2 알고리즘 (한 사이클)

1. nominal 제어 시퀀스에 가우시안 노이즈 주입 → K개 후보 생성: `u_k = ū + ε_k`, `ε ~ N(0, Σ)`
2. 각 후보를 동역학으로 전방 롤아웃, 누적 비용 `S_k` 평가
3. softmax 가중 평균으로 업데이트: `w_k = exp(−S_k/λ) / Σ exp(−S_j/λ)`, `ū ← ū + Σ w_k·ε_k`
4. 첫 제어만 실행, 시퀀스 shift 후 다음 주기 warm-start (receding horizon)

이론적 뿌리는 확률적 최적제어의 path integral / 정보이론적 유도 (Williams et al., 2017 "Information-Theoretic MPC"가 표준 정식화).

핵심 하이퍼파라미터:

| 파라미터 | 역할 | 튜닝 포인트 |
|---|---|---|
| λ (temperature) | 탐색↔수렴 다이얼 | 작으면 greedy, 크면 평균화·둔함 |
| Σ (노이즈 공분산) | 탐색 범위 | 작으면 국소최적 갇힘, 크면 유효 샘플 수 붕괴 |
| K (샘플 수) | 성능 직결 | 보통 GPU 병렬화 전제 (예외: Reference-Free) |
| 비용 함수 | 자유도 최대 | 비볼록·불연속 OK. 단 하드 제약 불가 → 페널티 처리 |

### 11.3 계보 — DDP 계열과의 위치

온라인 최적제어의 솔버 축은 두 갈래: **미분 기반**(DDP/iLQR, direct method, Convex MPC — 미분 필요·매끄러운 동역학 가정·수렴 빠름) vs **샘플링 기반**(MPPI, CEM, Predictive Sampling — 미분 불요·접촉 그대로 처리·샘플 수가 성능). 모델 충실도 축(SRBD→kinodynamic→full dynamics)은 별개 축이며, RL까지 치면 세 갈래 + 하이브리드가 현재 지형.

**R.pet 컨트롤러 지도에서의 위치**: A안(Convex MPC+WBIC)·B안(kinodynamic OCP)·C안(full-dynamics OCP)·aligator 로드맵이 전부 미분 기반이고, contact-implicit MPPI(sit/getup)가 샘플링 기반의 첫 실전 투입 사례.

| | 미분 기반 (DDP/OCP) | 샘플링 기반 (MPPI) |
|---|---|---|
| 접촉 처리 | 스케줄 사전 지정 또는 부드러운 접촉 모델 필요 | 물리엔진이 암묵 처리 (contact-implicit 공짜) |
| 비용 함수 | 미분 가능해야 | 제약 없음 (if문도 OK) |
| 계산 | CPU 실시간 가능 | 보통 GPU 병렬 롤아웃 필요 |
| 고차원 액션 | 강함 | 약함 → 저차원 파라미터화(스플라인·gait 파라미터)로 우회 |
| 수렴 보장 | 국소 수렴 이론 있음 | 약함, 노이즈 설정에 민감 |

### 11.4 최신 연구 동향 (2024 하반기–2026)

다리 로봇 적용의 고질병 — 타임스텝별 독립 노이즈가 만드는 비평활 제어, 접촉 전환 불연속에서의 고분산 — 을 푸는 방향으로 수렴 중. 세 갈래 트렌드:

**① diffusion 연결 + annealing으로 분산 문제 해결**
- **DIAL-MPC** (LeCAR Lab, ICRA 2025 best paper 최종후보): MPPI–단일스텝 diffusion의 이론적 연결에 기반, 노이즈 분산을 diffusion처럼 점진 축소하며 해를 반복 정제. full-order 토크 레벨 실시간 제어로 사족 점프·등반을 학습 없이 달성. MJX 기반 오픈소스 + sim2real 파이프라인 공개.

**② 저차원 파라미터화로 샘플 효율 극대화**
- **Reference-Free Sampling MPC** (2025말–2026): cubic Hermite 스플라인 파라미터화 + diffusion식 annealing으로 **CPU에서 30 샘플만으로 실시간**. gait 패턴·접촉 시퀀스 사전 정의 없이 trot~gallop·점프가 emergent하게 발현. Go2 실기 검증, 시뮬에서 백플립·물구나무·휴머노이드까지.
- **샘플링 전략 체계 연구** (arXiv 2601.01409, 2026.01): 구조화된 제어 파라미터화 관점에서 다리 로봇용 MPPI 샘플링 설계를 체계 정리 — 튜닝 레퍼런스로 유용.

**③ 알고리즘 하이브리드·보강**
- **MPOPI** (IROS 2025 계열): MPPI + CE + CMA(공분산 온라인 적응) 결합으로 샘플 효율 개선.
- **Feedback-MPPI** (2025): 롤아웃 미분으로 고주파 폐루프 보정 — 매 스텝 전체 재최적화 없이 저수준 제어기 대체 시도.
- **BC-MPPI** (2025): MPPI 약점인 제약 보장을 확률적 제약 레이어로 보완.
- **MTP** (TMLR 2025/ICLR 2026): 그래프 경로 샘플링으로 전역 다양성 확보 — MPPI/CEM의 국소최적 갇힘 겨냥. hydrax에 업스트림.

툴링:

| 도구 | 특징 |
|---|---|
| hydrax | JAX/MJX GPU 샘플링 MPC 라이브러리. MPPI·CEM·PS·MTP 일괄 지원, 온라인 domain randomization |
| dial-mpc | DIAL-MPC 공식 구현. Brax/MJX 기반, sim2real 파이프라인 포함 |
| MJPC | MuJoCo 공식 예측 제어 툴 (Predictive Sampling 원조) |

### 11.5 R.pet 적용 함의

- **이미 검증됨**: contact-implicit MPPI가 rump 접촉 전환(sit/getup)을 정공법으로 해결 — 접촉 스케줄이 본질인 maneuver에서 샘플링 접근의 구조적 장점이 실증됐다. `getup_mppi.py`는 CI-OCP 결과의 비교 기준선(모델프리 상한)으로 재사용.
- **aligator 트랙과의 관계**: 대체가 아닌 상호 검증. 미분 기반 CI-OCP가 gather성 접촉 시퀀스를 스스로 발견하는지를 gather-seeded MPPI 궤적(z=0.51) 기준으로 판정.
- **soft contact 반전**: aligator 로드맵에서 위험 요소였던 MuJoCo soft contact가 MPPI에서는 장점(롤아웃 시뮬레이터 = 모델, 불일치 원천 제거).
- **주의점**: 앞다리 3-DoF/뒷다리 4-DoF 비대칭과 다리 무거운 질량 분포 때문에 노이즈 스케일 Σ를 다리별로 다르게 잡아야 할 가능성 높음. raw 토크 공간(14~17-DoF) 직접 샘플링은 비효율 — Reference-Free식 스플라인 파라미터화가 유력한 다음 스텝.
- **탐색 한계 교훈**: 순수 MPPI 탐색은 gather 같은 비자명 전략을 스스로 못 찾았다(zf≈0.26 캡 오판의 원인). 좋은 kinematic seed가 성패를 갈랐다 — **MPPI는 최적화기이지 발견기가 아니라는 점**을 전제로 쓸 것.

---

## 부록: 즉시 실행 가능한 액션 아이템

0. **★ H0.5 착수: RL 동료에게 정책 아티팩트 + 메타데이터(관측 스펙/정규화/게인) 요청 → python mujoco.viewer 최소 루프부터** (§6 H0.5 체크리스트)
1. DTC 논문 정독 (arXiv 2309.15462) — 특히 학습 중 planner 호출 구조와 보상 설계 절
2. RL 동료와 1시간 세션: 위 5가지 패턴 공유 후 H1(sit/getup 폐루프화)의 인터페이스 합의 — "내가 궤적 N개를 어떤 포맷으로 주면 되는가"
3. gather-seeded MPPI 궤적 생성기를 초기 조건 randomize 가능하게 리팩터 (H1의 데이터 준비)
4. hydrax / dial-mpc 코드베이스에 R.pet MJCF 이식 실험 (패턴 E 탐색, 주말 프로젝트 규모)
5. Suh–Pang–Tedrake smoothing 관점 논문 1편 읽고 RL 동료와 토론 — 공통 언어 구축용
6. aligator: RTI(1~3 이터레이션) + shift warm-start 먼저 적용 후 속도 재측정 (§9.2 체크리스트 순서대로)
7. 6-DoF 머리 gaze task의 OCP 정식화 초안 작성 — 특허 후속 청구항 소재 검토와 병행 (§10.3-①)
