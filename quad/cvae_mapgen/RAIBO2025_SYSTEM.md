# Raibo2025 전체 시스템 스펙 + 우리 DTC 갭 분석

Raibo2025 (Kim/Hwangbo, arXiv 2506.02835)는 **CVAE 하나가 아니라 3모듈 + 학습환경** 파이프라인. CVAE(=map generator)는 그 중 **학습용 커리큘럼** 한 조각. 나머지 부분("학습환경세팅 등")을 논문 supplementary에서 추출.

## 1. 전체 구조 (3 모듈)
| 모듈 | 역할 | 시점 |
|---|---|---|
| **Tracker** | 발판 추종 저수준 RL 컨트롤러(actor+estimator) | 학습+배포 |
| **Planner** | 실시간 발판 계획(sampling+filtering) | 배포 |
| **Map generator (CVAE)** | 지형 난이도 커리큘럼 | **학습만** ← 우리가 구현한 것 |

## 2. Tracker (저수준 RL) — Network details
- **Actor** MLP [512,128]. obs **167-dim** = O_p(자기수용: 자세·각속도·관절pos/vel) + O_h(이력 0.01/0.02/0.03s 전) + O_t(발당 미래 발판 2개=CoM→타깃 변위·time_front/back) + unObs(추정 선속도 3-dim). 출력 **12-dim 관절 위치타깃**.
- **Critic** MLP [512,128]. actor와 같으나 unObs 대신 **실제 선속도**(sim GT).
- **State estimator** GRU(128)+MLP[64,16]. 입력 42-dim(O_p+prev_action)→선속도 3-dim. **supervised**.
- **Contact estimator** MLP[256,64]→발 접촉 4-dim. 접촉 0.06s↑이면 target index 갱신.
- **PPO**(Table S1): num epoch 16·γ0.995·λ0.95·lr 2e-4·max_grad_norm 0.5.
- **Rewards**(Table S4): Target(sparse k_ts1=9.4/dense k_td=0.30/last k_tl=5)·Style(torque·slip·foot gather 종/횡[25cm]·**bound**[바운딩 게이트]·joint vel·stop·impact·smooth)·Constraint(joint limit). ★게이트=바운딩(우리=고정trot).
- **종료**: 내부 충돌 OR base tilt >110°.

## 3. 학습환경 세팅
- **300 병렬 env**, Raisim, **에피소드 4.2s**, **sim 2ms / control 10ms(100Hz)**, env당 **디딤돌 10개** 순차.
- **도메인 랜덤화**(Table S2): PD게인 ±10%·제어지연 0~30%·관측 ±10%·초기상태 ±10%·이력관측 ±10%·발질량 ±7%·베이스질량 0~40%·COM(0~15/2/2cm)·관성 0~50%·마찰 **0.4~1.0**.
- **커리큘럼**: 초기(Table S3 5-stage 고정률)→경쟁(CVAE map generator, Algorithm 1).

## 4. Planner (배포 실시간)
- sampling + 순차 filtering: performance filter(6 주변샘플·safe radius 6cm)·spike filter(거칠기)·collision filter(비행 중 충돌, boundary estimator=충돌체 높이 grid). 8후보→물리롤아웃 평가→best. detached thread 100Hz.

## 5. ★우리 DTC(RobotSW_IsaacLab=A 도메인) vs Raibo2025 갭
| 항목 | 우리 DTC(현재) | Raibo2025 | 갭 |
|---|---|---|---|
| Tracker 게이트 | **고정 trot** | **바운딩**(가변) | 큼 |
| 발판 planner | **오프라인 TAMOLS 캐시** | **온라인 sampling+filter** | 접근 다름 |
| 상태추정 | (제어기 KF) | GRU estimator(정책 내) | 다름 |
| 지형 커리큘럼 | **손수 10-lane**(frontier ~6-7) | **CVAE map generator** ← 이식 후보 | **CVAE가 대체/증강** |
| obs | proprio+heightmap+foothold | proprio+O_h+O_t+unObs(167) | 유사 |
| 도메인랜덤 | (일부) | Table S2 상세 | 보강 여지 |

→ **CVAE(커리큘럼)가 가장 이식성 높은 조각**(tracker 무관하게 학습 커리큘럼에 붙음). 우리 손수 10-lane을 CVAE로 대체.
→ tracker(바운딩)·planner(온라인 sampling)는 **우리 DTC 설계와 근본적으로 다름** = 채택 여부는 별도 큰 결정.

## 6. 무엇을 우리가 할 수 있나 (충돌·범위)
- ✅ **CVAE map generator** = 완료(`cvae_mapgen.py`, 논문 충실·검증). 프레임워크 독립.
- ⏭ **ψ → 지형 기하**(standalone, 다음): Components-of-ψ로 CVAE 출력 ψ를 실제 stepping-stone 월드 포즈로 변환. `_build_gap_terrain` 대체용 브리지.
- ⚠ **Tracker·학습환경·planner 전체 = RobotSW_IsaacLab(A 도메인)**. 우리 DTC에 이미 tracker+env 존재(설계 다름). 재구현 아니라 **CVAE 통합**(A 조율) + Table S2 도메인랜덤·Table S4 보상 참조가 실질 작업.

## 7. 정직한 결론
"학습환경세팅"의 큰 부분(tracker·PPO·보상·도메인랜덤·planner)은 **우리 DTC에 이미 있고(A 도메인)**, Raibo2025의 그것과 설계가 다름. 우리가 독립으로 기여할 조각은 **①CVAE 커리큘럼(완료) ②ψ→지형 기하 브리지(다음)**. 나머지(바운딩 tracker·온라인 planner 채택)는 우리 DTC 방향을 바꾸는 별도 결정 → A와 조율 필요.
