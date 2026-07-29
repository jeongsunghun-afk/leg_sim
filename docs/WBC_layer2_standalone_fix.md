# 2층 standalone WBC 수정 계획서 — TAMOLS → WBC (DTC baseline)

> 작성 2026-07-29. TAMOLS 추종 실행층을 **2층(TAMOLS→완전WBC)**으로 확정(3층 TAMOLS→SRBD MPC→WBIC 대신). **DTC 정합**(추종층 WBC↔RL 교체). **핵심 철학: 완벽화 아니라 "충분한 baseline" 확보 후 RL로.**

## 0. 왜 2층인가 (전략)

- **DTC = 2층**: TO(TAMOLS류)→**RL 정책 추종**, MPC 없음. 우리 2층(TAMOLS→WBC)은 **추종층만 WBC↔RL 교체하면 DTC**. 3층은 MPC 버려야 해 DTC와 안 맞음.
- **WBC = baseline·스텝스톤**: RL의 모델기반 비교기준·인프라 첫 벽돌. **RL이 넘을 것이므로 WBC를 MPC 수준까지 완벽화하는 건 비효율**(RL이 학습으로 푸는 걸 수작업 재현).
- **★핵심**: 2층 WBC의 근본 약점(예측 부재·task 경합)이 **정확히 RL이 해결하는 지점**. 따라서 WBC는 "그럭저럭 추종"까지만, 강건성은 RL로.

## 1. 현재 상태 (객관 진단)

우리 WBIC은 **이미 완전 WBC**: 결정변수=[q̈, f(접촉력)], 부유베이스 동역학 hard constraint(M₆q̈+h₆=ΣJᵀf), f를 QP서 직접 최적화, base xy/z·자세·W_AM·마찰콘. GRF 참조(lam)는 soft(w_lam)일 뿐 → w_lam 낮추면 standalone(="RSL 모드").

**standalone WBC vs 3층 MPC (offline 첫계획, t=0.75):**
| | 2층 standalone | 3층 MPC |
|---|---|---|
| z (0.5 유지) | **0.396 침하** | 0.504 유지 |
| y 드리프트 | **−0.21** | +0.07 |
| tilt | **26.5°** | 19.2° |

**문제 (원인별):**
| # | 문제 | 원인 |
|---|---|---|
| 1 | **z 유지력 부족**(핵심) | 접촉전이 시 base 처짐을 순간반응 z-task가 사후대응→누적. MPC는 예측(0.28s)으로 사전보상 |
| 2 | **y 횡드리프트** | task 경합 — z-게인↑ 하면 y 악화(trade-off) |
| 3 | **tilt 열세** | 위 결과 |

**근본**: standalone WBC=순간적(예측 없음) + weighted task 경합. MPC=예측적. (ETH WBC도 instantaneous인데 됨=계층적 우선순위·정교 힘분배 차이).

**시도됨**: z-task 게인 env화(KP_Z·KD_Z·W_Z)+계획 z가속 ff. KP_Z 200→600으로 z 0.396→0.443 부분개선하나 y/tilt 악화(경합). **단순 게인으론 부족**.

## 2. 계획 (최소 baseline 지향)

### Phase 0 — TAMOLS 계획 품질 수정 (필수·최우선, 공유자산)
WBC·RL 둘 다 좋은 계획을 필요로 함. **이게 우선.**
1. **base 속도 프로파일**: solve_fast가 깨끗한 입력엔 양호(후진 없음, overshoot만). 오염 입력 시 후진. → 온라인은 TAM_CLEANV로 회피. 계획 자체 velocity 정규화/bound는 선택.
2. **발판 대칭**: 마지막 phase만 nominal cost → 전 phase Y-대칭 정규화(선택, H4는 부차라 후순위).
3. **성공**: offline 첫계획이 "추종 가능한" 품질(단조전진·대칭·GIAC 안정).

### Phase 1 — 최소 2층 WBC baseline (완벽화 X)
standalone WBC를 "그럭저럭 추종"까지. **MPC 따라잡기 목표 아님.**
1. **task 우선순위 개선**: z/자세 task를 swing/posture보다 우선(계층적 or 가중 재조정). z-게인 단독↑는 y 경합→우선순위로 접근.
2. **접촉전이 완충**: 전이 시 처짐 최소화(swing commitment·SW_DUR 매칭 이미 있음, 힘분배 부드럽게).
3. **W_AM=30 유지**(④ lateral 억제 확인됨).
4. **성공기준(baseline)**: offline 단일사이클 **falls=0·tilt<15°**(MPC의 19°와 동급이면 충분, 완벽 X). z 침하 <0.05m.

### Phase 2 — 판단 게이트
- **P1이 baseline(tilt<15·falls=0) 달성 → RL로 넘어감**(2층 확보, 목적 달성).
- **P1이 과도한 튜닝 요구 → 즉시 RL로**(WBC 완벽화는 비효율, baseline 미달이어도 RL이 해결).
- **어느 쪽이든 RL이 다음** — WBC에 몇 주 쏟지 않음.

### Phase 3 — RL(DTC) 전환
1. TAMOLS 참조(발판·base pose·twist·accel·관절·height scan)를 RL 관측/보상으로.
2. IsaacLab quad env(hind_leg RL 트랙 재사용) + 참조추종 보상.
3. WBC baseline과 비교(RL이 넘는지).

## 3. 성공 기준 (2층 WBC는 "충분히"만)

- **NOT 목표**: MPC(3층) 완전 따라잡기, tilt<5°, 완벽 강건.
- **목표(baseline)**: offline 단일사이클 falls=0·tilt<15°·z침하<0.05 = "TAMOLS 계획이 2층으로 추종 가능함" 실증 + RL 비교기준.
- 이 달성/근접하면 **즉시 Phase 3(RL)**.

## 4. 자산 (재사용)

- WBC: wbic_track([q̈,f] QP, 완전 WBC). RSL_TRACK=standalone(2층), TAM_MPC=3층.
- 게인 env: KP_Z·KD_Z·W_Z(z-task)·W_AM·W_BASE_XY·w_ori·w_yaw.
- TAMOLS: solve_fast·online_replan·TAM_DUMP(계획 덤프).
- 진단: TAM_DBG(z·yaw·comy·footz), 첫계획 /tmp/tamols_first.txt.
- 계획서 형제: TAMOLS_online_tracking_fix.md(online ⑤·gait sync).

## 5. 리스크·판단

- **2층 WBC가 baseline조차 어려우면**: 계획 품질(P0) 문제인지 WBC 정식화 문제인지 분리. 계획이 좋은데 WBC가 안 되면 → **바로 RL**(WBC 한계=RL 이유).
- **과투자 경계**: WBC 튜닝이 수확체감이면 즉시 RL. 메모리가 반복 확정한 "모델기반 추종=벽, RL=강건" 준수.
- **online(⑤ gait sync)은 2층 baseline 후**: offline 2층 되면 online 통합(재anchor gait sync)은 형제 계획서(TAMOLS_online_tracking_fix.md §7)로. 단 online도 baseline이면 RL 우선.

## 6. 즉시 다음 액션
**Phase 1.1** — standalone WBC의 task 우선순위 재조정(z/자세 > swing/posture). 게인 단독이 아니라 **가중 위계**로 z침하·y경합 동시 개선 시도. offline 첫계획 falls=0·tilt<15 목표. 달성/수확체감이면 **Phase 3(RL)**.
