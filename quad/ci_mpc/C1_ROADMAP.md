# C-1 로드맵 — HOUND식 해석 그래디언트 실시간 CI-MPC

**목표** 접촉 타이밍을 실시간(40Hz) 최적화하는 CI-MPC로 gap/stepping **완전** 크로싱(착지 안정화 포함).
**동기** (a) 복구서 확인: C-2/참조 하이브리드는 gap1을 **부분 크로싱(base0.79)** 하나 **직후 낙상 = post-crossing settle을 참조 레버로 못 잡음(레버 소진)**. 착지 안정화 = 접촉 타이밍을 최적화가 직접 정해야 함 → C-1.

**왜 C-2가 아니라 C-1** ([[ci-mpc-track]] §실시간성 판정): C-2(mjx_ilqr)는 jacfwd(forward-mode AD)로 접촉물리 통과 → 제어1스텝 2~5만 mjx.step = **완성해도 offline**. HOUND 70μs/40Hz는 **자동미분 안 씀** = 해석 도함수. 즉 실시간 = C-2 더개발이 아니라 **C-1 별도 구현**.

---

## 참조
- **Kim et al. 2023** (KAIST HOUND, `backup/docs/5.47650_Contact_implicit_Model_P.pdf`) — 원 논문.
- 플랜파일 `elegant-skipping-quasar.md` Phase 2 = 이 로드맵의 원안(Python-first 정확도 → C++ 속도).
- 스택: Crocoddyl(Box-FDDP) + Pinocchio(해석 도함수) — 둘 다 `proxddp` env 설치됨.

## 핵심 = 접촉임펄스 동역학 + 해석 그래디언트 (novel)
| 구성 | forward (동역학) | backward (그래디언트) |
|---|---|---|
| **접촉** | 하드접촉 임펄스: 속도공간 Signorini LCP + Coulomb + max-dissipation, Hwangbo2018 block Gauss-Seidel | **완화 상보성 해석 그래디언트** ∂λ/∂q, ∂λ/∂q̇, ∂λ/∂u (닫힌형식, 식11-19·26) |
| **적분** | semi-implicit Euler + drift 보상 φ/dt(식25) | — |
| **동역학** | Pinocchio ABA | Pinocchio ABA 미분(해석, O(n)) + contact-Jacobian kinematic hessian |
| **완화** | ρ (HOUND 2.0서 시작) | — |

## 단계 (매 단계 검증 게이트)

### C1.0 · 접촉임펄스 forward 동역학 (Python, `ci_action.py`)
- `DifferentialActionModelAbstract` 서브클래스. Hwangbo block-GS 임펄스 솔버.
- **검증**: 동일 상태·토크서 **MuJoCo 스텝과 일치**(sim2sim 갭 측정). 낙하·접촉·마찰 케이스.

### C1.1 · FD 그래디언트 sanity
- 유한차분으로 ∂x'/∂x, ∂x'/∂u 계산 → Box-FDDP에 꽂아 **평지 트롯 falls=0**(느려도 됨).
- **게이트**: FD-CI가 fixed-schedule보다 gap서 나은가? (타이밍 적응 원리 증명, 속도 무관)

### C1.2 · 해석 그래디언트 (★novel·최고난도)
- 완화 상보성 해석 미분(식11-19)로 ∂λ/∂(q,q̇,u) 닫힌형식. FD와 대조(정확성 검증=관문).
- **게이트**: 해석≈FD (rel err<1e-3), Box-FDDP 수렴.

### C1.3 · gap/stepping 완전 크로싱 (Python)
- 비용(식): regulating(SO3 log) + foot slip/clearance(sigmoid) + air time + symmetric. 사전 발판/스케줄 **없음**.
- **게이트**: gap1 **완전** 크로싱(착지 안정화 = C-2가 못한 post-crossing settle 해결). stepping stones.

### C1.4 · C++ 실시간 포팅
- dt25ms·N20·40Hz·Box-FDDP max-iter4. 해석 그래디언트 ~70μs 목표. 상태추정 `state_estimator.hpp` 재사용.
- **게이트**: 40Hz 실시간 + gap/stepping 폐루프 크로싱.

## 리스크 (정직)
- **C1.2(해석 그래디언트)가 관문**: 하드접촉 임펄스 미분(식19/26)이 미묘, 참조코드 없음. FD 대조가 안전망.
- **sim2sim 갭**: Pinocchio 하드접촉 ↔ MuJoCo soft sphere 불일치. C1.0서 정량화(선행 필수). 클수록 C-1 배포가치↓(C-2/RL이 MuJoCo-native라 유리).
- **규모**: 수주. Python-first(C1.0-3) 후 C++(C1.4). C1.2 실패 시 폴백=RL 증류.

## 착수 조건 재확인 (게이트, [[mpc-rl-hybrid-roadmap]])
C-1은 **RL 하이브리드 결과 후** 진입 권장(RL이 험지 풀면 불필요). 지금 (c) 착수=사용자 지시 a→c→b. 선행 권장=**C1.0 sim2sim 갭 측정**(작으면 C-1 가치↑, 크면 재고).

## 다음 실행
1. **C1.0**: `ci_action.py` 스켈레톤 = 접촉임펄스 forward + MuJoCo 대조. (이 세션 파운데이션)
2. 이후 C1.1~1.4는 포커스 세션별로. C1.2가 핵심 연구 관문.
