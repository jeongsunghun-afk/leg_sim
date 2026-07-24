# Contact-Implicit MPC (CI-MPC / C-1) 개발 리포트

> 02_Leg 17-DOF 사족보행 로봇 · 모델기반 제2트랙(접촉 타이밍 온라인 발견)
> 기준 논문: **Kim et al. 2023, "Contact-Implicit MPC: Controlling Diverse Quadruped Motions Without Pre-Planned Contact Modes or Trajectories"** (KAIST HOUND)
> 작성: 2026-07-23 (C++ 포팅 안정화 반영) · 코드: `simulation/quad/ci_mpc/` · C++: `simulation/quad/cpp/ci_mpc/`

---

## 0. 요약 (TL;DR)

- **목표**: 접촉 스케줄·발판을 **고정 입력으로 두지 않고**, 몸통 목표(desired pose)만 주면 **어느 발이 언제 어디를 딛을지를 online으로 발견**하는 모델기반 제어기(HOUND식 CI-MPC)를 우리 스택(Pinocchio + 직접구현 FDDP)에 구현.
- **달성**: 논문의 두 핵심 — **① hard 접촉 forward**(penetration-launch 없음) + **② 접촉 임펄스 λ의 해석 그래디언트 ∂λ/∂(q,v,u)**(FD 대비 **EXACT**) — 를 구현·검증. **평지 walking 2.7~3.6s 안정 실증**(base 직립·전진 스텝·발산 없음).
- **미완**: 완전 sustained walking(무한 보행)은 optimizer의 **hard forward**가 fine dt를 요구 → Python 긴 horizon 불가 = **C++ 프론티어**. 현재는 soft forward + exact λ그래디언트 하이브리드.
- **자세 전환(desired pose)**: walk·crouch·sit(nose-up)·lie를 desired pose만 주고 online 실증(§4.7) + GUI/뷰어(A vs CI 비교).
- **C++ 포팅(전체 스택)**: relaxed 그래디언트·tangent 선형화·FDDP·step_kkt(hard active-set)·receding MPC까지 포팅, **Python과 iteration 단위 정확 일치** 검증. **★★해석 그래디언트(HOUND 핵심) 구현·검증·측정 완료**(2026-07-24, §5.6): FD 44-geom을 kinematic Hessian+RNEA 트릭으로 대체 → 값 exact(=FD 1.66e-9)·**solve 195→82ms 2.4×**. Python이 convention 벽으로 못 한 지점을 C++서 돌파.

---

## 1. 배경과 동기

### 1.1 왜 CI-MPC인가
현행 배포 컨트롤러(A = WBIC + SRBD MPC)와 B/C(OCP)는 모두 접촉 스케줄(T_ds/T_ss)을 **고정 입력**으로 두는 fixed-schedule이다. 발 cadence·step length가 지형 간격과 우연히 맞는 gap만 넘는다(속도-공명). 계획층(스텝 길이·타이밍·base-발판 협조)의 부재가 험지 크로싱의 구조적 한계였다.

**Kim 2023 CI-MPC**가 정답 계열: 사전계획된 접촉모드·발판 없이 **접촉모드·발판·타이밍을 온라인으로 동시 발견**한다. 스택이 우리와 겹침(Pinocchio 전신동역학 + DDP).

### 1.2 논문의 핵심 설계
> **hard 접촉 forward + smooth(relaxed) 그래디언트 backward의 분리**

- **Forward roll-out**: hard contact(Signorini 상보성 + Coulomb) 임펄스로 물리적으로 crisp한 동역학.
- **Backward pass**: **접촉 임펄스의 해석 그래디언트**(relaxed 완화 상보성, eq 26)를 계산해 탐색 방향 산출.
- 사전 발판/궤적 없음. 비용은 몸통 SE(3) 참조 + foot-slip/clearance/air-time/symmetry.

---

## 2. 개발 단계 (C1.0 → C1.5)

### C1.0 — de-risk (해석 도함수 검증)  `c1_gradient_check.py`
가장 어려운 부분(하드접촉 해석미분)이 우리 스택에서 가능한지 먼저 확인.
- Pinocchio `constraintDynamics` + `computeConstraintDynamicsDerivatives`가 제공하는 ∂ddq/∂(q,v,τ)가 **유한차분(FD)과 일치**(상대오차 1e-6~1e-9).
- **결론**: HOUND가 수주간 손유도한 하드접촉 해석 그래디언트가 **Pinocchio 내장으로 de-risk**. C-1 난이도·기간 대폭↓.

### C1.1 — contact-implicit forward + iLQR OCP  `ci_action.py` · `ci_ocp.py`
- `ci_action.py`: 부드러운 단방향 접촉(softplus/sigmoid 완화 상보성) forward + 해석 그래디언트. 미분가능.
- `ci_ocp.py`: 매니폴드(freeflyer) tangent iLQR. `pin.dIntegrate` 연쇄.
- **결과**: crisp 접촉 서기 안정화 OCP **J 913→28.7 (97%↓·α=1.0)** 수렴. autodiff-free 궤적최적화 작동 실증.
- 발견: crisp(stiff) 접촉이 short-horizon서 soft보다 오히려 잘 수렴. explicit spring은 dt 작아야 안정.

### C1.2 — 해석 그래디언트 (soft force)  `ci_action.dynamics_derivatives`
- `computeABADerivatives` + 접촉력 연쇄로 ∂ddq/∂(q,v,τ) 해석 계산.
- FD 검증: **∂v·∂τ 정확(1e-10)**, ∂q 1.5e-2(getFrameVelocityDerivatives LWA convention 잔여).

### C1.3 — solver 스택: 긴 horizon 2블로커 분해  `ci_ocp.py` · `ci_ocp_ms.py`

**(A) relaxed backward — 조건수 문제**
- 측정: hard(stiff) 접촉의 선형화는 본질적으로 ill-conditioned(cond(A) 1e7). well-conditioned = soft + 작은 dt.
- **HOUND 트릭 구현**: forward는 stiff(물리 crisp) 유지 + **backward 그래디언트만 완화**(`KN_G=800`) → cond(A) 6.8e5 → **4.9e3 (139배↓)**, 수렴 horizon 100→200ms.
- 방향이 반직관: "물리에 맞춘 그래디언트"가 아니라 "일부러 완화한 그래디언트"가 well-conditioned(DDP는 정확 stiff Jacobian 불요).

**(B) multiple-shooting FDDP — nominal 발산 문제**  `ci_ocp_ms.py`
- single-shooting iLQR은 긴 horizon서 open-loop nominal 발산으로 ~100ms서 깨짐.
- **Mastalli 2020 FDDP 직접 구현**(crocoddyl 바인딩 우회): 각 노드 상태를 결정변수로 + gap(dynamics defect)이 물리 불일치 흡수 → nominal=서기 초기화(발산 원천 제거). gap 주입 backward(V_x⁺=V_x+V_xx·f̄) + feasibility-driven forward(gap (1-α) 수축) + merit(J+GAP_W·Σ|gap|).
- **결과**: N30/50/100 = 300ms·500ms·**1초** 모두 gap 7.6→0.0000 폐쇄(feasible). single-shooting 200ms 벽 돌파.

### C1.4 — hard 접촉 forward + λ 해석 그래디언트 + walking 실증  `ci_action.step_kkt·dyn_derivs_kkt` · `ci_mpc_walk.py`
(§3 아키텍처, §4 여정 참조) — **논문 핵심 완성 + walking 2.7~3.6s 실증**.

### C1.5 — C++ 실시간 · sustained walking (미착수)
완전 sustained walking = hard forward(velocity-impulse 대dt 또는 fine dt) 필요 = C++ 프론티어. HOUND 기준 dt25ms·N20·40Hz·max-iter4·그래디언트 ~70μs.

---

## 3. 최종 아키텍처 (HOUND식)

```
┌─ 입력: desired 몸통 pose 참조 (base 미션) + nominal stance 관절각(고정) ─┐
│                                                                        │
│   ┌──────────── MPC optimizer (receding horizon) ────────────┐        │
│   │  forward roll-out : soft 접촉 (빠름·안정)  ※C++면 hard      │        │
│   │  backward         : ★ λ 해석 그래디언트 ∂λ/∂(q,v,u) (EXACT) │        │
│   │  solver           : multiple-shooting FDDP (gap+merit)     │        │
│   └────────────────────────┬───────────────────────────────────┘        │
│                            │ 계획 (q_plan, v_plan, u_ff)                  │
│   ┌────────────────────────▼───────────────────────────────────┐        │
│   │  저수준 (HOUND §6.3): PD+FF fine rate (1kHz) 계획 추종        │        │
│   │    u_cmd = u_ff + Kp(q_plan−q) + Kd(v_plan−v)                │        │
│   └────────────────────────┬───────────────────────────────────┘        │
│   ┌────────────────────────▼───────────────────────────────────┐        │
│   │  sim/실행 : step_kkt (hard 접촉, constraintDynamics+active-set)│       │
│   └──────────────────────────────────────────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
```

**입력의 성격 (중요)**
- **base 참조** = 능동 지정하는 "미션"(전진속도·자세·높이·yaw). 태스크마다 바꿈.
- **관절 참조** = **nominal stance 고정**, 전 horizon 동일(보간·궤적 아님). regularization 역할.
- **걸음·발판·접촉 스케줄** = 처방 안 함. foot-slip cost로 **online 창발**.
- (즉 다리각도 **궤적**이 아니라 홈 자세 하나만. 자세 전환[앉기 등]은 목표 관절각도 desired에 포함.)

**세 핵심 조각**
| 조각 | 구현 | 상태 |
|---|---|---|
| ① hard 접촉 forward | `step_kkt` = pin.constraintDynamics(proper KKT) + active-set(단방향) | ✅ penetration-launch 없음 |
| ② **λ 해석 그래디언트** ∂λ/∂(q,v,u) | `dyn_derivs_kkt` = computeConstraintDynamicsDerivatives → `lin_AB_kkt` | ✅ **FD EXACT** (A 5.5e-8) |
| ③ 저수준 추종 | PD+FF fine rate(≤1ms) MPC 계획 추종 | ✅ (제어 유지간격 안정 핵심) |

---

## 4. 개발 여정 — 막힘과 돌파 (정직)

이 프로젝트의 가치는 **어디서 막히고 왜 뚫렸는지**에 있다.

1. **접근 교정 (논문 정독)**: 초기엔 Raibert식 **발궤적을 처방**했으나 뒤로 가고 붕괴. 논문 §5.1.2 재확인 → **HOUND는 발궤적·발판 처방 안 함**. foot-slip/clearance cost(eq22, `l_f=c_f·Σ S(-30φ)‖v_t‖²`)로 **걸음이 창발**(발 낮으면 접선속도 벌점 → 전진하려면 발 들 수밖에). 발궤적 처방 제거 후 전진 창발.

2. **단일 OCP 한계 → receding MPC**: 단일 open-loop OCP 0.6s는 feasibility+균형+스텝 동시 해결 불가. 걸음은 **receding-horizon MPC(HOUND 40Hz 폐루프)의 창발현상**임을 특정.

3. **근본 블로커 = soft 접촉**: receding MPC도 walking서 ~1s 발산. 원인 = **동적 스텝 touchdown서 발이 soft spring 파고듦 → 거대 복원력(kn·depth) → base 발사**(base_z 0.95로 튐). soft는 gentle 모션엔 OK, 동적 touchdown엔 부적합. → HOUND가 hard 접촉 쓰는 정확한 이유.

4. **hard 접촉 구현 시행착오**:
   - `step_hard`(velocity-projection 임펄스): penetration-launch는 막았으나 **에너지주입 발산**(explicit 스킴의 미묘한 불일치). 폐기.
   - **`step_kkt`(pin.constraintDynamics, proper KKT accel-level)**: 서기 안정 + penetration-launch 없음. ✅
   - **★핵심 교훈**: 초기 hard 발산은 접촉이 아니라 **제어 유지간격**(0.02s held PD = 발산 · 0.001s = 안정)이 원인. = **HOUND §6.3(저수준 PD+FF fine rate 추종)이 필수인 이유**.

5. **walking 실증**: sim=step_kkt(hard) + PD+FF fine 추종 통합 → soft에선 발산하던 config가 **2.7s 안정 보행**(base 직립·전진 0.25m). 하지만 ~2.7s 후 속도 runaway(soft **planner**의 penetration 경향).

6. **★논문 핵심 (λ 그래디언트)로 runaway 해결**: 사용자 지적("핵심은 λ의 그래디언트")대로 접촉 임펄스의 해석 그래디언트를 구현(`dyn_derivs_kkt`/`lin_AB_kkt`, FD **EXACT**). soft 그래디언트(1.5~13% 오차)를 대체 → **runaway 소멸**(2.7s → 3.6s 발산없음, vx 폭발 사라짐). 접촉 임펄스의 정확한 그래디언트가 optimizer에 올바른 정보를 줘 공격적 lunge를 방지.

7. **자세 전환(desired pose) 실증**: 몸통 목표(base_z·pitch·발 목표)만 주면 CI-MPC가 online 수행. `_stance_q` IK + POSE 모드(VX=0·foot-slip 자동 off). walk(전진 0.16m)·crouch(0.42→0.28)·sit(0.22+nose-up 0.25rad=진짜 개-앉기)·lie(0.20) 전부 발산 없음. A 컨트롤러의 룰베이스 전환을 일반 solver로 대체 실증. GUI(Tkinter)+MuJoCo 뷰어로 A vs CI 비교.

8. **★★C++ 포팅 안정화 — `dIntegrate` setZero 함정**: 그래디언트 코어(dyn_relaxed·lin_AB)를 C++로 포팅하니 서기 OCP가 `Vxx=1e50` 폭발 + crash. 격리 디버깅으로 근본원인 규명:
   - `dyn_relaxed`(ddq_dq=1058.68)는 Python과 완전 일치인데 `lin_AB`의 tangent A만 불일치(28→204) → tangent 단계로 격리. 수동 블록 dqn_dq=4.89 vs 재유도 69.59(같은 공식·같은 w·같은 model) → 차이는 `dIntegrate` 출력행렬뿐.
   - **★근본원인: pinocchio `dIntegrate`는 블록대각만 쓰고 off-diagonal은 안 지운다** → 출력행렬 `setZero()` 선행 필수. 미초기화 재사용 메모리의 쓰레기값이 A를 7배 부풀려 Vxx 25스텝 누적 폭발. `ci_relaxed.cpp`가 "검증됨"이던 건 **fresh malloc이 우연히 0**이라 통과한 것(운).
   - **수정 후**: Vxx=7171(Python 7678 일치)·crash 소멸·J 18→16.8 단조 수렴. **검증1(Python 일치)**: Python OCP를 relaxed 그래디언트로 돌리면 J 18→16.8, 각 iter α·J가 C++와 완전 동일 → **C++가 Python relaxed-gradient OCP를 iteration 단위 정확 재현**. (16.8 vs soft-데모 11.1 = relaxed 그래디언트[논문핵심, soft forward와 불일치] vs soft-force 그래디언트 방법차이, 버그 아님.)
   - **검증2(일관성=correctness)**: forward도 relaxed(`step_relaxed`)로 바꿔 backward와 동일 동역학으로 만들면(`FWD_REL=1`) **전 iter α=1.0 clean Newton 하강**(J 66→44). soft forward(불일치)는 α 0.1/0.05로 정체. → **relaxed 그래디언트가 relaxed 동역학의 정확한 도함수임을 확인**(단순 하강이 아니라 일관 Newton). 물리 결과(종단오차·base_z)는 두 forward 유사 = 25ms·|v0|1.03서 서기 한계.

---

## 5. 핵심 발견·교훈

1. **hard 접촉 ≠ well-conditioned**: hard의 선형화는 본질적으로 ill-conditioned(cond 1e7). HOUND의 트릭은 **hard forward + 완화(또는 exact-KKT) backward 분리**.
2. **λ 그래디언트가 진짜 핵심**: 접촉 임펄스의 정확한 ∂λ/∂(q,v,u)가 walking 안정성을 좌우. soft force 그래디언트(근사)로는 runaway.
3. **hard 발산의 진짜 원인 = 제어 유지간격**: 접촉이 아니라 held control interval. fine rate(≤1ms) PD+FF 추종이 필수.
4. **soft spring의 한계**: gentle 모션 OK, 동적 touchdown서 penetration-launch. walking엔 hard 필수.
5. **걸음은 처방이 아니라 창발**: 발궤적/발판 처방 금지. foot-slip cost + 몸통 참조 → 접촉 online 발견.
6. **성능 = fine dt**: constraintDynamics forward는 정확하나 fine dt 필요 → Python 긴 horizon 느림. 실시간은 C++.
7. **★C++ 포팅 함정 = `dIntegrate` setZero** (메커니즘):
   - **왜 블록대각만 쓰나**: 상태가 매니폴드(free-flyer SE3 + 관절 R)라 적분 `q⊕v`가 관절별 독립 → Jacobian이 구조적 블록대각(base 6×6·관절 1×1). off-diagonal(관절 간 커플링)은 수학적으로 0. pinocchio는 성능상 **0이 아닌 대각 블록만 쓰고 off-diagonal은 안 건드림**(호출자가 0으로 넣었다 가정 = 규약).
   - **왜 터졌나**: `MatrixXd J(nv,nv)`는 미초기화(힙 쓰레기값). dIntegrate가 대각만 덮어써 off-diagonal에 쓰레기 잔존 → `dqn_dq=dInt0+…`가 그 쓰레기 상속 → tangent A_fro 28→204(물리적으로 말 안 되는 DOF 커플링) → backward Riccati가 A를 25스텝 반복 곱 → **Vxx 기하 누적 폭발(1e50)** → 게인 NaN → crash.
   - **왜 단독은 통과했나**: 갓 할당 행렬이 OS zero-page라 우연히 0(운). OCP 루프는 힙 재사용이라 쓰레기 노출.
   - **교훈**: **pinocchio `dIntegrate`/`dDifference` 출력행렬은 항상 `MatrixXd::Zero` 선행.** 회귀 가드 = `ci_relaxed.cpp`가 수동 vs 라이브러리(CiDyn) A 대조(‖diff‖<1e-9).
8. **동일 이름 다른 함수 주의**: Python OCP 데모의 `lin_AB`(generic)=soft-force 도함수(forward와 일치), C++ `ci.lin_AB`=relaxed 해석 그래디언트(논문 핵심). 둘 다 유효하나 수렴 깊이가 다름(11.1 vs 16.8). 포팅·비교 시 어느 그래디언트인지 명확히.

---

## 5.5 논문 대비 구현 차이 — 완화항 `εI` vs `ρD`

논문(HOUND, image 1/3)의 relaxed 그래디언트 공식과 우리 구현을 대조하면 **뼈대는 동일**하고 **완화항 딱 한 군데**가 다르다.

| | 논문 (image 3/4) | 구현 (`RELAX_MODE=eps`, 기본) | 논문판 (`RELAX_MODE=D`) |
|---|---|---|---|
| 역행렬 안 | `A_cc + **ρD**` | `A_cc + εI` | `A_cc + ρD` |
| D 정의 | `diag(1/λ_l²)`, **법선만** | 없음(상수 ε·I 전방향) | `diag(1/λ_n²)` 법선전용 |
| 실현 상보성 | `vⁿ∘λⁿ = ρ`(상수) | `vⁿ∘λⁿ = −ε·λ²`(변동) | `vⁿ∘λⁿ = ρ`(상수) |
| forward | `vⁿλⁿ=ρ` Newton | `λ=−(A+εI)⁻¹b` 선형 | `vⁿλⁿ=ρ` Newton(λⁿ>0) |

- **★"v·λ=ρ 실현" 서술은 부정확했다**: εI는 Tikhonov 정규화라 `v=−ελ` → `vⁿ∘λⁿ=−ε·λ²`(상수 아님). forward 수치검증: εI판 `vⁿ∘λⁿ=−1.1e-5`(변동) vs ρD판 `1e-4`(상수 ρ). 기존 FD-EXACT 검증은 **εI 시스템의 자기일관성**을 확인한 것이지 논문 ρ 일치가 아니었다.
- **논문판 ρD 구현·검증**(`RELAX_MODE=D`): 두 모드 다 **FD EXACT**(∂q~7e-10). D=diag(1/λ_n²) 법선전용, forward는 `vⁿλⁿ=ρ` Newton(법선 λⁿ>0 내부점).
- **어디서 다른가(make/break 스윕)**: firm contact(λⁿ>0)선 εI=ρD **동일**(A_fro 비 1.00). 발 분리(λⁿ<0)로 갈수록 ρD가 ~30% 작고 부드러운 gradient + **λⁿ>0(단방향=발은 밀기만, 물리적) 강제**, εI는 λⁿ<0(당김=비물리) 허용 + gradient 더 큼.
- **★성능 실증(보행 receding MPC, VX=0.3)**: εI vs ρD 비교 — ρD가 보행 내내 **base를 ~0.05–0.13m 높게 유지**(스텝 중 덜 주저앉음)·붕괴 전 **48% 더 전진**(0.671 vs 0.606 최대). 둘 다 ~2.1s서 붕괴(근본 walking 불안정=soft planner, 완화선택 무관). firm contact선 차이 없음. (caveat: ε1e-3·ρ1e-4는 강도 단위 달라 완전통제 ablation 아님, 각자 합리적 기본값.)
- **★★현재 기본=ρD**(Python·C++ 양쪽, 2026-07-23). `RELAX_MODE=D` 기본, `eps`는 하위호환 옵션. **C++ ρD Newton 포팅=Python과 정확 일치**(make/break서 εI 26.47·ρD 18.16 양쪽 6자리 일치). setZero 회귀 가드는 εI로 유지.
- **함의**: 서기/보행(firm contact)엔 εI≈ρD. **gap/stepping처럼 make/break가 잦은 상황엔 ρD가 부드럽고 물리적**(λⁿ>0=발은 밀기만) = 논문이 ρD 쓰는 이유. 이제 논문 표준 채택.

---

## 5.6 FD 그래디언트 vs 해석 그래디언트 (실시간 핵심 레버)

접촉 임펄스 그래디언트 `∂λ/∂q`(정확히는 `∂A_cc/∂q`·`∂b_cc/∂q`·`∂(M⁻¹Jᵀλ)/∂q`)를 구하는 **두 방식** — 결과값은 같고 속도가 다르다.

| | FD 그래디언트 (유한차분) | 해석 그래디언트 (analytic) |
|---|---|---|
| 원리 | 관절 q를 ±ε 흔들어 결과 차이/ε | 미분 공식을 직접 조립(흔들지 않음) |
| 계산 | **44회 기하 재평가**(nv=22 × ±2, `ci_relaxed.cpp:71`) | kinematic Hessian ×4발 + RNEA도함수 ×2 |
| 사용 도구 | geom() 반복 호출 | `getFrameKinematicHessian`(∂J/∂q) + `computeRNEADerivatives`(∂M/∂q) |
| solve/step | **195ms** | **82ms** |
| 정확도 | 기준(ε 의존 미세오차) | FD와 **1.66e-9 일치**(exact) |

**해석 조립 공식** (`dyn_relaxed`의 `analytic_grad` 분기):
```
dbg  = Σ_b H(a,b,j)·q̇_free[b]          ← ∂b_cc/∂q  (kinematic Hessian)
Term1= Σ_b H(a,b,j)·(Wλ)[b] ;  JᵀΛ = Σ_a H(a,b,j)·λ[3k+a]
DMy  = RNEAderiv(q,0,Wλ).dtau_dq − RNEAderiv(q,0,0)   ← ∂(M·y)/∂q (RNEA 트릭)
dWl  = M⁻¹(JᵀΛ − DMy) ;  DA = Term1 + Jcc·dWl          ← ∂(M⁻¹Jᵀλ)/∂q
dl_dq = A⁻¹·DA − A⁻¹·(dbg + Jcc·dt·aq)                 ← 최종 ∂λ/∂q
```

- **왜 중요**: FD의 44회 재평가가 solve의 지배적 비용. 해석은 이를 없애 **2.4× 가속** + **정확도 향상**(ε 없음). HOUND가 40Hz 실시간(그래디언트 ~70μs)을 낸 방법이 바로 이것.
- **왜 Python이 못 했나**: Python 트랙은 `getFrameVelocityDerivatives`의 LWA(LOCAL_WORLD_ALIGNED) convention 불일치로 ∂q항이 1.5e-2 잔차. C++ `getFrameKinematicHessian`은 convention이 정확해 통과(de-risk: `ci_kinhess_test.cpp` ∂b_cc/∂q 1.2e-11·∂M⁻¹/∂q 5.7e-8).
- **활성화**: `env ANALYTIC=1`(기본 0). 남은 = 해석 kinhess/RNEA 자체 최적화(→ HOUND 70μs) + foot-slip `∂vt/∂q`도 kinematic Hessian로 exact화(gait 품질).

---

## 6. 파일 구성 (`simulation/quad/ci_mpc/`)

| 파일 | 역할 |
|---|---|
| `ci_action.py` | 접촉 forward + 그래디언트. soft(`step`·`dynamics_derivatives`) + **hard(`step_kkt`·`dyn_derivs_kkt`)** + λ 반환 |
| `ci_ocp.py` | tangent iLQR 선형화. `lin_AB`(relaxed soft) + **`lin_AB_kkt`(λ 그래디언트)** |
| `ci_ocp_ms.py` | multiple-shooting FDDP(gap 주입·feasibility-driven·merit) + foot-slip/air-time cost |
| **`ci_mpc_walk.py`** | **receding-horizon MPC walking** (planner + λ그래디언트 + hard sim + PD+FF 추종) |
| `c1_gradient_check.py` | 제약동역학 해석 도함수 vs FD 검증(de-risk) |
| `model_bridge.py` | URDF↔MJCF Pinocchio 브리지 |
| `ci_mpc_gui.py` · `ci_mpc_viewer.py` | GUI(Tkinter)+뷰어(A vs CI 비교 replay) |
| `sim2sim_probe.py` | 궤적 sim2sim 갭 프로브 |

### C++ 포팅 (`simulation/quad/cpp/ci_mpc/`)
| 파일 | 역할 |
|---|---|
| `ci_dyn.hpp` | CiDyn 라이브러리: 모델 로드·stance IK·step_soft/step_kkt·**dyn_relaxed(FD+해석 그래디언트)·lin_AB(tangent)·lin_AB_multi(multi-rate)**·foot-slip·in_gap. ★`dIntegrate` 전 setZero |
| `ci_relaxed.cpp` | 그래디언트 값 대조 검증 + **회귀 가드**(수동 vs CiDyn ‖diff‖<1e-9 + **해석 vs FD** 1.66e-9) |
| `ci_ocp_test.cpp` | single-shooting iLQR 서기 OCP. Python relaxed OCP와 iter단위 일치 검증 |
| `ci_fddp_test.cpp` | multiple-shooting FDDP(gap·feasibility·merit) |
| `ci_kkt_test.cpp` | hard active-set forward(step_kkt) 발 뜸 검증 |
| `ci_kinhess_test.cpp` | 해석 그래디언트 primitive de-risk(kinematic Hessian·RNEA 트릭 vs FD) |
| **`ci_mpc_run.cpp`** | **receding-horizon MPC**(solve_fddp + step_kkt sim + PD+FF fine 추종 + gait/gap 참조 + 프로파일). `env ANALYTIC=1`=해석 그래디언트 |

> C++는 **그래디언트 코어 + FDDP + step_kkt + receding MPC + 해석 그래디언트**까지 포팅·검증. 남은 = 해석 그래디언트 속도 추가 최적화(→70μs)·스텝 중 균형 튜닝.

---

## 7. 주요 파라미터 (`ci_mpc_walk.py`)

| 파라미터 | 기본 | 의미 |
|---|---|---|
| `HARD` | 1 | sim=step_kkt(hard)+PD+FF fine 추종 |
| `HARD_PLAN` | 0→**1** | backward=**exact λ그래디언트**(runaway 소멸 핵심) |
| `HARD_FWD` | 0 | optimizer forward=hard(fine dt 필요=C++). 0=soft(빠름) |
| `CTRL_DT` | 0.001 | 저수준 제어 갱신간격(≤1ms 필수, 안정 핵심) |
| `CF` / `C1S` | 2500 / -30 | foot-slip/clearance cost(걸음 창발). HOUND c_f=1·c1=-30 |
| `AIR_W` | 100 | air-time φ² 벌점(발 오래 들지마) |
| `W_BASE` | 50 | base z·자세 균형(↑=직립, 과하면 웅크림) |
| `VXVEL`/`W_BVEL` | 120/50 | base 전진속도 추종 / 속도 감쇠(crash 억제) |
| `KP_T`/`KD_T` | 150/12 | 저수준 PD+FF 추종 게인 |

**실행 예 (walking)**:
```
env HARD=1 HARD_PLAN=1 VX=0.15 CF=2500 W_BASE=50 CTRL_DT=0.001 \
  /home/jsh/miniforge3/envs/proxddp/bin/python \
  /home/jsh/문서/jsh/simulation/quad/ci_mpc/ci_mpc_walk.py
```

---

## 8. 현재 상태 · 남은 작업

**✅ 완성·검증 (Python)**
- solver 스택(relaxed backward·multiple-shooting FDDP)
- hard 접촉 forward(step_kkt) — penetration-launch 없음
- **λ 해석 그래디언트 ∂λ/∂(q,v,u) — FD EXACT** (논문 핵심)
- **relaxed 상보성 그래디언트(dyn_derivs_relaxed) — FD EXACT** (커스텀, ε 완화)
- walking 실증 — 2.7~3.6s 안정(base 직립·전진 스텝·발산 없음)
- **자세 전환(desired pose)** — walk·crouch·sit·lie online 실증 + GUI/뷰어

**✅ 완성·검증 (C++)**
- 그래디언트 코어(dyn_relaxed·lin_AB) 포팅 — **Python과 iteration 단위 정확 일치**
- **논문판 ρD 포팅**(RELAX_MODE=D 기본) — make/break서 Python ρD와 6자리 일치(18.159240)
- 서기 iLQR OCP 안정화 — `dIntegrate` setZero 버그 수정(Vxx 1e50→7171, crash 소멸)
- **multiple-shooting FDDP 포팅**(`ci_fddp_test.cpp`) — gap 완전폐쇄(feasible)·α=1.0 clean 수렴·N=100(0.1s, single-shooting 4배)·종단오차 0.338. relaxed forward+backward 일관 필수(soft forward는 gap 미폐쇄).
- **★★해석 그래디언트(HOUND 핵심) 포팅·검증·속도**(`ci_dyn.hpp` `analytic_grad`, 2026-07-24) — FD 44-geom 루프를 **kinematic Hessian(∂J/∂q)+RNEA 트릭(∂M⁻¹/∂q)** 해석 조립으로 대체. **값 exact**(해석=FD ‖diff‖1.66e-9)·**solve 195→82ms=2.4×↑**. Python이 kinematic 도함수 convention 벽으로 못 한 지점을 C++서 돌파(§5.6).
- 회귀 가드(ci_relaxed 대조, εI/ρD 양쪽 + 해석 vs FD)

**⏳ 남음**
| 항목 | 이유/방향 |
|---|---|
| 완전 sustained walking | runaway 지연됐으나 완전 제거 아님 |
| optimizer의 hard forward(HARD_FWD) | fine dt 필요 → **C++ 프론티어**(velocity-impulse 대dt 솔버 또는 fine dt) |
| **★★C++ step_kkt(hard active-set)** | ✅ **해소·스텝 달성**. `step_kkt`(constraintDynamics+active-set 단방향) 포팅·검증(FL굽힘→발 0.106m 뜸). ci_mpc_run 통합(SIM_KKT+PLAN_SOFT soft planner+gait)→**발 실제로 뜸(5.3cm)=진짜 스텝!** |
| **★★C++ fine-rate PD+FF 추종** | ✅ HOUND §6.3(계획을 h≤0.001s로 PD+FF 추종, KP_T/KD_T). u0 held(0.01s)의 base 붕괴(0.21)→**유지(0.33~0.40)+발 5.3cm 스텝** |
| **C++ 전진 보행(미붕괴, but 거침)** | △ anti-runaway 튜닝 → 3s 미붕괴·전진 0.478m. **★정직 교정(뷰어 확인)**: base_z 요동 12cm·기울기 8~19°·발 관통=**거친 전방 shuffle이지 깨끗한 보행 아님**. base_z만 본 이전 "안정" 판정은 과대평가. ★근본=CI-MPC로 basic walk 생성이 어긋난 접근(고정 관절테이블 추종=동역학 비일관). **walk엔 보행생성기 필요**(Raibert gait), CI-MPC 가치=험지 접촉발견 |
| **★★C++ 40Hz 실시간 프로파일/최적화** | ✅ 병목=FDDP solve(FD 기하 도함수 195ms·sim step_kkt 0.3ms만). ①knob(LIN_NSUB·N·ITERS) 축소=N15·ITERS3 20.4ms/step 실시간(단 품질저하). ②**★해석 그래디언트(ANALYTIC=1)로 195→82ms=2.4×↑·값 exact**(§5.6). 실시간+풀품질=해석+적당한 축소 조합(진행중), HOUND 70μs엔 해석 kinhess/RNEA 추가 최적화 |
| **C++ 험지(gap) 크로싱** | ✅ 인프라: CiDyn.in_gap로 step_kkt 틈 위 발=지지없음 + gref_gap 발판 solid ground shift(재IK). baseline=걷다 gap서 앞발 빠져 붕괴(gap 물리 확인). 발판회피시 gap 너머 과신전→vx lunge runaway→붕괴 = Python perceptive-nav "gap-edge lunge" 동일 프론티어. 안정 크로싱=capture/RL(연구급) |
| **C++ 40Hz 실시간** | 현 Python-급(offline). HOUND 해석 그래디언트 속도 최적화 |
| 더 깊은 수렴(J 11.1) | soft-force 그래디언트(dynamics_derivatives) C++ 포팅 시(옵션) |
| 다양한 동작(rearing·선회) | 미실증. **구조는 지원** |

**★★C++ CI-MPC 스택 현황**: 그래디언트(ρD)·iLQR·FDDP·multi-rate·foot-slip·gait참조·receding MPC·**step_kkt(hard active-set)** 전부 완성·검증. **★진짜 스텝 보행 달성**(step_kkt sim+soft planner+gait→발 5.3cm 뜸). 모델기반 CI-MPC 파이프라인 전체가 C++로 동작. **남은=스텝 중 균형 튜닝**(base 붕괴 방지)+40Hz 실시간.

**★해석 그래디언트 완성**(2026-07-24): HOUND 핵심(해석 접촉 그래디언트)을 C++서 구현·검증(exact=FD)·측정(2.4×). Python이 convention 벽으로 못 한 지점 돌파. **foot-slip ∂c/∂q도 exact화**(∂vt/∂q kinematic Hessian + 접촉점 오프셋 높이항, rel 7.56e-11).

**★★실시간+품질 스윕 결론(2026-07-24)** — 정직한 핵심 발견:
- **해석 그래디언트 = FD 폐루프 완전 동일**: 같은 config서 ANALYTIC=0(FD)·1(해석) 궤적이 **전진 1.792m·base_z·tilt 6자리 동일**(해석이 FD와 exact하므로 당연). 차이는 **solve 속도뿐**(33.9 vs 43.9ms). 속도 이득은 substep 수에 비례(N15·LIN2서 1.3×, N30·LIN10서 2.4×).
- **∴ 붕괴는 그래디언트 정확도와 직교**: 지속 walk 붕괴는 **속도 폭주(terminal-value 부재) 프론티어**. FD·해석 똑같이 붕괴. 그래디언트를 exact화해도 안 풀림(cost/terminal 문제).
- **속도 폭주 관찰**: 목표 vx=0.2인데 폐루프서 vx가 0.4→0.5→…→2.2로 가속(foot-slip 추진을 속도추종이 못 잡음). 강한 속도조절(VXVEL 60→400·W_BVX 8)로 초반 vx≈0.27(목표 근접)까지 잡으나 ~1.5s엔 붕괴. best config(N15·I4·LIN2·VXVEL400)=근실시간 34ms(40Hz 1.4×), 초반 0.6~0.9s는 walk 유사(base_z 0.40·발 lift 0.11).
- **결론**: **실시간 속도는 해석 그래디언트로 달성**(exact·34ms). **지속 clean walk는 여전히 미해결 프론티어** = terminal value function / capture-point가 필요(속도 폭주 근절). 이 트랙의 배포 컨트롤러는 A(WBIC) 불변.

**★★DCM(capture point) 터미널 cost — 속도 폭주 근절 building block**(2026-07-24):
- **원리**: 유한 horizon MPC(0.15s)가 horizon 너머 발산 모드(CoM 속도)를 못 봐 폭주. DCM ξ=x+v/ω(ω=√(g/z))는 LIP 발산 성분에 집중 페널티(대각 Qf는 발산/수렴 모드에 가중 분산). d=e_x+e_vx/ω로 상태오차서 조립·rank-1 hessian. env `W_DCM`(기본 0=무회귀)·`DCM_TF`(터미널 배율). 러닝+터미널 노드+merit 반영.
- **★결과(정직)**: DCM가 **수평 속도 폭주를 확실히 잡음**(step 0.6s에서 vx **1.07→0.02**, W_DCM1000·TF8·BVZ10·VXVEL120·N20). **그러나 실패가 높이-지지 붕괴로 이동**(~0.6–0.9s base_z 0.40→0.12 가라앉음 = 스윙 중 지지 부족, 다중접촉 타이밍 문제).
- **함의**: 각 cost 항이 한 실패를 잡으면 **다음 결합 실패가 드러남** → **고정-게이트 + CI-MPC 지속 walk는 marginal**, 진짜 해법은 whole-body balance(컨트롤러 A의 WBIC). DCM은 원리적 building block으로 속도 폭주를 해결. 근실시간 유지(26~35ms, 40Hz 1.1~1.4×).

**다음 갈림길**: (1) whole-body balance 항 추가로 모델기반 지속 walk 시도(높이-지지 결합 해결), (2) **RL 피벗**([[mpc-rl-hybrid-roadmap]]) — 지속 강건 walk를 RL로, CI-MPC는 교사/험지, (3) **CI-MPC를 험지 접촉-타이밍 전용**으로(walk는 A). **배포 컨트롤러 A(WBIC)는 불변** — 이 트랙은 모델기반 제2 트랙.

---

## 10. 참조-자세 추종 검증 (2026-07-24)

**질문**: 참조 자세(desired pose)를 주면 잘 추종하는가? 서기(VX=0, Rq=qstar 0.42)로 측정.

- **접촉모델 일치가 관건**: planner·sim이 같은 접촉모델이면 잘 추종.
  - relaxed planner + **relaxed sim**(일치): 3초 안정 — base_z 0.42→**0.418**(2mm)·tilt **2.4°**·드리프트 2mm.
  - relaxed planner + **hard-KKT sim**(불일치): 침하·붕괴(base_z→0.033·tilt 17°).
- **버그 수정**: `W_BASE`(base z·자세 가중)가 `if(VX>0)`에서만 적용돼 서기(VX=0)에선 무효였음(기본 20 고정) → VX 무관 적용. `RHO`·`RU` env도 노출.
- **HARD_FWD**(planner forward=step_kkt, HOUND식 hard forward): hard-KKT sim에서 서기 크게 개선 — **tilt 17→4.8°·침하 0.033→0.235**. planner-sim forward 일치의 효과.
- **잔존 침하(3초간 0.42→0.24)는 미해결**: base 높이 Jᵀ 피드백(pitch 커플링으로 악화)·수직감쇠·control-reg(RU)·joint 가중·iters 모두 무효/미미. HARD_FWD시 forward는 일치하므로 남은 불일치는 backward 그래디언트(relaxed)뿐인데, **backward 완화는 HOUND의 의도된 설계**(hard 선형화=cond 1e7 ill-conditioned, §5.1). 즉 잔존은 저수준·cost 튜닝으로 안 풀리고 hard backward도 역효과 유력 = 짧은 horizon·relaxed 그래디언트 하의 marginal standing 문제(연구급).
- **정직한 결론**: 참조-자세 추종은 **접촉모델이 일관되면 정확히 작동**(2mm). hard-KKT sim에선 HARD_FWD로 크게 개선되나 완전 유지엔 못 미침(잔존 침하). 배포 정지·자세는 A(WBIC) 담당 불변.

### 10.1 CI-MPC 4액션 산출물 (2026-07-24, CI-MPC 트랙 마무리)

목표: CI-MPC가 **눕기·앉기·서기·두발서기**를 잘 하는가. env `TGT_BZ`(높이)·`PITCH`(base nose-up)·`LIFT_FRONT`(앞두발 들기). 준정적이라 relaxed 일관 접촉(4발 자세는 물리적 타당).

| 액션 | env | 결과 | 수치 |
|---|---|---|---|
| **서기** | TGT_BZ=0.42 | ✅ | base_z 0.411·tilt 2.3°·드리프트 3mm·3s |
| **눕기** | TGT_BZ=0.20 | ✅ | base_z **0.206**(A 눕기와 일치)·tilt 2.3°·3s |
| **앉기** | PITCH=0.35·TGT_BZ=0.32 | ✅ | nose-up ~20° 자세 유지·base_z 0.314·3s |
| **두발서기** | LIFT_FRONT=1·COM_X뒤·PITCH | ✗ | 0.9s 붕괴(tilt 76°) = 축소 지지면 균형 프론티어 |

- **3/4 성공**: 준정적 자세(서기·눕기·앉기)는 CI-MPC가 목표를 **정확히 수행·유지**(±수 mm, 3s). 눕기 0.206은 A의 haunch/lie 높이와 일치.
- **두발서기만 붕괴**: 앞발은 정상적으로 뜸(step_kkt 변수접촉 작동)이나 **2발 축소 지지면 능동 균형에 실패**. 이는 marginal 균형 프론티어 = **RL/WBIC 균형층이 담당할 부분**(순수 CI-MPC 한계, HOUND 품질 미달).
- **뷰어**: `act_stand.npy`·`act_lie.npy`·`act_sit.npy`(scratchpad, replay_viewer.py).
- **CI-MPC 트랙 정리 결론**: **준정적 자세 생성·유지는 CI-MPC 강점(A보다 일반적=임의 목표 대응)**. 능동 균형이 필요한 동작(두발서기·동적 보행·gap)은 프론티어 → RL(DTC) 담당. 이 산출물이 CI-MPC의 검증된 능력 경계.

---

## 9. 관련 문서
- 파이프라인: `docs/pipeline_ci_mpc.html` (§13 C1.0~C1.5)
- 파라미터: `docs/params_ci_mpc.html` (CI-MPC walking 섹션)
- 논문: `quad/backup/docs/5.47650_Contact_implicit_Model_P.pdf`
</content>
