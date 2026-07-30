# Contact-Implicit MPC (CI-MPC / C-1) 개발 리포트

> **상태: ★종결 (2026-07-30 기준, 2026-07-24 결정).** 로버스트 험지극복 모델로 **부적합** 확정 — 수행분만 남기고 접음. 배포 컨트롤러 A(WBIC)는 불변. 이후 방향 = TOWR(SRBD-TO) 데이터 + RL(AMP/APT-RL).
>
> 02_Leg 17-DOF 사족보행 로봇 · 모델기반 제2트랙(접촉 타이밍 온라인 발견) · 기준 논문: **Kim et al. 2023, "Contact-Implicit MPC" (KAIST HOUND)** · 코드: `simulation/quad/ci_mpc/`(Python) · `simulation/quad/cpp/ci_mpc/`(C++)

---

## 0. 요약 (TL;DR)

- **목표**: 접촉 스케줄·발판을 고정 입력으로 두지 않고 몸통 목표(desired pose)만 주면 **어느 발이 언제 어디를 딛을지를 online으로 발견**하는 모델기반 제어기(HOUND식 CI-MPC)를 우리 스택(Pinocchio + 직접구현 FDDP)에 구현.
- **달성(수행분)**: 논문의 두 핵심 — ① hard 접촉 forward(penetration-launch 없음) + ② 접촉 임펄스 λ의 해석 그래디언트 ∂λ/∂(q,v,u)(FD 대비 **EXACT**) — 를 구현·검증. Python·C++ 전체 스택(그래디언트 코어·FDDP·step_kkt·receding MPC·해석 그래디언트) 포팅·iteration 단위 일치. 평지 walking 2.7~3.6s·준정적 자세(서기/눕기/앉기) 실증.
- **종결 사유**: 폐루프가 현실(hard) 접촉에서 일관되게 균형 프론티어 붕괴(gap 낙상·walk 속도폭주·서기 외란 0.3 붕괴·두발/깊은앉기 붕괴). relaxed의 안정성은 발 접착 아티팩트. **로버스트 험지·데이터·배포 어디에도 CI-MPC가 최선 아님**(§1 참조).
- **트랙의 값**: 음성 결과(negative result)로 **RL 방향 결정을 근거 있게 확정** + 접촉-암시 방법·해석 그래디언트·DCM의 깊은 이해.

---

## 1. 트랙 종결 결론 — 왜 접었나 (2026-07-24)

**결정(사용자)**: CI-MPC 트랙을 수행분만 남기고 종결. 추가 개발 중단.

**로버스트 험지극복 모델로 부적합**:
- **우리 실험**: 폐루프가 현실(hard) 접촉에서 일관되게 균형 프론티어 붕괴(gap 낙상·walk 속도폭주·서기 외란 0.3 붕괴·두발/깊은앉기 붕괴). relaxed forward의 안정성·외란강건성은 **발 접착(과제약, `step_relaxed`가 4발 고정집합) 아티팩트**로, 실제 hard 접촉선 marginal.
- **발명자 신호**: KAIST HOUND 팀조차 최신 험지 in-the-wild를 CI-MPC가 아니라 **RL(APT-RL, SRBD-TO pretrain)** 로 수행. CI-MPC가 답이면 그들이 썼을 것.
- **구조적 위치**: 데이터 생성 = SRBD-TO(TOWR, 빠름·18만 궤적)·강건 실행 = RL·CI-MPC는 둘 사이에 끼어 어느 쪽도 최선 아님. 파쿠르도 RL 지배. AMP엔 retargeting이면 충분(운동학 특징만), APT-RL엔 SRBD-TO 데이터 = 둘 다 CI-MPC 불필요.

**CI-MPC의 존재 이유**: 학습 없이(zero-shot) 모델기반으로 접촉을 사전처방 없이 창발시키는 통합 제어기(rearing·비스크립트 접촉·복구). 연구 niche는 분명하나 우리 병목(강건성·지각·규모)과 겹치지 않음.

**재개 시 TODO**:
1. **자기충돌 cost**(캡슐 선분거리: 다리=thigh_joint→foot·몸통=Base±0.28 캡슐·seg-seg 최근접점 페널티. calf 프레임 degenerate라 미사용. 설계만 했고 미검증·되돌림).
2. **보행 세팅**(walk 게이트/속도 튜닝 — 현재 walk 붕괴 상태).

---

## 2. 배경과 동기 (요약)

현행 배포 컨트롤러(A = WBIC + SRBD MPC)와 B/C(OCP)는 모두 접촉 스케줄(T_ds/T_ss)을 **고정 입력**으로 둔다. 발 cadence·step length가 지형 간격과 우연히 맞는 gap만 넘는다(속도-공명). 계획층(스텝 길이·타이밍·base-발판 협조)의 부재가 험지 크로싱의 구조적 한계였다.

**Kim 2023 CI-MPC**는 사전계획된 접촉모드·발판 없이 접촉모드·발판·타이밍을 온라인으로 동시 발견한다(스택이 우리와 겹침: Pinocchio 전신동역학 + DDP). 논문 핵심 설계 = **hard 접촉 forward(물리 crisp) + smooth(relaxed) 그래디언트 backward의 분리**. 비용은 몸통 SE(3) 참조 + foot-slip/clearance/air-time/symmetry이며, 걸음은 처방이 아니라 이 cost로 창발한다.

---

## 3. 최종 아키텍처 (HOUND식)

```
┌─ 입력: desired 몸통 pose 참조 (base 미션) + nominal stance 관절각(고정) ─┐
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

**입력의 성격**: base 참조 = 능동 지정 "미션"(전진속도·자세·높이·yaw, 태스크마다 바꿈). 관절 참조 = **nominal stance 고정**(전 horizon 동일, 궤적 아님·regularization 역할). 걸음·발판·접촉 스케줄 = 처방 안 함, foot-slip cost로 online 창발. (자세 전환[앉기 등]은 목표 관절각도만 desired에 포함.)

| 세 핵심 조각 | 구현 | 상태 |
|---|---|---|
| ① hard 접촉 forward | `step_kkt` = pin.constraintDynamics(proper KKT) + active-set(단방향) | ✅ penetration-launch 없음 |
| ② **λ 해석 그래디언트** ∂λ/∂(q,v,u) | `dyn_derivs_kkt` = computeConstraintDynamicsDerivatives → `lin_AB_kkt` | ✅ **FD EXACT**(A 5.5e-8) |
| ③ 저수준 추종 | PD+FF fine rate(≤1ms) MPC 계획 추종 | ✅ (제어 유지간격 안정 핵심) |

---

## 4. 개발 여정 — 막힘과 돌파

어디서 막히고 왜 뚫렸는지가 이 트랙의 실질이다.

1. **접근 교정(논문 정독)**: 초기엔 Raibert식 발궤적을 처방했으나 뒤로 가고 붕괴. 논문 §5.1.2 재확인 → HOUND는 발궤적·발판 처방 안 함. foot-slip/clearance cost(`l_f=c_f·Σ S(-30φ)‖v_t‖²`)로 걸음이 창발(발 낮으면 접선속도 벌점 → 전진하려면 발 들 수밖에). 처방 제거 후 전진 창발.

2. **단일 OCP 한계 → receding MPC**: 단일 open-loop OCP 0.6s는 feasibility+균형+스텝 동시 해결 불가. 걸음은 receding-horizon MPC(HOUND 40Hz 폐루프)의 창발현상임을 특정.

3. **single-shooting → multiple-shooting FDDP**: single-shooting iLQR은 긴 horizon서 open-loop nominal 발산으로 ~100ms서 깨짐(horizon 스캔 N10 OK·N15 실패). **Mastalli 2020 FDDP 직접 구현**(crocoddyl 바인딩 우회): 각 노드 상태를 결정변수로 + gap(dynamics defect)이 물리 불일치 흡수 → 모든 노드를 서기로 초기화(발산 원천 제거). gap 주입 backward + feasibility-driven forward + merit(J+GAP_W·Σ|gap|). 결과: N30/50/100 = 300ms·500ms·**1초** 모두 gap 7.6→0.0000 폐쇄(feasible). 200ms 벽 돌파.

4. **근본 블로커 = soft 접촉**: receding MPC도 walking서 ~1s 발산. 원인 = 동적 스텝 touchdown서 발이 soft spring 파고듦 → 거대 복원력(kn·depth) → base 발사(base_z 0.95로 튐). soft는 gentle 모션엔 OK, 동적 touchdown엔 부적합 = HOUND가 hard 접촉 쓰는 정확한 이유.

5. **hard 접촉 구현 시행착오**: `step_hard`(velocity-projection 임펄스)는 penetration-launch는 막았으나 에너지주입 발산(explicit 스킴 불일치)으로 폐기. **`step_kkt`(pin.constraintDynamics, proper KKT accel-level + active-set 단방향)**: 서기 안정 + penetration-launch 없음. ✅ **핵심 교훈**: 초기 hard 발산은 접촉이 아니라 **제어 유지간격**(0.02s held PD = 발산 · 0.001s = 안정)이 원인 = HOUND §6.3(저수준 PD+FF fine rate 추종)이 필수인 이유.

6. **walking 실증**: sim=step_kkt(hard) + PD+FF fine 추종 통합 → soft에선 발산하던 config가 2.7s 안정 보행(base 직립·전진 0.25m). 단 ~2.7s 후 속도 runaway(soft **planner**의 penetration 경향).

7. **★λ 그래디언트로 runaway 해결**: 사용자 지적("핵심은 λ의 그래디언트")대로 접촉 임펄스의 해석 그래디언트를 구현(`dyn_derivs_kkt`/`lin_AB_kkt`, FD **EXACT**). soft 그래디언트(1.5~13% 오차)를 대체 → runaway 소멸(2.7s → 3.6s 발산없음, vx 폭발 사라짐). 완전 제거는 hard forward도 필요(soft forward의 penetration 경향 잔존).

8. **★★C++ 포팅 최대 함정 — `dIntegrate` setZero**: 그래디언트 코어를 C++로 포팅하니 서기 OCP가 `Vxx=1e50` 폭발 + crash. 근본원인 = **pinocchio `dIntegrate`는 블록대각만 쓰고 off-diagonal은 안 지운다** → 출력행렬 `setZero()` 선행 필수. 미초기화 재사용 메모리의 쓰레기값이 tangent A를 7배 부풀려(28→204) Vxx 25스텝 누적 폭발. `ci_relaxed.cpp`가 "검증됨"이던 건 fresh malloc이 우연히 0이라 통과한 것(운). 수정 후 Vxx=7171(Python 7678 일치)·crash 소멸. **회귀 가드** = 수동 vs 라이브러리(CiDyn) A 대조(‖diff‖<1e-9).

---

## 5. 핵심 발견·교훈

1. **hard 접촉 ≠ well-conditioned**: hard의 선형화는 본질적으로 ill-conditioned(cond 1e7). HOUND의 트릭은 hard forward + 완화(또는 exact-KKT) backward 분리. 방향이 반직관 — "물리에 맞춘 그래디언트"가 아니라 "일부러 완화한 그래디언트"가 well-conditioned(DDP는 정확 stiff Jacobian 불요). backward만 완화(`KN_G=800`)로 cond(A) 6.8e5→4.9e3(139배↓), 수렴 horizon 100→200ms.
2. **λ 그래디언트가 진짜 핵심**: 접촉 임펄스의 정확한 ∂λ/∂(q,v,u)가 walking 안정성을 좌우. soft force 그래디언트(근사)로는 runaway.
3. **hard 발산의 진짜 원인 = 제어 유지간격**: 접촉이 아니라 held control interval. fine rate(≤1ms) PD+FF 추종이 필수.
4. **soft spring의 한계**: gentle 모션 OK, 동적 touchdown서 penetration-launch. walking엔 hard 필수.
5. **걸음은 처방이 아니라 창발**: 발궤적/발판 처방 금지. foot-slip cost + 몸통 참조 → 접촉 online 발견.
6. **성능 = fine dt**: constraintDynamics forward는 정확하나 fine dt 필요 → Python 긴 horizon 느림. 실시간은 C++.
7. **동일 이름 다른 함수 주의**: Python OCP 데모의 `lin_AB`(generic)=soft-force 도함수(forward와 일치), C++ `ci.lin_AB`=relaxed 해석 그래디언트(논문 핵심). 둘 다 유효하나 수렴 깊이 다름(11.1 vs 16.8). 포팅·비교 시 어느 그래디언트인지 명확히.

---

## 6. 논문 대비 구현 차이 — 완화항 `εI` vs `ρD`

논문(HOUND)의 relaxed 그래디언트와 우리 구현은 뼈대는 동일하고 완화항 한 군데가 다르다.

| | 논문 (`RELAX_MODE=D`) | 초기 구현 (`RELAX_MODE=eps`) |
|---|---|---|
| 역행렬 안 | `A_cc + **ρD**`, D=diag(1/λ_n²) 법선전용 | `A_cc + εI` (상수 ε·I 전방향) |
| forward | `vⁿλⁿ=ρ` Newton (법선 λⁿ>0 내부점) | `λ=−(A+εI)⁻¹b` 선형 |
| 실현 상보성 | `vⁿ∘λⁿ=ρ`(상수) | `vⁿ∘λⁿ=−ε·λ²`(변동, Tikhonov) |

- **정정**: "εI가 v·λ=ρ를 실현"은 부정확했다. εI는 Tikhonov 정규화라 `vⁿ∘λⁿ=−ε·λ²`(상수 아님). 기존 FD-EXACT 검증은 εI 시스템의 자기일관성을 확인한 것이지 논문 ρ 일치가 아니었음. **논문판 ρD 구현·검증**: 두 모드 다 FD EXACT(∂q~7e-10).
- **make/break 스윕**: firm contact(λⁿ>0)선 εI=ρD **동일**(A_fro 비 1.00). 발 분리(λⁿ<0)로 갈수록 ρD가 ~30% 작고 부드러운 gradient + λⁿ>0(단방향=발은 밀기만, 물리적) 강제, εI는 λⁿ<0(당김=비물리) 허용.
- **성능(보행 receding MPC, VX=0.3)**: ρD가 base를 ~0.05–0.13m 높게 유지·붕괴 전 48% 더 전진(0.671 vs 0.606). 둘 다 ~2.1s서 붕괴(근본 walking 불안정=soft planner, 완화선택 무관).
- **★현재 기본=ρD**(Python·C++ 양쪽, 2026-07-23, 커밋 aa9e278). C++ ρD Newton 포팅=Python과 정확 일치(make/break서 εI 26.47·ρD 18.16 6자리 일치). setZero 회귀 가드는 εI로 유지. gap/stepping처럼 make/break가 잦은 상황엔 ρD가 부드럽고 물리적 = 논문이 ρD 쓰는 이유.

---

## 7. FD 그래디언트 vs 해석 그래디언트 (실시간 핵심 레버)

접촉 임펄스 그래디언트를 구하는 두 방식 — 결과값은 같고 속도가 다르다.

| | FD 그래디언트 | 해석 그래디언트 (analytic) |
|---|---|---|
| 계산 | **44회 기하 재평가**(nv=22 × ±2) | kinematic Hessian ×4발 + RNEA 도함수 ×2 |
| 도구 | geom() 반복 호출 | `getFrameKinematicHessian`(∂J/∂q) + `computeRNEADerivatives`(∂M/∂q) |
| solve/step | **195ms** | **82ms** |
| 정확도 | 기준(ε 의존 미세오차) | FD와 **1.66e-9 일치**(exact) |

해석 조립 (`dyn_relaxed`의 `analytic_grad` 분기): ∂b_cc/∂q=kinematic Hessian, ∂(M·y)/∂q=RNEA 트릭(`RNEAderiv(q,0,y).dtau_dq − RNEAderiv(q,0,0)`, **VectorXd 복사 필수** — d.tau 참조 반환이라 안 하면 0), ∂λ/∂q = A⁻¹·DA − A⁻¹·(dbg + Jcc·dt·aq).

- **의의**: FD의 44회 재평가가 solve의 지배 비용. 해석은 이를 없애 **2.4× 가속** + 정확도 향상(ε 없음). HOUND가 40Hz 실시간(그래디언트 ~70μs)을 낸 방법.
- **왜 Python이 못 했나**: Python 트랙은 `getFrameVelocityDerivatives`의 LWA convention 불일치로 ∂q항 1.5e-2 잔차. C++ `getFrameKinematicHessian`은 convention 정확해 통과(de-risk: ∂b_cc/∂q 1.2e-11·∂M⁻¹/∂q 5.7e-8). **Python이 막힌 지점을 C++서 돌파.**
- **foot-slip ∂c/∂q도 exact화**(0927980): ∂vt/∂q kinematic Hessian + 접촉점 오프셋 높이항, rel 7.56e-11.
- **활성화**: `env ANALYTIC=1`(기본 0). 커밋 4a7f519.

---

## 8. 완성·검증 산출물 (종결 시점 유효)

**Python** — solver 스택(relaxed backward·multiple-shooting FDDP) · hard 접촉 forward(step_kkt, penetration-launch 없음) · **λ 해석 그래디언트 ∂λ/∂(q,v,u) — FD EXACT**(논문 핵심) · relaxed 상보성 그래디언트(dyn_derivs_relaxed) — FD EXACT · walking 2.7~3.6s 안정 · 자세 전환(desired pose) walk·crouch·sit·lie online 실증 + GUI/뷰어.

**C++** — 그래디언트 코어(dyn_relaxed·lin_AB) Python과 iteration 단위 정확 일치 · 논문판 ρD 포팅(make/break서 Python과 6자리 일치, 18.159240) · 서기 iLQR OCP(dIntegrate setZero 수정, Vxx 1e50→7171) · multiple-shooting FDDP(gap 완전폐쇄·α=1.0 clean·N=100·종단오차 0.338) · **step_kkt(hard active-set) 포팅**(SIM_KKT+soft planner+gait → 발 5.3cm 뜸 = 진짜 스텝) · fine-rate PD+FF 추종(base 붕괴 0.21→유지 0.33~0.40) · **해석 그래디언트(HOUND 핵심)** 구현·검증(=FD 1.66e-9)·측정(195→82ms 2.4×) · 회귀 가드(ci_relaxed 대조, εI/ρD + 해석 vs FD).

**참조-자세 추종 (2026-07-24)** — planner·sim 접촉모델 일치가 관건. relaxed planner + relaxed sim(일치): 3초 안정, base_z 0.42→0.418(2mm)·tilt 2.4°·드리프트 2mm. relaxed planner + hard-KKT sim(불일치): 침하·붕괴(base_z→0.033·tilt 17°). HARD_FWD(planner forward=step_kkt)로 hard sim 서기 크게 개선(tilt 17→4.8°·침하 0.033→0.235)하나 완전 유지엔 못 미침(3초간 0.42→0.24 잔존 침하 = 짧은 horizon·relaxed 그래디언트 하의 marginal standing, 연구급). 버그 수정: `W_BASE`가 `if(VX>0)`에서만 적용돼 서기(VX=0)에선 무효였음 → VX 무관 적용.

**CI-MPC 4액션 산출물 (2026-07-24, 트랙 마무리)** — env `TGT_BZ`(높이)·`PITCH`(base nose-up)·`LIFT_FRONT`(앞두발 들기). 준정적이라 relaxed 일관 접촉이 물리적 타당.

| 액션 | env | 결과 | 수치 |
|---|---|---|---|
| 서기 | TGT_BZ=0.42 | ✅ | base_z 0.411·tilt 2.3°·드리프트 3mm·3s |
| 눕기 | TGT_BZ=0.20 | ✅ | base_z **0.206**(A 눕기와 일치)·tilt 2.3°·3s |
| 앉기 | PITCH=0.35·TGT_BZ=0.32 | ✅ | nose-up ~20° 자세 유지·base_z 0.314·3s |
| 두발서기 | LIFT_FRONT=1·COM_X뒤·PITCH | ✗ | 0.9s 붕괴(tilt 76°) = 축소 지지면 균형 프론티어 |

**결론**: 준정적 자세 생성·유지는 CI-MPC 강점(A보다 일반적 = 임의 목표 대응). 능동 균형이 필요한 동작(두발서기·동적 보행·gap)은 프론티어 → RL(DTC) 담당. 이것이 CI-MPC의 검증된 능력 경계.

**DCM(capture point) 터미널 cost** — 유한 horizon MPC가 발산 모드(CoM 속도)를 못 봐 폭주하는 것을 DCM ξ=x+v/ω 페널티로 잡음. 수평 속도 폭주를 확실히 잡으나(step 0.6s에서 vx 1.07→0.02) 실패가 높이-지지 붕괴로 이동(base_z 0.40→0.12). **각 cost 항이 한 실패를 잡으면 다음 결합 실패가 드러남** → 고정-게이트 + CI-MPC 지속 walk는 marginal, 진짜 해법은 whole-body balance(A의 WBIC)임을 재확인.

---

## 9. 파일 구성

### Python (`simulation/quad/ci_mpc/`)
| 파일 | 역할 |
|---|---|
| `ci_action.py` | 접촉 forward + 그래디언트. soft(`step`·`dynamics_derivatives`) + **hard(`step_kkt`·`dyn_derivs_kkt`)** + λ 반환 |
| `ci_ocp.py` | tangent iLQR 선형화. `lin_AB`(relaxed soft) + **`lin_AB_kkt`(λ 그래디언트)** |
| `ci_ocp_ms.py` | multiple-shooting FDDP(gap 주입·feasibility-driven·merit) + foot-slip/air-time cost |
| **`ci_mpc_walk.py`** | **receding-horizon MPC walking**(planner + λ그래디언트 + hard sim + PD+FF 추종) |
| `c1_gradient_check.py` | 제약동역학 해석 도함수 vs FD 검증(de-risk) |
| `model_bridge.py` | URDF↔MJCF Pinocchio 브리지 |
| `ci_mpc_gui.py` · `ci_mpc_viewer.py` · `replay_viewer.py` | GUI(Tkinter)+뷰어(A vs CI 비교 replay) |

### C++ (`simulation/quad/cpp/ci_mpc/`)
| 파일 | 역할 |
|---|---|
| `ci_dyn.hpp` | CiDyn 라이브러리: 모델 로드·stance IK·step_soft/step_kkt·dyn_relaxed(FD+해석 그래디언트)·lin_AB(tangent)·lin_AB_multi·foot-slip·in_gap. ★`dIntegrate` 전 setZero |
| `ci_relaxed.cpp` | 그래디언트 값 대조 + **회귀 가드**(수동 vs CiDyn ‖diff‖<1e-9 + 해석 vs FD 1.66e-9) |
| `ci_ocp_test.cpp` | single-shooting iLQR 서기 OCP. Python relaxed OCP와 iter단위 일치 검증 |
| `ci_fddp_test.cpp` | multiple-shooting FDDP(gap·feasibility·merit) |
| `ci_kkt_test.cpp` | hard active-set forward(step_kkt) 발 뜸 검증 |
| `ci_kinhess_test.cpp` | 해석 그래디언트 primitive de-risk(kinematic Hessian·RNEA 트릭 vs FD) |
| **`ci_mpc_run.cpp`** | **receding-horizon MPC**(solve_fddp + step_kkt sim + PD+FF fine 추종 + gait/gap 참조). `env ANALYTIC=1`=해석 그래디언트 |

---

## 10. 주요 파라미터 (`ci_mpc_walk.py`)

| 파라미터 | 기본 | 의미 |
|---|---|---|
| `HARD` | 1 | sim=step_kkt(hard)+PD+FF fine 추종 |
| `HARD_PLAN` | 1 | backward=exact λ그래디언트(runaway 소멸 핵심) |
| `HARD_FWD` | 0 | optimizer forward=hard(fine dt 필요=C++). 0=soft(빠름) |
| `CTRL_DT` | 0.001 | 저수준 제어 갱신간격(≤1ms 필수, 안정 핵심) |
| `CF` / `C1S` | 2500 / -30 | foot-slip/clearance cost(걸음 창발). HOUND c_f=1·c1=-30 |
| `AIR_W` | 100 | air-time φ² 벌점 |
| `W_BASE` | 50 | base z·자세 균형(↑=직립, 과하면 웅크림) |
| `VXVEL`/`W_BVEL` | 120/50 | base 전진속도 추종 / 속도 감쇠(crash 억제) |
| `KP_T`/`KD_T` | 150/12 | 저수준 PD+FF 추종 게인 |
| `RELAX_MODE` | D | ρD(논문판, 기본) / eps(하위호환) |

**실행 예 (walking)**:
```
env HARD=1 HARD_PLAN=1 VX=0.15 CF=2500 W_BASE=50 CTRL_DT=0.001 \
  /home/jsh/miniforge3/envs/proxddp/bin/python \
  /home/jsh/문서/jsh/simulation/quad/ci_mpc/ci_mpc_walk.py
```

---

## 11. 관련 문서
- 파이프라인: `docs/pipeline_ci.html` (§13 C1.0~C1.5 로드맵)
- 파라미터: `docs/params_ci.html` (CI-MPC walking 섹션)
- 논문: `quad/backup/docs/5.47650_Contact_implicit_Model_P.pdf`
