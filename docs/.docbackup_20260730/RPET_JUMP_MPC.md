# RPET_JUMP_MPC.md — R.pet 점프 구현 브리핑 (OCP 주력 / MPPI 검증)

> R.pet 17-DoF(waist, all-ankle) 모델의 제자리 점프(pronk) → 전진 점프 구현 계획.
> 접촉 시퀀스가 **알려진** 문제이므로 OCP(고정 접촉 스케줄)가 정공법이고,
> MPPI는 검증·착지 강건화·타이밍 탐색의 보조 트랙으로 쓴다.
>
> 관련 문서: `RPET_ALIGATOR_MPC.md`(Phase 0–3), sit/getup 기술 기록(§3 착지 교훈,
> §8 MPPI), 토크·각속도 분석리포트 §7.5(run 스윕, ankle ω 한계),
> `RPET_HEAD_GAZE_MPC.md`(본 작업 완료 후 착수).
>
> **왜 점프가 sit/getup과 다른가 (방법 선택의 근거):**
> sit/getup은 접촉 시퀀스 자체가 미지수(rump on/off)라 MPPI가 정공법이었다.
> 점프는 접촉 시퀀스가 자명하다 — `4발 stance(추진) → flight → 4발 stance(착지)`.
> 접촉 스케줄이 알려진 문제는 미분 기반 OCP의 홈그라운드다 (crocoddyl 점프
> 예제가 표준 레퍼런스). 따라서 **OCP 주력, MPPI 보조**로 역할을 고정한다.

---

## 0. 목표와 완료 정의

### 0.1 목표 (단계별)

| 단계 | 목표 동작 | 성공 기준 |
|---|---|---|
| M1 | 제자리 점프 (pronk) | 발끝 clearance ≥ 0.10 m, 착지 falls=0, tilt < 10° |
| M2 | 전진 점프 | 전방 ≥ 0.3 m + M1 기준 유지 |
| M3 | 연속 점프 (착지→재점프) | 3회 연속 falls=0 |

### 0.2 범위 제외 (명시적 non-goal)

- 장애물 인지 기반 점프 (target은 높이/거리 파라미터로 지정)
- 온라인 실시간 점프 MPC (본 문서는 **오프라인 궤적 + 추종 배포**까지.
  온라인화는 aligator RTI 성숙 후 별도 문서)
- twist/backflip 등 곡예 (M3 완료 후 확장 항목으로만 §7에 기재)

### 0.3 전제 자산

- 17-DoF MJCF/URDF (질량 15.24 kg, 다리 무거운 분포), ankle 8:1 기어
- A안 Convex MPC+WBIC (착지 후 회복에 재사용), wbic_stance
- crocoddyl/aligator 환경, contact-implicit MPPI (`getup_mppi.py`)
- run 스윕에서 확보한 ankle ω 실측 한계 데이터 (STEP=0.08 → ω~40 rad/s,
  STEP=0.16 → ω 156 rad/s 스파이크·대량 낙상) — **점프 추진기 ω 예산의 기준**

---

## 1. Phase J0 — 타당성 계산 + 스크립트 점프 기준선 (선행 필수)

OCP를 돌리기 전에 "이 로봇이 물리적으로 얼마나 뛸 수 있는가"의 상한을
해석적으로 박아둔다. sit/getup의 교훈: **상한을 먼저 규명하면 이후 모든
결과의 판정 기준이 생긴다** (zf≈0.26 사례의 재사용).

### 1.1 해석적 상한 계산 (스크립트 1개, `jump_feasibility.py`)

```
이륙 속도 → 높이:   h = v_z² / (2g)
필요 충격량:        J = m·v_z  (m = 15.24 + payload)
추진 구간 t_thrust 동안 평균 잉여 GRF:  F_avg = J/t_thrust + m·g
관절 토크 요구(정적 근사): τ_i ≈ J_leg,iᵀ(q)·F  를 crouch→신전 구간에서 적분
```

- [ ] crouch 깊이(z0) × 추진 시간(t_thrust) 그리드에서, 토크 한계·ankle ω 한계를
      만족하는 최대 v_z 계산 → **도달 가능 h 상한 맵** 산출
  - crouch 하한은 sit/getup에서 검증된 wbic_stance 스쿼트 z=0.29를 시작점으로,
    IK 가지-flip 영역(z≲0.40에서 rear thigh flip 전례)은 검증된 update_stand_qhome
    경로의 자세만 사용 — **저높이 IK 새로 풀지 말 것** (기존 교훈)
  - ankle ω 예산: 추진 말기 신전 속도에서 ankle ω < 40 rad/s(run 안전 실측치)
    를 1차 제약으로, 완화 실험은 J3에서
- [ ] 다리 무거운 분포 보정: 다리 질량이 총질량의 큰 비중 → 추진 중 다리 자체
      가속에 토크가 소모된다. 정적 근사에 다리 관성항 보정 추가
      (`crba` 대각 블록으로 1차 추정)

### 1.2 스크립트 점프 기준선 (`jump_scripted.py`, SGU 패턴 재사용)

- [ ] open-loop 3상 스크립트: crouch(wbic_stance로 z0 정착) → thrust(관절 신전
      속도 프로파일, J0.1의 최선 그리드점) → flight(다리 tuck 고정) →
      touchdown 감지 → 저-PD 흡수 → wbic_stance 회복
- [ ] 목적: (a) J0.1 상한 맵의 시뮬 검증, (b) OCP가 넘어야 할 성능 하한,
      (c) touchdown 감지·착지 파이프라인을 OCP보다 먼저 완성

### 1.3 J0 완료 기준

- [ ] 상한 맵 산출 + 스크립트 점프로 h ≥ 상한의 50% 달성, 착지 falls 여부 기록
      (falls=0 불요 — 착지는 J2에서 해결. 단 이륙 자체는 성립해야 함)
- [ ] **판정 게이트**: 상한 h < 0.05 m이면 점프는 하드웨어 개선(토크/기어비)
      선행 과제로 재분류하고 본 문서 중단 — 이 판정 자체가 산출물

---

## 2. Phase J1 — OCP 오프라인 궤적 (주력 트랙)

### 2.1 문제 정식화 (crocoddyl 우선, aligator 이식은 J1.4)

고정 접촉 스케줄 다상(multi-phase) OCP:

```
Phase A (crouch→thrust): 4발 접촉, N_A 노드
  - 접촉: 4× ContactModel3D (발끝 point contact)
  - 비용: CoM 상향 가속 유도(터미널 CoM 속도 v_z*), 자세 정칙화,
          토크·상태 정칙화, 마찰원뿔(선형화) 페널티/제약
Phase B (flight): 접촉 없음, N_B 노드
  - 비용: 다리 tuck 목표 자세(clearance), **각운동량 정칙화(핵심, §2.2)**,
          터미널: 착지 준비 자세 + tilt 최소
Phase C (landing): 4발 접촉 복귀, N_C 노드
  - 임팩트 모델: crocoddyl ActionModelImpulseFwdDynamics (터치다운 순간)
  - 비용: 착지 후 CoM 정착, GRF 평활, 최종 stand 자세
```

- [ ] 노드/시간 초기값: A 0.25 s / B는 v_z*로부터 `t_fly = 2v_z*/g` / C 0.3 s,
      dt 10 ms. **위상 시간은 1차에서 고정**, J1.3에서 스윕
- [ ] 제약: 토크 한계, 관절 range, ankle ω < 40 rad/s (J0 예산),
      thrust 중 GRF ≥ 0 (마찰원뿔 포함)
- [ ] warm-start: J0.2 스크립트 점프 궤적을 초기해로 주입
      — **sit/getup 교훈 그대로: 솔버는 최적화기이지 발견기가 아니다.**
      cold start 수렴 실패는 버그가 아니라 기본 예상값으로 취급

### 2.2 flight 상의 핵심 물리 — 다리 무거운 분포의 역습

flight에서 base는 underactuated: 총 각운동량 보존. **다리가 무거우므로
공중 tuck 동작 자체가 몸통을 크게 회전시킨다** (일반 사족 대비 효과 큼).
이것이 R.pet 점프의 최대 난소이자, 잘 쓰면 자세 교정 수단이다.

- [ ] flight 비용에 각운동량 잔차 추가: 이륙 시점 각운동량 L₀를 최소화하는
      항을 Phase A 터미널에, flight 중에는 "터치다운 tilt 최소"를 터미널로
      → 최적화가 tuck 타이밍·좌우 대칭을 스스로 조정하게 함
- [ ] 진단 로그: 이륙 순간 L₀, 터치다운 순간 tilt·각속도 — 이 둘의 상관이
      튜닝의 나침반 (L₀ 크면 무조건 착지 tilt로 되돌아옴)

### 2.3 J1 검증 (완료 기준)

- [ ] 궤적 레벨(솔버 출력 자체): h ≥ J0 상한의 70%, 토크·ω 마진 ≥ 10%,
      터치다운 tilt < 5°, 수렴 이터레이션·시간 기록
- [ ] 단위 테스트: 임팩트 전후 운동량 정합, 수치미분 대비 비용 그래디언트
- [ ] MuJoCo 재생(open-loop torque replay)은 **실패해도 정상** — soft contact
      불일치 때문(B안 전례). 판정은 J2 추종 후에

### 2.4 aligator 이식 (병행 가능, 비차단)

- [ ] 동일 정식화를 aligator/ProxDDP로: 하드 제약(토크·ω·마찰원뿔)을 ALM으로
      승격 — crocoddyl 페널티 대비 마진 신뢰도 개선이 목적
- [ ] §9(속도 진단) 체크리스트 선적용: 상태 스케일링, μ 증대. 점프 OCP는
      오프라인이므로 풀이 시간 수 분도 허용 — **속도 문제로 J1을 막지 말 것**

---

## 3. Phase J2 — 추종·배포 (착지가 본체)

sit/getup의 최대 교훈 3개가 전부 여기서 재등장한다:
(1) 관절 PD는 base 자세를 못 잡는다, (2) 균형 자세는 wbic_stance로,
(3) open-loop은 접촉 전환 타이밍에 극도로 민감하다(NG45 전례).

### 3.1 상별 추종 전략

```
crouch:    wbic_stance (검증된 스쿼트 경로 그대로)
thrust:    관절 PD + OCP 토크 피드포워드 (τ_ff + Kp(q_ref−q) + Kd(q̇_ref−q̇))
           — 짧고(≤0.3 s) 4발 접촉이라 open-loop성 허용
flight:    관절 PD로 tuck 궤적 추종 (base는 어차피 제어 불가)
touchdown: 감지 즉시 → 저-PD 임팩트 흡수 (0.05~0.1 s) → wbic_stance 회복
           — 저-PD는 "발이 몸 밑에 있는 짧은 흡수"에만, 홀드는 즉시 wbic로
```

- [ ] **touchdown 감지기**: 발끝 접촉력(시뮬) 또는 발 높이+하강속도 휴리스틱.
      감지 지연 1 스텝이 착지 성패를 가르므로 감지기 자체를 단위 테스트
      (오탐: flight 중 조기 감지 / 미탐: 침투 후 지연 감지, 양쪽 시나리오)
- [ ] **상 전환 트리거는 시간이 아니라 이벤트로**: thrust→flight는 GRF 소멸,
      flight→landing은 touchdown 감지 — NG45 교훈(시간 스케줄의 취약성) 반영
- [ ] wbic_stance 인계 게이트: sit/getup에서 확립한 이중 조건(명목+실제 bz)
      패턴 재사용 — 침투 상태에서 조기 인계 방지

### 3.2 강건성 스윕

- [ ] 초기 조건 randomize (자세 ±2°, z ±0.01 m) × 20회: falls=0 목표
- [ ] soft contact 파라미터 ±20% 스윕: 착지 성공률 유지 확인
      (B안 전례 — soft floor가 결과를 왜곡할 수 있으므로 명시적 스윕)

### 3.3 J2 완료 기준 (= M1 달성)

- [ ] C++ 뷰어 배포, clearance ≥ 0.10 m, 착지 falls=0 (20/20), tilt < 10°
- [ ] 실측 h vs J1 궤적 h vs J0 상한의 3단 비교표 — 간극 원인 기록
      (sit/getup의 "이상적 rollout 47° vs 배포 60°" 형식 그대로)

---

## 4. Phase J3 — MPPI 보조 트랙 (검증·강건화·탐색)

MPPI의 역할 3개 — **OCP 대체가 아님을 문서·커밋에 명시**:

### 4.1 교차 검증 (모델프리 상한)

- [ ] contact-implicit MPPI로 동일 점프 태스크 (비용: h 최대 + tilt 페널티,
      OCP 궤적을 seed로). MPPI h vs OCP h 비교:
  - MPPI ≫ OCP → OCP 정식화/제약이 과보수 (마진 재검토)
  - MPPI ≈ OCP → 상한 근접, 정상
  - `getup_mppi.py`의 기준선 역할 패턴 재사용

### 4.2 착지 강건화 (MPPI가 실제로 이길 수 있는 유일한 구간)

- [ ] 터치다운 직전 상태에서 시작하는 **짧은 호라이즌(0.3 s) landing MPPI**:
      교란된 착지 상태(각속도·tilt 랜덤)에서 회복 토크 시퀀스 탐색
- [ ] 산출물: 교란 → 회복 전략의 데이터셋 (RL 하이브리드 H1 확장 소재 —
      착지 tracking policy 학습용 궤적 라이브러리)

### 4.3 타이밍 탐색

- [ ] (t_thrust, crouch z0) 2D 스윕을 MPPI/시뮬 평가로 — run 스윕(T×STEP_H)과
      동일한 인프라 재사용. J1의 위상 시간 고정을 여기서 보정

### 4.4 J3 완료 기준

- [ ] 3단 비교(J0 상한 / OCP / MPPI) 표 + landing MPPI 회복률 리포트

---

## 5. Phase J4 — 전진 점프·연속 점프 (M2, M3)

- [ ] M2: J1 정식화에 터미널 CoM 수평 변위 추가 (v_x* 도입, flight 시간 재계산).
      착지 시 수평 운동량 흡수 — 마찰원뿔 제약이 여기서 활성화됨 (미끄러짐 주의:
      눕기 슬라이드 버그 전례처럼 착지 슬라이드를 지표에 포함)
- [ ] M3: 착지 → wbic_stance 정착 판정(기존 이중 게이트) → crouch 재진입 루프.
      연속 3회. 상태기계는 sit/getup 전환 매트릭스 프레임에 `jump` 상태 추가
- [ ] 회귀: `jump` 상태 추가 후 기존 전환 매트릭스 전 항목 falls=0 유지

---

## 6. 함정 사전 등록

### 6.1 MuJoCo soft contact — 추진 침투
thrust 피크 GRF에서 발끝 침투가 커지며 실효 추진 거리·에너지가 깎인다.
대응: J2.2의 파라미터 스윕 + 침투 깊이 로깅. OCP(강체 접촉)와의 h 간극의
1차 용의자로 사전 지목해 둔다.

### 6.2 ankle ω 스파이크
추진 말기 신전 + tuck 개시가 겹치는 순간 ankle ω 급증 (run STEP=0.16 전례:
156 rad/s → 대량 낙상). 대응: OCP ω 제약을 하드로(aligator ALM), 추종 시
ω 로거로 상시 감시, tuck 개시를 이륙 후 50 ms 지연.

### 6.3 이륙 순간 비대칭 → 공중 회전
앞 3-DoF/뒤 4-DoF 비대칭 + 좌우 GRF 불균형이 L₀를 만든다. 대응: §2.2의
L₀ 최소화 터미널 항 + 이륙 GRF 좌우 대칭 비용. 진단: L₀ vs 터치다운 tilt 상관 로그.

### 6.4 touchdown 감지 실패 모드
조기 감지(flight 중 오탐) → wbic가 허공에서 GRF 요구 → 발산.
지연 감지 → 침투 상태 임팩트 → 튐. 대응: §3.1 감지기 단위 테스트 2종 필수.

### 6.5 시간 스케줄 취약성 (NG45 재발 방지)
위상 전환을 시간으로 걸면 준정적 sit/getup보다 훨씬 민감하다.
대응: §3.1 — 전환은 전부 이벤트 트리거. 시간은 타임아웃 안전장치로만.

### 6.6 임팩트 모델 불일치
crocoddyl 임펄스 모델(순간 임팩트) vs MuJoCo soft contact(유한 시간 흡수).
착지 직후 0.05 s는 궤적 추종을 포기하고 저-PD 흡수로 넘기는 설계가 대응책
(§3.1) — 임팩트 구간을 "추종하지 않는 구간"으로 명시.

---

## 7. 확장 백로그 (본 문서 범위 밖, 기록만)

- twist jump (yaw 회전 점프) — §2.2 각운동량 프레임 그대로 확장 가능
- 장애물 점프 (clearance 제약 추가)
- 온라인 점프 MPC (aligator RTI 성숙 후)
- 착지 tracking policy RL 학습 (J3.2 데이터셋 → 하이브리드 로드맵 H1 확장)
- 6-DoF 머리 반작용 질량 결합 (RPET_HEAD_GAZE_MPC.md G4 — 공중 자세 교정에
  머리 스윙 활용. **점프가 머리 작업보다 선행하는 이유가 바로 이 결합점**:
  점프의 flight 자세 문제가 head reaction mass의 최고 테스트베드다)

---

## 8. 파일 구조 제안

```
rpet_jump/
├── feasibility/
│   └── jump_feasibility.py          # J0.1 상한 맵
├── scripted/
│   └── jump_scripted.py             # J0.2 기준선 (SGU 패턴)
├── ocp/
│   ├── jump_ocp_crocoddyl.py        # J1.1–J1.3
│   └── jump_ocp_aligator.py         # J1.4
├── deploy/
│   ├── jump_state_machine.cpp       # J2 (crouch/thrust/flight/land 상태기계)
│   ├── touchdown_detector.cpp       # J2 감지기
│   └── jump_tracker.cpp             # τ_ff + PD 추종
├── mppi/
│   ├── jump_mppi.py                 # J3.1 (getup_mppi.py 확장)
│   └── landing_mppi.py              # J3.2
├── eval/
│   ├── jump_metrics.py              # h, clearance, tilt, L₀, ω, falls 로거
│   └── sweep_runner.py              # J2.2 / J3.3 스윕
└── tests/
    ├── test_touchdown_detector.py   # §6.4 오탐/미탐 2종
    └── test_impulse_consistency.py  # J1 임팩트 정합
```

## 9. 마일스톤 요약

| Phase | 산출물 | 완료 기준 | 예상 규모 |
|---|---|---|---|
| J0 | 상한 맵 + 스크립트 점프 | 상한 산출, 이륙 성립, **go/no-go 판정** | 3–5일 |
| J1 | OCP 점프 궤적 | h ≥ 상한 70%, tilt < 5° (궤적 레벨) | 1–2주 |
| J2 | C++ 배포 (M1) | clearance ≥ 0.10 m, falls 0/20, 3단 비교표 | 1–2주 |
| J3 | MPPI 검증·landing 데이터셋 | 3단 비교 + 회복률 리포트 | 1주 (병행 가능) |
| J4 | 전진·연속 점프 (M2, M3) | 각 기준 + 전환 매트릭스 회귀 | 1–2주 |

권장 순서: **J0 → J1 → J2** 직렬 (J0의 go/no-go가 게이트),
J3는 J1 완료 후 병행, J4는 M1 달성 후.

## 10. 커밋 규율

- Phase 단위 브랜치 (`feat/jump-j0-feasibility`, `feat/jump-j2-deploy`, ...)
- 모든 결과에 전제 병기 (sit/getup 원칙): HEAD 없음/payload 0/soft contact
  기본값 등 조건 명기. 특히 J0 상한은 "현 토크·기어비 전제"임을 못 박을 것
- 3단 비교표(J0 상한/OCP/배포 실측)는 이후 모든 동적 maneuver의 표준 리포트
  형식으로 승격 — 리포트 템플릿화
