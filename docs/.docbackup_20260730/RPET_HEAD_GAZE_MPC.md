# RPET_HEAD_GAZE_MPC.md — 6-DoF 머리 시선·전신 협조 구현 브리핑

> R.pet 17-DoF(waist, all-ankle) 모델에 6-DoF 머리(목) 체인을 추가하고,
> ① 예측형 시선 안정화, ② 반작용 질량(reaction mass) 협조, ③ 시선–균형 중재를
> 단계적으로 구현하기 위한 개발 문서. Claude Code / 직접 구현 겸용.
>
> 관련 문서: `RPET_ALIGATOR_MPC.md`(Phase 0–3 로드맵), sit/getup 기술 기록 §8,
> MPC-RL 하이브리드 로드맵 §10.3.
>
> 특허 연계: 능동 시선 제어 모듈(듀얼 IMU 방진 피드백) = 반응층.
> 본 문서의 산출물 = 예측층. 두 층은 인터페이스만 공유하고 독립 동작해야 한다
> (특허 청구항 분리 관점에서도 결합도를 낮게 유지할 것).

---

## 0. 목표와 범위

### 0.1 최종 목표 상태

```
[예측층: 본 문서]                         [반응층: 특허 모듈]
MPC/OCP base 예측 궤적                    듀얼 IMU 잔여 진동 측정
      │                                        │
      ▼                                        ▼
head feedforward q_ff(t), q̇_ff(t)  ──(+)──  IMU 피드백 보정 Δq_fb
                                       │
                                       ▼
                              head joint 명령 (6-DoF)
```

- 예측층이 보행 진동의 **저주파·예측 가능 성분**(gait 주기 성분)을 선제 상쇄
- 반응층이 **잔여 고주파·비예측 성분**(지면 충격, 모델 오차)을 흡수
- 어느 한 층이 죽어도 다른 층 단독으로 동작 (graceful degradation)

### 0.2 범위 제외 (명시적 non-goal)

- 카메라 영상 기반 visual servoing (target은 3D 좌표로 주어진다고 가정)
- 머리 하드웨어 설계 자체 (단, §7.2 질량 예산 가이드는 설계팀 전달용)
- 반응층(IMU 피드백) 내부 수정 — 인터페이스만 정의

### 0.3 전제

- 기반 모델: `quad_real_17dof_waist_sphere.mjcf` (17-DoF) → 23-DoF로 확장
- 기존 스택: Convex MPC + WBIC(A안), crocoddyl/aligator OCP, contact-implicit MPPI
- WBIC에 각운동량 보상 task, 자세목표 quat 주입(sit_pitch) 패턴 구현 경험 있음
  → G2의 gaze task는 이 두 패턴의 직접 확장이다

---

## 1. 수학 정식화 (전 Phase 공통)

### 1.1 좌표계와 기호

| 기호 | 의미 |
|---|---|
| `W` | world frame |
| `B` | base(몸통) frame |
| `C` | camera frame (head 말단 링크에 고정) |
| `z_C(q) ∈ R³` | 카메라 광축 단위벡터 (world 표현), C frame의 +z축 |
| `p_C(q) ∈ R³` | 카메라 원점 위치 (world) |
| `p_T ∈ R³` | 응시 목표점 (world) |
| `u(q) = (p_T − p_C) / ‖p_T − p_C‖` | 목표 방향 단위벡터 |
| `q = [q_base(7); q_legs(14); q_head(6)]` | 23-DoF 일반화 좌표 (base는 pos+quat) |

### 1.2 gaze 오차 정의 (3안 비교 — G2에서 (b) 채택 권장)

**(a) 내적 비용**: `c = 1 − z_Cᵀu`
간단하지만 오차가 클 때 그래디언트가 죽고(π 근처), 방향 정보가 스칼라로 뭉개짐.

**(b) 축 오차 벡터 (채택)**: `e_gaze = z_C × u ∈ R³` (소각 근사에서 회전 오차의 sin)
- 3차원 잔차라 task Jacobian이 자연스럽고, WBIC/OCP 양쪽에 그대로 들어감
- 주의: `z_C = −u`(정반대)에서 특이 → §8.1의 fallback 필요
- 광축 둘레 roll은 무구속 (2-DoF task). roll까지 잡으려면 (c) 사용

**(c) full SO(3) 오차**: 목표 자세 `R_des = look_at(p_C, p_T, up)`에 대해
`e = log(R_desᵀ R_C)∨`. sit_pitch에서 쓴 `mju_quat2Vel` 패턴과 동일 계열.
카메라 수평 유지(roll 제어)까지 필요하면 이걸로 승격.

### 1.3 task Jacobian

`e_gaze = z_C × u`의 시간 미분 (target 정지 가정 시 u̇의 p_C 의존항 포함):

```
ė_gaze = −[u]× ż_C + [z_C]× u̇
ż_C = ω_C × z_C = −[z_C]× J_ω(q) q̇        (J_ω: C frame 각속도 Jacobian)
u̇  = (I − uuᵀ)/d · ṗ_C = −(I − uuᵀ)/d · J_v(q) q̇   (d = ‖p_T − p_C‖)
⇒ J_gaze = [u]× [z_C]× J_ω + [z_C]× (I − uuᵀ)/d · J_v   ∈ R^{3×22}
```

- Pinocchio: `getFrameJacobian(model, data, cam_frame_id, LOCAL_WORLD_ALIGNED)`
  → 상하 3행 분리로 `J_v`, `J_ω` 획득
- MuJoCo(WBIC 쪽): `mj_jacSite`로 camera site의 `jacp`(=J_v), `jacr`(=J_ω)
- `d`가 크면 `u̇` 항은 무시 가능 (원거리 응시 근사, d > 2 m면 drop 권장)

### 1.4 base 진동 → head 보상의 해석적 관계 (G3 경량 버전용)

base가 자세 섭동 `δθ_B`(roll/pitch/yaw), 위치 섭동 `δp_B`를 가질 때,
광축을 목표에 고정하기 위한 head 보상(1차 근사):

```
δθ_head ≈ −R_BH δθ_B  −  R_BH [ (I − uuᵀ)/d · δp_B ]_각도화
```

원거리 응시(d 큼)면 둘째 항 무시 → **δθ_head ≈ −δθ_B (base 역회전)**.
즉 경량 피드포워드는 "MPC가 예측한 base 자세 궤적의 역회전을 head 관절에 IK로 배분"
이면 충분하다. 이것만으로도 gait 주기 성분의 대부분이 잡힌다 (G3-lite).

---

## 2. Phase G0 — 모델 확장 (23-DoF)

### 2.1 작업 항목

- [ ] `quad_real_23dof_head.mjcf` 생성: 기존 17-DoF + neck/head 체인 6관절
  - 관절 구성(예시, 실제 하드웨어 확정 전 placeholder):
    `neck_yaw → neck_pitch → neck_roll → head_yaw → head_pitch → head_roll`
  - 각 관절 `range`, `damping`, `armature`를 실제 액추에이터 스펙으로. 미정이면
    다리 ankle 값에서 스케일 다운한 placeholder + `TODO(hw)` 주석
- [ ] camera **site** 추가 (head 말단): `<site name="cam_optical" ...>`
  - site의 +z가 광축이 되도록 quat 정렬. **광축 정의는 이 site 하나로 단일화**
    (WBIC/OCP/MPPI 전부 이 site만 참조 — 정의 분산 금지)
- [ ] URDF 동기 버전 생성 (Pinocchio/aligator용). SW2URDF 경유 시
  **foot-link 질량 흡수 버그 전례** 있으므로 head 말단 링크 inertial 반드시 검수
- [ ] 질량/관성: 설계 미확정이므로 파라미터화
  - `HEAD_MASS` sweep 대상: {0.5, 1.0, 1.5, 2.0} kg (§7.2 예산 실험)
  - 관성은 균질 실린더/박스 근사로 질량에서 유도

### 2.2 검증 (G0 완료 기준)

- [ ] `mj_step` 정지 자세에서 총 질량 = 15.24 kg + HEAD_MASS 확인
- [ ] Pinocchio `crba` 질량행렬 조건수 로깅: 17-DoF 대비 악화율 기록
  (HEAD_MASS별 — G4/aligator 수렴성 진단의 기준 데이터)
- [ ] 기존 17-DoF 회귀: head 관절 고정(freeze) 시 trot/run/sit-getup 전환 매트릭스
  falls=0 유지 (head 추가가 기존 스택을 깨지 않음을 먼저 보장)

---

## 3. Phase G1 — 반응층 기준선 + 인터페이스 동결

목적: 예측층의 효과를 측정할 **기준선**과, 특허 모듈과의 **합산 인터페이스** 확정.

### 3.1 작업 항목

- [ ] 반응형 안정화의 시뮬 재현: base IMU(=base 자세·각속도 센서)로
  `Δq_fb = −Kp·δθ_B − Kd·ω_B`를 head 관절에 매핑하는 단순 PD 역회전 컨트롤러
  (특허 모듈의 동작을 근사하는 시뮬 대역 — 실물 모듈 코드가 아님)
- [ ] 인터페이스 동결 (헤더 1개로):

```cpp
// head_cmd_interface.h
struct HeadCmd {
  double q_ff[6];    // 예측층: 피드포워드 각도
  double dq_ff[6];   // 예측층: 피드포워드 각속도
  double q_fb[6];    // 반응층: IMU 피드백 보정 (특허 모듈 출력 자리)
  // 최종 명령 = q_ff + q_fb (관절 range로 clamp, §5.2의 중재 로직이 배분)
};
```

- [ ] 평가 지표 정의 및 로거 구현 (전 Phase 공통 사용):
  - `gaze_err_rms` [deg]: ∠(z_C, u)의 RMS
  - `gaze_err_p95` [deg]
  - `head_torque_rms`, `head_power_mean`
  - `pixel_blur_proxy`: 광축 각속도 RMS (카메라 블러 대리 지표)

### 3.2 기준선 측정 매트릭스

| 조건 | gait | 측정 |
|---|---|---|
| head 고정 (보상 없음) | trot 1.0 / run 2.18 m/s | gaze_err 상한 확인 |
| 반응층만 (G1 PD) | trot / run | 반응층 단독 성능 = 예측층이 넘어야 할 선 |

---

## 4. Phase G2 — WBIC gaze task (기존 스택, 최단 경로 데모)

OCP 없이 **지금 스택으로 1~2주 내 데모 가능한 층**. sit_pitch 자세목표 주입과
각운동량 task 두 패턴의 합성이므로 코드 재사용률이 높다.

### 4.1 작업 항목

- [ ] WBIC task 목록에 `TaskGaze` 추가
  - 잔차: `e_gaze = z_C × u` (§1.2-b), Jacobian: §1.3 (mj_jacSite 기반)
  - task 동역학: `ẍ_des = −Kp·e_gaze − Kd·ė_gaze` (Kp≈100, Kd≈20에서 시작)
- [ ] 우선순위 배치 (기존 계층에 삽입):

```
1. contact 유지           (기존)
2. CoM / base 자세        (기존 — 균형이 항상 gaze보다 상위)
3. 각운동량 보상          (기존)
4. gaze (신규)            ← 여기
5. posture (nominal head = 정면)  (기존 posture에 head 6-DoF 추가)
```

- [ ] head 관절 torque/속도 한계를 WBIC QP 부등식에 추가
- [ ] target 입력 경로: 고정점 / 원궤도 이동점 / 조작자 지정, 3모드 스위치

### 4.2 검증 (G2 완료 기준)

- [ ] 정지(wbic_stance) + 이동 target: gaze_err_rms < 2°
- [ ] trot 1.0 m/s + 고정 target: gaze_err_rms < 5° (head 고정 대비 ≥70% 감소)
- [ ] 회귀: gaze task 추가 후에도 보행 falls=0, CoM 오차 악화 < 10%
- [ ] 주의: G2는 **반응형**이다 (현재 상태 기반 servo). 예측이 아님.
  G3와의 차이를 리포트에서 명확히 구분해 기록할 것 — 이 구분이 특허 논리의 핵심.

---

## 5. Phase G3 — 예측형 피드포워드 (①의 본체)

### 5.1 G3-lite: 해석적 피드포워드 (권장 선행)

MPC(A안)는 매 주기 base 예측 궤적 `{p_B(t_k), θ_B(t_k)}_{k=0..N}`을 이미 계산한다.
이를 §1.4의 역회전 관계로 head 레퍼런스에 배분:

- [ ] MPC 출력에서 base 자세 예측열 추출하는 hook 추가 (A안 코드 수정 최소화:
  기존 예측 상태 버퍼를 read-only 참조)
- [ ] `head_ff_generator`: 예측열 → `δθ_head(t)` → 6관절 IK 배분
  - 배분 규칙: pitch/roll 보상은 neck_pitch/roll 우선, 잔여를 head_pitch/roll로;
    yaw는 head_yaw 우선 (가동범위·속도한계 고려한 waterfall)
  - 출력은 `HeadCmd.q_ff/dq_ff` (G1 인터페이스)
- [ ] 지연 보상: 제어 지연 `τ_delay`만큼 예측열을 앞당겨 샘플링
  (`t_query = t + τ_delay`) — 실기 이식 시 핵심 파라미터
- [ ] G2의 WBIC gaze task와 병행 시: gaze task는 잔차만 처리하도록
  `u` 계산에 q_ff 반영 (이중 보상 방지)

### 5.2 G3-full: OCP gaze cost (aligator/crocoddyl)

전신 OCP 비용에 gaze 잔차를 넣어 "보행 진동을 미리 상쇄하는 머리 궤적"을
최적화가 직접 생성하게 한다. G3-lite가 잘 되면 이건 정밀화 단계.

- [ ] crocoddyl `ResidualModelAbstract` 상속: `ResidualGaze`
  - `calc`: §1.2-b, `calcDiff`: §1.3의 J_gaze (Pinocchio frame Jacobian)
  - 단위 테스트: 수치미분 대비 해석 Jacobian 오차 < 1e-6
- [ ] aligator 스테이지 비용에 `w_gaze · ‖e_gaze‖²` 추가
  - 초기 가중치: `w_gaze = 0.1 × w_base_pose`에서 시작 (균형이 항상 우위)
- [ ] head 관절 한계는 ALM 부등식으로 (aligator 강점 활용)
- [ ] **오프라인 모드 우선**: §9(aligator 속도 진단)에 따라 G3-full은
  실시간이 아니라 "정밀 head+body 궤적 라이브러리 생성기"로 먼저 가치를 낸다.
  실시간화는 RTI 적용 이후 별도 판단.

### 5.3 검증 (G3 완료 기준)

- [ ] trot/run에서 층별 ablation (동일 시드, 각 30 s × 5 회):

| 구성 | 기대 서열 |
|---|---|
| head 고정 | 최악 (기준) |
| 반응층만 (G1) | 개선 |
| G2 WBIC servo만 | 개선 |
| G3-lite ff만 | G2와 비등 이상 (예측 성분에서 우위) |
| **G3 ff + 반응층** | **최선** — 2층 구조의 가치 실증 |

- [ ] 목표: run 2.18 m/s에서 `ff+반응층`의 gaze_err_rms가 반응층 단독 대비 ≥40% 감소
- [ ] 이 ablation 표가 특허 후속(예측층 청구항)의 실험 근거 — 결과를 커밋에 보존

---

## 6. Phase G4 — 반작용 질량 실험 (②)

### 6.1 OCP 트랙

- [ ] G3-full 셋업에서 급회전 태스크: yaw rate step (0 → 2 rad/s) 추종 OCP
  - 각운동량 정칙화 가중치를 낮춰 head 사용 자유도 부여
  - 관찰: 최적해가 head를 몸통 회전 반대로 스윙하는가 (각운동량 상쇄)
  - 정량: swing 다리 기인 각운동량 피크 vs head 기여 각운동량의 상쇄율
- [ ] 착지 교정: 낙하 초기 자세에 tilt 부여 → 착지 전 head 스윙으로 자세 교정하는
  해가 나오는지 (치타 꼬리 효과 재현)

### 6.2 MPPI 트랙 (탐색 실험)

- [ ] 기존 contact-implicit MPPI에 head 6-DoF 추가하되 **Σ 블록 분리**:
  - `Σ = blkdiag(Σ_legs, Σ_head)`, `Σ_head`는 별도 스케일 파라미터
  - raw 23-DoF 토크 샘플링 비효율 대비: head는 3-DoF gaze 방향 파라미터화
    (§10.4 교훈) 또는 스플라인 노트 파라미터화 옵션 병행
- [ ] 복구(getup) maneuver 재실행: head 자유 vs head 고정 비교
  - 가설: head 스윙이 자발적으로 나타나고 bounce가 추가 감소
  - **주의**: sit/getup 교훈 — MPPI는 발견기가 아니다. head 스윙이 안 나오면
    "불가"가 아니라 "seed 필요"로 해석하고, G4.1의 OCP 해를 seed로 재시도

### 6.3 검증 (G4 완료 기준)

- [ ] 급회전: head 자유 시 yaw 추종 오차 또는 tilt 피크가 head 고정 대비 개선
  (개선 없으면 HEAD_MASS 스윕으로 유효 질량 하한 규명 — 그 자체가 설계 입력)

---

## 7. Phase G5 — 시선–균형 중재 (③)

### 7.1 작업 항목

- [ ] **소프트 중재 (기본)**: WBIC/OCP의 가중치·우선순위 구조가 이미 중재를 수행
  (균형 상위, gaze 하위). 여기서는 상황 적응만 추가:
  - 험지/외란 감지 시(예: tilt > 임계, 접촉 이상) `w_gaze`를 램프 다운
  - 안정 회복 후 램프 업 (히스테리시스 포함)
- [ ] **가동범위 소진 핸들링**: head 관절이 range 경계 δ 이내로 접근하면
  base yaw 목표에 gaze 잔차의 yaw 성분을 점진 주입
  - `yaw_cmd_base += k_leak · sat(e_gaze,yaw)` (k_leak 램프, 보행 명령과 합성)
  - 이는 "몸이 고개를 도와 돌아서는" 동작 — 데모 가치 높음
- [ ] 상태기계 연동: sit/getup 등 전환 중에는 gaze task 자동 저가중
  (전환 매트릭스 falls=0 회귀 보장)

### 7.2 하드웨어 설계 피드백 (설계팀 전달물)

- [ ] G0 조건수 데이터 + G4 유효 질량 데이터로 **head 질량/관성 예산표** 작성:
  - 반작용 질량 효과의 하한 질량 vs OCP 수렴성(ill-conditioning) 상한 질량
  - 원위 질량 최소화 원칙: 무거운 부품(액추에이터)은 목 기저부로

---

## 8. 함정과 대응 (사전 등록)

### 8.1 gaze 특이점
`z_C = −u`(목표가 정후방)에서 §1.2-b 잔차 특이. 대응: `z_Cᵀu < cos(150°)`이면
gaze task 비활성 + base yaw 재배향 모드로 전환 (G5.1의 leak 로직 재사용).

### 8.2 이중 보상 (G2 servo × G3 ff 충돌)
ff가 이미 상쇄한 진동을 servo가 또 보상하면 과보상 발진. 대응: §5.1처럼
servo 잔차 계산에 ff 반영, 또는 G3 활성 시 G2 게인 1/3로 다운.

### 8.3 반응층과의 합산 발진
예측층 오차를 반응층이 먹고, 그 보정이 예측층 가정과 어긋나는 루프.
대응: 주파수 분리 — 예측층은 gait 주기 대역(≤ 2·gait Hz)만 담당하도록
q_ff에 저역필터, 반응층이 고주파 전담. 분리 컷오프는 gait 주기의 2배에서 시작.

### 8.4 ill-conditioning (기존 리스크의 확장)
다리 무거운 분포 + 원위 head 질량. 대응: G0의 조건수 로깅을 CI 지표로,
aligator μ 증대·상태 스케일링(§9.2 체크리스트 4번) 우선 적용.

### 8.5 MuJoCo/Pinocchio 프레임 불일치
site 광축 정의가 두 모델에서 어긋나면 G2(WBIC)와 G3(OCP)가 서로 다른 축을
쫓는다. 대응: G0에서 동일 q에 대해 두 라이브러리의 `z_C` 일치 단위 테스트
(오차 < 1e-9) — MJCF↔URDF 변환 회귀에 포함.

### 8.6 target 근접 시 u̇ 폭주
d → 0에서 §1.3의 (I−uuᵀ)/d 발산. 대응: `d_min = 0.5 m` clamp + 근접 시
gaze task 게인 스케일 다운.

---

## 9. 파일 구조 제안

```
rpet_head/
├── models/
│   ├── quad_real_23dof_head.mjcf        # G0
│   └── quad_real_23dof_head.urdf        # G0 (Pinocchio용)
├── include/rpet_head/
│   ├── head_cmd_interface.h             # G1 (동결 인터페이스)
│   ├── gaze_math.h                      # §1 잔차/Jacobian (양 라이브러리 공용)
│   └── head_ff_generator.h              # G3-lite
├── src/
│   ├── task_gaze_wbic.cpp               # G2
│   ├── head_ff_generator.cpp            # G3-lite
│   └── arbitration.cpp                  # G5
├── ocp/
│   ├── residual_gaze.py|.cpp            # G3-full (crocoddyl/aligator)
│   └── turn_reaction_mass.py            # G4.1
├── mppi/
│   └── getup_head_mppi.py               # G4.2 (기존 getup_mppi.py 확장)
├── eval/
│   ├── gaze_metrics.py                  # G1 지표 로거
│   └── ablation_runner.py               # G3.3 ablation 매트릭스
└── tests/
    ├── test_jacobian_fd.py              # 해석 vs 수치미분
    └── test_frame_consistency.py        # §8.5
```

## 10. 마일스톤 요약

| Phase | 산출물 | 완료 기준 | 의존성 |
|---|---|---|---|
| G0 | 23-DoF 모델 + 조건수 데이터 | 17-DoF 회귀 falls=0 | — |
| G1 | 반응층 기준선 + 인터페이스 | 지표 로거 동작 | G0 |
| G2 | WBIC gaze task 데모 | trot에서 err ≥70%↓ | G0 |
| G3 | 예측 ff (lite→full) | ablation: ff+fb 최선, ≥40%↓ | G1, G2, (full: aligator) |
| G4 | 반작용 질량 실증 | 급회전 개선 or 질량 하한 규명 | G3-full 셋업 |
| G5 | 중재 로직 + 설계 예산표 | 전환 매트릭스 회귀 falls=0 | G2–G4 |

권장 착수 순서: **G0 → G1 → G2** (기존 스택만으로 3주 내 첫 데모)
→ G3-lite (예측층 가치 실증 + 특허 실험 근거) → 이후 병렬.

## 11. 커밋 규율

- Phase 단위 브랜치 (`feat/head-g0-model`, `feat/head-g2-wbic-gaze`, ...)
- 각 Phase 완료 시 검증 지표를 커밋 메시지에 숫자로 박제
  (sit/getup 문서의 "결론에는 전제를 병기" 원칙 준수 — 특히 G2 servo와
  G3 예측의 구분, HEAD_MASS 조건을 반드시 명기)
- G3.3 ablation 결과는 특허 후속 검토 자료이므로 원시 로그까지 보존
