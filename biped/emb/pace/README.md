# pace — 액추에이터 마찰 측정 · PACE 파라미터 식별

실기(SHM 경유)에서 **정지마찰 · 쿨롱마찰 · 점성감쇠 · 반사관성**을 뽑아
프로젝트가 쓰는 `ROTOR_I / JDAMP / JFRIC` 로 내보낸다.

레퍼런스: `motorcortex-python-tools/automatic_testing_examples`
(VECTIONEER — 저속 사인 가진 + 위치/속도 대비 토크 로깅 + HTML 리포트 구조를 계승).

## 실행

```bash
# 1) 하드웨어 없이 추정기 수학만 검증 (모터 무동작, 언제든 안전)
python3 actuator_test.py --selftest

# 2) 실기 — Emb 기동 후 5초 이상 지난 뒤
python3 actuator_test.py --ch 0 --tests friction
python3 actuator_test.py --ch 0 --tests friction,pace
python3 actuator_test.py --all --tests friction,pace     # spec 의 installed_channels 전부
```
리포트: `results/output.html` · 데이터셋: `results/pace_dataset_ch##.npz`

### 실행 전 필수 확인
1. **Emb 기동 후 5초 경과** — `halGait` 초기화 게이트(100+4500 tick @1kHz) 전에는
   Emb 가 SHM 명령을 아예 읽지 않는다.
2. **모터 명령 writer 는 하나만** — `app/biped_emb.py`, `RobotTestGait`, `diag/mot_test` 종료.
   (하니스가 `preflight()` 에서 검사하고 거부한다.)
3. 로봇 거치(크레인/스탠드).
4. 텔레메트리 신선도 — Emb 는 EtherCAT 이 OP 를 잃어도 마지막 값을 계속 재발행하며
   플래그까지 세운다. `hwio` 가 무변화 지속시간을 재서 차단하지만, 사전에
   `diag/stt_probe 8` 로 "값이 갱신됨" 을 확인하는 게 좋다.

## 안전 설계

- **가진은 전부 위치+게인(임피던스) 모드.** 생 토크명령을 쓰지 않는다 →
  토크가 `Kp·err` 로 상한이 걸려 폭주가 구조적으로 불가능. 토크는 *측정*해서 쓴다.
- `hwio.Hardware` 가 매 틱 강제: 위치한계 · 속도 · 추종오차 · 토크(지속시간 포함) · **stale**.
  위반 시 즉시 limp 후 `SafetyAbort`.
- **종료는 언제나 limp** — 정상·예외·Ctrl-C·SIGTERM 전부 같은 경로.
  (명령을 "안 쓰는" 것은 정지가 아니다. Emb 는 마지막 명령을 1kHz 로 영원히 재전송한다.)
- 인가는 측정각 래치 후 게인 0→목표 **0.3s 램프**(스텝 토크 스파이크 방지).
- `kp≤60 / kd≤3` 하드클립 — 단위 스케일 실수가 그대로 드라이버로 가지 않게.

## 측정 방법과 그 이유

### 양방향 상쇄 (이 구현의 핵심)
같은 구간을 +/− 방향으로 통과시키면
```
τ⁺ = +f(v) + g(q) + bias        f(v)      = (τ⁺ − τ⁻)/2    ← 마찰만
τ⁻ = −f(v) + g(q) + bias   →    g(q)+bias = (τ⁺ + τ⁻)/2    ← 중력+바이어스
```
로 분리된다. 셀프테스트가 이걸 정량으로 보여준다 — **상쇄를 빼면 JFRIC 이 +92% 과대**.
그리고 `g+bias` 가 속도에 따라 변하면 그 자체가 이상신호(자세이동·열드리프트)라
품질 지표로도 쓴다.

### (A) 정지마찰 breakaway
목표각을 `0.6 deg/s` 로 아주 천천히 밀어 `Kp·err` 토크를 키우다가 `|q̇|` 가 임계를
넘는 순간의 토크를 기록. 방향당 3회 반복해 산포를 본다.

### (B) 등속 스윕 → **JFRIC · JDAMP**
여러 속도(2~70 deg/s)를 양방향으로 통과, 가감속 구간을 뺀 중앙 60% 구간만 평균.
`f(v) = τ_c + b·v` 회귀 → `τ_c=JFRIC`, `b=JDAMP`.
데이터가 충분하면 Stribeck `τ_c + (τ_s−τ_c)·exp(−(v/v_s)²) + b·v` 도 시도해
더 잘 맞을 때만 채택.

### (C) 저속 사인 (레퍼런스 방식)
`0.05 Hz` 사인 → 위치-토크 / 속도-토크 마찰 루프 플롯.
midstroke 히스테리시스 폭 ≈ `2·τ_c` 라 (B) 결과의 교차검증에 쓴다.

### (D) PACE 처프 → **ROTOR_I**
`0.2→4 Hz` 처프로 가속도를 여기시키고
```
τ = I_total·q̈ + b·q̇ + τ_c·tanh(q̇/ε) + A·sin q + B·cos q + c
```
를 회귀(선형 LSQ → scipy 비선형 정련). 그리고
```
ROTOR_I = (I_total − I_link) / N²
```
`I_link` = 그 자세에서의 관절축 링크관성. **MJCF 에서 얻어야 한다**(아래 참조).

> **어느 시험의 값을 쓸 것인가**
> 셀프테스트 기준 — `JFRIC/JDAMP` 는 **(B) 등속 스윕이 정확**(상대오차 0.0%),
> 처프 회귀는 속도노이즈 때문에 JDAMP 가 ~20% 흔들린다.
> 반면 `I_total`(→ROTOR_I)은 가속도 여기가 필요해 **(D) 처프에서만** 나온다.
> → **JFRIC/JDAMP 는 (B), ROTOR_I 는 (D)** 를 채택하는 것을 권장.

## 식별 가능성 확인 (반드시 볼 것)

리포트의 다음 항목이 신뢰도를 결정한다.
- **파라미터별 ±표준편차** — 값과 같은 자릿수면 그 파라미터는 식별되지 않은 것이다.
  특히 `I_total ± ` 가 값의 50% 를 넘으면 처프 상한주파수/진폭을 올려 재측정.
- **조건수** — 수백을 넘으면 파라미터 간 상관이 커 분리 신뢰도가 떨어진다.
- **R²·잔차 RMS** — R² 가 0.8 미만이면 모델이 못 잡는 성분(백래시·시간지연·온도).
- **중력+bias 의 속도 무관성** — 속도에 따라 변하면 상쇄 전제가 깨진 것.

## 노트북에서 마무리할 것 (원래의 PACE = CMA-ES sim-매칭)

로봇에는 `mujoco`/`cma` 가 없다. 그래서 역할을 나눴다.
- **로봇**: 가진·수집·해석적 회귀 → `results/pace_dataset_ch##.npz`
- **노트북**: 그 npz 로 MuJoCo 재현매칭 `Σ(sim−real−bias)²` 를 CMA-ES 로 최소화.
  회귀 추정값을 `x0` 로 주면 수렴이 훨씬 빠르다. 목적함수 골격은
  `tests/act_identify_pace.py` 의 `CMAES_OBJECTIVE` 에 그대로 적어두었다.

`I_link` 도 노트북에서 얻는 게 정확하다 — 시험 자세에서
`mj_fullM` 의 해당 관절 대각성분 `M[i,i]` 가 곧 관절축 링크관성이다
(armature 를 0 으로 둔 모델에서 읽을 것).

## 스펙 입력 (`spec.yaml`)

`TODO` 로 남은 값이 있으면 하니스가 실행 시 목록을 찍고, 그 값에 의존하는 환산은
"미확정" 으로 보고한다(없는 값을 추측해 그럴듯한 숫자를 내지 않는다).

| 항목 | 왜 필요한가 | 현재 |
|---|---|---|
| `units.torque_frame` | 보고 토크가 모터축인지 관절축인지. 관절환산이 N배 달라진다 | **TODO** |
| `joints[].I_link` | `ROTOR_I = (I_total−I_link)/N²` 분리에 필수 | **TODO** |
| `joints[].kt_nm_per_a` | 전류-토크 교차검증(측정 tau 와 cur 이 일치하는지) | **TODO** |
| `joints[].gear` | 반사관성·관절환산 | 7 / 7 / 10.5 / 8.4 (문서값) |
| `joints[].q_min/max` | 시험 각도한계 | hip ±20° (checklist 확정값) |

⚠ 문서 간 불일치 두 건 — 확인 필요:
1. **foot 감속비**: `quad/PARAMS.md:91` 은 14.0(+GEAR_FOOT 0.5714 → 8:1),
   `docs/sim2real_checklist_17dof.html`(2026-07-21) 은 **8.4 실값**. 여기서는 8.4 채택.
2. **hip 가동범위**: checklist 는 **±20° 확정**("구 ±35° 폐기") 인데
   `emb/config/biped_emb.yaml` 은 아직 ±35. 시험은 보수적으로 ±20 을 따른다.

## 파일

```
pace/
├── spec.yaml                        하드웨어 스펙 + 안전한계 (사용자 입력)
├── hwio.py                          안전 SHM I/O (한계·stale·limp·인가램프)
├── actuator_test.py                 하니스 + --selftest
├── tests/act_measure_friction.py    (A)(B)(C) 마찰
├── tests/act_identify_pace.py       (D) PACE 처프 식별 + npz 내보내기
└── templates/base.html              리포트 템플릿
```

⚠ 이 디렉터리는 git 미추적이다. `~/simulation` 은 clone/sync 가 미추적 파일을 지우므로
(2026-08-05 에 `emb/diag/`·`emb/net/` 가 그렇게 삭제됐다) **커밋해 둘 것**.
