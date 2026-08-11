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
python3 actuator_test.py --ch 0 --tests latency        # 왕복지연·로스트모션
python3 actuator_test.py --ch 0 --tests friction,latency,pace
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
목표각을 `0.6 deg/s` 로 아주 천천히 밀어 `Kp·err` 토크를 키우다가, **위치가**
`move_thresh_deg`(0.25°) 이상 실제로 움직인 시점을 파단으로 본다. 방향당 3회 반복.

- **속도로 판정하면 안 된다** — 이 로봇은 정지 중에도 속도 노이즈가 ±15 deg/s 라
  임계 2 deg/s 를 상시 초과한다. 초기 구현이 그래서 τ_s(0.12) < τ_c(0.51) 라는
  물리적으로 불가능한 값을 냈다. 위치 노이즈는 ~0.01° 로 25배 여유가 있다.
- 기록하는 값은 검출 시점의 순간토크가 아니라 **정착 이후 누적 최대토크**다.
  풀리는 순간 가속하며 추종오차가 줄어 토크가 이미 떨어져 있기 때문 —
  정지마찰의 정의는 "파단 직전 버틴 최대토크" 다.

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

> **어느 시험의 값을 쓸 것인가** (★실기 실측으로 갱신 — 초기 권고를 뒤집었다)
> - **JDAMP → (D) 처프.** (B) 스윕은 최고 35 deg/s(=0.61 rad/s)까지밖에 못 돌려
>   점성 기여가 쿨롱마찰의 0.5% 라 신호에 잡히지 않는다. HL 은 0.0042(사실상 0),
>   HR 은 **−0.0415 음수**까지 나왔다(R²=0.9992 인데도). 코드가 이제 이를 감지해 무효화한다.
> - **I_total(→ROTOR_I) → (D) 처프.** 가속도 여기가 필요해 처프에서만 나온다.
> - **JFRIC → 용도별.** 저속 정지·유지는 (B) 0.50~0.52, 보행 등 동적은 (D) 0.38.
>   Stribeck 때문에 갈리는 것이지 오류가 아니다.
>
> ⚠ `--selftest` 의 "(B) 상대오차 0.0%" 는 **합성 데이터 한정**이다. 합성 신호에는
> Stribeck 이 없어 선형 f(v)=τ_c+b·v 가 정확히 맞을 뿐, 실기에는 해당하지 않는다.

## 식별 가능성 확인 (반드시 볼 것)

리포트의 다음 항목이 신뢰도를 결정한다.
- **파라미터별 ±표준편차** — 값과 같은 자릿수면 그 파라미터는 식별되지 않은 것이다.
  특히 `I_total ± ` 가 값의 50% 를 넘으면 처프 상한주파수/진폭을 올려 재측정.
- **조건수** — 수백을 넘으면 파라미터 간 상관이 커 분리 신뢰도가 떨어진다.
- **R²·잔차 RMS** — R² 가 0.8 미만이면 모델이 못 잡는 성분(백래시·시간지연·온도).
- **중력+bias 의 속도 무관성** — 속도에 따라 변하면 상쇄 전제가 깨진 것.

## ★두 트랙 — 축별 해석회귀 / PACE 전축 재현매칭 (2026-08-11 정리)

| | 축별 해석회귀 | **PACE 전축 동시** |
|---|---|---|
| 가진 | 한 축, 나머지는 지그 고정 | **전 축 동시**(비상관 처프) |
| 모델 | `τ = I_ii·q̈ + b·q̇ + τ_c·sgn + g` (**대각만**) | 시뮬레이터가 `M(q)·C·g` 를 전부 들고 있음 |
| 목적함수 | 토크 회귀 | **Σ(q_sim − q_real)²** — 궤적 재현 |
| 순환 문제 | ★있다. 드라이버 τ 가 `kp·err` 로 재구성됨 | **없다.** τ 를 아예 안 쓴다 |
| 파라미터 정밀도 | 축별로 **또렷함** | 스칼라 목적함수라 일부 조합이 sloppy |
| 조건 | 실제 보행과 다름(타축 정지) | **배포 조건과 같음** |

⇒ **상호 보완이다.** 축별 값을 CMA-ES 의 **초기값·탐색범위**로 넣고, PACE 로 마무리한다.
⚠PACE 는 **강체 부분이 맞다는 걸 전제**한다. MJCF 질량·관성이 틀리면 CMA-ES 가 그 오차를
  armature/damping/friction 으로 흡수해 "잘 맞는데 물리적으로 틀린" 값을 낸다.
  → 그래서 축별로 `I_link` 를 먼저 검증했다(foot 예측 대비 **−1.0%**, 2026-08-11).

```bash
# ① 지그를 **빼고** 전축 동시 처프 수집 (위치+게인 모드라 토크 자기제한)
python3 collect_multichirp.py --dry     # 하드웨어 미접촉 설계검사(상관·한계)
python3 collect_multichirp.py           # → results/pace_multichirp.npz
# ② MuJoCo 롤아웃 + CMA-ES
~/.venv-mujoco/bin/python pace_cmaes.py results/pace_multichirp.npz
```

★**mujoco 는 이제 로봇(Pi)에 있다** — `~/.venv-mujoco` (aarch64 휠, `cma` 포함).
  종전 README 의 "노트북에서 마무리" 는 **틀린 전제**였다. 설치를 시도해 본 적이 없었을 뿐이다.

### 자기충돌 포락선 (MJCF 꼭짓점 2^8 전수)
```
전축 동일 진폭   ±10° 안전 · ±15° 부터 두 발 충돌(HL_sphere↔HR_sphere −70mm)
hip 만 ±8° 제한  나머지 ±30° 까지 전부 안전
```
원인은 hip 이다 — 내전하면 발이 모인다. 그래서 `spec.pace_multi.amp_deg` 는 hip 만 5°다.

## 스펙 입력 (`spec.yaml`)

`TODO` 로 남은 값이 있으면 하니스가 실행 시 목록을 찍고, 그 값에 의존하는 환산은
"미확정" 으로 보고한다(없는 값을 추측해 그럴듯한 숫자를 내지 않는다).

| 항목 | 왜 필요한가 | 현재 |
|---|---|---|
| `units.torque_frame` | 보고 토크가 모터축인지 관절축인지. 관절환산이 N배 달라진다 | ✅ **joint** (36° 명령→출력축 36° 육안확인) |
| `joints[].I_link` | `ROTOR_I = (I_total−I_link)/N²` 분리에 필수 | ✅ hip=**0.0** (다리 미장착). 장착 시 MJCF 값으로 교체 |
| `joints[].kt_nm_per_a` | 전류-토크 교차검증(측정 tau 와 cur 이 일치하는지) | **TODO** |
| `joints[].gear` | 반사관성·관절환산 | 7 / 7 / 10.5 / 8.4 (문서값) |
| `joints[].q_min/max` | 시험 각도한계 | hip ±20° (checklist 확정값) |

**기어 구조**(사용자 확인 2026-08-05): 전 관절이 **동일 모터 + 7:1** 이고 관절별로
추가 감속단만 붙는다 — calf 10.5 = 7×1.5, foot **8.4** = 7×1.2. 총비 8.4 가 맞다
(`quad/PARAMS.md:91` 의 14.0 은 재기어 이전값이라 stale).
⇒ `ROTOR_I` 는 모터축 상수라 **전 관절 공통**이고, armature = `ROTOR_I·N²` 로 파생된다.

⚠ 관절한계는 spec.yaml 이 `emb/config/biped_emb.yaml` 보다 **8축 전부 더 좁다**.
hip ±20° 만 근거가 있고(checklist 2026-07-21 확정, config 는 아직 ±35),
나머지 6축은 미장착 상태의 임의 시험한계다. 장착 시 재확인할 것.

## 파일

```
pace/
├── spec.yaml                        하드웨어 스펙 + 안전한계 (사용자 입력)
├── hwio.py                          안전 SHM I/O (한계·stale·limp·인가램프)
├── actuator_test.py                 하니스 + --selftest
├── RESULTS.md                       ★실측 결과(HL_hip·HR_hip)
├── tests/act_measure_friction.py    (A)(B)(C) 마찰
├── tests/act_identify_pace.py       (D) PACE 처프 식별 + npz 내보내기
├── tests/act_measure_latency.py     (E) 왕복지연 · 로스트모션 · 2엔코더 백래시(대기)
└── templates/base.html              리포트 템플릿
```

⚠ `~/simulation` 은 clone/sync 가 **미추적 파일을 지운다**(2026-08-05 에 `emb/diag/`·
`emb/net/` 가 그렇게 삭제됐다). 로봇에서 만든 것은 반드시 커밋할 것.
빌드 산출물·플롯은 `emb/.gitignore` 로 제외하고, `results/*.npz`(원시 측정데이터)는
로봇 없이 재현 불가하므로 커밋한다.
