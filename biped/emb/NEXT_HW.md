# 다음 실기 세션 체크리스트 (작성 2026-08-06)

**전제**: 하드웨어가 **다리 전부 조립된 biped** 로 바뀌었다. 그 전 작업은 전부 "다리 미장착
(hip 2축 단품)" 상태에서 한 것이라, 여기 적힌 항목 상당수가 **전제가 깨져서** 다시 해야 하는 것들이다.

**순서가 있다.** 3(JOG 각축 검증)이 안 끝나면 5~7 은 의미가 없거나 위험하다.
숫자를 건너뛰지 말 것.

> ⚠⚠ **stand/walk 는 9 번을 끝내기 전까지 금지.**
> **실기배포는 C++ 기준**인데 **C++→SHM 구간이 아직 없다**(§9).
> (§8 Python 통일은 2026-08-06 완료 — sim 8/8 통과.)
> jog/home/hold 는 이와 무관하니 3~7 은 그대로 진행해도 된다.

---

## 0. 매 세션 시작 전 (고정 절차)

- [ ] 로봇 **거치**(크레인/스탠드). 다리가 달렸으므로 이전보다 훨씬 중요하다.
- [ ] 전원/기동 순서: **모터전원 ON → Emb 기동 → 5초 대기 → 명령**
      (`halGait` 초기화 게이트 = 100+4500 tick @1kHz. 이 전엔 명령이 무시된다)
- [ ] **모터 명령 writer 는 한 번에 하나만** — `app/biped_emb.py` · `RobotTestGait` · `mot_test` ·
      `pace/actuator_test.py` 중 택1. 겹치면 서로 명령을 덮어쓴다.
- [ ] 종료 순서: writer 종료 → Emb 종료 → 모터전원 OFF.
      ⚠ 명령을 "안 쓰는" 것은 정지가 아니다. Emb 는 마지막 명령을 1kHz 로 영원히 재전송한다.

```bash
# Emb 기동 (⚠ 직후 4.5s 동안 전 관절이 0°로 램프 — 다리 달렸으니 주변 확인)
cd ~/ZSource/RobotEmbedded/build && sudo ./src/RobotEmbedded 2>&1 \
  | grep --line-buffered -aE "EtherCAT|Slave|WKC|Fail" > /tmp/emb.log
```

---

## 1. 기본 생존 확인 (5분)

- [ ] `pgrep -x RobotEmbedded` — 기동 확인
- [ ] EtherCAT 유통: `cat /sys/class/net/eth0/statistics/tx_packets` 를 1초 간격 2회.
      **~1000/s 증가면 정상, 0 이면 EtherCAT 정지** → Emb 재기동(자가복구 불가)
- [ ] `cd emb/diag && ./stt_probe 8` — "값이 갱신됨" 이어야 함.
      "값이 전혀 안 변함" = OP 이탈. ⚠ **플래그는 신선도를 증명하지 못한다**(Emb 가 정지
      데이터를 신선한 것처럼 발행함) — 반드시 값 변화로 판정.
- [ ] **드라이버 파워단 생존**: Emb 기동 직후 4.5초 램프에서 관절이 실제로 움직였는가.
      안 움직였으면 파워단 래치오프 → **모터 전원 재투입**(Emb 재기동만으로는 안 풀림).

> 2026-08-05 에 EtherCAT 슬레이브 이탈이 하루 2회 발생했고 원인 미규명이다.
> 재발하면 `~/BIPED_EMB_HANDOFF.md` 의 "다음에 잡을 데이터" 항목을 수집할 것.

---

## 2. 배선 확인 → `installed_channels` 갱신

**기구 조립 ≠ 전기 배선.** 8축이 실제로 EtherCAT 에 물려 통신하는지 먼저 본다.

- [ ] `./stt_probe 8` 로 채널별 값이 **실제로 변하는지** 확인 (관절을 손으로 조금 돌려보면 확실)
- [ ] 통신되는 채널을 **두 파일 모두** 에 반영 (한쪽만 고치면 시험 하네스와 배포 앱이 갈린다)
  - [ ] `emb/config/biped_emb.yaml` → `meta.installed_channels`
  - [ ] `emb/pace/spec.yaml` → `meta.installed_channels`
- [ ] GUI 에서 미장착 축이 **어두운 LED**, 실장축만 초록인지 확인
      (`ok N/N` 의 분모가 실장축 수로 나와야 한다)

---

## 3. ★JOG 각축 검증 — `sign` · `offset_deg` · 한계  (가장 중요, 선행조건)

현재 8축 중 **hip 2축만 sign 실측 확정**이고, `offset_deg` 는 **전축 0.0 = 미확정**,
나머지 6축은 sign·게인이 전부 **미검증 외삽값**이다. 이게 안 끝나면 5~8 이 전부 무의미하다.

축 하나씩:

- [ ] **Off** 로 시작(limp) → 안전 자세 확인
- [ ] **JOG** 진입 (슬라이더가 현재 실측각으로 자동 정렬됨)
- [ ] 슬라이더를 **소량(+)** 이동하고 관찰
  - 반대로 움직이면 → `biped_emb.yaml` 의 `sign` 뒤집기 (±1)
  - 슬라이더 0 인데 관절이 0 이 아니면 → `offset_deg` 로 0점 보정
  - 물리 한계 전에 멈춰야 하면 → `min_deg`/`max_deg` (및 `jog.range_frac`) 조정
- [ ] 8축 전부 방향·0점·한계 확정 → config 저장 → **git commit**

⚠ 게인은 hip 만 실측(kp40/kd2)이고 나머지는 ×5 외삽이다. thigh/calf/foot 은 부하가 다르므로
`mot_test` 로 축별 재검증할 것. `Kd` 는 속도 노이즈를 그대로 토크로 증폭하니 함부로 올리지 말 것.

---

## 4. home 모드 실기 검증  (코드는 있는데 실기에서 한 번도 안 돌았다)

mock 검증만 끝난 상태다. GUI 는 Pi 에 `dearpygui` 가 없어 띄워보지도 못했다.

- [ ] **먼저 CLI 로** (dearpygui 없이 가능, 더 안전):
      `/tmp/biped_cmd.json` 에 `{"mode":"home", "seq":N}` 을 20Hz 로 써서 확인
      (⚠ `seq` 를 매번 증가시킬 것 — 워치독이 내용 변화로 생존을 판정한다)
- [ ] 확인할 것: 전 축 **동시 출발·동시 도착**, 계단 없이 부드러운 출발, 도착 후 홈 유지,
      `home_at_goal` 이 true 로 뜨는지
- [ ] `home.q_deg` 재설정 — **현재 전 축 0° 는 다리를 쭉 편 자세**다. 다리를 단 지금은
      실제 기립 자세(무릎 굽힘)로 잡아야 한다. 굽힘이 크면 `jog.range_frac`(현재 0.5)에
      걸려 클램프되고 기동 시 경고가 뜬다 → `range_frac` 부터 검토.
- [ ] (선택) GUI: `dearpygui` 설치. aarch64 휠 있음. PEP 668 → venv 또는 `--break-system-packages`

⚠ `home.q_deg` 의 "0" 은 **모터 엔코더 0점**이다 (jog/home 경로는 `offset_deg` 를 안 거친다).
3번 캘리브레이션이 끝나야 기계적 홈과 일치한다.

---

## 5. ★안전한계 재산정 (중력) — PACE 돌리기 전 필수

현재 값은 전부 **다리 없을 때** 정한 것이다. 다리 중력토크가 들어오면 시작하자마자
트립하거나, 피하려고 임계를 올리면 보호가 사라진다.

- [ ] 각 축의 **중력토크 실측**: JOG 로 자세를 잡고 `hold` 상태의 `fTorque` 를 읽는다
- [ ] `emb/pace/spec.yaml` `safety` 재산정
  - [ ] `tau_trip_nm` (현재 8.0) — 중력토크 + 여유
  - [ ] `err_max_deg` (현재 12.0) — kp40 에서 12° = 8.4 Nm 라 `tau_trip` 과 맞물린다
  - [ ] `vel_trip_dps` (현재 200.0) — 관성이 커졌으니 재검토
- [ ] `emb/config/biped_emb.yaml` `safety` 도 같은 근거로 갱신 (배포 앱용)

---

## 6. `hold_others` 실기 검증  (2026-08-06 추가, **실기 미검증**)

시험축 외 채널을 각자의 측정위치에 kp40/kd2 로 잡는 기능. 켜져 있다
(`spec.safety.hold_others: true`).

- [ ] 거치 상태에서, 사람이 보는 앞에서 첫 실행
- [ ] 확인: 시험축만 움직이고 나머지는 자세를 유지하는가
- [ ] `check_hold()` 가 도는지 — 홀드축을 손으로 밀면 `err_max` 초과 시 limp + 중단돼야 함

---

## 7. PACE 측정 (다리 장착 상태)

**축마다 목적이 다르다** (RESULTS.md §11-b). 예상값과 대조할 것:

| 축 | 링크비중 | 목적 | 예상 `I_total` |
|---|---|---|---|
| hip | 81.4% | **MJCF `I_link` 검증** (ROTOR_I=7.4e-4 대입) | 0.1951 |
| thigh | 76.7% | 같음 | 0.1557 |
| calf | 26.1% | 둘 다 어중간 | 0.1104 |
| foot | **3.7%** | **`ROTOR_I` 독립 추출** | 0.0542 |

- [ ] **시험 자세를 홈(q=0)으로 고정.** `I_link` 는 자세 의존이다 —
      hip 은 무릎 −40° 에서 +15% 변한다(0.1589 → 0.1822). 다른 자세로 잴 거면
      `~/.venvs/mj/bin/python emb/pace/extract_ilink.py` 로 그 자세 값을 다시 뽑을 것.
- [ ] `friction` — RESULTS.md §9 의 "다리 장착 상태 미측정" 항목. 베어링 하중이 늘어 마찰이
      커졌을 것이다. 양방향 상쇄 설계라 중력은 소거된다(개념상 다리 장착에 적합).
- [ ] `pace`(chirp) — 위 표의 `I_total` 대조. **예측과 크게 어긋나면 MJCF 관성을 의심**할 것
      (그게 이 측정의 주 목적이다. MJCF 는 MPC/WBIC 의 근거인데 실기 대조를 한 번도 안 했다).
- [ ] `frf` — spec.yaml 이 *"다리 장착 후에 측정할 것"* 이라고 명시한 항목. **지금이 적기.**
      단 `amp_nm_free`(0.45) 는 무부하 파단토크 기준이라 재산정 필요.

### ⚠ 순수토크 계열은 조건부 — `torque` / `backlash` / `frf`

`act_probe_torque_mode.py` 헤더에 이렇게 써 있다:
*"**다리가 없는 지금이 이 시험을 하기에 가장 안전한 시점이다.**"*

Kp=Kd=0 이라 토크가 자기제한되지 않는다. `torque_mode.tau_max_nm`(1.4) ·
`backlash.tau_max_nm`(0.45) 는 **무부하 파단토크 0.71 Nm 기준**이라, 다리 중력토크가
그보다 크면 시험축이 그냥 무너진다.

- [ ] 5번(중력토크 실측)이 끝난 뒤, 그 값으로 `tau_max` 재산정
- [ ] `hold_others` 가 확실히 동작하는지 6번에서 확인된 뒤에만
- [ ] `torque` / `backlash` 는 이미 다리 없이 측정 완료(§2-b, 백래시 HL 0.1133° / HR 0.0752°).
      **재측정 가치가 낮다면 건너뛰는 것도 정당한 선택이다.**

---

## 8. ★★Python 을 C++ 튜닝값에 맞춘다 (동기화 방향: **C++ → Python**)

**개발 파이프라인**: `Python(탐색·프로토타이핑) → C++(이식) → 실기배포(SHM)`
**단, 현재 튜닝값의 기준은 C++ 다.** 실측 물리 반영과 안정성 재스윕이 C++ 에서 이뤄졌고
(`cpp/STABILITY_MAP.md`), **실기배포도 C++ 기준**이다(§9).

> ⚠⚠ **이번 동기화 방향은 C++ → Python 이다.** 평소 흐름(Python→C++)과 반대이므로
> 헷갈리지 말 것. **C++ 값을 Python 으로 되돌리는 일은 절대 없다** — T_STEP 0.38 ·
> ROTOR_I 7.4e-4 는 낙상 스윕으로 얻은 값이고, Python 의 0.24 / 1e-4 는 그 스윕
> **이전** 값이다.

갈렸던 6개 (2026-08-06 **통일 완료**, §8-a):

| 파라미터 | **C++ = 기준** | Python (구값) | armature 영향 |
|---|---|---|---|
| `ROTOR_I` | **7.4e-4** | 1e-4 | hip 0.0363 vs **0.0049** |
| `JDAMP` / `JFRIC` | **0.09 / 0.38** | 0.1 / 0.5 | 작음 |
| `GEAR` foot | **8.4** | 8.0 | foot 0.0522 vs 0.0064 |
| `T_STEP` | **0.38** | 0.24 | ★결정적 |
| `K_RETURN` | **0.15** | 0.45 | |

> `TAU_PEAK` foot 은 **C++·Python 둘 다 96** 이라 갈린 항목이 아니다.
> 다만 그 96 자체가 감속비와 안 맞는다 → §8-d.

**왜 Python 도 맞춰야 하나** — **Python 에서 하는 탐색이 통째로 헛돌기 때문**이다.
armature 가 7.4배 과소한 모델 위에서 튜닝하면 그 결과는 C++/실기로 넘어가는 순간
성립하지 않는다. 2026-08-06 실측(§8-b)으로 증명됐다:

| 물리 | 보행 | 결과 |
|---|---|---|
| ROTOR_I 1e-4 (구) | T_STEP 0.24 (구) | 15s 무낙상 tilt 8.8° — **"잘 되는 것처럼 보인다"** |
| ROTOR_I 7.4e-4 (실측) | T_STEP 0.24 (구) | **2.12s 낙상** tilt 66° |
| ROTOR_I 7.4e-4 (실측) | T_STEP 0.38 (C++) | 15s 무낙상 tilt 5.4° ✅ |

즉 구 Python 에서 "안정적"이라고 결론 낸 것들은 **실기에서 성립한다는 보장이 전혀 없다.**

> ⚠ **오해 정정 (2026-08-06)**: 이 문서 초판은 "물리 상수가 갈려서 파리티 검증이
> 불능이다" 를 주된 이유로 적었는데 **틀렸다.** `dump_biped_*.py` 는 `M`·`h`·`Jc`·
> `tau_peak`·`gains` 를 **Python 이 계산해서 C++ 에 먹이는** 구조라, C++ 는 Python 의
> M(=armature 가 이미 반영된)을 그대로 쓴다. 그래서 파리티 도구는 **알고리즘 일치만**
> 검증하고 물리 상수 차이는 원리적으로 못 잡는다.
> 실측으로 확인: 구 상수(1e-4/0.24)로 덤프해도 파리티 통과(오차 5.8e-12).
> ⇒ **파리티 OK 는 "상수가 같다" 를 뜻하지 않는다.** 상수 일치는 별도로 눈으로 볼 것.

> ℹ️ **오해 정정**: `app/biped_emb.py` 의 stand/walk 는 Python `model_ctrl` 에 물려 있지만,
> Pi 에 `mujoco`·`qpsolvers` 가 **미설치**라 import 가 실패하고 **hold 폴백**된다
> (`biped_emb.py:287`). 즉 지금 당장 stale 한 값으로 실기가 도는 상태는 아니다.
> 다만 **설계된 안전장치가 아니라 우연**이다 — Pi 에 그 둘을 설치하는 순간 열린다.
> (§9 에서 이 경로 자체를 정리한다.)

### 재튜닝 방향은 이미 실증돼 있다 (`biped_control.hpp:22-29`)

```
ROTOR_I 1e-4/2e-4/4e-4/5e-4 = 15s 무낙상 · 6e-4 = 9.4s · 7.4e-4 = 2.18s 낙상
7.4e-4 + T_STEP 0.32 = 15s 무낙상 tilt 2.7° (전 설정 중 최량) · 0.40/0.50 = 낙상
⚠ 스윙게인을 올리는 것은 역효과 (SW_KP 800→1600/3200/5920 = 1.16/1.10/0.65s 낙상)
  — 대역 부족이 아니라 **토크 포화**이기 때문. 필요가속도 ∝ 1/T² 이라 스텝을 늦추는 게 해법.
```
이후 leg-odom 야코비안 편향을 제거하며 T_STEP 0.32→**0.38**, K_RETURN 0.45→**0.15** 로 재스윕
(상세: `cpp/STABILITY_MAP.md`).

### 8-a. ✅ 통일 완료 (2026-08-06)

- [x] `biped_wbic.py` — `ROTOR_I` 1e-4 → **7.4e-4** · `JDAMP/JFRIC` 0.1/0.5 → **0.09/0.38**
      · `GEAR` foot 8.0 → **8.4**
- [x] `biped_step.py` — `T_STEP` 0.24 → **0.38** · `K_RETURN` 0.45 → **0.15**
- [x] 전수 대조 결과 **나머지 22개 상수는 원래 일치**했다 (DS_FRAC · STEP_H · K_CAP ·
      CAP_CLAMP · SW_KP/KD · K_RET_LAT · K_LAT · SPREAD · GAP_MIN/MAX · SS_* · TRIG_Y ·
      GVEC · STANCE_KD · W_ORI/POST/ANKLE · MU_EFF · LAMZ_MIN · MPC_N/DT · head_lead · Q_HOME).
      갈린 건 위 6개뿐이었다.

### 8-b. ✅ 검증 결과 (2026-08-06, MuJoCo 3.9.0)

| 검사 | 결과 |
|---|---|
| C++ 기준 재현 (`biped_sim` vx0.15 15s) | 15.00s 무낙상 · x=2.185 · tilt 2.8° |
| **Python 통일 후** (동일 조건) | **15.00s 무낙상 · x=2.197 · tilt 3.5°** (전진거리 0.5% 차) |
| **속도대역 스윕** (12s × 8조건) | **8/8 통과** — 정지 · 전진 0.05/0.10/0.15/0.20 · 후진 −0.10 · 측방 0.05 · 선회 0.2 (tilt 전부 5~6°) |
| **반증**: 실측물리 + 구 T_STEP 0.24 | **2.12s 낙상** — C++ 기록 2.18s 를 3% 오차로 재현 ✅ |
| WBIC 파리티 | 오차 2.5e-11 ✅ |
| MPC 파리티 | 오차 3.2e-12 ✅ |

⇒ **실측 하드웨어 값에서 제어가 정상 동작한다.** C++ 이 경고했던
*"vx=0.15 단일 조건으로 잡은 값"* 리스크도 속도대역 스윕으로 해소됐다(단 sim 기준).

재현 (Pi):
```bash
~/.venvs/mj/bin/python -c "import mujoco; print(mujoco.mj_versionString())"   # 3.9.0 이어야 함
cd ~/simulation/biped && VX=0.15 T=15 ~/.venvs/mj/bin/python biped_mpc_wbic.py
cd cpp && LD_LIBRARY_PATH=$HOME/mujoco/lib ./build/biped_sim ../biped_from_quad.mjcf 0.15 15
```
⚠ **MuJoCo 3.9.0 을 쓸 것.** 3.11 은 `d.qM` → `d.M` 으로 API 가 바뀌어 Python 컨트롤러가
`AttributeError` 로 죽는다(`mj_fullM` 시그니처도 바뀜).

### 8-c. 남은 일

- [ ] `quad/PARAMS.md:147` 의 표(`Irot 1e-4` / `jdmp·jfrc 0.1/0.5`)가 여전히 구값이다 → 갱신.
- [ ] ★foot `tau_peak` 불일치 — 아래 8-d.

### 8-d. ⚠ 미해결: foot `tau_peak` 이 감속비와 안 맞는다

`tau_peak ÷ gear` 는 모터 피크토크(12.0 Nm)여야 하는데 foot 만 어긋난다:

| 축 | gear | tau_peak | ÷gear |
|---|---|---|---|
| hip · thigh | 7.0 | 84 | 12.00 ✅ |
| calf | 10.5 | 126 | 12.00 ✅ |
| **foot** | **8.4** | **96** | **11.43** ❌ (12×8.4 = **100.8** 이어야) |

`GEAR` foot 을 8→8.4 로 고칠 때 `tau_peak` 이 따라가지 않았다. 네 군데가 전부 다르다:

| 위치 | foot 값 | 유래 |
|---|---|---|
| `cpp/biped_control.hpp:46` `tau_peak8` | 96 | 12 × 8 (재기어 이전) |
| `biped_wbic.py` `TAU_PEAK` | 96 | 위와 동일 (통일 유지) |
| `pace/spec.yaml` | **100.8** | 12 × 8.4 — **물리적으로 맞는 값** |
| MJCF `actuatorfrcrange` | **168** | 12 × 14 — **최초 14:1 시절 값** |

- [ ] 실물 감속비 확정 후 통일. **96 은 4.8% 보수적(안전 방향)** 이라 급하진 않다.
- [ ] ⚠ 96 → 100.8 은 **토크 포화 한계를 바꾼다.** T_STEP 튜닝이 바로 그 포화에서 나온
      값이므로, 고치면 안정성 스윕을 **다시 돌려야 한다.** 통일과 같은 커밋에서 하지 말 것.

---

## 9. ★★실기배포는 C++ 기준 — 그런데 SHM 구간이 아직 없다 (핸드오프 미완료 #4)

**실기에 나가는 것은 C++ 다.** Python 은 탐색·프로토타이핑 전용이고 실기에 붙지 않는다.
그런데 **`cpp/src` 에 SHM 배선이 한 줄도 없다.**
(`grep -rn "SHM\|bridge_\|RobotSharedMem" cpp/src cpp/CMakeLists.txt` → **0 건**)

현재 C++ 타깃은 전부 시뮬/검증용이다:

| 타깃 | 용도 | 상태 |
|---|---|---|
| `biped_sim` | MJCF 로드 → BipedControl → `mj_step` 헤드리스 | ✅ 로봇에서 빌드·실행 OK (vx0.15) |
| `biped_view` | 뷰어 | ✅ |
| `biped_mpc_parity` / `biped_wbic_parity` | Python 대조 | ✅ (단 §8 때문에 현재 불일치) |
| **실기 배포 타깃** | — | ❌ **없음** |

`deploy_loop.hpp` 는 이름과 달리 **sim 전용**이다 — 지연·추정·보상을 모사하지만 물리는
`mj_step`(호출자)이다(`deploy_loop.hpp:4,103`).

### 할 일

- [ ] `mj_step` 자리에 **실모터 read/write** 를 넣는 HardwareInterface 작성.
      기존 `hal/shm_bridge.cpp`(C ABI)를 그대로 링크하면 된다 —
      `bridge_init/read/write_mit/enable` 이 이미 있고 Python 쪽에서 검증된 경로다.
- [ ] **변환 계약** (핸드오프 #4 명시):
      SHM(deg) ↔ 컨트롤러(rad) · **Kp/Kd 는 Nm/rad 그대로 전달** · `tau_ff` 는 `fTorque`
- [ ] 안전장치를 Python 앱(`app/biped_emb.py`)에서 이식 — 이미 실측으로 다듬어진 것들이다:
      워치독 · tilt/토크/속도 E-stop(래치 포함) · **종료 시 limp 반복기록**
      (⚠ `bridge_enable(0)` 만으로는 정지가 아니다. Kp=Kd=0 을 실제로 써야 한다)
- [ ] 실기 배포 타깃을 `CMakeLists.txt` 에 추가
- [ ] **`app/biped_emb.py` 의 Python stand/walk 분기 제거** — 배포가 C++ 기준으로 확정된
      이상 이 분기는 실기에서 돌아선 안 되는 경로다. 지금은 Pi 에 mujoco·qpsolvers 가
      없어서 import 실패로 hold 폴백되는데(`biped_emb.py:287`), 그건 **설계가 아니라 우연**이다.
      코드에서 명시적으로 막을 것.
      → 정리 후 Python 앱의 역할은 **off/jog/home/hold(각축 검증·복귀)** 로 한정된다.

### 역할 정리 (혼동 방지)

| | 역할 | 실기에 나가나 |
|---|---|---|
| Python `biped_*.py` | 알고리즘 탐색·프로토타이핑 (sim) | ❌ |
| Python `emb/app/biped_emb.py` | **off/jog/home/hold** — 각축 검증·홈복귀 | ✅ (모델기반 제외) |
| C++ `cpp/src` | 튜닝 기준 + **stand/walk 실기배포** | ✅ (§9 포팅 후) |

### 해소된 항목

- ~~foot 감속비 불일치~~ → **8.4 로 확정**. `biped_control.hpp:60` 에
  *"GEAR foot 8 → 8.4 (총 감속비 8.4 = 7×1.2 추가단, 사용자 확인 2026-08-05)"*.
  `pace/spec.yaml`(8.4)도 일치. **`biped_wbic.py` 의 8.0 과 `quad/PARAMS.md` 의 14.0 이 stale.**

### 남은 확인 (5초)

- [ ] **calf/foot 모터 라벨 육안 확인** — "8축 전부 RO100 동일 모터" 가설의 마지막 고리.
      `tau_peak ÷ gear` 가 8축 모두 12.0 Nm 로 일치하지만, 그 수치 자체가 `12 × N` 으로
      계산된 값일 수 있어 순환일 여지가 있다. 라벨 한 번 보면 끝난다.

---

## 참고 — 상태 요약 (2026-08-06 기준)

| 항목 | 상태 |
|---|---|
| 액추에이터 식별(다리 미장착) | ✅ 완료 — ROTOR_I 7.4e-4 · JFRIC · JDAMP · 백래시 · FRF · 데이터시트 대조 |
| `I_link` | ✅ 8축 추출 완료(`extract_ilink.py`), spec.yaml 반영 |
| spec.yaml TODO | ✅ 0 개 |
| home 모드 | 코드 완료, **실기 미검증** |
| `hold_others` | 코드 완료, **실기 미검증** |
| `installed_channels` | 코드 완료, 값은 `[0,4]` (배선 확인 후 갱신) |
| sign / offset | hip 2축 sign 만 확정. offset 전축 미확정 |
| 나머지 6축 | 기구 조립됨, 배선·검증 미확인 |
| C++ 컨트롤러 (sim) | ✅ **튜닝 기준** — 실측 물리 + 재스윕 완료 (ROTOR_I 7.4e-4 · T_STEP 0.38) |
| Python 컨트롤러 (탐색용) | ✅ **C++ 로 통일 완료**(2026-08-06) — 속도대역 8/8 · 파리티 OK (§8) |
| foot `tau_peak` | ⚠ 96 vs 100.8 vs 168 세 값 공존, 감속비와 불일치 (§8-d) |
| **C++ → SHM 실기배포** | ❌ **미착수** — `cpp/src` 에 SHM 배선 0 건. 핸드오프 #4 (§9) |

관련 문서: [pace/RESULTS.md](pace/RESULTS.md) (§10 데이터시트, §11 다리 조립 후) ·
[README.md](README.md) (각축 검증 절차) · `~/BIPED_EMB_HANDOFF.md` (Emb 결함·운용절차, git 밖)
