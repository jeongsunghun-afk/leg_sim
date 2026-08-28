# biped/emb — 실기(Emb) 배포 인터페이스

biped **모델기반 제어기(MPC+WBIC)** 를 RGA `RobotSharedMem`(Gait) 실모터에 연결하는 배포 프로젝트.
`quad_ctrl` 의 HAL 경계 원칙(State가 유일 입력·HAL이 유일 부작용·config 파라미터화)을 따른다.

> **1단계 목표(현재): GUI 명령 → 각축이 잘 움직이는지 인터페이스 검증(per-axis JOG).**
> 모델기반 Stand/Walk 는 배선 완료(`control/model_ctrl.py`) — jog 로 부호·오프셋·한계 확정 후 실행.
> 언어=Python 우선(검증된 컨트롤러·루프·GUI 재사용). RT 부족 시 C++ 이관(`RobotTestGait` 골격 참조).

## 입출력 인자 — **md80 ↔ MCU ↔ Emb ↔ 우리** 파이프라인

각 경계에서 **무엇이 어떤 단위·좌표로 오가는지**가 이 프로젝트가 반복해서 걸린 지점이다.
같은 "각도" 가 단계마다 뜻이 다르다.

### 층위와 경계

```
 우리 (Python/C++)        모델각 q_joint  [deg]   ← MJCF 관절각
    │  joint_map.q_joint_to_ch          ★커플링 되먹임 + 부호·감속비 + offset
    ▼
 채널각 q_ch [deg]  ─ SHM(MotGeneral_t, **float16**) ─▶  RobotEmbedded (1 kHz 폴링)
                                                          │  EtherCAT  (Pi ↔ transfer 보드)
                                                          ▼
                                                     transfer 보드 (EtherCAT 슬레이브)
                                                          │  SPI       (transfer ↔ MCU)
                                                          ▼
                                                        MCU  ─ FDCAN ─▶  **md80 드라이버**
                                                                          (MIT 임피던스 모드)
 ◀───────────────  역경로로 같은 필드가 돌아온다  ───────────────
```

왕복지연 **8.39 ± 0.79 ms** (직접 실측 — `act_measure_latency.py`).

### 쓰기 — `bridge_write_mit(q_des_deg, dq_des_dps, tau_ff_nm, kp, kd, n)`

채널마다 `MotGeneral_t` 하나를 채워 `RobotMemGait_SetMotorCommand16` 으로 넣는다.

| SHM 필드 | 넣는 값 | 단위 | 비고 |
|---|---|---|---|
| `ucDevID` | 채널 index | — | 0~9 |
| `ucMode` | **1** | — | MIT/임피던스 고정 |
| `fPosition` | `q_des` | **채널 deg** | `enable=0` 이면 마지막 값 유지 |
| `fVelocity` | `dq_des` | 채널 dps | 안 주면 0 |
| `fTorque` | `tau_ff` | **Nm(채널)** | 감속기 **전** |
| `fGainKp` / `fGainKd` | 게인 | 채널 | `enable=0` 이면 **0 = limp** |
| `fGainKi` | 0 | — | 미사용 |

⚠**전부 `float16` 이다.** 분해능이 값 크기에 비례한다:

| 값 | 격자 |
|---|---|
| 각도 100° | **0.0625°** |
| 각도 180° | 0.125° |
| 토크 20 Nm | 0.0156 Nm |
| 게인 100 | 0.0625 |

⇒ 발목 마찰 데드밴드가 0.85° 인 것에 비하면 각도 격자는 문제가 아니다. 다만
**0.0625° 미만의 명령 변화는 전달되지 않는다** — 미세 보정을 설계할 때 하한이다.

### 읽기 — `bridge_read(q_deg, dq_dps, tau_nm, cur_a, rpy, acc, gyro, conn, stt)`

| | 단위 | 비고 |
|---|---|---|
| `q` / `dq` | **채널 deg / dps** | 드라이버가 보고하는 값 |
| `tau` | **Nm(채널)** | 감속기 전 |
| `cur` | A | |
| `rpy` / `acc` / `gyro` | deg / — / — | ⚠**전부 0 이다** — `IMU_RECOVERY.md` 참조(미해결) |
| `conn` / `stt` | int | ⚠ 래치오프를 **안 알려준다**(정상값 그대로) |

### ★각도 좌표가 셋이다 — 헷갈리면 반드시 틀린다

| 좌표 | 정의 | 어디서 쓰나 |
|---|---|---|
| **모델각** `q_joint` | MJCF 관절각 | 제어기·시뮬 |
| **raw각** `q_raw` | `q_foot + coef·q_calf` (발목만) | **엔코더가 실제로 재는 값** |
| **채널각** `q_ch` | `sign·gear_k·q_raw + offset` | SHM·드라이버 |

```
쓰기:  q_joint ──커플링 되먹임──▶ q_raw ──sign·gear_k, offset──▶ q_ch ──▶ SHM
읽기:  SHM ──▶ q_ch ──역변환──▶ q_raw ──커플링 풀기──▶ q_joint
```

### ★게인도 좌표가 둘이다 — **`k²` 만큼 다르다**

각도가 들어갈 때 한 번, 토크가 나올 때 한 번 — **두 번** 변환된다:

```
Δq_ch  = s·Δq_raw              (s = sign·gear_k.  ★**raw** 다 — 모델각이 아니다)
τ_ch   = kp_ch · Δq_ch
τ_raw  = s·τ_ch                (일률보존: τ_raw·q̇_raw = τ_ch·q̇_ch)

⇒ τ_raw = kp_ch · s² · Δq_raw = **kp_ch · gear_k² · Δq_raw**   (부호는 제곱에서 상쇄)
```

**`kp_raw = kp_ch · gear_k²`** · `kd` 도 같다. 단위는 **둘 다 Nm/rad**.

⚠**2026-08-21 정정.** 이 자리에 `Δq_ch = s·Δq_joint` 라 적혀 있었고 결과를 `kp_joint` 라
불렀다. 바로 위 좌표표(`q_ch = sign·gear_k·q_raw + offset`)와 **모순**이다 — 채널각은
raw각의 함수지 모델각의 함수가 아니다. 같은 오류가 `biped_emb.yaml`·`RL_INTERFACE.md` 에
복제돼 있었다. hip·thigh·calf 는 raw==모델각이라 값이 안 갈려 오래 안 보였고 **발목에서만**
갈린다. (`pace_cmaes.py` 는 이미 옳게 적어 두고 있었다.)

★**모델각(관절) 게인은 스칼라로 존재하지 않는다.** 발목이 calf 에 물려 있어 모델각 좌표
강성은 **행렬**이 된다:

```
K_joint = Aᵀ·diag(kp_raw)·A            A = [[1,0],[1,1]]  (calf, foot)

K_raw   = diag(180.0, 43.2)                  ← 축별 스칼라로 쓸 수 있다
K_joint = [[223.2, 43.2], [43.2, 43.2]]      ← **비대각이 있다**
```

무릎만 1° 어긋나고 발목이 0° 여도 **발목 관절토크가 0.754 Nm** 나온다. 반작용이 되돌아와
calf 모델각 강성은 180 → **223.2 (+24%)** 가 된다. ⇒ 축별로 숫자 하나를 적는 형식 자체가
**raw 좌표에서만 성립한다.**

### ★게인 이름 규칙 (2026-08-21 확정)

| 이름 | 좌표 | hip / thigh / calf / foot | 어디서 쓰나 |
|---|---|---|---|
| **`kp_ch`** | 채널각 | 100 / 50 / **80** / **30** | config·SHM·드라이버 — **PD 가 실제로 닫히는 곳** |
| **`kp_raw`** | raw각 `= kp_ch·gear_k²` | 100 / 50 / **180** / **43.2** | 상태 JSON·모니터·트립각·MuJoCo 롤아웃·npz |
| **`K_joint`** | 모델각 — **행렬** | calf 대각 223.2 · 비대각 43.2 | 이론 논의용. **스칼라로 쓰지 말 것** |
| ~~`kp_joint`~~ | — | **폐기** | 스칼라 관절게인은 존재하지 않는다 |
| ~~`kp_leg`~~ | — | **폐기** | 좌표를 안 말한다 — 아래처럼 세 뜻으로 쓰였다 |

⚠`kp_leg` 가 폐기된 이유 — **한 이름이 세 가지**였다:

| 옛 이름 | 실체 | 새 이름 |
|---|---|---|
| `joint_map.self.kp_leg` | 채널좌표 게인, 다리순 8 배열 | **`kp_ch_leg`** |
| `hw_interface.write_torque(kp_leg=)` | 게인이 아니라 **배율**(0 = 순수토크) | **`kp_scale`** |
| 상태 JSON 키 `kp_leg` | C++ 은 ×k², Python 은 채널 그대로 — **같은 키가 다른 좌표** | **`kp_raw`** |

★**접미사 규칙**: `_ch`/`_raw` 는 **좌표**, `_leg` 는 **8축 다리순 배열**(좌표 아님).
게인 이름에는 좌표를 **반드시** 단다.

⚠`hip`·`thigh` 는 `gear_k=1` 이라 세 좌표가 같아서 **헷갈리지 않는다.** calf·foot 만 갈린다 —
그래서 실수해도 두 축에서만 티가 난다.

### 트립각은 **1승**이다

트립은 **채널토크**로 걸린다(`biped_deploy` 가 `hs.tau_nm` 을 그대로 비교).
⇒ 트립 시점의 raw 토크는 `tau_trip·gear_k`, raw 강성은 `kp_ch·gear_k²` 이므로

```
트립각 = (tau_trip·gear_k) / (kp_ch·gear_k²·배율) = **tau_trip / (kp_ch·gear_k·배율)**

가장 예민한 calf: 15 / (80·1.5) = 0.125 rad = **7.16°/배율**   (×10 에서 0.72°)
```

⚠2승으로 쓰면 calf 가 4.77°/배율로 나온다. `biped_deploy.cpp` 가 2026-08-21 까지 그랬다.
`POS_KP_SCALE_MAX` 를 5→10 으로 올린 근거가 이 7.16° 이므로 **안전 판단에 직결된다.**

★확인 — 실측 예가 단위를 못박는다: *"hip 오차 4.03° → 명령 7.03 Nm"*
`100 × 4.03 × π/180 = 7.03` ✓. SHM 으로는 각도를 **도**로 보내지만 드라이버가
내부에서 라디안으로 변환한다.

⚠**`gear_k` 는 감속비가 아니다.** 드라이버가 전 축을 7:1 로 착각해서 생긴 **배율**이다
(`calf` 1.5 = 10.5/7 · `foot` 1.2 = 8.4/7). 실제 감속비는 `N = 7·gear_k`.
⚠**`coef` 는 그것과 무관한 링키지 커플링**이다(값 1.0, 경사계로 실측 확정).
원인이 다른 둘이 같은 두 축(calf·foot)에 겹쳐 있어 특히 헷갈린다.
⚠쓰기 마지막에 채널각을 **±180 으로 포화**시킨다 — Emb 가 초과분을 클램프가 아니라
**래핑**하므로(`halGait.cpp:666-671`) 181° 가 −179° 로 뒤집혀 반대편으로 날아간다.

상세는 `emb/RL_INTERFACE.md`(각도규약 전문) · `emb/pace/RESULTS.md` §1-b(커플링과 손실 좌표).

### 계층별 주기

| 계층 | 주기 | 무엇을 하나 | 어디 코드 |
|---|---|---|---|
| **md80 드라이버 (MCU)** | 미확인 | MIT 임피던스 `τ = kp·err + kd·derr + τ_ff` (`ucMode=1`) | 벤더 펌웨어 |
| **RobotEmbedded** | **1 kHz** | EtherCAT 사이클 + gait HAL(기동 램프·SHM 통과) | 벤더 (`~/ZSource/RobotEmbedded`) |
| 우리 제어기 (Python) | **500 Hz** | jog 램프·안전·좌표변환 | `emb/app/biped_emb.py` |
| 우리 제어기 (C++) | **500 Hz** | 같은 역할, 실기 writer | `cpp/src/biped_deploy.cpp` |
| WBIC | **500 Hz** | 전신 QP → 관절토크 | `biped_mpc_wbic.py` |
| MPC | **50 Hz** | 보행 계획 (호라이즌 `N=14 × DT=0.02` = **0.28 s**) | 같은 파일 (`MPC_DECIM=10`) |
| 시뮬 적분 | 500 Hz | MuJoCo `timestep=0.002` | MJCF |
| 식별 수집 | **400 Hz** | 다축 처프 로깅 | `emb/pace/collect_multichirp.py --rate` |

`ctrl_hz` 는 `emb/config/biped_emb.yaml` 한 곳에서 온다 — Python·C++ 이 같은 값을 읽는다.

⚠**어느 것도 실시간이 아니다** (2026-08-14 실측):

| | |
|---|---|
| `RobotEmbedded` 스케줄링 | `SCHED_OTHER` · 우선순위 0 · affinity 0-3 |
| 사용자 RT 한도 (`ulimit -r`) | **0** — `chrt` 를 쓸 권한이 없다 |

1 kHz EtherCAT 루프조차 일반 우선순위다. 커널이 아무 때나 **20~45 ms 선점**할 수 있고,
그게 실제로 관측된다:

| 조건 | 20 ms 넘는 멈춤 | 누적 지연 |
|---|---|---|
| 400 Hz · 경합 있음(`biped_sim` 100%) | 10회 / 12000틱 | 0 |
| 400 Hz · `biped_monitor`(13.8%) 정리 후 | **2회** | +0.01 s |

★**주기 예산은 남는다** — 주기 중앙값이 공칭 그대로(2.50 ms)이고 누적 지연이 0 이다.
문제는 드문 **큰 멈춤**이지 rate 가 아니다. `rate` 를 낮춰도 같은 선점이 같은 크기로 난다.
⇒ 개선하려면 **경합 정리**(무료)가 먼저고, 근본은 RT 권한이다.
  ⚠순서가 중요하다 — 우리 스크립트만 RT 로 올리면 EtherCAT 루프를 선점해 **더 나빠진다.**
    `RobotEmbedded` 를 먼저 높게, 그 아래에 우리를 둔다.

왕복지연 **8.39 ± 0.79 ms** — 이 값은 `pace_cmaes` 가 모델의 `T_d` 로 쓰고,
C++ 배포는 `LAT_COMP_MS` 로 지연보상에 쓴다.

---

## 데이터 흐름
```
RobotSharedMem(모터·IMU)                          GUI(teleop_emb)
        │ read                                         │ /tmp/biped_cmd.json
        ▼                                              ▼
  ShmBackend ──► HwInterface(매핑·IMU변환) ──► [ModeFSM 디스패치] ──► write ──► 모터
   (deg 채널)      (rad·quat, 8-DOF)          off/jog/hold/stand/walk        (MIT)
                                                       │ /tmp/biped_state.json
                                                       ▼ (실측각·tilt·hz)
```

## 디렉토리 (quad_ctrl 대응)
```
emb/
├── config/biped_emb.yaml     파라미터 단일점: 관절맵(채널↔biped·부호·오프셋·한계)·게인·jog·안전
├── hal/                      SHM 경계(채널·deg)
│   ├── shm_bridge.cpp/.h     RobotSharedMem 얇은 C ABI → libbipedshm.so (C 의존 격리)
│   ├── build_bridge.sh       ★Pi에서 브리지 컴파일
│   ├── backend.py            Backend ABC + RawState
│   ├── shm_backend.py        실HW: ctypes로 .so 로드
│   └── mock_backend.py       데스크톱 데모(SHM 없이 명령 에코)
├── interface/
│   ├── joint_map.py          biped 8-DOF(rad) ↔ Gait 채널(deg): 부호·오프셋·한계
│   └── hw_interface.py       매핑+IMU변환 묶음(jog·hold·torque write, ctrl_state)
├── control/
│   ├── jog.py                per-axis 저속 위치 램프 ← ★각축 검증
│   ├── home.py               홈 자세로 S-curve 복귀(전축 동시출발·동시도착)
│   ├── mode_fsm.py           off/jog/home/hold/stand/walk 상태기계
│   └── model_ctrl.py         (미사용) 모델기반 래퍼 — ★배포는 C++ cpp/build/biped_deploy
├── app/biped_emb.py          메인 RT 루프 + 상태발행 (off·jog·home·hold 만)
└── run_emb.sh                런처
```

★GUI 는 이 트리에 없다. `../teleop_gui_biped.py` 하나로 통합돼 있다
  (커밋 9454912 에서 각축 JOG 패널을 흡수하며 `gui/teleop_emb.py` 제거,
   7953c5c 에서 `quad/` → `biped/` 로 이동). 실행: `../run_gui_only.sh`
```

## 실행

### 데스크톱 데모(SHM 없이 jog·GUI 검증)
```bash
MOCK=1 ./run_emb.sh
# GUI에서 JOG 버튼 → 슬라이더로 축별 이동 → '실측'이 명령을 따라오는지 확인(mock=에코)
# 부분연결/에러 시뮬 = LED 3색 확인:
MOCK=1 MOCK_CONNECTED="1,2,3" MOCK_FAULT="3" ./run_emb.sh   # 1,2=초록 3=노랑 나머지=회색
```

### 축별 상태 LED — 임베디드 보고 모니터 (제어 아님)
- GUI JOG 패널 각 행: **[●LED] 관절명 [슬라이더] 실측°**.
- **LED 3색**(임베디드가 SHM에 쓴 값을 그대로 반영):
  - 🟢 초록 = **정상**(통신 O + `ucStatus`=0)
  - 🟡 노랑 = **에러**(통신 O + `ucStatus`≠0 — 과열/과전류/엔코더 등 펌웨어 보고)
  - ⚫ 회색 = **두절**(무통신, 0.5s 미수신)
- 상단에 **`정상 N / 에러 M / 두절 K / 8`** = 몇 축이 살아있고/에러/죽었는지 한눈에.
- **모터 on/off는 없음**: 이 계층에서 개별 모터를 켜고 끌 수 없음(임베디드 소관). **2개만 연결해도 동작** — 미배선 모터는 통신이 없어 명령이 무효화될 뿐, 루프는 정상.
- 판정 원천: 브리지가 채널별 `MotorStatus16` 수신(통신) + `MotGeneral_t.ucStatus`(에러). ★`ucStatus` 비트 의미는 모터/펌웨어 정의 → 현재 `≠0=에러` 가정, 실기 스펙 확인 후 세분화. ★임베디드가 통신손실 시 갱신을 멈추는지도 확인 필요(안 멈추면 두절 판정 불가).

### 실기(Pi/Emb)
```bash
# ① 브리지 빌드(RobotSharedMem.h·RobotTestGait/inc·libRobotSharedMem 필요)
bash hal/build_bridge.sh
# ② 실행 — ShmBackend 자동 사용(.so 없으면 mock 폴백)
./run_emb.sh
```

## 각축 검증 절차 (부호·오프셋·한계 캘리브레이션)
1. **Off** 로 시작(limp) → 로봇을 안전 자세로.
2. **JOG** 진입 → 슬라이더가 현재 실측각으로 정렬됨.
3. 한 관절씩 슬라이더 소량(+) 이동:
   - 로봇이 **반대로** 움직이면 config `sign` 을 뒤집는다(±1).
   - 슬라이더 0인데 관절이 0이 아니면 `offset_deg` 로 0점 보정.
   - 물리 한계 전에 멈추려면 `min_deg/max_deg`(및 `jog.range_frac`) 조정.
4. 8축 모두 방향·0점·한계 확정 → config 저장.
5. 그 다음에야 **Stand**(모델기반 서기) — 그 전엔 매핑 미확정이라 위험.

## 관절 매핑 (기본)
| biped(제어기) | Gait 채널 | 축 |
|---|---|---|
| HL_hip / HR_hip | 0 / 4 | roll |
| HL_thigh / HR_thigh | 1 / 5 | pitch |
| HL_calf / HR_calf | 2 / 6 | knee pitch |
| HL_foot / HR_foot | 3 / 7 | ankle pitch |
| (허리 WaR/WaP) | 8 / 9 | biped 모델 외 → 고정 홀드 |

## 남은 실기 확정 항목 (코드에 TODO 표기)
- **IMU 자이로 인덱스**(`shm_bridge.cpp` `IDX_OF_IMU_AVEL`) — 추정기용 각속도. jog 불필요, stand 전 확정.
- **발 접촉 판정**(`hw_interface._estimate_contact`) — 힘센서 부재 시 발목토크 임계. stand 전 확정.
- **sign·offset·limit** — 위 각축 검증으로 실측 확정.
- **ctrl_hz vs 모델 timestep** — 모델기반은 모델 dt 로 페이싱(현재 config ctrl_hz). walk 고속=RT 부족 시 C++ 이관.

## 의존성
- Python: `numpy`, `pyyaml`, `dearpygui`(GUI), `mujoco`+`qpsolvers`(stand/walk만).
- 브리지(Pi): `RobotSharedMem.h`/`libRobotSharedMem`, `RobotTestGait/inc`(defineConfigMotor.h 등).
