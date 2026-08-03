# biped/emb — 실기(Emb) 배포 인터페이스

biped **모델기반 제어기(MPC+WBIC)** 를 RGA `RobotSharedMem`(Gait) 실모터에 연결하는 배포 프로젝트.
`quad_ctrl` 의 HAL 경계 원칙(State가 유일 입력·HAL이 유일 부작용·config 파라미터화)을 따른다.

> **1단계 목표(현재): GUI 명령 → 각축이 잘 움직이는지 인터페이스 검증(per-axis JOG).**
> 모델기반 Stand/Walk 는 배선 완료(`control/model_ctrl.py`) — jog 로 부호·오프셋·한계 확정 후 실행.
> 언어=Python 우선(검증된 컨트롤러·루프·GUI 재사용). RT 부족 시 C++ 이관(`RobotTestGait` 골격 참조).

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
│   ├── mode_fsm.py           off/jog/hold/stand/walk 상태기계
│   └── model_ctrl.py         모델기반(MPC+WBIC) 래퍼 — biped_deploy 포팅(stand/walk)
├── app/biped_emb.py          메인 RT 루프 + 상태발행
├── gui/teleop_emb.py         JOG 패널(8관절 슬라이더·실측) + 모드버튼 + walk 조이스틱
└── run_emb.sh                런처(app + GUI)
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
