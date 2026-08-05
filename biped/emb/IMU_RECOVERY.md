# IMU 복구 계획 — 실기 IMU 가 전부 0 인 원인과 조치

**작성** 2026-08-05 · **상태** 진단 완료(반증 검증 통과) · 조치 미실시
**증상** SHM(key 1234) `fIMUBuf` 102 float 전부 0. 그런데 `IsUpdatedIMU()` 는 **1** 을 반환.
**영향** `biped_emb.py` 의 tilt E-stop 이 `hypot(roll,pitch)` 로 계산되므로 **tilt ≡ 0 → 임계 40° 에 영원히 도달 불가 = 완전 무력.**

> ⚠ **"값이 0" 보다 "신선한 0" 이 더 위험하다.** freshness 검사로 걸러지지 않아
> 하류가 "유효한 수평 자세" 로 오해한다. 우리 브리지에는 이미 방어를 넣었다(5절).

---

## 0. 결론 요약

| 층 | 원인 | 우리가 고칠 수 있나 |
|---|---|---|
| ① 우리 브리지 | 존재하지 않는 심볼 `IDX_OF_IMU_AVEL` → 자이로 상수 0 | ✅ **완료**(재빌드 필요) |
| ② Emb 인덱싱 | `halIMU.cpp:164-167` 이 파싱 결과를 **0행**에 쓰는데 게터는 **3/4행**을 읽음 | ⚠ RobotEmbedded 소스 |
| ③ Emb 발행 | `engRobot.cpp` 가 매 주기 로컬 0 배열을 SHM 에 덮어씀 · GYRO/QUAT 슬롯 미기입 | ⚠ RobotEmbedded 소스 |
| ④ Emb 포트 | `DEFINED_HEAD_SIMULATOR_ENA` 로 IMU UART open 이 컴파일 제외 | ⚠ 위험, 조건부 |

**좋은 소식**: ②③④ 는 **MCU 펌웨어가 아니다.** RobotEmbedded 는 Pi 위에서 도는 Linux
C++ 앱이고 소스가 이 장비에 있다(`/home/rpetubt/ZSource/RobotEmbedded`). 재플래싱 없이
**재컴파일로 끝난다.** 수정량은 총 5줄 수준.

**더 좋은 소식**: biped 에 필요한 건 **Body IMU** 인데 그 포트 `/dev/ttyAMA4` 는
**이미 존재한다**(아래 4절). 없는 건 Head IMU 용 `ttyAMA3` 뿐이다.
⇒ **부팅 설정 변경도 재부팅도 필요 없을 가능성이 높다.**

---

## 1. ② 인덱싱 버그 — 가장 확실하고 가장 값싼 수정

`RobotEmbedded/modules/sensIMU/halIMU.cpp:164-167`

```c
memcpy(&m_fAccel_MpSEC2[0], &fAccl[0], sizeof(float) * 3);   // ← [0]
memcpy(&m_fGyro_RADpSEC[0], &fGyro[0], sizeof(float) * 3);   // ← [0]
memcpy(&m_fArpy_DEG[0],     &fArpy[0], sizeof(float) * 3);   // ← [0]
memcpy(&m_fMagnetic[0],     &fMagn[0], sizeof(float) * 3);   // ← [0]

m_fTemperature[unDevID] = fTemp;                             // ← [unDevID] 정상!
```

배열은 `static float m_fArpy_DEG[ENUM_DevID_Uart_NUM][3]`(28-29행)이고,
게터는 `m_fArpy_DEG[DEF_DevID_IMU_Body]` = **4행**을 반환한다(59행, `halIMU.h:29`).
4행에 쓰는 코드는 `halIMU_Init` 의 `memset` 뿐 → **Body 슬롯은 영구 0.**

**바로 아래 169행만 `[unDevID]` 로 올바르게 인덱싱돼 있어 오타임이 확실하다.**
게다가 갱신 플래그는 정상적으로 세워지므로(177-182행) "신선한 0" 이 만들어진다.

**수정**: `[0]` → `[unDevID]` 네 곳.
**조건부 컴파일 아님** — `halIMU.cpp` 전체에 `#if/#ifdef` 가 단 한 건도 없다. 무조건 빌드된다.

> ⚠ **`RobotEmbeddedTest/modules/sensIMU/halIMU.cpp` 가 바이트 단위로 동일하다.**
> 요청 시 두 트리 모두 대상임을 명시할 것.

**위험**: 중간. 지금까지 0 이던 Head/Body ARPY·ACCL 이 갑자기 실제 값이 된다.
소비처가 `engRobot.cpp:426-427` 외에 `algKinematics.cpp:486-487`(헤드 자세 계산)과
`halHead` 이므로, **0 을 전제로 튜닝돼 있던 헤드 킨매틱스가 즉시 다른 출력을 낼 수 있다.**
⇒ 반드시 **로봇을 지지대에 올리고 모터 비여자 상태**에서 먼저 확인할 것.

---

## 2. ③ 발행 경로 — 0 덮어쓰기와 GYRO 누락

`RobotEmbedded/engines/engRobot.cpp`

- **415-416**: `float fIMUStt_Head/Body[LEN_OF_IMU_DATA] = {0,};` — 매 호출 로컬 0 초기화
- **430-431 / 438-439**: 갱신 비트가 설 때만 **ARPY(10-12)·ACCL(4-6) 6개만** 복사
- **465 / 493 / 494**: 갱신 비트와 **무관하게 무조건** SHM 에 씀
  ⇒ 갱신이 없던 주기에는 **17 float 전부 0 이 SHM 을 덮어쓴다**
- 이 함수(400-500) 전체에 **GYRO(7-9)·QUAT(0-3)·MAGN·TEMP 에 대입하는 문장이 하나도 없다**

또한 `RobotSharedLib/src/RobotSharedMem_Gait.cpp:1521-1523` 의 `SetIMU` 는
**검증 루프가 빈 채로**(본문이 주석뿐) `memcpy` 직후 `ucIsUpdated_IMU=1` 을 무조건 세운다.
⇒ 이것이 "신선한 0" 의 발생원이다.

> ⚠ **제2 생산자 주의**: `modules/ctrlGait/halGaitJig.cpp:2084` 가 **동일한 Gait SHM
> ForeC 슬롯(0-16)** 에 쓰는 경로가 하나 더 있다(빌드 대상에 포함됨). 데이터원이
> `halIMU_GetPtrMsgData(...Body_ARPY/ACCL)` = 4행이라 ①과 같은 이유로 역시 0 이다.
> ②를 고치면 이쪽도 같이 살아난다.

**요청 사항**
1. 갱신 비트가 서지 않은 주기에는 SHM 쓰기를 건너뛸 것(또는 직전 값 유지)
2. **GYRO(7-9) 를 채울 것** — 패킷에는 이미 들어온다(`interfaceIMU.cpp:325-348` 에서
   ARPY/GYRO/ACCL/MAGN/DIST/TEMP/TIME 디코드). 복사만 안 하고 있다.
3. `SetIMU` 의 빈 검증 루프를 채우거나, 최소한 전부 0 이면 `ucIsUpdated_IMU` 를 세우지 말 것

**QUAT(0-3) 은 요청 대상이 아니다** — IMU 패킷에 쿼터니언 필드 자체가 없고
`halIMU.h:48-61` MsgID enum 에도 없다. **ARPY(오일러) → quat 변환은 호스트에서 한다**
(이미 `hw_interface.py:49` 가 ZYX 변환으로 하고 있다).

---

## 3. ④ UART open 게이트 — **매크로 한 줄만 지우면 실기가 안 뜬다**

`RobotEmbedded/inc/define/defineGeneral.h:44` `#define DEFINED_HEAD_SIMULATOR_ENA`

`interfaceManager.cpp:113-125` 에서 `#ifdef` 쪽이 **빈 블록**이고 `#else` 쪽에만
IMU_Head(devID 3)/IMU_Body(devID 4) 의 `intfUART_Open` 이 있다. FD 배열은 0 으로
초기화되고(44행) 읽기 경로(586행)가 `nFileDesc <= 0` 으로 **로그 없이 조용히** 실패한다.
수신 스레드는 계속 도는 반쪽 상태다(`moduleManager.cpp:213` 의 `halIMU_Receive` 는
매크로 밖이라 무조건 호출됨).

### 🔴 이 매크로는 IMU 말고 3가지를 더 바꾼다 — 그냥 끄면 안 된다

| 위치 | 매크로가 하는 일 | 끄면 |
|---|---|---|
| `halGait.cpp:514-519` | 초기화 카운터 강제 증가 | Gait MCU 수신 없으면 초기화가 영원히 안 끝남 |
| `halHead.cpp:488-493` | 위와 동일(Head) | 동일 |
| `algKinematics.cpp:1310-1314` | 자세계산 입력을 **지령**(`fEqCmd_deg`)으로 치환 | **실측**(`fEqStt_deg`)으로 바뀌어 자세 루프가 닫힘 |

여기에 더해 **devID 3 → `/dev/ttyAMA3` 이 존재하지 않아** open 실패가
`intfUART.cpp:120-123` → `interfaceManager_Init` → `mainInit.cpp:28-51` 로 전파되어
**communicationManager / moduleManager / mainLoop / mainPeriodic / mainThread 초기화가
전부 스킵된다 ⇒ 실기가 아예 기동하지 않는다.**

---

## 4. ✅ 핵심 발견 — Body IMU 포트는 이미 존재한다

이 장비에서 직접 확인(2026-08-05):

```
/dev/ttyAMA0  ttyAMA1  ttyAMA2  ttyAMA4  ttyAMA10      ← ttyAMA3 없음
config.txt:  dtoverlay=uart0/1/2/4/5-pi5               ← uart3 없음
```

- **IMU_Head(devID 3) → `/dev/ttyAMA3`**: 없음 → `dtoverlay=uart3-pi5` + 재부팅 필요
- **IMU_Body(devID 4) → `/dev/ttyAMA4`**: **존재함** ✅

**biped 가 필요한 것은 base 자세 = Body IMU 다.** 따라서 요청은 이렇게 좁힐 수 있다:

> **IMU_Body 의 `intfUART_Open` 만 매크로 밖으로 꺼내고, Head 는 그대로 두거나
> 실패해도 초기화 체인을 중단하지 않게 할 것.**

이러면 부팅 설정 변경도 재부팅도 불필요하고, 위 3가지 부작용(초기화 카운터·자세 루프)도
건드리지 않는다. **가장 안전하고 가장 좁은 변경이다.**

⚠ 다만 **Body IMU 가 물리적으로 ttyAMA4 에 실제 결선돼 있는지는 소스만으로 확인 불가**다.
하드웨어 담당 확인 필요.

---

## 5. ✅ 우리 쪽 조치 — 완료

`emb/hal/shm_bridge.cpp`

1. **존재하지 않는 심볼 제거.** `IDX_OF_IMU_AVEL` 은 SDK 어디에도 없었다
   (`/usr/include/RobotSharedMem.h`·ZSource 전체 grep 0건). 실명은 **`IDX_OF_IMU_GYRO`(=7)**.
   `#ifndef` 폴백이 항상 `-1` 로 정의되어 사용처 `if (... >= 0)` 이 **컴파일 타임에 접혀**
   자이로가 무조건 0 이었다. 경고 하나 없이 통과한다.
   (배포된 `libbipedshm.so` 역어셈블로 확증: rpy/acc 는 `ldr`, gyro 자리는 `movi #0x0` 상수.)
   ⇒ 올바른 심볼로 교체 + 폴백을 `#error` 로 바꿔 **조용한 0 채움을 원천 차단**.

2. **"신선한 0" 방어.** 가속도 3축 크기가 사실상 0 이면(`|a|² ≤ 0.25`) IMU 유효 마스크를
   세우지 않는다. 정상 IMU 는 정지 중에도 중력 ~9.81 m/s² 를 반드시 읽으므로
   전부 0 은 물리적으로 불가능하다. 상류가 **IMU 없음을 인지**하게 만든다.

> 🔴 **재빌드 필요.** `libbipedshm.so`(15:26:37)가 `shm_bridge.cpp`(18:11:54)보다 오래됐다.
> Pi 에서 `hal/build_bridge.sh` 를 다시 돌리기 전까지 **실행 바이너리는 여전히 자이로를 0 으로 채운다.**

---

## 6. 선행 확인 (조치 전 반드시)

**ZSource 소스와 실제로 Pi 에서 도는 임베디드 바이너리가 같은 빌드라는 보장이 없다.**
코드만으로는 확인 불가하므로, 위 요청을 넣기 전에 실측으로 한 번 대조할 것:

```bash
# RobotEmbedded 가동 중에
cd ~/simulation/biped/emb && ./diag/shm_dump      # IsUpdatedIMU 와 RPY 실값 대조
```

- `IsUpdatedIMU=1` 인데 RPY 가 0 → 진단대로(②③)
- `IsUpdatedIMU=0` → 다른 층의 문제. 진단 재검토 필요

---

## 7. 대체 안전장치 — IMU 없이 지금 걸 수 있는 것

**tilt E-stop 이 무력한 동안 런타임 보호가 워치독 하나뿐이었다.**
`emb/pace/hwio.py` 에 **이미 있고 실측으로 확정된** 임계값들을 배포 앱으로 승격했다
(2026-08-05, `config/biped_emb.yaml` + `app/biped_emb.py`):

| 트립 | 임계 | 동작 |
|---|---|---|
| 토크 | 8.0 Nm 가 **50 ms 연속** | limp·래치 (순간 스파이크는 통과) |
| 속도 | 200 deg/s | 즉시 limp·래치 |
| 워치독 | 명령 500 ms 무응답 | 홀드→limp (기존) |

모의 백엔드로 3종 발화 확인 완료(토크 지속=트립 / 속도=트립 / 순간 스파이크=미트립).

⚠ **OFF 모드에서는 검사하지 않는다** — 무여자 상태에서 사람이 다리를 밀 때까지 트립으로
잡으면 재기동이 불가능해진다.

---

## 8. 우선순위

| 순위 | 항목 | 지금(다리 미장착) | 다리 장착 후 |
|---|---|---|---|
| 1 | `build_bridge.sh` 재빌드 | 소스-바이너리 불일치 해소 | 동일 |
| 2 | `diag/shm_dump` 실측 대조(6절) | 요청 전 필수 | 동일 |
| 3 | ② `halIMU.cpp` 인덱싱 수정 요청 | 낮음 | — |
| 4 | ④ **IMU_Body open 만** 매크로 밖으로(4절) | 낮음 | 🔴 **필수** |
| 5 | ③ GYRO 발행 + 0 덮어쓰기 수정 | 낮음 | 🔴 **필수**(추정기 gyro 필요) |

**다리 미장착 현 상태에서 tilt E-stop 부재의 실질 위험은 사실상 0 이다** — 넘어질 몸이 없다.
현재 실장은 hip 2축(ch0/ch4)뿐이다.

**그러나 다리를 다는 순간 4·5 는 필수가 된다.** 상태추정기가 base 자세와 자이로를 요구하고
(`state_estimator.hpp` 의 `quat_wxyz`/`gyro` 인자), tilt E-stop 이 유일한 낙상 보호가 된다.

---

## 9. 담당·요청 문구 초안

**대상**: RobotEmbedded 유지보수 담당 (`halIMU.cpp` 헤더 기준 Lee, Hyoung-Sang)

> RobotEmbedded 의 IMU 데이터가 SHM 에 항상 0 으로 올라옵니다. 세 가지 확인 부탁드립니다.
>
> 1. `modules/sensIMU/halIMU.cpp:164-167` 이 파싱 결과를 `[unDevID]` 가 아니라 `[0]` 행에
>    씁니다. 바로 아래 169행 `m_fTemperature[unDevID]` 는 정상이라 오타로 보입니다.
>    게터(53/59행)가 Head=3 / Body=4 행을 읽으므로 Body 슬롯이 영구 0 입니다.
>    (`RobotEmbeddedTest` 사본도 동일합니다.)
> 2. `engines/engRobot.cpp` 가 갱신 비트와 무관하게 매 주기 로컬 0 배열을 SHM 에 씁니다(465행).
>    또한 성공 경로에서도 ARPY/ACCL 만 복사하고 **GYRO(7-9) 는 채우지 않습니다** —
>    패킷에는 들어오는데 복사만 누락된 것으로 보입니다. 상태추정에 자이로가 필요합니다.
> 3. `interfaceManager.cpp:113` 의 `DEFINED_HEAD_SIMULATOR_ENA` 가 IMU UART open 을 통째로
>    막고 있습니다. 다만 이 매크로가 초기화 카운터·자세계산 입력까지 함께 묶고 있고
>    `/dev/ttyAMA3`(Head) 가 이 장비에 없어서, 매크로를 끄면 기동 자체가 실패합니다.
>    **`/dev/ttyAMA4` 는 존재하므로 IMU_Body 의 open 만 매크로 밖으로 분리**해 주실 수
>    있을까요? 그러면 부팅 설정 변경 없이 body IMU 만 살릴 수 있습니다.
>
> 추가로, `RobotSharedLib` 의 `SetIMU` 가 내용 검증 없이 `ucIsUpdated_IMU=1` 을 세워
> "신선한 0" 이 만들어집니다. 값이 0 인 것보다 위험해서(freshness 로 못 거름) 함께
> 봐주시면 좋겠습니다.
