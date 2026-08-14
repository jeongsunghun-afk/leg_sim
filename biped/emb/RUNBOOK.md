# 실기 기동 런북 — 터미널별 실행 순서

## ⚡ 한눈에 — 터미널 배치

| | 터미널 1 (Emb) | 터미널 2 (뷰어) | 터미널 3 (제어기·시험) |
|---|---|---|---|
| ① | `emb_ctl.sh reset` | | |
| ② | *모터 전원 OFF → 3초 → ON* | | |
| ③ | `emb_ctl.sh start` | | |
| ④ | | `./run_hw.sh` | |
| ⑤ | | | `biped_emb.py --start-mode off` |
| ⑥ | | | (시험 전) `pkill -f 'app/biped_emb.py'` → `actuator_test.py …` |

> ⚠ **`reset` 을 터미널 3 에 두지 말 것.** reset 은 Emb 도 죽인다 —
> 터미널 1 에서 방금 띄운 Emb 가 사망하고, 그 터미널이 통째로 헛일이 된다.
> `reset` · 전원사이클 · `start` 는 **반드시 같은 터미널(1)에 모은다.**

> ⚠ Emb 와 제어기를 **같은 터미널에 두지 말 것.** `biped_emb.py` 는 포그라운드로 물고
> 있어서 그 터미널을 못 쓴다. `emb_ctl.sh start` 는 백그라운드로 띄우고 바로 돌아온다.


매 세션 이 순서를 따른다. 배경·근거는 `NEXT_HW.md`, 사고 이력은 각 항목의 ⚠ 참조.

**전제 — writer 는 언제나 하나다.** 모터에 명령을 쓰는 프로세스(`RobotEmbedded` /
`biped_emb.py` / `actuator_test.py` / `RobotTestGait` / `mot_test`)가 둘 이상이면
서로의 명령을 덮어쓴다. 아래 순서는 그걸 지키도록 짜여 있다.

---

## 0. 시작 전 — 싹 정리 (터미널 1)

```bash
cd ~/simulation/biped/emb && diag/emb_ctl.sh reset
```

writer → Emb → **되살리는 래퍼 셸**까지 전부 지운다.

> ⚠ **`reset` 은 Emb 도 죽인다.** 반드시 Emb 기동 **전**에만 쓸 것.
> 제어기 기동 절차(§2) 안에 넣으면 방금 띄운 Emb 가 죽는다.

EtherCAT 이 OP 를 잃었던 경우(값 동결)에는 여기서 **모터 전원 OFF → 3초 → ON**.
Emb 재기동만으로는 슬레이브가 안 올라온다.

---

## 1. 임베디드 (터미널 1)

```bash
cd ~/simulation/biped/emb && diag/emb_ctl.sh start
```

중복 가드 + 신선도(`MotorStatus16`) 확인까지 한다. `✓ MotorStatus16 신선` 이 떠야 성공.

### 또는 — 터미널에 직접 띄운다 (★로그를 눈으로 보려면 이쪽)

```bash
pgrep -cx RobotEmbedded          # ★반드시 먼저. 0 이 아니면 띄우지 말 것
cd ~/ZSource/RobotEmbedded/build && ./src/RobotEmbedded    # sudo 없이

# 다른 터미널에서 신선도 확인
cd ~/simulation/biped/emb && diag/emb_ctl.sh fresh
```

터미널에 직접 띄우면 stdout 이 **줄단위 버퍼링**이라 로그가 실시간으로 보인다.
EtherCAT 동결의 원인이 아직 미제이므로, **동결 순간에 뭐라고 찍는지 직접 보는 것**이
지금 가장 값어치 있다. Ctrl+C 도 즉시 듣는다(sudo 래퍼가 없으므로).

⚠**이 경로는 파일로 안 남는다**(터미널 출력). 그래서 디스크 걱정은 없지만, 창을 닫으면
  증거도 사라진다. **나중에 볼 로그가 필요하면 `emb_ctl.sh start`** 를 쓸 것 —
  거기엔 반복 스팸 솎기(1/500)가 들어 있다.
  ⚠직접 실행에 `> /tmp/emb.log` 를 붙이지 말 것. 필터가 없어 **시간당 1.7GB** 로 자란다
    (2026-08-14 실측: 3.5GB 도달, 디스크 79%). 굳이 붙이려면 `emb_ctl.sh` 의 awk 를 같이 쓸 것.

⚠ 대신 `start` 가 해주던 두 가지가 빠진다:
  ① **중복 가드** — 위 `pgrep` 을 잊으면 그대로 중복된다(2026-08-12 에 4개까지 늘어
     전 채널이 얼어붙었다). ② **신선도 확인** — `emb_ctl.sh fresh` 로 대신한다.

⚠ **`sudo` 를 붙이지 말 것.** `setcap cap_net_admin,cap_net_raw=eip` 가 적용돼 있고
SHM perms 가 666 이라 비루트로 충분하다. sudo 로 띄우면 래퍼가 2개 끼어
**Ctrl+C 가 실제 프로세스에 도달하지 않는다** — 죽은 줄 알고 다시 띄우면 중복 writer 가
되고 root 소유라 kill 도 어렵다(실측: sudo → SIGINT 무시 · 직접실행 → 0.11초 종료).

---

## 2. 제어기 (터미널 3)

```bash
pkill -f 'app/biped_emb.py'
cd ~/simulation/biped/emb && python3 app/biped_emb.py --start-mode off
```

**`backend=shm` 으로 떠야 한다.** `backend=mock` 이면 SHM 라이브러리를 못 찾은 것이다.

> ⚠ 여기에 `emb_ctl.sh reset` 을 넣지 말 것 — §1 에서 띄운 Emb 가 죽는다.
> 제어기만 재시작할 때는 위 `pkill -f 'app/biped_emb.py'` 로 충분하다.

---

## 3. 뷰어 + GUI (터미널 2)

```bash
cd ~/simulation/biped && ./run_hw.sh
```

---

## 4. 영점 캘리브레이션

**제어기가 `off` 모드로 떠 있어야 한다** — `calib_zero.py` 는 제어기가 발행하는
상태파일을 읽는다. 제어기를 죽이면 읽을 게 없다.

```bash
# 터미널 3: 제어기가 off 모드로 떠 있는 상태에서
# 로봇을 기준자세(지그)에 물리적으로 맞춘 뒤 — 터미널 4:
cd ~/simulation/biped/emb && python3 diag/calib_zero.py            # 계산만
cd ~/simulation/biped/emb && python3 diag/calib_zero.py --apply    # config 갱신
```

`--apply` 는 **config 파일만** 고친다. 모터에는 아무것도 쓰지 않는다.
적용 후 **제어기·Emb 재기동**이 필요하다 → §0 부터 다시.

---

## 5. 단축 테스트

제어기와 하니스는 **둘 다 writer** 다. 반드시 제어기를 먼저 끈다.

```bash
pkill -f 'app/biped_emb.py'
cd ~/simulation/biped/emb/pace

python3 actuator_test.py --ch 3 --ch 7 --tests torque --pose neutral   # ① foot
python3 actuator_test.py --ch 1 --ch 5 --tests torque --pose neutral   # ② thigh
python3 actuator_test.py --ch 2 --ch 6 --tests torque --pose neutral   # ③ calf
python3 actuator_test.py --ch 0 --ch 4 --tests torque --pose neutral   # ④ hip
```

`--ch` 는 반복 가능하다 — 좌우 짝을 한 번에 돌린다(한 프로세스 안에서 순차 실행).
`--pose neutral` 은 thigh 를 중력중립각(+21.7°)으로 옮겨 홀드 처짐을 없앤다.

### ★순서에는 이유가 있다 — 임의로 바꾸지 말 것

- **① foot 이 맨 앞** — 유일하게 **같은 날 기준선**이 있다(HL 0.637 · HR 0.658 Nm, 2026-08-12).
  bias 도 0.032 로 사실상 0 이라 값이 바뀔 이유가 없다. 재현되면 위치의존 bias·중력추종
  정착이 기존 경로를 안 건드렸다는 뜻이고, **다르면 여기서 멈춘다**. 나머지 세 축엔
  비교 기준이 없어서, foot 을 통과시키지 않으면 이상값이 축 특성인지 코드 탓인지 못 가린다.
- **② thigh** — bias ≈ 0 이라 안전하고, 파라미터가 하나도 없는 신규 축이다.
- **③ calf 를 hip 앞에** — **토크 규약(채널 vs 관절)을 여기서 가른다.** gear_k 가 1 인 hip
  으로는 구분이 안 되고, calf 는 1.5 라 차이가 가장 크다(채널 0.806 vs 관절 1.209 Nm).
  틀리면 프로브가 중단하면서 두 후보를 같이 찍는다. 그걸 **알고 hip 에 들어가야** 한다 —
  hip 의 bias 5.25 가 규약 탓에 틀렸다면 5 Nm 급 오차다.
- **④ hip 이 맨 뒤** — 토크가 가장 크고(bias 5.25 + swing 2.5 = 피크 7.75 Nm),
  2026-08-12 의 **EtherCAT 동결 2회가 모두 hip 시험 중**이었다(기전 미확인).

시험이 끝나면 하니스가 `limp` 하고 빠진다. 제어기를 다시 쓰려면 §2 로.

---

## 6. 2점 평발 stand — **모델기반 제어의 첫 실기 동작**

PACE 세팅 뒤 처음 시도할 동작. 보행보다 훨씬 쉽다(스텝 없음·가속 없음·토크 포화 없음).
**그러나 `stand` 는 이 로봇에서 한 번도 실기에서 돈 적이 없다** — 여기까지는 전부
jog/home/hold 였다. 아래는 그 첫 시도를 위한 절차다.

### 6-a. ⚠⚠ 먼저 — **HOME 자세로 바닥에 내려놓으면 안 된다**

영점(전축 0°) 자세의 발 기하를 재보면 (`cpp/build/flat_home --eval 0 0 0`):

```
  밑창 기울기 (z_toe−z_heel) = −0.14378 m   ❌ 발끝만 닿는다(까치발)
  CoM − 밑창중심             = +0.11065 m   접촉 반길이 1.27cm 대비 여유 −772.9%
```

**발끝으로 서서 CoM 이 11cm 앞에 있는 자세다. 내려놓는 즉시 앞으로 넘어진다.**
평발 자세는 발목이 **−59.8°** 다. 즉 **매달린 상태에서 평발 자세를 먼저 만들고**,
발바닥이 지면과 평행함을 눈으로 확인한 뒤 크레인을 내려야 한다.

| | thigh | calf | foot |
|---|---:|---:|---:|
| HOME(영점) | 0.00° | 0.00° | 0.00° |
| **평발 stand** | **+3.68°** | **−23.87°** | **−59.81°** |

⚠발목 59.8° 는 큰 이동이다. `home.max_speed_dps` 15 dps 기준 **약 4초**가 걸린다.

### 6-b. 시뮬에서 먼저 (기준값 확보)

```bash
cd ~/simulation/biped/cpp && ./verify.sh          # 첫 항목이 flat stand 다
# 단독 실행:
CONTACT=1 ./build/biped_sim ../biped_flatfoot.mjcf 0.0 15
#   기대: 생존 15.00s(무낙상) · base=(−0.010, 0.001, 0.438) · tilt 0.2°
```

### 6-c. 실기 절차

```bash
# §0~§3 을 먼저 마친 상태에서 (Emb·제어기·뷰어 기동, 영점 확인)
pkill -f 'app/biped_emb.py'                        # ★writer 는 하나만. 배포와 겹치면 안 된다
cd ~/simulation/biped/cpp
./build/biped_deploy --mjcf ../biped_flatfoot.mjcf
#   기동 로그에서 반드시 확인:  cmode=1 **2점 평발(정적 자세유지)**
#   ★--mjcf 를 빼면 기본이 점발(biped_from_quad)이라 cmode=0 으로 뜬다 — 그러면 stand 가 아니다.
```

순서 (한 단계씩, 각 단계 후 멈춰서 확인):

1. **매달린 채** `mode=hold` — 현재 자세 유지되는지, `loop_hz` 500 근처인지
2. **매달린 채** 평발 자세로 이동 → **발바닥이 지면과 평행한지 눈으로 확인**
3. 크레인을 **천천히** 내려 발바닥 전체가 닿게 한다. 하중은 아직 크레인이 받는다
4. 하중을 서서히 로봇에 넘긴다. **`hold` 상태로** 한다 — 아직 `stand` 가 아니다.
   ⚠**순서를 뒤집지 말 것.** 발이 살짝만 닿은 채로 `stand` 를 켜면 접촉이 미약해
     WBIC 가 요구하는 λ 를 지면이 못 내주고 **QP 가 매 틱 실패**한다(§6-c2).
     그 상태에서 크레인을 마저 내리면 **폐루프가 죽은 채로 하중이 실린다.**
   ★여기서 **α(토크 스케일)가 공짜로 측정된다.** 값 모니터의 `Δq`·`≈Nm` 열을 보고
     아래 예측(매달린 채 hold, 평발자세)과 대조한다. 실측이 크면 α<1(토크가 약함):

   | 축 | 중력토크 τ_g | kp | 예상 처짐 |
   |---|---:|---:|---:|
   | hip   | ±5.247 Nm | 100 | ±3.00° |
   | thigh | −2.28 Nm  |  50 | −2.6°  |
   | calf  | +0.411 Nm |  80 | +0.29° |
   | foot  | −0.189 Nm |  30 | −0.36° |

   ⚠**foot 으로 판단하지 말 것** — 중력 0.19 Nm 가 정지마찰(0.63~0.71 Nm)에 완전히 묻힌다.
     hip 이 5.2 Nm 로 가장 크니 신호가 제일 깨끗하다.
5. `mode=stand` — 크레인 줄은 **느슨하게 남겨둔 채로**
6. 30초 이상 유지되면 성공. 줄을 완전히 풀기 전에 최소 한 번 외란(살짝 밀기)을 준다

### 6-c2. ★접지가 됐는지 어떻게 아나 — QP 실패율

값 모니터에 `QP실패 N% · K=n · com_err Nmm` 이 뜬다. **접지 판정의 유일한 지표**다.

| | 접촉 K | QP 실패 | com_err |
|---|---:|---|---|
| **접지 정상** | **4** | **~0%** | 6 mm (줄어든다) |
| **발이 뜸/미약** | **3 이하** | **~95%** | 127 mm 에서 **고정** |

(2026-08-14 시뮬 실측 — 베이스를 용접하고 발을 공중에 띄운 모델로 잰 값.)

⚠**이게 왜 중요한가:** 접촉이 부족하면 QP 가 해를 못 찾고 폴백인 **중력보상 홀드**로
  떨어진다. 그런데 그때 로봇은 **겉보기에 안정돼 보인다** — 정착하고, 안 넘어지고,
  tilt 도 크지 않다. 그래서 **"매달린 채로도 stand 가 되네" 라고 오판하기 쉽다.**
  실제로는 폐루프가 죽고 개루프로 도는 상태다. `com_err` 이 **안 줄고 고정**되는 게 특징이다.

※ "다리가 휘젓는다"(NEXT_HW §0)보다 이 서술이 정확하다 — 실측해보니 발산이 아니라
  **조용한 실패**였다. 발산보다 오히려 알아채기 어렵다.

### 6-d. 이 시점의 미해결 — 알고 들어갈 것

| 항목 | 상태 | 의미 |
|---|---|---|
| **IMU 전 값 0** | 미복구 | **tilt E-stop 이 무력하다**(`hypot(roll,pitch)≡0`). 넘어져도 소프트가 못 잡는다 ⇒ **크레인이 유일한 안전장치다** |
| **자세 피드백 없음** | 위와 동일 | 제어기가 base 기울기를 모른 채 선다. 기울기 누적을 잡을 수단이 없다 |
| 토크 트립 | 15 Nm / 50 ms **래치** | 걸리면 `off` 명령으로만 풀린다. stand 는 정적이라 여유는 크다 |
| 속도 트립 | 200 dps 즉시 래치 | 발목 채널각은 모델각의 1.2배임에 주의 |
| 워치독 | 500 ms | **명령 내용이 바뀌어야** 생존 판정. `seq` 를 증가시킬 것 |
| 바닥 마찰 μ | **미측정** | 시뮬 1.6 · WBIC 계획 0.566 · 실기 미상. 미끄러우면 발이 벌어진다 |
| `coef`·`gear_k` | 정량 미측정 | 발목 명령이 계수 오차만큼 어긋난다. 큰 발목각(−59.8°)이라 영향이 커진다 |

⚠**τ 스케일 α 미검증**은 stand 에도 걸린다 — 중력보상 토크가 α 배 어긋나면 그만큼 처진다.

### 6-d2. 값 모니터 — **측정 vs 명령**을 눈으로 본다

`./run_hw.sh` 가 3D 자세 뷰어·GUI 와 함께 **값 모니터(그래프 창)** 를 띄운다(2026-08-13 추가).
3D 뷰어는 자세만 보여주지 명령을 얼마나 못 따라오는지는 안 보인다 —
stand 처럼 정적인 동작에선 그 편차가 판단의 전부다.

```bash
./run_hw.sh                        # 뷰어 + GUI + 값 모니터(그래프)
MON=text ./run_hw.sh               # 그래프 대신 표(터미널). SSH 전용일 때
MON=0 ./run_hw.sh                  # 값 모니터 없이
MON_ARGS="--win 20" ./run_hw.sh    # 시간창 20초
QUAD_STATE=/tmp/biped_state.json ~/.venvs/gui/bin/python monitor_plot.py   # 따로 띄우기
```

**그래프 창(`monitor_plot.py`)** — 축을 **하나 고르면** 그 축의 위치·속도·토크 3개를
측정(파랑) vs 명령(주황)으로 그린다. 8축 × 3량 × 2계열 = 48줄을 한 번에 그리면 아무것도
안 보이기 때문이다. 판단은 늘 "어느 축이 이상한가" → "그 축이 왜" 순서로 가고,
앞은 3D 뷰어·표가, **뒤를 이 창이** 한다.
★이력은 **전 축을 항상 버퍼링**한다 — 축을 바꿔도 과거가 살아 있다.

**표(`monitor_state.py`)** — 8축을 한 화면에. `--ch` 로 채널각도 같이 본다.
`≈Nm` 열이 핵심이다: 위치오차를 그 축의 kp 로 토크환산한 값
(드라이버 MIT `τ = kp·err[rad] + kd·derr`, kp 1 = 0.0175 Nm/deg). 각도만 보면
kp100 인 hip 과 kp30 인 foot 을 같은 잣대로 재게 된다.

둘 다 **읽기 전용**이라 writer 충돌을 만들지 않는다. 단위는 전부 **모델각**이다.
⚠**C++ 배포(`biped_deploy`)에서는 q명령·dq명령 이 비는 게 정상**이다 — 순수 토크모드
(kp=kd=0)라 위치명령이 존재하지 않는다. 그때는 토크만 비교하면 된다.

⚠그래프 창은 **teleop GUI 와 같은 경로**(setsid + 명시적 DISPLAY)로 띄운다.
처음엔 `x-terminal-emulator` 로 터미널 창을 띄웠는데 안 떴다 — 이 기기의 그것은
`gnome-terminal.wrapper` 라 폐기된 `-e` 분기로 갔고, 런처가 반환하지 않으며,
에러를 `/dev/null` 로 버려 조용히 실패했다. 에뮬레이터마다 규약이 달라 계속 깨진다.

관절별로 `q·dq·τ` 를 **측정 / 명령 / 편차**로 나란히 보여준다. 전부 **모델각**이고
채널각은 `--ch` 로만 나온다. 읽기 전용이라 writer 충돌을 만들지 않는다.

- **`≈Nm` 열이 핵심이다** — 위치오차를 그 축의 kp 로 토크환산한 값이다
  (드라이버 MIT 법칙 `τ = kp·err[rad] + kd·derr`, kp 1 = 0.0175 Nm/deg).
  각도만 보면 kp100 인 hip 과 kp30 인 foot 을 같은 잣대로 재게 된다.
- 색: 노랑 = 주의, 빨강 = 위험. 토크 임계는 **런타임 트립 15 Nm 과 같은 값**으로 잡았다.
- 하단 `세션 최대` 는 순간 스파이크를 잡는다 — 눈으로는 못 본다.
- **C++ 배포(`biped_deploy`)에서는 q명령·dq명령 이 `—` 로 뜨는 게 정상**이다.
  순수 토크모드(kp=kd=0)라 위치명령이 존재하지 않는다. 그때는 토크 열만 보면 된다.

### 6-e. 실패하면 어디를 보나

- **앞/뒤로 서서히 기운다** → CoM 이 밑창중심에서 벗어났다는 뜻. `flat_home` 으로 자세를
  다시 확인하고, 실기 링크질량이 CAD 와 다른지 의심할 것(torso 2.8kg 은 실측이지만
  `I_link` 는 MJCF 유래다).
- **무릎이 처진다** → 중력보상 토크 부족. τ 스케일 α 또는 `gear_k` 를 의심.
  `q_leg_deg` 와 명령의 정상편차를 보면 갈린다.
- **발목만 어긋난다** → `coef`(커플링) 오차. 무릎각이 클수록 커지므로 −23.9° 에서
  이미 보인다. `diag/couple_check.py` 를 **무여자로** 재측정할 것.
- **채널이 죽는다** → §"고장 판별" 로. 다축 동시 부하는 stand 가 처음이다.

---

## ★MD80 응답 패킷 — 우리가 볼 수 있는 것의 전부

출처: `docs/MD-80 CAN 통신 정리_이형상_20260511.pdf` (DEFAULT_RESPONSE) + 벤더 확인 2026-08-14.
구조는 `EMB(Pi) ──EtherCAT── MCU(RGA LAN9252 8AXIS) ──FDCAN── MD80` 이고,
**MD80 이 안 주는 것은 MCU 도 못 만든다.** 그래서 이 표가 정보의 상한이다.

| BYTE | 내용 | 타입 | 우리에게 오나 |
|---|---|---|---|
| 0 | FRAME ID (0x0A) | uint8 | — |
| **1–2** | **ERROR VECTOR** | uint16 | ⚠**하위 8bit 만** → `ucStatus` |
| **3** | **MOTOR TEMPERATURE** | uint8 [°C] | ⚠SHM 엔 오는 듯(`fAccelrationOrTemperture`)하나 **브리지가 안 읽는다** |
| 4–7 | MAIN ENCODER POSITION | float [rad] | ✅ `fPosition` |
| 8–11 | MAIN ENCODER VELOCITY | float [rad/s] | ✅ `fVelocity` |
| 12–15 | **MOTOR TORQUE** | float [Nm] | ✅ `fTorque` |
| 16–19 | OUTPUT ENCODER POSITION | float | ❌ **MCU 에서 삭제**(패킷 이슈) |
| 20–23 | OUTPUT ENCODER VELOCITY | float | ❌ 〃 |

여기서 곧바로 따라오는 것 셋:

- **전류는 어디에도 없다.** `fCurrent` 는 채울 원천이 없어 `fTorque` 복사본이다
  (`pace/hwio.py:134` 가 비트단위 일치를 확인해 뒀다). **전류 요청은 접는다** —
  있어도 `τ = Kt·Iq` 라 Kt 뒤에 있어 토크 스케일 α 를 못 푼다.
- **엔코더는 모터축 하나뿐이다.** OUTPUT(관절축) 엔코더가 MCU 에서 잘린다
  ⇒ `diag/couple_check.py` 의 **(A) 경우가 확정**이다. 커플링은 소프트로 풀 수밖에 없고,
  `joint_map` 이 하는 그대로가 맞다.
- **`ucStatus` = ERROR VECTOR 하위 8bit.** 아직 정제 전 원값이다.
  ⚠**상위 8bit 는 잘려서 안 온다** — 거기 있는 비트는 못 본다.

### 래치오프가 나면 `ucStatus` 부터 본다

값 모니터에 **`err` 열**로 원값이 hex 로 뜬다(0 이 아니면 빨강). CSV 에도 `*_stt` 로 남는다.
종전에는 `health` 문자열(`ok`/`fault`)만 나가서 **"죽었다"는 것만 알고 왜인지는 못 봤다.**

MAB SDK 의 비트 정의(`~/CANdle-SDK/candlelib/src/MD/MDStatus.hpp`)를 참고 대조군으로 쓸 수 있다.
`hardwareStatus` 는 **하위 6bit** 에 원인이 몰려 있어 잘려도 살아남는다:

```
0 OverCurrent   1 OverVoltage   2 UnderVoltage
3 MotorTemp     4 MosfetTemp    5 ADCCurrentOffset
```

⚠단 ERROR VECTOR 가 `hardwareStatus` 인지 `quickStatus`(요약 비트) 인지는 **미확인**이다.
  `quickStatus` 라면 bit4 가 "하드웨어 에러 있음" 요약이고 상세는 별도 레지스터다.
  **실제 값을 한 번 보면 갈린다** — 그게 지금 `err` 열을 넣은 이유다.

## 고장 판별 — 값이 얼어붙었을 때

Emb 는 EtherCAT OP 를 잃어도 **프로세스가 계속 돌고, 마지막 버퍼를 재발행하며,
`IsUpdated` 플래그까지 1 로 세운다.** 플래그·프로세스 존재로는 판별 불가.

```bash
cd ~/simulation/biped/emb/pace && python3 - <<'EOF'
import time, numpy as np, yaml, ctypes as C
sp=yaml.safe_load(open('spec.yaml')); n=int(sp['shm']['n_channel'])
lib=C.CDLL(sp['shm']['lib']); F=C.POINTER(C.c_float); I=C.POINTER(C.c_int)
lib.bridge_init.argtypes=[C.c_int]; lib.bridge_read.argtypes=[F]*7+[I,I]
z=lambda k=n: np.zeros(k,np.float32)
q,dq,tau,cur=z(),z(),z(),z(); rpy,acc,gyr=z(3),z(3),z(3)
cn=np.zeros(n,np.int32); st=np.zeros(n,np.int32)
p=lambda a:a.ctypes.data_as(F); ip=lambda a:a.ctypes.data_as(I)
lib.bridge_init(3000); S=[]; t0=time.monotonic()
while time.monotonic()-t0<5.0:
    lib.bridge_read(p(q),p(dq),p(tau),p(cur),p(rpy),p(acc),p(gyr),ip(cn),ip(st))
    S.append(np.concatenate([q.copy(),dq.copy(),rpy.copy()])); time.sleep(0.02)
A=np.array(S); u=len(np.unique(A,axis=0))
print(f"표본 {len(A)} · 고유 조합 {u} · IMU rpy 변화폭 {A[:,2*n:2*n+3].ptp(axis=0)}")
print("살아있다" if u>5 else "**얼어붙음 — 모터 전원 OFF/ON 후 §0 부터**")
EOF
```

살아 있으면 IMU rpy 가 늘 미세하게 흔들린다. 변화폭이 정확히 0 이면 동결이다.
`diag/shm_dump` 는 갱신을 기다리며 막혀 표본이 1개만 모이므로 이 판별에 쓸 수 없다.

### 로그가 비어 있다면 — **버퍼링이다**

`/tmp/emb.log` 가 0바이트여도 RobotEmbedded 가 안 쓰는 게 아니다. stdout 이 파일·파이프면
libc 가 **블록 버퍼링**(4~8KB)으로 바꾸는데, Emb 는 영원히 도니 flush 가 안 되고
`pkill` 로 죽이면 버퍼가 **통째로 버려진다**. 터미널로 직접 띄우면(줄단위 버퍼링) 콸콸 보인다.

⇒ `emb_ctl.sh start` 는 `stdbuf -oL -eL` 로 줄단위를 강제한다. 이제 로그가 실시간으로 쌓인다.

```bash
tail -f /tmp/emb.log        # 다른 터미널에서 실시간 확인
```

⚠sudo 폴백 경로만은 stdbuf 로 감쌀 수 없다(sudoers 가 정확한 바이너리 경로만 허용).
그 경로로 떨어지면 로그가 다시 안 보인다 — 무권한 기동이 되는 한 쓰이지 않는다.
