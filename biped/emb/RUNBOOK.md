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
4. 하중을 서서히 로봇에 넘긴다. 이때 `q_leg_deg` 가 명령과 얼마나 벌어지는지 본다
5. `mode=stand` — 크레인 줄은 **느슨하게 남겨둔 채로**
6. 30초 이상 유지되면 성공. 줄을 완전히 풀기 전에 최소 한 번 외란(살짝 밀기)을 준다

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
