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
