# 실기 기동 런북 — 터미널별 실행 순서

매 세션 이 순서를 따른다. 배경·근거는 `NEXT_HW.md`, 사고 이력은 각 항목의 ⚠ 참조.

**전제 — writer 는 언제나 하나다.** 모터에 명령을 쓰는 프로세스(`RobotEmbedded` /
`biped_emb.py` / `actuator_test.py` / `RobotTestGait` / `mot_test`)가 둘 이상이면
서로의 명령을 덮어쓴다. 아래 순서는 그걸 지키도록 짜여 있다.

---

## 0. 시작 전 — 싹 정리 (터미널 A)

```bash
cd ~/simulation/biped/emb && diag/emb_ctl.sh reset
```

writer → Emb → **되살리는 래퍼 셸**까지 전부 지운다.

> ⚠ **`reset` 은 Emb 도 죽인다.** 반드시 Emb 기동 **전**에만 쓸 것.
> 제어기 기동 절차(§2) 안에 넣으면 방금 띄운 Emb 가 죽는다.

EtherCAT 이 OP 를 잃었던 경우(값 동결)에는 여기서 **모터 전원 OFF → 3초 → ON**.
Emb 재기동만으로는 슬레이브가 안 올라온다.

---

## 1. 임베디드 (터미널 A)

```bash
cd ~/simulation/biped/emb && diag/emb_ctl.sh start
```

중복 가드 + 신선도(`MotorStatus16`) 확인까지 한다. `✓ MotorStatus16 신선` 이 떠야 성공.

<details><summary>직접 띄우고 싶다면</summary>

```bash
pgrep -cx RobotEmbedded          # ★반드시 먼저. 0 이 아니면 띄우지 말 것
cd ~/ZSource/RobotEmbedded/build && ./src/RobotEmbedded    # sudo 없이
```

⚠ 이 방식은 **중복을 막아주지 않는다.** 위 `pgrep` 을 잊으면 그대로 중복된다
(2026-08-12 에 4개까지 늘어 전 채널이 얼어붙었다).

⚠ **`sudo` 를 붙이지 말 것.** `setcap cap_net_admin,cap_net_raw=eip` 가 적용돼 있고
SHM perms 가 666 이라 비루트로 충분하다. sudo 로 띄우면 래퍼가 2개 끼어
**Ctrl+C 가 실제 프로세스에 도달하지 않는다** — 죽은 줄 알고 다시 띄우면 중복 writer 가
되고 root 소유라 kill 도 어렵다(실측: sudo → SIGINT 무시 · 직접실행 → 0.11초 종료).
</details>

---

## 2. 제어기 (터미널 B)

```bash
pkill -f 'app/biped_emb.py'
cd ~/simulation/biped/emb && python3 app/biped_emb.py --start-mode off
```

**`backend=shm` 으로 떠야 한다.** `backend=mock` 이면 SHM 라이브러리를 못 찾은 것이다.

> ⚠ 여기에 `emb_ctl.sh reset` 을 넣지 말 것 — §1 에서 띄운 Emb 가 죽는다.
> 제어기만 재시작할 때는 위 `pkill -f 'app/biped_emb.py'` 로 충분하다.

---

## 3. 뷰어 + GUI (터미널 C)

```bash
cd ~/simulation/biped && ./run_hw.sh
```

---

## 4. 영점 캘리브레이션

**제어기가 `off` 모드로 떠 있어야 한다** — `calib_zero.py` 는 제어기가 발행하는
상태파일을 읽는다. 제어기를 죽이면 읽을 게 없다.

```bash
# 터미널 B: 제어기가 off 모드로 떠 있는 상태에서
# 로봇을 기준자세(지그)에 물리적으로 맞춘 뒤 — 터미널 D:
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

python3 actuator_test.py --ch 0 --ch 4 --tests torque --pose neutral   # hip
python3 actuator_test.py --ch 1 --ch 5 --tests torque --pose neutral   # thigh
python3 actuator_test.py --ch 3 --ch 7 --tests torque --pose neutral   # foot
python3 actuator_test.py --ch 2 --ch 6 --tests torque --pose neutral   # calf
```

`--ch` 는 반복 가능하다 — 좌우 짝을 한 번에 돌린다(한 프로세스 안에서 순차 실행).
`--pose neutral` 은 thigh 를 중력중립각(+21.7°)으로 옮겨 홀드 처짐을 없앤다.

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

`/tmp/emb.log` 는 **0바이트다** — RobotEmbedded 는 stdout 에 아무것도 안 쓴다.
`emb_ctl.sh log` 의 EtherCAT 요약이 늘 비어 있는 이유이고, 로그 기반 진단은 불가능하다.
