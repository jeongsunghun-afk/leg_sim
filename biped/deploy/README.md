# biped 실배포 (sim → 실모터 HW)

컨트롤러(MPC+WBIC)는 **순수 토크**를 출력한다. 배포 = 플랜트만 교체 — `RobotInterface`
백엔드를 `SimInterface`(MuJoCo) → `HardwareInterface`(실모터)로 바꾸면 컨트롤러·게인·GUI 그대로.

```
컨트롤러(MuJoCo=동역학 모델)                RobotInterface (플랜트 경계)
  c.control(dt) → d.ctrl(tau) ──LowCmd──▶  write():  sim=mj_step / HW=setTorqueRef
  M·J·bias 계산  ◀──LowState── apply_state() ◀─read(): sim=d 스냅샷 / HW=인코더+IMU(+추정)
```

## 실행
```bash
# sim 검증(배포 API가 biped_run과 동일 결과인지) — GUI 함께
python biped_deploy.py --backend sim --view      # 또는 헤드리스(--view 생략)
#   GUI:  cd ../../quad && python teleop_gui_biped.py     (같은 /tmp/biped_cmd.json)

# 실배포 (HardwareInterface 구현 후)
python biped_deploy.py --backend hw
```
GUI **Off 전원** = `enable_motors(False)` = 토크차단(limp) / 실HW=motor disable. 재투입=Stand.

## 구성
| 파일 | 역할 |
|---|---|
| `robot_interface.py` | `LowState`·`LowCmd`(SDK 규약) + `RobotInterface` + `SimInterface`(동작) + `HardwareInterface`(스텁) |
| `biped_deploy.py` | 배포 루프(read→apply→control→write). `--backend sim/hw`. GUI JSON 채널 |

관절 순서 = `[HL_hip, HL_thigh, HL_calf, HL_foot, HR_hip, HR_thigh, HR_calf, HR_foot]` (= MuJoCo qpos[7:]).

## 실배포 체크리스트 (HardwareInterface 채우기)

### A. 모터 버스 (`read`/`write`)
- [ ] CAN/EtherCAT/DDS 초기화, 8모터 ID 매핑(위 관절 순서).
- [ ] **부호·감속비·오프셋 캘리브레이션** — sim 관절부호와 실모터 방향 일치(제로자세 정렬).
- [ ] 토크 명령 = `setTorqueRef(tau)` (WBIC=순수 ff토크, kp=kd=0). 단위[Nm] 확인.
- [ ] 토크 한계 = `TAU_PEAK=[84,84,126,96]×2` (biped_wbic). 모터 Peak과 대조 → 메모리 `02leg-motor-spec`.

### B. IMU (`read`)
- [ ] quat(base 자세)·gyro(각속도)·acc 스트림. **좌표계 = base body-frame** 정렬(축 부호).
- [ ] IMU→base 링크 오프셋 보정. 지연[ms] 측정(WBIC 안정성 영향).

### C. ★base 상태추정 (`apply_state`) — 최대 sim2real 갭
- [ ] base **위치/선속도**는 직접 측정 불가 → **leg-odometry + IMU EKF** 필요.
      (stance 발 world 고정 가정 + FK로 base pos, IMU로 fusion.) WBIC CoM/속도 task가 이걸 씀.
- [ ] 없으면 CoM task 부정확 → 균형 저하. checklist C(상태추정 노이즈·지연) 참조.
- [ ] MuJoCo 모델(`HardwareInterface.m`)로 FK/자코비안 재사용 가능.

### D. 안전 (`write`/`enable_motors`)
- [ ] 토크 **slew-rate 제한**·관절각/속도 한계 클램프(발산 시 폭주 방지).
- [ ] **E-stop**(하드웨어 킬) + 통신 워치독(타임아웃→enable_motors(False)).
- [ ] 낙상감지 시 sim은 자동리셋 → **실HW는 안전정지(damp 또는 disable)**로 교체(`biped_deploy` 해당 분기).
- [ ] 첫 기동: 크레인 서스펜션/저게인·저속(vx≤0.1)부터. `no-unphysical-sim2real-hacks` 준수(상태 강제절단 금지).

### E. 운용
- [ ] 제어율 = sim 500Hz(dt=0.002). 실루프 지터 측정, 실시간 보장(RT 커널/우선순위).
- [ ] 점 발 한계(선회·측방 marginal) 그대로 → 급기동은 RL 몫. 전진/후진+gentle선회만 로버스트.

## 참조
- 메모리 `rbq-sdk-reference`(SDK 규약·함수명) · `02leg-motor-spec`(모터 스펙) · `sim2real-checklist-17dof`(갭 상세) · `no-unphysical-sim2real-hacks`.
- 상위 `../README.md`(제어 구조) · 개발기록 `biped-mpc-reimpl`.
