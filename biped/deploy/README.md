# biped 실배포 (sim → 실모터 HW)

컨트롤러(MPC+WBIC)는 **순수 토크**를 출력한다. 배포 = 플랜트만 교체 — `RobotInterface`
백엔드를 `SimInterface`(MuJoCo) → `HardwareInterface`(실모터)로 바꾸면 컨트롤러·게인·GUI 그대로.

```
컨트롤러(MuJoCo=동역학 모델)                RobotInterface (플랜트 경계)
  c.control(dt) → d.ctrl(tau) ──LowCmd──▶  write():  sim=mj_step / HW=setTorqueRef
  M·J·bias 계산  ◀──LowState── apply_state() ◀─read(): sim=d 스냅샷 / HW=인코더+IMU
                                    ▲ base pose/vel = StateEstimator(leg-odometry, 센서만)
```

## ★상태추정기 (leg-odometry) — 센서만으로 full state
`StateEstimator`(robot_interface.py) = **17-DOF quad와 동일 계보**(state_estimator.hpp Python 포팅).
절대 base(GPS/모캡) **안 씀** = 실로봇 동일. **센서만**(관절 q/dq + IMU quat/gyro + 발 접촉) →
base 위치·선속도 복원. HardwareInterface.apply_state가 자동 사용 → HW는 **센서만 붙이면 full state**.
- 원리: stance 발 정지 가정 → `v_base = −(ω×R·p_foot + R·v_foot)` 평균 + 저역통과(α=0.4), 적분→위치.
- 자세=IMU quat 직접, 각속도=gyro 직접. 위치=속도 적분이라 **드리프트 존재**(실기와 동일, 절대보정 없음).
- 검증(sim, 병행 개루프): walk 0.15 6s = **속도오차 0.01~0.04 m/s(양호)·위치오차 ~10cm(≈12% 드리프트)**.
- 폐루프(`--est-ctrl`, 추정값으로 제어): ~6s 안정 후 드리프트 → 점발 biped는 순수 leg-odom만으론 marginal.
  ★실배포 강건화 = **IMU accel 융합 EKF + 접촉이벤트 위치보정**(D 참조). quad(4발 안정)는 폐루프 OK.

## 실행
```bash
# sim 검증(배포 API가 biped_run과 동일 + 추정오차 리포트) — GUI 함께
python biped_deploy.py --backend sim --view          # 헤드리스=--view 생략, 1s마다 추정오차 출력
python biped_deploy.py --backend sim --est-ctrl      # ★폐루프: 추정 base로 제어(추정기 품질 검증)
#   GUI:  cd ../../quad && python teleop_gui_biped.py   (같은 채널, 추정오차 표시)

# 실배포 (HardwareInterface 구현 후) — base는 추정기가 센서만으로 복원
python biped_deploy.py --backend hw
```
GUI **Off 전원** = `enable_motors(False)` = 토크차단(limp) / 실HW=motor disable. 재투입=Stand.
GUI 상태줄 = `추정(leg-odom) 오차: pos …cm  vel …m/s` (GT 대비, biped_deploy 실행 시).

## 구성
| 파일 | 역할 |
|---|---|
| `robot_interface.py` | `LowState`·`LowCmd`(SDK 규약) + **`StateEstimator`(leg-odometry)** + `RobotInterface` + `SimInterface`(동작) + `HardwareInterface`(스텁) |
| `biped_deploy.py` | 배포 루프(read→apply→control→write) + 추정오차 추출. `--backend sim/hw` `--est-ctrl`. GUI JSON 채널 |

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

### C. base 상태추정 (`apply_state`) — ★leg-odometry 구현 완료, 강건화가 실배포 관건
- [x] **leg-odometry `StateEstimator` 구현**(위 섹션). HardwareInterface.apply_state가 센서만으로 base 복원.
- [x] sim 검증(개루프 병행): 속도오차 양호·위치 드리프트 ~12%(절대보정 없음, 실기와 동일).
- [ ] ★실배포 강건화: **IMU accel 융합 EKF**(위치 드리프트↓) + **접촉이벤트 기반 위치보정**(발 착지 시 앵커).
      점발 biped는 순수 leg-odom 폐루프가 ~6s marginal(`--est-ctrl` 실측) → 실기 전 강건화 필수.
- [ ] 접촉 검출 정확도(발 힘센서/추정)가 추정 품질 좌우 — swing 발 오검출 시 속도 튐.

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
