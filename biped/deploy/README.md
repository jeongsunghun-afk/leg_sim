# biped 실배포 (sim → 실모터 HW)

컨트롤러(MPC+WBIC)는 **순수 토크**를 출력한다. 배포 = 플랜트만 교체 — `RobotInterface`
백엔드를 `SimInterface`(MuJoCo) → `HardwareInterface`(실모터)로 바꾸면 컨트롤러·게인·GUI 그대로.

```
컨트롤러(MuJoCo=동역학 모델)                RobotInterface (플랜트 경계)
  c.control(dt) → d.ctrl(tau) ──LowCmd──▶  write():  sim=mj_step / HW=setTorqueRef
  M·J·bias 계산  ◀──LowState── apply_state() ◀─read(): sim=d 스냅샷 / HW=인코더+IMU
                                    ▲ base pose/vel = StateEstimator(leg-odometry, 센서만)
```

## ★상태추정기 (leg-odometry) — 센서만으로 full state (관찰자로는 유효)
`StateEstimator`(robot_interface.py) = **17-DOF quad와 동일 계보**(state_estimator.hpp Python 포팅) + IMU 센서(MJCF).
절대 base(GPS/모캡) **안 씀** = 실로봇 동일. **센서만**(관절 q/dq + IMU quat/gyro/accel + 발 접촉) → base 위치·선속도 복원.
- 원리: stance 발 정지 가정 → `v_base = −(ω×R·p_foot + R·v_foot)` 평균 + 저역통과(α), 적분→위치.
- 자세=IMU quat 직접, 각속도=gyro 직접. 위치=속도 적분이라 **드리프트 존재**(절대보정 없음, 실기와 동일).
- 개루프(관찰자) 검증: 속도추정 steady 구간 양호(~0.01~0.05 m/s), 위치는 완만 드리프트. **모니터·sim2real 갭 정량·RL 관측 입력엔 유효.**

## ★★결정적 발견 — 점발 biped 폐루프(MPC×추정)는 고전 방식으로 비성립
`--est-ctrl`(추정 base로 제어, 물리는 GT) 20s 스윕 결과, **어떤 튜닝도 폐루프를 못 살림**:

| 설정 | 20s 낙상수 |
|---|---|
| **완벽추정(=GT)** | **0** (harness·컨트롤러 정상 확인) |
| leg-odom α0.2 (최선) | 45 (≈0.4s마다 낙상) |
| leg-odom α0.4 기본 | 98 |
| + IMU accel 융합 | 133 (악화) |
| + 접촉앵커 dwell | 327 (악화) |

**결론**: 점발 biped는 **base 속도추정 오차에 근본 과민** → 고전 leg-odom(+EKF/앵커 시도 포함) 폐루프 제어 **성립 불가**.
이는 튜닝 문제 아님(완벽추정=0 vs 최선=45). **선회·측방이 "RL 몫"인 것과 동일한 점발 근본 취약성.**
→ **실 점발 biped 배포 = RL 정책 필수**(추정 노이즈+점발 불안정을 end-to-end 학습). 고전 스택+추정기 볼트온으론 배포 불가.
(quad 4발은 넓은 지지면이라 같은 추정기로 폐루프 OK — biped와의 결정적 차이.) 상세 기록=메모리 biped-mpc-reimpl.

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

### C. base 상태추정 (`apply_state`) — ★leg-odometry 구현(관찰자 유효), 폐루프는 RL 몫으로 판명
- [x] **leg-odometry `StateEstimator` 구현**(위 섹션). HardwareInterface.apply_state가 센서만으로 base 복원.
- [x] IMU 센서(MJCF gyro/accel/quat) + 개루프 관찰자 검증 + `--est-ctrl` 폐루프 스윕.
- [x] **★판명: 고전 폐루프 비성립**(위 표). 완벽추정=0낙상 vs 최선 leg-odom=45낙상/20s. EKF·앵커·accel 모두 악화.
- [ ] ~~고전 추정기 강건화~~ → **폐루프 강건성은 RL 정책이 담당**(추정 노이즈 흡수 end-to-end). ref_lib 핸드오프 경로.
- [x] 접촉 검출 정확도(발 힘센서/추정)가 추정 품질 좌우 — swing 발 오검출 시 속도 튐(실배포 시 힘센서 권장).

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
