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
- 속도: stance 발 정지 가정 → `v_base = −(ω×R·p_foot + R·v_foot)` 평균 + 저역통과(α).
- **높이 z: 접촉발이 지면(z=0)에 있다는 사실로 직접 측정**(`base_z = −pfw_z`) = **드리프트 없음**. ★폐루프 안정의 핵심.
- 수평 xy: 속도 적분(드리프트 허용 — 컨트롤러가 상대량만 써서 무해). (선택 `k_anchor`=접촉앵커 보정)
- 자세=IMU quat 직접, 각속도=gyro 직접. `contact_height=True`(기본), 평지 가정(`ground_z`).

## ★★결정적 발견 — 폐루프 성립의 열쇠 = 접촉 기반 **높이(z)** 추정
`--est-ctrl`(추정 base로 제어, 물리는 GT) 진단 스윕. **어느 추정 성분이 폐루프를 죽이나** 분리:

| 주입(20s walk 0.15) | 낙상 | 해석 |
|---|---|---|
| 완벽추정(=GT) | 0 | harness·플랜트 정상 |
| 위치 GT · **속도=추정** | 0 | ★속도 추정은 문제 없음(스파이크 무관) |
| 속도 GT · **위치=추정** | 144 | ★위치 추정이 killer |
| 위치=추정 · **z만 GT** | 0 | ★★killer의 정체 = **높이(z)** (xy는 167cm 드리프트해도 무해) |

**진단**: leg-odom **위치 적분은 z(높이)가 드리프트**하는데, z가 틀리면 WBIC 높이 task가 즉시 무너짐.
xy 드리프트는 컨트롤러가 상대량만 써서 무해(수 m 드리프트해도 생존). **해결 = z를 적분하지 말고 접촉으로 직접 측정**:
접촉발이 지면(z=0)에 있다는 사실 → `base_z = −(발_base기준_z)`, **드리프트 없음**(발이 땅에 붙어 절대 관측 가능).

**검증(접촉높이 on, 실제 전체 추정기, GT 없음)**:

| 모드 | 접촉높이 OFF | **접촉높이 ON** |
|---|---|---|
| walk 0.15 / 0.20 (40s) | 95 낙상 | **0** (xy 140cm 드리프트 무해) |
| stance (40s) | 다수 | **0** |
| 후진 −0.15 (40s) | — | 63 (잔여, 후진은 최약 모드·GT/완벽추정은 0) |
| **전진 + 센서노이즈**(σ 실측~4배, 30s) | — | **0** (깨끗함 덕 아님=진짜 강건, `NOISE=1~4`) |

**정정된 결론**: 점발 biped 폐루프는 **비성립이 아님**. **접촉 기반 높이 추정 하나로 전진·정지 폐루프가 0낙상**(고전 MPC+추정기만으로 배포 가능). 이전의 "RL 필수" 판정은 **z 관측 누락**이 원인이었음.
→ **전진 locomotion은 고전 스택으로 실배포 가능**. RL은 여전히 선회·측방·후진·외란 강건성 담당(가치 유지). (quad는 4발이라 애초에 z가 여러 발로 잘 잡혀 이 이슈 없었음.) 상세=메모리 biped-mpc-reimpl.

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

### C. base 상태추정 (`apply_state`) — ★leg-odometry + 접촉높이, 전진 폐루프 성립
- [x] **leg-odometry `StateEstimator` 구현**(위 섹션). HardwareInterface.apply_state가 센서만으로 base 복원.
- [x] IMU 센서(MJCF gyro/accel/quat) + 개루프 관찰자 + `--est-ctrl` 폐루프 진단 스윕.
- [x] **★접촉 기반 높이 추정 → 전진·정지 폐루프 0낙상/40s**(고전 스택 배포 가능). killer=z 드리프트로 규명·해결.
- [ ] 후진(-0.15)·선회·측방·외란 강건성 = 잔여 → **RL 정책 담당**(ref_lib 핸드오프). 후진은 최약 모드.
- [ ] **평지 가정(ground_z=0)** — 험지 배포 시 지형 높이맵 or 발별 접촉 z 추정 필요.
- [x] 접촉 검출 정확도(발 힘센서/추정)가 추정 품질 좌우 — swing 발 오검출 시 튐(실배포 시 힘센서 권장).

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
