# biped — 뒷다리 2족 보행 (MPC + WBIC, event-DCM)

quad 17-DOF 뒷다리(HL/HR)를 추출한 8-DOF biped를 **성숙 MPC+WBIC 스택**으로 제어.
점 발 동적 균형(capture-point)·event-based DCM 게이트·base-frame 발배치·heading-hold.
**현황**: 저속 양방향(±0.25 m/s) 25s+ 무낙상 직진 + gentle 선회 + 실시간 GUI 조종.
**C++ 배포 완료**(Python 파리티 1e-11, C++ 뷰어도 같은 GUI로 조종).

## 실행

```bash
# ── Python (개발·튜닝) ──
./run_gui_biped.sh               # 뷰어 + GUI 동시 (★사용자 터미널에서)
python3 biped_view.py            # 단독 뷰어 (VX=0.15 로 보행)
VIEW=0 T=25 python3 biped_run.py # 헤드리스 검증

# ── C++ (배포·실시간) ──
./run_gui_cpp.sh                 # C++ 뷰어(biped_view) + 같은 GUI
cd cpp && ./build/biped_sim ../biped_from_quad.mjcf 0.15 15   # 헤드리스 폐루프

# ── 녹화 ──
./record_biped.sh               # GUI + OBS
```
**GUI**(teleop_gui_biped): **좌스틱=전후(vx)/측방(vy) · 우스틱=선회(wz)** (우클릭 고정) ·
vx/vy/wz·몸통높이 슬라이더 · 버튼 **RESET / Off 전원 / Stand 서기 / Walk 이동**(17-DOF 순서·테마).
**Off 전원**=모터 토크차단(limp, 실HW=motor disable)·재투입=Stand. env=`proxddp`(런처 자동).
Python·C++ 어느 컨트롤러든 **같은 GUI·JSON 채널**(/tmp/biped_cmd.json)로 조종.

## 실배포 (실모터 HW) — `deploy/`
컨트롤러는 순수 토크 출력 → 플랜트만 교체(`RobotInterface`: sim↔실모터). `python deploy/biped_deploy.py --backend sim|hw`.
**★상태추정기(leg-odometry, 17-DOF와 동일 계보)**: 센서만(관절 q/dq + IMU + 접촉)으로 base pose/vel 복원 →
HW는 **센서만 붙이면 full state**. GT 대비 오차 추출(sim 검증·GUI 표시). `--est-ctrl`=폐루프 검증.
`deploy/robot_interface.py`(LowCmd/LowState·StateEstimator)·`biped_deploy.py`(배포 루프)·`README.md`(A~E 체크리스트). 상세=`deploy/README.md`.

## 문서 안내 — **무엇을 하려는지로 고른다**

| 하려는 것 | 문서 | 줄 |
|---|---|---|
| 실기를 **지금 켠다** | [`emb/RUNBOOK.md`](emb/RUNBOOK.md) — 터미널별 실행 순서 | 452 |
| 실기 **인터페이스**를 이해한다 | [`emb/README.md`](emb/README.md) — SHM↔제어기 배선·ModeFSM | 103 |
| **액추에이터 값**을 MJCF 에 넣는다 | [`emb/pace/RESULTS.md`](emb/pace/RESULTS.md) — **최종 파라미터·신뢰도·근거** | 856 |
| 그 값을 **어떻게 쟀는지** 본다 | [`emb/pace/README.md`](emb/pace/README.md) — 측정 절차·도구 | 181 |
| **RL 정책**을 실기에 올린다 | [`emb/RL_INTERFACE.md`](emb/RL_INTERFACE.md) — 각도규약·커플링·한계 | 560 |
| RL **레퍼런스 궤적**을 쓴다 | [`ref_lib/README.md`](ref_lib/README.md) — DTC 교사 핸드오프 | 41 |
| **얼마나 빨리/크게** 걸을 수 있나 | [`cpp/STABILITY_MAP.md`](cpp/STABILITY_MAP.md) — 속도×스텝시간 안정영역 | 421 |
| sim → **실모터**로 옮긴다 | [`deploy/README.md`](deploy/README.md) — 플랜트 교체·상태추정 | 118 |
| **다음 실기 세션**을 준비한다 | [`emb/NEXT_HW.md`](emb/NEXT_HW.md) — 체크리스트(다리 장착 이후) | 964 |
| **IMU 가 0** 이다 | [`emb/IMU_RECOVERY.md`](emb/IMU_RECOVERY.md) — 원인·조치(미실시) | 238 |

**단일 출처 규칙** — 같은 값이 두 문서에 있으면 반드시 갈라진다. 실제로 그랬다:
| 값 | **단일 출처** | 나머지는 |
|---|---|---|
| 액추에이터 파라미터(`armature`·`damping`·`frictionloss`·`delay`·`coef`) | `emb/pace/RESULTS.md` | 인용만 하고 **복사하지 말 것** |
| 마찰·파단토크 실측 | `emb/pace/spec.yaml` 의 `friction.measured_*` | RESULTS 는 그 스냅숏이다 |
| 각도규약·부호·감속비 | `emb/config/biped_emb.yaml` | 문서는 설명만 |

---

## 파일 (활성)
**Python**
| 파일 | 역할 |
|---|---|
| `biped_from_quad.mjcf` | 모델 (quad 뒷다리 + 최소몸통, sphere발, GEARBOX, 바닥강성) |
| `biped_wbic.py` | WBIC(양발 stance) + GEARBOX + 게인·상수 |
| `biped_step.py` | event-DCM 게이트 + **base-frame 발배치**(dcm_target) + swing 궤적 |
| `biped_mpc_wbic.py` | SRBD MPC(2발) + WBIC 추종 + heading-hold + 선회/좌우. **메인** |
| `biped_run.py` | 실행기: 컨트롤러+뷰어 + JSON 명령채널(v/vy/w/body_h/mode) + 상태발행 |
| `biped_view.py` | 단독 뷰어(낙상 자동리셋) · `biped_ref_export.py` RL 레퍼런스 생성 |
| `teleop_gui_biped.py` | 슬림 GUI (dearpygui, 듀얼스틱 + 각축 JOG 패널). ★커밋 7953c5c 에서 `quad/` → `biped/` 로 이동 |
| `run_gui_biped.sh` | sim 원샷 런처(컨트롤러+뷰어+GUI) |
| `run_gui_only.sh` | **GUI 만** — 실기 JOG 검증용(컨트롤러는 `emb/app/biped_emb.py` 따로) |
| `run_gui_cpp.sh`·`record_biped.sh` | 기타 런처 |

**C++** (`cpp/`, 배포용 — 성숙 quad C++과 동일 구조·파리티 검증)
| 파일 | 역할 |
|---|---|
| `cpp/src/biped_mpc.hpp` | Di Carlo SRBD MPC 2발 (파리티 3e-11) |
| `cpp/src/biped_wbic.hpp` | 단일 전신 QP WBIC (파리티 6e-12) |
| `cpp/src/biped_control.hpp` | 통합 컨트롤러 (게이트+발배치+MPC+WBIC+GEARBOX) |
| `cpp/src/state_estimator.hpp` | ★상태추정(leg-odom+접촉높이) — Python deploy 포팅. 배포 폐루프용 |
| `cpp/src/biped_sim.cpp` | 헤드리스 sim (`EST_CTRL=1`=추정 폐루프·falls카운트) · `biped_view.cpp` GLFW 뷰어+GUI연동(`EST_CTRL=1`=배포 폐루프) |
| `cpp/dump_biped_{mpc,wbic}.py`·`*_parity.cpp` | Python 파리티 검증 |
| `cpp/CMakeLists.txt` | 빌드 (eiquadprog+Eigen+mujoco+GLFW, proxddp env) |

하위: `ref_lib/`(RL 레퍼런스+핸드오프) · `renders/`·`archive/`(옛 시도) · `meshes/`·`urdf/`(옛 자산).

## 제어 구조 (17-DOF quad와 동일 백본)
- **MPC**(50Hz, Di Carlo SRBD): 균형 GRF 계획. x_ref 속도=명령(body)·yaw=heading목표(드리프트 보정)·wz 선회.
- **WBIC**(500Hz): MPC GRF 추종 + 스윙발 + roll/pitch 레벨링 + posture. GEARBOX(반사관성).
- **게이트**: event-based(측방 DCM이 swing측 넘을 때 착지). 점 발이라 고정시간보다 안전.
- **발배치**: base-frame capture-point(yaw 변환=다리 꼬임 해결) + SPREAD 벌림 + 스탠스 최소간격.

**한계**(점 발 근본): 빠른 선회·좌우이동·무한 균형은 marginal → RL 몫. 전진/후진+gentle선회는 로버스트.

상세 개발기록·튜닝 = 메모리 `biped-mpc-reimpl` · 전략 = `../docs/roadmap_hybrid.html` · RL 레퍼런스 = `ref_lib/README.md`.
