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
vx/vy/wz·몸통높이 슬라이더 · **Stand / Walk / RESET**. env=`proxddp`(런처 자동).
Python·C++ 어느 컨트롤러든 **같은 GUI·JSON 채널**(/tmp/biped_cmd.json)로 조종.

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
| `../quad/teleop_gui_biped.py` | 슬림 GUI (dearpygui, 듀얼스틱) |
| `run_gui_biped.sh`·`run_gui_cpp.sh`·`record_biped.sh` | 원샷 런처 |

**C++** (`cpp/`, 배포용 — 성숙 quad C++과 동일 구조·파리티 검증)
| 파일 | 역할 |
|---|---|
| `cpp/src/biped_mpc.hpp` | Di Carlo SRBD MPC 2발 (파리티 3e-11) |
| `cpp/src/biped_wbic.hpp` | 단일 전신 QP WBIC (파리티 6e-12) |
| `cpp/src/biped_control.hpp` | 통합 컨트롤러 (게이트+발배치+MPC+WBIC+GEARBOX) |
| `cpp/src/biped_sim.cpp` | 헤드리스 sim · `biped_view.cpp` GLFW 뷰어+GUI연동 |
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
