# 02_Leg 문서 인덱스 — 여기만 보면 전부 찾음

흩어진 문서·스크립트·메모리를 이 한 곳에서 매핑. **컨트롤러 배포 = 17-DOF (C++/Python)**.

## 📄 문서 (simulation/docs/)
| 문서 | 역할 | 형식 |
|---|---|---|
| **[RUN.md](RUN.md)** | 실행 레시피(빌드·GUI·헤드리스·지형·회귀·데모) | md |
| **[MAINTENANCE.md](MAINTENANCE.md)** | 품질 프로세스(하네스·스킬·파리티·문서동기화 규칙) | md |
| **[params.html](params.html)** | 파라미터 값 **전체** + Python↔C++ 파리티 + 튕김조절 | 아티팩트 |
| **[pipeline.html](pipeline.html)** | **A** — MPC+WBIC 파이프라인·솔버 문제정의·I/O 변수(배포) | 아티팩트 |
| **[pipeline_bc.html](pipeline_bc.html)** | **B·C** — Kinodynamics OCP+TSID / FullDynamics OCP+Riccati 파이프라인(연구) | 아티팩트 |
| **[pipeline_rl.html](pipeline_rl.html)** | **RL** — RGA `RobotSW_IsaacLab` 파쿠르 정책(RMA MLP teacher-student, 비순환) 파이프라인 + A/B·C(MPC/OCP) 대조 | 아티팩트 |
| **[pipeline_ci_mpc.html](pipeline_ci_mpc.html)** | **CI-MPC** — 고정스케줄 한계·후보3안(§15)·모델구조·입출력·비용·샘플링 CI(MJX/GPU) 파이프라인·진행(연구, `quad/ci_mpc/`) | 아티팩트 |
| **[datasheet_load_17dof.html](datasheet_load_17dof.html)** | 관절별 τ·ω peak/RMS 부하 데이터시트(trot·walk, 실모터 한계 대비) | 아티팩트 |
| **[sim2real_checklist_17dof.html](sim2real_checklist_17dof.html)** | 실기 이식 갭(액추에이터 물리·미모델·운용) | 아티팩트 |
| **[biped.html](biped.html)** | **biped** — 2족 MPC+WBIC. 점발(동적보행)/평발(정적서기) 접촉모드·통합 1점/2점 전환·상태·개발여정·평발보행 진행(C++ 배포) | 아티팩트 |
| [RECORDING.md](RECORDING.md) | 화면 녹화 가이드(NVENC/OBS) | md |
| [DEVLOG.md](DEVLOG.md) | 개발일지 + 참조 메모(연대기) | md |

문서 원칙: **값은 params에만**, 나머지는 포인터로 위임(중복 금지). params=값 / pipeline=수식·구조 / sim2real=실기 갭.

## 🔧 코드·도구 (simulation/)
| 위치 | 내용 |
|---|---|
| `quad/quad_mpc_wbic_17dof.py` + `teleop_gui_17dof.py` | **배포 A′(Python)** |
| `quad/cpp/` (quad_control·mpc·trot_controller·trot_sim·trot_view) | **배포 C++(1kHz)** |
| `quad/run_gui.sh` · `run_gui_py.sh` | **원샷 런처**(뷰어+GUI, 기본=종합코스). C++ / Python |
| `quad/gen_jump.sh` | 점프 궤적 생성(J1 OCP→J2 변환→/tmp/jump_traj.txt) |
| `quad/cpp/verify.sh` | **하네스** — 표준 회귀 배터리(`./verify.sh [--python]`) |
| `quad/PARAMS.md` | 파라미터 원본(md, = params.html 소스) |
| `quad/make_terrains.py` + `quad_terrain_*.mjcf` | 테스트 지형(3레인 course·계단·험지·마찰·gap·stepping·soft) |
| `quad/record_demo.sh` | 데모 녹화 |
| `.claude/skills/controller-review/SKILL.md` | **스킬** — 검수 SOP(`/controller-review`) |
| `quad/offline/jump/` · `offline/getup/` | 오프라인 궤적(점프 OCP·기립 gather) 생성 파이프라인 |
| `quad/tools/` | 모델 빌드(build_real_quad_17dof·gen_sphere_17dof·plot_perfoot) |
| `quad/research/quad_mpc_wbic.py` + `teleop_gui.py` | 14-DOF(구 본선, 연구) |
| `biped/` | **뒷다리 2족 보행**(MPC+WBIC event-DCM·base-frame·GUI). Python+**C++ 배포**(cpp/, 파리티 1e-11). 실행 `biped/run_gui_biped.sh`(Py)·`run_gui_cpp.sh`(C++). 상세=`biped/README.md`·메모리 biped-mpc-reimpl |
| `simple_mpc/` | 구조 B/C(FullDynamics·Centroidal, marginal 연구) |
| `gait_sim/` | ★구 gait_sim 연구노트(v13/v14 다수 md) — 히스토리, 아카이브 후보 |

## 🧠 메모리 (~/.claude/.../memory) — 프로젝트 방향·진단
`MEMORY.md`(인덱스) + biped-wbic-mpc-project(메인) · 02leg-motor-spec · joint-load-trot-walk · quad-abc-io-structure · haunch-sit-posture · getup-trajopt-wip · rbq-sdk-reference · no-unphysical-sim2real-hacks · sim2real-checklist(·17dof).
→ sim2real 체크리스트 **값·표는 위 HTML이 정본**, 메모리는 결정·통찰만(중복 슬림).

## 🗺️ 로드맵 / 향후 기능
| 문서 | 내용 | 상태 |
|---|---|---|
| **[MPC_RL_하이브리드_전략_리포트.md](MPC_RL_하이브리드_전략_리포트.md)** | MPC/RL 하이브리드 5패턴(A=MPC교사·B=계층·C=주입·D=안전필터·E=샘플링prior) + R.pet 로드맵(H0~H3) + **§9 crocoddyl/aligator 실시간화**(RTI·호라이즌다이어트·모델계층화) + 6-DoF머리. ★**다음단계=crocoddyl C++ 실시간 OCP**의 전략 근거. | 전략·설계 |
| **[roadmap_hybrid.html](roadmap_hybrid.html)** | 위 리포트의 **실행판** — RGA `RobotSW_IsaacLab` 실측 반영. 패턴별 적용성·sim2sim 왕복 파이프라인·Phase 0~4 우선순위(Go2로 브리지 검증→R_Skeleton 하이브리드). R_Skeleton=velocity-command·RMA 확인. | 아티팩트 |
| **[RPET_HEAD_GAZE_MPC.md](RPET_HEAD_GAZE_MPC.md)** | 6-DoF 머리(목) 체인 추가 → 예측형 시선 안정화·반작용 질량 협조·시선/균형 중재. 단계 **G0(23-DoF 모델)→G1(반응층 기준선)→G2(WBIC gaze task)→G3(예측 ff, lite/full)→G4(반작용 질량)→G5(중재)**. 착수순서 G0→G1→G2(기존 스택 3주 데모). | 설계완료·미착수 |
| RPET_ALIGATOR_MPC.md | (위 문서가 참조하는 Phase 0–3 로드맵 — aligator OCP) | 참조·아직 없음 |

**★ head(머리·시선) 기능을 TODO에 추가할 땐 위 [RPET_HEAD_GAZE_MPC.md](RPET_HEAD_GAZE_MPC.md)를 정본으로 참조** — 모델 확장(23-DoF)·gaze 잔차(z_C×u)·WBIC task 삽입 위치·검증 지표·함정이 이미 정리됨. 기존 스택(WBIC 각운동량 task·sit_pitch 자세목표 주입)의 직접 확장이라 G2까지는 재사용률 높음.

## ✅ 유지 워크플로 (요약)
1. 변경 → `cd quad/cpp && ./verify.sh` (회귀 PASS 확인).
2. 파라미터/게인 바꿨으면 `--python`으로 파리티, params 문서 갱신.
3. 스킬 `/controller-review`가 이 절차를 자동 수행(MAINTENANCE.md 기준).
4. 논리단위 커밋(요청 시). 상세=[MAINTENANCE.md](MAINTENANCE.md).

## 🗑️ 정리됨 (2026-07-09)
- 삭제: `RBQGUI-x86_64.AppImage`(167M)·`squashfs-root/`(448M) = RBQ GUI 앱·압축해제본(써드파티, 재다운로드 가능).
- 상위 흩어진 노트(`실행코드`·`obs.md`·`개발일지`·`메모`) → 이 docs/로 통합.
