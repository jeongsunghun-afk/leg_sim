# 02_Leg 문서 인덱스 — 여기만 보면 전부 찾음

흩어진 문서·스크립트·메모리를 이 한 곳에서 매핑. **컨트롤러 배포 = 17-DOF (C++/Python)**.

## 📄 문서 (simulation/docs/)

### 전체 파이프라인
| 문서 | 역할 | 형식 |
|---|---|---|
| **[pipeline_fullstack.html](pipeline_fullstack.html)** | **전체 시스템 프레임워크** — 형태→기능(기구 4요소)·4계보(CI-MPC·DTC·APT·AMP) 통합 배포 아키텍처·지각/SLAM/계획/제어 층구조(F1–F3)·**실제 코드 대조·수정사항(F4–F5, DOF R¹⁶·다중 repo 통합)**·**부록: nav 통합·RL 정책 구조(구 pipeline_nav·pipeline_rl 흡수, F6–F7)** | 아티팩트 |
| **[rl_module_train.html](rl_module_train.html)** | **RL 모듈 구축·학습 가이드** — 제어층 정책을 **어떻게 만들고 학습하는가**. 두 참조골격(DTC 계획↔제어·APT-RL 3단계)·학습 3단계(표현/RL+RMA/지각증류)·**모듈 카탈로그(ActorCriticRMA: actor·estimator·history/priv_encoder·critic — 배포 vs 학습전용)**·**보상설계 실증(2점 평발 plateau→shuffle→limp 3재균형)**·예정 Depth CNN+GRU·PACE 정합. fullstack이 개요면 이건 **학습법 심화** | 아티팩트 |

### 제어기별 (pipeline · params)
| 제어기 | pipeline | params | 비고 |
|---|---|---|---|
| **A** (MPC+WBIC, **배포**) | [pipeline_a.html](pipeline_a.html) | [params_a.html](params_a.html) | 솔버 문제정의·I/O·값전체+파리티+튕김조절 |
| **B·C·D1** (OCP/NMPC, 연구·분석) | [pipeline_bcd1.html](pipeline_bcd1.html) | (pipeline 내) | Kinodyn OCP+TSID / FullDyn OCP+Riccati / OCS2 NMPC — 각 **구조·입출력** |
| **제어기 비교** (A·B·C·D1·CI) | [제어기_비교.html](제어기_비교.html) | — | 5종 **구조·솔버·I/O·실시간·지형·성능** 대조 + 지형유형 정직비교 |
| **CI** (Contact-Implicit, **종결**) | [pipeline_ci.html](pipeline_ci.html) · [CI-MPC_개발_리포트.md](CI-MPC_개발_리포트.md) | [params_ci.html](params_ci.html) | 종결(§11): 로버스트 험지 부적합 확정. 완성분=해석그래디언트·자세(서기/눕기/앉기). 상세=리포트 |

### 실행 · 품질 · 기타
| 문서 | 역할 | 형식 |
|---|---|---|
| **[RUN.md](RUN.md)** | 실행 레시피(빌드·GUI·헤드리스·지형·회귀·데모) | md |
| **[MAINTENANCE.md](MAINTENANCE.md)** | 품질 프로세스(하네스·스킬·파리티·문서동기화 규칙) | md |
| **[TOWR_모델기반지형_리포트.md](TOWR_모델기반지형_리포트.md)** | **TOWR 트랙** — 오프라인 지형 planning·추종 한계·폐루프 조사 결론(모델기반 상한→RL) | md |
| **[datasheet_load_17dof.html](datasheet_load_17dof.html)** | 관절별 τ·ω peak/RMS 부하 데이터시트(trot·walk, 실모터 한계 대비) | 아티팩트 |
| **[sim2real_checklist_17dof.html](sim2real_checklist_17dof.html)** | 실기 이식 갭(액추에이터 물리·미모델·운용) | 아티팩트 |
| **[biped.html](biped.html)** | **biped** — 2족 MPC+WBIC. 점발/평발 접촉모드·1점/2점 전환·개발여정(C++ 배포) | 아티팩트 |
| [RECORDING.md](RECORDING.md) · [DEVLOG.md](DEVLOG.md) | 화면 녹화 가이드 · 개발일지(연대기) | md |

문서 원칙: **값은 params에만**, 나머지는 포인터로 위임(중복 금지). params=값 / pipeline=수식·구조 / sim2real=실기 갭. nav·RL은 fullstack으로 통합(개별 pipeline_nav·pipeline_rl 폐지).

## 🔧 코드·도구 (simulation/)
| 위치 | 내용 |
|---|---|
| `quad/quad_mpc_wbic_17dof.py` + `teleop_gui_17dof.py` | **배포 A′(Python)** |
| `quad/cpp/` (quad_control·mpc·trot_controller·trot_sim·trot_view) | **배포 C++(1kHz)** |
| `quad/run_gui.sh` · `run_gui_py.sh` | **원샷 런처**(뷰어+GUI, 기본=종합코스). C++ / Python |
| `quad/gen_jump.sh` | 점프 궤적 생성(J1 OCP→J2 변환→/tmp/jump_traj.txt) |
| `quad/cpp/verify.sh` | **하네스** — 표준 회귀 배터리(`./verify.sh [--python]`) |
| `quad/PARAMS.md` | 파라미터 원본(md, = params_a.html 소스) |
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
| **[TAMOLS_online_tracking_fix.md](TAMOLS_online_tracking_fix.md)** | TAMOLS whole-body 추종 안정화(P0~P3). ④lateral=W_AM·⑤online=re-anchor frame. GM-observer·swing commitment. | 진행·진단완료 |
| **[WBC_layer2_standalone_fix.md](WBC_layer2_standalone_fix.md)** | 2층(TAMOLS→WBC) baseline. standalone WBC z침하=QP 힘품질. 논문(계층적 WBC·GM-observer) 해법. "최소 baseline→RL" 전략. | 진행·게이트 |
| **[D1_OCS2_porting.md](D1_OCS2_porting.md)** | ★**현재 트랙** — OCS2 통합 perceptive NMPC를 02_Leg에 포팅(강건 모델기반). OCS2 클론됨(ros2, ocs2_perceptive·legged_robot·python_interface). 관문=Jazzy/Humble 빌드. | 착수·빌드 |
| **[pipeline_tamols.html](pipeline_tamols.html)** | TAMOLS 계획층(A 실행스택 위 ③④ 주입)·TAM_BASE base-발판 협조 실증·한계·D1포팅/DTC. | 아티팩트 |
| **[B_elevation_perceptive_NMPC.md](B_elevation_perceptive_NMPC.md)** | B→Grandia식 통합 perceptive NMPC 계획. ★**보류**(D1/OCS2 채택, B는 20Hz 실시간 열세). | 보류 |
| **[MPC_RL_하이브리드_전략_리포트.md](MPC_RL_하이브리드_전략_리포트.md)** | MPC/RL 하이브리드 5패턴(A=MPC교사·B=계층·C=주입·D=안전필터·E=샘플링prior) + R.pet 로드맵(H0~H3) + **§9 crocoddyl/aligator 실시간화**(RTI·호라이즌다이어트·모델계층화) + 6-DoF머리. ★**다음단계=crocoddyl C++ 실시간 OCP**의 전략 근거. | 전략·설계 |
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
- [모델기반 갭크로싱 탐색리포트](모델기반_갭크로싱_탐색리포트.html) — TOWR·TAMOLS·실행3시도·실시간측정 종합, C++ 착수 근거
