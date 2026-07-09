# 02_Leg 문서 인덱스 — 여기만 보면 전부 찾음

흩어진 문서·스크립트·메모리를 이 한 곳에서 매핑. **컨트롤러 배포 = 17-DOF (C++/Python)**.

## 📄 문서 (simulation/docs/)
| 문서 | 역할 | 형식 |
|---|---|---|
| **[RUN.md](RUN.md)** | 실행 레시피(빌드·GUI·헤드리스·지형·회귀·데모) | md |
| **[MAINTENANCE.md](MAINTENANCE.md)** | 품질 프로세스(하네스·스킬·파리티·문서동기화 규칙) | md |
| **[params.html](params.html)** | 파라미터 값 **전체** + Python↔C++ 파리티 + 튕김조절 | 아티팩트 |
| **[pipeline.html](pipeline.html)** | MPC+WBIC 파이프라인·솔버 문제정의·I/O 변수 | 아티팩트 |
| **[sim2real_checklist_17dof.html](sim2real_checklist_17dof.html)** | 실기 이식 갭(액추에이터 물리·미모델·운용) | 아티팩트 |
| [RECORDING.md](RECORDING.md) | 화면 녹화 가이드(NVENC/OBS) | md |
| [DEVLOG.md](DEVLOG.md) | 개발일지 + 참조 메모(연대기) | md |

문서 원칙: **값은 params에만**, 나머지는 포인터로 위임(중복 금지). params=값 / pipeline=수식·구조 / sim2real=실기 갭.

## 🔧 코드·도구 (simulation/)
| 위치 | 내용 |
|---|---|
| `quad/quad_mpc_wbic_17dof.py` + `teleop_gui_17dof.py` | **배포 A′(Python)** |
| `quad/cpp/` (quad_control·mpc·trot_controller·trot_sim·trot_view) | **배포 C++(1kHz)** |
| `quad/cpp/verify.sh` | **하네스** — 표준 회귀 배터리(`./verify.sh [--python]`) |
| `quad/PARAMS.md` | 파라미터 원본(md, = params.html 소스) |
| `quad/make_terrains.py` + `quad_terrain_*.mjcf` | 테스트 지형(계단/험지/마찰/course) |
| `quad/record_demo.sh` | 데모 녹화 |
| `.claude/skills/controller-review/SKILL.md` | **스킬** — 검수 SOP(`/controller-review`) |
| `quad/quad_mpc_wbic.py` | 14-DOF(구 본선, 연구) |
| `biped/` | 초기 biped phase(README·mjcf·urdf, 브랜치 biped-wbic-mpc) |
| `simple_mpc/` | 구조 B/C(FullDynamics·Centroidal, marginal 연구) |
| `gait_sim/` | ★구 gait_sim 연구노트(v13/v14 다수 md) — 히스토리, 아카이브 후보 |

## 🧠 메모리 (~/.claude/.../memory) — 프로젝트 방향·진단
`MEMORY.md`(인덱스) + biped-wbic-mpc-project(메인) · 02leg-motor-spec · joint-load-trot-walk · quad-abc-io-structure · haunch-sit-posture · getup-trajopt-wip · rbq-sdk-reference · no-unphysical-sim2real-hacks · sim2real-checklist(·17dof).
→ sim2real 체크리스트 **값·표는 위 HTML이 정본**, 메모리는 결정·통찰만(중복 슬림).

## ✅ 유지 워크플로 (요약)
1. 변경 → `cd quad/cpp && ./verify.sh` (회귀 PASS 확인).
2. 파라미터/게인 바꿨으면 `--python`으로 파리티, params 문서 갱신.
3. 스킬 `/controller-review`가 이 절차를 자동 수행(MAINTENANCE.md 기준).
4. 논리단위 커밋(요청 시). 상세=[MAINTENANCE.md](MAINTENANCE.md).

## 🗑️ 정리됨 (2026-07-09)
- 삭제: `RBQGUI-x86_64.AppImage`(167M)·`squashfs-root/`(448M) = RBQ GUI 앱·압축해제본(써드파티, 재다운로드 가능).
- 상위 흩어진 노트(`실행코드`·`obs.md`·`개발일지`·`메모`) → 이 docs/로 통합.
