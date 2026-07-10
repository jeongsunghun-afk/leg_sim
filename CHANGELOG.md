# CHANGELOG

버전 규칙: **v메이저.마이너** (기존 태그 체계 계승, 최신 태그 v14.5.5).
메이저=큰 능력/구조 변화 · 마이너=기능 증분·수정. 태그는 `git tag -a vX.Y`.

---

## [진행중] crocoddyl C++ 실시간 OCP (v16 목표)

점프를 offline replay → **C++ live-solve**로. 로드맵=[MPC/RL 리포트 §9](docs/MPC_RL_하이브리드_전략_리포트.md).
- **S0 (완료)**: crocoddyl C++ 빌드 통합 de-risk. `cpp/ocp/build_check.cpp` + CMake `ocp_check` 타겟(crocoddyl 3.2.1=std::shared_ptr, `-lcrocoddyl -lpinocchio_default`). 컴파일·링크·solve 확인.
- **S1 (다음)**: 점프 OCP C++ 포팅(`offline/jump/jump_ocp.py` → C++), Python parity.
- **S2**: §9 실시간화(RTI 1~3이터+warm-start·호라이즌 다이어트).
- **S3**: 배포 통합(점프모드 live-solve).

---

## v15.0 — 17-DOF 배포 완성 baseline (2026-07-10)

최신 태그 v14.5.5(2026-05-18) 이후 272 커밋을 정리한 **17-DOF 실배포 마일스톤**. 여기서부터 버전관리 재개.

### 배포 (deploy)
- **17-DOF** (허리 능동 + sphere발, 37.9kg): C++ 1kHz 실시간(`quad/cpp/`) + Python 레퍼런스(`quad_mpc_wbic_17dof.py`). 게인 자동감지.
- 물리: convex SRBD MPC(GRF) + WBIC 단일 QP + 반응형 Raibert 발배치.

### 보행 (locomotion)
- gait 프리셋: **walk / trot / run / stairs** (GUI 토글, 속도·발높이·base height 자동세팅).
- 자동차식 **허리 조향**(Ackermann, yaw-rate 캡), 제자리 스핀 공존.
- **perceptive 지형적응**: mj_ray 단일레이(group2)로 착지높이·몸통높이 적응. 높이천장 없음.
- 서기=멈춘 위치 홀드(홈으로 안 빨려감), 서기 0.52.

### 점프 (jump)
- **offline OCP**(crocoddyl, Python `offline/jump/`) → `/tmp/jump_traj.txt` → **C++ 추종 재생** + WBIC 착지.
- 전방 점프(JUMP_VX, 기본 0.6=~0.14m 전방) / 수직 선택.

### 자세 (posture)
- 서기 / 눕기(wbic 저크라우치) / 앉기(haunch sit) / 기립(gather 궤적, 튕김 완화 KP80).

### 지형 (terrain)
- **3레인 종합코스**(좌우±2.4m): 마찰→소프트 / 험지→계단 / 갭→스테핑.
- 개별 씬: friction(μ0.5/0.3/0.1 priority)·soft(매트리스)·gap·stepping·stairs·rough. 회귀용 verify.

### 도구·구조
- **원샷 런처**: `run_gui.sh`(C++)·`run_gui_py.sh`(Python), 기본=종합코스. 녹화 `record_demo.sh`.
- **mjcf 일원화** → `quad/mjcf/`. 스크립트 폴더 정리: `offline/{jump,getup}`·`tools`·`research`.
- 회귀 하네스 `cpp/verify.sh` (walk/trot/run + course, falls=0). 문서 `docs/`.

### 다음 (roadmap)
- **crocoddyl C++ 실시간 OCP**: 점프를 offline replay가 아닌 C++ live-solve로 전환 검토.
- head/gaze(RPET_HEAD_GAZE_MPC), 급계단(offline궤적/lookahead), sim2real(실측 로터관성·상태추정).
