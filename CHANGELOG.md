# CHANGELOG

버전 규칙: **v메이저.마이너** (기존 태그 체계 계승, 최신 태그 v14.5.5).
메이저=큰 능력/구조 변화 · 마이너=기능 증분·수정. 태그는 `git tag -a vX.Y`.

---

## [진행중] q_home LUT — RT-safe 자세 (sim=실배포 동일 아키텍처)

body_h 조절·점프standup 렉 = update_stand_qhome이 높이변경마다 300회 IK를 제어루프서 돌려 per-step 스파이크→실시간 페이싱 밀림. 실로봇선 1kHz 데드라인 위반=저크 위험(더 심각).
**해결: q_home LUT.** 시작 시 높이(0.18~0.55, 5mm격자)별 q_home(발목 포함)·com_ref·foot_hip_off·foot_gz0를 300회 IK로 표화 → RT는 **선형보간만**(IK 없음). 발목=상수·hip/thigh/calf만 높이변화라 보간 정확. self-check: 보간 vs 직접IK 최대오차 **0.0195°**(walk높이 0.50은 격자점=오차0) → walk 회귀 없음(verify 4/4). ★warm-start(발목버그·미수렴)와 달리 값이 cold와 동일. C++·Python 완전 parity. LUT_CHECK env=self-check. 실로봇 RT 아키텍처와 동일(무거운 IK를 루프서 제거).

## [진행중] crocoddyl C++ 실시간 OCP (v16 목표)

점프를 offline replay → **C++ live-solve**로. 로드맵=[MPC/RL 리포트 §9](docs/MPC_RL_하이브리드_전략_리포트.md).
- **S0 (완료)**: crocoddyl C++ 빌드 통합 de-risk. `cpp/ocp/build_check.cpp` + CMake `ocp_check` 타겟(crocoddyl 3.2.1=std::shared_ptr, `-lcrocoddyl -lpinocchio_default`). 컴파일·링크·solve 확인.
- **S1 (완료)**: 점프 OCP C++ 포팅(`cpp/ocp/jump_ocp.cpp`). crocoddyl 전체(contact·cost·FDDP·warm-start) → **iter54·cost1.28·apex0.282m Python 완전일치**. (q_crouch/q_stand는 mj_crouch IK 산물 → 임시로 Python DUMP_Q0 로드, IK 포팅은 후속)
- **S2 (완료)**: 실시간 판단. 점프=1회 기동이라 crouch 구간(450ms) 안에 solve만 끝나면 됨. iter 스윕: iter1=76ms·apex0.31(유효점프), iter5~8=~150ms·apex0.282(완전수렴) — **전부 crouch 예산 내 → 점프 live-solve 실시간 충족**. (RTI/호라이즌다이어트는 연속제어=보행용, 1회 점프엔 불요). argv[3]=maxit·chrono 계측.
- **S3-a (완료)**: C++ OCP self-contained화. mj_crouch(MuJoCo IK)+mj2pin을 C++ 포팅 → Python DUMP_Q0 제거, 순수 C++로 crouch/stand 계산+OCP solve. 결과 iter54·cost1.284·apex0.282m 여전히 완전 일치. jump_ocp 타겟에 MuJoCo 링크.
- **S3-b (다음)**: 배포 통합 — jump_ocp를 호출가능 함수로 리팩터→trot_view에 crocoddyl 링크→점프모드가 jump-press 시 live-solve(crouch중)→신선 궤적 실행. 또는 on-demand 생성(별도 solve→/tmp→기존 replay).

### B/C C++ 포팅 판단 (데이터 측정, 2026-07-10)
사용자 질문: A(MPC+WBIC)가 C++로 대배수 빨라졌듯 B/C·시뮬도 C++면 빨라져 고속보행 되나?
- **측정(B centroidal, pixi env)**: `mpc.iterate` OCP solve = **40ms 평균(warmup포함)·steady~20ms = 25Hz 상한**, per-step cycle ~24ms → **solve(C++ aligator)가 지배, glue 수 ms**.
- **결론: 아니오.** A는 계산이 **Python**이라 C++화로 대배수. B/C는 무거운 계산(OCP solve)이 **이미 C++**이고 그게 병목(40ms) → 드라이버 C++화는 glue(수ms)만 제거, 25→~30Hz. 시뮬(MuJoCo)도 이미 C. **고속보행 한계=언어 아닌 OCP수렴(§9 모델계층화)**, 고속은 A(C++·~2m/s)가 담당. B/C 가치=저속 정밀 접촉기동.

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
