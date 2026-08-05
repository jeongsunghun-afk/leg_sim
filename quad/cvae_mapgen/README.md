# cvae_mapgen — Raibo2025 competitive CVAE map generator (독립 프로토타입)

Raibo2025 = **Kim/Hwangbo, "High-speed control and navigation for quadrupedal robots on complex and discrete terrain"** (KAIST RAI Lab, arXiv **2506.02835**, 2025.6). 그 논문의 **map generator**(경쟁적 CVAE 지형 커리큘럼)를 프레임워크 독립 PyTorch로 구현한 프로토타입.

## 무엇을 하나
지형 난이도를 **손수 짠 커리큘럼(lane)** 대신 **학습된 생성모델**이 자동 조절. tracker(보행 정책)가 강해질수록 CVAE가 "성공 가능한 지형 분포"를 학습하고 α로 확장해 **프론티어를 물리적 가능영역 안에서 밀어붙임**(r 0.4→1.6m·x_tilt→90° 벽주행). 논문서 이 방식이 고차원 지형 파라미터에서 기존 커리큘럼보다 우수.

## 아키텍처 (논문 Fig.2/6)
- **ψ (6D)** = 각 디딤돌의 이전 대비 상대 포즈 `[r, φ, θ, x_tilt, y_tilt, h]`. (정확 6성분=논문 supplementary "Components of psi"; 여기선 대표값+물리 가능영역 `PSI_LO/HI`.)
- **CVAE** (encoder/decoder MLP):
  - 조건 `y = [직전 ψ, 직전2 ψ, T_last]` (T_last=다음이 마지막 타깃인지).
  - encoder: `(ψ, y) → (μ, logvar)` / **decoder(=map generator): `(z, y) → ψ`** — 생성 시 디코더만.
  - 생성: `z ~ N(0, α·I)`, **α = 난이도(분산) knob**. 물리 가능영역으로 clamp.
  - 손실 = reconstruction(MSE) + KL.
- **경쟁적 커리큘럼**(`CompetitiveCurriculum`):
  1. 현 CVAE+α로 지형(ψ 시퀀스, 자기회귀 y=직전2ψ) 생성.
  2. tracker 시도 → **성공한 (ψ,y)** 수집(순차: 한 번 실패하면 이후 못 감).
  3. 에피소드당 넘은 디딤돌 평균 > **9.3/10**이면 → 성공 버퍼로 CVAE **재학습** + **α 확대**(프론티어 밀기).

## 실행 (self-test)
```bash
python cvae_mapgen.py      # torch 환경(예: GPU 서버 isaac-5.1)
```
MockTracker(라운드마다 skill 성장)와 경쟁 루프를 돌려 **생성 ψ의 프론티어(r_max→~1.6·x_tilt_max→~90°)가 확장**되는지 확인 = 논문 Fig.6C 정성 재현. 실 tracker 없이 생성기 동역학만 격리 검증.

## RobotSW_IsaacLab (DTC P3) 통합 인터페이스
★이 프로토타입은 **A 워크스트림(RobotSW_IsaacLab) 충돌 회피**를 위해 독립 구현. 통합 시 3곳만 연결:
1. **MockTracker → 실 tracker 성공신호**: `step_round`의 `tracker.attempt(psi)`를 실제 에피소드 결과(디딤돌 넘음 여부)로. 성공률 집계는 env가 이미 가진 정보(현 DTC의 `foothold`/terrain_level 로직 재사용).
2. **ψ → 지형 빌더**: `generate()`가 낸 ψ(디딤돌 상대포즈)를 실제 stepping-stone 월드 배치로 변환. 현 `quad17_env`의 `_build_gap_terrain`/stepping 빌더 자리에 ψ-기반 배치 삽입. (현 10-lane 손수 커리큘럼을 대체/증강.)
3. **재학습 트리거**: 현 terrain_level 승강(promote>0.8) 대신(또는 함께) 9.3/10 기준으로 CVAE 재학습 호출.

현 DTC 상태(리포트): 손수 10-lane 커리큘럼으로 terrain_level frontier ~6-7(≈13-17cm gap) 달성·level9(20cm) 미달=kinematic/gait 한계. **CVAE map generator는 이 프론티어를 학습적으로 밀 후보**(논문서 1.6m·90° 달성). 단 우리 로봇 물리한계 안에서만 확장됨(가능영역 clamp).

## 상태 / 한계 (정직)
- 프로토타입 = CVAE + 경쟁루프 + mock 검증. **실 tracker·실 지형 빌더 미연결**(통합은 A/RobotSW_IsaacLab).
- ψ 6성분 정확 정의는 논문 supplementary 확인 + 우리 지형 파라미터로 재매핑 필요(현재 대표 6D).
- α 스케줄·KL β·재학습 빈도는 튜닝 대상(논문 algorithm S1 상세는 supplementary).
- 물리 가능영역(`PSI_LO/HI`)은 우리 17-DOF 로봇 실측 reach/gait로 조정 필요(현재 논문값 근사).
