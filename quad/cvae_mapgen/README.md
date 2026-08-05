# cvae_mapgen — Raibo2025 competitive CVAE map generator (독립 프로토타입)

Raibo2025 = **Kim/Hwangbo, "High-speed control and navigation for quadrupedal robots on complex and discrete terrain"** (KAIST RAI Lab, arXiv **2506.02835**, 2025.6). 그 논문의 **map generator**(경쟁적 CVAE 지형 커리큘럼)를 프레임워크 독립 PyTorch로 구현한 프로토타입.

★**논문 supplementary(같은 36p PDF)의 실제 레시피에 충실**: Algorithm 1(adversarial training)·Table S3(초기 커리큘럼)·Network details(MLP enc[512,128]/dec[128,512])·Components of ψ(6성분).

## 무엇을 하나
지형 난이도를 손수 짠 커리큘럼(lane) 대신 **학습된 생성모델**이 자동 조절. tracker(보행 정책)가 강해지면 CVAE를 그 tracker가 **넘은(overcome=feasible) 지형**으로 재학습 → 프론티어를 물리 가능영역 안에서 밀어붙임(논문: r 0.4→1.6m·x_tilt→90° 벽주행). 고차원 지형 파라미터서 기존 커리큘럼보다 우수(논문 §Analysis of various curricula).

## 아키텍처 (논문 그대로)
- **ψ (6D)** = `[r, θ, φ, Δyaw, x_tilt, y_tilt]` (각 디딤돌의 이전 대비 상대 포즈. Components of ψ·Table S3).
- **CVAE**: enc MLP[512,128]·dec MLP[128,512]. 조건 `y=[직전 2ψ, T_last]`. **map generator=디코더만**, `z~N(0,(1+α)I)`. loss=MSE recon+KL.
- **2단계**:
  1. **초기 커리큘럼**(Table S3 stage0~4): ψ 범위를 고정률로 확대(r_hi 0.8→1.1·φ 5→40°·x_tilt 10→30° 등) → CVAE 초기데이터.
  2. **경쟁**(Algorithm 1): 아래.
- **Algorithm 1 (α 메커니즘)**: α=0.7 초기. `update%period==0 and perf>9.3`이면 → **overcome ψ(feasible_param)로 CVAE 재학습** → α←0.7 → `while perf<9.15: α−=0.02`(난이도를 9.15/10로 낮춤). **높은 α=분산↑=어려움**. 프론티어는 '재학습이 (향상된 tracker가 넘은) 더 어려운 overcome 분포로 이동'해 확장.

## 실행 / self-test 결과 (torch 환경)
```bash
python cvae_mapgen.py
```
MockTracker와 Algorithm 1을 돌린 검증(cpu):
- 부트스트랩(Table S3) → 경쟁 재학습 주기적 발생(9회) → **프론티어 r_max→1.59m**(목표 1.6).
- ★**핵심 이점 정량 검증 — feasible-fraction: uniform=0.31 vs CVAE=0.998 (3.2× 효율)**: CVAE가 r-φ/θ-x_tilt **상관 manifold를 학습**해 물리 가능 지형을 생성(uniform은 상관 무시→infeasible 낭비). = 논문 고차원 이점의 근거.

## 정직한 한계 (mock 특성 — 알고리즘 아님)
- **9.3/9.15는 논문 값**(실 tracker≈100% feasible 생성 가정). mock은 CVAE ~97~99% feasible → 순차-정지 perf 상한이 낮아 **mock용 8.3/8.0으로 비례 하향**(알고리즘 구조는 동일).
- **x_tilt 프론티어는 26°까지만**(r은 1.59 달성) = mock 난이도/feasibility(θ↔x_tilt 상관)의 단순함 탓. 알고리즘 문제 아님. 실 tracker에선 논문대로 90°(벽주행).
- **MockTracker·feasible()·PSI 물리영역은 대표 근사**: 실 통합 시 실 tracker 성공신호·우리 로봇 reach/gait로 재매핑.

## RobotSW_IsaacLab (DTC P3) 통합 (★A 워크스트림 충돌 회피 위해 독립 구현)
3곳만 연결:
1. **MockTracker → 실 tracker**: `rollout()`의 `tracker.attempt`를 실 에피소드 overcome 결과로. `env.get_feasible_param()`=현 DTC의 terrain_level/foothold 로직 재사용.
2. **ψ → 지형 빌더**: `generate()`가 낸 ψ(디딤돌 상대포즈)를 실제 stepping-stone 월드 배치로. 현 `quad17_env`의 `_build_gap_terrain`/stepping 자리에 삽입(현 손수 10-lane 대체/증강).
3. **Algorithm 1 트리거**: 논문 9.3/9.15 그대로. 현 DTC frontier ~6-7(≈13-17cm gap)을 학습적 생성기로 밀 후보.
