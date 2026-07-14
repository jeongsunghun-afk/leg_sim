# biped 레퍼런스 라이브러리 — RL 추종(DTC) 핸드오프

MPC 개발자(모델기반) → RL 담당자 핸드오프. `biped_mpc_wbic`(제자리 균형+0.4m/s 보행)를
DTC 교사로 써서 뽑은 레퍼런스 궤적. **RL이 이를 추종해 모델기반 ~8s 한계를 넘어 로버스트 균형을 학습**한다.

> 생성: `python biped_ref_export.py` · 모델: `../biped_from_quad.mjcf`(CM_HL 뒷다리, 8-DOF, sphere발) · 50Hz

## 파일
| 파일 | 명령 | 길이 | 비고 |
|---|---|---|---|
| `biped_ref_inplace.npz` | vx=0.0 | 6.0s (300f) | 제자리 균형 |
| `biped_ref_walk02.npz` | vx=0.2 | 5.7s (283f) | 전진 1.38m |
| `biped_ref_walk03.npz` | vx=0.3 | 6.0s (300f) | 전진 1.78m (~0.4m/s) |

## npz 키 (프레임별 배열, N×dim)
**full (env 재생 / tracking 보상용)**
- `root_pos`(3) · `root_quat`(4, **wxyz** MuJoCo 규약) · `q`(8) · `dq`(8)
- `foot`(6=2발×xyz world) · `contact`(2, HL/HR) · `tau`(8) · `lam`(6=2발×GRF)
- `t`(s) · `vx`(스칼라) · `dt`(0.02)

**★DTC subset (정책 관측 = "작은 서브셋", 전략리포트 §3 패턴 A)**
- `swing_tgt`(2) — 스윙 발디딤 2D 목표 (우리 capture-point 출력)
- `ref_q`(8) — 목표 관절각 (**DTC ablation: 관측 포함이 수렴 결정타**)
- `cs`(2) — 접촉 스케줄 (event-based gait 출력)

## RL 태스크 설계 (DTC 패턴 A)
`RobotSW_IsaacLab`의 hind_leg / R_Skeleton_amp 태스크에 이식:
- **관측** = proprio(base ang vel·proj gravity·q−q_default·dq·prev_action) + **DTC subset**(swing_tgt·ref_q·cs). 특권/history 인코더(RMA)는 기존 그대로.
- **보상** = tracking: `w_q·‖q−ref_q‖ + w_base·‖base−root_ref‖ + w_foot·‖foot−ref_foot‖ + w_contact·접촉일치` + 생존/평활.
- **관절 순서 주의**: 우리 MJCF = HL[hip,thigh,calf,foot]·HR[...] (8). IsaacLab hind_leg 관절순과 매핑 테이블 필요.
- **인프라 재사용**: R_Skeleton_amp의 `SkeletonMotionLoader`(DeepMimic-txt) 구조에 맞춰 변환하면 reset-from-motion·AMP까지 활용 가능. (현 npz → 그 포맷 변환 스크립트는 RL측 합의 후)

## 왜 이 방식 (오늘 실증)
- 모델기반 biped 균형은 **~7.88s에서 plateau**(점 발·leg-heavy·가변높이). event-based DCM gait로 0.69→7.88s까지 올렸으나 무한 균형은 hand-tuning으로 한계.
- **RL이 흡수할 부분**: 반응형 균형·외란 복구·분포밖 회복. 레퍼런스(우리)가 발디딤·관절·접촉을 주면 RL은 tracking만 배우면 됨(희소보상 회피, 수렴 빠름).
- 배포용 C++ 포팅은 **RL로 알고리즘 최종화 후**(재작업 방지).

## 관련
- 컨트롤러: `../biped_mpc_wbic.py`(MPC+WBIC+event-DCM) · `../biped_step.py`(gait/발배치) · `../biped_wbic.py`(WBIC·GEARBOX)
- 전략: `../../docs/MPC_RL_하이브리드_전략_리포트.md`(패턴 A·DTC) · `../../docs/roadmap_hybrid.html`
- 메모리: `biped-mpc-reimpl` · `mpc-rl-hybrid-roadmap`
