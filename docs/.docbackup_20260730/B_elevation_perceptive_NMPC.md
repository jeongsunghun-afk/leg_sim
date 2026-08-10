# B승격: 통합 Perceptive NMPC (Grandia식) → DTC

> 작성 2026-07-29. injection(짜깁기, 마진 얇음)의 취약성을 근본 해결하는 **모델기반 정공** = B(quad_centroidal, kino-dynamic)를 **Grandia 2023식 통합 폐루프 perceptive NMPC**로 승격. 이후 DTC(RL 추종).

## 0. 왜 B승격인가 (전략)

**injection의 취약성 규명(2026-07-29)**: A Raibert+TAMOLS+selectFoot+갭회피 = **짜깁기(stitched)** → 층간 타협·마진 얇음("불안불안", 뷰어 확인). 연속지형 ~0.24m·이산갭 형상민감.

**Grandia perceptive NMPC(D1/ANYmal 실기)**가 강건한 이유 = **하나의 폐루프 통합 NMPC**가 base+발판+힘을 지형-aware로 co-optimize. 짜깁기 아님.

**B(quad_centroidal)가 정확히 이 기반**: kino-dynamic centroidal(다리관성 포함)+발 결정변수(FOOT_DECISION)+폐루프 MPC = Grandia/OCS2 계열. **B에 완전 perceptive를 얹으면 = 통합 perceptive NMPC.**

**DTC 정합**: B(또는 TAMOLS) 참조 → RL 추종. B 통합 NMPC = 모델기반 강건 baseline + DTC 참조공급.

## 1. B 현재 상태 (자산)

`quad/simple_mpc/quad_centroidal_17dof.py` (실행 `run_centroidal_17dof.sh`, pixi `~/simple-mpc/.pixi/envs/default/bin/python`+CONDA_PREFIX):
- **안정 보행**: 17-DOF, falls=0, ~0.4 m/s(명령0.5의 88%), ~20Hz(aligator ProxDDP RTI).
- **보유**: FOOT_DECISION(발 자유변수)·WBVY(측방)·STIFF(접촉정합)·WAIST_LOCK(허리)·**setHeightmap(발판 z)·HM_BASE(base z 지형적응)**.
- **부분 perceptive**: 연속 경사(15° ramp) base z 상승 등반(tilt<5°). **discrete(gap/stepping)·base 자세·발판 XY 지형회피 미흡**.
- 실기갭: 20Hz(A는 1kHz)·속도 0.4(A 1.85).

## 2. 목표 = Grandia식 완전 perceptive NMPC

Grandia 2023 핵심 요소를 B OCP에 embed:
- **① 지형-aware 발판(SDF)**: 발판 XY를 gap/edge 회피(signed distance field), 유효 지지면 위로. B는 발판-z만(setHeightmap) → **발판-XY 지형제약 추가**.
- **② 지형 base pose**: base z(HM_BASE 있음) + **base 자세(roll/pitch) 지형 정렬** 추가.
- **③ 통합 co-optimize**: base+발판+힘을 한 OCP서(B는 이미 centroidal+발결정=구조 보유). GIAC/안정 유지.
- **④ 폐루프**: B는 이미 MPC(매사이클 재해). 유지.

## 3. 계획

### Phase 0 — B 재가동·perceptive 현황 확인 (선행)
1. B 실행 확인(pixi env, run_centroidal_17dof.sh 평지 falls=0 재현).
2. 현 perceptive(setHeightmap·HM_BASE) 지형 테스트: 연속경사 OK·discrete 실패 재확인. 어디까지 되는지 baseline.

### Phase 1 — 지형-aware 발판 (①, 핵심)
1. B OCP에 **발판 XY 지형제약/비용**: heightmap→footScore/SDF, gap/edge 회피. (TAMOLS의 foothold-on-ground 10^4·edge avoidance 참조).
2. 발판이 유효 지지면(stone) 위로 최적화되게. B의 발 결정변수(FOOT_DECISION)에 지형항 추가.
3. 검증: discrete stepping/gap서 발판이 stone 위 배치(낙상 감소).

### Phase 2 — 지형 base 자세 (②)
1. base 자세(roll/pitch)를 지형 평면(stance 발 fit)에 정렬(TAMOLS base pose alignment 식34 참조).
2. HM_BASE(base z)와 통합.
3. 검증: 경사·discrete서 base가 지형 따라 기울며 안정.

### Phase 3 — 벤치·강건성 비교
1. B perceptive vs injection(A+TAMOLS) vs A: 동일 지형(연속경사·discrete gap+height)·지표(falls/tilt/속도/외란복구).
2. **핵심 가설**: 통합 NMPC(B)가 짜깁기(injection)보다 강건(마진↑, 외란복구↑).
3. 실시간: 20Hz 유지 or HPIPM/condensing으로 향상(sim은 20Hz 허용 가능).

### Phase 4 — DTC 전환 (B승격 후)
1. B(또는 TAMOLS) 통합 NMPC 참조(발판·base pose·twist·힘·접촉) → RL 관측/보상.
2. IsaacLab quad env(hind_leg RL 재사용) + 참조추종 보상.
3. RL이 B baseline 넘는지(외란·험지 강건성).

## 4. 참조

- **Grandia 2023** "Perceptive Locomotion through NMPC"(OCS2, D1/ANYmal). [[d1-navbothub-mpc-amp-analysis]]·[[perceptive-nav-tamols]].
- **TAMOLS**(2206.14049) — 지형 발판/base 비용(§V: foothold-on-ground·edge avoidance·base alignment), GM-observer. 정독본 참조.
- **DTC**(2309.15462) — TO→RL 추종. [[dtc-aptrl-papers]].
- B 코드=quad_centroidal_17dof.py, simple_mpc(KinodynamicsID=B의 WBC). §5.5 안정화 config(STIFF·FD·WBVY).

## 5. 리스크·판단

- **B는 20Hz·0.4m/s** — 실시간·속도는 A 열세. 하지만 B승격 값어치=**지형 통합 NMPC**(A SRBD 불가·injection 취약 해결). 속도경쟁 아님.
- **simple_mpc OCP 수정** = 라이브러리 작업(C++ 훅). 발판 지형제약 추가가 Phase1 관문.
- **폴백**: B perceptive 정체 시 injection(작동함)이 모델기반 baseline 잔존. RL(DTC)이 최종 강건.
- **판단 게이트**: Phase 3 벤치서 B 통합이 injection보다 강건 확인 → DTC. 아니면 injection baseline + DTC 직행.

## 6. 즉시 다음 액션
**Phase 0** — B(quad_centroidal_17dof.py) pixi env 재가동, 평지 falls=0 재현 + 현 perceptive(연속경사 OK·discrete 한계) 확인. 그 후 Phase 1(발판 지형제약).
