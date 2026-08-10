# TOWR·모델기반 지형계획 트랙 리포트

**기간** 2026-07-21 ~ 07-27 · **작성** 2026-07-22 · **갱신** 2026-07-27
**한 줄 결론** 모델기반 오프라인 지형 planning은 성공(TOWR), 그러나 **fast 험지 실시간 추종의 벽은 안정성**이고 주입 API가 아니다 → **robust 불연속 험지 = RL 하이브리드**로 확정. TOWR·브리지는 RL 교사/오프라인 참조로 편입.
**07-27 확증** ①진짜 ethz-adrl/towr **빌드**해 우리 모델 적용(flat/block Optimal, towr_cd는 phase-timing 없는 **단순화판**임이 판명) ②§4.1이 지목한 **C++ 발판주입 훅을 실제 구현**(`setExternalFoothold`)·평지 falls=0 검증 ③**TOWR=오프라인 계획기(rviz 시각화)일 뿐 제어기 아님** 규명 ④옵션3 = **B heightmap도 갭 실패** → 로버스트 모델기반 갭 baseline 부재 재확인. **결론 불변**(RL 피벗), 근거만 더 단단해짐.

관련: [MPC_RL 하이브리드 전략 리포트](MPC_RL_하이브리드_전략_리포트.md) · [RPET 지형맵 통합](RPET_TERRAIN_MAP_INTEGRATION.md) · 메모리 `b-elevation-tamols-towr-track` · `mpc-rl-hybrid-roadmap`

---

## 1. 배경·목표

험지(gap/stepping) 크로싱이 **접촉 타이밍·발판을 지형에 맞춰 최적화하지 못하는** 구조적 한계에 걸려 있었다(A/B/C·D1 모두 고정스케줄 또는 반응형 greedy). 그림의 6개 elevation-map 플래너 중 **TOWR**(Winkler, phase-based, 센트로이달, IPOPT, 준-오프라인)를 도입해:
1. 지형인지 최적 궤적(발판·타이밍)을 오프라인 생성하고,
2. 이를 추종해 **모델기반 험지 크로싱을 실증**하며,
3. 모델기반의 상한을 그어 **RL이 필요한 영역을 정의**하는 것이 목표.

> 결정: D1(OCS2) 승격 대신 TOWR. D1은 A/B와 구조 중복(NMPC vs SRBD)이고 타이밍 미해결. TOWR가 "모델기반 지형계획기"의 정점.

---

## 2. 구축 자산

| 파일 | 역할 | 상태 |
|---|---|---|
| `quad/towr/towr_cd.py` | **TOWR-in-CasADi** — SRBD+footholds(지형위)+마찰콘+ROM. gait trot/crawl, 지형 flat/step/gap/platgap, **가변 cadence** | ✅ planning 작동 |
| `quad/towr/towr_track.py` | WBIC-lite 추종(힘균형+GRF) | ✅ STEP / ❌ fast |
| `quad/towr/towr_wbic.py` | **QP-WBIC 프로토타입**(proxsuite, 풀 동역학) | ⚠️ 부분작동 |
| `simple_mpc/towr_track_B.py` | **B의 WBIC(KinodynamicsID/TSID) 브리지** + phase-leash 재생 | ✅ slow / ❌ fast |
| `quad/mjcf/quad_terrain_step·platgap.mjcf` | 검증 지형 씬 | ✅ |

**재구현 결정(07-22)**: C++ TOWR(ifopt/ROS/catkin) 빌드 대신 **CasADi 번들 IPOPT로 우리 스택에 재구현**(PACE식 방법론 이식). casadi 3.7 + IPOPT + pinocchio + proxsuite 확인.
**★07-27 정정**: 위 `towr_cd`는 **고정 접촉스케줄만 쓰는 단순화판**으로, **TOWR 핵심인 phase-based 타이밍 최적화가 빠짐**(그래서 갭서 수동 cadence 핵 필요했음). 진짜 원조 **ethz-adrl/towr(ifopt+IPOPT)를 `towr_ext/`에 빌드 완료**(towr-example solve 0.21s·게이트 자동창발). 우리 SRBD 적용 드라이버=`towr_ext/towr/towr/test/zero2leg_example.cc`(+`zero2leg_model.h`): **flat/block=Optimal**(위상 지속시간 발별 자동최적화 확인=towr_cd가 못하던 것), gap(0.5m)·narrowgap(0.25/0.15m)=**미수렴**(우리 로봇 ROM 스텝≤0.26m 한계). towr_cd는 경량 참조로 잔존.

**SRBD 파라미터**(pinocchio URDF 추출): m=38.016kg, 관성 diag(0.94, 2.52, 2.24), 공칭 발위치 ±0.30/±0.16, base_h 0.50, μ=0.6.

---

## 3. 결과

### 3.1 오프라인 planning — ✅ 성공
- **flat/step/gap 궤적 solve**: 깊은 갭 착지 0개(발이 지형 정합·갭 회피), 계단 base 0.50→0.56 상승, SRBD 동역학 잔차 ~1e-6.
- **★GAP planning 핵심 = 짧은 stance**: 긴 crawl stance(Tg0.80·0.64s)는 발판 하나가 base 0.19m 이동을 커버해야 해 유효 ROM(±0.13)을 ±0.035로 조여 갭 근처 **infeasible**. **Tg≤0.40(stance≤0.28s)** 로 줄이면 발이 갭 밖으로 빨리 재배치 → feasible + **갭 회피 완벽**. 전역 위상오프셋 8개 스윕 = 전부 infeasible로 "정렬이 아닌 stance 지속시간" 확증.
- **가변 cadence**: 플랫폼 느림(Tg0.80)·갭 근처 빠름(Tg0.40) 궤적 생성 → planning 성공.

### 3.2 추종 — ✅ slow / ❌ fast
| 추종기 | slow(step·평지 crawl Tg0.80) | fast(gap Tg0.40) |
|---|---|---|
| **STEP 엔드투엔드**(WBIC-lite) | ✅ base 0.50→0.64 등반, falls=0, tilt7° | — |
| **QP-WBIC 프로토타입** | 정지 안정(tilt<3°)·첫스윙 통과 | ❌ 지속보행 미달 |
| **B WBIC 브리지**(TSID) | ✅ **tilt1.7° 완주·base z유지** | ❌ 정체/크라우치/전복 |

- **STEP 크로싱**: 모델기반 지형 크로싱 엔드투엔드 최초 실증(TOWR 계획→추종→0.10m 단 등반).
- **B 브리지**: B의 성숙 TSID(접촉전이·soft접촉 처리)로 slow 참조를 견고 추종. **KP_BASE=40이 firm base의 핵심**(7은 처짐).

### 3.3 fast 갭 미완의 근본원인 — **개루프 재생의 벽**
게인·leash·firm 지지발·가변 cadence 다수 시도, 최선 x~0.88(갭 근처)까지나 미완주. 원인은 튜닝이 아니라:
- B 자체 게이트는 **MPC가 매 10ms 현재상태서 재계획하는 폐루프**라 안정.
- **개루프 TOWR 재생**은 fast cadence서 SRBD계획↔full-dynamics 로봇 상태불일치가 누적 → 회복 불가. slow는 불일치가 작아 추종 OK.

---

## 4. ★폐루프 조사 — 결정적 결론

"TOWR footholds/timing을 B의 MPC에 참조 주입하면(폐루프 유지) fast 갭이 되는가?"를 조사.

### 4.1 주입 API는 완비돼 있다 (기술적으로 전부 가능)
| API | 주입 대상 |
|---|---|
| `OCP.setReferencePose(t, foot, pose)` / `setPoseBase(t, pose)` / `setReferenceState(t, x)` | per-node 발판·base·상태 |
| `MPC.generateCycleHorizon(contact_states)` | **임의 접촉 스케줄(timing)** |
| `MPC.setHeightmap` + `nearestValid` | 지형 발판 갭회피 |

### 4.2 그러나 B는 **이미** 모델기반 폐루프 지형발판을 구현
`MPC::updateStepTrackerReferences`(매 iterate) = Raibert 발판 → heightmap 갭회피(갭이면 최근접 유효셀로 이동) → OCP 주입. **= 그림의 "Online foothold opt."(Jenelten 2020) 폐루프 그 자체**. A의 footScore/selectFoot도 동일 계보.

### 4.3 B 폐루프의 상한 = 연속지형 (데이터)
- 슬로프 등반 ✅ (TAMOLS base높이 적응, 15° 램프 tilt<5°)
- **discrete 붕괴**: TOWR가 계획한 그 **platgap에 B 네이티브 폐루프 → t=0.3에 base 0.955 폭등·tilt62°·즉시 텀블**. stepping stones도 첫 스톤서 낙상. A도 gap/stepping "속도-공명 취약"(정렬된 갭만).

### 4.4 결론 — 벽은 주입이 아니라 **안정성**
- TOWR-최적 발판/timing을 주입해도 **안정성 벽을 못 넘음**(+ setReferencePose는 매 iterate Raibert가 덮어 C++ 수정 필요).
- **∴ TOWR→MPC 실시간 주입 = 저효용**(B 폐루프가 이미 존재·상한 동일). **개발 권장 안 함.**

### 4.5 ★07-27 후속 — C++ 훅 실제 구현·검증(그래도 결론 불변)
§4.1이 지목한 "매 iterate Raibert가 덮어 C++ 수정 필요"를 **실제 구현**: `simple_mpc` C++에 `MPC::setExternalFoothold(ee,pos)`/`clearExternalFootholds()` 추가(`updateStepTrackerReferences`서 Raibert next_pose_를 외부 world 발판으로 대체, heightmap보다 우선). 바인딩·재빌드 완료. Python 배선=`quad_centroidal_17dof.py` env `TOWR_INJECT=<traj.json>`(base x 위상잠금→발별 다음 착지 주입). **평지+flat TOWR 발판 주입=falls=0 검증**(훅 작동 확인). **그러나 갭 관철은 여전히 막힘**: ①TOWR가 우리 로봇 갭 계획을 수렴 못함(ROM 한계), ②B heightmap 네이티브도 갭 실패(옵션3: quad_terrain_gap서 VX0.2·0.3 첫 플랫폼 낙상). **∴ 로버스트 모델기반 갭 baseline 자체가 없어 TOWR 주입이 개선할 대상이 없음 → §4.4 결론(저효용·RL 피벗) 재확인.** 훅은 좋은 계획소스 생기면 재사용 가능한 자산으로 잔존.

---

## 5. 하이브리드 경계 (확정)

| 층 | 담당 영역 | 구현 |
|---|---|---|
| **모델기반 폐루프** | 연속 + 구조화/정렬 지형 **소유** | A footScore / B heightmap (배포·검증) |
| **RL** | 임의 불연속 험지 **robust** | 착수 대기(WBIC·지형맵·footScore substrate 재사용) |
| **TOWR** | **오프라인 최적 참조** (매핑 구조지형 계획 or RL 교사) | ✅ 구축, 실시간 주입 ❌ |

**"RL 하이브리드가 답" = 재확인** — 단 이번 조사로 경계가 정량화됨. 모델기반 폐루프는 연속지형을 소유하고, robust 불연속은 RL이 담당하며, **TOWR는 실시간 MPC 주입이 아니라 오프라인 참조/RL 교사 다리(C-2)로 편입**된다.

---

## 6. RL 착수 시 재사용 자산 (핸드오프)

- `quad/towr/` — TOWR 오프라인 최적 궤적 생성 (**RL 교사·레퍼런스**)
- `simple_mpc/towr_track_B.py` — B WBIC 브리지 (slow 참조 추종, **오프라인 궤적 증류 다리**)
- B/A의 heightmap·footScore·WBIC substrate
- [MPC_RL 하이브리드 전략 리포트](MPC_RL_하이브리드_전략_리포트.md) — 하이브리드 5패턴·H0~H3 로드맵

> RL 소스는 별도 제공 예정. 제공 시 위 자산을 교사/substrate로 하이브리드 H0부터 착수.

---

## 7. 백로그 (RL 무관, 언제든)
- 경사 마찰콘 회전 (sloped stepping stones)
- InEKF → CoCo-InEKF 상태추정기 격상

---

## 8. 커밋 이력 (이 트랙)
- `b19dbd4` TOWR-in-CasADi + STEP 크로싱 엔드투엔드
- `51ed119` GAP planning 해결(짧은 stance)·tracking 풀WBIC 필요
- `c4942f8` QP-WBIC 프로토타입(부분작동)
- `1f551ff` B WBIC 브리지(slow 견고·fast 개루프 벽)·가변 cadence
- (폐루프 조사 결론 = 메모리 `mpc-rl-hybrid-roadmap` 기록)
