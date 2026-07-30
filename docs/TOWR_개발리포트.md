# TOWR·모델기반 지형계획 개발 리포트

**상태: 피벗 (모델기반 상한 → RL) · 2026-07-30**
TOWR 오프라인 지형 planning은 수렴(flat/step/block Optimal)하나, **추종·폐루프 조사가 모델기반의 상한을 확정** — 개루프 재생은 측방 드리프트로 발산, 폐루프(B MPC)는 이미 존재하고 상한이 동일(불연속 험지 붕괴). ∴ **robust 불연속 험지 = RL**. TOWR·B 브리지·C++ 발판주입 훅은 폐기가 아니라 **RL 교사/오프라인 참조 자산**으로 편입.

**기간** 2026-07-21 ~ 07-28 · 관련: [DTC 개발리포트](DTC_개발리포트.md) · 메모리 `b-elevation-tamols-towr-track`·`terrainmap-elevation-pointcloud`(지형맵)

---

## 1. 배경·목표

험지(gap/stepping) 크로싱이 **접촉 타이밍·발판을 지형에 맞춰 최적화하지 못하는** 구조적 한계에 걸려 있었다(A/B/C·D1 모두 고정 스케줄 또는 반응형 greedy). TOWR(Winkler, phase-based 센트로이달 TO, IPOPT)를 도입해:
1. 지형인지 최적 궤적(발판·타이밍)을 오프라인 생성하고,
2. 이를 추종해 **모델기반 험지 크로싱을 실증**하며,
3. 모델기반의 상한을 그어 **RL이 필요한 영역을 정의**하는 것이 목표.

> 결정: D1(OCS2) 승격 대신 TOWR. D1은 A/B와 구조 중복(NMPC vs SRBD)이고 타이밍 미해결이라, footholds+timing을 지형에 맞춰 최적화하는 모델기반 지형계획기로 TOWR 선택.

---

## 2. 구축 자산

| 파일 | 역할 | 상태 |
|---|---|---|
| `quad/towr/towr_cd.py` | **TOWR-in-CasADi(단순화판)** — SRBD+footholds(지형 위)+마찰콘+ROM. gait trot/crawl, 지형 flat/step/gap/platgap, 수동 가변 cadence | ✅ planning 작동 |
| `quad/towr/towr_cd.cpp` | 위의 C++ 포팅(Python과 1e-6 일치, `548259d`) | ✅ |
| `quad/towr/towr_track.py` | WBIC-lite 추종(SRBD 힘균형+GRF 분배) | ⚠️ 부양base 불안정 |
| `quad/towr/towr_wbic.py` | QP-WBIC 프로토타입(proxsuite, 풀 동역학) | ⚠️ 부분작동 |
| `simple_mpc/towr_track_B.py` | **B의 WBIC(KinodynamicsID/TSID) 브리지** + phase-leash 재생 | ✅ 정적 crawl / ❌ 동적게이트 |
| `quad/towr/towr_ext/` | **원조 ethz-adrl/towr(ifopt+IPOPT) 빌드** + 02_Leg 드라이버 | ✅ flat/block Optimal |
| `quad/mjcf/quad_terrain_step·platgap.mjcf` | 검증 지형 씬 | ✅ |

**재구현 결정(07-22)**: C++ TOWR(ifopt/ROS/catkin) 대신 CasADi 번들 IPOPT로 재구현(`towr_cd`). **★07-24 정정**: `towr_cd`는 **고정 접촉스케줄만 쓰는 단순화판**으로 TOWR 핵심인 **phase-based 타이밍 최적화가 빠짐**(그래서 갭서 수동 cadence 핵이 필요했음). **원조 ethz-adrl/towr(Winkler2018)를 `quad/towr/towr_ext/`에 빌드**(towr-example solve 0.21s·게이트 자동창발). 우리 SRBD 드라이버=`towr_ext/.../test/zero2leg_example.cc`(+`zero2leg_model.h`, `OptimizePhaseDurations` 사용).

**SRBD 파라미터**(pinocchio URDF 추출): m=38.016 kg, 관성 diag(0.941, 2.521, 2.236), 공칭 발위치 x±0.30 y±0.16, base_h 0.50, μ=0.6, max_dev{0.13, 0.09, 0.08}.

---

## 3. 오프라인 planning — ✅ 수렴

### 3.1 towr_cd(단순화판)
- **flat/step/gap solve**: 깊은 갭 착지 0개(발이 지형 정합·갭 회피), 계단 base 0.50→0.563 상승, SRBD 동역학 잔차 ~1e-6.
- **★GAP planning 핵심 = 짧은 stance**: 긴 crawl stance(Tg0.80·0.64 s)는 발판 하나가 base 0.19 m 이동을 커버해야 해 유효 ROM(±0.13)을 ±0.035로 조여 갭 근처 **infeasible**. **Tg≤0.40(stance≤0.28 s)** 로 줄이면 발이 갭 밖으로 빨리 재배치 → feasible + 갭 회피 완벽. 전역 위상오프셋 8개 스윕은 전부 infeasible = 문제는 "정렬이 아니라 stance 지속시간" 확증.

### 3.2 원조 TOWR(zero2leg, 02_Leg 적용, 07-27)
- **flat = Optimal 2.0 s**·**block(계단턱) = Optimal 7.8 s**(1e-6 클린): **위상 지속시간을 발별로 자동 최적화**(towr_cd가 못하던 phase-based 타이밍이 우리 로봇서 작동 확인).
- **gap(0.5 m)·narrowgap(0.25/0.15 m) = 미수렴**: constraint 40→2.4 감소하나 dual 2.6e4 정체(100 s/736 iter). 원인=우리 로봇 ROM 스텝 ≤0.26 m 한계 + 포물선 갭이 "부드러워" 발이 경사면 stance 가능(진짜 hole 아님) + trotw+phase-timing은 변수↑로 오히려 발산. TOWR 공지 난케이스.
- 함정: IPOPT `max_cpu_time`은 스레드합산 CPU초라 mumps 멀티스레드서 조기종료 → **`max_wall_time`**(IPOPT3.14+) 사용.
- 출력=`quad/towr/traj_towr_{flat,block,gap}.json`.

---

## 4. 추종 — 정적 crawl만 완주, 동적게이트는 개루프의 벽

### 4.1 결과 요약
| 추종기 | 정적/준정적 crawl | 동적게이트(walk/trot) |
|---|---|---|
| **WBIC-lite**(towr_track.py, SRBD 힘균형) | STEP 크로싱은 초기 성공했으나(아래 정정) | ❌ 부양base 미보상 |
| **QP-WBIC 프로토타입**(towr_wbic.py) | 정지 안정(tilt<3°)·첫 스윙 통과 | ❌ 지속보행 미달 |
| **B WBIC 브리지**(TSID, towr_track_B.py) | ✅ **tilt1.7° 완주·base z유지** | ❌ 스윙전환서 측방붕괴 |

- **원조 TOWR 궤적 TSID 추종(07-27)**: fly-trot 낙상 1.4 s(관절속도 34>19.7 rad/s 초과)·walking-trot 낙상 2.2 s(1.6 s까지 tilt<1° 완벽 후 스윙서 y드리프트)·**정적 crawl(X0.4·T7) ✅ 완주**(tilt1.5°, `traj_towr_flatcrawl2.json`). = **개루프 TSID 재생은 항상 3발지지 crawl만 완주**, 2발지지 동적게이트는 스윙전환서 측방붕괴.
- **B TSID 브리지 slow=견고**: KP_BASE=40이 firm base 핵심(7은 처짐). QP-WBIC 프로토타입이 못한 접촉전이·soft접촉을 B의 성숙 TSID가 처리.

### 4.2 ★개루프 vs 폐루프 직접대조 — 원인 확정(07-27)
같은 로봇·같은 TSID로 개루프 재생(towr_track_B.py) vs 폐루프 MPC(quad_centroidal_17dof.py) 평지 대조:

| | 개루프 TOWR 재생 | 폐루프 B MPC |
|---|---|---|
| y좌표 | −0.001 → **−0.466 발산** | ±0.005 유지 |
| tilt | 0.8 → **94° 전복** | 0.8~1.7° |
| 결과 | 낙상 2.2 s | **falls=0 정상보행** |

**원인 확정 = 되먹임 없는 측방 드리프트 발산.** 동적게이트(2발지지)는 도립진자라 스윙발이 실제 CoM 가는 곳에 착지해야 받아내는데, 개루프는 SRBD계획↔실물 불일치로 매 스텝 밀린 걸 되돌릴 방법이 없어 누적 → 지지밖 → 전복. 폐루프 MPC는 매 사이클 측정상태로 발판 재결정 → 드리프트 즉시 교정. **동적게이트 완주 = 폐루프 필수.**

### 4.3 ★정직 정정(07-28, provenance 재검증)
초기 "STEP 엔드투엔드 falls=0·tilt7°"(b19dbd4)의 궤적파일 `traj_flat/gap/step.json`은 **손상(truncated)** 확인. 유효한 `traj_towr_*`로 재실측하니 **WBIC-lite는 평지 walk/trot 포함 전 궤적 낙상**(flat x0.47/tilt76·gap tilt74·narrowgap tilt61). = SRBD 힘기반 WBIC-lite가 TOWR 부양base를 못 잡음(포괄 확인). 재현 검증된 완주는 **B TSID의 정적 crawl뿐**. (B·C·TOWR+B-TSID의 과거 수치는 simple_mpc C++ 바인딩 미설치로 현 env 재검증 불가 = 확정 아님.)

---

## 5. 폐루프 조사 — 결정적 결론

"TOWR footholds/timing을 폐루프 MPC에 참조 주입하면 fast 갭이 되는가?"를 조사.

### 5.1 B는 **이미** 모델기반 폐루프 지형발판을 구현
`MPC::updateStepTrackerReferences`(매 iterate) = Raibert 발판 → heightmap 갭회피(갭이면 최근접 유효셀 이동) → OCP 주입. **= "Online foothold opt."(Jenelten 2020) 폐루프 그 자체.** A의 footScore/selectFoot도 동일 계보. 단 B의 주입 훅 실체는 `velocity_base`·`setHeightmap`·`x_reference`(base-z)뿐 = **임의 발판시퀀스 주입 훅 없음**(setReferencePose는 iterate가 매 사이클 덮음).

### 5.2 B 폐루프의 상한 = 연속지형
- 연속 슬로프 등반 ✅(15° 램프 tilt<5°, base높이 지형적응).
- **discrete 붕괴**: TOWR가 계획한 그 platgap에 B 네이티브 폐루프 → t=0.3에 base 0.955 폭등·tilt62°·즉시 텀블. quad_terrain_gap(0.16 단차·0.32 갭)도 VX0.2/0.3 첫 플랫폼서 낙상. A도 gap/stepping "속도-공명 취약"(정렬된 갭만).

### 5.3 ★C++ 외부발판 주입 훅 구현·검증(07-27) — 그래도 결론 불변
§5.1의 "매 iterate Raibert가 덮음"을 실제 해소: `simple_mpc` C++에 `MPC::setExternalFoothold(ee,pos)`/`clearExternalFootholds()` 추가(`updateStepTrackerReferences`서 Raibert next_pose_를 외부 world 발판으로 대체, heightmap보다 우선). 바인딩·재빌드 완료. Python 배선=`quad_centroidal_17dof.py` env `TOWR_INJECT=<traj.json>`(base x 위상잠금 → 발별 다음 착지 주입, placement-only).
- **평지+flat TOWR 발판 주입 = falls=0 검증**(훅 작동 + 폐루프가 불완전매칭 외부발판 받고도 균형유지, y드리프트 −0.17 = 개루프 −0.47보다 훨씬 작음).
- **그러나 갭 관철은 여전히 막힘**: ①TOWR가 우리 로봇 갭 계획을 수렴 못함(ROM), ②B heightmap 네이티브도 갭 실패. **∴ 로버스트 모델기반 갭 baseline 자체가 없어 TOWR 주입이 개선할 대상이 없음.**

### 5.4 결론 — 벽은 주입이 아니라 안정성
- TOWR-최적 발판/timing을 주입해도 **안정성 벽을 못 넘음**(B 폐루프가 이미 존재·상한 동일).
- **∴ TOWR→MPC 실시간 주입 = 저효용, 개발 권장 안 함.** 훅은 좋은 계획소스가 생기면 재사용 가능한 자산으로 잔존.
- **robust 불연속 험지 = RL 확정.**

---

## 6. 하이브리드 경계 (확정)

| 층 | 담당 영역 | 구현 |
|---|---|---|
| **모델기반 폐루프** | 연속 + 구조화/정렬 지형 소유 | A footScore / B heightmap (배포·검증) |
| **RL** | 임의 불연속 험지 robust | 착수 대기(WBIC·지형맵·footScore substrate 재사용) |
| **TOWR** | 오프라인 최적 참조(매핑 구조지형 계획 or RL 교사) | ✅ 구축, 실시간 주입 ❌ |

모델기반 폐루프는 연속지형을 소유하고, robust 불연속은 RL이 담당하며, **TOWR는 실시간 MPC 주입이 아니라 오프라인 참조/RL 교사 다리(C-2)로 편입**된다.

---

## 7. RL 착수 시 재사용 자산 (핸드오프)

- `quad/towr/` — TOWR 오프라인 최적 궤적 생성 + `towr_ext/`(원조 phase-based) → **RL 교사·레퍼런스**.
- `simple_mpc/towr_track_B.py` — B WBIC 브리지(정적 crawl 참조 추종) → **오프라인 궤적 증류 다리**.
- `MPC::setExternalFoothold` C++ 훅 — 계획소스 확보 시 발판 주입 재사용.
- B/A의 heightmap·footScore·WBIC substrate.
- [DTC 개발리포트](DTC_개발리포트.md) — 하이브리드 5패턴·H0~H3 로드맵.

> RL 소스 제공 시 위 자산을 교사/substrate로 하이브리드 H0부터 착수.

---

## 8. 커밋 이력 (이 트랙)
- `b19dbd4` TOWR-in-CasADi + STEP 크로싱 엔드투엔드(초기, 이후 궤적 손상 확인)
- `51ed119` GAP planning 해결(짧은 stance)·tracking 풀WBIC 필요
- `c4942f8` QP-WBIC 프로토타입(부분작동)
- `1f551ff` B WBIC 브리지(정적 crawl 견고·동적게이트 개루프 벽)·가변 cadence
- `548259d` towr_cd C++ 포팅(Python 1e-6 일치)
- 원조 towr 빌드·zero2leg 드라이버·C++ 발판주입 훅·provenance 재검증 = 메모리 `b-elevation-tamols-towr-track` 기록
