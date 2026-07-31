# TAMOLS Whole-Body 온라인 추종 — 개발 리포트

> **상태: 종결(구조적 막다른길) → 험지 크로싱 RL 피벗 확정 (2026-07-30 기준).** ETH식 완전 TAMOLS 계획(base pose+발판+GIAC 안정)을 A의 실행층(SRBD MPC + WBIC)으로 online receding-horizon 추종하려던 시도. 평지 안정화 진단(P0~P2)까지 수행했으나 **full TAMOLS 단발 WBIC 재생은 구조적 막다른길로 확정(2026-07-15, perceptive-nav-tamols)**. 아래 진단·수정 내역은 종결 근거이자 재사용 자산으로 보존한다.
>
> ※ 범위: 이 리포트는 **A 실행층 위 whole-body 추종**(§2~6, 평지)과 **갭 크로싱 + C++ TAMOLS 솔버**(§7)를 다룬다. 모델기반 지형계획(TOWR) 트랙은 `TOWR_개발리포트.md` 참조.

## 1. 결론 (상위 최종 위치)

- **full TAMOLS → WBIC 단발 재생 = 구조적 막다른길.** 저수준(WBIC)은 문제없으나(임의 plan 추종 가능), 계획층에 **3가지가 부재**: ①스텝길이 적응 X(Raibert=속도×stance 고정) ②접촉타이밍 적응 X(고정 게이트 클럭=속도공명 근본원인) ③base-발판 협조 최적화 X. tamols-rl은 발판 1개/다리=수신 horizon 플래너라 단발 2s 재생 자체가 부적합.
- **★확정 피벗**: 단발 WBIC-TAMOLS 정리 → **험지 크로싱 = RL(하이브리드)**. A(보행/기립/점프 배포)는 모델기반 유지, 험지 크로싱만 RL이 발판/타이밍 정책을 담당하되 **WBIC·지형맵·footScore를 실행층/관측으로 재사용**.
- 아래 진단(④ lateral=W_AM, ⑤ online=re-anchor death spiral)과 자산은 위 결론의 실측 근거이며, 향후 합성 정상-cadence plan으로 추종기 재검증 시(B2/B3 모델기반 실시간 OCP 착수 시) 출발점이다.

## 2. 시도한 것 (목표와 단계)

**목표**: 발판만 주입(injection)이 아닌 **base·자세·발판 통합 추종**(ETH TAMOLS+WBC 방식). 발판만 주입하는 injection(A gait + TAMOLS 발판)은 falls=0으로 이미 작동 — A의 검증된 gait가 lateral 드리프트를 우회한다. 순수 whole-body 추종은 그 문제에 정면으로 부딪힌다.

**단계 정의**:
- **P0**: offline 단일 사이클(0.8s) 추종 falls=0 · tilt<5°
- **P1**: offline 다사이클(체인) 추종 falls=0 (연속 3s+)
- **P2**: online receding-horizon 평지 falls=0 · 3s+ (death spiral 제거)
- **P3**: online + 지형(heightmap) 크로싱, injection과 성능 비교

## 3. 진단 (평지 online 추종이 ~1초 내 낙상)

| # | 문제 | 원인 | 상태 |
|---|---|---|---|
| 1 | z 침하 | RSL이 SRBD MPC 우회(gravity-comp만, base z-hold 약함) | ✅ `TAM_MPC`로 해결(offline 확인) |
| 2 | phantom stance | `SW_DUR`(0.4) > 계획 swing 위상(0.2) → 발 반만 스윙 | ✅ `SW_DUR` env 매칭 |
| 3 | base 후진 lurch | 오염 입력 시 solve_fast 후진 vx | ✅ `TAM_CLEANV`(X clean) |
| 4 | **lateral/yaw 드리프트** | CoM 지지폴리곤 유지 실패(body-sway 추종 부족) | ✅ P0에서 `W_AM`으로 해결(§4) |
| 5 | death spiral | 추종오차→오염상태→나쁜 replan→악화 | ❌ online 고유(§5) |

**핵심**: 개별요인(1·2·3)은 downstream 손잡이로 해결. 솔버 계획은 **깨끗한 입력엔 양호**(후진 없음)이므로 병목은 계획이 아니라 **추종의 lateral 로버스트니스**(우리 WBC의 CoM-지지 유지 성숙도 < RSL). 검증된 가설: **H2(각운동량 미제어)=참**(단일지지 스윙 시 발 flail·다리 반력이 yaw/roll 각운동량 유발) / **H4(발판 비대칭)=거의 무효** / **H5(SRBD 실행층 한계)=online에서 재확인**.

## 4. Phase 0 결과 — 각운동량 task(W_AM)가 ④의 핵심 (offline, 2026-07-29)

offline 첫계획(정지 덤프 `TAM_DUMP`) 추종에서:
- **W_AM 기본→30**: t=0.75 tilt **49.9 → 19.2°**(절반↓). **t=0.5까지 tilt<6.3°**(이전 ~50). → P0 목표 tilt<5° 근접.
- 각운동량 감쇠(KD_AM)·W_ORI 추가는 미미(plateau ~19). 마지막 스윙(FL)에서 tilt 19로 튐 = 사이클 경계 효과(체인에서 해소 기대).
- **H4(발판 대칭화) 거의 무효**(tilt 49.8, yaw만 8→2 약간). foot 비대칭은 부차.

→ **Phase 0 대체로 성공**: W_AM=30으로 offline 단일사이클 t=0.5까지 tilt<6.3°.

## 5. online(P2) 결과 — ⑤ re-anchoring death spiral (online 고유 벽)

- **online은 W_AM으로 안 고쳐짐.** online+W_AM=30: 여전히 낙상(**yaw −143° 스핀** · z 침하). W_YAW=30 · REPLAN_DT 0.4/0.8 모두 실패(yaw −132° / −47°).
- **⑤ re-anchoring death spiral이 online 고유 병목**(offline엔 없음). yaw 급발산 = 재anchor가 yaw 참조/발판을 오염시켜 누적. WBIC yaw task(W_YAW)로 안 잡힘 = **참조 자체가 오염**되는 문제.
- 시도한 완충책: swing foot commitment(절대시간+target 동결)로 재anchor 대응, replan 초기조건을 측정상태 대신 이전 계획상태 blend(오염 완충) 검토 — 그러나 **death spiral이 구조적**(④가 유발, ⑤가 online에서 증폭)이라 downstream 손잡이로 못 풂.

## 6. 종결 근거와 재사용 자산

**왜 막다른길인가**: P0(offline W_AM)은 확보됐으나 P2(online 폐루프)에서 re-anchoring death spiral을 못 풂. 근본은 §1의 **계획층 3부재** — tamols-rl 플래너가 수신 horizon(1스텝) 구조라 단발 재생 정합이 불가하고, 우리 실행층은 그 참조의 오염을 흡수할 계획층이 없다. downstream WBC 게인(W_AM·W_YAW·KD_AM)으로는 참조 오염을 교정할 수 없음이 실측으로 확정.

**재사용 자산**(이번 세션 구축):
- `TAM_MPC=1`: SRBD MPC 기반 추종(z 침하 해결)
- `TAM_CLEANV=1`: X 전진 clean(후진 회피, Y sway 유지)
- `TAM_DUMP=<file>`: online 계획 → load_tamols 포맷 덤프(offline 격리·대칭성 진단)
- `SW_DUR` env: offline도 읽음(계획 위상 매칭)
- `W_AM=30`: 각운동량 task(④ lateral/yaw 억제 근거)
- swing foot commitment: 절대시간+target 동결(재anchor 대응)
- 발판 대칭화·깨끗계획 후처리 스크립트, 지형(hsteps·dsteps·dsteps2·trench·gapcourse)

**이관**: 험지 크로싱은 RL(하이브리드) 트랙으로. 위 WBIC 실행층·지형맵·footScore는 RL의 실행층/관측으로 재사용(콜드스타트 아님). 모델기반 실시간 OCP(B2/B3)는 살아있으나 현재 미착수 — 착수 시 합성 정상-cadence plan으로 이 추종기를 재검증한다.

## 6.1 재확인 — Phase 4 교차 벤치 + D1 참조 평가 (2026-07-31)

사용자 요청("TAMOLS+RSL를 D1 참조로 재앵커·base_z 침하 근본해결")에 따라 **독립 재조사**했고, §3~§6 결론을 **정량 재확인**했다.

**Phase 4 교차 벤치**(동일 17-DOF·동일 지형·연속지형, VX=0.2, C++ trot_sim, 첫 tilt>90° base_x / 최종 상태):
| 지형 | A(반응형·plain) | injection(TAMOLS 발판주입) | pure-online RSL | D1(perceptive NMPC) |
|---|---|---|---|---|
| slope 15° | **x2.17·z0.838·falls0 완주** | x0.6·z0.50 **stall(미등반)** | x0.34·**z0.166 붕괴** | x2.65·z0.84(crest 전복) |
| rough | **x2.14·falls0 완주** | x0.6 **stall** | x0.21·**z0.156 붕괴** | x2.12(이후 전복) |

**진단(TAM_DBG 실측)**: pure-online RSL 붕괴는 base_z droop이 아니라 **yaw 급발산(0→−50°)+횡드리프트(comy 0→−0.27)=붕괴**(t≈0.4s부터). §3·§5의 ④lateral/⑤death-spiral·H2(각운동량) 재확인. injection stall은 **TAMOLS 발판이 명목(fwd0=0)이라 전방 Raibert 스텝 부재→A의 전진추진 제거**(원인 규명).

**D1 참조가 근본해결 못하는 이유**: D1가 base_z·yaw를 잡는 것은 (a)통합 NMPC의 base 궤적 + (b)WeightedWbc **모멘텀률 base 6D FF**(A WBIC엔 없음·순수 PD) + (c)연속 gait클럭 덕. 이를 TAMOLS+RSL에 이식하려면 **사실상 D1 컨트롤러 자체**가 된다(이미 보유·연속지형 작동). 순수 WBC 게인/FF로는 **참조 오염(재앵커 death spiral)**을 못 고침이 §6 결론 그대로. 연속 world 클럭+전방 발판(TAM_CONT 실험)은 stall만 미미 개선(x0.34→0.44)·붕괴 잔존이라 폐기.

**결론 재확정**: 연속지형=**A(반응형) 또는 D1(perceptive)**가 강건(실측). pure-online TAMOLS+RSL은 구조적 막다른길(재확인). 이산험지=RL. D1 참조의 실질 = "연속지형은 D1/A 쓰라"이며, 그것이 이미 결론.

## 7. 별도 실행 시도 — 갭 크로싱 + C++ TAMOLS 솔버 (2026-07-27~28)

§2~6은 A 실행층 위 whole-body **평지** 추종이었고, 이건 **불연속 지형(갭) 크로싱**을 TAMOLS 계획 + C++ 전용 솔버로 실행한 별도 시도다. 결론은 같다 — **계획(TAMOLS)은 OK, 추종기(executor)가 벽**. (구 `모델기반_갭크로싱_탐색리포트.html` 통합.)

### 7.1 계획 — TAMOLS는 깨끗한 갭 크로싱 계획을 낸다
tamols-rl(ianpedroza, Drake)에 02_Leg 파라미터(m=37.9·nominal 0.52·μ0.6·sphere발) 적응, GAP=0.20m서 feasible solve. `add_gap_avoid_footholds`(앞발=갭 너머·뒷발=갭 앞) 하나로 발이 solid에 straddle(뒤 0.42·앞 0.87), base pitch ±13.4→±10.4°, 전진 0→0.73(갭 통과). 단 이건 **옵티마이저 해의 품질**이지 로봇이 건넜다는 실행 검증이 아님.

### 7.2 실시간 측정 — Drake는 583× 느려 불가
2-phase(최소 0.8s) cold solve 383ms=2.6Hz(실시간 50Hz와 19×), warm-start도 4~13Hz로 미달. Drake는 연구·오프라인용(범용 NLP) → **실시간 TAMOLS = 튜닝이 아니라 C++ 전용 솔버 재구현**(583× 못 메움).

### 7.3 ★C++ TAMOLS 솔버 — 완성 (planner)
`quad/cpp/tamols/`, Drake 단계별 정합 검증:
- **정식화**: 스플라인·지형·제약5(초기·연속·friction·kinematic·GIAC Eq17)·비용3(track·foothold·nominal). Drake residual 0~e-6·비용 rel 1e-9 ✅
- **솔버**: SQP-RTI(eiquadprog QP + ℓ1 merit + 적응형 LM). Drake해=고정점·섭동서 feasible 수렴 ✅
- **실시간**: 해석 Jacobian(cost/eq/비GIAC + GIAC block-sparse FD) → **<20ms(5-iter 11.7ms) 실시간 달성** ✅
- **cold-start**: elastic-mode globalization + 동역학일관 Hermite init → **Drake seed 없이 self-contained 수렴**(eq 1e-16, ineq 1e-6) ✅

### 7.4 폐루프 크로싱 — ❌ 추종기서 낙상
플랜→WBIC 참조 변환(`tamols_track/export_traj`)→A(WBIC+SRBD MPC) 추종. **접근(t=0~1.25s)은 깨끗이 추종**(yaw~0°·tilt<15°·falls=0), 그러나 **크로싱(t=1.25~1.75)서 계획을 1.7× 오버슛**(로봇 x=1.037 vs 계획 종단 0.724): 앞발 원거리 착지→몸통 급전진→**CoM이 앞 스탠스발 넘어 전방tip**→yaw 스핀 19→80°→낙상.
- **근본원인=추종기**: A의 SRBD MPC는 **x,y 위치를 추종 안 함(Qdiag px=py=0), 속도만 추종하는 정상상태 조절기**라 크로싱의 감속 GRF를 2발 대각지지서 못 만듦. 변형(yaw홀드 W_YAW·1.6× 느린재생·발판/base 대칭 solve) 전부 다른방식 실패.
- ★이전 "falls=0 크로싱" 보고는 **marginal·재현 불가**였음을 정정.

### 7.5 자산·결론
자산: `quad/cpp/tamols/`(C++ 솔버)·`quad/cpp/src/terrain_map.hpp`(footScore/edgeSDF/slope)·`quad/tamols/tamols_02leg.py`(Drake 레퍼런스)·`quad/mjcf/quad_tamols_gap.mjcf`(갭 검증 씬). TOWR(오프라인 발판최적화)는 우리 로봇 ROM서 미수렴(상세=`TOWR_개발리포트.md`).
**결론**: C++ TAMOLS 솔버는 완성(실시간+cold-start)이나 폐루프 크로싱은 A 추종기서 실패 = §1과 동일하게 **병목=추종기** → TAMOLS/TOWR 계획 재사용 + **RL 추종(DTC)** 이 남은 길.
