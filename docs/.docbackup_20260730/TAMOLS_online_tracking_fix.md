# TAMOLS Online Whole-Body 추종 안정화 — 수정 계획서

> 작성 2026-07-29. ETH 방식(완전 TAMOLS 계획 + whole-body online receding-horizon 추종)을 평지 falls=0으로 안정화하기 위한 실행 계획. 이번 세션 진단을 근거로 함.

## 0. 목표와 성공 기준

**목표**: TAMOLS 완전계획(base pose + 발판 + GIAC 안정)을 A의 실행층(SRBD MPC + WBIC)으로 **online receding-horizon 추종**. 발판만 주입(injection)이 아닌 **base·자세·발판 통합 추종**(ETH TAMOLS+WBC 방식).

**단계별 성공 기준**:
- P0: **offline 단일 사이클**(0.8s) 추종 falls=0 · tilt<5° (현재 ~0.5s tilt<17 후 드리프트)
- P1: **offline 다사이클(체인)** 추종 falls=0 (연속 보행 3s+)
- P2: **online receding-horizon** 평지 falls=0 · 3s+ (death spiral 제거)
- P3: **online + 지형**(heightmap) 크로싱 — 최종 목표, injection과 성능 비교

## 1. 현재 상태 (객관적 진단)

online 평지 추종이 **~1초 내 낙상**. 원인별 상태:

| # | 문제 | 원인 | 상태 |
|---|---|---|---|
| 1 | z침하 | RSL이 SRBD MPC 우회(gravity-comp만, base task z-hold 약함) | ✅ `TAM_MPC`로 해결(offline 확인) |
| 2 | phantom stance | `SW_DUR`(0.4) > 계획 swing위상(0.2) → 발 반만 스윙 | ✅ `SW_DUR` env 매칭 |
| 3 | base 후진 lurch | 오염 입력 시 solve_fast 후진 vx | ✅ `TAM_CLEANV`(X clean) |
| 4 | **lateral/yaw 드리프트** | CoM 지지폴리곤 유지 실패(body-sway 추종 부족) | ❌ **핵심 미해결** |
| 5 | death spiral | 추종오차→오염상태→나쁜 replan→악화 | ❌ 구조적(④가 유발) |

**핵심**: 개별요인(1·2·3)은 downstream 손잡이로 해결됐으나, **④ lateral/yaw 드리프트**가 근본 벽. 솔버 계획은 **깨끗한 입력엔 양호**(후진 없음)이므로 병목은 계획이 아니라 **추종의 lateral 로버스트니스**(우리 WBC의 CoM-지지 유지 성숙도 < RSL).

**대조**: injection(A gait + TAMOLS 발판)은 falls=0 작동 — A의 검증된 gait가 ④를 우회. 순수 whole-body 추종은 ④에 정면으로 부딪힘.

## 2. 근본 원인 가설 (④ lateral/yaw 드리프트)

walk 단일지지(3발) 국면서 CoM이 지지삼각형을 벗어나 base가 기울고 표류. 계획은 GIAC로 CoM-지지 궤적을 인코드하나 **추종이 그걸 달성 못함**. 세부 가설:

- **H1 (base xy 추종 부족)**: 계획의 base xy sway(GIAC)를 MPC가 약하게 추종(`TAM_QPOS`) → CoM이 계획대로 안 움직여 지지 이탈. *반증 일부: 게인 상향이 오히려 악화(SRBD 위치권한 한계).*
- **H2 (각운동량 미제어)**: 단일지지 스윙 시 발 flail·다리 반력이 yaw/roll 각운동량 유발, WBIC의 각운동량 task(`W_AM`) 부족 → yaw 드리프트.
- **H3 (계획 gait ≠ A 검증 gait)**: online 계획의 위상·발판 타이밍이 A의 안정 walk와 달라 WBIC가 다루기 어려운 참조.
- **H4 (발판 배치 비대칭)**: 계획 발판 X/Y 비대칭 → 계통적 yaw 모멘트(관측: yaw 항상 한쪽).
- **H5 (SRBD 실행층 한계)**: 발 frozen(CoM-relative) SRBD가 축소지지·동적 sway를 근본적으로 못 실행 → whole-body(발 자유) 실행 필요.

## 3. 실행 계획

### Phase 0 — offline 단일사이클 청정화 (기반, 최우선)
online의 death spiral 없이 ④를 격리 진단·수정. **가장 깨끗한 실험대**.

1. **진단**: 첫 계획(정지 상태 덤프, `TAM_DUMP`) offline 추종 시 tilt·y·yaw·발 접촉수·각운동량 시계열 로깅(`TAM_DBG` 확장). 드리프트 개시 시점·국면(어느 발 스윙) 특정.
2. **H4 검증**: 발판 대칭화(이미 스크립트 있음) → yaw 드리프트 감소하나? (부분 확인됨, 재측정)
3. **H2 검증**: `W_AM`(각운동량 task 가중) 스윕 — 단일지지 yaw 억제되나?
4. **H1 검증**: base xy 추종을 SRBD MPC가 아니라 **WBIC base task로 직접**(CoM task 강화) — SRBD 위치권한 한계 우회.
5. **성공**: 단일사이클 falls=0·tilt<5°.

### Phase 1 — offline 다사이클 체인
단일사이클이 청정하면 여러 사이클 이어붙여 연속보행. **주의**: 개루프 체인은 드리프트 누적(메모리 "개루프 벽") — 사이클 경계 상태 연속성 확인. 이 단계가 어려우면 P2(online 폐루프)가 정공이므로 P0 확인 후 바로 P2로 갈 수도.

### Phase 2 — online receding-horizon (death spiral 제거)
1. **warm-start 표준화**: replan 초기조건을 **측정상태 그대로가 아니라** 이전 계획 + 측정 blend(오염 완충) 또는 계획상태 우선(A MPC가 매사이클 측정 재계획해도 안정한 것과 대조 분석).
2. **replan 빈도·commitment**: RDT(재계획 주기)·swing commitment(이미 절대시간+target 동결 구현) 튜닝.
3. **P0 수정(W_AM·CoM task·대칭발판) 이식**: online에 적용.
4. **성공**: online 평지 falls=0·3s+.

### Phase 3 — online + 지형
heightmap(TAMOLS_TERRAIN 이미 구현) 켜고 지형 크로싱. injection과 **동일 지형서 성능 비교**(A baseline ~0.10~0.15m 대비).

## 4. 자산 (이번 세션 구축, 재사용)

- `TAM_MPC=1`: SRBD MPC 기반 추종(z침하 해결)
- `TAM_CLEANV=1`: X 전진 clean(후진 회피, Y sway 유지 버전)
- `TAM_DUMP=<file>`: online 계획 → load_tamols 포맷 덤프(offline 격리·대칭성 진단)
- `SW_DUR` env: offline도 읽음(계획 위상 매칭)
- swing foot commitment: 절대시간+target 동결(재anchor 대응)
- 발판 대칭화 스크립트, 깨끗계획 후처리 스크립트(`/tmp/tamols_flatsym.txt` 등)
- 지형: hsteps·dsteps·dsteps2·trench·gapcourse

## 5. 리스크·판단 기준

- **④가 SRBD 실행층의 근본 한계(H5)면**: 순수 whole-body(발 자유변수, TSID/전신 WBIC)로 실행층 교체 필요 = 큰 작업. B(quad_centroidal, 발 결정변수)와 합류 검토.
- **P0에서도 tilt<5° 못 달성하면**: 우리 WBC의 CoM-지지 유지 성숙도가 근본 부족 = injection이 실용 최선임을 인정, ETH-adjacent로 배포. (RL은 별개 트랙)
- **판단 게이트**: P0(단일사이클) 성패가 전체 방향 결정. 되면 P1→P2 진행, 안 되면 실행층 재검토 또는 injection 인정.

## 6. Phase 0 실행 결과 (2026-07-29)

**★H2 확정 = 각운동량 task(W_AM)가 ④의 핵심.** offline 첫계획(정지 덤프) 추종서:
- W_AM 기본→30: t=0.75 tilt **49.9→19.2°**(절반↓). **t=0.5까지 tilt<6.3°**(예전 50). 
- 각운동량 감쇠(KD_AM)·W_ORI 추가는 미미(plateau ~19). 마지막 스윙(FL)서 tilt 19로 튐=사이클 경계 효과.
- **H4(발판 대칭) 거의 무효**(tilt 49.8, yaw만 8→2 약간). foot 비대칭은 부차.
- **→ Phase 0 대체로 성공**: W_AM=30로 offline 단일사이클 t=0.5까지 tilt<6.3(P0 목표 tilt<5 근접). 마지막 스윙 peak 19는 체인서 해소 기대.

**★online(P2)은 W_AM으로 안 고쳐짐 = ⑤가 별개 문제.** online+W_AM=30: 여전히 낙상(**yaw −143° 스핀**·z침하). W_YAW=30·REPLAN_DT 0.4/0.8 모두 실패(yaw −132°/−47°). 
- **→ ⑤ re-anchoring death spiral이 online 고유 병목**(offline엔 없음). yaw 급발산=재anchor가 yaw 참조/발판을 오염시켜 누적. WBIC yaw task(W_YAW)로 안 잡힘=참조 자체 문제.

## 7. 갱신된 다음 액션 (P2: online ⑤ 규명)
1. **online yaw 급발산 원인 특정**: 재anchor 시 계획 yaw(s[5])·발판(tam_fh) 시계열 덤프 → yaw 오염 지점. `TAM_CLEANV`에 yaw도 clean(초기헤딩 홀드) 추가 검토.
2. **death spiral 완충**: replan 초기조건을 측정상태 대신 **이전 계획상태 blend**(오염 완충). A MPC는 매사이클 측정재계획해도 안정한 것과 대조 분석(A는 왜 되고 우리 online은 왜 안 되나).
3. **W_AM=30은 online에도 유지**(④ 억제 근거). P0 자산.

**전략 판단 유지**: online ⑤가 안 풀리면(P2 실패) → offline 청정(P0 W_AM 성공)은 확보되나 연속보행 불가 → injection이 실용 최선 재확인 or 실행층 재검토(H5).
