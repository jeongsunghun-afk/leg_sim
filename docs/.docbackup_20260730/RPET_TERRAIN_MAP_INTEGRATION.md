# RPET_TERRAIN_MAP_INTEGRATION.md — elevation map 통합 브리핑 (terrain_z 심 확장)

> 실기 perception(elevation map)을 컨트롤러에 붙이는 통합 설계.
> 결론 먼저: **새 배선은 없다.** 기존 `terrain_z(x,y)` 포인트 쿼리 심(seam)을
> "레이어드 그리드맵 계약(TerrainMap)"으로 넓히는 것이 통합의 전부이고,
> 실제 난점은 배선이 아니라 **odom 프레임 정합 + 비동기/지연 처리**다.
>
> 관련 문서: 하이브리드 리포트 §8.3(ptgt·cs·q_ref 서브셋 — 본 맵이 ptgt 품질의
> 상류), leg-odometry/ZUPT 추정기 설계, latency 링버퍼 설계,
> `RPET_JUMP_MPC.md`(perceptive 착지), stepping stone 로드맵(TAMOLS 소비자).
> 핵심 문헌: TAMOLS = Jenelten et al., T-RO 2022, arXiv 2206.14049
> (공식 코드 없음 — §5-P2 조사 결과 참조).
>
> 전제: 파이프라인 ③의 perceptive가 `terrain_z(x,y)` 포인트 쿼리로 구현되어
> 있고 백엔드는 sim=mj_ray. "백엔드만 교체, 인터페이스 동일" 원칙 유지.

---

## 0. 설계 원칙 (전 Phase 공통)

1. **심은 하나** — 컨트롤러(③ Raibert), TAMOLS, nav가 전부 같은 TerrainMap
   계약에 붙는다. 소비자별 전용 경로를 만들지 않는다.
2. **컨트롤러는 절대 블록되지 않는다** — 맵은 수십 Hz, 컨트롤러는 1 kHz.
   읽기는 lock-free(더블버퍼 + atomic 스왑), 파생 레이어 계산은 루프 밖.
3. **sim/real 파리티** — sim 백엔드(mj_ray→grid)와 real 백엔드
   (elevation_mapping)가 같은 계약을 구현. 테스트는 sim에서 열화 주입으로 선행.
4. **드리프트는 상대기하로 이긴다** — 맵과 발판 타깃을 같은 odom 프레임에서
   다루면 절대위치가 드리프트해도 국소 상대기하(발↔지형)는 정확하다.

---

## 1. 계약(Contract) 정의 — TerrainMap

```cpp
// terrain_map.h — 백엔드 무관 계약
// sim 구현: MjRayTerrainMap (mj_ray 래핑 → 그리드 캐시)
// real 구현: ElevationMapAdapter (perception 노드 구독)
struct Submap {
  double res;            // [m/cell]
  double ox, oy;         // 원점 (odom frame)
  int    nx, ny;
  Grid   elevation;      // 원시 높이
  Grid   filtered;       // 평활화 높이 (컨트롤러 z()의 소스)
  Grid   slope;          // 경사 [rad]
  Grid   roughness;      // 국소 분산
  Grid   edgeSDF;        // 엣지까지 부호거리 [m] (음수=엣지 위)
  Grid   footScore;      // 발판 적합도 [0,1] (아래 §3.2 합성 규칙)
  uint64_t stamp_ns;     // 맵 타임스탬프 (지연 정합용, §4.3)
};

struct TerrainMap {
  double z(double x, double y) const;        // 포인트 높이 — 현 컨트롤러 그대로 사용
  bool   valid(double x, double y) const;    // 맵 밖/구멍/신뢰도 미달 → false
  Submap submap(double cx, double cy, double r) const;  // TAMOLS/planner용
};
```

계약의 소비자 지도:

| 소비자 | 사용 API | 용도 |
|---|---|---|
| ③ Raibert / ④ 몸통높이 | `z()`, `valid()` | 착지 높이 보정, com_h0+지형 — **현 로직 무변경** |
| footScore nudge (§5-P1) | `submap()` (footScore만) | 발판 타깃 미세 이동 |
| TAMOLS (§5-P2) | `submap()` 전체 | base+발판 동시 최적화 |
| nav/global planner (후순위) | `submap()` (slope/roughness) | 경로 비용 |
| RL 관측 (H2 연동) | `submap()` → height samples | 정책 지형 관측 — 같은 심에서 추출 |

**valid()=false의 의미 계약**: "모른다"이지 "평지다"가 아니다. 소비자별 폴백을
명시한다 — ③은 마지막 유효값 유지 후 타임아웃 시 평지 가정 + 보수 모드
(STEP_H 하향), TAMOLS는 해당 셀을 발판 후보에서 제외.

---

## 2. 아키텍처 — 어디서 무엇이 계산되는가

```
[perception 노드]  elevation_mapping (수십 Hz, odom frame, 로봇중심 스크롤)
      │ grid_map msg (or 서비스) — 레퍼런스 확정 대상 ①
      ▼
[맵 노드 = TerrainMapNode]  (맵 갱신율에서만 동작, 1 kHz 밖)
   - 수신 → odom 정합 확인 → 파생 레이어 계산 (filtered/slope/edgeSDF/footScore)
   - Submap 이중버퍼 back에 기록 → atomic 포인터 스왑 (front 공개)
   - 맵 지연 특성화 로깅 (stamp_ns vs now)
      │ front 포인터 (lock-free read)
      ▼
[컨트롤러 1 kHz]  z()/valid()만 — 스왑된 front를 읽기 전용 참조
[TAMOLS / planner 50 Hz~이벤트]  submap() — 동일 front 참조
```

- 파생 레이어(edgeSDF·footScore·slope)는 **맵 갱신 시 1회만** 계산. 1 kHz
  루프 안에서 어떤 레이어 계산도 금지 (원칙 2). elevation_mapping 계열이면
  grid_map 필터 체인이 slope/roughness를 이미 제공할 가능성 — 레퍼런스에서
  확인(확정 대상 ③), 있으면 맵 노드는 edgeSDF/footScore만 추가 계산.
- 스왑 패턴은 기존 CMDFILE/STATE_PUB 원자교체와 동일 — 코드 재사용.

---

## 3. 파생 레이어 계산 규칙 (맵 노드)

### 3.1 기본 레이어
- `filtered`: elevation의 중앙값/가우시안 평활 (구멍은 inpainting 후 valid 마스크 유지)
- `slope`: filtered의 그래디언트 노름 → atan
- `roughness`: 국소 창(예: 3×3 발 크기) 표준편차
- `edgeSDF`: "낙차 > 임계(예: 0.05 m)" 셀 집합까지의 부호거리 변환

### 3.2 footScore 합성 (초기 규칙 — 튜닝 대상)
```
footScore = w1·sat(1 − slope/slope_max)
          + w2·sat(1 − roughness/rough_max)
          + w3·sat(edgeSDF/margin)          // 엣지에서 발 반경+여유만큼 떨어질 것
          × valid mask
초기값: slope_max=25°, rough_max=0.03 m, margin=발 반경+0.03 m, w=(0.4,0.2,0.4)
```
- 이 스칼라 장 하나가 P1(nudge)과 P2(TAMOLS 발판 비용)의 공용 입력이 된다.
- 스코어 정의를 바꿔도 소비자 코드는 불변 — 계약 뒤에 숨긴다.

> **⚠ 점접촉(sphere 발) 주의**: 02_Leg 발=**점접촉**(발바닥 패치 없음). 그래서 footScore는
> "패치가 지형에 얼마나 밀착하나"가 아니라 **"이 한 점이 안전·안정한 지지점인가"** 다.
> ① **margin = 발바닥 크기가 아니라 = sphere 반경(foot_r) + 발배치 불확실성 σ_place**.
>    σ_place는 §6 열화스윕(latency/노이즈 발끝오차)에서 근거를 댄다 — 임의값 금지.
> ② **edgeSDF가 지배항**(sphere가 엣지서 굴러떨어지거나 갭에 빠지지 않게), slope 보조,
>    **roughness는 저비중**(패치 밀착이 무의미하니 뾰족돌기/구덩이 신호로만). 기본 w=(0.4,0.2,0.4).
> ③ sphere 양면: 지름만큼 작은 갭·거칠기를 bridging(험지 관대) / 엣지 굴러떨어짐(edge margin 결정적).
> ④ **게이트 무관**: footScore는 발배치 층이라 walk·trot·run·stairs 전부 적용(스윙발 타깃 이동).
>    불연속 지형은 정적안정·정밀배치인 walk/stairs가 본선, trot은 완만 지형.
> (구현: `cpp/src/terrain_map.hpp` MjRayTerrainMap — margin=foot_r+placement_margin.)

---

## 4. 실제 난점 3종 — 프레임·비동기·지연

### 4.1 프레임: odom 위의 로봇중심 스크롤 맵
- 맵 프레임은 **estimator의 odom** (leg-odometry/ZUPT 출력) 위에 태운다.
  world 절대 프레임 금지 — 드리프트가 발판 오차로 직결된다.
- 발판 타깃(ptgt)도 같은 odom 프레임에서 쿼리 → 절대위치가 흘러도
  "발과 그 앞 지형"의 상대기하는 정확 (원칙 4).
- 스크롤 방식(로봇중심 이동 창)일 때 Submap의 ox/oy가 매 갱신 이동함 —
  소비자는 좌표를 항상 (x,y) 절대 odom 좌표로 쿼리하고 인덱스를 저장하지
  말 것 (스왑 사이에 인덱스 의미가 바뀐다).
- 확정 대상 ②: perception이 주는 프레임이 odom인지 map인지, 스크롤 여부.

### 4.2 비동기: 더블버퍼 + atomic 스왑
- 맵 노드가 back 버퍼 완성 → `front.store(back, release)` → 이전 front는
  다음 사이클에 back으로 재사용 (2버퍼 순환). 컨트롤러는 `load(acquire)`
  후 그 포인터로 일관 읽기 — 갱신 중간의 찢어진 맵을 절대 보지 않는다.
- 컨트롤러 측에서 스왑 감지 시(포인터 변경) ③의 착지높이 보정이 계단식으로
  튀지 않도록 z() 소비부에 슬루(기존 V/VY/WZ 슬루 패턴) 적용.

### 4.3 지연 정합: 상태와 맵의 타임스탬프 맞춤
- 맵에는 perception 지연(취득→정합→맵 갱신)이 있다. 상태 x̂(t_now)로
  맵(stamp_ns = t_now − τ_map)을 그냥 쿼리하면, 이동 중 발판이 τ_map·v 만큼
  어긋난다 (0.1 s × 2 m/s = 0.2 m — 발판 하나가 통째로 어긋나는 수준).
- 대응: **latency 링버퍼를 맵에도 적용** — 상태 이력 버퍼에서 맵 스탬프
  시각의 자세 x̂(t_map)를 꺼내 그 기준으로 쿼리하거나, 맵 좌표를 현재
  odom으로 보정. 상태지연 특성화에서 만든 인프라 재사용.
- 선행 작업: **τ_map 특성화** — sim에서 인위 지연 주입으로 "지연 vs 발판
  오차 vs falls" 곡선을 먼저 뽑는다 (§6). 이 곡선이 real 지연 허용 예산이 된다.

---

## 5. 붙이는 순서 — 각 단계가 독립 검증되는 점진 통합

### P0. 백엔드 스왑 (최저 위험 첫 통합)
- [ ] `MjRayTerrainMap` 구현: mj_ray를 그리드로 캐시 → TerrainMap 계약 충족
      (sim에서 계약 자체를 먼저 검증 — real 연결 전에 심을 살아있게 만든다)
- [ ] `ElevationMapAdapter` 구현: 레퍼런스 확정 3종(§7) 반영
- [ ] 현 컨트롤러가 z()/valid()만으로 실맵 위 보행 — **로직 무변경으로 통과**
- 완료 기준: 평지+완만 험지에서 기존 sim 성능과 동등 (falls=0, 속도 추종 동등)
- 의미: perception→controller 심이 살아있음을 증명. 이후 단계는 소비자 추가일 뿐.

### P1. footScore nudge — "가난한 자의 TAMOLS"
- [ ] ③ Raibert 출력 ptgt를 반경 r_nudge(초기 0.08 m) 내 footScore 최고 셀로
      이동. 이동량 상한 + 히스테리시스(매 틱 후보가 바뀌어 발이 떨리지 않게)
- [ ] cs/스윙 위상은 무변경 — 순수하게 "어디" 만 개선
- 완료 기준: 엣지 인접 지형에서 엣지 위 착지율 유의미 감소, 평지 회귀 무열화
- 의미: 옵티마이저 없이 발판 품질 확보. stepping stone 이전의 저비용 중간 단계이자,
  TAMOLS 도입 후에도 폴백 모드로 잔존.

### P2. TAMOLS 소비 — base+발판 동시 최적화
- [ ] TAMOLS(또는 동급 지형 인지 planner)가 submap()을 읽어 발판+base 포즈
      최적화 → ptgt·base ref를 ③④ 자리에 공급, WBIC 추종
- [ ] 갱신은 touch-down 이벤트 기반으로 시작 (DTC 방식), 예산 되면 상향
- [ ] P1 nudge는 TAMOLS 실패/타임아웃 시 폴백으로 유지 (graceful degradation)
- 완료 기준: 불연속 발판(stepping stone 축소판)에서 P1 대비 성공률 우위
- 의미: H2(DTC화)의 planner 상류가 완성됨 — RL 관측 서브셋(ptgt)의 품질 상한이
  여기서 결정된다.

#### P2 조사 결과 (2026-07): TAMOLS 원논문은 있고, 공식 코드는 없다

**원논문 (확정):** Jenelten, Grandia, Farshidian, Hutter, "TAMOLS: Terrain-Aware
Motion Optimization for Legged Systems", IEEE T-RO vol. 38 no. 6, pp. 3395–3413,
2022. arXiv **2206.14049**. 핵심: heightmap 조건 하에 base 포즈+발디딤 동시
최적화, 나쁜 국소최적 회피용 **graduated optimization**(맵을 여러 평활화
수준으로 점진 정밀화 — §3.1 `filtered` 레이어 설계와 직결), 접촉력 없는
안정성 판별식, direct collocation 전사, **온라인 10 ms 이내 풀이**
(실시간 planner 후보인 근거).

**공식 레퍼런스 코드: 없음.** ETH RSL은 TAMOLS(및 DTC)를 오픈소스로 공개하지
않았다. 서드파티로 tamols-rl(ianpedroza, Go2 + Isaac Gym: TAMOLS 발디딤
planner 파이썬 재구현 + RL 하위 제어기 계층 결합)이 존재 — 구조가 H2 구도와
같아 참고 가치는 있으나 학생 프로젝트 수준, 코드 품질 무검증 전제로 볼 것.

**P2 실행 시 선택지 3종:**

| 선택지 | 내용 | 판단 |
|---|---|---|
| ① 논문 기반 자체 구현 | collocation + graduated opt. 정식화는 논문에 상세, aligator/crocoddyl 경험으로 감당 가능 | 공수 큼 — stepping stone이 실로드맵에 오를 때만 |
| ② 공개 대안 채택 | 같은 그룹 후속작 Grandia et al. "Perceptive Locomotion through NMPC"(T-RO 2023) 계열, OCS2 legged 모듈(B안 시절 simple-mpc/OCS2 생태계 접점). perception 쪽(elevation_mapping, grid_map)은 공개 — §1 real 백엔드로 그대로 사용 | planner 대안 1순위 후보 |
| ③ P1 연장 운용 | footScore nudge("가난한 자의 TAMOLS")로 발판 품질 대부분 확보 | **기본값 — 당분간 이것으로 충분** |

**권장: 당분간 ③(P1로 충분), stepping stone급 태스크가 실제 로드맵에 올라오는
P2 착수 시점에 ①↔② 재평가.** 본 절의 "TAMOLS(또는 동급)" 표기는 이 유보를
의도한 것.

---

## 6. sim 병행 검증 — 맵 열화 주입 (하드웨어 전 규명)

sim 백엔드에 열화를 주입해 "맵이 나빠지면 발판이 얼마나 망가지나"를 정량화:

| 열화 축 | 주입 방법 | 측정 |
|---|---|---|
| 노이즈 | elevation에 σ_z 가우시안 (0~0.03 m 스윕) | 발판 오차, falls |
| 지연 | 맵 stamp 인위 지연 (0~200 ms 스윕) | §4.3 곡선 — 지연 예산 산출 |
| occlusion | 진행방향 부채꼴 invalid 마스크 | valid()=false 폴백 동작 검증 |
| 드리프트 | odom에 인위 바이어스 주입 | 상대기하 강건성 확인 (원칙 4 검증) |

- 각 축은 P0 완료 직후부터 돌릴 수 있다 (P1/P2와 병행).
- 이 표의 결과가 곧 real 레퍼런스에 요구할 스펙(허용 지연, 필요 정밀도)이 된다
  — "레퍼런스를 받아서 맞추는" 게 아니라 "요구 스펙을 들고 받으러 가는" 순서.

---

## 7. 레퍼런스 수령 시 확정할 3가지 (어댑터 사양 결정 항목)

1. **출력 타입/메시지**: grid_map(ROS) / costmap / 서비스 쿼리 중 무엇인가
   → ElevationMapAdapter의 형태(구독 vs 폴링) 결정
2. **프레임 규약**: odom/base/world 중 무엇에 실리나, 로봇중심 스크롤 여부,
   드리프트 처리 주체 → §4.1의 정합 방식 확정
3. **rate·latency·기존 레이어**: 갱신율과 지연 특성, footScore/edgeSDF/slope를
   perception이 이미 주는지 → 맵 노드가 계산할 레이어 목록 확정

---

## 8. 함정 사전 등록

- **스왑 순간의 발판 점프**: 맵 갱신으로 z()가 계단식 변화 → ③ 착지높이가
  틱 사이에 튐. 대응: §4.2 슬루 + swing 중인 다리의 ptgt.z는 스윙 시작 시점
  값으로 동결(스윙 중 재쿼리 금지, touch-down 후 갱신).
- **valid 경계에서의 채터링**: 로봇이 맵 가장자리에서 valid↔invalid 반복 →
  보수 모드 진입/이탈 발진. 대응: valid 판정에 히스테리시스(시간+공간 마진).
- **footScore 국소 진동**: 이웃 셀 점수가 비슷하면 nudge 후보가 매 갱신 바뀜.
  대응: 현 후보에 보너스 가중(sticky), 이동량 데드밴드.
- **인덱스 보관 버그**: 스크롤 맵에서 (i,j) 저장 후 스왑되면 다른 위치를 가리킴.
  대응: 소비자는 항상 odom (x,y)로 보관·쿼리 (§4.1). 코드리뷰 체크 항목화.
- **1 kHz 오염**: 누군가 편의로 루프 안에서 submap()을 부르는 순간 원칙 2 붕괴.
  대응: submap()에 호출 스레드 assert (컨트롤러 스레드에서 호출 시 abort in debug).
- **레이어 계산 지연 누적**: 맵 노드 계산이 맵 주기를 넘으면 지연이 스택.
  대응: 맵 노드 처리시간 로깅 + 예산 초과 시 레이어 해상도 강등.

---

## 9. 파일 구조 제안

```
rpet_terrain/
├── include/rpet_terrain/
│   ├── terrain_map.h              # §1 계약 (헤더 온리)
│   └── layers.h                   # §3 파생 레이어 규칙
├── src/
│   ├── mjray_terrain_map.cpp      # P0 sim 백엔드
│   ├── elevation_map_adapter.cpp  # P0 real 백엔드 (§7 확정 후)
│   ├── terrain_map_node.cpp       # §2 맵 노드 (레이어 + 더블버퍼)
│   └── foot_nudge.cpp             # P1
├── eval/
│   ├── degradation_runner.py      # §6 열화 주입 스윕
│   └── terrain_metrics.py         # 발판 오차, 엣지 착지율, falls
└── tests/
    ├── test_contract_parity.py    # sim/real 백엔드 동일 쿼리 → 동일 의미 검증
    ├── test_swap_consistency.cpp  # 더블버퍼 찢어짐 없음 (스트레스)
    └── test_latency_alignment.py  # §4.3 스탬프 정합
```

## 10. 마일스톤 요약

| 단계 | 산출물 | 완료 기준 | 의존성 |
|---|---|---|---|
| P0 | TerrainMap 계약 + 양 백엔드 + 백엔드 스왑 보행 | 기존 성능 동등 (로직 무변경) | 레퍼런스 §7 확정 (real측만) |
| §6 | 열화 스윕 리포트 | 지연/노이즈 예산 곡선 | P0 (sim측) |
| P1 | footScore nudge | 엣지 착지율 감소, 평지 무열화 | P0 |
| P2 | TAMOLS 통합 | 불연속 발판에서 P1 대비 우위 | P0, P1(폴백) |

권장 순서: P0(sim) → §6 병행 → P1 → [레퍼런스 수령, §7 확정, P0 real] → P2.
sim측 P0와 §6은 레퍼런스 없이 오늘 시작 가능 — 확정 3종은 어댑터 한 파일에만
격리되어 있으므로 대기 항목이 전체를 막지 않는다.

## 11. 커밋 규율

- 단계 브랜치: `feat/terrain-p0-contract`, `feat/terrain-p1-nudge`, ...
- 결과에 전제 병기: 맵 해상도/갱신율/지연 조건, sim 백엔드 여부 명기
- §6 열화 곡선은 원시 로그 보존 — real 레퍼런스 스펙 협상 자료
