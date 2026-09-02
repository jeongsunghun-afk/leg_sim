# 표준 발판(Standard Foothold) + TAMOLS 방향

> RL에 붙이는 발판생성이 3갈래(TAMOLS 캐시 / D1 OCS2 / 휴리스틱 snap)로 쌓이며 **앵커링 의미가 소스마다 달라진** 문제를 표준화한다.
> 관련: [[dtc-17dof-development]], `TAMOLS_개발리포트.md`, `D1_OCS2_개발리포트.md`.

## 문제 요약 (2026-08-20 진단)
하나의 `_foothold_target` 버퍼에 **의미가 다른 발판**이 섞여 들어감:

| 소스 | xy 결정 | z | 지형-고정? |
|---|---|---|---|
| stepping / parkour (snap) | Raibert점 → **돌/안전셀 중심 스냅** | 지형 | ✅ 지형 못박힘 |
| gap 캐시 | base-상대(재앵커) | 평지 | △ lookup이 gap거리 정합 |
| 계단 캐시 · D1 | **base-상대(재앵커)** | tread 표면 | ❌ **xy가 로봇 따라다님** |

→ 재앵커 경로(gap/계단/D1)에서 발판 **xy는 지형이 아니라 base 기준 공칭 보행패턴**, 지형정보는 **z에만**.
이게 (1) 발판 마커가 로봇 발밑을 따라다니는 인상, (2) **foothold_track이 늘 느슨(~0.10)** 한 근본 원인. 진짜 DTC(NMPC가 절대 지형좌표 발판 계획→RL 추종)와의 구조적 차이.

## 표준 발판 정의 (canonical)
1. **표현** = **월드 프레임 지형-고정 절대 좌표**. 진행좌표 `s = base_x − x0`로 플랜에서 선택. xy는 코리도/지형에 못박히고 **라이브 base의 lateral/longitudinal 오차에 안 흔들림**.
2. **앵커** = 라이브 base가 아니라 **진행/코리도 참조**:
   ```
   ref_base_xy(s) = (x0 + s, lane_y)         # 코리도 센터라인(라이브 base 아님)
   world_xy       = ref_base_xy(s) + Rz(ref_yaw)·foot_rel_xy(s)
   world_z        = terrain_surface(world_xy)   # tread/stone top
   ```
   (등가: 재앵커 후 xy를 지형 feature[tread/stone]에 스냅.)
3. **obs** = 정책엔 **base-상대**로 인코딩(그대로). 단 타겟이 지형-고정이므로 이제 base-상대 obs가 **"지형 발판이 앞-좌 X만큼"이라는 지형 신호를 담음**.
4. **reward** = 지형-고정 절대 타겟 추종 → **xy 추종이 지형 신호를 담음**(느슨함 해소).
5. **소스 통일** = TAMOLS 캐시 / D1 / 휴리스틱 전부 이 인터페이스로 emit. 스냅 경로(stepping/parkour)는 이미 준수 → **재앵커 경로(gap/계단/D1)를 이 표준으로 이관**.

## TAMOLS 방향
- TAMOLS 캐시와 D1을 **동일한 "발판 플랜 공급자"** 로 취급, 뒤에 **단일 지형-고정 앵커링 인터페이스**.
- **Go2 계단 TAMOLS 캐시 생성**(현재 계단 캐시=02_Leg base_h 0.52뿐 → 대칭 커버리지).
- **gait 지형별 표준 확정**: gap=trot / 계단·stepping=walk (§P2.8 근거=walk가 tread 발판 10× 정밀·정적안정 3발지지).
- **fn/rn 전 생성기 auto**(flat-solve): 02_Leg 하드코딩(0.51/0.06) 제거.

## 하나씩 해결 순서
1. **#1 지형-고정 앵커링 표준** — env 통일 헬퍼, D1/stair 경로 이관, 플래그 A/B. 학습으로 foothold_track 개선 검증. ← 표준 핵심
2. **#2 Go2 계단 TAMOLS 캐시** — `cache_gen_go2_stairs`(Go2 기하 z0 0.34·hip±0.1934/0.142).
3. **#3 gait 표준화** — 지형별 확정 + 문서.
4. **#4 fn/rn auto 통일** — `cache_gen.cpp` 하드코딩 제거.

---

## gait 지형별 표준 (#3 확정)
코드는 이미 지형-적절하게 동작(생성기 기본값). 이를 **표준으로 확정**:

| 지형 | gait | 근거 | 코드 |
|---|---|---|---|
| gap | **trot** | 대각쌍, 빠른 횡단(관용지형) | `cache_gen`/`cache_gen_go2` = OnlineCfg 기본(walk=bound=false=trot) |
| 계단 | **walk** | 정적안정 3발지지·tread 발판 10× 정밀(§P2.8) | `cache_gen_stairs`/`_go2_stairs` = `apply_cache_gait` 기본 walk |
| stepping | **walk** | 좁은 돌 정밀배치·GIAC 유효(§P4c) | RL `GO2_GAIT=walk` |
| D1(계단) | (OCS2 gait 스케줄) | NMPC 자체 접촉계획 | D1 참조 |

→ gap generator는 암묵적 기본(trot), stair generator는 명시적(`TAMOLS_GAIT`). 캐시 churn 방지 위해 gap을 명시화하진 않음(기본이 이미 trot). 지형별 gait는 **캐시에 baked**되므로 env는 지형에 맞는 캐시를 선택만.

## 진행 기록 (2026-08-20)
- **#1 지형-고정 앵커링** — Go2 D1 경로 구현+A/B 학습 진행중(`GO2_FOOTHOLD_ABS` 플래그, v4 vs v3 재앵커).
- **#2 Go2 계단 TAMOLS 캐시** — ✅ `cache_gen_go2_stairs.cpp`(base_h0.34·Go2 hip) 생성·빌드·검증(24셀 실패0), 서버 `direct/go2/tamols_stair_cache_go2/` 스테이징. **물리 발견: Go2 저스탠스(0.34)라 step_h 0.12~0.15서 앞발이 up-tread에 못 닿음**(02_Leg 0.52는 됨) = Go2 계단 도달 한계. RL 배선은 D1이 이미 Go2 계단 커버하므로 선택적 후속(TAMOLS vs D1 A/B용).
- **#3 gait 표준** — ✅ 위 표로 확정(코드 무변, 이미 지형-적절).
- **#4 fn/rn** — ⚠️ **블라인드 통일 불가로 판명**. 02_Leg 하드코딩 게이트(fn0.51/rn0.06)가 **실제 flat-solve 공칭(fn≈0.375/rn≈-0.185)과 불일치**(스테일). 그러나 배포 quad17 gap 캐시(작동 terrain_level~9)가 이 게이트로 빌드됨 → auto 재생성=캐시 변경=quad17 RL 영향. **생성기를 배포캐시와 일치(하드코딩) 유지+불일치 코드 문서화**. Go2는 처음부터 auto(정합). 진짜 수정=별도 re-cal+quad17 gap RL 재검증(블라인드 금지).

## #1 결과 — 지형-고정 앵커링 A/B (2026-08-20)
`GO2_FOOTHOLD_ABS` 플래그로 Go2 D1 경로 구현. **smoke 입증**: base +0.1m/+0.05rad 이동 시 발판 world_y 이동 = OFF(재앵커) +0.100 / **ON(지형-고정) 0.000**(코리도 못박힘). v4(ON) vs v3(OFF) A/B:

| | v3 재앵커 | v4 지형-고정 |
|---|---|---|
| terrain_level | ~3.9 (돌파 iter~450) | 3.5↑ (돌파 iter~333) = **등반 유지** |
| foothold_track | ~0.10 | **~0.10 (변화 없음)** |

**결론:** 지형-고정 앵커링은 **표현으로서 올바르고**(진짜 DTC 의미론=지형 신호 보유, smoke 입증) **등반에 무해**(오히려 돌파 약간 빠름). **단 foothold_track를 못 올림** → 발판 추종의 느슨함은 재앵커가 아니라 **soft scale(0.3)+연속 tread가 정밀배치 미강제** 탓(전체 조사결론=관용지형 발판 잉여와 일치). **∴ 표준=지형-고정 채택**(올바른 표현·무해)하되, 추종 tightness는 별개 레버(scale↑ or 지형강제). **선택적 후속: 지형-고정+높은 foothold scale이 재앵커+높은scale(과추종 게이밍)과 달리 tight한 지형 발판추종을 여는지** = 표준이 실제로 뭔가 unlock하는지 테스트.

## 후속 결과 — 지형-고정 + 높은 foothold scale (v5, 2026-08-20)
표준(지형-고정)이 발판을 실제로 유용하게 만드는지 검증: v5 = 지형-고정 + foothold scale **0.3→1.0**(2048 envs, GPU1). 재앵커+높은scale은 과추종 게이밍(foothold 폭주+terrain_level 7.9→2.5 후퇴)이었음.

| | v4 지형-고정 scale0.3 | **v5 지형-고정 scale1.0** | 옛 재앵커 scale1.0 |
|---|---|---|---|
| foothold_track | ~0.10 | **~0.45 (4~5×↑)** | 폭주(게이밍) |
| terrain_level | ~4.1 (최종) | ~2.5 (후퇴 없음) | 7.9→2.5 후퇴 |

**결론:** **지형-고정 + 높은 scale = tight한 지형 발판추종(0.45) + 게이밍 없음** ↔ 재앵커+높은scale은 게이밍. **∴ 표준의 실질 가치 확정: 지형-고정은 발판을 "게이밍 없이 진짜 추종 가능한 지형점"으로 만든다**(재앵커는 base-상대 패턴이라 높은scale서 corrupt). 단 climb은 v4보다 낮음(2.5 vs 3.9)=①정밀 발판추종이 연속 tread선 climb과 약한 trade-off(tread는 정밀 xy 불필요=관용지형 발판잉여 재확인) ②env수 confound(2048 vs 4096). **∴ 언제 tight를 쓸지는 지형 의존: 연속 tread/관용지형=soft(v4) 유리, 정밀배치 강제 지형(stepping stone·gap)=tight(v5) 유리.** 표준 자체(지형-고정 표현)는 항상 올바르고 무해; scale은 지형별 튜닝.

### 정의적 비교 — confound 제거 (v5b, 4096, 2026-08-21, GPU2 전용)
v5(2048) climb의 env수 confound를 제거하려 v5b = 지형-고정 + scale 1.0 **4096**(v4와 동일)을 재실행. 최종(iter 2999, seed 42 동일):

| 4096 동일 | v4 soft (scale 0.3) | **v5b tight (scale 1.0)** |
|---|---|---|
| foothold_track | 0.10 | **0.46 (4.6×↑)** |
| terrain_level | 4.12 | **3.26** |

**tight 발판추종의 순수 climb-cost = 4.12→3.26 ≈ 21%**(v5 2048의 2.86보다 v5b 4096이 3.26으로↑=낮은 climb 일부는 env수 탓). **게이밍 붕괴 아님**(terrain_level 유지). **∴ 최종: 지형-고정 표준은 발판을 게이밍 없이 tight 추종가능(0.46)하게 만들고, 그 대가는 연속 tread에서 climb ~21%↓(정밀 xy 불필요한 지형이라 예상된 완만한 trade-off). 정밀배치 강제 지형(stepping/gap)에선 이 tight 추종이 순이득이 될 것.**

---

## ★표준 재조립 (2026-08-27 사용자 확정)

**진단:** 기존 발판층 = WTW 볼트온 — 발판 점 xy + 상수 스윙(cmd9) + xy-only touchdown 보상. same-height(z 상수)에서는 매치드 0.12 vs 9.00으로 발판층 필수성이 실증됐으나, **표준(DTC/TAMOLS)과의 구조 갭**이 varied-height에서 드러남: ①발판 생성이 지각(elevation map) 미소비(특권 테이블) ②스윙이 타겟 Δz 무지 ③추종 보상에 z 없음.

**표준 사슬로 재조립:**
```
elevation map → 3D 발판 생성 → 지형-인지 스윙 궤적 참조(TAMOLS h_s1/h_s2) → 3D 추종 보상
```

| 단계 | 내용 | 게이트 |
|---|---|---|
| P1 (env) | 3D touchdown 오차 / 타겟-조건 스윙 apex+맵 클리어런스 / 타겟 z=RayCaster 지각(테이블 fallback) | GO2_FH_Z3D · GO2_SWING_TERRAIN · GO2_FH_Z_FROM_HMAP |
| P2 (TAMOLS) | footfall별 스윙 스플라인 export (공간 인덱스, 시간클록 無) | swing_L*.csv |
| P3 (실험) | varied-height 매치드 **A(emergent) vs D(표준사슬)** — same-height와 동일 엄밀도(유일차이=발판·스윙층, 같은 보행기준, 렌더+z텔레메트리 검증) | — |

**승계 불변식:** soft scale(0.3)·공간 타겟+next-ahead(follower deadlock 방지)·레인게이트·anti-crawl·오프라인 게이트 선행·매치드 프로토콜·중반판단 금지.

**함정필드 경과 (P5m~p):** 오프라인=greedy 붕괴 vs A* 전해결(선택-필수 지형 존재 증명, 성과) / RL=v2·v3 실패(입구 deadlock→next-ahead 수정 후 C4 진행중). C4 실패 시 함정필드 RL은 표준 사슬(P3) 이후 재평가.
