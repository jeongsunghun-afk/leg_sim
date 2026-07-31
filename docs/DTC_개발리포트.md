# DTC 개발리포트 — Deep Tracking Control (MPC 교사 + RL 추종)

> **상태: 활성 (DTC 트랙 정본) · 2026-07-31.** 17-DOF quad(02_Leg) 위 DTC(모델기반 planner → RL 추종). **P0(자산)·P1(속도 워커) 완료 · P2(절차적 발판 tracker: IK obs로 lazy-agent 규명) 완료 · ★P2.5 실제 TAMOLS 오프라인 캐시 파이프라인 구축 → 4096-env 학습 착수(초기 foothold_track↑) · P3(지형 커리큘럼·CVAE) 진행.**
> 이 문서 = DTC 트랙의 개발 기록·설계 근거. (전신 = MPC–RL 하이브리드 전략 리포트; 실행 트랙이 DTC로 좁혀져 이 이름으로 통합.) 학습법 심화는 `rl_module_train.html`, 작업 메모리는 `dtc-17dof-development`.

---

## 1. DTC란 — 패턴 A(MPC가 교사) (★우리가 택한 하이브리드)

모델기반 planner(TO/MPC)가 최적 레퍼런스를 생성하고 RL 정책이 강건하게 **추종(tracking)**. 원전 = **DTC(Jenelten et al., Science Robotics 2024, arXiv 2309.15462)**. TO의 정확한 발 배치·계획 + RL의 강건성·반사를 결합.

```
[온라인 planner: TAMOLS]              [RL 정책 (PPO, 오프라인 학습)]
지형 elevation map + 명령              관측 = 고유수용감각 + 지형 샘플
  → 발디딤 + base 포즈 동시 최적화          + 레퍼런스 "작은 서브셋":
  → touch-down 시점마다 재계획              · 2D 발디딤 좌표 · IK 목표 관절각 · 접촉 스케줄
       └──────── 레퍼런스 ──────────────┘  → 관절 명령 (고주파 추종 + 반사)
```

DTC가 잘 되는 4가지 (우리가 그대로 가져올 것):
1. **레퍼런스는 "작은 서브셋"만 노출** — 발디딤 2D·**IK 목표 관절각**·접촉 스케줄. 이 정보 병목이 모델기반 취약점(상태추정·비전 오차)을 가려 강건성·planner 교체 불변성을 만든다(다른 NMPC로 zero-shot 교체해도 발디딤 오차 3cm).
2. **IK 목표 관절각을 관측에 포함**한 것이 학습 수렴의 결정타(Fig 7D ablation 실증). ← ★우리 P2가 처음에 빠뜨렸다가 재발견한 바로 그 신호(§3.P2).
3. **가변 업데이트** — touch-down 순간에만 재계산. 50Hz로 올리면 실패율 7.11%p 추가 감소.
4. **레퍼런스 다양성 = 일반화** — 지형·명령·초기조건 randomize를 레퍼런스 생성 단계에서 확보.
- **성능(원전)**: 발디딤 평균 오차 **2.3cm**, 흔들 널빤지(맵오차 0.4m)·비전차단 계단 생존, 0.6m 틈새·1.8m 외나무다리 100%.

**왜 이 패턴인가**: 모델기반 폐루프의 상한은 **연속·구조화 지형**(A footScore/B heightmap = 슬로프 등반 OK, discrete stepping stones 붕괴). 임의 불연속 험지 강건 크로싱은 **RL 몫**으로 확정(2026-07-22 경계, `TOWR_개발리포트.md`). DTC = 그 둘을 잇는 다리(모델기반 정밀 계획 + RL 강건 추종).

---

## 2. 목표 아키텍처 — Kim 2025 (Raibo, Hwangbo/KAIST)

DTC(2024, ETH)보다 우리 목표(**사족·불연속지형**)에 더 가까운 **Kim et al. 2025 (arXiv 2506.02835, Raibo)** 를 목표 아키텍처로 삼는다. planner+tracker = DTC 구조를 갭·stepping·벽·4m/s로 발전. 핵심 3:
1. **tracker** = MLP actor + **GRU state estimator**(= 우리 ActorCriticRMA 골격) · 발판 추종 · target-index 접촉 0.06s 갱신 · 뒷발=앞발자리 · 보상(target/slip/impact/foot_gather 25cm/bound).
2. **competitive CVAE map generator**(적대적 커리큘럼) — tracker가 9.3/10 넘으면 CVAE 재학습해 더 어려운 지형 자동생성(벽주행·1.6m 갭). → P3 커리큘럼 도입 검토.
3. **planner** = 샘플링+순차필터(performance·spike·collision)+8후보 물리 롤아웃, 충돌=경계추정 MLP(RMS 2.27cm).
- ★한계 = 온보드 지각 없음(Vicon+사전맵) → 우리 perception/SLAM 트랙이 채우는 상보 갭.

---

## 3. 개발 현황 — P0 ~ P3

### P0 — 17-DOF quad 자산 (USD + cfg) ✅
MJCF(`quad_real_17dof_waist_sphere.mjcf`, 17관절=4다리×hip/thigh/calf/foot + FB_waist) → USD 변환 + `rga.py` `QUAD_17DOF_CFG`(Peak토크 hip/thigh84·calf126·foot100.8·waist84, self-collision on). 검증: **17관절·22바디·spawn z 0.52·물리안정** — 통과.
- USD 디버깅 3건: ①worldBody spurious ArticulationRoot 제거 ②baked z 0.5235 중복→init z=0 ③meshdir 절대경로 중복→상대경로. + 접촉 3건(instanceable 콜라이더·발 sphere·`/Robot/Base/.*` prim). ★근본해결=MJCF→USD 재변환서 지면·중첩·발콜라이더 처리(현재 런타임 workaround).
- ★교훈(사용자): 학습 전엔 "기립 안정" 판단 무의미 = P0 기준은 로드·관절·높이·물리안정뿐.

### P1 — 속도 워커 (velocity tracking) ✅
`Quad17-Velocity-Direct-v0`(hind_leg env 2→4다리+허리, quad trot, ActorCriticRMA/PPOParkour). action=17·obs=proj_grav3+cmd3+joint 17×3+trot clock4. 크라우치 default. **수렴: model_31200, reward~31, epLen 999**.
- ★즉시종료 버그(epLen 1.00·무학습): USD 내 **유령 바닥면 `/Robot/worldBody/floor`** 가 step0 스퓨리어스 접촉→base_contact 종료. 수정=`_disable_baked_floor_collision()` + init pos z=0.57. → epLen 999 회복. **교훈: 스모크 통과 ≠ 작동**(감시가 잡음).

### P2 — 발판 추종 tracker (진행 중) ★오늘의 핵심
DTC tracker를 **절차적 랜덤 발판**(실제 TAMOLS의 대역, from-scratch)으로 구현. obs에 발판 블록 +28, touchdown XY `−log(err²+eps)` 추종보상, **ablation gate**(발판 obs를 scramble했을 때 발 배치가 나빠져야 = 진짜 추종)로 검증.

**진단 여정 (전부 gate로 검증, 코드 위험 0으로 원인 좁힘):**

| 시험 | 방법 | gate(none vs scramble) | 결론 |
|---|---|---|---|
| 1차 구현 | 발판 obs+28·touchdown 보상 | 17.18 ≈ 17.22 cm (model_1500), 16.63 ≈ 16.59 cm (model_3000) | ❌ **obs 완전 무시** |
| ② 탈상관 | jitter 0.05→0.18 | 25.56 ≈ 25.57 cm | ❌ gap 0 |
| ③ 인센티브 | foothold scale 1→10 | 21.94 ≈ 21.84 cm | ❌ gap 0 |

- **1차 실패 원인 (DTC 논문 대조로 확정)**: ①**lazy-agent** — DTC가 명시 경고한 실패("optimizer가 상태에 적응→에이전트가 게으르게 보상 획득"). 우리 절차적 발판이 자연 걸음새 위치(±5cm)라 obs 안 읽어도 공짜 보상. ②**IK 목표 관절각 obs 누락** — DTC Fig 7D의 #1 학습가능성 레버("foothold accuracy is intrinsically complicated to learn"). §1의 그 신호를 우리가 빠뜨렸음.
- ②·③ 반증 → 원인 = **탈상관·인센티브가 아니라 학습가능성**(직교좌표 발판 obs를 관절 액션으로 매핑하는 걸 정책이 못 배움).

**① IK 목표 관절각 obs 구현 + 물리 검증 ✅ (2026-07-30):**
- Jacobian 1차 IK: 다리별 `Δq = J⁺·(p_target − p_foot)`(hip/thigh/calf 3관절) → obs **89→101(+12)**. ablation을 IK 블록까지 확장(cartesian+IK 동시 blind = gate 유효성).
- ★검증(스모크≠작동이라 값 검증): FD vs 해석 Jacobian **MATCH(rel 0.13~1.2%)** · 다리 간 누설 0 · e2e(dq가 발을 목표로?) **cos=1.000·mag 1.00** = delta_q가 발을 정확히 목표로 이동.
- **Run-D**(① IK + jitter 0.18 + scale 10, from-scratch) 학습 중 — Run-C(scale 10, IK 없음, gap 0)와 **IK만 다른 A/B**. gate@1500 대기.

**분기 (Run-D gate)**: gap 열리면 → **IK가 enabler = 평지 발판추종 해결**, 수렴·정확도(cm) 튜닝. gap 0이면 → 평지+소프트보상 추종은 근본 부족 확정 → **P3의 구조적 발판+지형 물리압력(DTC-identical env)** 으로 조기 이행.
- ★참고: DTC도 tracking을 **지형 커리큘럼**(갭·stepping)에서 학습했다(평지는 배포 결과). 평지 소프트보상 단독 추종은 물리적 압력이 없어 본질적으로 어려울 수 있음.

### P2.5 — 오프라인 TAMOLS 캐시 파이프라인 (실제 TAMOLS 발판) ★2026-07-31 핵심
P2의 "절차적 랜덤 발판"을 **실제 C++ TAMOLS 플랜**으로 대체 = DTC 논문 그대로 "full TAMOLS planner → RL 추종". 단 **온라인 solve가 IsaacLab 스케일서 불가**해 오프라인 캐시로 실현(= APT-RL 방식).

**아키텍처 결정 — in-loop 불가 → 오프라인 캐시:**
```
[in-loop DTC 원전]  4096 env × touchdown마다 TAMOLS solve
   벤치(순수 C++ OpenMP, 20코어, tamols/bench_batch.cpp): warm solve 13ms → 배치 ~400 solves/s
   4096 env @0.4s 재풀이 = 10,240 solves/s 필요 → 25× 부족 (서버 64코어도 alloc경합 포함 ~500 env 상한)
   = DTC가 CPU 클러스터+2주 쓴 이유. 우리 자원선 불가.
        ↓ pivot
[오프라인 캐시 = APT-RL]  TAMOLS를 학습 전 격자로 사전풀이 → 학습중 조회+재앵커(solve 0)
```
- **핵심 통찰**: TAMOLS 플랜은 로컬 프레임(base 원점·전방+x)이라 절대 위치/yaw 무관 → 플랜 모양은 `(vx, gap폭, gap거리)`만으로 결정 → 격자 사전풀이 가능. 학습중 env가 `world = base_pos + Rz(yaw)·local`로 재앵커. 재앵커가 로봇을 따라가 reactive·나머지 균형(capture-point)은 RL이 = **DTC 분업**. (in-loop이 pure-online서 실패한 근본원인 = capture-point 부재였고, RL이 그걸 채우는 게 DTC 존재이유 → 오프라인 캐시도 동일 분업.)

**캐시 (`tamols/cache_gen.cpp` → `cache/{footholds,base,contacts}.bin`+meta.json)**: `vx{0.2~0.6} × width{0.06~0.26} × gapd{-0.2~0.8}` = **1230셀**, 셀당 로컬 발판(4×3)+base궤적(51×12)+접촉(51×4). 50초 생성, 실패 0.
- **도달성 수정**(중요): aggressive straddle-init 끔(`OnlineCfg.straddle_init` 신규 플래그, 기본 true=컨트롤러 불변) + **foot-in-gap 게이트**(nominal 발판[앞~0.51·뒤~0.06]이 실제 gap에 빠질 때만 회피, 아니면 nominal). → 먼 gap의 도달불가 straddle(FL 1.0m 앞) 제거. 검증: straddle이 두 밴드(앞발 gapd~0.4·뒷발~-0.1)에만·전부 도달가능(최대 hip 앞 ~0.29m).
- **width 축**: env gap폭(0.05~0.26 커리큘럼 램프)을 캐시가 커버 → 폭 불일치 해소.

**env 배선 (`RobotSW_IsaacLab/.../quad17/quad17_env.py`, worker 위임)**: `_regen_footholds`의 절차적 Raibert를 캐시 lookup+재앵커로 대체(토글 `use_tamols_cache`, Raibert는 A/B용 보존). leg 매핑(cache FL,FR,RL,RR ↔ env sole 기하 부호매칭 `cache2sole=[2,3,0,1]`, 검증 assert), 3축 lookup(vx·per-gap폭[스트립간격서 도출]·gap거리[searchsorted]), forward-vx 한정(vy=yaw=0, TAMOLS가 vx-only). **obs=101 불변**(발판 출처만 교체). 스모크(64-env) 통과: 캐시 로드·매핑·무오류·foothold_track 산출.

**학습 착수 (GPU2 4096-env, tmux `dtc_gap`)**: 20000 iter, **2.4s/iter → ~13h**(주말 완주).

**★iter 1500 진단 — 두 층으로 갈림(정직):**
- ✅ **추종 층 = 해결**: **foothold_track 0.016→0.27**(iter7→1500, ~15× 상승) + **foot_in_gap ≈ −0.0005**(발이 gap에 거의 안 빠짐=straddle 작동) + Mean reward −7→**+6.5**(양전환). = **P2의 근본 실패(base상대 발판=obs 무시)를 지형유래 발판이 극복** = DTC 오프라인 메커니즘 검증.
- ⚠️ **균형/보행 층 = 현 병목**: **종료=base_contact 지배**(~4초 후 토르소 낙상, time_out 0) + **progress 0.033**(전진 거의 못함) + **epLen ~200/1000**(생존 20%). 발판은 맞추나 **안정된 보행으로 실행 못함** = parkour서 본 무거운 로봇 균형 문제. **spawn 근처(거의 평지)서 이미 낙상**.

**★Go2 파이프라인 대조 (현 병목 관련)**: 현재 quad17 DTC env는 Go2 파쿠르의 두 요소가 없음 — ①**heightmap 지각**(`num_heights=0`, DTC가 의도적으로 foothold 참조로 대체) ②**정식 IsaacLab 적응형 터레인 커리큘럼**(현재=`_build_gap_terrain` 손수 공간 gap 램프, Go2=`TerrainGeneratorCfg(curriculum=True)` level 승강). ②가 병목 기여 후보(평지 걷기 마스터 전에 gap 마주). 단 parkour 포팅(①②다 갖춤)도 tilt했음=원인은 **gait-free가 무거운 로봇엔 too hard**였고, 현재 고정 trot은 그 문제 없음(그래서 추종 성공).

**★iter 5000 균형 판정 = 정체 확정**: foothold_track 0.28→**0.34**(계속 상승, 추종 우수) but **epLen ~200 평탄**(190~218 진동, 상승無)·**progress ~0.03 평탄**·**base_contact ~20 평탄**(낙상 안 줆). = 3200 iter 동안 **발판 추종만 개선·균형/생존은 정체**. → 정책대로 **커리큘럼 전환**.

**★적응형 터레인 level 커리큘럼 전환 (2026-07-31, 자율)**: dtc_gap 중단(체크포인트 저장) → **커리큘럼 학습 발사**(tmux `dtc_curric`, `QUAD17_TERRAIN_CURRICULUM=1`). env 배선(worker): `_build_gap_terrain`을 **10 난이도 lane**으로 확장(level0=평지→level9 gap 0.20m, +y로 lane 배치), per-env `_terrain_level` 진행거리 기반 승강(promote>0.8·demote<0.4·corridor 8m), per-level gap 테이블로 캐시 통합 유지, `Metrics/terrain_level` 로그, 토글 OFF=현재 동작 불변. 스모크 통과(10 lane 빌드·terrain_level 로그·무오류). **가설**: 잘 되는 것(고정 trot+TAMOLS 발판 추종) + 빠진 것(평지부터 걷기 배우는 적응형 커리큘럼) 조합 → 균형 병목 해소.

**★★iter 1500 결과 = 커리큘럼 정공 검증(2026-08-01)**: iter ~1100 breakthrough(RL 전형 slow-start→급상승) — **epLen 150→~500**(4→10초 생존)·**progress 0.008→0.22**(전진!)·**base_contact 종료 28→7**(낙상 4×↓)·foothold_track→0.56·**Mean reward 22.7**·**★terrain_level 0→0.57**(평지 마스터 후 **gap 레벨 승급 시작**). = **"빠진 것=적응형 커리큘럼"이 균형 병목의 원인**이었음이 검증. dtc_gap(정체 epLen200·progress0.03)와 극명 대조. **정책대로 개선이니 지속**(gait 튜닝 불요).

**★★iter 8751 = 강한 성공 궤적**: **terrain_level 2.17→~4.6-5.5**(level 5 ≈ 11cm gap까지 크로싱, 계속 상승)·epLen 567·progress 0.22·falls 7·reward ~26 유지. = 로봇이 **걷기 + 점점 넓은 gap 크로싱을 순차 학습**. DTC 오프라인+커리큘럼 정공 작동. 완주(20000 iter)까지 terrain_level 승급 지속 모니터(→최대 level 9=20cm gap 도달 여부).
- **남은 것(구조)**: base궤적·접촉 캐시는 저장했으나 첫 배선은 발판만(base=env 램프·gait=고정 trot). 3D 지형(계단/슬로프)=full base 플랜 후속. in-loop 재검토=서버 코어 확대/GPU-batched QP 시.

### P3 — 지형·강건성·CVAE (예정)
지형 heightmap → 발판에 물리 압력(갭=헛디디면 낙상), Kim2025 competitive CVAE 커리큘럼, 실제 TAMOLS 참조(`tamols_02leg.py`)로 절차적 발판 대체 ← **P2.5서 실현(오프라인 캐시)**, height scan(발→목표 직선).

---

## 4. 하이브리드 맥락 — 5패턴 중 DTC=패턴 A (나머지 요약)

DTC(패턴 A) 외 4패턴은 상황별 보조:
- **B (RL 상위/MPC 하위·계층)**: RL이 발디딤·gait 의도, MPC/WBC가 물리 실현. RLOC·GLIDE.
- **C (RL을 MPC 내부 주입)**: C-1 학습 가치함수=terminal cost / C-2 잔차 모델(=액추에이터 네트워크·**TOWR 오프라인 참조**) / C-3 RL로 MPC 하이퍼 튜닝.
- **D (MPC=안전계층)**: predictive safety filter·CBF, 또는 MPC 출력 위 RL 잔차(80/20). 실기 보험.
- **E (샘플링 MPC + 학습 prior)**: MPPI를 학습 정책으로 warm-start. sit/getup의 gather seed가 수동 버전.

**역할 분담 원칙**: "RL이 잘 되면 내 궤적·모델·안전계층 덕이고, 안 되면 내 MPC가 백업." MPC 담당 = 모델·레퍼런스 생성기·보상설계 자문·하위 WBC·safety filter. (상세 용어집·문헌은 `dtc-aptrl-papers`·`rl-backbone-8papers` 메모리.)

---

## 5. MPC 교사 공급 · OCP 실시간화 (요약)

**핵심**: RL에게 주는 것은 "실시간 제어기"가 아니라 "궤적·신호". 교사는 오프라인이라 실시간성 불요 → RL 협업 즉시 가능. A(Convex MPC+WBIC, 실시간)는 학습 루프 내 online 참조(②), aligator full-dynamics는 오프라인 정밀 궤적 라이브러리(①).

- **점프 live-solve OCP = 완료**(커밋 57e4e7b·c5cf034→fd8bc95→6bb3142): C++ in-loop solve(캐싱 464→122ms), crocoddyl→aligator 하드 BoxConstraint 이관, wbic_jump(폐루프 GRF) apex 0.546·falls=0.
- **보행 실시간 OCP = 실험 후 미채택**(2026-07-13): B 호라이즌 다이어트 T50=53Hz(marginal)·T40붕괴. 병목=OCP 안정/수렴(02_Leg 무거운 다리 ill-conditioning). **실시간 보행은 A로 충분** → OCP는 저속 정밀 접촉기동 전용.

---

## 6. 로드맵 (DTC 중심)

- **H0.5 sim2sim 브리지** — RL 정책 아티팩트(관측 스펙·정규화·게인) → MuJoCo(C++ 뷰어) 배포. 함정=quaternion(wxyz vs xyzw)·속도 프레임·관절 순서/부호 매핑. RGA `RobotSW_IsaacLab` 클론 확보(R_Skeleton=R.pet·hind_leg 태스크·ActorCriticRMA·obs 46·ONNX+estimator).
- **H1 sit/getup DTC화** — gather-seeded MPPI 궤적 레퍼런스로 RL tracking(축소판 DTC).
- **H2 보행 DTC화** — A/run 프리셋을 레퍼런스 생성기로, RL tracker 배포. WBIC 하위(패턴 B) vs 완전 증류(패턴 A)는 예산·안전으로 결정. ← **현재 P2가 이 방향의 quad17 실증.**
- **H3 안전계층** — RL 배포 시 MPC predictive safety filter 상시.

---

## 참조
- **Jenelten "DTC"** (Science Robotics 2024, arXiv 2309.15462) — 원전. PDF=`quad/backup/docs/2309.15462v2.pdf`.
- **Kim 2025 (Raibo)** (arXiv 2506.02835) — 목표 아키텍처. PDF=`~/다운로드/2506.02835v1.pdf`.
- 코드: `RobotSW_IsaacLab/.../direct/quad17/`(env·cfg·agents) · `rga.py` QUAD_17DOF_CFG · `tamols/tamols_02leg.py`(TAMOLS 참조).
- 관련 문서: `rl_module_train.html`(모듈 학습법) · `D1_OCS2_개발리포트.md`(모델기반 배포 NMPC) · `TOWR_개발리포트.md`(모델기반 상한→RL).
- 메모리: `dtc-17dof-development`(P0~P3 작업기록) · `dtc-aptrl-papers` · `rl-backbone-8papers` · `perceptive-nav-tamols`.
