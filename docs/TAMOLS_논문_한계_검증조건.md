# TAMOLS 논문 — 한계점 · 검증 지형조건 (★모델기반 개발 참조용)

> 원전: F. Jenelten, R. Grandia, F. Farshidian, M. Hutter, **"TAMOLS: Terrain-Aware Motion Optimization for Legged Systems"**, IEEE RA-L 2022, arXiv:2206.14049v2. 로봇=ANYmal(12 DOF). 우리 세션(2026-07-31) 논문 정독 발췌.
> **용도: 우리 모델기반 full-TAMOLS 개발(full-dyn OCP + TAMOLS)의 참조** — ①목표 지형범위 설정(어디까지 노리나) ②기대 한계 파악(모델기반이 근본적으로 못 넘는 선). 우리가 실측한 "full-dyn OCP도 slope stall·gap fall"이 이 논문 한계와 어떻게 맞물리는지 대조.

---

## 1. 명시적 한계 (§VII-G Limitations)

1. **★동역학 모델 근사 (major drawback)** — TAMOLS의 GIAC 안정성 보장(weak contact stable)은 **수평 접촉면**에서만 성립. **발 하나라도 기울어진 평면**에 있으면 WBC가 no-slip 조건 위해 계획 궤적에서 이탈. 일반 지형(비수평)으로 확장은 이론상 prop 1·2를 hard 제약화하면 가능하나 **그런 발판 찾기가 매우 어렵다**고 명시.
2. **full kinematics 미반영** — 간이 운동학 제약(task-space)이 다리 과신장은 막으나 **무릎 관절 충돌(knee joint collision)은 여전히 문제**. → DTC 논문(2309.15462) 기준 **TAMOLS 결합 최대 극복 높이 ≈ 0.40 m**(이 한계 때문).
3. **elevation map 품질 강의존** — 상태추정 드리프트가 map을 odometry 대비 이동시킴 → 실제 발위치와 계획 발위치 **불일치** → **극단적 경우 시스템 불안정화**.

(추가: GM observer 실측서도 정지 시 힘 추정치가 0으로 수렴 안 함 = 부정확한 질량/CoM 모델오차. GM=실기 외란/모델오차용, sim은 무의미.)

---

## 2. 검증 지형조건 (§VII Results)

- **로봇**: ANYmal (ruggedized quadruped, 12 actuated DOF), 실기+sim. LiDAR 2개, elevation map 20Hz(GPU), 상태추정 400Hz(CPU).
- **예측 호라이즌**: **1 gait cycle**(다리당 1스텝). 계산: trot **6.3 ms**(SOTA CMO 48배)·fast trot 2.1 ms·SQP 1~2 iter 최다 수렴.
- **검증 gait**: trot · fast trot · running trot · amble · pace · running pace · crawl (7종).

| 지형 | 조건 | 결과(성공률/거동) |
|---|---|---|
| **계단(sim)** | 12 tread, 18회 승/강, trot. 다리 12.7% 연장 변형·tread 1.27× | TAMOLS **18/18**(trot/amble/running trot/pace) · crawl 16/18↑·14/18↓(대보폭=downstairs 접촉불일치·upstairs 무릎충돌 취약) · (구 batch search trot 10/18↑·14/18↓) |
| **계단(실기)** | **20 tread, 29×17 cm, 36° 경사**, trot, 명령 0.45(실현 0.37 m/s) | 스텝당 0/1/2 tread 클리어. tracking 오차=계단 기하로 feasible space 축소 |
| **속도변조(sim)** | 12스텝 승, fast trot | >0.9 m/s=2 tread/step · <0.9=2·1 교대 · **~0.45~0.55 m/s=1 tread/step(최적)** |
| **갭(실기)** | pallet+slope 갭, ambling, 장애물 30 cm 간격. **갭 높이 20 cm·폭 27 cm**, 명령 0.7 m/s | 급경사부 회피·**갭 안 밟음**(5회 중 RH발 1회만 갭 진입) |
| **stepping stone(실기)** | 경사 나무벽돌, trot, 명령 0.4 m/s. **벽돌 20×20×50 cm, 인접 갭 20 cm** | 발판을 돌 중앙 배치(h_s1 gradient 페널티) |

**요지(§VIII 결론)**: GIAC=미분가능·접촉력 free 동적안정 척도(ZMP만큼 복잡하나 SRBD의 큰 유효범위) + graduated optimization으로 발판·base pose를 rough 지형서 공동최적화. 일반화=rough·human-engineered 환경(계단·갭·stepping stone).

---

## 3. 모델기반 개발 함의 (우리가 쓸 것)

**(a) 우리 모델기반이 노릴 지형범위 (검증조건이 상한 제시)** — TAMOLS(모델기반)가 실기서 넘은 범위 = 우리 목표 상한:
- 계단 **~29×17cm·36° 경사**(1 tread/step 최적 0.45~0.55 m/s, 2 tread는 >0.9 m/s)
- 갭 **폭~27cm·높이~20cm** · stepping stone **20cm 벽돌·갭 20cm**
- 예측호라이즌 = **1 gait cycle**, gait 7종(trot~crawl). 이보다 큰 이산/비수평은 모델기반 단독 곤란.

**(b) 우리 실측 findings ↔ 논문 한계 대조 (2026-07-31)**:
- full-dynamics OCP = **평지 base 안정화(underactuated 닫기) 해결** ✅.
- 지형서 **slope stall·gap fall** = 논문 **한계①(수평 접촉면 가정, 기울어진 발판서 WBC 이탈)의 실증**. 즉 우리 실패가 TAMOLS 근본 한계와 정확히 동종 → **모델기반 추종의 공통 하드월**임을 논문이 사전 예고.
- ⇒ 모델기반 개발 시: **비수평 접촉(경사·계단 tread)·전진력이 근본 취약점**. prop 1·2를 hard 제약화(비수평 발판 허용)하거나 full-kinematics(무릎충돌)까지 넣어야 논문 한계①②를 넘음 = 대공사.

**(c) 모델기반의 근본 한계선 (넘기 어려운 것)**:
- ①비수평 접촉면(기울어진 발판) — WBC no-slip 이탈. ②무릎충돌 ~0.4m(간이 운동학). ③map 품질 의존(상태추정 드리프트→불안정).
- 이 선 너머(임의 이산험지·큰 단차)는 모델기반 한계 → RL(DTC) 영역. 참조=[[full-tamols-modelbased-tracker]]·`DTC_개발리포트.md`.
