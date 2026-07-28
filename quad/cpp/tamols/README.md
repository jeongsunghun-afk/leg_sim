# C++ 실시간 TAMOLS (모델기반 지형 joint MPC) — 개발

**목표** — D1(OCS2 Perceptive NMPC) 청사진을 **우리 MuJoCo/C++ 스택에 native로** 구현.
base 스플라인 + 발판 + GIAC 안정성 + 지형(footScore/edge)을 **공동최적화**하는 실시간 지형 MPC.
= B 제어기의 실시간화(발판만 greedy → base+발판 joint 온라인 협조). APT-RL 전 모델기반 정공법 1판.

**왜 D1 포팅이 아니라 native 구현** — D1(OCS2/Gazebo)은 MuJoCo 작동 보장 없음(sim2sim+ROS+로봇적응 삼중 리스크).
우리가 원하는 건 코드가 아니라 **알고리즘**(OCS2식 joint 지형 MPC + 빠른 QP). Drake 프로토타입으로 정식화는 검증됨 → C++로 포팅.

## 접근

- **정식화**: Drake `quad/tamols/tamols_02leg.py`(우리가 갭회피·GIAC로 검증) → C++ 포팅.
- **솔버**: SQP-RTI(1 iteration warm-start) + 빠른 QP(**eiquadprog**, 이미 quad/cpp WBIC에 있음). 실시간(20ms) 목표.
  - Drake 오프라인 11.7s(583× 느림)의 원인 = 범용 NLP·전체수렴. 극복 = 짧은 horizon + RTI + 커스텀 QP.
- **지형**: `quad/cpp/src/terrain_map.hpp`(footScore/edgeSDF/slope) 재사용.
- **실행**: `quad/cpp` WBIC(eiquadprog)로 계획 추종 + MuJoCo 폐루프(`trot_sim`·`DISABLE_FLOOR`·`quad_tamols_gap.mjcf`).

## 변수/제약/비용 (Drake 대응)

| 항목 | Drake | C++ |
|---|---|---|
| base 스플라인 | `spline_coeffs[phase]` (base_dims×order) | `TamolsState.a[phase]` MatrixXd(6×4) |
| 발판 | `p` (num_legs×3) | `p` Matrix4x3 |
| GIAC slack | `epsilon` (num_phases) | `epsilon` VectorXd |
| 스플라인 평가 | `helpers.py` pos/vel/acc | `tamols.hpp` pos/vel/acc ✅검증(1e-10) |
| 초기·연속 | `add_initial_constraints` | `constraints.hpp` ✅검증 |
| GIAC 동역학 | `add_dynamics/giac_constraints` (Eq17) | `constraints.hpp` ✅검증(4.6e-6) |
| kinematic reach | `add_kinematic_constraints` (l_min/max) | `constraints.hpp` ✅검증 |
| friction cone | `add_friction_cone_constraints` | `constraints.hpp` ✅검증 |
| 비용 | tracking/foothold/nominal (활성) | `costs.hpp` ✅검증(rel 1e-9) |

## 진행 상태

- ✅ **스플라인 평가 핵심** (`tamols.hpp`) — pos/vel/acc, 초기조건 매핑. `test_spline.cpp` 검증(FD 1e-10·초기조건 정확).
- ✅ **지형 처리** (`terrain_proc.hpp`) — h_s1(gaussian)·h_s2(virtual floor)·∇h/∇h_s1/∇h_s2(5점 FD). `test_terrain.cpp` **Python(Drake) 정합 검증**(gaussian 5e-9·∇ 1e-7). ★부호주의: scipy convolve1d=convolution이라 미분커널 뒤집음([-1,8,0,-8,1]).
- ✅ **제약 5종 완성** (`constraints.hpp`) — 초기·위상연속·friction·kinematic reach·**GIAC(Eq17: 17a 마찰콘·17b 다중접촉·17c/d 이중지지)**. Drake feasible 해 로드해 residual 전부 검증(초기 0·위상연속 8e-9·friction 0·kinematic 0·**GIAC 4.6e-6**). L̇=I·dω+ω×Iω(각운동량), det=a·(b×c). `test_constraints.cpp`.
- ✅ **비용 3종 완성** (`costs.hpp`) — tracking(x속도)·foothold_on_ground(양선형 높이보간)·nominal_kinematic. Drake `EvalBinding` 값과 대조 rel~1e-9(track 0.1131·foot 8e-6·nom 3e-5). base_pose_align·edge·prev·smoothness는 Drake서 비활성(주석). `test_costs.cpp`.
- ✅ 갭회피 발판 구속 = 선형 bound(앞발 ≥gap_hi+margin·뒷발 ≤gap_lo−margin), Drake `add_gap_avoid_footholds`. 솔버 조립 시 부등식 bound로 편입.
- ✅ **SQP-RTI 솔버**(`tamols_qp.hpp`, eiquadprog) — 결정벡터 pack/unpack(nz=137) + 비용 GN 최소제곱(H=2JᵀJ·g=2JᵀR) + 제약 선형화(FD Jacobian) + ℓ1 merit 라인서치 + **적응형 Levenberg-Marquardt(trust-region reg)**. Drake 해=고정점(eq 1e-14·ineq 1.6e-7), 섭동서 feasible 복귀. `test_qp.cpp`.
  - **정확성·수렴 ✓** / **실시간 아직**: FD Jacobian 5.2 ms/iter(반복당 ~822 함수평가), 5-iter RTI 26 ms(>20ms). → **해석 Jacobian 교체가 실시간 경로**(반복당 10–50×↑ 기대). FD는 검증용.
- ⬜ 해석 Jacobian(비용·제약) → 실시간 20ms
- ⬜ WBIC 폐루프 통합 + 갭 MuJoCo 검증(vs Drake 오프라인)
- ✅ **플랜→WBIC 참조 변환기**(`tamols_track.hpp`) — TAMOLS 해(base 스플라인·발판·게이트)를 매 스텝 (com_ref·yaw·contacts·swing 궤적)으로. `test_track.cpp` 검증: base 연속·트롯 스케줄·스윙끝↔발판 6e-4·x 0→0.73(갭전진).
- ⬜ **MuJoCo 폐루프 러너**(`tamols_sim.cpp`) — 변환기로 ctrl.wbic_track 구동 + quad_tamols_gap.mjcf 갭 크로싱 검증(per-gait 검증 역할)

**검증 방식** — 각 조각을 Drake Python 출력과 대조(같은 입력 → 오차 <1e-5). `test_*.cpp` 회귀.

## 빌드/실행

```bash
cd /home/jsh/문서/jsh/simulation/quad/cpp
g++ -O2 -std=c++17 tamols/test_spline.cpp -I/usr/include/eigen3 -o tamols/test_spline && ./tamols/test_spline
```

관련: 모델기반 갭크로싱 리포트(`docs/모델기반_갭크로싱_탐색리포트.html`) · Drake `quad/tamols/` · TAMOLS 논문 `~/다운로드/논문/TAMOLS_2206.14049.pdf`

## 해석 Jacobian (실시간화) — 진행
- ✅ **비용 잔차 해석 Jacobian**(`tamols_jac.hpp`) — tracking(선형)·foothold(양선형 ∂h)·nominal(∂R_B). FD 대조 오차 2.5e-10·**9× 빠름**(26.5µs vs 237µs). `test_jac.cpp`.
- ✅ **등식 제약 해석 Jacobian**(초기·위상연속=선형 상수). FD 대조 1.4e-10. `test_jac.cpp`.
- ✅ **부등식 해석 Jacobian(비GIAC 299행)** — friction·kinematic(2차)·bounds·foot_y·gap·terminal. FD 대조 7e-10. GIAC 179행=mask(FD 대상).
- ✅ **GIAC 블록-sparse FD** — phase-k 행은 a[k]+p+eps(k)만 의존 → 관련변수만 FD. 전체 부등식 Jacobian(해석+GIAC) 1083µs(vs full FD 4315µs=**4×**), 정합 7e-10.
- ✅ **★실시간 달성** (`solve_fast`, tamols_jac.hpp) — 해석 Jacobian 솔버 **2.26 ms/iter**(FD 6.53 대비 2.9×), **5-iter RTI = 11.7 ms < 20ms(50Hz)**. FD와 동일 수렴. `test_fast.cpp`.
- **C++ TAMOLS = 실시간 온라인 TO(DTC) 가능**. 남은: warm-start receding-horizon 완전크로싱·WBIC 온라인 통합.
