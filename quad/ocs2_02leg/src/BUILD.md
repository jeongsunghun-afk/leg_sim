# 02_Leg OCS2 포팅 — 소스·빌드 (D1 트랙)

OCS2 워크스페이스(`quad/ocs2_ws/`)는 크기 때문에 **gitignore**됨. 우리가 작성/수정한 소스를 여기에 보존한다. 재현 시 아래대로 ocs2_ws에 배치·수정 후 빌드.

## 우리 작성 소스 (이 디렉토리)
- `test02legLoad.cpp` — Phase 1 검증(인터페이스 로드 + STANCE/TROT MPC 평지 계획 생성).
- `test02legMujoco.cpp` — Phase 2 MuJoCo 폐루프 브리지(상태변환·MPC 재계획·ff토크/WBC·관절PD).
- `wbc_02leg.hpp` — Phase 2b weighted QP WBC(legged_control식). **base-task 정식화 미완**(진행중).

배치: 위 3개를 `ocs2_ws/src/ocs2/ocs2_robotic_examples/ocs2_legged_robot/test/`에 복사.

## OCS2 소스 수정 (2곳)

### 1) `ocs2_legged_robot/src/common/ModelSettings.cpp` — jointNames/contactNames를 task.info서 로드
`loadModelSettings()`의 `modelFolderCppAd` 로드 직후에 추가:
```cpp
// 02_Leg port: task.info의 리스트로 관절/접촉명 오버라이드(없으면 ANYmal 기본 유지)
loadData::loadStdVector(filename, fieldName + ".jointNames", modelSettings.jointNames, verbose);
loadData::loadStdVector(filename, fieldName + ".contactNames3DoF", modelSettings.contactNames3DoF, verbose);
loadData::loadStdVector(filename, fieldName + ".contactNames6DoF", modelSettings.contactNames6DoF, verbose);
```

### 2) `ocs2_legged_robot/CMakeLists.txt` — 실행파일 2개 추가 (`ament_package()` 앞)
```cmake
add_executable(test02legLoad test/test02legLoad.cpp)
target_include_directories(test02legLoad PRIVATE include ${PROJECT_BINARY_DIR}/include)
target_link_libraries(test02legLoad ${PROJECT_NAME})
target_compile_options(test02legLoad PRIVATE ${OCS2_PINOCCHIO_FLAGS})
install(TARGETS test02legLoad RUNTIME DESTINATION lib/${PROJECT_NAME})

set(CONDA_ENV "/home/jsh/miniforge3/envs/proxddp")
find_path(MJ_INC mujoco/mujoco.h PATHS ${CONDA_ENV}/include)
find_library(MJ_LIB mujoco PATHS ${CONDA_ENV}/lib)
find_path(EIQP_INC eiquadprog/eiquadprog-fast.hpp PATHS ${CONDA_ENV}/include)
find_library(EIQP_LIB eiquadprog PATHS ${CONDA_ENV}/lib)
if(MJ_INC AND MJ_LIB)
  add_executable(test02legMujoco test/test02legMujoco.cpp)
  target_include_directories(test02legMujoco PRIVATE include ${PROJECT_BINARY_DIR}/include)
  # ★conda/include(mujoco·eiquadprog)를 -idirafter로 검색 맨 뒤에 → OCS2가 -isystem으로 넣는
  #   시스템 pinocchio(/opt/ros/humble)가 이겨 conda pinocchio shadow(ABI불일치·FK깨짐) 방지. 필수.
  target_compile_options(test02legMujoco PRIVATE -idirafter ${CONDA_ENV}/include ${OCS2_PINOCCHIO_FLAGS})
  target_link_libraries(test02legMujoco ${PROJECT_NAME} ${CONDA_ENV}/lib/libstdc++.so ${MJ_LIB} ${EIQP_LIB})
  set_target_properties(test02legMujoco PROPERTIES BUILD_RPATH "${CONDA_ENV}/lib" INSTALL_RPATH "${CONDA_ENV}/lib")
  install(TARGETS test02legMujoco RUNTIME DESTINATION lib/${PROJECT_NAME})
endif()
```

## 빌드
```bash
cd quad/ocs2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select ocs2_legged_robot \
  --cmake-args -DCMAKE_CXX_FLAGS="-include cstddef -include cstdint"
```
함정: `size_t 미선언`→위 cmake flag / urdfdom→`apt liburdfdom-dev` / conda pinocchio 충돌→`env -u PYTHONPATH`.

## 실행 (작업디렉토리=quad/)
```bash
source /opt/ros/humble/setup.bash && source ocs2_ws/install/setup.bash
EXE=ocs2_ws/install/ocs2_legged_robot/lib/ocs2_legged_robot
CFG="ocs2_02leg/config/task.info ocs2_02leg/urdf/02leg_ocs2.urdf ocs2_02leg/config/reference.info"

# Phase 1: MPC 계획 생성(STANCE 힘균형 + TROT 대각쌍)
$EXE/test02legLoad $CFG ocs2_02leg/config/gait.info

# Phase 2: MuJoCo 폐루프 (MJCF 추가). WBC=1 정적 STANCE=solid(falls=0) / 동적=미달
MJCF=mjcf/quad_real_17dof_waist_sphere.mjcf
WBC=1 VX=0 $EXE/test02legMujoco $CFG $MJCF stance 3.0      # WBC 정적 (falls=0, solid)
WBC=1 VX=0.2 $EXE/test02legMujoco $CFG $MJCF trot 3.0      # WBC 동적 (미달=접촉전환 warm-start)
```

## 뷰어 (GLFW)
```bash
quad/ocs2_02leg/run_view.sh stance 0        # 정적 STANCE solid 확인(마우스=카메라)
quad/ocs2_02leg/run_view.sh trot 0.2        # 동적(진행중)
```

## env 노브
- 실행: `WBC=1`(WBC 저수준)·`VIEW=1`(뷰어)·`VX`(전진 m/s)·`PD_ONLY=1`(ff+PD nominal 홀드)·`SETTLE`(gait 전 STANCE settle s)·`MPC_HZ`(재계획률).
- WBC 가중: `W_BASE`·`W_F`·`W_SW`·`W_REG`·`W_POST`.
- WBC 게인: `KP_B/KD_B`(base pos)·`KP_O/KD_O`(base ori)·`KP_F/KD_F`(Cartesian swing)·`KP_JS/KD_JS`(joint swing).
- WBC 모드: `SWING_JOINT=1`(joint-space swing)·`BASE_HARD=1`(base 6D hard 제약).
- 진단: `DBG`·`WBC_DBG`·`TROT_DBG`(접촉플래그·|w|·계획vs실제). task.info `sqpIteration`=SQP 반복수.

## 현 상태 (2026-07-29)
- ✅ **정적 STANCE solid**(falls=0, tilt 0.8°). WBC 동역학 tau≈ff, swing 추종 정확, MPC standalone 계획 정확.
- 🔶 **동적 locomotion 미달** — 최심층 병목=**접촉스케줄 전환 warm-start**(STANCE→swing 순간 MPC 해 garbage). = MPC solver 엔지니어링 필요(파라미터 튜닝 아님).

상세 진행/결론 = `docs/D1_OCS2_porting.md`.
