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

# Phase 2: MuJoCo 폐루프 (MJCF 추가). ff+PD=정적STANCE falls=0 / WBC=1은 base-task 미완
MJCF=mjcf/quad_real_17dof_waist_sphere.mjcf
VX=0 $EXE/test02legMujoco $CFG $MJCF stance 3.0            # ff+PD 정적 (falls=0)
WBC=1 VX=0 $EXE/test02legMujoco $CFG $MJCF stance 3.0      # WBC (진행중)
```
env: `VX`(전진목표 m/s)·`WBC=1`(WBC 사용)·`PD_ONLY=1`(nominal 홀드)·`MPC_HZ`·`KP/KD`·`W_BASE/W_F/W_REG/W_SW`(WBC 가중)·`DBG/WBC_DBG`(진단).

상세 진행/결론 = `docs/D1_OCS2_porting.md`.
