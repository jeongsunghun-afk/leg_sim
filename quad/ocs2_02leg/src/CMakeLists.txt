cmake_minimum_required(VERSION 3.5)
project(ocs2_legged_robot)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(ocs2_core REQUIRED)
find_package(ocs2_oc REQUIRED)
find_package(ocs2_ddp REQUIRED)
find_package(ocs2_mpc REQUIRED)
find_package(ocs2_sqp REQUIRED)
find_package(ocs2_ipm REQUIRED)
find_package(ocs2_robotic_tools REQUIRED)
find_package(ocs2_pinocchio_interface REQUIRED)
find_package(ocs2_centroidal_model REQUIRED)
find_package(ocs2_robotic_assets REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED COMPONENTS system filesystem log_setup log)

# Resolve for the package path at compile time.
configure_file(
  "${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}/package_path.h.in"
  "${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}/package_path.h"
  @ONLY
)

add_library(${PROJECT_NAME}
  src/common/ModelSettings.cpp
  src/dynamics/LeggedRobotDynamicsAD.cpp
  src/constraint/EndEffectorLinearConstraint.cpp
  src/constraint/FrictionConeConstraint.cpp
  src/constraint/ZeroForceConstraint.cpp
  src/constraint/NormalVelocityConstraintCppAd.cpp
  src/constraint/ZeroVelocityConstraintCppAd.cpp
  src/initialization/LeggedRobotInitializer.cpp
  src/reference_manager/SwitchedModelReferenceManager.cpp
  src/foot_planner/CubicSpline.cpp
  src/foot_planner/SplineCpg.cpp
  src/foot_planner/SwingTrajectoryPlanner.cpp
  src/gait/Gait.cpp
  src/gait/GaitSchedule.cpp
  src/gait/LegLogic.cpp
  src/gait/ModeSequenceTemplate.cpp
  src/LeggedRobotInterface.cpp
  src/LeggedRobotPreComputation.cpp
)
target_link_libraries(${PROJECT_NAME}
  pinocchio::pinocchio
  Boost::filesystem
  Boost::system
  dl
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
    ${pinocchio_INCLUDE_DIRS}
)
ament_target_dependencies(${PROJECT_NAME}
  ocs2_core
  ocs2_oc
  ocs2_ddp
  ocs2_mpc
  ocs2_sqp
  ocs2_ipm
  ocs2_robotic_tools
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  ocs2_robotic_assets
)
target_compile_options(${PROJECT_NAME} PUBLIC ${OCS2_PINOCCHIO_FLAGS})

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
)
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  EXPORT export_${PROJECT_NAME}
  DESTINATION share/${PROJECT_NAME}/cmake
)

ament_export_include_directories(
  include
  ${PROJECT_BINARY_DIR}/include
  ${pinocchio_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(
  ocs2_core
  ocs2_oc
  ocs2_ddp
  ocs2_mpc
  ocs2_sqp
  ocs2_ipm
  ocs2_robotic_tools
  ocs2_pinocchio_interface
  ocs2_centroidal_model
  ocs2_robotic_assets
  pinocchio
  Eigen3
  Boost
)

####################
# 02_Leg port test #
####################
add_executable(test02legLoad test/test02legLoad.cpp)
target_include_directories(test02legLoad PRIVATE include ${PROJECT_BINARY_DIR}/include)
target_link_libraries(test02legLoad ${PROJECT_NAME})
target_compile_options(test02legLoad PRIVATE ${OCS2_PINOCCHIO_FLAGS})
install(TARGETS test02legLoad RUNTIME DESTINATION lib/${PROJECT_NAME})

# MuJoCo 폐루프 브리지 (Phase 2) — conda proxddp env의 MuJoCo 링크
set(CONDA_ENV "/home/jsh/miniforge3/envs/proxddp")
find_path(MJ_INC mujoco/mujoco.h PATHS ${CONDA_ENV}/include)
find_library(MJ_LIB mujoco PATHS ${CONDA_ENV}/lib)
if(MJ_INC AND MJ_LIB)
  message(STATUS "MuJoCo: ${MJ_LIB} → test02legMujoco 빌드")
  find_path(EIQP_INC eiquadprog/eiquadprog-fast.hpp PATHS ${CONDA_ENV}/include)
  find_library(EIQP_LIB eiquadprog PATHS ${CONDA_ENV}/lib)
  # qpOASES(legged_wbc 솔버) — 소스 빌드(coin-or). 있으면 WBC_LEGGED가 qpOASES 사용(semi-def H robust).
  set(QPOASES_DIR "/home/jsh/qpOASES")
  find_library(QPOASES_LIB qpOASES PATHS ${QPOASES_DIR}/build/libs)
  add_executable(test02legMujoco test/test02legMujoco.cpp)
  target_include_directories(test02legMujoco PRIVATE include ${PROJECT_BINARY_DIR}/include)
  # ★conda/include(mujoco·eiquadprog)는 -idirafter로 검색 맨 뒤에 배치 → OCS2가 -isystem으로
  #   먼저 넣는 시스템 pinocchio(/opt/ros/humble)가 이겨 conda pinocchio shadow(ABI 불일치·FK깨짐) 방지.
  target_compile_options(test02legMujoco PRIVATE -idirafter ${CONDA_ENV}/include ${OCS2_PINOCCHIO_FLAGS})
  # conda libstdc++(GLIBCXX_3.4.32 ⊇ 시스템)로 MuJoCo+OCS2 ABI 동시충족
  find_library(GLFW_LIB glfw PATHS ${CONDA_ENV}/lib)
  target_link_libraries(test02legMujoco ${PROJECT_NAME} ${CONDA_ENV}/lib/libstdc++.so ${MJ_LIB} ${EIQP_LIB} ${GLFW_LIB})
  set(TEST02_RPATH "${CONDA_ENV}/lib")
  if(QPOASES_LIB)
    message(STATUS "qpOASES: ${QPOASES_LIB} → WBC_LEGGED가 qpOASES 사용")
    target_include_directories(test02legMujoco PRIVATE ${QPOASES_DIR}/include)
    target_compile_definitions(test02legMujoco PRIVATE WBC_USE_QPOASES)
    target_link_libraries(test02legMujoco ${QPOASES_LIB})
    set(TEST02_RPATH "${TEST02_RPATH};${QPOASES_DIR}/build/libs")
  endif()
  set_target_properties(test02legMujoco PROPERTIES BUILD_RPATH "${TEST02_RPATH}" INSTALL_RPATH "${TEST02_RPATH}")
  install(TARGETS test02legMujoco RUNTIME DESTINATION lib/${PROJECT_NAME})
else()
  message(WARNING "MuJoCo 미발견 → test02legMujoco 건너뜀")
endif()

############
# Testing ##
############

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(${PROJECT_NAME}_test
    test/AnymalFactoryFunctions.cpp
    test/constraint/testEndEffectorLinearConstraint.cpp
    test/constraint/testFrictionConeConstraint.cpp
    test/constraint/testZeroForceConstraint.cpp
  )
  target_include_directories(${PROJECT_NAME}_test PRIVATE
    test/include
    ${PROJECT_BINARY_DIR}/include
  )
  target_link_libraries(${PROJECT_NAME}_test
    gtest_main
    ${PROJECT_NAME}
  )
  target_compile_options(${PROJECT_NAME}_test PRIVATE ${OCS2_PINOCCHIO_FLAGS})
endif()

ament_package()
