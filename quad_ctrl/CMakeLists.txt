cmake_minimum_required(VERSION 3.16)
project(quad_ctrl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen (header-only). 필요시: cmake -DEIGEN_INCLUDE=/path/to/eigen3
find_path(EIGEN_INCLUDE Eigen/Dense
  HINTS ${EIGEN_INCLUDE} $ENV{CONDA_PREFIX}/include/eigen3 /usr/include/eigen3 /usr/local/include/eigen3)
if(NOT EIGEN_INCLUDE)
  message(FATAL_ERROR "Eigen 못 찾음. -DEIGEN_INCLUDE=... 로 지정")
endif()

include_directories(${CMAKE_SOURCE_DIR} ${EIGEN_INCLUDE})

# 데모(MockHal, MuJoCo 불필요) — 골격 컴파일·배선 검증
add_executable(sim_main app/sim_main.cpp)

# ── 이관 1단계: sim_bridge (MujocoHal + quad/cpp TrotCtrl wrap) ──
#   quad/cpp 검증 컨트롤러를 HAL 경계 뒤에서 구동. sim_bridge == trot_sim(GT) 검증용.
set(ENV_PREFIX "/home/jsh/miniforge3/envs/proxddp" CACHE PATH "conda env prefix (mujoco/eiquadprog)")
set(QUAD_CPP_SRC "${CMAKE_SOURCE_DIR}/../quad/cpp/src")
find_path(MUJOCO_INCLUDE mujoco/mujoco.h              PATHS ${ENV_PREFIX}/include)
find_library(MUJOCO_LIB  mujoco                        PATHS ${ENV_PREFIX}/lib)
find_path(EIQP_INCLUDE   eiquadprog/eiquadprog-fast.hpp PATHS ${ENV_PREFIX}/include)
find_library(EIQP_LIB    eiquadprog                    PATHS ${ENV_PREFIX}/lib)
if(MUJOCO_LIB AND EIQP_LIB AND EXISTS ${QUAD_CPP_SRC}/quad_control.hpp)
  add_executable(sim_bridge app/sim_bridge.cpp)
  target_include_directories(sim_bridge PRIVATE ${CMAKE_SOURCE_DIR} ${QUAD_CPP_SRC}
                             ${MUJOCO_INCLUDE} ${EIGEN_INCLUDE} ${EIQP_INCLUDE})
  target_link_libraries(sim_bridge PRIVATE ${ENV_PREFIX}/lib/libstdc++.so ${MUJOCO_LIB} ${EIQP_LIB})
  target_link_options(sim_bridge PRIVATE "-Wl,-rpath-link,${ENV_PREFIX}/lib")
  message(STATUS "sim_bridge: MuJoCo=${MUJOCO_LIB} eiquadprog=${EIQP_LIB}")
else()
  message(STATUS "sim_bridge SKIP (MuJoCo/eiquadprog/quad_cpp 없음) — sim_main 골격만 빌드")
endif()

# TODO(3단계): app/robot_main.cpp(real RT 루프) + real_hal(모터/IMU) + EKF estimator.
