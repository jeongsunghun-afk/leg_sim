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

# ── 이관 3단계: robot_main (RealHal + EkfEstimator + TrotBridge, 실기 RT 루프) ── ★Pi 전용
#   RobotSharedMem.h(Pi에만)+MuJoCo(ARM)+eiquadprog 존재 시에만 빌드. 데스크톱은 자동 skip.
#   ※Pi 빌드=ENV_PREFIX(또는 MUJOCO/EIQP 경로)를 Pi의 MuJoCo(ARM)로 지정 필요.
if(EXISTS "/usr/include/RobotSharedMem.h" AND MUJOCO_LIB AND EIQP_LIB AND EXISTS ${QUAD_CPP_SRC}/quad_control.hpp)
  find_library(RSHM_LIB RobotSharedMem)
  add_executable(robot_main app/robot_main.cpp)
  target_compile_definitions(robot_main PRIVATE QC_HAVE_ROBOT_SHM)
  target_include_directories(robot_main PRIVATE ${CMAKE_SOURCE_DIR} ${QUAD_CPP_SRC}
                             ${MUJOCO_INCLUDE} ${EIGEN_INCLUDE} ${EIQP_INCLUDE})
  target_link_libraries(robot_main PRIVATE ${MUJOCO_LIB} ${EIQP_LIB} ${RSHM_LIB} rt pthread)  # RobotTestGait와 동일 링크
  message(STATUS "robot_main(Pi): RobotSharedMem=${RSHM_LIB}")
else()
  message(STATUS "robot_main SKIP (RobotSharedMem.h 없음=데스크톱) — Pi에서만 빌드")
endif()

# ── 실기 진단도구(Pi 전용) — MuJoCo 불필요(RealHal은 Eigen+RobotSharedMem 만 의존) ──
#   chan_probe : 읽기전용 채널배치/IMU 실측(모터 무동작).
#   hal_smoke  : RealHal 순수토크 경로 end-to-end 검증(단일 채널·속도가드·토크캡).
if(EXISTS "/usr/include/RobotSharedMem.h")
  find_library(RSHM_LIB RobotSharedMem)
  foreach(_t chan_probe hal_smoke)
    add_executable(${_t} tools/${_t}.cpp)
    target_compile_definitions(${_t} PRIVATE QC_HAVE_ROBOT_SHM)
    target_include_directories(${_t} PRIVATE ${CMAKE_SOURCE_DIR} ${EIGEN_INCLUDE})
    target_link_libraries(${_t} PRIVATE ${RSHM_LIB} rt pthread)
  endforeach()
  message(STATUS "tools(Pi): chan_probe · hal_smoke")
endif()
