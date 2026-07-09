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

# TODO(2단계): MuJoCo 링크 + MujocoHal 소스 → 실 sim.
#   find_package(mujoco) 또는 $ENV{CONDA_PREFIX}/lib 링크. eiquadprog(WBIC), 재기어/GEARBOX 이관.
# TODO(3단계): app/robot_main.cpp(real RT 루프) + real_hal(모터/IMU) + EKF estimator.
