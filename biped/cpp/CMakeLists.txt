cmake_minimum_required(VERSION 3.16)
project(biped_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# 의존성: eigen · eiquadprog · mujoco. **conda 전용이 아니다** (2026-08-19 정정).
#   실제 확인된 조합:
#     Pi     eigen=/usr/include/eigen3 · eiquadprog=/opt/ros/jazzy · mujoco=~/mujoco
#     노트북 eigen=/usr/include/eigen3 · eiquadprog=/opt/ros/humble(apt ros-humble-eiquadprog)
#            mujoco=pip 휠(site-packages/mujoco) — **lib 이 include 옆에 있고 `lib/` 하위가 아니다**
#   ⇒ 못 찾으면 -D 로 직접 준다:
#     cmake -S . -B build -DEIQP_INCLUDE=... -DEIQP_LIB=... -DMUJOCO_INCLUDE=... -DMUJOCO_LIB=...
#   ⚠MuJoCo 는 **3.9.0** 이어야 한다(헤더 mjVERSION_HEADER=3009000).
#     3.11 은 d.qM→d.M · mj_fullM 시그니처가 바뀌어 이 코드가 안 빌드된다.
set(ENV_PREFIX "$ENV{CONDA_PREFIX}")
if(NOT ENV_PREFIX)
  set(ENV_PREFIX "/home/jsh/miniforge3/envs/proxddp")
endif()
find_path(EIGEN_INCLUDE Eigen/Dense PATHS ${ENV_PREFIX}/include/eigen3 /usr/include/eigen3 REQUIRED)
find_path(EIQP_INCLUDE  eiquadprog/eiquadprog-fast.hpp PATHS ${ENV_PREFIX}/include REQUIRED)
find_library(EIQP_LIB   eiquadprog PATHS ${ENV_PREFIX}/lib REQUIRED)
message(STATUS "eigen: ${EIGEN_INCLUDE} · eiquadprog: ${EIQP_LIB}")

add_executable(biped_mpc_parity src/biped_mpc_parity.cpp)
target_include_directories(biped_mpc_parity PRIVATE ${EIGEN_INCLUDE} ${EIQP_INCLUDE})
target_link_libraries(biped_mpc_parity PRIVATE ${EIQP_LIB})

add_executable(biped_wbic_parity src/biped_wbic_parity.cpp)
target_include_directories(biped_wbic_parity PRIVATE ${EIGEN_INCLUDE} ${EIQP_INCLUDE})
target_link_libraries(biped_wbic_parity PRIVATE ${EIQP_LIB})

find_path(MUJOCO_INCLUDE mujoco/mujoco.h PATHS ${ENV_PREFIX}/include REQUIRED)
find_library(MUJOCO_LIB  mujoco PATHS ${ENV_PREFIX}/lib REQUIRED)
add_executable(biped_sim src/biped_sim.cpp)
target_include_directories(biped_sim PRIVATE ${EIGEN_INCLUDE} ${EIQP_INCLUDE} ${MUJOCO_INCLUDE})
target_link_libraries(biped_sim PRIVATE ${EIQP_LIB} ${MUJOCO_LIB})

# ── ★실기 배포 (§9) — mj_step 자리에 실모터 read/write ────────────────────────
#   libbipedshm 은 **링크하지 않는다**(dlopen). Pi 전용 라이브러리라 링크 의존을 만들면
#   노트북 빌드가 통째로 깨진다. 실행 시점에만 필요하게 둔다 → -ldl 만 건다.
add_executable(biped_deploy src/biped_deploy.cpp)
target_include_directories(biped_deploy PRIVATE ${EIGEN_INCLUDE} ${EIQP_INCLUDE} ${MUJOCO_INCLUDE})
target_link_libraries(biped_deploy PRIVATE ${EIQP_LIB} ${MUJOCO_LIB} dl)

# ── ★실기 모니터 뷰어 — 엔코더 표시 전용(제어·물리 없음) ────────────────────
# ★JointMap Python↔C++ 패리티 시험. MuJoCo 불필요(deploy_hw.hpp 만).
#   한 점 검증으로는 못 잡는 divergence 를 한계 박스 전 꼭짓점으로 훑는다.
add_executable(biped_parity src/biped_parity.cpp)

# ── ★평발 home 자세(Qflat8) 진단·재산출 — 기하조건 3개를 Newton 으로 푼다 ──────
add_executable(flat_home src/flat_home.cpp)
target_include_directories(flat_home PRIVATE ${MUJOCO_INCLUDE})
target_link_libraries(flat_home PRIVATE ${MUJOCO_LIB})
target_link_directories(flat_home PRIVATE ${ENV_PREFIX}/lib)
set_target_properties(flat_home PROPERTIES BUILD_RPATH ${ENV_PREFIX}/lib)

add_executable(biped_monitor src/biped_monitor.cpp)
target_include_directories(biped_monitor PRIVATE ${EIGEN_INCLUDE} ${MUJOCO_INCLUDE})
target_link_libraries(biped_monitor PRIVATE ${MUJOCO_LIB} glfw dl)
target_link_directories(biped_sim PRIVATE ${ENV_PREFIX}/lib)
set_target_properties(biped_sim PROPERTIES BUILD_RPATH ${ENV_PREFIX}/lib)

# ── 뷰어(biped_view) — GLFW 가 있을 때만 빌드한다 ─────────────────────────────
#   ★조건부인 이유: GLFW 는 렌더링 전용이라 헤드리스 검증(biped_sim·parity)에는 불필요하다.
#     무조건 타겟으로 두면 GLFW 가 없는 환경에서 CMake 가 **generate 단계에서 실패**해
#     biped_sim 조차 빌드되지 않는다(로봇 Pi 에서 실제로 그랬다 — 노트북은 conda 로
#     GLFW 가 딸려와 드러나지 않던 이식성 문제).
#   설치: sudo apt install -y libglfw3-dev   (aarch64/x86_64 공통)
find_path(GLFW_INCLUDE GLFW/glfw3.h PATHS ${ENV_PREFIX}/include /usr/include)
find_library(GLFW_LIB glfw PATHS ${ENV_PREFIX}/lib /usr/lib)   # 기본 검색경로가
                                                               # aarch64/x86_64 를 알아서 찾는다
if(GLFW_INCLUDE AND GLFW_LIB)
  message(STATUS "glfw: ${GLFW_LIB} → biped_view 빌드")
  add_executable(biped_view src/biped_view.cpp)
  target_include_directories(biped_view PRIVATE ${EIGEN_INCLUDE} ${EIQP_INCLUDE} ${MUJOCO_INCLUDE} ${GLFW_INCLUDE})
  target_link_libraries(biped_view PRIVATE ${EIQP_LIB} ${MUJOCO_LIB} ${GLFW_LIB})
  target_link_directories(biped_view PRIVATE ${ENV_PREFIX}/lib)
  set_target_properties(biped_view PROPERTIES BUILD_RPATH ${ENV_PREFIX}/lib)
else()
  message(STATUS "glfw 없음 → biped_view 건너뜀(헤드리스 타겟은 정상 빌드). "
                 "뷰어가 필요하면: sudo apt install -y libglfw3-dev")
endif()
