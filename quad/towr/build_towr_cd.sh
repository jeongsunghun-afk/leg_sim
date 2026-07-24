#!/usr/bin/env bash
# TOWR C++ (towr_cd.cpp) 빌드 — proxddp env의 casadi+ipopt 링크.
#   실행: bash build_towr_cd.sh  → towr_cd_bin 생성
#   사용: LD_LIBRARY_PATH=$ENV/lib TERRAIN=flat GAIT=trot OUT=traj_flat_cpp.json ./towr_cd_bin
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV=${ENV:-/home/jsh/miniforge3/envs/proxddp}
g++ -O2 -std=c++17 "$HERE/towr_cd.cpp" \
  -I"$ENV/include" -L"$ENV/lib" -lcasadi -Wl,-rpath,"$ENV/lib" \
  -o "$HERE/towr_cd_bin"
echo "✅ 빌드: $HERE/towr_cd_bin  (실행 시 LD_LIBRARY_PATH=$ENV/lib 필요 — ipopt 플러그인 로드)"
