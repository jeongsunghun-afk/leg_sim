#!/bin/bash
# build_bridge.sh — shm_bridge 를 libbipedshm.so 로 컴파일. ★Pi(Emb)에서 실행.
#   RobotSharedMem.h(/usr/include) + RobotTestGait/inc(defineConfigMotor.h 등) + libRobotSharedMem 필요.
#   산출물 경로는 config/biped_emb.yaml 의 shm.lib 와 일치시킬 것.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
RTG="${RTG_ROOT:-/home/jsh/문서/jsh/RobotTestGait}"     # RobotTestGait 루트(inc/ 헤더용)
OUT="${OUT:-$RTG/build/libbipedshm.so}"

echo "브리지 컴파일: $HERE/shm_bridge.cpp → $OUT"
g++ -O2 -fPIC -shared -std=c++17 \
    -I"$RTG/inc" -I/usr/include \
    "$HERE/shm_bridge.cpp" \
    -o "$OUT" \
    -lRobotSharedMem -lrt -lpthread \
  && echo "✅ 빌드 OK: $OUT" \
  || { echo "❌ 빌드 실패 — RobotSharedMem.h/lib 및 RobotTestGait/inc 경로 확인"; exit 1; }
