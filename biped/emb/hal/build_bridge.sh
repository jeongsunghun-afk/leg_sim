#!/bin/bash
# build_bridge.sh — shm_bridge 를 libbipedshm.so 로 컴파일. ★Pi(Emb)에서 실행.
#   RobotSharedMem.h(/usr/include) + RobotTestGait/inc(defineConfigMotor.h 등) + libRobotSharedMem 필요.
#   산출물 경로는 config/biped_emb.yaml 의 shm.lib 와 일치시킬 것.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# ★2026-08-05 경로 수정. 기본값이 `/home/jsh/문서/jsh/RobotTestGait` 로 **다른 장비(랩탑)
#   경로**였다. 이 Pi 에서는 그런 디렉터리가 없어 헤더를 못 찾고 빌드가 실패했다.
#   그 결과 libbipedshm.so 가 소스보다 오래된 채로 방치돼 있었다(소스만 고치고 반영 안 됨).
RTG="${RTG_ROOT:-$HOME/ZSource/RobotTestGait}"        # RobotTestGait 루트(inc/ 헤더용)
# ★산출물은 config/biped_emb.yaml 의 shm.lib 와 반드시 일치해야 한다.
#   기존 기본값($RTG/build/…)은 yaml 이 가리키는 emb/hal/ 과 달라서, 빌드가 성공해도
#   앱은 **옛 .so 를 계속 로드**하게 된다. yaml 기준으로 맞춘다.
OUT="${OUT:-$HERE/libbipedshm.so}"

echo "브리지 컴파일: $HERE/shm_bridge.cpp → $OUT"
g++ -O2 -fPIC -shared -std=c++17 \
    -I"$RTG/inc" -I/usr/include \
    "$HERE/shm_bridge.cpp" \
    -o "$OUT" \
    -lRobotSharedMem -lrt -lpthread \
  && echo "✅ 빌드 OK: $OUT" \
  || { echo "❌ 빌드 실패 — RobotSharedMem.h/lib 및 RobotTestGait/inc 경로 확인"; exit 1; }
