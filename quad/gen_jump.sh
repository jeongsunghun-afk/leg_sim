#!/usr/bin/env bash
# 점프 궤적 생성: J1 OCP(crocoddyl) → J2 추종변환 → /tmp/jump_traj.txt (C++ 점프모드가 로드해 재생)
#   사용: bash gen_jump.sh [VX]    VX=전방 이륙속도 m/s (0=수직 제자리, 기본 0.6=전방 점프)
#   예:   bash gen_jump.sh 0       → 수직 제자리 점프 / bash gen_jump.sh 0.9 → 더 멀리
# ★/tmp는 휘발성 → 재부팅/점프 안 될 때 이 스크립트로 재생성.
set -e
PXI=/home/jsh/miniforge3/envs/proxddp/bin/python
cd "$(dirname "${BASH_SOURCE[0]}")"
VX="${1:-0.6}"   # ★기본=전방 점프(0.6). 배포 점프는 전방(로봇 정면 방향)
JUMP_VX="$VX" "$PXI" offline/jump/jump_ocp.py
"$PXI" offline/jump/jump_track.py
echo "완료: /tmp/jump_traj.txt (전방 이륙속도 VX=$VX m/s)"
