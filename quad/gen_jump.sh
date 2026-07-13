#!/usr/bin/env bash
# 점프 궤적 생성 (C++ crocoddyl OCP): jump_ocp가 OCP solve + pin→mj 변환 → /tmp/jump_traj.txt 직접 출력.
#   사용: bash gen_jump.sh [VX] [maxit]   VX=전방 이륙속도 m/s(0=수직 제자리, 기본 0.6), maxit=FDDP 반복(기본 200)
#   예:   bash gen_jump.sh 0        → 수직 제자리 / bash gen_jump.sh 0.9 → 더 멀리
# ★S3-b: C++ 단일 실행이 구 Python 2단계(jump_ocp.py + jump_track.py)를 대체(Python판은 offline/jump/에 보존).
#   pin 16-DOF(허리 lock) OCP → MuJoCo 17-DOF replay 포맷 변환까지 C++서 수행. /tmp는 휘발성 → 안 될 때 재생성.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # simulation/quad
CPP="$HERE/cpp"
URDF="${URDF:-$HERE/../02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf}"
VX="${1:-0.6}"; MAXIT="${2:-200}"

[ -f "$URDF" ] || { echo "❌ URDF 없음: $URDF (URDF=... 로 경로 지정)"; exit 1; }
cmake --build "$CPP/build" --target jump_ocp >/dev/null 2>&1 || { echo "❌ jump_ocp 빌드 실패 — crocoddyl/pinocchio 환경 확인"; exit 1; }

JUMP_OUT=/tmp/jump_traj.txt "$CPP/build/jump_ocp" "$URDF" "$VX" "$MAXIT"
echo "완료: /tmp/jump_traj.txt (C++ crocoddyl OCP · 전방 이륙속도 VX=$VX m/s)"
