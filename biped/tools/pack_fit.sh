#!/usr/bin/env bash
# pack_fit.sh — 적합(CMA-ES)을 **다른 컴퓨터에서 돌리기 위한** 꾸러미를 만든다.
#
# ★왜 (2026-08-14, 사용자 요청)
#   적합은 저장된 npz 만 쓰는 **순수 오프라인 계산**이다. 실기도, SHM 도, 드라이버도
#   필요 없다(`pace_cmaes.py` 의 import 는 numpy·mujoco·yaml·cma 뿐).
#   그런데 이 기기(Pi 4, 4코어)에서 돌리면 두 가지 손해가 있다:
#     ① **1시간 반**이 걸린다 — 노트북이면 대개 5~15분이다.
#     ② CPU 경합이 **드라이버 래치오프**를 부른다. 파이썬 GC 로 330ms 루프정지가
#        찍힌 전례가 있고(2026-08-12), 그동안 드라이버에 명령이 안 간다.
#        ⇒ 적합을 여기서 돌리는 동안은 실기를 못 만진다. 그게 진짜 비용이다.
#
# ⚠MJCF 와 메시를 **그대로** 넣는다. 메시(37MB)는 충돌형상뿐이고 롤아웃에서는
#   base 고정·바닥 제거라 `d.ncon == 0` 이지만, 모델 파일이 두 기기에서 달라지면
#   결과를 비교할 수 없다. 재현성이 전송량보다 중요하다.
#
# 사용:
#     bash tools/pack_fit.sh                       # 최신 npz 하나
#     bash tools/pack_fit.sh a.npz b.npz           # 특정 npz 들
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
BIPED=$(dirname "$HERE")
ROOT=$(dirname "$BIPED")
B=$(basename "$BIPED")
cd "$ROOT"

if [ "$#" -gt 0 ]; then
  NPZ=("$@")
else
  # 최신 수집본 하나. *_cmaes.npz(적합 산출물)는 입력이 아니므로 뺀다.
  mapfile -t NPZ < <(ls -t "$BIPED"/emb/pace/results/pace_multichirp*.npz 2>/dev/null \
                     | grep -v _cmaes | head -1)
fi
[ "${#NPZ[@]}" -gt 0 ] || { echo "✗ 넣을 npz 가 없다. 경로를 인자로 줄 것"; exit 1; }

MESHDIR=$(grep -o 'meshdir="[^"]*"' "$BIPED/biped_flatfoot.mjcf" | sed 's/meshdir="//;s/"//')
FILES=("$B/emb/pace/pace_cmaes.py" "$B/emb/pace/spec.yaml"
       "$B/biped_flatfoot.mjcf" "$B/tools/compare_params.py" "$B/${MESHDIR%/}")
for f in "${NPZ[@]}"; do FILES+=("${f#"$ROOT"/}"); done

for f in "${FILES[@]}"; do
  [ -e "$ROOT/$f" ] || { echo "✗ 없다: $f"; exit 1; }
done

OUT="$ROOT/pace_fit_bundle.tar.gz"
tar czf "$OUT" "${FILES[@]}"
echo "✓ $OUT  ($(du -h "$OUT" | cut -f1))"
echo
echo "── 노트북에서 ─────────────────────────────────────────────────────"
echo "  scp rpetubt@<이 기기 IP>:$OUT ."
echo "  tar xzf pace_fit_bundle.tar.gz"
echo "  python3 -m venv .venv && .venv/bin/pip install mujoco numpy pyyaml cma"
echo
for f in "${NPZ[@]}"; do
  echo "  .venv/bin/python $B/emb/pace/pace_cmaes.py \\"
  echo "      $B/emb/pace/results/$(basename "$f") --holdout 0.2 --pin hip"
done
echo
echo "  # 먼저 이걸로 **꾸러미가 온전한지** 확인할 것 (CMA-ES 없이 초기값만 평가)"
echo "  .venv/bin/python $B/emb/pace/pace_cmaes.py \\"
echo "      $B/emb/pace/results/$(basename "${NPZ[0]}") --holdout 0.2 --eval-only"
echo
echo "  # 결과 npz 를 이 기기로 되가져와서 대조"
echo "  scp *_cmaes.npz rpetubt@<이 기기 IP>:$BIPED/emb/pace/results/"
echo "  python3 $B/tools/compare_params.py .../<이름>_cmaes.npz"
echo "───────────────────────────────────────────────────────────────────"
echo "⚠결과는 기기가 달라도 **같아야 한다** — CMA-ES 시드가 고정이 아니라 완전히"
echo "  같진 않지만, RMS 와 파라미터가 크게 갈리면 모델·데이터가 다른 것이다."
