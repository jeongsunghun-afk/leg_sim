#!/bin/bash
# dwell_diag.sh — EST_DWELL 이 "추정기 문제"를 고쳤는지 "로봇 거동"을 바꿨는지 가른다.
#
# 짝 스윕 표는 GT 만 저장해서 둘을 구분 못 한다. 여기선 마지막 줄의 추정치와 GT 를 둘 다 뽑아
#   est_err = |x_est − x_gt|   ← 추정기가 맞아졌나 (헤더 주장: 0.46 → 0.01)
#   gt_x    = |x_gt|           ← 로봇이 실제로 얼마나 흘렀나
# 를 따로 본다. 낙상시드는 reset() 때문에 둘 다 무의미하므로 제외한다.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MJ="${MUJOCO_DIR:-$HOME/mujoco}"
MJLIB="${MUJOCO_LIB_DIR:-$MJ/lib}"      # ⚠pip 휠이면 site-packages/mujoco 밑이다(sweep_paired.sh 주석 참조)
MODEL="${MODEL:-$HERE/../biped_from_quad.mjcf}"
OUT="${1:-/tmp/dwell_diag.tsv}"
DWELLS="${DWELLS:-0 5}"
SEEDS="${SEEDS:-12}"
T_STEP="${T_STEP:-0.30}"
K_RETURN="${K_RETURN:-0.15}"
DUR="${DUR:-40}"
VX="${VX:-0.00}"
JOBS="${JOBS:-20}"

DEPLOY="EST_CTRL=1 ACT_LAT_MS=8.4 LAT_COMP_MS=8.4 LAT_COMP_KIN=1"
NOISE_ENV="ENCQ_N=7.64e-5 ENCDQ_N=0.0368"

run_one() {   # $1=dwell $2=seed
  local out falls ex gx tilt
  out=$(cd "$HERE" && env LD_LIBRARY_PATH="$MJLIB:${LD_LIBRARY_PATH:-}" $DEPLOY $NOISE_ENV \
        T_STEP="$T_STEP" K_RETURN="$K_RETURN" EST_DWELL="$1" SEED="$2" \
        timeout 600 ./build/biped_sim "$MODEL" "$VX" "$DUR" 2>/dev/null | tail -n 1)
  falls=$(sed -n 's/.*falls=\([0-9]*\).*/\1/p' <<<"$out")
  ex=$(sed -n 's/.*base=(\([^,]*\),.*/\1/p' <<<"$out")
  gx=$(sed -n 's/.*GT=(\([^,]*\),.*/\1/p' <<<"$out")
  tilt=$(sed -n 's/.*tilt=\([0-9.]*\).*/\1/p' <<<"$out")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "${falls:-99}" "${ex:-9.99}" "${gx:-9.99}" "${tilt:-999}"
}
export -f run_one
export HERE MJLIB MODEL DEPLOY NOISE_ENV T_STEP K_RETURN DUR VX

{ printf "DWELL\tseed\tfalls\test_x\tgt_x\ttilt\n"
  for w in $DWELLS; do for s in $(seq 1 "$SEEDS"); do echo "$w $s"; done; done \
    | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"'
} > "$OUT"

echo "== EST_DWELL 진단 — T_STEP=$T_STEP K_RETURN=$K_RETURN vx=$VX ${DUR}s · 시드 $SEEDS 개 =="
python3 - "$OUT" <<'PY'
import sys, collections
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
cell = collections.defaultdict(list)
for r in rows:
    if len(r) < 6: continue
    cell[r[0]].append((int(float(r[2])), float(r[3]), float(r[4]), float(r[5])))
print(f"  {'DWELL':<8}{'falls':<8}{'est_err|x|':<13}{'gt|x|':<10}{'tilt':<8}{'n_surv'}")
for w in sorted(cell, key=float):
    v = cell[w]
    nf = sum(1 for f, *_ in v if f > 0)
    surv = [(e, g, t) for f, e, g, t in v if f == 0]
    if surv:
        ee = sum(abs(e - g) for e, g, _ in surv) / len(surv)
        gg = sum(abs(g) for _, g, _ in surv) / len(surv)
        mt = max(t for _, _, t in surv)
        print(f"  {w:<8}{f'{nf}/{len(v)}':<8}{ee:<13.3f}{gg:<10.3f}{mt:<8.1f}{len(surv)}")
    else:
        print(f"  {w:<8}{f'{nf}/{len(v)}':<8}{'—전멸':<13}{'—':<10}{'—':<8}0")
print()
print("  est_err = |추정x − GTx| : 추정기가 맞아졌는지. 헤더 주장대로면 dwell 켜면 확 준다.")
print("  gt|x|   = 실제 이동량   : 로봇이 실제로 흘렀는지. 낙상시드는 reset() 때문에 제외했다.")
PY
echo "원본: $OUT"
