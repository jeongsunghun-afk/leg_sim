#!/bin/bash
# sweep_stability.sh — 속도 × 스텝시간 안정영역 지도.
#
# ★왜 필요한가: 2026-08-05 실측 ROTOR_I(7.4e-4) 반영 후 T_STEP 을 0.24→0.32 로 재튜닝했는데,
#   그 근거가 **vx=0.15 단일조건 4점 스윕**뿐이었다. 실기 배포 전에 속도대역 전반에서
#   안정한지 확인해야 한다. (0.40/0.50 에서 다시 낙상한 것으로 보아 최적점이 좁다.)
#
# 사용: ./sweep_stability.sh [출력파일]
#   env 로 격자 조정: VXS="0.05 0.1 …" TSTEPS="0.24 …" ROTOR_I=… DUR=15
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/tmp/sweep_stability.tsv}"
MJ="${MUJOCO_DIR:-$HOME/mujoco}"
MODEL="${MODEL:-$HERE/../biped_from_quad.mjcf}"
VXS="${VXS:-0.05 0.10 0.15 0.20 0.25 0.30}"
TSTEPS="${TSTEPS:-0.24 0.28 0.32 0.36 0.40}"
DUR="${DUR:-15}"
JOBS="${JOBS:-4}"

[ -x "$HERE/build/biped_sim" ] || { echo "biped_sim 없음 — 먼저 빌드"; exit 1; }

run_one() {   # $1=vx $2=T_STEP
  local out
  out=$(cd "$HERE" && LD_LIBRARY_PATH="$MJ/lib" T_STEP="$2" \
        timeout 300 ./build/biped_sim "$MODEL" "$1" "$DUR" 2>&1 | tail -n 1)
  # "vx=0.15 · 생존 15.00s(무낙상) · base=(…) tilt=8.8°"
  local surv tilt fell
  surv=$(sed -n 's/.*생존 \([0-9.]*\)s.*/\1/p' <<<"$out")
  tilt=$(sed -n 's/.*tilt=\([0-9.]*\).*/\1/p' <<<"$out")
  fell=$(grep -q 무낙상 <<<"$out" && echo 0 || echo 1)
  printf "%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "${surv:-0}" "${tilt:-999}" "$fell"
}
export -f run_one; export HERE MJ MODEL DUR

{ printf "vx\tT_STEP\tsurvive_s\ttilt_deg\tfell\n"
  for v in $VXS; do for t in $TSTEPS; do echo "$v $t"; done; done \
    | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"'
} > "$OUT"

echo "== 안정영역 지도 (숫자=생존초, ✗=낙상) =="
python3 - "$OUT" <<'PYEOF'
import sys
rows=[l.split('\t') for l in open(sys.argv[1]).read().splitlines()[1:] if l.strip()]
d={(r[0],r[1]):(float(r[2]),float(r[3]),int(r[4])) for r in rows}
vxs=sorted({r[0] for r in rows}, key=float); ts=sorted({r[1] for r in rows}, key=float)
print("vx\\T " + "".join(f"{t:>12s}" for t in ts))
for v in vxs:
    cells=[]
    for t in ts:
        s_,tilt,fell=d.get((v,t),(0.0,999.0,1))
        cells.append(f"{'X '+format(s_,'.1f')+'s':>12s}" if fell else f"{format(s_,'.0f')+'s '+format(tilt,'.1f')+chr(176):>12s}")
    print(f"{v:>5s} " + "".join(cells))
ok=[(v,t) for (v,t),(s_,ti,f) in d.items() if not f]
print(f"\n무낙상 {len(ok)}/{len(d)}")
if ok:
    best=min(ok, key=lambda k: d[k][1])
    print(f"최소 tilt: vx={best[0]} T_STEP={best[1]} -> tilt {d[best][1]:.1f} deg")
    from collections import defaultdict
    per=defaultdict(list)
    for v,t in ok: per[t].append(float(v))
    print("T_STEP 별 무낙상 속도범위:")
    for t in sorted(per, key=float):
        print(f"  T={t}: vx {min(per[t]):.2f}~{max(per[t]):.2f} ({len(per[t])}개)")
PYEOF
