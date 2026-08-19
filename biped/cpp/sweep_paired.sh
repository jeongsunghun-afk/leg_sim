#!/bin/bash
# sweep_paired.sh — 발디딤 게인 × 추정기 짝 스윕. **셀당 N시드 낙상률**로 낸다.
#
# ═══ 왜 기존 sweep_stability.sh 로는 안 되나 ═══
#
# ① **셀당 1회는 근거가 못 된다.** 이 시스템은 혼돈적으로 민감하다 — 실측:
#      JFRIC.calf 0.5720            → 2점 stand 60s 무낙상
#      JFRIC.calf 0.572000000000572 → **26.28s 낙상**   (상대차 **1e-12**)
#    1e-12 는 **부동소수점 반올림 수준**이다. 즉 단일 실행의 낙상/무낙상은
#    컴파일 플래그·CPU(x86 vs ARM)만 달라도 뒤집힌다. **기기 간에 안 옮겨진다.**
#    ⇒ 옮겨지는 건 **통계**뿐이다. 셀마다 시드를 여러 개 돌려 **낙상률**을 본다.
#    ⚠1e-9·1e-6 섭동은 안 뒤집혔다 — 크기 문제가 아니라 사실상 난수라는 뜻이다.
#
# ② 기존 스크립트는 `vx × T_STEP` 만 훑고 **배포조건(EST_CTRL·지연)이 없다.**
#    지금 풀려는 문제는 배포경로에서만 나타난다.
#
# ③ 시드를 주려면 **센서노이즈가 켜져 있어야** 한다(SEED 는 노이즈 실현만 바꾼다).
#    2026-08-05 실측값을 쓴다: ENCQ_N=7.64e-5 rad · ENCDQ_N=0.0368 rad/s.
#
# ═══ 무엇을 푸는가 ═══
#
#   `EST_DWELL` 을 켜면 제자리 드리프트가 사라진다(60초 0.43m → 0.02m).
#   그런데 **낙상이 생긴다**(2회/60초). 발디딤 게인이 그 편향 위에서 맞춰져 있어서다.
#   2026-08-05 에 같은 일이 있었고, 그때도 추정기 수정과 게인을 **짝으로** 스윕해 풀었다.
#   ⇒ 목표: 낙상률 0 이면서 드리프트가 작은 (T_STEP, K_RETURN, EST_DWELL) 을 찾는다.
#
# ═══ 사용 ═══
#   ./sweep_paired.sh [출력.tsv]
#   격자: TSTEPS="0.26 0.30 0.34" KRETS="0.10 0.15 0.20" DWELLS="0 5 10" SEEDS=5
#   그 외: VX=0.00 DUR=40 JOBS=8
#
#   ★노트북(다중코어)에서 JOBS 를 코어수-2 로 주고 돌릴 것. 최종 채택점만 Pi 에서 재확인.
#   ⚠MuJoCo 는 **3.9.0** 이어야 한다(3.11 은 d.qM→d.M · mj_fullM 시그니처가 바뀌어 안 빌드된다).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/tmp/sweep_paired.tsv}"
MJ="${MUJOCO_DIR:-$HOME/mujoco}"
MODEL="${MODEL:-$HERE/../biped_from_quad.mjcf}"

TSTEPS="${TSTEPS:-0.26 0.30 0.34}"
KRETS="${KRETS:-0.10 0.15 0.20}"
DWELLS="${DWELLS:-0 5 10}"
SEEDS="${SEEDS:-5}"
VX="${VX:-0.00}"
DUR="${DUR:-40}"
JOBS="${JOBS:-4}"

# 배포조건 — verify.sh 의 DEPLOY_ENV 와 같아야 한다. 여기에 실측 센서노이즈를 더한다.
DEPLOY="EST_CTRL=1 ACT_LAT_MS=8.4 LAT_COMP_MS=8.4 LAT_COMP_KIN=1"
NOISE_ENV="ENCQ_N=7.64e-5 ENCDQ_N=0.0368"

[ -x "$HERE/build/biped_sim" ] || { echo "✗ build/biped_sim 없음 — 먼저 빌드"; exit 1; }

run_one() {   # $1=T_STEP $2=K_RETURN $3=EST_DWELL $4=seed
  local out falls gx tilt
  out=$(cd "$HERE" && env LD_LIBRARY_PATH="$MJ/lib" $DEPLOY $NOISE_ENV \
        T_STEP="$1" K_RETURN="$2" EST_DWELL="$3" SEED="$4" \
        timeout 600 ./build/biped_sim "$MODEL" "$VX" "$DUR" 2>/dev/null | tail -n 1)
  falls=$(sed -n 's/.*falls=\([0-9]*\).*/\1/p' <<<"$out")
  gx=$(sed -n 's/.*GT=(\([^,]*\),.*/\1/p' <<<"$out")
  tilt=$(sed -n 's/.*tilt=\([0-9.]*\).*/\1/p' <<<"$out")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "${falls:-99}" "${gx:-9.99}" "${tilt:-999}"
}
export -f run_one; export HERE MJ MODEL VX DUR DEPLOY NOISE_ENV

{ printf "T_STEP\tK_RETURN\tDWELL\tseed\tfalls\tgt_x\ttilt\n"
  for t in $TSTEPS; do for k in $KRETS; do for w in $DWELLS; do
    for s in $(seq 1 "$SEEDS"); do echo "$t $k $w $s"; done
  done; done; done | xargs -P "$JOBS" -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"'
} > "$OUT"

echo "== 셀별 집계 (낙상률 · 평균|드리프트| · 최대tilt) — 시드 $SEEDS 개 =="
python3 - "$OUT" <<'PY' 2>/dev/null || echo "  (python3 없음 — 원본 TSV 를 직접 볼 것: $OUT)"
import sys, collections
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
cell = collections.defaultdict(list)
for r in rows:
    if len(r) < 7: continue
    cell[(r[0], r[1], r[2])].append((int(float(r[4])), abs(float(r[5])), float(r[6])))
print(f"  {'T_STEP':<8}{'K_RET':<8}{'DWELL':<7}{'낙상률':<9}{'평균|x|':<10}{'최대tilt'}")
out = []
for k, v in cell.items():
    nf = sum(1 for f, _, _ in v if f > 0)
    out.append((nf / len(v), sum(x for _, x, _ in v) / len(v), max(t for _, _, t in v), k, len(v)))
for rate, mx, mt, k, n in sorted(out):
    mark = "  ★" if rate == 0 and mx < 0.10 else ""
    print(f"  {k[0]:<8}{k[1]:<8}{k[2]:<7}{f'{int(rate*n)}/{n}':<9}{mx:<10.3f}{mt:<8.1f}{mark}")
print("\n  ★ = 낙상 0 + 드리프트 10cm 미만. 이게 후보다.")
print("  ⚠단일 셀이 0/N 이라도 N 이 작으면 우연이다 — 후보는 SEEDS 를 늘려 재확인할 것.")
PY
echo
echo "원본: $OUT"
