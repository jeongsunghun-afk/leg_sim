#!/bin/bash
# est_diag.sh — 추정기 설정(config)별로 **추정오차와 실제 드리프트를 갈라서** 잰다.
#
# ═══ 왜 sweep_paired.sh 로는 안 되나 ═══
#   그쪽은 (T_STEP × K_RETURN) 게인 격자의 **낙상률**이 목적이라 GT 만 저장한다.
#   그래서 "추정기가 맞아졌나" 와 "로봇이 실제로 흘렀나" 를 구분하지 못한다. 여기선 가른다:
#       est_err = |추정x − GTx|   ← 추정기 품질
#       gt|x|   = |GTx|           ← 실제 이동량
#       est|x|  = |추정x|         ← 추정기가 본 것 (0 에 가까우면 **전진을 아예 못 본 것**)
#   ⚠낙상시드는 reset() 이 로봇을 원점으로 되돌리므로(biped_control.hpp:478) 셋 다 무의미하다.
#     → 생존시드만 집계한다. sweep_paired.sh 집계와 같은 규칙이다.
#
# ═══ 왜 config 단위인가 — 추정기 knob 은 짝으로 동작한다 ═══
#   dwell 은 나쁜 속도표본을 **버리고**, anchor 는 그 결과 생기는 적분오차를 **위치로 묶는다.**
#   한쪽만 켜고 재면 결론이 뒤집힌다 — 실제로 그런 일이 있었다(2026-08-19, 아래).
#   그래서 knob 하나가 아니라 **조합 전체**를 한 줄로 놓고 비교한다.
#
# ═══ 사용 ═══
#   ./est_diag.sh [출력.tsv]
#   CONFIGS 는 "라벨|ENV=VAL ENV=VAL" 을 줄바꿈으로 나열한다. 기본은 아래 2×2.
#   그 외: SEEDS=12 T_STEP=0.30 K_RETURN=0.15 VX=0.00 DUR=40 JOBS=20
#          NOISE_ENV="" 로 주면 **무노이즈** 비교(옛 측정 재현용).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
MJ="${MUJOCO_DIR:-$HOME/mujoco}"
MJLIB="${MUJOCO_LIB_DIR:-$MJ/lib}"      # ⚠pip 휠이면 site-packages/mujoco 밑이다(sweep_paired.sh 주석 참조)
MODEL="${MODEL:-$HERE/../biped_from_quad.mjcf}"
OUT="${1:-/tmp/est_diag.tsv}"

SEEDS="${SEEDS:-12}"
T_STEP="${T_STEP:-0.30}"
K_RETURN="${K_RETURN:-0.15}"
DUR="${DUR:-40}"
VX="${VX:-0.00}"
JOBS="${JOBS:-20}"

# ★기본 격자 = **C++ 기본 vs Python 배포 기본**의 2×2.
#   deploy/biped_deploy.py:69-71 은 dwell_steps=15 · k_anchor=0.05 로 **둘 다 켜져 있다.**
#   C++ deploy_loop.hpp 는 둘 다 0 이다 — 포팅에서 빠졌다. 그 차이가 뭘 하는지 가른다.
CONFIGS="${CONFIGS:-cpp기본|EST_DWELL=0 EST_ANCHOR=0
앵커만|EST_DWELL=0 EST_ANCHOR=0.05
dwell만|EST_DWELL=15 EST_ANCHOR=0
py등가|EST_DWELL=15 EST_ANCHOR=0.05}"

DEPLOY="${DEPLOY:-EST_CTRL=1 ACT_LAT_MS=8.4 LAT_COMP_MS=8.4 LAT_COMP_KIN=1}"
NOISE_ENV="${NOISE_ENV-ENCQ_N=7.64e-5 ENCDQ_N=0.0368}"   # ⚠'-' 이므로 빈 문자열 지정 가능

[ -x "$HERE/build/biped_sim" ] || { echo "✗ build/biped_sim 없음 — 먼저 빌드"; exit 1; }

CFGFILE=$(mktemp); printf "%s\n" "$CONFIGS" > "$CFGFILE"
trap 'rm -f "$CFGFILE"' EXIT

run_one() {   # $1=config 행번호 $2=seed
  local line label cenv out falls ex gx tilt
  line=$(sed -n "${1}p" "$CFGFILE")
  label="${line%%|*}"; cenv="${line#*|}"
  out=$(cd "$HERE" && env LD_LIBRARY_PATH="$MJLIB:${LD_LIBRARY_PATH:-}" $DEPLOY $NOISE_ENV $cenv \
        T_STEP="$T_STEP" K_RETURN="$K_RETURN" SEED="$2" \
        timeout 600 ./build/biped_sim "$MODEL" "$VX" "$DUR" 2>/dev/null | tail -n 1)
  falls=$(sed -n 's/.*falls=\([0-9]*\).*/\1/p' <<<"$out")
  ex=$(sed -n 's/.*base=(\([^,]*\),.*/\1/p' <<<"$out")
  gx=$(sed -n 's/.*GT=(\([^,]*\),.*/\1/p' <<<"$out")
  tilt=$(sed -n 's/.*tilt=\([0-9.]*\).*/\1/p' <<<"$out")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$label" "$2" "${falls:-99}" "${ex:-9.99}" "${gx:-9.99}" "${tilt:-999}"
}
export -f run_one
export HERE MJLIB MODEL DEPLOY NOISE_ENV T_STEP K_RETURN DUR VX CFGFILE

NCFG=$(wc -l < "$CFGFILE")
{ printf "idx\tconfig\tseed\tfalls\test_x\tgt_x\ttilt\n"
  for i in $(seq 1 "$NCFG"); do for s in $(seq 1 "$SEEDS"); do echo "$i $s"; done; done \
    | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"'
} > "$OUT"

echo "== 추정기 config 비교 — T_STEP=$T_STEP K_RETURN=$K_RETURN vx=$VX ${DUR}s · 시드 $SEEDS 개 =="
[ -n "$NOISE_ENV" ] && echo "   노이즈: $NOISE_ENV" || echo "   노이즈: **없음**"
python3 - "$OUT" <<'PY'
import sys, collections
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])][1:]
cell = collections.defaultdict(list)
order = {}
for r in rows:
    if len(r) < 7: continue
    order.setdefault(r[1], int(r[0]))
    cell[r[1]].append((int(float(r[3])), float(r[4]), float(r[5]), float(r[6])))
# ★실행 실패(falls=99)는 **낙상이 아니다** — 분모에서 빼고 err 열로 따로 낸다.
#   섞어 세면 "8/12 낙상" 같은 유령이 생긴다(2026-08-19 마찰 실험에서 실제로 그랬다).
ERR = 99
print(f"  {'config':<12}{'falls':<9}{'est_err':<10}{'gt|x|':<10}{'est|x|':<10}{'tilt':<8}{'n_surv':<8}{'err'}")
for w in sorted(cell, key=lambda k: order[k]):
    v = cell[w]
    ne  = sum(1 for f, *_ in v if f == ERR)
    nok = len(v) - ne
    nf  = sum(1 for f, *_ in v if 0 < f < ERR)
    surv = [(e, g, t) for f, e, g, t in v if f == 0]
    serr = f"✗{ne}" if ne else ""
    if surv:
        ee = sum(abs(e - g) for e, g, _ in surv) / len(surv)
        gg = sum(abs(g) for _, g, _ in surv) / len(surv)
        eg = sum(abs(e) for e, _, _ in surv) / len(surv)
        mt = max(t for _, _, t in surv)
        print(f"  {w:<12}{f'{nf}/{nok}':<9}{ee:<10.3f}{gg:<10.3f}{eg:<10.3f}{mt:<8.1f}{len(surv):<8}{serr}")
    else:
        print(f"  {w:<12}{f'{nf}/{nok}':<9}{'—전멸':<10}{'—':<10}{'—':<10}{'—':<8}{0:<8}{serr}")
print()
print("  est_err=|추정x−GTx| · gt|x|=실제이동 · est|x|=추정이 본 것 (낙상시드 제외)")
print("  ⚠err 열은 **실행 실패**다(낙상 아님). 0 이 아니면 그 행을 믿지 말 것.")
print("  ⚠est|x| ≈ 0 인데 gt|x| 가 크면 **추정기가 전진을 아예 못 본 것**이다.")
print("    컨트롤러는 그 추정을 믿으므로(발디딤이 K_RETURN·err_b) 되잡지 않는다 — 낙상 0 이라 안 보인다.")
PY
echo "원본: $OUT"
