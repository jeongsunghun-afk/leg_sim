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
#   격자: TSTEPS="0.26 0.30 0.34" KRETS="0.10 0.15 0.20" ALPHAS="0.4 0.7 1.0" SEEDS=5
#   그 외: VX=0.00 DUR=40 JOBS=8
#
#   ★노트북(다중코어)에서 JOBS 를 코어수-2 로 주고 돌릴 것. 최종 채택점만 Pi 에서 재확인.
#   ⚠MuJoCo 는 **3.9.0** 이어야 한다(3.11 은 d.qM→d.M · mj_fullM 시그니처가 바뀌어 안 빌드된다).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-/tmp/sweep_paired.tsv}"
MJ="${MUJOCO_DIR:-$HOME/mujoco}"
# ★MuJoCo 공유라이브러리 디렉터리. **`$MJ/lib` 이 아닐 수 있다** —
#   pip 휠로 깔면 `site-packages/mujoco/libmujoco.so.3.9.0` 처럼 패키지 바로 밑에 있다.
#   ⚠종전엔 LD_LIBRARY_PATH 를 `$MJ/lib` 로 **덮어써서** 그 경우 실행이 안 됐다
#     (밖에서 export 해도 소용없다 — env 로 치환하기 때문). 그래서 분리·append 한다.
MJLIB="${MUJOCO_LIB_DIR:-$MJ/lib}"
MODEL="${MODEL:-$HERE/../biped_from_quad.mjcf}"

TSTEPS="${TSTEPS:-0.26 0.30 0.34}"
KRETS="${KRETS:-0.10 0.15 0.20}"
# ★2026-08-19 3번째 격자축을 **DWELL → EST_ALPHA** 로 교체했다. dwell 은 기각됐고
#   (배포조건에서 추정오차 2.6배·낙상 6/8), 대신 **저역통과 alpha 가 진범**으로 나왔다:
#     alpha 0.2→0.235 · 0.4(기본)→0.182 · 0.7→0.104 · **1.0→0.034 m** (추정오차, 단조)
#   ⚠옛 TSV 는 3열이 DWELL 이다. 집계는 **헤더를 읽어** 구분하니 섞어 쓰지만 말 것.
ALPHAS="${ALPHAS:-0.4 0.7 1.0}"
SEEDS="${SEEDS:-5}"
VX="${VX:-0.00}"
DUR="${DUR:-40}"
JOBS="${JOBS:-4}"

# 배포조건 — verify.sh 의 DEPLOY_ENV 와 같아야 한다. 여기에 실측 센서노이즈를 더한다.
DEPLOY="EST_CTRL=1 ACT_LAT_MS=8.4 LAT_COMP_MS=8.4 LAT_COMP_KIN=1"
NOISE_ENV="ENCQ_N=7.64e-5 ENCDQ_N=0.0368"

# AGG_ONLY=1 이면 스윕을 다시 돌리지 않고 기존 TSV 만 재집계한다.
#   ★집계 규칙을 고칠 때마다 스윕을 다시 돌릴 이유가 없다. 원본 TSV 는 그대로다.
#   실제로 이게 필요했다 — 낙상시드를 빼는 규칙으로 바꾸자 189런의 **결론이 뒤집혔다**.
AGG_ONLY="${AGG_ONLY:-0}"

if [ "$AGG_ONLY" != 1 ]; then
[ -x "$HERE/build/biped_sim" ] || { echo "✗ build/biped_sim 없음 — 먼저 빌드"; exit 1; }
fi

run_one() {   # $1=T_STEP $2=K_RETURN $3=EST_ALPHA $4=seed
  local out falls gx tilt
  out=$(cd "$HERE" && env LD_LIBRARY_PATH="$MJLIB:${LD_LIBRARY_PATH:-}" $DEPLOY $NOISE_ENV \
        T_STEP="$1" K_RETURN="$2" EST_ALPHA="$3" SEED="$4" \
        timeout 600 ./build/biped_sim "$MODEL" "$VX" "$DUR" 2>/dev/null | tail -n 1)
  falls=$(sed -n 's/.*falls=\([0-9]*\).*/\1/p' <<<"$out")
  gx=$(sed -n 's/.*GT=(\([^,]*\),.*/\1/p' <<<"$out")
  tilt=$(sed -n 's/.*tilt=\([0-9.]*\).*/\1/p' <<<"$out")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "${falls:-99}" "${gx:-9.99}" "${tilt:-999}"
}
export -f run_one; export HERE MJ MJLIB MODEL VX DUR DEPLOY NOISE_ENV

if [ "$AGG_ONLY" != 1 ]; then
{ printf "T_STEP\tK_RETURN\tALPHA\tseed\tfalls\tgt_x\ttilt\n"
  for t in $TSTEPS; do for k in $KRETS; do for w in $ALPHAS; do
    for s in $(seq 1 "$SEEDS"); do echo "$t $k $w $s"; done
  done; done; done | xargs -P "$JOBS" -n 4 bash -c 'run_one "$0" "$1" "$2" "$3"'
} > "$OUT"
else
  [ -s "$OUT" ] || { echo "✗ AGG_ONLY=1 인데 $OUT 이 없다"; exit 1; }
  echo "(AGG_ONLY=1 — 스윕은 건너뛰고 $OUT 만 재집계한다)"
fi

echo "== 셀별 집계 =="
python3 - "$OUT" <<'PY' 2>/dev/null || echo "  (python3 없음 — 원본 TSV 를 직접 볼 것: $OUT)"
import sys, collections, math

# ═══ 왜 낙상시드를 드리프트 집계에서 빼는가 ═══
#   낙상하면 sim 이 reset() 으로 로봇을 **원점으로 되돌린다**. 그 뒤 남은 시간 동안
#   원점 근처에 있으니 |x| 가 **인위적으로 작게** 찍힌다. 그래서 낙상 많은 셀일수록
#   드리프트가 좋아 보이는 역전이 생긴다(실측: DWELL=10 이 7/7 낙상인데 |x| 최소).
#   ⇒ 드리프트·tilt 는 **살아남은 시드만** 집계한다. STABILITY_MAP 의 경고와 같은 규칙이다.

ERR = 99          # run_one 이 출력 파싱에 실패했을 때 넣는 sentinel
ALPHA = 0.05

def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))

def cp_upper(k, n, alpha=ALPHA):
    """낙상률의 Clopper-Pearson 95% 상한. 0/7 이면 0.41 — 즉 n=7 로는 '0%' 를 못 주장한다."""
    if k >= n: return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > alpha / 2: lo = mid
        else: hi = mid
    return (lo + hi) / 2

_all = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
# ★3열의 의미는 파일마다 다르다(옛 파일=DWELL · 새 파일=ALPHA). **헤더에서 읽는다.**
KNOB = _all[0][2] if _all and len(_all[0]) > 2 else "KNOB"
rows = _all[1:]
cell = collections.defaultdict(list)
for r in rows:
    if len(r) < 7: continue
    cell[(r[0], r[1], r[2])].append((int(float(r[4])), abs(float(r[5])), float(r[6])))

nerr = sum(1 for v in cell.values() for f, _, _ in v if f == ERR)
print("  ⚠드리프트·tilt 는 **낙상 안 한 시드만** 집계한다 — 낙상하면 reset() 이 원점으로")
print("    되돌려 |x| 가 인위적으로 작게 찍힌다. 낙상 셀의 드리프트는 읽으면 안 된다.")
print()
print(f"  {'T_STEP':<8}{'K_RET':<8}{KNOB:<7}{'falls':<8}{'상한95%':<10}"
      f"{'|x|surv':<10}{'tilt_surv':<11}{'n_surv'}".replace("DWELL  ", f"{KNOB:<7}"))

out = []
for k, v in cell.items():
    n = len(v)
    nf = sum(1 for f, _, _ in v if f > 0)
    surv = [(x, t) for f, x, t in v if f == 0]
    dx = sum(x for x, _ in surv) / len(surv) if surv else None
    mt = max(t for _, t in surv) if surv else None
    out.append((nf / n, dx if dx is not None else float("inf"), nf, n, dx, mt, len(surv), k))

for rate, _sort, nf, n, dx, mt, ns, k in sorted(out):
    up = cp_upper(nf, n)
    # ★ = 낙상 0 + 드리프트 10cm 미만 + 상한이 10% 아래(= 시드가 충분히 많다). 이게 채택 후보다.
    mark = "  ★" if nf == 0 and dx is not None and dx < 0.10 and up < 0.10 else ""
    sdx = f"{dx:.3f}" if dx is not None else "—전멸"
    smt = f"{mt:.1f}" if mt is not None else "—"
    print(f"  {k[0]:<8}{k[1]:<8}{k[2]:<7}{f'{nf}/{n}':<8}{f'≤{up*100:.0f}%':<10}"
          f"{sdx:<10}{smt:<11}{ns}{mark}")

print()
print("  ★ = 낙상 0 · 생존드리프트 <10cm · 낙상률 95%상한 <10%. 셋 다여야 채택 후보다.")
print("  ⚠'상한95%' 는 Clopper-Pearson. 0/7 은 상한 41% 라 0/7 끼리는 **구분이 안 된다** —")
print("    후보를 가르려면 SEEDS 를 늘려야 한다(0/30 이면 상한 12%, 0/60 이면 6%).")
if nerr:
    print(f"  ✗ 파싱실패(falls={ERR}) {nerr}건 — 해당 런의 출력이 비었다. 원본 확인 필요.")
PY
echo
echo "원본: $OUT"
echo "재집계만: AGG_ONLY=1 $0 $OUT"
