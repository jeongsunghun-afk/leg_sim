#!/usr/bin/env bash
# run_deploy_hw.sh — 실기 배포기(biped_deploy)를 **식별된 파라미터로** 띄운다.
#
# ★왜 이 파일인가 (2026-08-24): 무중력 브래킷 측정으로 축별 중력 부족분이 확정됐는데,
#   그 값들이 env-var 라 매번 손으로 치면 반드시 빠뜨리거나 옛값을 쓴다.
#   측정의 출처와 값을 **한 곳에** 박아 둔다. 값을 바꾸려면 여기를 고칠 것.
#
# ── 측정 요약 (2026-08-24 · 점발 Qhome8 자세 · 브래킷 판독 = 마찰 소거) ──
#     축          g_lo    g_hi     g*     밴드   τ_c/α    비고
#     HL_hip      1.10    1.30    1.20    0.20            (평발자세 측정. hip 중력은 자세무관)
#     HR_hip      1.00    1.35    1.18    0.35    0.92
#     HL_thigh    1.10    1.90   ★1.50    0.80    0.68    ⚠아래 경고
#     HR_thigh    0.70    1.50    1.10    0.80    0.66    (PACE 식별 0.604 와 일치 = 판독 검증)
#     calf/foot   측정 불가(중력 ≪ 마찰). 1.00 유지.
#
# ★★핵심 발견 — **밴드(마찰폭)는 좌우 동일한데 g* 만 1.36배 밀려 있다.**
#     밴드 = 2τc/(α·G_CAD) 가 같다  ⇒  α 도 마찰도 좌우 같다
#     g*   = G_real/(α·G_CAD) 만 다르다  ⇒  **왼다리 실제 중력토크가 36% 크다**
#   질량이 36% 다를 수 없으니 기하다: 왼무릎(calf) 실제각이 보고각과 25~30° 어긋나야
#   나오는 크기고, 육안 관찰("초기자세 좌우 다름")·HL_calf 영점 4.25° 이동·커플링 잔차
#   3.11° 미설명·2026-08-11 풀리 재조임 이력이 전부 같은 곳을 가리킨다.
#   ⇒ **왼쪽 무릎 벨트/풀리 미끄러짐.**
#
# ⚠⚠HL_thigh 1.50 은 그 미끄러짐을 **흡수한 값**이다. float(무중력) 실험엔 쓸 수 있지만,
#   벨트가 또 미끄러지면 그 순간 틀어진다. **walk 는 왼무릎을 수리하기 전엔 금지.**
#   수리 후: 지그 영점 재캘리브레이션 → gen_grav_table --apply → g* 재측정(1.10 근처로
#   돌아오면 수리 성공이 정량 확인된다) → 이 파일 갱신.
#
# 사용:  ./run_deploy_hw.sh            # 1점 점발(기본. walk 용)
#        ./run_deploy_hw.sh flat       # 2점 평발(stand 용. ⚠foot gear_k 1.2/1.6 미해결)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

MJCF="$HERE/biped_from_quad.mjcf"                  # 1점 점발
[ "${1:-}" = "flat" ] && MJCF="$HERE/biped_flatfoot.mjcf"

# ── ①float(무중력) 축별 중력배율 — 측정된 중립점 g* 그대로 ──────────────────
#   이 값이면 무중력에서 전 축이 중립이다(뜨지도 지지도 않음). GUI 배율은 이 위에
#   공통 계수로 곱해진다(×1.00 이 이 값 그대로라는 뜻).
export GRAV_SCALE_JOINT="1.20,1.50,1.00,1.00,1.18,1.10,1.00,1.00"

# ── ②stand/walk 토크보정 — 같은 부족분을 WBIC 토크에 건다 ──────────────────
#   1/g* = α·(G_CAD/G_real) 이므로 τ 에 g* 를 곱하면 실제 출력이 모델 의도값이 된다.
#   ⚠HL_thigh 1.50 경고는 위와 동일 — 벨트 수리 전 walk 금지.
#   ⚠calf/foot 은 측정 불가라 1.00 = 무보정. thigh 값으로 외삽하지 **않는다**
#     (축퇴 원인이 기하로 밝혀진 이상, 안 잰 축에 배율을 지어내면 안 된다).
export STAND_TAU_SCALE_JOINT="1.20,1.50,1.00,1.00,1.18,1.10,1.00,1.00"

echo "[run_deploy_hw] MJCF=$MJCF"
echo "[run_deploy_hw] GRAV_SCALE_JOINT=$GRAV_SCALE_JOINT"
echo "[run_deploy_hw] STAND_TAU_SCALE_JOINT=$STAND_TAU_SCALE_JOINT"
echo "[run_deploy_hw] ⚠walk 는 왼무릎 벨트 수리 전 금지 — 헤더 주석 참조"
cd "$HERE/cpp"
exec ./build/biped_deploy --mjcf "$MJCF" --start-mode off
