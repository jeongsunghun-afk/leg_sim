#!/usr/bin/env python3
"""couple_magnitude.py — calf→foot 커플링 계수의 **크기**를 경사계로 잰다.

★왜 이게 필요한가 (2026-08-14)
  `couple_check.py` 는 (A)/(B) 만 가른다. 실기 결과가 **(A)** 였다:
      HL  calf 115.9° → foot_ch 기울기 +0.0001
      HR  calf 148.9° → foot_ch 기울기 +0.0000
  엔코더가 모터축에 있어 커플링을 **아예 못 본다**(|c| < 2e-4). 기울기가 실제 크기와
  무관하게 0 이므로 **엔코더로는 원리적으로 크기를 못 잰다.**

  PACE 로도 안 됐다. coef 를 탐색변수로 넣어 봤더니:
      fit_v2 (coef 고정 1.000)  적합 0.4050 · 따로 뺀 구간 0.3967
      fit_v3 (coef 자유→1.064)  적합 0.4058 · 따로 뺀 구간 0.3983
  자유로 풀면 오히려 나빠진다. 데이터가 c=1 을 반박하지도, 확정하지도 않는다.
  ⇒ **엔코더 밖으로 나가야 한다.** 남은 건 발의 물리각을 외부 센서로 재는 것뿐이다.

★원리 — 보상계수 k 를 우리가 정하고, 발의 절대각을 읽는다.
  명령:      q_raw_foot = q_foot_target + k·q_calf        ← k 는 **우리가** 정한다
  기구:      q_foot     = q_raw_foot − c·q_calf           ← c 가 **재려는 값**
  직렬사슬:  발_절대각  = q_calf + q_foot                  (대퇴 기준. 대퇴는 잡아 둔다)
             = q_foot_target + q_calf·(1 + k − c)

      ⇒  Δ발_절대각 = Δq_calf·(1 + k − c)
      ⇒  **c = 1 + k − Δ발_절대각 / Δq_calf**

  k=0 (발목 채널각 고정):  Δtilt = Δq_calf·(1−c)   → c=1 이면 **0°**  ← 영점시험
  k=1 (평소 보상 유지):    Δtilt = Δq_calf·(2−c)   → c=1 이면 Δq_calf 만큼 같이 돈다

★★k=0 과 k=1 을 **둘 다** 하는 게 맞다 (사용자 제안, 2026-08-14).
  두 측정의 차는 c 가 소거되어 **정확히 Δq_calf** 다:
      (1+1−c)·Δq − (1+0−c)·Δq = Δq
  즉 두 번 재서 차가 스윙각과 맞으면 **경사계·스윙각·미끄러짐까지 계측계 전체가
  검증**된다. 그 눈금 위에서 k=0 값이 (1−c)·Δq 를 준다.
  한 번만 재면 그 눈금이 없다 — 어긋나도 c 탓인지 계측 탓인지 못 가른다.

⚠자기측정 함정과의 관계 — 2026-08-10 에 "부호 −1 이 맞다" 고 오판한 그 함정은
  **엔코더로 잴 때**의 문제다(발목 모터가 우리 보상을 추종하니 자기 명령을 되잰다).
  경사계는 외부 센서라 명령이 뭐든 진짜 물리각을 본다. 그래서 k=1 도 유효하다.

⚠⚠**발목 채널각으로 직접 명령한다** — 평소 명령경로(`q_joint_to_ch`)를 쓰지 않는다.
  그 경로는 config 의 `couple_coef` 를 자동으로 얹으므로 k 를 우리가 못 정한다.
  여기서는 채널공간에서 직접:
      Δq_ch_foot = k · (s_foot/s_calf) · Δq_ch_calf ,   s = sign·gear_k
  두 다리 다 s_foot/s_calf = 0.8 이다(±1.2 / ±1.5). 알려진 실측과 맞다 —
  "calf 관절 +30° → 채널각 HL calf−45/foot−36" 에서 36/45 = 0.8.

측정 절차:
  1) 제어기를 띄우고 이 스크립트를 실행한다 (발목이 잡히고, **무릎만 자유**가 된다)
  2) **발 링크**에 경사계를 대고 무릎을 한쪽 끝으로 옮겨 **손으로 잡은 채** 읽는다
     ⚠**발바닥 평면이 있다고 가정하지 말 것.** 접촉 구성이 용도별로 둘이다 —
       stand 는 2점(heel+toe), **보행은 1점(point foot)** 이다(2026-08-14 사용자 확인).
       발 링크의 **평평한 가공면** 아무 데나 쓰면 된다. 두 번 다 **같은 면에 같은
       방향으로** 대는 것만 지키면 절대 영점은 상관없다(차이만 쓰므로).
  3) 반대쪽 끝으로 천천히 옮겨 다시 읽는다  ← 이 두 자세가 스윙의 양 끝이어야 한다
  4) Ctrl-C → 두 값을 입력하면 c 가 나온다
  5) `--k 1.0` 으로 한 번 더 → 두 결과의 차가 Δq_calf 와 맞는지 확인

⚠무릎이 자유이므로 정강이·발이 **중력으로 처진다.** 손으로 받치고 천천히 움직일 것.
⚠재는 것은 **발 링크의 절대 자세**다(정강이 기준 관절각이 아니다). 둘을 헷갈리면
  부호와 크기가 다 달라진다 — Δ발_절대각 = Δq_calf·(1+k−c) 이고, 관절각 쪽은
  Δq_foot = (k−c)·Δq_calf 다.
⚠대퇴는 잡아 둔다(hold). 대퇴가 움직이면 절대각 관계가 깨진다 — 그것도 검사한다.

사용:
  python3 diag/couple_magnitude.py --leg HL              # k=0, 영점시험
  python3 diag/couple_magnitude.py --leg HL --k 1.0      # 눈금용
  python3 diag/couple_magnitude.py --selftest            # 하드웨어 없이 수식만 검증
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml

from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EMB, "pace"))
sys.path.insert(0, os.path.join(EMB, "interface"))

LEGS = {"HL": ("HL_calf", "HL_foot"), "HR": ("HR_calf", "HR_foot")}


# ── 수식 (하드웨어와 무관 — selftest 가 이걸 검증한다) ──────────────────────
def coef_from_tilt(d_tilt_deg: float, d_qcalf_deg: float, k: float) -> float:
    """c = 1 + k − Δtilt/Δq_calf.  (독스트링 상단 유도 참조)"""
    if abs(d_qcalf_deg) < 1e-9:
        raise ValueError("Δq_calf 가 0 이다 — 무릎을 움직이지 않았다")
    return 1.0 + k - d_tilt_deg / d_qcalf_deg


def tilt_from_coef(c: float, d_qcalf_deg: float, k: float) -> float:
    """역방향 — 예측 Δtilt. 표를 찍고 selftest 로 왕복시키는 데 쓴다."""
    return d_qcalf_deg * (1.0 + k - c)


def _selftest() -> int:
    print("■ 수식 왕복 검증 — coef_from_tilt ∘ tilt_from_coef = 항등")
    bad = 0
    for k in (0.0, 0.8, 1.0):
        for c in (0.70, 0.90, 1.00, 1.06, 1.20):
            for dq in (-150.0, -60.0, 45.0, 100.0):
                t = tilt_from_coef(c, dq, k)
                c2 = coef_from_tilt(t, dq, k)
                ok = abs(c2 - c) < 1e-12
                bad += not ok
                if not ok:
                    print(f"  ✗ k={k} c={c} dq={dq}: Δtilt={t:.4f} → c={c2:.12f}")
    print(f"  {'✓ 전부 통과' if not bad else f'✗ {bad}건 실패'} "
          f"(k 3 × c 5 × Δq 4 = 60 조합)")

    # ★핵심 성질 — k=1 과 k=0 의 차는 c 와 **무관하게** Δq_calf 다. 이게 눈금이다.
    print("\n■ 눈금 성질 — (k=1 측정) − (k=0 측정) = Δq_calf, c 와 무관")
    for c in (0.70, 1.00, 1.30):
        dq = 100.0
        d = tilt_from_coef(c, dq, 1.0) - tilt_from_coef(c, dq, 0.0)
        ok = abs(d - dq) < 1e-9
        bad += not ok
        print(f"  {'✓' if ok else '✗'} c={c:.2f}: 차 {d:+.4f}° (기대 {dq:+.1f}°)")

    print("\n■ 예측표 — 100° 스윙에서 읽힐 값")
    print(f"  {'c':>6}{'k=0 Δtilt':>12}{'k=1 Δtilt':>12}")
    for c in (0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10):
        print(f"  {c:>6.2f}{tilt_from_coef(c, 100.0, 0.0):>12.1f}"
              f"{tilt_from_coef(c, 100.0, 1.0):>12.1f}")
    print("  ⇒ k=0 은 c=1 에서 0° 다(영점시험). 경사계 분해능 0.5° 면 Δc 0.005 를 본다.")
    return 1 if bad else 0


# ── 실기 ────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=sorted(LEGS), default="HL")
    ap.add_argument("--k", type=float, default=0.0,
                    help="보상계수. 0=발목 채널각 고정(영점시험) · 1=평소 보상(눈금)")
    ap.add_argument("--kp", type=float, default=None, help="발목 홀드 kp (기본 spec)")
    ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--min-span", type=float, default=40.0,
                    help="이보다 작은 무릎 스윙은 감도가 안 나온다 [관절°]")
    ap.add_argument("--out-dir", default=None,
                    help="추적 저장 위치 (기본 emb/pace/results). **시험은 임시 경로를 줄 것** — "
                         "기본값에 쓰면 실기 측정 결과를 덮어쓴다")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()

    import hwio
    import actuator_test as at
    from joint_map import JointMap

    spec = yaml.safe_load(open(os.path.join(EMB, "pace", "spec.yaml"), encoding="utf-8"))
    cfg = yaml.safe_load(open(os.path.join(EMB, "config", "biped_emb.yaml"), encoding="utf-8"))
    jm = JointMap(cfg)
    n_ch = int(spec["shm"]["n_channel"])

    calf_n, foot_n = LEGS[a.leg]
    ci, fi = jm.names.index(calf_n), jm.names.index(foot_n)
    js = cfg["joints"]

    def s_of(i):
        return float(js[i].get("sign", 1)) * float(js[i].get("gear_k", 1.0))

    s_calf, s_foot = s_of(ci), s_of(fi)
    ratio = s_foot / s_calf                    # Δq_ch_foot = k·ratio·Δq_ch_calf

    # ★config 가 실제로 쓰는 커플링 계수를 찍는다 — k=1 이 "평소 보상" 과 같은지 확인용
    cfg_coef = next((float(j.get("couple_coef", 1.0)) for j in js
                     if j["name"] == foot_n and j.get("couple_from")), None)

    print("=" * 72)
    print(f"■ calf→foot 커플링 **크기** 측정 — {a.leg} · k = {a.k:g}")
    print("=" * 72)
    print(f"  채널        calf ch{ci} (s={s_calf:+.2f})  ·  foot ch{fi} (s={s_foot:+.2f})")
    print(f"  s_foot/s_calf = {ratio:+.4f}   ⇒ Δq_ch_foot = {a.k:g}·{ratio:+.4f}·Δq_ch_calf")
    print(f"  config 의 couple_coef = {cfg_coef}  "
          f"({'k=1 이 평소 보상과 같다' if cfg_coef == 1.0 else '⚠k=1 과 다르다'})")
    if a.k == 0.0:
        print("  ⇒ **영점시험** — c=1 이면 발바닥 절대각이 안 변해야 한다")
    else:
        print(f"  ⇒ 발목이 무릎을 따라간다. c=1 이면 Δtilt = {a.k:g}+1−1 = "
              f"{a.k:g}×Δq_calf 만큼 돈다")

    sf, g = spec["safety"], spec["gains"]
    box = at._ch_limit_box(spec, pin_home=True)
    lo_f, hi_f = box[fi]

    # 홀드: 시험 다리의 calf 만 자유. 나머지 전부(반대다리 · hip · thigh)를 잡는다.
    #   ★thigh 를 반드시 잡아야 한다 — 절대각 관계가 "대퇴 고정" 을 전제한다.
    # ★관절이 정의된 채널만 잡는다. spec.shm.n_channel 은 **10** 인데 관절은 8 이다
    #   (ch8·9 는 미사용). range(n_ch) 로 잡으면 _ch_limit_box 에 없어 KeyError 다.
    #   미사용 채널은 kp=kd=0 으로 두는 게 맞다 — 잡을 것이 없다.
    #   ⚠box 를 쓰므로 **box 정의 뒤**에 와야 한다(2026-08-14 NameError 로 한 번 터졌다).
    hold = tuple(c for c in range(min(n_ch, len(jm.names)))
                 if c not in (ci, fi) and c in box)
    lim = hwio.Limits(q_min=lo_f, q_max=hi_f,
                      tau_trip=float(sf["tau_trip_nm"]), tau_trip_ms=float(sf["tau_trip_ms"]),
                      vel_trip=float(sf["vel_trip_dps"]), err_max=float(sf["err_max_deg"]),
                      stale_ms=float(sf["stale_ms"]),
                      kp_max=float(g["kp_max"]), kd_max=float(g["kd_max"]))
    # ★spec 의 hold_kp/kd 는 **축별 dict** 다(스칼라도 허용). 시험축(발목) 것을 꺼낸다.
    #   종전엔 float() 로 통째로 감싸 TypeError 였다 — 축마다 I·gear_k 가 달라
    #   스칼라 하나로는 못 맞춘다는 게 spec 주석의 결론인데 그걸 못 읽은 것이다.
    def _gain(key, dflt):
        v = sf.get(key, dflt)
        if isinstance(v, dict):
            return float(v.get(fi, v.get(str(fi), dflt)))
        return float(v)

    kp = a.kp if a.kp is not None else _gain("hold_kp", 40.0)
    kd = a.kd if a.kd is not None else _gain("hold_kd", 2.0)
    print(f"  발목 홀드게인 kp {kp:g} · kd {kd:g}")

    trace = []
    clipped = [False]
    aborted = [False]
    with hwio.Hardware(spec["shm"]["lib"], n_ch, float(spec["shm"]["rate_hz"]), lim,
                       int(spec["shm"]["recv_wait_ms"]), 0.3,
                       hold_channels=hold,
                       hold_kp=sf.get("hold_kp", 40.0),
                       hold_kd=sf.get("hold_kd", 2.0)) as hw:
        # ★홀드축마다 **자기 상자**를 등록한다. self.lim 은 시험축(발목) 상자이므로
        #   그대로 두면 홀드축 판정이 엉뚱한 범위로 이뤄진다 (homing.py:236 과 같은 이유).
        #   _ch_limit_box 는 (lo, hi) 튜플을 준다 — Limits 가 아니다.
        for c in hold:
            _lo, _hi = box[c]
            hw.lim_ch[c] = replace(lim, q_min=float(_lo), q_max=float(_hi))
        hw.lim_ch[fi] = replace(lim, q_min=lo_f, q_max=hi_f)
        print(f"  홀드 ch {list(hold)} · 자유 ch [{ci}] · 시험축 ch {fi}"
              f"   (전체 {n_ch}채널 중 관절 {len(jm.names)}개)")
        q_foot0 = hw.arm(fi, kp, kd)
        hw.read(fi)
        q_calf0 = float(hw._q[ci])
        thigh_ch = [c for c in hold if "thigh" in jm.names[c]]
        thigh0 = {c: float(hw._q[c]) for c in thigh_ch}

        # ★스윙 한계를 **미리** 알려 준다. k>0 이면 발목 명령이 상자에 먼저 부딪힌다.
        if a.k != 0.0:
            room = min(q_foot0 - lo_f, hi_f - q_foot0)
            max_ch = room / abs(a.k * ratio)
            print(f"\n  ⚠k={a.k:g} 이라 발목 명령이 움직인다. 상자 [{lo_f:.1f},{hi_f:.1f}] ·"
                  f" 현재 {q_foot0:.1f} → 편측여유 {room:.1f}°")
            print(f"    ⇒ 무릎 스윙 한계 **채널 {max_ch:.1f}° = 관절 {max_ch/abs(s_calf):.1f}°**"
                  f" (한쪽 방향). 넘으면 잘리고 그 구간은 무효다")
        print(f"\n  발목 래치 {q_foot0:+.2f}° · 무릎 시작 {q_calf0:+.2f}° (채널각)")
        print("  ── 무릎만 손으로 천천히 왕복시킬 것. 발목·발은 잡지 말 것 ──")
        print("  ── 양 끝에서 경사계를 읽고, 끝나면 Ctrl-C ──\n")

        t0 = time.monotonic()
        try:
            while True:
                q_ch_calf = float(hw._q[ci])
                dq_ch = q_ch_calf - q_calf0
                q_cmd = q_foot0 + a.k * ratio * dq_ch
                if q_cmd < lo_f - 1e-9 or q_cmd > hi_f + 1e-9:
                    if not clipped[0]:
                        print(f"\n  ⚠⚠명령 {q_cmd:+.1f}° 가 상자 [{lo_f:.1f},{hi_f:.1f}] 를"
                              f" **넘었다 — 여기부터 무효**다. 되돌리거나 다시 시작할 것.")
                    clipped[0] = True
                s = hw.step(fi, q_cmd, kp, kd)
                trace.append((time.monotonic() - t0, q_ch_calf, s.q_deg, q_cmd, s.tau))
                if len(trace) % 200 == 0:
                    print(f"\r  무릎 ch {q_ch_calf:+7.2f}° (관절 {q_ch_calf/s_calf:+7.2f}°)"
                          f" · 스윙 {(max(x[1] for x in trace)-min(x[1] for x in trace)):6.2f}°"
                          f" · 발목 {s.q_deg:+7.2f}° τ {s.tau:+5.2f}  ", end="", flush=True)
        except KeyboardInterrupt:
            print("\n  ── 정지 ──")
        except hwio.SafetyAbort as e:
            # ★트립해도 **추적을 살린다** (2026-08-14). 종전엔 예외가 with 밖으로
            #   나가 그때까지 모은 자료가 통째로 날아갔다. 손으로 무릎을 미는 시험이라
            #   대퇴 홀드가 밀려 트립하는 건 드문 일이 아니고, 그때 "왜 트립했나" 를
            #   보려면 바로 그 추적이 필요하다.
            print(f"\n  ⚠⚠트립: {e}\n     추적은 저장한다. 아래 경고를 먼저 볼 것.")
            aborted[0] = True
        thigh_drift = {c: float(hw._q[c]) - thigh0[c] for c in thigh_ch}

    if not trace:
        print("✗ 표본이 하나도 없다 — arm 직후에 끝났다"); return 1
    T = np.array(trace, float)
    qc = T[:, 1]
    i_lo, i_hi = int(np.argmin(qc)), int(np.argmax(qc))
    span_ch = float(qc[i_hi] - qc[i_lo])
    span_j = span_ch / s_calf                       # 관절각 스윙(부호 포함)

    # ★저장 위치를 인자로 뺀다 (2026-08-14). 오프라인 시험이 같은 경로에 쓰는 바람에
    #   **실기 측정 추적을 덮어썼다.** 파일명이 (다리, k) 로만 정해져 충돌한다.
    #   시험은 --out-dir 로 임시 경로를 준다.
    out = os.path.join(a.out_dir or os.path.join(EMB, "pace", "results"),
                       f"couple_mag_{a.leg}_k{a.k:g}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, t=T[:, 0], q_ch_calf=qc, q_ch_foot=T[:, 2], q_cmd_foot=T[:, 3],
             tau_foot=T[:, 4], k=a.k, ratio=ratio, s_calf=s_calf, s_foot=s_foot,
             leg=a.leg, span_ch=span_ch, span_joint=span_j, clipped=clipped[0])

    if len(T) < 50:
        # ★저장 뒤에 판정한다 (2026-08-14). 종전엔 여기서 바로 return 이라
        #   **왜 일찍 끝났는지 볼 자료가 사라졌다.** 조기 트립은 원인 규명이 전부다.
        print(f"\n  ✗ 표본이 {len(T)}개뿐이다(≥50 필요) — 시작 직후 끝났다.")
        print(f"     추적은 저장했다: {out}")
        if aborted[0]:
            print("     안전트립이 원인이다. 위 트립 메시지를 볼 것.")
        if clipped[0]:
            print(f"     발목 명령이 상자 [{lo_f:.1f},{hi_f:.1f}] 를 넘었다 —"
                  f" k={a.k:g} 이면 무릎 스윙이 발목 명령을 끌고 간다. --k 0 은 안 그렇다.")
        return 1

    print(f"\n■ 무릎 스윙 — 채널 {span_ch:+.2f}° = **관절 {span_j:+.2f}°**")
    print(f"  (양 끝: ch {qc[i_lo]:+.2f}° @ t={T[i_lo,0]:.1f}s  ↔  "
          f"{qc[i_hi]:+.2f}° @ t={T[i_hi,0]:.1f}s)")
    print(f"  ✓ 저장: {out}")

    bad = False
    # ★스윙이 0 이면 여기서 멈춘다 (2026-08-14). 종전엔 아래 0.5/|span| 에서
    #   ZeroDivisionError 로 터졌다 — "무릎을 안 움직였다" 는 **사용자 실수**인데
    #   추적(traceback)이 뜨면 코드 고장으로 읽힌다. 원인을 그대로 말해 준다.
    if abs(span_j) < 1e-6:
        print("\n  ✗ 무릎이 움직이지 않았다 — 스윙 0°. 측정이 성립하지 않는다.")
        print("     ch{} 가 자유인지 확인하고, 무릎을 손으로 양 끝까지 옮긴 뒤"
              " Ctrl-C 할 것.".format(ci))
        return 1
    if abs(span_j) < a.min_span:
        print(f"  ⚠스윙이 {abs(span_j):.1f}° 로 작다(권장 ≥{a.min_span:.0f}°). "
              f"경사계 0.5° 오차가 Δc {0.5/abs(span_j):.3f} 로 번진다")
    ferr = np.abs(T[:, 2] - T[:, 3])          # |실측 발목채널 − 명령|
    print(f"\n■ 발목 추종오차 — 평균 {ferr.mean():.3f}° · 95% "
          f"{np.percentile(ferr,95):.3f}° · 최대 {ferr.max():.3f}°")
    # ★이게 크면 발목이 안 잡힌 것이고, 그만큼 Δtilt 가 줄어 c 가 1 쪽으로 편향된다.
    #   경사계 분해능(0.5°)과 같은 자릿수면 측정의 의미가 없다. 관절각으로 환산해 본다.
    ferr_j = ferr.max() / abs(s_foot)
    if ferr_j > 0.5:
        print(f"  ⚠최대오차가 관절각 {ferr_j:.2f}° 다 — 경사계 분해능(0.5°)보다 크다.")
        print(f"     발목이 밀렸다는 뜻이고 Δtilt 가 그만큼 **줄어든다**(c 가 1 쪽으로 편향).")
        print(f"     --kp 를 올려 다시 잴 것(지금 {kp:g}).")
        bad = True
    for c, d in thigh_drift.items():
        if abs(d) > 1.0:
            print(f"  ⚠⚠{jm.names[c]} 가 {d:+.2f}° 밀렸다 — **절대각 관계의 전제가 깨졌다**")
            bad = True
    if clipped[0]:
        print("  ⚠⚠발목 명령이 상자에 잘렸다 — 이 측정은 **무효**다")
        bad = True
    if aborted[0]:
        print("  ⚠⚠안전트립으로 끝났다 — 끝 자세를 실제로 유지했는지 확인할 것")
        bad = True

    print(f"\n■ 예측 — 이 스윙({span_j:+.1f}°)에서 c 별 Δ발바닥_절대각")
    print(f"  {'c':>6}{'Δtilt 기대':>12}")
    for c in (0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10):
        print(f"  {c:>6.2f}{tilt_from_coef(c, span_j, a.k):>12.1f}")

    print("\n■ 경사계 값 입력 (발바닥 절대각, 도. 빈 줄이면 건너뜀)")
    try:
        ta = input(f"   무릎 ch {qc[i_lo]:+.2f}° 자세에서 : ").strip()
        tb = input(f"   무릎 ch {qc[i_hi]:+.2f}° 자세에서 : ").strip()
    except (EOFError, KeyboardInterrupt):
        ta = tb = ""
    if not ta or not tb:
        print("  건너뜀 — npz 는 저장됐다. 나중에 값으로 c 를 계산할 수 있다:")
        print(f"    c = 1 + {a.k:g} − Δtilt / ({span_j:+.2f})")
        return 0

    d_tilt = float(tb) - float(ta)
    c_est = coef_from_tilt(d_tilt, span_j, a.k)
    err = 0.5 / abs(span_j) * 2                  # 경사계 ±0.5° 두 번 → Δc 불확도
    print(f"\n  Δtilt = {d_tilt:+.2f}°  ·  Δq_calf = {span_j:+.2f}°  ·  k = {a.k:g}")
    print(f"  ⇒ **c = {c_est:.4f} ± {err:.4f}**  (경사계 ±0.5° 가정)")
    if bad:
        print("  ⚠위 경고 때문에 이 값은 못 믿는다. 고치고 다시 잴 것.")
        return 1
    if abs(c_est - 1.0) <= err:
        print(f"  ⇒ **c = 1 과 구별되지 않는다.** 현 가정(coef=1)을 유지하면 된다.")
    else:
        print(f"  ⇒ **c ≠ 1 이다** ({(c_est-1)*100:+.1f}%). config 의 couple_coef 와"
              f" MJCF tendon 을 함께 고칠 것 — 한쪽만 고치면 앞뒤가 안 맞는다.")
    other = 1.0 if a.k == 0.0 else 0.0
    print(f"\n  ★다음: `--k {other:g}` 로 한 번 더 재서 **눈금**을 확인할 것.")
    print(f"    두 측정의 Δtilt 차이는 c 와 무관하게 정확히 Δq_calf({span_j:+.1f}°) 여야 한다.")
    print("    안 맞으면 c 가 아니라 계측계(경사계·미끄러짐·대퇴 이동)가 틀린 것이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
