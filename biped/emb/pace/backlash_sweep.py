#!/usr/bin/env python3
"""backlash_sweep.py — 1차(q_ch)·2차(aux) 엔코더 동시기록·히스테리시스 분석.

  ⚠⚠ 한계 (2026-09-07 신뢰성 검증 — 이 도구로 '백래시 확정' 하지 말 것):
    ① **무게추 달고 돌리면 백래시 측정 불가.** 매단 추=단방향(중력)이라 기어 메시가 한 flank 에
       늘 물려 dead-zone 을 안 지난다(추 무거울수록 더 안 보임). 나오는 값은 백래시 아니라 한
       flank 의 마찰/탄성 히스테리시스. 진짜 백래시는 출력 **클램프 + ±토크 0-통과**
       (bench_actuator_full.py backlash 페이즈).
    ② **calf/foot 은 aux=벨트 앞단** → 감속단(7:1)만 보고 **벨트 유격 못 봄**. 무릎 실제 유격은
       벨트에 있고 **손실측 7°**(발끝 42mm, DEV_STATUS_20260903) — 이 도구엔 안 나온다.
    ③ q_ch·aux 둘 다 **float16 LSB 0.0625°** → 0.03~0.07°는 분해능 바닥(>0 확인 불가).
       ~3 LSB 인 thigh 0.18°(관절측 aux) 정도만 신뢰.
  ⇒ 용도: hip/thigh 를 **무부하 대칭 스윙**으로 감속단 비틀림 대략 보기. 백래시 확정용 아님.

  기어(감속단) 히스테리시스: 느린 처프로 한 축을 왕복시키며
   1차(모터측 q_ch)·2차(출력축 aux) 엔코더를 **틱마다 동시 기록**하고 히스테리시스 루프를 분석한다.

  왜: home 잔차 3~4° 중 감속단 비틀림 몫을 2점(float/home) 비교로는 ~1° 로만 어림했다.
      연속 왕복이면 방향전환점의 (aux−q_ch) 점프 = **백래시**, 하중(τ)에 비례하는 성분 = **컴플라이언스**,
      각도에 비례하는 성분 = **스케일차**, 상수 = **영점차** 로 깨끗이 분리된다 → 적정/교체 판단 근거.

  전제 (2026-09-04):
    · deploy 를 **AUX_MODE=1** 로 띄웠을 때처럼 이 스크립트도 AUX_MODE=1 환경에서 실행(bridge 가 init 시 env 를 읽음).
    · ⚠**매달린 채만**(0x5A 가 MD80 명령프레임 바꾸는지 RGA 미확인). 한 축씩, 진폭 ≤15°.
    · ⚠모터 명령 writer 는 하나 — **biped_deploy 를 먼저 종료**(있으면 이 스크립트가 거부한다).
    · hip/thigh: aux = 관절각(진짜 링크). calf/foot: aux = 벨트 **앞단** → 여기서 잡히는 건 감속단 몫이고
      벨트 슬립은 안 보인다(벨트는 별도 시험).

  사용 (Pi):
    cd /home/rpetubt/simulation/biped/emb/pace
    AUX_MODE=1 python3 backlash_sweep.py --ch 1                 # HL_thigh, 기본 f 0.05→0.5Hz · 60s · ±10°
    AUX_MODE=1 python3 backlash_sweep.py --ch 0 --amp 8 --T 45
    python3 backlash_sweep.py --selftest                        # 하드웨어 없이 분석로직만(합성 히스테리시스)
  출력: /tmp/backlash_<축>_<시각>.csv/.npz/.png + 요약 1줄.
"""
from __future__ import annotations

import argparse, os, sys, time, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 배포게인(채널 kp/kd) — hwio._raw_write_all 주석과 동일 근거(spec hold_kp = biped_emb.yaml joints)
GAIN_BY_NAME = {"hip": (100.0, 6.0), "thigh": (50.0, 4.0), "calf": (80.0, 3.5), "foot": (30.0, 2.0)}


# ══════════════════════════════════════════════════════════════════════════
#  분석 (하드웨어 무관 — selftest 가 이 함수를 검증한다)
# ══════════════════════════════════════════════════════════════════════════
def analyze(t, q_cmd, q, tau, aux, f0, f1, T, amp):
    """반환 dict: offset[deg], scale[%], backlash_plateau[deg], backlash_reversal[deg],
       compliance[deg/Nm], n_rev, moving_frac. 규약: diff = aux − q_ch."""
    t = np.asarray(t, float); q = np.asarray(q, float); aux = np.asarray(aux, float)
    q_cmd = np.asarray(q_cmd, float); tau = np.asarray(tau, float)
    diff = aux - q
    dq_cmd = np.gradient(q_cmd, t)
    f_inst = f0 + (f1 - f0) * np.clip(t, 0, T) / T
    env = amp * 2 * np.pi * np.maximum(f_inst, 1e-6)          # 순간 최대 |dq_cmd|
    moving = np.abs(dq_cmd) > 0.3 * env                         # 방향전환 근방(백래시 전이구간) 제외
    if moving.sum() < 20:
        raise RuntimeError("이동 구간 샘플 부족 — 진폭/주파수/시간 확인")
    # ① 영점차·스케일차: 이동구간에서 diff = a + b·q 최소제곱
    A = np.column_stack([np.ones(moving.sum()), q[moving]])
    (a, b), *_ = np.linalg.lstsq(A, diff[moving], rcond=None)
    r = diff - (a + b * q)                                      # 잔차 = 백래시 + 컴플라이언스 + 잡음
    # ② 백래시(평탄부법): +방향 이동 중 r 의 중앙값 − −방향 중앙값
    pos = moving & (dq_cmd > 0); neg = moving & (dq_cmd < 0)
    bl_plateau = float(np.median(r[pos]) - np.median(r[neg])) if pos.any() and neg.any() else float("nan")
    # ③ 백래시(전환점 점프법): dq_cmd 부호가 바뀌는 지점 전후 r 의 중앙값 차
    s = np.sign(dq_cmd); rev = np.where(np.diff(s) != 0)[0] + 1
    jumps = []
    dt = float(np.median(np.diff(t)))
    for i in rev:
        W = max(6, int(0.12 / (max(f_inst[i], 1e-3) * dt)))     # 주기의 ~12%
        lo, hi = i - W, i + W
        if lo < 0 or hi >= len(r): continue
        before = np.median(r[lo:i - W // 4]); after = np.median(r[i + W // 4:hi])
        jumps.append(after - before)
    jumps = np.array(jumps)
    bl_rev = float(np.median(np.abs(jumps))) if len(jumps) else float("nan")
    # ④ 컴플라이언스: 이동구간 r vs τ 기울기(deg/Nm). τ 는 보고 토크(명령측 에코 가능) → '겉보기'
    comp = float("nan")
    if np.std(tau[moving]) > 1e-3:
        # 백래시 성분을 방향별로 제거한 뒤 회귀
        rr = r.copy(); rr[pos] -= np.median(r[pos]); rr[neg] -= np.median(r[neg])
        comp = float(np.polyfit(tau[moving], rr[moving], 1)[0])
    tau_range = float(np.ptp(tau[moving])) if moving.any() else float("nan")   # 부하 수준(τ 진폭)
    resid_p2p = float(np.percentile(r, 95) - np.percentile(r, 5))               # 잔차 폭(백래시+컴플+잡음)
    return dict(offset=float(a), scale_pct=float(b * 100), backlash_plateau=abs(bl_plateau),
                backlash_reversal=bl_rev, compliance_deg_per_nm=comp, n_rev=int(len(jumps)),
                moving_frac=float(moving.mean()), tau_range=tau_range, resid_p2p=resid_p2p,
                r=r, moving=moving, pos=pos, neg=neg, rev=rev)


def verdict(bl):
    """가이드라인(사양 아님 — 최종 기준은 RGA/CubeMars 감속기 스펙): 출력축 백래시."""
    if not np.isfinite(bl): return "판정불가"
    if bl < 0.3: return "우수(<0.3°)"
    if bl < 1.0: return "보통(0.3~1.0°) — 감시"
    return "과대(>1.0°) — 조정/교체 검토"


def report(name, res, log=print):
    bl = np.nanmedian([res["backlash_plateau"], res["backlash_reversal"]])
    log(f"  [{name}] 영점차 a={res['offset']:+.2f}°  스케일차 b={res['scale_pct']:+.2f}%  "
        f"백래시 평탄부={res['backlash_plateau']:.2f}° / 전환점={res['backlash_reversal']:.2f}° (n_rev={res['n_rev']})  "
        f"겉보기컴플라이언스={res['compliance_deg_per_nm']:+.3f}°/Nm  → **백래시≈{bl:.2f}° : {verdict(bl)}**")
    log(f"           부하 τ범위={res['tau_range']:.2f} Nm · (aux−q_ch) 잔차폭={res['resid_p2p']:.2f}° "
        f"[무부하와 비교: τ범위 커졌는데 잔차폭·컴플라이언스 그대로면 감속단은 하중에도 견고]")
    return bl


def save_plot(path, name, t, q, aux, res):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (플롯 생략 — matplotlib 없음: {e})"); return None
    r, pos, neg, rev = res["r"], res["pos"], res["neg"], res["rev"]
    fig, ax = plt.subplots(2, 1, figsize=(9, 8))
    ax[0].plot(q[pos], (aux - q)[pos], ".", ms=2, label="moving +")
    ax[0].plot(q[neg], (aux - q)[neg], ".", ms=2, label="moving -")
    ax[0].set_xlabel("primary q_ch [deg]"); ax[0].set_ylabel("aux - q_ch [deg]")
    ax[0].set_title(f"{name}: hysteresis (width ~ backlash {res['backlash_plateau']:.2f} deg)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(t, r, lw=.8); ax[1].vlines(t[rev], r.min(), r.max(), colors="r", alpha=.25, lw=.6)
    ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("residual r [deg] (offset/scale removed)"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig); return path


# ══════════════════════════════════════════════════════════════════════════
#  하드웨어 실행
# ══════════════════════════════════════════════════════════════════════════
def run_hw(a):
    import yaml
    from bench_actuator_full import open_hw, bind_aux          # main 가드 있음 — import 안전
    if subprocess.run(["pgrep", "-f", "build/biped_deploy"], capture_output=True).returncode == 0:
        print("✗ biped_deploy 가 떠 있음 — 모터 writer 는 하나여야 한다. 먼저 종료:"
              "\n    pkill -f build/biped_deploy"); return 2
    if os.environ.get("AUX_MODE", "0") != "1":
        print("✗ AUX_MODE=1 이 아님 — 2차 엔코더가 안 온다. `AUX_MODE=1 python3 backlash_sweep.py ...`"); return 2
    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    name = next((j["name"] for j in spec["joints"] if int(j["ch"]) == a.ch), f"ch{a.ch}")
    kp, kd = a.kp, a.kd
    if kp is None or kd is None:
        g = next((v for k, v in GAIN_BY_NAME.items() if k in name.lower()), (50.0, 3.0))
        kp = kp if kp is not None else g[0]; kd = kd if kd is not None else g[1]
    amp = min(abs(a.amp), 15.0); f1 = min(a.f1, 1.0)
    print(f"[backlash] {name}(ch{a.ch})  kp={kp} kd={kd}  처프 {a.f0}→{f1}Hz · {a.T}s · ±{amp}°"
          f"\n  ⚠ 로봇이 **매달려** 있고 이 축이 자유롭게 움직일 수 있는지 확인. 3초 후 시작(Ctrl+C 중단)…")
    time.sleep(3)
    hw = open_hw(spec); read_aux = bind_aux(hw)
    if read_aux is None: print("✗ .so 에 bridge_aux 없음(구 빌드)"); return 2
    rows = []
    try:
        hw.arm(a.ch, kp, kd)
        if read_aux(a.ch) is None:
            print("✗ aux 값이 안 옴(bridge_aux≠1) — AUX_MODE 전달/펌웨어 확인"); hw.safe_hold(); return 2
        center = float(hw.read(a.ch)[0])
        def qcmd(t):   # 선형 처프: 위상 φ = 2π(f0 t + (f1−f0) t²/2T)
            return center + amp * np.sin(2 * np.pi * (a.f0 * t + (f1 - a.f0) * t * t / (2 * a.T)))
        t0 = time.monotonic(); k = 0
        while True:
            t = time.monotonic() - t0
            if t >= a.T: break
            s = hw.step(a.ch, float(qcmd(t)), kp, kd)
            ax = read_aux(a.ch) or (float("nan"), float("nan"))
            rows.append((t, s.q_cmd_deg, s.q_deg, s.dq_dps, s.tau, s.cur, ax[0], ax[1]))
            k += 1
            if k % max(1, int(1.0 / hw.dt)) == 0:
                print(f"    {t:5.1f}/{a.T:.0f}s  q={s.q_deg:7.2f} aux={ax[0]:7.2f} Δ={ax[0]-s.q_deg:+.2f} τ={s.tau:+.2f}", flush=True)
            nxt = t0 + k * hw.dt; slp = nxt - time.monotonic()
            if slp > 0: time.sleep(slp)
        hw.goto(a.ch, center, kp, kd, speed_dps=10.0)
    except KeyboardInterrupt:
        print("\n  중단 — 안전정지")
    finally:
        try: hw.safe_hold()
        except Exception: pass
    if len(rows) < 50: print("✗ 데이터 부족"); return 1
    R = np.array(rows, float)
    t, q_cmd, q, dq, tau, cur, aux, auxv = R.T
    ok = np.isfinite(aux)
    res = analyze(t[ok], q_cmd[ok], q[ok], tau[ok], aux[ok], a.f0, f1, a.T, amp)
    bl = report(name, res)
    ts = time.strftime("%Y%m%d_%H%M%S"); base = f"/tmp/backlash_{name}_{ts}"
    np.savetxt(base + ".csv", R, delimiter=",", header="t,q_cmd,q_ch,dq,tau,cur,aux,aux_vel", comments="")
    np.savez(base + ".npz", t=t, q_cmd=q_cmd, q=q, dq=dq, tau=tau, cur=cur, aux=aux, aux_vel=auxv,
             offset=res["offset"], scale_pct=res["scale_pct"], backlash=bl, kp=kp, kd=kd, amp=amp, f0=a.f0, f1=f1)
    p = save_plot(base + ".png", name, t[ok], q[ok], aux[ok], res)
    print(f"  → {base}.csv / .npz" + (f" / .png" if p else ""))
    return 0


# ══════════════════════════════════════════════════════════════════════════
#  selftest — 합성 히스테리시스로 analyze() 검증
# ══════════════════════════════════════════════════════════════════════════
def _selftest():
    dt, T, f0, f1, amp = 0.005, 60.0, 0.05, 0.5, 10.0
    a_true, b_true, B_true = -2.60, 0.02, 1.20            # 영점차 · 스케일차 · 백래시
    t = np.arange(0, T, dt)
    q_cmd = amp * np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * T)))
    q = np.roll(q_cmd, 2); q[:2] = q_cmd[:2]                # 2틱 추종지연
    dq_cmd = np.gradient(q_cmd, t)
    rng = np.random.default_rng(0)
    # 출력축은 이동방향 반대로 B/2 만큼 뒤처진다(사구간) + 영점차 + 스케일차 + 잡음
    aux = a_true + (1 + b_true) * q - (B_true / 2) * np.sign(dq_cmd) + rng.normal(0, 0.02, len(t))
    tau = 50.0 * np.deg2rad(q_cmd - q)                     # kp·오차(합성)
    res = analyze(t, q_cmd, q, tau, aux, f0, f1, T, amp)
    bl = report("SELFTEST", res)
    ok = (abs(res["offset"] - a_true) < 0.1 and abs(res["scale_pct"] - b_true * 100) < 0.3
          and abs(bl - B_true) / B_true < 0.15 and res["n_rev"] > 10)
    p = save_plot("/tmp/backlash_selftest.png", "SELFTEST", t, q, aux, res)
    print(f"  기대: a={a_true:+.2f} b={b_true*100:+.2f}% B={B_true:.2f}  → {'PASS' if ok else 'FAIL'}" + (f"  (플롯 {p})" if p else ""))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ch", type=int, help="채널 0~7 (HL hip,thigh,calf,foot = 0..3 · HR = 4..7)")
    ap.add_argument("--amp", type=float, default=10.0, help="진폭[deg] (≤15 클램프)")
    ap.add_argument("--f0", type=float, default=0.05, help="시작 주파수[Hz] — 백래시는 느린 전환에서 드러남")
    ap.add_argument("--f1", type=float, default=0.5, help="끝 주파수[Hz] (≤1.0 클램프, 폐루프 대역 아래)")
    ap.add_argument("--T", type=float, default=60.0, help="지속[s]")
    ap.add_argument("--kp", type=float, default=None); ap.add_argument("--kd", type=float, default=None)
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--selftest", action="store_true", help="하드웨어 없이 분석로직만 검증")
    a = ap.parse_args()
    if a.selftest: return _selftest()
    if a.ch is None: ap.error("--ch 필요 (또는 --selftest)")
    return run_hw(a)


if __name__ == "__main__":
    sys.exit(main())
