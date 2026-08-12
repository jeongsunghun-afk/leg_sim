#!/usr/bin/env python3
"""design_excitation.py — 가진 궤적의 **식별가능성**을 설계 단계에서 잰다 (하드웨어 미접촉).

★왜 필요한가 — 지금 설계검사(collect_multichirp --dry)는 **축간 상관**을 본다. 그건 대리지표다.
  PACE 의 목적함수는
        cost(θ) = Σ_t Σ_i ( q_sim,i(t;θ) − q_real,i(t) )²
  이고 θ 근처에서 선형화하면
        Δcost ≈ ‖S·Δθ‖²,      S = ∂q_sim/∂θ        (감도행렬)
  ⇒ 좋은 가진의 조건은 둘이고, **둘 다 파라미터(θ) 기준이지 축 기준이 아니다**:
      (1) 각 파라미터의 ‖S_p‖ 가 **측정 노이즈보다 클 것**  — 아니면 애초에 식별 불가
      (2) S 의 열들이 **서로 평행하지 않을 것**             — 평행하면 두 파라미터가 맞바꿔진다
  축간 상관을 낮추는 건 (2)에 도움이 되지만 (2) 자체가 아니다.
  ★반례가 우리 구성에 실제로 있다: `ROTOR_I` 는 **8축이 공유**하는 하나의 값이라,
    축 궤적을 아무리 비상관화해도 `JDAMP` 와 평행해질 수 있다. 축 상관은 그걸 못 본다.

★여태 확인된 것은 `ROTOR_I` **하나뿐**이다
  2026-08-11 손계산: `ROTOR_I` +10% → 궤적변화 0.058°, 잡음 0.02° 대비 **SNR 3**
  (a348781 커밋 메시지 · NEXT_HW.md:745). 그 계산은 **코드에 없고** 나머지 8개
  (JDAMP×4 · JFRIC×4)는 감도를 재 본 적이 없다. 이 도구가 그 자리를 메운다.

★측정 방식은 CMA-ES 가 실제로 하는 것과 **같아야 한다**
  pace_cmaes 는 `--window`(0.5s)마다 실측 상태로 재초기화한다. 그래서 여기서도 섭동
  롤아웃을 **공칭 궤적 상태로 창마다 재초기화**한다. 30초를 통짜 개루프로 적분한 감도를
  재면 적분 발산분이 섞여 실제 식별력보다 **크게** 나온다(낙관 편향).

★한계 — 이건 **시뮬 안에서의 감도**다
  모델이 틀린 부분(미모델 마찰·백래시·지연)은 여기 안 잡힌다. "SNR 이 충분하다"는
  "이 모델이 맞다면 식별 가능하다"는 뜻이지 "실측이 잘 나온다"가 아니다.

사용:
    ~/.venv-mujoco/bin/python design_excitation.py                # 현재 spec 평가
    ~/.venv-mujoco/bin/python design_excitation.py --T 10         # 짧게(빠른 확인)
    ~/.venv-mujoco/bin/python design_excitation.py --compare 자유형식.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.dirname(HERE)
BIPED = os.path.dirname(EMB)
sys.path[:0] = [HERE, os.path.join(HERE, "tests"), os.path.join(EMB, "interface")]

import pace_cmaes as P                                    # noqa: E402
from joint_map import JointMap                            # noqa: E402

DEG = np.pi / 180.0


def build_traj(mc, jm, cfg_all, T, rate):
    """collect_multichirp 과 **같은 함수**를 쓴다 — 설계와 수집이 갈리면 분석이 무의미해진다."""
    import collect_multichirp as cm
    n = jm.n_leg
    amps, f0, k, phi = cm.chirp_bank(mc, n, T)
    ramp = float(mc.get("ramp_s", 2.0))
    home = np.array([float(x) for x in cfg_all["home"]["q_deg"]])[:n]
    tt = np.arange(0.0, T, 1.0 / rate)
    q_cmd = np.array([home + cm.q_at(t, amps, f0, k, phi, ramp) for t in tt])
    return tt, q_cmd, home, dict(amp=amps, f0=f0, f1=f0 + k * T, phi=phi)


def rollout_free(m, d, idx, q_cmd, kp, kd, dt, q0):
    """공칭 롤아웃 — 재초기화 없이 끝까지. q·dq 를 함께 돌려준다(섭동 롤아웃의 기준상태)."""
    import mujoco
    N, n = q_cmd.shape
    q = np.empty((N, n)); dq = np.empty((N, n))
    m.opt.timestep = dt
    d.qpos[:] = 0; d.qvel[:] = 0
    for i, (_, qa, dofa, _) in enumerate(idx):
        d.qpos[qa] = q0[i] * DEG
    mujoco.mj_forward(m, d)
    for t in range(N):
        for i, (_, qa, dofa, aid) in enumerate(idx):
            q[t, i] = d.qpos[qa] / DEG
            dq[t, i] = d.qvel[dofa] / DEG
            d.ctrl[aid] = kp[i] * (q_cmd[t, i] - q[t, i]) * DEG - kd[i] * d.qvel[dofa]
        mujoco.mj_step(m, d)
    return q, dq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--config", default=os.path.join(EMB, "config", "biped_emb.yaml"))
    ap.add_argument("--mjcf", default=os.path.join(BIPED, "biped_from_quad.mjcf"))
    ap.add_argument("--T", type=float, default=None, help="길이[s] (기본 spec)")
    ap.add_argument("--window", type=float, default=0.5, help="재초기화 창[s] — pace_cmaes 와 맞출 것")
    ap.add_argument("--pert", type=float, default=0.10, help="섭동 크기(상대). 감도는 이 값 기준으로 보고")
    ap.add_argument("--noise", type=float, default=0.02, help="측정 잡음[°] — SNR 기준")
    ap.add_argument("--dt", type=float, default=None, help="적분 timestep[s] (기본 1/rate)")
    a = ap.parse_args()

    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    cfg_all = yaml.safe_load(open(a.config, encoding="utf-8"))
    jm = JointMap(cfg_all)
    mc = spec["pace_multi"]
    T = float(a.T or mc.get("duration_s", 30.0))
    rate = float(spec["shm"]["rate_hz"])
    dt = float(a.dt or 1.0 / rate)

    tt, q_cmd, home, des = build_traj(mc, jm, cfg_all, T, rate)
    names = list(jm.names)

    # ★게인은 **관절공간**으로. collect_multichirp 이 npz 에 저장하는 것과 같은 환산이다
    #   (kp_joint = kp_ch·k²). 여기서 갈리면 감도가 통째로 틀린다.
    import actuator_test as at
    kp_ch = at._gain(mc["kp"]); kd_ch = at._gain(mc["kd"])
    kp = np.array([kp_ch[c] * jm.k[i] ** 2 for i, c in enumerate(jm.ch)])
    kd = np.array([kd_ch[c] * jm.k[i] ** 2 for i, c in enumerate(jm.ch)])
    gear_n = np.array([float([x for x in spec["joints"] if x["ch"] == c][0]["gear"])
                       for c in jm.ch])

    m = P.load_fixed_base(a.mjcf)
    import mujoco
    d = mujoco.MjData(m)
    idx = P.joint_index(m, names)
    x0, lo, hi = P.init_bounds(a.spec, names, False)
    plabels = ["ROTOR_I"] + [f"JDAMP.{k}" for k in P.KINDS] + [f"JFRIC.{k}" for k in P.KINDS]

    print("■ 가진 궤적 식별가능성 (하드웨어 미접촉)")
    print(f"  {os.path.basename(a.mjcf)} · {T:.0f}s · {rate:.0f}Hz · dt {dt*1e3:.1f}ms "
          f"· 창 {a.window:.2f}s · 섭동 ±{a.pert*100:.0f}% · 잡음 {a.noise:.3f}°")
    print(f"  진폭[°] {np.round(des['amp'],1)}")
    print(f"  f_end[Hz] {np.round(des['f1'],2)}")

    win = max(1, int(round(a.window / dt)))
    P.apply_params(m, idx, gear_n, x0, False, names)
    q_nom, dq_nom = rollout_free(m, d, idx, q_cmd, kp, kd, dt, home)

    # ── 감도행렬 S: 열 하나가 파라미터 하나 ────────────────────────────────
    #   중앙차분. 창 재초기화를 켠 채로 재야 CMA-ES 가 보는 것과 같다.
    S = np.empty((q_nom.size, len(x0)))
    for p in range(len(x0)):
        col = []
        for sgn in (+1, -1):
            xp = x0.copy(); xp[p] = x0[p] * (1.0 + sgn * a.pert)
            P.apply_params(m, idx, gear_n, xp, False, names)
            col.append(P.rollout(m, d, idx, q_nom, dq_nom, q_cmd, kp, kd, dt, win))
        S[:, p] = ((col[0] - col[1]) * 0.5).ravel()        # +pert 1단위당 궤적변화[°]
    P.apply_params(m, idx, gear_n, x0, False, names)       # 원복

    rms = np.sqrt(np.mean(S ** 2, axis=0))
    snr = rms / a.noise
    print(f"\n  ── (1) 감도: 파라미터 +{a.pert*100:.0f}% 당 궤적변화 ──")
    print(f"  {'파라미터':<14}{'초기값':>11}{'RMS[°]':>10}{'SNR':>8}   판정")
    for p in np.argsort(-rms):
        v = "식별가능" if snr[p] >= 5 else ("★경계(반복수집 필요)" if snr[p] >= 2 else "❌식별 불가")
        print(f"  {plabels[p]:<14}{x0[p]:>11.4g}{rms[p]:>10.4f}{snr[p]:>8.1f}   {v}")

    # ── (2) 파라미터 간 혼동 ───────────────────────────────────────────────
    Sn = S / np.where(rms > 0, rms, 1.0)
    C = (Sn.T @ Sn) / S.shape[0]
    C /= np.sqrt(np.outer(np.diag(C), np.diag(C)))
    off = np.abs(C - np.eye(len(x0)))
    print(f"\n  ── (2) 파라미터 간 상관 (|r|>0.9 면 그 둘은 맞바꿔진다) ──")
    pairs = [(off[i, j], i, j) for i in range(len(x0)) for j in range(i + 1, len(x0))]
    for r, i, j in sorted(pairs, reverse=True)[:5]:
        mark = "  ★혼동" if r > 0.9 else ("  주의" if r > 0.7 else "")
        print(f"    {plabels[i]:<14}↔ {plabels[j]:<14}  r={r:+.3f}{mark}")
    sv = np.linalg.svd(Sn, compute_uv=False)
    print(f"\n  감도행렬 조건수 {sv[0]/max(sv[-1],1e-30):.1f}  "
          f"(정규화 후. 크면 어떤 파라미터 조합이 궤적에 거의 안 나타난다)")

    # 참고: 축 상관 — 종전 지표. 남겨서 대조할 수 있게 한다.
    Cq = np.corrcoef(q_cmd.T)
    print(f"  [참고] 종전 지표 '축간 상관' 최대 {np.abs(Cq-np.eye(len(names))).max():.3f}")

    weak = [plabels[p] for p in range(len(x0)) if snr[p] < 2]
    conf = [f"{plabels[i]}↔{plabels[j]}" for r, i, j in pairs if r > 0.9]
    print()
    if weak:
        print(f"  ❌ 식별 불가(SNR<2): {', '.join(weak)}")
        print(f"     → 진폭·f_end 를 올리거나(토크예산 확인), 그 파라미터를 축별 시험으로 고정할 것")
    if conf:
        print(f"  ★혼동(|r|>0.9): {', '.join(conf)}")
        print(f"     → 위상·주파수 배치를 바꾸거나, 한쪽을 축별 측정으로 고정할 것")
    if not weak and not conf:
        print("  ✅ 전 파라미터 SNR≥2 · 혼동쌍 없음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
