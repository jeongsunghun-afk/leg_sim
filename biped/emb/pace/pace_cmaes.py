#!/usr/bin/env python3
"""pace_cmaes.py — PACE 본절차. MuJoCo 롤아웃 + CMA-ES 로 **궤적 재현매칭**.

목적함수:  Σ_t Σ_i w_i·(q_sim,i(t) − q_real,i(t))²      ← 드라이버 τ 를 **쓰지 않는다**
  ⇒ 우리를 괴롭히던 순환 문제(보고 τ 가 kp·err 로 재구성됨)가 **구조적으로 소멸**한다.
    실기가 위치+게인으로 돌았으니 시뮬도 **같은 제어기·같은 게인**으로 돌린다.

탐색 파라미터 (MJCF dof 속성):
      dof_armature      = ROTOR_I · N²      ← 로터 반사관성
      dof_damping       = JDAMP             ← 점성
      dof_frictionloss  = JFRIC             ← 쿨롱마찰
  기본은 **축종류별 묶음**(hip/thigh/calf/foot) — 8축이 같은 모터라 ROTOR_I 는 하나,
  마찰·감쇠는 기구가 다르니 4개씩. 총 1+4+4 = **9개**.
  `--per-axis` 로 축별(1+8+8=17)도 가능하나 sloppy 해진다.

★왜 창 단위로 재초기화하나
  30초를 개루프로 적분하면 편향이 누적돼 발산한다 — 모델 결함이 아니라 적분의 성질이다.
  `--window` 마다 **실측 상태로 재초기화**해 국소 정합도와 드리프트를 분리한다
  (pace_validate.py 와 같은 처리).

★base 는 **고정**하고 **바닥을 없앤다** — 실기가 매달린(거치) 상태이기 때문이다.
  ⚠freejoint 만 빼면 torso 가 `pos="0 0 0.5257"` 에 고정되는데, 그 높이는 **발이 바닥에
    닿도록** 계산된 값이다. 그대로 두면 시뮬 로봇이 **바닥을 딛고 선다** —
    2026-08-11 실제로 이걸 놓쳐서 "8축 전부 트립" 이라는 **가짜 결론**을 냈다.
    가진 4초 내내 두 발이 100% 접촉 중이었고, 지면이 다리를 붙잡아 추종오차가
    13~32° 로 폭증했다. 실기에는 없는 힘이다.
  ⇒ floor geom 을 제거한다. 확인: 롤아웃 중 `d.ncon == 0` 이어야 한다.

⚠**PACE 는 강체 부분이 맞다는 걸 전제한다.** MJCF 질량·관성이 틀리면 CMA-ES 가 그 오차를
  armature/damping/friction 으로 흡수해 "잘 맞는데 물리적으로 틀린" 값을 낸다.
  → 그래서 축별로 I_link 를 먼저 검증했다(foot 예측 대비 −1.0%, 2026-08-11).

사용:
    ~/.venv-mujoco/bin/python pace_cmaes.py results/pace_multichirp.npz
    ~/.venv-mujoco/bin/python pace_cmaes.py --selftest        # 하드웨어·실측 불필요
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(os.path.dirname(HERE))
DEG = np.pi / 180.0

# 축종류별 묶음 — 8축이 같은 모터(RO100)라 ROTOR_I 는 공유, 나머지는 기구별
KINDS = ("hip", "thigh", "calf", "foot")


def kind_of(name: str) -> str:
    for k in KINDS:
        if k in name:
            return k
    raise ValueError(name)


def load_fixed_base(mjcf_path: str):
    """freejoint 를 뺀 모델을 만들어 로드. 거치(매단) 상태와 일치시킨다."""
    import mujoco
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    for parent in root.iter():
        for fj in list(parent.findall("freejoint")):
            parent.remove(fj)
        # ★바닥 제거 — 매단 상태를 재현한다(위 주석 참조). 안 지우면 발이 지면에 눌린다.
        for g in list(parent.findall("geom")):
            if g.get("name") == "floor":
                parent.remove(g)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=os.path.dirname(mjcf_path),
                                     delete=False) as f:
        tree.write(f, encoding="unicode")
        tmp = f.name
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)
    return m


def joint_index(m, names):
    """npz 의 축 순서 → MJCF dof/qpos/actuator 인덱스. **이름으로** 맞춘다(순서 가정 금지)."""
    import mujoco
    out = []
    for nm in names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_joint")
        if jid < 0:
            raise SystemExit(f"✗ MJCF 에 관절 {nm}_joint 가 없다")
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        if aid < 0:
            raise SystemExit(f"✗ MJCF 에 actuator {nm} 가 없다")
        out.append((jid, int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]), aid))
    return out


def apply_params(m, idx, gear_n, p, per_axis: bool, names):
    """탐색 벡터 → MJCF dof 속성. p 는 log 공간이 아니라 실공간(양수 제약은 clip)."""
    n = len(idx)
    if per_axis:
        rot = p[0:1].repeat(n)
        dmp, frc = p[1:1 + n], p[1 + n:1 + 2 * n]
    else:
        rot = p[0:1].repeat(n)
        kd_ = {k: p[1 + i] for i, k in enumerate(KINDS)}
        kf_ = {k: p[5 + i] for i, k in enumerate(KINDS)}
        dmp = np.array([kd_[kind_of(x)] for x in names])
        frc = np.array([kf_[kind_of(x)] for x in names])
    for i, (_, _, dof, _) in enumerate(idx):
        m.dof_armature[dof] = max(rot[i], 1e-9) * gear_n[i] ** 2
        m.dof_damping[dof] = max(dmp[i], 0.0)
        m.dof_frictionloss[dof] = max(frc[i], 0.0)


def rollout(m, d, idx, q_real, dq_real, q_cmd, kp, kd, dt, win_steps):
    """창 단위 재초기화 롤아웃. 실기와 **같은 제어법칙**을 시뮬 안에서 돌린다."""
    import mujoco
    N, n = q_real.shape
    q_sim = np.empty_like(q_real)
    m.opt.timestep = dt
    for s in range(0, N, win_steps):
        e = min(s + win_steps, N)
        # ★창 시작마다 실측 상태로 재초기화 — 개루프 적분 발산과 모델오차를 분리한다
        for i, (_, qa, dofa, _) in enumerate(idx):
            d.qpos[qa] = q_real[s, i] * DEG
            d.qvel[dofa] = dq_real[s, i] * DEG
        mujoco.mj_forward(m, d)
        for t in range(s, e):
            for i, (_, qa, dofa, aid) in enumerate(idx):
                q_sim[t, i] = d.qpos[qa] / DEG
                err = (q_cmd[t, i] - d.qpos[qa] / DEG) * DEG
                d.ctrl[aid] = kp[i] * err - kd[i] * d.qvel[dofa]
            mujoco.mj_step(m, d)
    return q_sim


def cost_of(q_sim, q_real, w=None):
    r = (q_sim - q_real)
    if w is not None:
        r = r * w
    return float(np.sqrt(np.mean(r ** 2)))          # RMS[deg]


def load_data(path):
    z = np.load(path, allow_pickle=True)
    return dict(t=z["t"], q=z["q"], q_cmd=z["q_cmd"], dq=z["dq"],
                kp=z["kp_joint"], kd=z["kd_joint"], gear_n=z["gear_n"],
                names=[str(x) for x in z["names"]], dt=float(z["dt"]))


def init_bounds(spec_path, names, per_axis):
    """초기값·탐색범위 — **축별 측정값**에서 온다. 이게 sloppy 를 줄이는 핵심이다."""
    import yaml
    sp = yaml.safe_load(open(spec_path, encoding="utf-8"))
    rot0 = 7.327e-4          # 2026-08-11 τ_ff 경로 실측(foot 좌우 평균)
    d0 = {"hip": 0.09, "thigh": 0.09, "calf": 0.09, "foot": 0.02}   # JDAMP (hip 외삽/foot 실측)
    f0 = {"hip": 0.38, "thigh": 0.38, "calf": 0.38, "foot": 0.44}   # JFRIC (foot 은 절편 실측)
    if per_axis:
        x0 = np.concatenate([[rot0], [d0[kind_of(x)] for x in names],
                             [f0[kind_of(x)] for x in names]])
    else:
        x0 = np.concatenate([[rot0], [d0[k] for k in KINDS], [f0[k] for k in KINDS]])
    lo = x0 * 0.3
    hi = x0 * 3.0
    return x0, lo, hi


def report(names, x, per_axis, log=print):
    log(f"  {'파라미터':<16}{'값':>12}")
    log(f"  {'ROTOR_I':<16}{x[0]:>12.4e}")
    if per_axis:
        n = len(names)
        for i, nm in enumerate(names):
            log(f"  {'JDAMP.'+nm:<16}{x[1+i]:>12.4f}")
        for i, nm in enumerate(names):
            log(f"  {'JFRIC.'+nm:<16}{x[1+n+i]:>12.4f}")
    else:
        for i, k in enumerate(KINDS):
            log(f"  {'JDAMP.'+k:<16}{x[1+i]:>12.4f}")
        for i, k in enumerate(KINDS):
            log(f"  {'JFRIC.'+k:<16}{x[5+i]:>12.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", help="collect_multichirp.py 산출물")
    ap.add_argument("--mjcf", default=os.path.join(BIPED, "biped_flatfoot.mjcf"))
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--window", type=float, default=0.5, help="재초기화 창[s]")
    ap.add_argument("--per-axis", action="store_true", help="축별 17모수(기본은 묶음 9모수)")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="뒤쪽 이 비율은 **적합에 안 쓰고** 검증에만 쓴다")
    ap.add_argument("--eval-only", action="store_true", help="초기값만 평가(CMA-ES 생략)")
    ap.add_argument("--st-T", type=float, default=4.0, help="셀프테스트 길이[s]")
    ap.add_argument("--st-dt", type=float, default=0.002,
                    help="셀프테스트 스텝[s]. ★Pi 는 느리다 — 1ms×12s 는 세대당 수십초다")
    ap.add_argument("--selftest", action="store_true",
                    help="알려진 파라미터로 합성데이터를 만들어 되찾는지 확인")
    a = ap.parse_args()

    try:
        import mujoco  # noqa: F401
    except ImportError:
        raise SystemExit("✗ mujoco 가 없다. ~/.venv-mujoco/bin/python 으로 실행할 것.")

    import mujoco
    m = load_fixed_base(a.mjcf)
    print(f"■ 모델 {os.path.basename(a.mjcf)} (base 고정) — nq={m.nq} nv={m.nv} nu={m.nu}")

    if a.selftest:
        return selftest(m, a)

    if not a.npz:
        raise SystemExit("npz 를 주거나 --selftest 를 쓸 것")
    D = load_data(a.npz)
    idx = joint_index(m, D["names"])
    d = mujoco.MjData(m)
    N = len(D["t"])
    win = max(1, int(round(a.window / D["dt"])))
    ncut = int(N * (1 - a.holdout))
    print(f"■ 데이터 {os.path.basename(a.npz)} — {N} 표본 · dt {D['dt']*1000:.1f}ms · "
          f"창 {a.window}s({win} 스텝)")
    print(f"  적합 구간 0~{ncut} · **hold-out {ncut}~{N}** ({a.holdout:.0%})")

    x0, lo, hi = init_bounds(a.spec, D["names"], a.per_axis)
    print(f"■ 초기값(축별 측정 유래) — {len(x0)} 모수, 범위 ×0.3~×3.0")
    report(D["names"], x0, a.per_axis)

    def evaluate(p, s, e):
        apply_params(m, idx, D["gear_n"], np.asarray(p, float), a.per_axis, D["names"])
        qs = rollout(m, d, idx, D["q"][s:e], D["dq"][s:e], D["q_cmd"][s:e],
                     D["kp"], D["kd"], D["dt"], win)
        return cost_of(qs, D["q"][s:e]), qs

    c0, _ = evaluate(x0, 0, ncut)
    h0, _ = evaluate(x0, ncut, N)
    print(f"\n■ 초기값 RMS — 적합 {c0:.4f}° · hold-out {h0:.4f}°")
    if a.eval_only:
        return 0

    try:
        import cma
    except ImportError:
        raise SystemExit("✗ cma 가 없다: ~/.venv-mujoco/bin/pip install cma")
    es = cma.CMAEvolutionStrategy(list(x0), 0.3, {
        "popsize": a.popsize, "maxiter": a.iters,
        "bounds": [list(lo), list(hi)],
        "CMA_stds": list(x0 * 0.5), "verbose": -9})
    print(f"\n■ CMA-ES — popsize {a.popsize} · 최대 {a.iters} 세대")
    best, bestc = np.array(x0), c0
    it = 0
    while not es.stop():
        X = es.ask()
        F = [evaluate(x, 0, ncut)[0] for x in X]
        es.tell(X, F)
        it += 1
        if min(F) < bestc:
            bestc = float(min(F)); best = np.array(X[int(np.argmin(F))])
        if it % 10 == 0 or it == 1:
            print(f"    세대 {it:>3}  최량 RMS {bestc:.4f}°")
    hb, qs = evaluate(best, ncut, N)
    print(f"\n■ 결과 — 적합 RMS {bestc:.4f}° · **hold-out RMS {hb:.4f}°** "
          f"(초기 {c0:.4f}/{h0:.4f})")
    print(f"  개선 적합 {(1-bestc/c0)*100:+.1f}% · hold-out {(1-hb/h0)*100:+.1f}%")
    if hb > bestc * 1.5:
        print("  ⚠hold-out 이 적합보다 크게 나쁘다 — **과적합**이다. 모수를 줄이거나"
              " 데이터를 늘릴 것")
    report(D["names"], best, a.per_axis)
    out = os.path.splitext(a.npz)[0] + "_cmaes.npz"
    np.savez(out, x=best, x0=x0, rms_fit=bestc, rms_holdout=hb,
             per_axis=a.per_axis, names=np.array(D["names"]))
    print(f"\n  ✓ 저장: {out}")
    return 0


def selftest(m, a) -> int:
    """★알려진 파라미터로 합성데이터를 만들어 CMA-ES 가 되찾는지 확인.

    실측을 믿기 전에 **추정기 자체**를 분리 검증한다. 이번 세션에서 여러 번 배운 것:
    합성검증은 반드시 **실제 조건**(같은 궤적·같은 창·같은 노이즈)으로 해야 한다.
    """
    import mujoco
    import yaml
    names = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot",
             "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
    idx = joint_index(m, names)
    d = mujoco.MjData(m)
    sp = yaml.safe_load(open(a.spec, encoding="utf-8"))
    mc = sp["pace_multi"]
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), "config",
                                           "biped_emb.yaml"), encoding="utf-8"))
    J = {j["name"]: j for j in cfg["joints"]}
    kp = np.array([J[n]["kp"] * J[n]["gear_k"] ** 2 for n in names])
    kd = np.array([J[n]["kd"] * J[n]["gear_k"] ** 2 for n in names])
    gear_n = np.array([float([x for x in sp["joints"]
                              if x["name"] == n][0]["gear"]) for n in names])

    T, dt = float(a.st_T), float(a.st_dt)
    tt = np.arange(0, T, dt)
    amps = np.array(mc["amp_deg"], float)
    f0 = np.array(mc["f_start_hz"], float)
    k_ = (np.array(mc["f_end_hz"], float) - f0) / T
    phi = (np.arange(8) * 0.6180339887) % 1.0 * 2 * np.pi
    q_cmd = np.array([amps * np.sin(2 * np.pi * (f0 * t + 0.5 * k_ * t * t) + phi)
                      for t in tt])

    x_true = np.concatenate([[7.327e-4], [0.11, 0.07, 0.13, 0.025],
                             [0.42, 0.31, 0.50, 0.44]])
    apply_params(m, idx, gear_n, x_true, False, names)
    win = max(1, int(round(a.window / dt)))
    # 참 궤적: 실측 자리에 시뮬을 넣고 창 재초기화 없이 한 번에 굴린다
    q0 = np.zeros((len(tt), 8)); dq0 = np.zeros_like(q0)
    q_true = rollout(m, d, idx, q0, dq0, q_cmd, kp, kd, dt, len(tt))
    dq_true = np.vstack([np.zeros(8), np.diff(q_true, axis=0) / dt])
    rng = np.random.default_rng(0)
    q_meas = q_true + rng.normal(0, 0.02, q_true.shape)      # 엔코더 잡음 0.02°
    print(f"■ 셀프테스트 — 합성 {T:.0f}s · dt {dt*1000:.0f}ms · 측정잡음 0.02°")

    def ev(p):
        apply_params(m, idx, gear_n, np.asarray(p, float), False, names)
        qs = rollout(m, d, idx, q_meas, dq_true, q_cmd, kp, kd, dt, win)
        return cost_of(qs, q_meas)

    x0, lo, hi = init_bounds(a.spec, names, False)
    print(f"  초기 RMS {ev(x0):.4f}°  ·  참값 RMS {ev(x_true):.4f}°")
    try:
        import cma
    except ImportError:
        raise SystemExit("✗ cma 가 없다: ~/.venv-mujoco/bin/pip install cma")
    es = cma.CMAEvolutionStrategy(list(x0), 0.3, {
        "popsize": a.popsize, "maxiter": a.iters, "bounds": [list(lo), list(hi)],
        "CMA_stds": list(x0 * 0.5), "verbose": -9})
    best, bestc = np.array(x0), ev(x0)
    it = 0
    while not es.stop():
        X = es.ask(); F = [ev(x) for x in X]; es.tell(X, F); it += 1
        if min(F) < bestc:
            bestc = float(min(F)); best = np.array(X[int(np.argmin(F))])
        if it % 10 == 0:
            print(f"    세대 {it:>3}  최량 RMS {bestc:.4f}°")
    print(f"\n  {'파라미터':<16}{'참값':>12}{'추정':>12}{'오차':>10}")
    lab = ["ROTOR_I"] + [f"JDAMP.{k}" for k in KINDS] + [f"JFRIC.{k}" for k in KINDS]
    ok = True
    for i, l in enumerate(lab):
        e = (best[i] / x_true[i] - 1) * 100
        good = abs(e) < 30
        ok &= good
        print(f"  {l:<16}{x_true[i]:>12.4g}{best[i]:>12.4g}{e:>9.1f}%"
              + ("" if good else "  ★"))
    print(f"\n  RMS {bestc:.4f}° · 셀프테스트 {'통과' if ok else '실패(30% 초과 항목)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
