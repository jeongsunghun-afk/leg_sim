#!/usr/bin/env python3
"""coef × JDAMP.calf 격자 — **축퇴 여부를 직접 재는** 진단.

★왜 (2026-08-14)
  coef 1-D 훑기에서 최소가 0.76 에 나왔는데, 축별로 뜯어보니 개선이 foot 이 아니라
  **calf 에서** 나왔다(−22%/−18%, foot 은 −3%). 가설은 "foot 잔차의 원인이 커플링
  계수"였으니 **어긋난 것**이다.

  기구학적으로 보면 coef 는 foot 액추에이터의 반력과 회전자 관성이 calf 로 얼마나
  실리는지를 정한다. 그러니 coef 를 낮추면 calf 가 받는 외란이 줄어 calf 잔차가 준다.
  ⇒ **JDAMP.calf 부족을 대신 메우고 있을 수 있다.** JDAMP.calf 는 지금까지 상자 벽에
    붙어 있던 미해결 항목이고, calf 감쇠는 실측이 없다(중력 잔차가 시험 진폭보다 커서).

  둘이 축퇴라면 (coef, JDAMP.calf) 평면의 최소가 **골짜기**로 길게 눕는다. 그러면
  coef 는 이 데이터로 식별 불가이고, 1-D 훑기의 0.76 은 물리값이 아니라
  JDAMP.calf 를 x0 에 묶어둔 탓에 생긴 그림자다.
  독립이라면 각 coef 마다 JDAMP.calf 를 다시 맞춰도 최소가 한 점에 남는다.

사용:  ~/.venv-mujoco/bin/python tests/scan_coef_jdamp.py [npz]
"""
import sys, pathlib
import numpy as np
import mujoco

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import pace_cmaes as P                                            # noqa: E402

NPZ = sys.argv[1] if len(sys.argv) > 1 else str(
    HERE.parent / "results/pace_multichirp_f0.4_dqfix.npz")
MJCF = str(HERE.parent.parent.parent / "biped_flatfoot.mjcf")
SPEC = str(HERE.parent / "spec.yaml")

COEFS = (0.76, 0.84, 0.92, 1.00)
DMULS = (0.5, 1.0, 2.0, 4.0)      # JDAMP.calf 를 x0 대비 몇 배로
I_DCALF = 3                       # [ROTOR_I, JDAMP.hip/thigh/calf/foot, JFRIC…]


def main() -> int:
    m = P.load_fixed_base(MJCF)
    d = mujoco.MjData(m)
    D = P.load_data(NPZ)
    nm = list(D["names"])
    idx = P.joint_index(m, nm)
    win = int(round(0.5 / D["dt"]))
    wrap = P.actuator_wrap(m, idx, nm, "tendon")
    x0, lo, hi, _ = P.init_bounds(SPEC, nm, False)
    fit, hold = P.split_segments(len(D["t"]), win, 0.2, "interleave")

    assert P.param_labels(nm, False)[I_DCALF] == "JDAMP.calf", \
        f"라벨이 밀렸다: {P.param_labels(nm, False)[I_DCALF]}"

    def ev(c, dmul, segs):
        x = np.array(x0)
        x[-1] = c
        x[I_DCALF] = float(np.clip(x0[I_DCALF] * dmul, lo[I_DCALF], hi[I_DCALF]))
        dyn, bias, dly, cf = P.split_params(x, len(nm), False)
        P.apply_params(m, idx, D["gear_n"], dyn, False, nm)
        P.set_coupling_coef(m, cf)
        Q = P.retarget_coupling(D["q"], nm, cf)
        QC = P.retarget_coupling(D["q_cmd"], nm, cf)
        DQ = P.retarget_coupling(D["dq"], nm, cf)
        ss = cnt = 0
        for s, e in segs:
            qs = P.rollout(m, d, idx, Q[s:e], DQ[s:e], QC[s:e], D["kp"], D["kd"],
                           D["dt"], win, bias=bias, delay_s=dly, wrap=wrap)
            r = qs - (Q[s:e] - bias)
            ss += float(np.sum(r ** 2)); cnt += r.size
        return float(np.sqrt(ss / cnt)), x[I_DCALF]

    print(f"■ coef × JDAMP.calf 격자   (x0 의 JDAMP.calf = {x0[I_DCALF]:.4f}, "
          f"상자 {lo[I_DCALF]:.3f}~{hi[I_DCALF]:.3f})")
    print(f"  {'':>7}" + "".join(f"{'×%.1f' % v:>10}" for v in DMULS)
          + f"{'행최소':>10}{'@':>7}")
    grid = np.zeros((len(COEFS), len(DMULS)))
    for i, c in enumerate(COEFS):
        row = []
        for j, dm in enumerate(DMULS):
            r, _ = ev(c, dm, fit)
            grid[i, j] = r; row.append(r)
        j = int(np.argmin(row))
        print(f"  {c:>7.2f}" + "".join(f"{v:>10.4f}" for v in row)
              + f"{row[j]:>10.4f}{'×%.1f' % DMULS[j]:>7}")

    i, j = np.unravel_index(int(np.argmin(grid)), grid.shape)
    print(f"\n  격자 최소  coef {COEFS[i]:.2f} · JDAMP.calf ×{DMULS[j]:.1f} "
          f"→ {grid[i, j]:.4f}")

    # ★판정: JDAMP.calf 를 각 coef 에서 다시 맞춘 뒤에도 coef 가 갈리나?
    prof = grid.min(axis=1)
    spread = float(prof.max() - prof.min())
    print(f"  coef 프로파일(각 행에서 JDAMP.calf 최적화 후): "
          + " ".join(f"{c:.2f}:{v:.4f}" for c, v in zip(COEFS, prof)))
    print(f"  프로파일 폭 {spread:.4f}° ({spread / prof.min():.1%})")
    if spread < 0.005:
        print("  ⇒ **축퇴다.** JDAMP.calf 를 풀면 coef 가 거의 안 갈린다.\n"
              "     coef 는 이 데이터로 식별 불가 — 1.0 에 고정하고 별도 시험이 필요하다.")
    else:
        print("  ⇒ 축퇴가 아니다. JDAMP.calf 를 풀어도 coef 최소가 남는다.")

    # hold-out 에서도 같은 결론인지 (적합에서만 나는 개선이면 과적합이다)
    print("\n  ★hold-out 대조 — 격자 최소점 vs coef=1 최적 JDAMP.calf")
    j1 = int(np.argmin(grid[COEFS.index(1.00)]))
    for lab, c, dm in (("격자 최소", COEFS[i], DMULS[j]),
                       ("coef=1.00", 1.00, DMULS[j1])):
        rf, _ = ev(c, dm, fit)
        rh, _ = ev(c, dm, hold)
        print(f"    {lab:<10} coef {c:.2f} ×{dm:.1f}   적합 {rf:.4f} · hold-out {rh:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
