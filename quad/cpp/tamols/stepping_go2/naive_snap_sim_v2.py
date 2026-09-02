#!/usr/bin/env python3
"""stepping_go2/naive_snap_sim_v2.py — baseline: the RL env's NAIVE nearest-stone
snap marched over the TRAP stone fields (stones_trapv2_L{0..9}.csv, the v2 IRREGULAR fields).

Emulates the env foothold rule (plan_meta.json "naive_rule"):
  * virtual base marches along +x at the nominal advance ADV = vadv * 4 * phase_dur
    (= 0.24 m/cycle; footfalls_L0 measured mean 0.2410 confirms),
  * each leg's Raibert touchdown nominal = base@swing-end + hip offset (+ half
    stride) — calibrated directly as the cycle-0 nom_(x,y) anchors from
    footfalls_L0.csv (identical across levels) advanced by ADV per cycle,
    nominal y held at the hip y (+-0.142, lane-local),
  * snap: nominal x <= X_SNAP (strip end 0.75 + cell 0.02) -> spawn strip
    z=0.15; else NEAREST stone by Euclidean center distance
    (_snap_xy_to_stone semantics), foot z = stone top_z,
  * march until each leg's nominal passes the field end x=4.25.

Scoring per level (transitions = consecutive same-leg pairs whose LANDING is on
a stone; strip->strip excluded, strip->first-stone included):
  (a) % transitions with |dz| > 0.10 (marginal) and > 0.14 (infeasible),
  (b) first-failure x (snapped cx of the first infeasible landing, touchdown order),
  (c) % of on-stone footfalls landing on designated TRAP stones (trapv2_meta.json).

Output: naive_trapv2_L{0..9}.csv (per-footfall) + per-level table on stdout.
Usage: python3 naive_snap_sim_v2.py [--adv 0.216]   (default: calibrated 0.24)
"""
import argparse
import csv
import json
import math
import os

D = os.path.dirname(os.path.abspath(__file__))

MARGINAL_DZ = 0.10   # Go2 conservative feasible step-up/down limit
INFEASIBLE_DZ = 0.14 # beyond this the robot is assumed to fail
STRIP_Z = 0.15
FIELD_X_END = 4.25
LEG_ORDER = ["HR", "FR", "HL", "FL"]  # touchdown order: walk(RR,FR,RL,FL)


def load_meta():
    pm = json.load(open(os.path.join(D, "plan_meta.json")))
    tm = json.load(open(os.path.join(D, "trapv2_meta.json")))
    return pm, tm


def calibrate(pm):
    """Anchors = cycle-0 nominals from footfalls_L0 (level-independent); ADV from plan_meta."""
    adv = pm["vadv"] * 4.0 * pm["phase_dur"]           # 0.24 m/cycle
    x_snap = 0.75 + pm["cell"]                          # 0.77 (matches footfalls: -1 up to 0.7699, snapped from 0.7715)
    anchors = {}
    with open(os.path.join(D, "footfalls_L0.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["cycle"]) == 0:
                anchors[r["leg"]] = (float(r["nom_x"]), float(r["nom_y"]))
    assert set(anchors) == set(LEG_ORDER)
    return adv, x_snap, anchors


def load_stones(lvl):
    rows = []
    with open(os.path.join(D, f"stones_trapv2_L{lvl}.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(dict(idx=int(r["idx"]), ix=int(r["ix"]), iy=int(r["iy"]),
                             cx=float(r["cx"]), cy=float(r["cy"]),
                             size=float(r["size"]), top_z=float(r["top_z"])))
    return rows


def role_of(idx, lm):
    if idx in lm["trap"]:
        return "trap"
    if idx in lm["path"]:
        return "path"
    if idx in lm["near"]:
        return "near_decoy"
    return "far_decoy"


def run_level(lvl, tm_lvl, adv, x_snap, anchors, write_csv=True, rng=None,
              jitter_x=0.0, jitter_y=0.0):
    stones = load_stones(lvl)
    lm = dict(path=set(tm_lvl["path_idx"]), trap=set(tm_lvl["trap_idx"]),
              near=set(tm_lvl["near_decoy_idx"]))

    # ---- march the naive rule -------------------------------------------
    foots = []  # touchdown order
    n_cyc = {leg: 0 for leg in LEG_ORDER}
    for leg in LEG_ORDER:
        ax, _ = anchors[leg]
        n_cyc[leg] = int(math.floor((FIELD_X_END - ax) / adv)) + 1
    for c in range(max(n_cyc.values())):
        for leg in LEG_ORDER:
            if c >= n_cyc[leg]:
                continue
            ax, ay = anchors[leg]
            nx, ny = ax + c * adv, ay
            if rng is not None:
                nx += rng.gauss(0.0, jitter_x) if jitter_x else 0.0
                ny += rng.gauss(0.0, jitter_y) if jitter_y else 0.0
            if nx <= x_snap:
                foots.append(dict(cycle=c, leg=leg, nom_x=nx, nom_y=ny, on="strip",
                                  stone_idx=-1, ix=-1, iy=-1, cx=nx, cy=ny,
                                  top_z=STRIP_Z, role=""))
            else:
                s = min(stones, key=lambda s: math.hypot(nx - s["cx"], ny - s["cy"]))
                foots.append(dict(cycle=c, leg=leg, nom_x=nx, nom_y=ny, on="stone",
                                  stone_idx=s["idx"], ix=s["ix"], iy=s["iy"],
                                  cx=s["cx"], cy=s["cy"], top_z=s["top_z"],
                                  role=role_of(s["idx"], lm)))

    # ---- score -----------------------------------------------------------
    prev = {}
    n_trans = n_marg = n_inf = 0
    first_fail = None  # (cx, cycle, leg, dz)
    for f in foots:
        p = prev.get(f["leg"])
        f["dz_prev"] = ""
        f["marginal"] = f["infeasible"] = 0
        if p is not None and f["on"] == "stone":
            dz = f["top_z"] - p["top_z"]
            f["dz_prev"] = f"{dz:.4f}"
            n_trans += 1
            if abs(dz) > MARGINAL_DZ:
                n_marg += 1
                f["marginal"] = 1
            if abs(dz) > INFEASIBLE_DZ:
                n_inf += 1
                f["infeasible"] = 1
                if first_fail is None:
                    first_fail = (f["cx"], f["cycle"], f["leg"], dz)
        prev[f["leg"]] = f

    on_stone = [f for f in foots if f["on"] == "stone"]
    n_on = len(on_stone)
    n_trap = sum(1 for f in on_stone if f["role"] == "trap")
    dist = {r: sum(1 for f in on_stone if f["role"] == r)
            for r in ("path", "near_decoy", "far_decoy", "trap")}

    if write_csv:
        cols = ["cycle", "leg", "nom_x", "nom_y", "on", "stone_idx", "ix", "iy",
                "cx", "cy", "top_z", "role", "dz_prev", "marginal", "infeasible"]
        with open(os.path.join(D, f"naive_trapv2_L{lvl}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in foots:
                w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k])
                            for k in cols})

    return dict(level=lvl,
                n_footfalls=len(foots), n_on_stone=n_on, n_transitions=n_trans,
                n_marginal=n_marg, n_infeasible=n_inf,
                marginal_pct=round(100.0 * n_marg / n_trans, 1) if n_trans else 0.0,
                infeasible_pct=round(100.0 * n_inf / n_trans, 1) if n_trans else 0.0,
                first_fail_x=round(first_fail[0], 3) if first_fail else None,
                first_fail_cycle=first_fail[1] if first_fail else None,
                first_fail_leg=first_fail[2] if first_fail else None,
                first_fail_dz=round(first_fail[3], 3) if first_fail else None,
                n_trap_hits=n_trap,
                trap_hit_pct=round(100.0 * n_trap / n_on, 1) if n_on else 0.0,
                land_path=dist["path"], land_near=dist["near_decoy"],
                land_far=dist["far_decoy"], land_trap=dist["trap"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adv", type=float, default=None,
                    help="override nominal advance m/cycle (default: vadv*cycle from plan_meta)")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--jitter-y", type=float, default=0.0,
                    help="Gaussian sigma [m] on nominal y (emulates base/foot drift)")
    ap.add_argument("--jitter-x", type=float, default=0.0)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    pm, tm = load_meta()
    adv, x_snap, anchors = calibrate(pm)
    if a.adv is not None:
        adv = a.adv
    print(f"calibration: ADV={adv:.4f} m/cycle  X_SNAP={x_snap:.2f}  anchors=" +
          " ".join(f"{l}:({anchors[l][0]:+.4f},{anchors[l][1]:+.4f})" for l in LEG_ORDER))

    if a.jitter_x or a.jitter_y:
        import random
        print(f"jitter MC: sx={a.jitter_x} sy={a.jitter_y} trials={a.trials} (no CSV, means reported)")
        res = []
        for lv in tm["levels"]:
            acc = None
            rng = random.Random(a.seed + lv["level"])
            for _ in range(a.trials):
                r = run_level(lv["level"], lv, adv, x_snap, anchors,
                              write_csv=False, rng=rng,
                              jitter_x=a.jitter_x, jitter_y=a.jitter_y)
                if acc is None:
                    acc = {k: [] for k in r}
                for k, v in r.items():
                    acc[k].append(v)
            fails = [v for v in acc["first_fail_x"] if v is not None]
            mean = lambda k: sum(acc[k]) / len(acc[k])
            res.append(dict(level=lv["level"],
                            n_footfalls=int(mean("n_footfalls")),
                            n_on_stone=int(mean("n_on_stone")),
                            n_transitions=int(mean("n_transitions")),
                            n_marginal=round(mean("n_marginal"), 1),
                            n_infeasible=round(mean("n_infeasible"), 1),
                            marginal_pct=round(mean("marginal_pct"), 1),
                            infeasible_pct=round(mean("infeasible_pct"), 1),
                            first_fail_x=round(sum(fails) / len(fails), 3) if fails else None,
                            first_fail_cycle="", first_fail_leg="",
                            first_fail_dz=None,
                            fail_trial_pct=round(100.0 * len(fails) / a.trials, 1),
                            n_trap_hits=round(mean("n_trap_hits"), 1),
                            trap_hit_pct=round(mean("trap_hit_pct"), 1),
                            land_path=round(mean("land_path"), 1),
                            land_near=round(mean("land_near"), 1),
                            land_far=round(mean("land_far"), 1),
                            land_trap=round(mean("land_trap"), 1)))
    else:
        res = [run_level(lv["level"], lv, adv, x_snap, anchors, write_csv=not a.no_csv)
               for lv in tm["levels"]]

    hdr = (f"{'lvl':>3} {'foot':>5} {'stone':>5} {'trans':>5} {'marg':>4} {'inf':>4} "
           f"{'marg%':>6} {'inf%':>6} {'1st-fail-x':>10} {'@':>8} "
           f"{'trap':>4} {'trap%':>6} {'path/near/far/trap':>19}")
    print(hdr)
    for r in res:
        ff = f"{r['first_fail_x']:.3f}" if r["first_fail_x"] is not None else "-"
        at = (f"{r['fail_trial_pct']:.0f}%tr" if "fail_trial_pct" in r
              else f"c{r['first_fail_cycle']}{r['first_fail_leg']}"
              if r["first_fail_x"] is not None else "-")
        print(f"{r['level']:>3} {r['n_footfalls']:>5} {r['n_on_stone']:>5} "
              f"{r['n_transitions']:>5} {r['n_marginal']:>4} {r['n_infeasible']:>4} "
              f"{r['marginal_pct']:>6.1f} {r['infeasible_pct']:>6.1f} {ff:>10} {at:>8} "
              f"{r['n_trap_hits']:>4} {r['trap_hit_pct']:>6.1f} "
              f"{r['land_path']:>4}/{r['land_near']}/{r['land_far']}/{r['land_trap']}")
    return res


if __name__ == "__main__":
    main()
