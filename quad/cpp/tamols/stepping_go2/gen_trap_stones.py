#!/usr/bin/env python3
"""stepping_go2/gen_trap_stones.py — TRAP variant of the varied-height stepping
stone fields where multi-step stone SELECTION genuinely matters.

Construction per level (same grid as stones_L{lvl}.csv — positions/sizes kept,
ONLY heights change):
  1. GOLDEN PATH: per leg-side, a forward chain of stones along the side's modal
     row taken from the verified TAMOLS plan (plan_L{lvl}.csv) — L0/1 = +-1 row
     pair, L2-5 = shared center row, L6-8 = inner pair, L9 = R@-0.28/L@0.0.
     Chain step = pitch = the min consecutive-stone center jump the plans used
     (calibrated reach band), edge gap <= max realized foot step (~0.358 m).
  2. Chain heights: ONE bounded random walk over columns (|step| <= 0.05 =>
     consecutive |dz| <= 0.06 > guaranteed feasible; Go2 feasible limit 0.10),
     reflected inside top-z [0.10, 0.26]; both side chains share the column
     value (zero roll step). Walk is redrawn until its range exceeds the safe
     field's z-range (variation strictly LARGER than stones_L{lvl}.csv).
     Strip (top z 0.15) -> first column also |dz| <= 0.05.
  3. TRAP STONES: decoys adjacent (8-neighborhood) to the golden path — the
     stones a nearest-snap would grab — get top z with |dz| > 0.16 vs EVERY
     neighboring path stone (raised or sunken; strip z=0.15 included as a ref
     for column-0 stones). Density ramps: frac = 0.125*(lvl-1) clipped to [0,1]
     (L0/1: 0 = sanity, L5: 0.5, L9: 1.0 = all decoys trapped); severity ramps:
     margin = 0.17 + 0.05*lvl/9 (+U(0,0.03)). Far decoys are trapped at the
     same frac vs their nearest path stone. Untrapped decoys stay feasible
     (|dz| <= 0.10 vs all adjacent path stones) = legitimate alternates.
  4. Deterministic: np.random.RandomState(SEED + level).
  5. Validated: path max |dz| <= 0.06, min trap dz > 0.16, chain step inside
     the plan-calibrated band, gap <= max foot step, brute walk of both chains
     under the Go2 limits, coverage to the last column.

Output: stones_trap_L{0..9}.csv (same 7-column format as stones_L*.csv, drop-in
for the env loader) + trap_meta.json (roles: path/near/far/trap indices).
"""
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260826
NUM_LEVELS = 10

STRIP_Z = 0.15          # spawn strip top z (x < 0.75)
Z_LO, Z_HI = 0.10, 0.26  # golden-path walk top-z bounds (range > safe field)
WALK_STEP = 0.05        # bounded random walk |step| (=> path |dz| <= 0.05)
PATH_DZ_MAX = 0.06      # required golden-path consecutive |dz| bound
LIMIT_FEAS = 0.10       # Go2 conservative feasible step-up/down between stones
TRAP_DZ_MIN = 0.16      # traps must exceed this vs neighboring path stones
TRAP_Z_MIN, TRAP_Z_MAX = 0.02, 0.55  # keep sunken traps above void, boxes sane


def trap_frac(lvl):
    """0 @ L0/1 (sanity), 0.5 @ L5, 1.0 @ L9 — piecewise-linear single formula."""
    return float(min(1.0, max(0.0, 0.125 * (lvl - 1))))


def trap_margin_base(lvl):
    return 0.17 + 0.05 * lvl / 9.0


def load_stones(lvl):
    rows = []
    with open(os.path.join(OUT, f"stones_L{lvl}.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(dict(idx=int(r["idx"]), ix=int(r["ix"]), iy=int(r["iy"]),
                             cx=float(r["cx"]), cy=float(r["cy"]),
                             size=float(r["size"]), top_z=float(r["top_z"])))
    return rows


def load_plan_calibration(lvl, cy_of_iy):
    """From the verified TAMOLS plan: modal chain row per leg side + the
    consecutive distinct-stone center-jump band actually used."""
    legs = defaultdict(list)
    with open(os.path.join(OUT, f"plan_L{lvl}.csv")) as f:
        for r in csv.DictReader(f):
            legs[r["leg"]].append((int(r["cycle"]), int(r["order"]),
                                   int(r["stone_idx"]),
                                   float(r["stone_cx"]), float(r["stone_cy"])))
    jumps, side_rows = [], {"R": Counter(), "L": Counter()}
    for leg, seq in legs.items():
        seq.sort()
        side = leg[1]  # FR/HR -> R, FL/HL -> L
        for a, b in zip(seq, seq[1:]):
            if a[2] != b[2]:
                jumps.append(math.hypot(b[3] - a[3], b[4] - a[4]))
        for s in seq:
            side_rows[side][round(s[4], 6)] += 1
    iy_side = {}
    for side in ("R", "L"):
        cy_modal = side_rows[side].most_common(1)[0][0]
        iy_side[side] = min(cy_of_iy, key=lambda iy: abs(cy_of_iy[iy] - cy_modal))
    return iy_side, (min(jumps), max(jumps))


def load_max_footstep(lvl):
    ff = defaultdict(list)
    with open(os.path.join(OUT, f"footfalls_L{lvl}.csv")) as f:
        for r in csv.DictReader(f):
            ff[r["leg"]].append((int(r["cycle"]), float(r["foot_x"]),
                                 float(r["foot_y"])))
    mx = 0.0
    for leg, seq in ff.items():
        seq.sort()
        for a, b in zip(seq, seq[1:]):
            mx = max(mx, math.hypot(b[1] - a[1], b[2] - a[2]))
    return mx


def draw_golden_walk(nx, safe_range, rng):
    """Bounded reflected random walk over columns, strip-anchored; redraw until
    its z-range strictly exceeds the safe field's range (and >= 0.08)."""
    need = max(safe_range + 0.01, 0.08)
    for _ in range(500):
        w, z = [], STRIP_Z
        for _ in range(nx):
            z = z + float(rng.uniform(-WALK_STEP, WALK_STEP))
            if z > Z_HI:
                z = 2 * Z_HI - z
            if z < Z_LO:
                z = 2 * Z_LO - z
            w.append(z)
        if (max(w) - min(w)) >= need:
            return w
    raise RuntimeError("golden walk range not achieved")


meta_levels = []
print(f"{'lvl':>3} {'n':>4} {'path':>5} {'near':>5} {'far':>4} {'trap':>5} "
      f"{'frac':>5} {'pathdz':>7} {'mintrap':>8} {'range':>6} {'safe_r':>6} "
      f"{'band':>13} {'rows(R,L)':>12} {'feasible':>8}")
for lvl in range(NUM_LEVELS):
    rng = np.random.RandomState(SEED + lvl)
    stones = load_stones(lvl)
    nx = max(s["ix"] for s in stones) + 1
    ny = max(s["iy"] for s in stones) + 1
    pitch = stones[ny]["cx"] - stones[0]["cx"] if nx > 1 else 0.0
    size = stones[0]["size"]
    by_grid = {(s["ix"], s["iy"]): s for s in stones}
    cy_of_iy = {iy: by_grid[(0, iy)]["cy"] for iy in range(ny)}
    safe_range = (max(s["top_z"] for s in stones) -
                  min(s["top_z"] for s in stones))

    iy_side, jump_band = load_plan_calibration(lvl, cy_of_iy)
    max_footstep = load_max_footstep(lvl)

    # ── 1+2. golden path: full column chain per side on the plan's modal rows
    path_iys = sorted(set(iy_side.values()))
    walk = draw_golden_walk(nx, safe_range, rng)
    path_set = set()
    for iy in path_iys:
        for ix in range(nx):
            path_set.add((ix, iy))
    z_new = {}
    for (ix, iy) in path_set:
        z_new[(ix, iy)] = walk[ix]          # both chains share the column value

    # ── classify decoys
    near, far = [], []
    for s in stones:
        key = (s["ix"], s["iy"])
        if key in path_set:
            continue
        adj = [(s["ix"] + dx, s["iy"] + dy) for dx in (-1, 0, 1)
               for dy in (-1, 0, 1) if not dx == dy == 0]
        adj_path = [k for k in adj if k in path_set]
        (near if adj_path else far).append((key, adj_path))

    # ── 3. traps (deterministic count via seeded shuffle)
    frac = trap_frac(lvl)
    n_trap_near = int(round(frac * len(near)))
    n_trap_far = int(round(frac * len(far)))
    order_near = [near[i] for i in rng.permutation(len(near))]
    order_far = [far[i] for i in rng.permutation(len(far))]
    trap_set, trap_ref = set(), {}

    def nearest_path_key(key):
        sx, sy = by_grid[key]["cx"], by_grid[key]["cy"]
        return min(path_set, key=lambda k: (by_grid[k]["cx"] - sx) ** 2 +
                                           (by_grid[k]["cy"] - sy) ** 2)

    def place_trap(key, ref_keys):
        refs = [walk[k[0]] for k in ref_keys]
        if key[0] == 0:
            refs.append(STRIP_Z)            # snap from the strip is also tempting
        m = trap_margin_base(lvl) + float(rng.uniform(0.0, 0.03))
        go_up = bool(rng.randint(2))
        z_up, z_dn = max(refs) + m, min(refs) - m
        z = z_up if go_up else z_dn
        if z < TRAP_Z_MIN:
            z = z_up
        if z > TRAP_Z_MAX:
            z = z_dn
        trap_set.add(key)
        trap_ref[key] = refs
        return z

    def place_feasible(key, ref_keys):
        refs = [walk[k[0]] for k in ref_keys]
        if key[0] == 0:
            refs.append(STRIP_Z)
        lo, hi = max(refs) - LIMIT_FEAS, min(refs) + LIMIT_FEAS
        mid = 0.5 * (lo + hi)
        return float(np.clip(mid + rng.uniform(-0.03, 0.03), lo, hi))

    for i, (key, adj_path) in enumerate(order_near):
        z_new[key] = (place_trap(key, adj_path) if i < n_trap_near
                      else place_feasible(key, adj_path))
    for i, (key, _) in enumerate(order_far):
        ref = [nearest_path_key(key)]
        z_new[key] = (place_trap(key, ref) if i < n_trap_far
                      else place_feasible(key, ref))

    # ── 5. validation
    path_dzs = []
    for iy in path_iys:                       # brute-walk each chain
        zs = [STRIP_Z] + [z_new[(ix, iy)] for ix in range(nx)]
        for a, b in zip(zs, zs[1:]):
            path_dzs.append(abs(b - a))
        assert all(d <= PATH_DZ_MAX + 1e-12 for d in path_dzs), f"L{lvl} path dz"
        assert all(d <= LIMIT_FEAS for d in path_dzs), f"L{lvl} path infeasible"
        # chain step length within the plan-calibrated reach band + gap check
        assert jump_band[0] - 1e-9 <= pitch <= jump_band[1] + 1e-9, \
            f"L{lvl} chain step {pitch} outside plan band {jump_band}"
        assert pitch - size <= max_footstep + 1e-9, f"L{lvl} gap unreachable"
        assert (nx - 1, iy) in path_set       # coverage to last column
    if len(path_iys) == 2:                    # roll step between the two chains
        for ix in range(nx):
            assert abs(z_new[(ix, path_iys[0])] -
                       z_new[(ix, path_iys[1])]) <= LIMIT_FEAS
    min_trap_dz = min((min(abs(z_new[k] - r) for r in trap_ref[k])
                       for k in trap_set), default=float("nan"))
    if trap_set:
        assert min_trap_dz > TRAP_DZ_MIN, f"L{lvl} weak trap {min_trap_dz}"
    for key, adj_path in near:                # untrapped near decoys stay feasible
        if key not in trap_set:
            assert all(abs(z_new[key] - walk[k[0]]) <= LIMIT_FEAS + 1e-9
                       for k in adj_path), f"L{lvl} untrapped decoy infeasible"
    assert all(TRAP_Z_MIN <= z_new[k] <= TRAP_Z_MAX for k in trap_set)
    walk_range = max(walk) - min(walk)
    assert walk_range > safe_range, f"L{lvl} variation not larger than safe"
    if lvl <= 1:
        assert not trap_set, f"L{lvl} must be trap-free sanity"

    # ── write CSV (same 7-column drop-in format as stones_L*.csv)
    with open(os.path.join(OUT, f"stones_trap_L{lvl}.csv"), "w") as f:
        f.write("idx,ix,iy,cx,cy,size,top_z\n")
        for s in stones:
            z = z_new[(s["ix"], s["iy"])]
            f.write(f"{s['idx']},{s['ix']},{s['iy']},{s['cx']:.10g},"
                    f"{s['cy']:.10g},{s['size']:.10g},{z:.10g}\n")

    n_trap = len(trap_set)
    rows_lbl = f"{cy_of_iy[iy_side['R']]:+.2f},{cy_of_iy[iy_side['L']]:+.2f}"
    print(f"{lvl:>3} {len(stones):>4} {len(path_set):>5} {len(near):>5} "
          f"{len(far):>4} {n_trap:>5} {frac:>5.3f} {max(path_dzs):>7.4f} "
          f"{(min_trap_dz if trap_set else float('nan')):>8.4f} "
          f"{walk_range:>6.3f} {safe_range:>6.3f} "
          f"{jump_band[0]:>6.3f}-{jump_band[1]:<6.3f} {rows_lbl:>12} {'OK':>8}")
    meta_levels.append(dict(
        level=lvl, n_stones=len(stones), nx=nx, ny=ny, pitch=pitch, size=size,
        rows_iy=dict(R=int(iy_side["R"]), L=int(iy_side["L"])),
        rows_cy=dict(R=cy_of_iy[iy_side["R"]], L=cy_of_iy[iy_side["L"]]),
        n_path=len(path_set), n_near_decoy=len(near), n_far_decoy=len(far),
        n_trap=n_trap, n_trap_near=n_trap_near, n_trap_far=n_trap_far,
        trap_frac=frac, trap_margin_base=trap_margin_base(lvl),
        path_max_dz=float(max(path_dzs)),
        min_trap_dz=(float(min_trap_dz) if trap_set else None),
        path_z_min=float(min(walk)), path_z_max=float(max(walk)),
        path_z_range=float(walk_range), safe_z_range=float(safe_range),
        plan_jump_band=[float(jump_band[0]), float(jump_band[1])],
        max_footstep_plan=float(max_footstep),
        path_idx=sorted(by_grid[k]["idx"] for k in path_set),
        trap_idx=sorted(by_grid[k]["idx"] for k in trap_set),
        near_decoy_idx=sorted(by_grid[k]["idx"] for k, _ in near)))

with open(os.path.join(OUT, "trap_meta.json"), "w") as f:
    json.dump(dict(
        kind="stepping_stone_trap_fields", seed=SEED,
        frame=("lane-local, identical grid to stones_L*.csv (positions/sizes "
               "unchanged, only top_z reassigned); spawn strip x<0.75 top "
               "z=0.15; stones x in [0.75,4.25]; void plane z=0"),
        golden_path=("per-side forward chain on the TAMOLS plan's modal row; "
                     "shared bounded random walk per column, |step|<=0.05, "
                     "top z in [0.10,0.26], range > safe field; strip-anchored"),
        traps=("decoys 8-adjacent to the path get |dz|>0.16 vs ALL neighboring "
               "path stones (strip z included at column 0), raised or sunken; "
               "frac=0.125*(lvl-1) in [0,1]; margin=0.17+0.05*lvl/9+U(0,0.03); "
               "far decoys trapped vs nearest path stone at the same frac; "
               "untrapped decoys kept feasible (|dz|<=0.10) as alternates"),
        limits=dict(step_feasible=LIMIT_FEAS, path_dz_max=PATH_DZ_MAX,
                    trap_dz_min=TRAP_DZ_MIN, walk_step=WALK_STEP,
                    z_lo=Z_LO, z_hi=Z_HI, strip_z=STRIP_Z),
        levels=meta_levels), f, indent=1)
print("wrote stones_trap_L{0..9}.csv + trap_meta.json")
