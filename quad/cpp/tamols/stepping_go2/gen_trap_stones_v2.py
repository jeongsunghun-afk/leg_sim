#!/usr/bin/env python3
"""stepping_go2/gen_trap_stones_v2.py — v2 TRAP stepping-stone fields:
IRREGULAR SPARSE (no grid). v1 kept the uniform grid, so the naive
nearest-snap's natural row choice coincided with the golden path (verdict:
naive 0% infeasible L0-8 — no discrimination). v2 removes that crutch:

 1. NO GRID. Blue-noise (min-distance rejection) layout in the corridor
    x in [0.75,4.25], y in [-1,1]; sizes ramp 0.40 -> 0.15 over L0..9 and
    filler density ramps DOWN, so nominal hip-track points often have no
    nearby feasible stone.
 2. GOLDEN PATH FIRST, OFF the hip tracks: per side a feasible chain
    (step dx in [0.28,0.45] (reach cap 0.53), consecutive |dz| <= 0.06 via a
    shared smooth height profile h(x) in [0.10,0.26], surface gap <= 0.45,
    strip-anchored) that WEAVES: lateral offset alternates +-[0.10,0.25]
    around the hip track (+-0.142), so following it requires the lateral
    modulation the naive rule cannot do.
 3. TRAPS AT THE GREEDY-PREFERRED SPOTS: the naive nominal footprint
    sequence (virtual base +x ADV=0.24/cycle, footfalls_L0 cycle-0 anchors,
    nominal y = hip y — exactly naive_snap_sim.py's rule) is recomputed; a
    ramping fraction of those nominals (L2 25% -> L9 90%) get a TRAP stone
    placed NEARER to the nominal than any other stone, with top_z isolated
    by >0.16 from EVERY non-trap stone within A* reach (radius 0.80 >
    hypot(0.53,0.55)) and vs the strip where strip-reachable, so a
    multi-step selector can never step onto them, while the greedy nearest
    rule prefers them by construction. Trap tiers cycle sunken/high/higher
    so consecutive trap landings also tend to differ.
 4. Deterministic (RandomState(SEED+1000+lvl)); L0/1 trap-free sanity;
    brute feasibility walk of the golden chains under the A* edge rules.

Output: stones_trapv2_L{0..9}.csv (same 7-column drop-in format; ix = golden
chain rank else -1; iy = 0:R-golden 1:L-golden 2:trap 3:filler) +
trapv2_meta.json (path_idx/trap_idx/near_decoy_idx roles + design stats incl.
the bait-rate KPI = % of naive nominals whose nearest stone is a trap).
"""
import csv
import itertools
import json
import math
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260826
NUM_LEVELS = 10

STRIP_Z, STRIP_X = 0.15, 0.75
X_MAX, Y_HALF = 4.25, 1.0
Z_LO, Z_HI = 0.10, 0.26        # golden height-profile band
H_DX, H_STEP = 0.35, 0.04      # h(x) control spacing / |step| => slope<=0.114/m
GOLD_DZ_MAX = 0.06
DX_LO, DX_HI = 0.28, 0.45      # golden chain step band (relax: 0.18..0.50)
OFF_LO, OFF_HI = 0.10, 0.25    # weave offset magnitude around hip track
MAX_REACH, DY_MAX, L_MAX = 0.53, 0.55, 0.45   # v2 A* edge bands
DZ_HARD = 0.10                 # Go2 feasible step (A* hard limit)
ON_STONE_TOL, Y_BAND = 0.02, (0.04, 0.34)
COVER_X = 4.0
TRAP_ISO_R = 0.80              # isolation radius > max edge hypot = 0.764
TRAP_DZ_MIN = 0.16             # required |dz| trap vs any reachable non-trap
TRAP_M = 0.17                  # placement margin (> TRAP_DZ_MIN)
TRAP_Z_MIN, TRAP_Z_MAX = 0.02, 0.55
HIP_Y = {"R": -0.142, "L": 0.142}
LEG_ORDER = ["HR", "FR", "HL", "FL"]
FIELD_X_END = 4.25


def size_of(lvl):
    return 0.40 - 0.25 * lvl / 9.0


def bait_target(lvl):
    return 0.0 if lvl < 2 else 0.25 + 0.65 * (lvl - 2) / 7.0


def n_filler_of(lvl):
    return max(6, 26 - 2 * lvl)


def filler_amp(lvl):
    return 0.02 + 0.004 * lvl


def filler_mind(lvl):
    return 0.50 + 0.02 * lvl


def calibrate():
    """Same calibration as naive_snap_sim.py: ADV from plan_meta, cycle-0
    nominal anchors from footfalls_L0.csv (level-independent)."""
    pm = json.load(open(os.path.join(OUT, "plan_meta.json")))
    adv = pm["vadv"] * 4.0 * pm["phase_dur"]
    x_snap = 0.75 + pm["cell"]
    anchors = {}
    with open(os.path.join(OUT, "footfalls_L0.csv")) as f:
        for r in csv.DictReader(f):
            if int(r["cycle"]) == 0:
                anchors[r["leg"]] = (float(r["nom_x"]), float(r["nom_y"]))
    assert set(anchors) == set(LEG_ORDER)
    return adv, x_snap, anchors


def nominal_points(adv, x_snap, anchors):
    """The naive rule's on-stone touchdown nominals, exact replica of
    naive_snap_sim.run_level's march."""
    n_cyc = {leg: int(math.floor((FIELD_X_END - anchors[leg][0]) / adv)) + 1
             for leg in LEG_ORDER}
    pts = []
    for c in range(max(n_cyc.values())):
        for leg in LEG_ORDER:
            if c >= n_cyc[leg]:
                continue
            ax, ay = anchors[leg]
            nx = ax + c * adv
            if nx > x_snap:
                pts.append(dict(leg=leg, cycle=c, x=nx, y=ay))
    return pts


def draw_profile(rng):
    """Shared smooth height profile h(x): bounded reflected random walk on
    control points every H_DX, anchored at strip z, range >= 0.10."""
    xs = np.arange(STRIP_X, X_MAX + 2 * H_DX, H_DX)
    for _ in range(500):
        zs = [STRIP_Z]
        for _ in xs[1:]:
            z = zs[-1] + float(rng.uniform(-H_STEP, H_STEP))
            if z > Z_HI:
                z = 2 * Z_HI - z
            if z < Z_LO:
                z = 2 * Z_LO - z
            zs.append(z)
        if max(zs) - min(zs) >= 0.10:
            return xs, np.array(zs)
    raise RuntimeError("profile range not achieved")


def h_of(hxs, hzs, x):
    return float(np.interp(x, hxs, hzs))


def side_usable(stone, side):
    lo = stone["cy"] - stone["size"] / 2 - ON_STONE_TOL
    hi = stone["cy"] + stone["size"] / 2 + ON_STONE_TOL
    band = Y_BAND if side == "L" else (-Y_BAND[1], -Y_BAND[0])
    return max(lo, band[0]) <= min(hi, band[1])


def overlap(a, b, margin=0.02):
    hs = (a["size"] + b["size"]) / 2 + margin
    return abs(a["cx"] - b["cx"]) < hs and abs(a["cy"] - b["cy"]) < hs


def surf_gap(a, b):
    hs = (a["size"] + b["size"]) / 2
    return math.hypot(max(0.0, abs(b["cx"] - a["cx"]) - hs),
                      max(0.0, abs(b["cy"] - a["cy"]) - hs))


def d2(a, x, y):
    return math.hypot(a["cx"] - x, a["cy"] - y)


class JamError(RuntimeError):
    """Construction jam (sampling dead-end) — outer loop retries a new
    RNG substream; distinct from validation assertion failures (bugs)."""


def build_chain(side, hxs, hzs, size_lvl, rng, stones):
    """Weaving golden chain for one leg side, feasible under the A* edge
    rules; coverage to COVER_X."""
    chain = []
    sign = 1 if rng.randint(2) else -1
    guard = 0
    while not chain or chain[-1]["cx"] + chain[-1]["size"] / 2 < COVER_X:
        guard += 1
        if guard >= 40:
            raise JamError(f"chain too long side {side}")
        placed = None
        for relax in (0, 1, 2):
            off_lo, off_hi = [(OFF_LO, OFF_HI), (0.06, 0.32),
                              (0.03, 0.38)][relax]
            dx_lo, dx_hi = [(DX_LO, DX_HI), (0.20, 0.50),
                            (0.15, 0.52)][relax]
            for _ in range(200):
                size = float(size_lvl * rng.uniform(0.95, 1.05))
                half = size / 2
                if not chain:
                    cx = float(rng.uniform(0.95, 1.10))
                else:
                    prev = chain[-1]
                    lo_cx = max(prev["cx"] + 0.10, COVER_X - half + 0.005)
                    if prev["cx"] + dx_hi >= COVER_X - half and \
                            lo_cx < X_MAX - half:
                        cx = float(rng.uniform(lo_cx, X_MAX - half))
                    else:
                        cx = prev["cx"] + float(rng.uniform(dx_lo, dx_hi))
                if cx > X_MAX - half:
                    cx = X_MAX - half
                s_ = (sign if not relax or rng.randint(2) else -sign)
                mag = float(rng.uniform(off_lo, off_hi))
                cy = HIP_Y[side] + s_ * mag
                cand = dict(cx=cx, cy=cy, size=size,
                            top_z=h_of(hxs, hzs, cx), role="gold")
                if not side_usable(cand, side):
                    continue
                if abs(cy) > Y_HALF - half:
                    continue
                if chain:
                    prev = chain[-1]
                    dx = cand["cx"] - prev["cx"]
                    if not (0.0 < dx <= MAX_REACH):
                        continue
                    if abs(cand["cy"] - prev["cy"]) > 0.50:
                        continue
                    if math.hypot(dx, cand["cy"] - prev["cy"]) > 0.52:
                        continue          # calibrated center-jump cap (0.53)
                    if abs(cand["top_z"] - prev["top_z"]) > GOLD_DZ_MAX:
                        continue
                    if surf_gap(prev, cand) > L_MAX - 0.01:
                        continue
                else:
                    dx = cand["cx"] - STRIP_X
                    if dx > MAX_REACH:
                        continue
                    if abs(cand["top_z"] - STRIP_Z) > GOLD_DZ_MAX:
                        continue
                    if max(0.0, dx - half) > L_MAX - 0.01:
                        continue
                if any(overlap(cand, s) for s in stones + chain):
                    continue
                placed = (cand, s_)
                break
            if placed:
                break
        if placed is None:
            raise JamError(f"chain stuck side {side} k={len(chain)}")
        chain.append(placed[0])
        sign = -placed[1]
    return chain


def choose_trap_z(qx, qy, stones, rng, tier):
    """Trap top_z isolated by > TRAP_DZ_MIN from every non-trap stone within
    TRAP_ISO_R (and the strip where strip-reachable); tiers sunken / high /
    higher for variety; prefer max z-distance to nearby traps."""
    refs = [s["top_z"] for s in stones if s["role"] != "trap"
            and d2(s, qx, qy) <= TRAP_ISO_R]
    if qx <= STRIP_X + MAX_REACH + 0.05:
        refs.append(STRIP_Z)
    if not refs:
        refs = [STRIP_Z]
    trap_zs = [s["top_z"] for s in stones if s["role"] == "trap"
               and d2(s, qx, qy) <= 0.60]
    u = float(rng.uniform(0.005, 0.03))
    z_sun = min(refs) - TRAP_M - u
    z_hi = max(refs) + TRAP_M + u
    z_hi2 = min(TRAP_Z_MAX, max(refs) + TRAP_M + 0.17 + u)
    pref = dict(sun=z_sun, hi=z_hi, hi2=z_hi2)[tier]
    cands = [z for z in (z_sun, z_hi, z_hi2)
             if TRAP_Z_MIN <= z <= TRAP_Z_MAX
             and all(abs(z - r) > TRAP_DZ_MIN + 0.004 for r in refs)]
    if not cands:
        return None
    def score(z):
        d = min((abs(z - t) for t in trap_zs), default=1.0)
        return (min(d, 0.20), -abs(z - pref))
    return max(cands, key=score)


def try_place_trap(p, stones, size_lvl, rng, tier):
    """Place a trap NEARER to nominal p than any existing stone, non-
    overlapping, z-isolated. Returns stone dict or None."""
    size_t = float(np.clip(0.50 * size_lvl, 0.09, 0.17))
    half = size_t / 2
    d_near = min(d2(s, p["x"], p["y"]) for s in stones)
    for r in (0.0, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18):
        if r >= d_near - 0.02:
            break
        ths = [0.0] if r == 0.0 else [k * math.pi / 4 for k in
                                      rng.permutation(8)]
        for th in ths:
            qx = p["x"] + r * math.cos(th)
            qy = p["y"] + r * math.sin(th)
            if not (STRIP_X + half <= qx <= X_MAX - half
                    and abs(qy) <= Y_HALF - half):
                continue
            cand = dict(cx=qx, cy=qy, size=size_t, role="trap")
            if any(overlap(cand, s, 0.01) for s in stones):
                continue
            z = choose_trap_z(qx, qy, stones, rng, tier)
            if z is None:
                continue
            cand["top_z"] = z
            return cand
    return None


def build_field(lvl, rng, noms):
    """One construction attempt: chains + traps + fillers.
    Raises JamError on a sampling dead-end."""
    size_lvl = size_of(lvl)
    hxs, hzs = draw_profile(rng)
    stones = []
    chains = {}
    for side in ("R", "L"):
        c = build_chain(side, hxs, hzs, size_lvl, rng, stones)
        chains[side] = c
        stones += c

    # ── traps at greedy-preferred spots ──
    tiers = itertools.cycle(["hi", "sun", "hi2"])
    n_b = int(round(bait_target(lvl) * len(noms)))
    sel = (sorted(rng.choice(len(noms), n_b, replace=False),
                  key=lambda i: noms[i]["x"]) if n_b else [])
    skipped = 0
    baited = []          # (nominal, dist-to-its-trap) for filler guard
    for i in sel:
        p = noms[i]
        near = min(stones, key=lambda s: d2(s, p["x"], p["y"]))
        dn = d2(near, p["x"], p["y"])
        if near["role"] == "trap":
            baited.append((p, dn))
            continue
        cand = try_place_trap(p, stones, size_lvl, rng, next(tiers))
        if cand is None:
            skipped += 1
            continue
        stones.append(cand)
        baited.append((p, d2(cand, p["x"], p["y"])))

    # ── sparse fillers (blue-noise; must not steal baited nominals,
    #    must stay z-isolated from reachable traps) ──
    n_f_target = n_filler_of(lvl)
    amp = filler_amp(lvl)
    mind = filler_mind(lvl)
    n_f = 0
    for _ in range(600):
        if n_f >= n_f_target:
            break
        size_f = float(size_lvl * rng.uniform(0.80, 1.00))
        half = size_f / 2
        qx = float(rng.uniform(STRIP_X + half, X_MAX - half))
        qy = float(rng.uniform(-(Y_HALF - half), Y_HALF - half))
        cand = dict(cx=qx, cy=qy, size=size_f, role="fill")
        if any(overlap(cand, s) for s in stones):
            continue
        if any(d2(s, qx, qy) < mind for s in stones
               if s["role"] == "fill"):
            continue
        if any(d2(cand, p["x"], p["y"]) <= dp + 0.02
               for (p, dp) in baited):
            continue
        traps_near = [s["top_z"] for s in stones if s["role"] == "trap"
                      and d2(s, qx, qy) <= TRAP_ISO_R]
        z = None
        for _ in range(8):
            zc = float(np.clip(h_of(hxs, hzs, qx)
                               + rng.uniform(-amp, amp), 0.02, 0.55))
            if all(abs(zc - t) > TRAP_DZ_MIN + 0.004 for t in traps_near):
                z = zc
                break
        if z is None:
            continue
        cand["top_z"] = z
        stones.append(cand)
        n_f += 1

    return stones, chains, baited, skipped, n_f, hxs, hzs


def main():
    adv, x_snap, anchors = calibrate()
    noms = nominal_points(adv, x_snap, anchors)
    meta_levels = []
    print(f"calibration: ADV={adv:.4f} X_SNAP={x_snap:.2f} "
          f"n_nominals(on-stone)={len(noms)}")
    print(f"{'lvl':>3} {'n':>4} {'gold':>5} {'trap':>5} {'fill':>5} "
          f"{'skip':>4} {'bait%':>6} {'tgt%':>5} {'golddz':>7} {'size':>5} "
          f"{'hrange':>6} {'feasible':>8}")
    for lvl in range(NUM_LEVELS):
        best = None
        tgt = 100.0 * bait_target(lvl)
        for attempt in range(30):
            rng = np.random.RandomState(SEED + 1000 + lvl + 100000 * attempt)
            try:
                field = build_field(lvl, rng, noms)
            except JamError:
                continue
            rate = 100.0 * sum(
                1 for p in noms
                if min(field[0], key=lambda s: d2(s, p["x"], p["y"]))["role"]
                == "trap") / len(noms)
            if best is None or rate > best[1]:
                best = (field, rate)
            # accept when the attempt meets the design KPI (within 5pp of the
            # ramp target, and strictly >50% at L5+ where the KPI is hard)
            if rate >= tgt - 5.0 and (lvl < 5 or rate > 50.0):
                break
        assert best is not None, f"L{lvl}: all construction attempts jammed"
        (stones, chains, baited, skipped, n_f, hxs, hzs), _ = best
        size_lvl = size_of(lvl)

        # ── index + roles ──
        for i, s in enumerate(stones):
            s["idx"] = i
            s["iy"] = dict(gold=None, trap=2, fill=3)[s["role"]]
        for side, iy in (("R", 0), ("L", 1)):
            for k, s in enumerate(chains[side]):
                s["ix"], s["iy"] = k, iy
        for s in stones:
            if s["role"] != "gold":
                s["ix"] = -1

        # ── validation ──
        gold_dzs = []
        for side in ("R", "L"):
            ch = chains[side]
            prev = dict(cx=STRIP_X, cy=0.0, size=0.0, top_z=STRIP_Z)
            for k, s in enumerate(ch):
                dx = s["cx"] - prev["cx"]
                dz = abs(s["top_z"] - prev["top_z"])
                gold_dzs.append(dz)
                assert 0.0 < dx <= MAX_REACH + 1e-9, f"L{lvl} {side} reach"
                assert dz <= GOLD_DZ_MAX + 1e-9, f"L{lvl} {side} dz {dz}"
                assert dz <= DZ_HARD, f"L{lvl} {side} infeasible"
                if k == 0:
                    assert max(0.0, dx - s["size"] / 2) <= L_MAX
                else:
                    assert abs(s["cy"] - prev["cy"]) <= DY_MAX + 1e-9
                    assert surf_gap(prev, s) <= L_MAX + 1e-9
                    assert math.hypot(dx, s["cy"] - prev["cy"]) <= 0.53, \
                        f"L{lvl} {side} center jump > 0.53"
                assert side_usable(s, side), f"L{lvl} {side} band"
                prev = s
            assert ch[-1]["cx"] + ch[-1]["size"] / 2 >= COVER_X - 1e-9
        traps = [s for s in stones if s["role"] == "trap"]
        for t in traps:
            for s in stones:
                if s is t or s["role"] == "trap":
                    continue
                if d2(s, t["cx"], t["cy"]) <= 0.78:
                    assert abs(s["top_z"] - t["top_z"]) > TRAP_DZ_MIN, \
                        f"L{lvl} trap {t['idx']} not isolated vs {s['idx']}"
            if t["cx"] <= STRIP_X + MAX_REACH:
                assert abs(t["top_z"] - STRIP_Z) > TRAP_DZ_MIN
            assert TRAP_Z_MIN - 1e-9 <= t["top_z"] <= TRAP_Z_MAX + 1e-9
        for (p, dp) in baited:
            near = min(stones, key=lambda s: d2(s, p["x"], p["y"]))
            assert near["role"] == "trap", f"L{lvl} bait stolen at {p}"
        if lvl <= 1:
            assert not traps, f"L{lvl} must be trap-free sanity"

        # ── bait-rate KPI over ALL on-stone nominals ──
        n_bait_hit = sum(
            1 for p in noms
            if min(stones, key=lambda s: d2(s, p["x"], p["y"]))["role"]
            == "trap")
        bait_rate = 100.0 * n_bait_hit / len(noms)

        # ── write CSV (drop-in 7-column format) ──
        with open(os.path.join(OUT, f"stones_trapv2_L{lvl}.csv"), "w") as f:
            f.write("idx,ix,iy,cx,cy,size,top_z\n")
            for s in stones:
                f.write(f"{s['idx']},{s['ix']},{s['iy']},{s['cx']:.10g},"
                        f"{s['cy']:.10g},{s['size']:.10g},{s['top_z']:.10g}\n")

        n_gold = len(chains["R"]) + len(chains["L"])
        hrange = float(hzs.max() - hzs.min())
        print(f"{lvl:>3} {len(stones):>4} {n_gold:>5} {len(traps):>5} "
              f"{n_f:>5} {skipped:>4} {bait_rate:>6.1f} "
              f"{100 * bait_target(lvl):>5.1f} {max(gold_dzs):>7.4f} "
              f"{size_lvl:>5.3f} {hrange:>6.3f} {'OK':>8}")
        meta_levels.append(dict(
            level=lvl, n_stones=len(stones), n_gold=n_gold,
            n_gold_R=len(chains["R"]), n_gold_L=len(chains["L"]),
            n_trap=len(traps), n_filler=n_f, n_bait_skipped=skipped,
            bait_target_pct=round(100 * bait_target(lvl), 1),
            bait_rate_pct=round(bait_rate, 1),
            n_nominals=len(noms), n_bait_hit=n_bait_hit,
            gold_max_dz=float(max(gold_dzs)),
            size_lvl=size_lvl, trap_size=float(np.clip(0.50 * size_lvl,
                                                       0.09, 0.17)),
            profile_range=hrange,
            path_idx=sorted(s["idx"] for s in stones if s["role"] == "gold"),
            trap_idx=sorted(s["idx"] for s in stones if s["role"] == "trap"),
            near_decoy_idx=sorted(s["idx"] for s in stones
                                  if s["role"] == "fill")))

    with open(os.path.join(OUT, "trapv2_meta.json"), "w") as f:
        json.dump(dict(
            kind="stepping_stone_trap_fields_v2_irregular", seed=SEED,
            frame=("lane-local, IRREGULAR sparse layout (no grid); spawn "
                   "strip x<0.75 top z=0.15; stones x in [0.75,4.25], "
                   "y in [-1,1]; void plane z=0"),
            golden_path=("per-side weaving chain off the hip track "
                         "(offset alternates +-[0.10,0.25] around y=+-0.142), "
                         "dx in [0.28,0.45] (reach cap 0.53), |dz|<=0.06 via "
                         "shared height profile h(x) in [0.10,0.26], "
                         "strip-anchored, coverage to x>=4.0"),
            traps=("placed NEARER than any other stone to a ramping fraction "
                   "(L2 25% -> L9 90%) of the naive nominal footprints "
                   "(ADV=0.24 march, footfalls_L0 anchors); top_z isolated "
                   ">0.16 vs every non-trap stone within radius 0.80 and vs "
                   "the strip where strip-reachable; tiers sunken/high/higher"),
            fillers=("blue-noise sparse alternates at h(x)+-amp(lvl), density "
                     "ramping down 26->8, min spacing 0.50+0.02*lvl, barred "
                     "from stealing baited nominals and from trap z-bands"),
            limits=dict(step_feasible=DZ_HARD, gold_dz_max=GOLD_DZ_MAX,
                        trap_dz_min=TRAP_DZ_MIN, max_reach=MAX_REACH,
                        dy_max=DY_MAX, l_max=L_MAX, z_lo=Z_LO, z_hi=Z_HI,
                        strip_z=STRIP_Z, cover_x=COVER_X),
            levels=meta_levels), f, indent=1)
    print("wrote stones_trapv2_L{0..9}.csv + trapv2_meta.json")


if __name__ == "__main__":
    main()
