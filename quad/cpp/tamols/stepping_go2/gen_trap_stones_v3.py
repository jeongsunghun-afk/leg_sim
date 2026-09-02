#!/usr/bin/env python3
"""stepping_go2/gen_trap_stones_v3.py — v3 TRAP stepping-stone fields.
v3 fixes the four render-diagnosed defects of v2:

 1. LEFT-CHAIN BUG FIXED: v2's relax tiers let a chain fall into a same-side
    line (L0 left chain: all 7 offsets +0.23..+0.32 = 0.30 m outboard of the
    hip track, dx 0.454..0.516 above the [0.28,0.45] spec). v3 enforces
    STRICT sign alternation of the weave offset at every relax tier (relax
    only widens the magnitude/dx lower bounds), validates alternation and
    requires similar stone counts on both sides (|nR-nL| <= 2).
 2. STRIDE CAP = CRUISE-STRIDE BUDGET, NOT GEOMETRIC REACH: v2 capped the
    center jump at the 0.53 geometric reach (TAMOLS walk plans). The trained
    trot cruises ~0.2 m/step at vx 0.4 / 2 Hz. v3 caps the DIAGONAL step
    sqrt(dx^2+dy^2) of consecutive same-chain stones at BUDGET(lvl) =
    0.32 (L0) -> 0.40 (L9) (dynamically plausible; still within reach), and
    dx <= 0.45 everywhere. The same per-level budget is exported in the meta
    for the A* edge bands.
 3. EASED ENTRY: for x in [0.75, 2.0] both hip tracks get denser full-width
    entry fillers (|cy| in [0.10,0.25]) so the per-side edge-to-edge support
    gap along the track is <= ENTRY_GAP(lvl) = 0.20 (L0) ramping sparser
    (+0.035/lvl); support-sequence |dz| <= 0.10. Weave amplitude also ramps:
    |offset| <= 0.10 (L0) -> 0.25 (L9). Golden-chain stones may overlap each
    other (shared smooth h(x), |dz|<=0.06 seams) — at L0 the path is a dense
    garden path ~ the old grid builder's L0-2.
 4. BAIT OUTSIDE THE ENTRY ZONE AT LOW-MID LEVELS: traps (v2 mechanism kept:
    strictly nearest to the naive nominal footprints, top_z isolated >0.16
    from every non-trap stone within radius 0.80, L0-1 trap-free, ramping
    fraction L2 25% -> L9 90% of ELIGIBLE nominals) are restricted to
    x > 2.0 for L2-4, anywhere for L5+.

Deterministic (RandomState(SEED+2000+lvl+100000*attempt)); golden chains are
brute-walk validated per level (|dz| <= 0.06, diag step <= BUDGET(lvl)).
Output: stones_trapv3_L{0..9}.csv (7-column drop-in; ix = golden chain rank
else -1; iy = 0:R-golden 1:L-golden 2:trap 3:filler/entry) + trapv3_meta.json.
"""
import csv
import itertools
import json
import math
import os

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260827
NUM_LEVELS = 10

STRIP_Z, STRIP_X = 0.15, 0.75
X_MAX, Y_HALF = 4.25, 1.0
Z_LO, Z_HI = 0.10, 0.26        # golden height-profile band
H_DX, H_STEP = 0.35, 0.04      # h(x) control spacing / |step| => slope<=0.114/m
GOLD_DZ_MAX = 0.06
DX_CAP = 0.45                  # hard forward-step cap (spec)
DZ_HARD = 0.10                 # Go2 feasible step (A* hard limit)
L_MAX = 0.45                   # max surface-to-surface gap
DY_MAX = 0.55                  # lateral band (diag budget binds first)
ON_STONE_TOL, Y_BAND = 0.02, (0.04, 0.34)
COVER_X = 4.0
TRAP_ISO_R = 0.80              # isolation radius >> max edge diag (<=0.40)
TRAP_DZ_MIN = 0.16             # required |dz| trap vs any reachable non-trap
TRAP_M = 0.17                  # placement margin (> TRAP_DZ_MIN)
TRAP_Z_MIN, TRAP_Z_MAX = 0.02, 0.55
HIP_Y = {"R": -0.142, "L": 0.142}
LEG_ORDER = ["HR", "FR", "HL", "FL"]
FIELD_X_END = 4.25

X_ENTRY_END = 2.0              # eased-entry zone end
TRACK_HALF = 0.10              # track band half-width around hip y (support test)
ENTRY_CY_LO, ENTRY_CY_HI = 0.10, 0.25   # entry filler |cy| band
ENTRY_DZ_MAX = 0.10            # support-sequence dz cap along the entry walk


def budget(lvl):
    """Cruise-stride diagonal budget for consecutive same-chain steps."""
    return 0.32 + 0.08 * lvl / 9.0


def entry_gap(lvl):
    """Max edge-to-edge support gap along each hip track in the entry zone."""
    return 0.20 + 0.035 * lvl


def amp_band(lvl):
    """Weave offset magnitude band, ramping small->large. The lower bound
    also ramps (closest lateral approach to the hip track), reaching the
    fix-1 band floor 0.10+ by L2 — this is what opens room for strictly-
    nearest traps beside the dense budget-capped chains at mid levels."""
    return min(0.06 + 0.02 * lvl, 0.12), 0.10 + 0.15 * lvl / 9.0


def trap_x_min(lvl):
    """Bait only outside the eased entry zone at low-mid levels."""
    return 2.0 if lvl <= 4 else STRIP_X


def size_of(lvl):
    """Steeper-early ramp 0.40 -> 0.15: mid levels reach the trapped/sparse
    regime (spec fix 3 target) — also opens clearance for strictly-nearest
    traps beside the chains."""
    return 0.40 - 0.25 * (lvl / 9.0) ** 0.75


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


def build_chain(side, lvl, hxs, hzs, size_lvl, rng, stones):
    """Weaving golden chain for one leg side under the CRUISE-STRIDE budget:
    consecutive diag step <= budget(lvl), dx <= 0.45, |dz| <= 0.06, STRICT
    offset-sign alternation, coverage to COVER_X. Golden stones may overlap
    each other (dense garden path at low levels); never other stones."""
    B = budget(lvl)
    a_lo0, a_hi = amp_band(lvl)
    chain, mags, signs = [], [], []
    first_sign = 1 if rng.randint(2) else -1
    guard = 0
    while not chain or chain[-1]["cx"] + chain[-1]["size"] / 2 < COVER_X:
        guard += 1
        if guard >= 45:
            raise JamError(f"chain too long side {side}")
        placed = None
        for relax in (0, 1, 2):
            a_lo = [a_lo0, 0.05, 0.04][relax]
            d_lo = [0.18, 0.15, 0.13][relax]
            for _ in range(300):
                size = float(size_lvl * rng.uniform(0.95, 1.05))
                half = size / 2
                if chain:
                    prev = chain[-1]
                    s_ = -signs[-1]              # STRICT alternation (fix 1)
                    dcy_allow = math.sqrt(max(B * B - d_lo * d_lo, 1e-9)) \
                        - 0.005
                    m_hi = min(a_hi, dcy_allow - mags[-1])
                    if m_hi < a_lo:
                        continue
                    mag = float(rng.uniform(a_lo, m_hi))
                    cy = HIP_Y[side] + s_ * mag
                    dcy = abs(cy - prev["cy"])
                    dx_hi = min(DX_CAP, math.sqrt(max(B * B - dcy * dcy,
                                                      1e-9)) - 1e-4)
                    if dx_hi <= d_lo:
                        continue
                    cx = prev["cx"] + float(rng.uniform(d_lo, dx_hi))
                else:
                    s_ = first_sign
                    mag = float(rng.uniform(a_lo, a_hi))
                    cy = HIP_Y[side] + s_ * mag
                    cx = float(rng.uniform(0.85, STRIP_X + B - 0.01))
                if cx > X_MAX - half:
                    cx = X_MAX - half
                cand = dict(cx=cx, cy=cy, size=size,
                            top_z=h_of(hxs, hzs, cx), role="gold")
                if not side_usable(cand, side):
                    continue
                if abs(cy) > Y_HALF - half:
                    continue
                if chain:
                    prev = chain[-1]
                    dx = cand["cx"] - prev["cx"]
                    if not (0.0 < dx <= DX_CAP):
                        continue
                    if math.hypot(dx, cand["cy"] - prev["cy"]) > B - 1e-6:
                        continue                 # cruise-stride budget (fix 2)
                    if abs(cand["top_z"] - prev["top_z"]) > GOLD_DZ_MAX:
                        continue
                    if surf_gap(prev, cand) > L_MAX - 0.01:
                        continue
                else:
                    dx = cand["cx"] - STRIP_X
                    if dx > B - 1e-6:
                        continue
                    if abs(cand["top_z"] - STRIP_Z) > GOLD_DZ_MAX:
                        continue
                    if max(0.0, dx - half) > L_MAX - 0.01:
                        continue
                # golden stones may overlap each other (same h(x) surface);
                # never overlap non-gold stones
                if any(overlap(cand, s) for s in stones
                       if s["role"] != "gold"):
                    continue
                placed = (cand, s_, mag)
                break
            if placed:
                break
        if placed is None:
            raise JamError(f"chain stuck side {side} k={len(chain)}")
        chain.append(placed[0])
        signs.append(placed[1])
        mags.append(placed[2])
    return chain


def track_supports(stone, hip):
    """Stone's reachable y-interval intersects the hip-track band."""
    if stone["role"] == "trap":
        return False
    lo = stone["cy"] - stone["size"] / 2 - ON_STONE_TOL
    hi = stone["cy"] + stone["size"] / 2 + ON_STONE_TOL
    return max(lo, hip - TRACK_HALF) <= min(hi, hip + TRACK_HALF)


def walk_track(stones, hip, gap):
    """Greedy frontier walk along one hip track over existing supports.
    Returns (frontier_reached, support_sequence)."""
    frontier, prev_z = STRIP_X, STRIP_Z
    seq = []
    while frontier < X_ENTRY_END:
        cands = [s for s in stones if track_supports(s, hip)
                 and s["cx"] - s["size"] / 2 <= frontier + gap + 1e-9
                 and s["cx"] + s["size"] / 2 > frontier + 1e-6
                 and abs(s["top_z"] - prev_z) <= ENTRY_DZ_MAX + 1e-9]
        if not cands:
            return frontier, seq
        s = max(cands, key=lambda s: s["cx"] + s["size"] / 2)
        frontier = s["cx"] + s["size"] / 2
        prev_z = s["top_z"]
        seq.append(s)
    return frontier, seq


def build_entry(side, lvl, hxs, hzs, size_lvl, rng, stones):
    """Eased entry (fix 3): fill each hip track in x in [0.75, 2.0] so the
    edge-to-edge support gap is <= entry_gap(lvl) with support |dz| <= 0.10.
    Returns list of new entry stones (role='entry')."""
    gap = entry_gap(lvl)
    hip = HIP_Y[side]
    placed = []
    guard = 0
    while True:
        guard += 1
        if guard > 40:
            raise JamError(f"entry stuck side {side}")
        frontier, seq = walk_track(stones + placed, hip, gap)
        if frontier >= X_ENTRY_END:
            return placed
        prev_z = seq[-1]["top_z"] if seq else STRIP_Z
        ok = None
        for margin in (0.0, -0.08):
            for _ in range(120):
                size_f = float(size_lvl * rng.uniform(0.80, 1.00))
                half = size_f / 2
                cx = frontier + gap * float(rng.uniform(0.35, 0.90)) + half
                sgn = 1.0 if hip > 0 else -1.0
                cy = sgn * float(rng.uniform(ENTRY_CY_LO, ENTRY_CY_HI))
                zc = float(np.clip(h_of(hxs, hzs, cx)
                                   + rng.uniform(-0.03, 0.03), 0.02, 0.55))
                if abs(zc - prev_z) > 0.05:
                    continue
                cand = dict(cx=cx, cy=cy, size=size_f, top_z=zc, role="entry")
                if not track_supports(cand, hip):
                    continue
                if any(overlap(cand, s, margin) for s in stones + placed
                       if s["role"] != "gold"):
                    continue
                if margin < 0.0:
                    # small overlap vs golds allowed only if the seam is benign
                    if any(overlap(cand, s, margin) and
                           abs(zc - s["top_z"]) > 0.08
                           for s in stones if s["role"] == "gold"):
                        continue
                elif any(overlap(cand, s, 0.0) for s in stones
                         if s["role"] == "gold"):
                    continue
                ok = cand
                break
            if ok:
                break
        if ok is None:
            raise JamError(f"entry filler stuck side {side} x={frontier:.2f}")
        placed.append(ok)


def choose_trap_z(qx, qy, stones, rng, tier):
    """Trap top_z isolated by > TRAP_DZ_MIN from every non-trap stone within
    TRAP_ISO_R (and the strip where strip-reachable); tiers sunken / high /
    higher for variety; prefer max z-distance to nearby traps."""
    refs = [s["top_z"] for s in stones if s["role"] != "trap"
            and d2(s, qx, qy) <= TRAP_ISO_R]
    if qx <= STRIP_X + TRAP_ISO_R:
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


def try_place_trap(p, stones, size_lvl, rng, tier, x_min):
    """Place a trap NEARER to nominal p than any existing stone, non-
    overlapping, z-isolated, center x >= x_min. Returns stone dict or None."""
    size_t = float(np.clip(0.45 * size_lvl, 0.09, 0.14))
    half = size_t / 2
    d_near = min(d2(s, p["x"], p["y"]) for s in stones)
    for r in np.arange(0.0, 0.181, 0.02):
        if r >= d_near - 0.015:
            break
        ths = [0.0] if r == 0.0 else [k * math.pi / 8 for k in
                                      rng.permutation(16)]
        for th in ths:
            qx = p["x"] + r * math.cos(th)
            qy = p["y"] + r * math.sin(th)
            if qx < x_min:
                continue
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
    """One construction attempt: chains + entry easing + traps + fillers.
    Raises JamError on a sampling dead-end."""
    size_lvl = size_of(lvl)
    hxs, hzs = draw_profile(rng)
    stones = []
    chains = {}
    for side in ("R", "L"):
        c = build_chain(side, lvl, hxs, hzs, size_lvl, rng, stones)
        chains[side] = c
        stones += c
    if abs(len(chains["R"]) - len(chains["L"])) > 2:
        raise JamError("chain count asymmetry")

    # ── eased entry (fix 3) ──
    entries = []
    for side in ("R", "L"):
        e = build_entry(side, lvl, hxs, hzs, size_lvl, rng, stones)
        entries += e
        stones += e

    # ── traps at greedy-preferred spots, outside entry zone at L2-4 ──
    xmin = trap_x_min(lvl)
    eligible = [i for i, p in enumerate(noms) if p["x"] > xmin]
    tiers = itertools.cycle(["hi", "sun", "hi2"])
    n_b = int(round(bait_target(lvl) * len(eligible)))
    # candidate order: sparse pockets first (descending distance to the
    # nearest existing stone; deterministic tie-break by index), until n_b
    # nominals are baited or candidates are exhausted
    sel = sorted(eligible,
                 key=lambda i: (-min(d2(s, noms[i]["x"], noms[i]["y"])
                                     for s in stones), i)) if n_b else []
    skipped = 0
    baited = []          # (nominal, dist-to-its-trap) for filler guard
    for i in sel:
        if len(baited) >= n_b:
            break
        p = noms[i]
        near = min(stones, key=lambda s: d2(s, p["x"], p["y"]))
        dn = d2(near, p["x"], p["y"])
        if near["role"] == "trap":
            baited.append((p, dn))
            continue
        cand = try_place_trap(p, stones, size_lvl, rng, next(tiers), xmin)
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

    return stones, chains, entries, baited, skipped, n_f, hxs, hzs


def main():
    adv, x_snap, anchors = calibrate()
    noms = nominal_points(adv, x_snap, anchors)
    meta_levels = []
    print(f"calibration: ADV={adv:.4f} X_SNAP={x_snap:.2f} "
          f"n_nominals(on-stone)={len(noms)}")
    print(f"{'lvl':>3} {'n':>4} {'goldR':>5} {'goldL':>5} {'entry':>5} "
          f"{'trap':>5} {'fill':>5} {'skip':>4} {'baitE%':>6} {'tgt%':>5} "
          f"{'golddz':>7} {'maxdiag':>7} {'B':>5} {'egap':>5} {'size':>5} "
          f"{'feasible':>8}")
    for lvl in range(NUM_LEVELS):
        best = None
        tgt = 100.0 * bait_target(lvl)
        xmin = trap_x_min(lvl)
        eligible = [p for p in noms if p["x"] > xmin]
        for attempt in range(30):
            rng = np.random.RandomState(SEED + 2000 + lvl + 100000 * attempt)
            try:
                field = build_field(lvl, rng, noms)
            except JamError:
                continue
            rate = (100.0 * sum(
                1 for p in eligible
                if min(field[0], key=lambda s: d2(s, p["x"], p["y"]))["role"]
                == "trap") / len(eligible)) if eligible else 0.0
            if best is None or rate > best[1]:
                best = (field, rate)
            if rate >= tgt - 5.0 and (lvl < 5 or rate > 50.0):
                break
        assert best is not None, f"L{lvl}: all construction attempts jammed"
        (stones, chains, entries, baited, skipped, n_f, hxs, hzs), _ = best
        size_lvl = size_of(lvl)
        B = budget(lvl)

        # ── index + roles ──
        for i, s in enumerate(stones):
            s["idx"] = i
            s["iy"] = dict(gold=None, trap=2, fill=3, entry=3)[s["role"]]
        for side, iy in (("R", 0), ("L", 1)):
            for k, s in enumerate(chains[side]):
                s["ix"], s["iy"] = k, iy
        for s in stones:
            if s["role"] != "gold":
                s["ix"] = -1

        # ── validation: golden brute walk (fix 1 + 2) ──
        gold_dzs, gold_diags = [], []
        for side in ("R", "L"):
            ch = chains[side]
            prev = dict(cx=STRIP_X, cy=0.0, size=0.0, top_z=STRIP_Z)
            prev_off = None
            for k, s in enumerate(ch):
                dx = s["cx"] - prev["cx"]
                dz = abs(s["top_z"] - prev["top_z"])
                gold_dzs.append(dz)
                assert 0.0 < dx <= DX_CAP + 1e-9, f"L{lvl} {side} dx {dx}"
                assert dz <= GOLD_DZ_MAX + 1e-9, f"L{lvl} {side} dz {dz}"
                assert dz <= DZ_HARD, f"L{lvl} {side} infeasible"
                off = s["cy"] - HIP_Y[side]
                assert 0.03 <= abs(off) <= 0.27, f"L{lvl} {side} off {off}"
                if k == 0:
                    assert dx <= B + 1e-9, f"L{lvl} {side} strip dx {dx}"
                    assert max(0.0, dx - s["size"] / 2) <= L_MAX
                else:
                    diag = math.hypot(dx, s["cy"] - prev["cy"])
                    gold_diags.append(diag)
                    assert diag <= B + 1e-9, \
                        f"L{lvl} {side} diag {diag:.3f} > budget {B:.3f}"
                    assert surf_gap(prev, s) <= L_MAX + 1e-9
                    assert prev_off * off < 0.0, \
                        f"L{lvl} {side} k={k} offset sign not alternating"
                assert side_usable(s, side), f"L{lvl} {side} band"
                prev, prev_off = s, off
            assert ch[-1]["cx"] + ch[-1]["size"] / 2 >= COVER_X - 1e-9
        assert abs(len(chains["R"]) - len(chains["L"])) <= 2, \
            f"L{lvl} chain asymmetry"

        # ── validation: eased entry (fix 3) ──
        for side in ("R", "L"):
            frontier, seq = walk_track(stones, HIP_Y[side], entry_gap(lvl))
            assert frontier >= X_ENTRY_END - 1e-9, \
                f"L{lvl} {side} entry gap walk stuck at {frontier:.2f}"
            pz = STRIP_Z
            for s in seq:
                assert abs(s["top_z"] - pz) <= ENTRY_DZ_MAX + 1e-9
                pz = s["top_z"]
        for s in entries:
            assert ENTRY_CY_LO - 1e-9 <= abs(s["cy"]) <= ENTRY_CY_HI + 1e-9

        # ── validation: traps (fix 4 + v2 mechanism) ──
        traps = [s for s in stones if s["role"] == "trap"]
        for t in traps:
            assert t["cx"] >= xmin - 1e-9, f"L{lvl} trap in entry zone"
            for s in stones:
                if s is t or s["role"] == "trap":
                    continue
                if d2(s, t["cx"], t["cy"]) <= 0.78:
                    assert abs(s["top_z"] - t["top_z"]) > TRAP_DZ_MIN, \
                        f"L{lvl} trap {t['idx']} not isolated vs {s['idx']}"
            if t["cx"] <= STRIP_X + TRAP_ISO_R:
                assert abs(t["top_z"] - STRIP_Z) > TRAP_DZ_MIN
            assert TRAP_Z_MIN - 1e-9 <= t["top_z"] <= TRAP_Z_MAX + 1e-9
        for (p, dp) in baited:
            near = min(stones, key=lambda s: d2(s, p["x"], p["y"]))
            assert near["role"] == "trap", f"L{lvl} bait stolen at {p}"
        if lvl <= 1:
            assert not traps, f"L{lvl} must be trap-free sanity"

        # ── bait-rate KPIs ──
        def nearest_is_trap(p):
            return min(stones,
                       key=lambda s: d2(s, p["x"], p["y"]))["role"] == "trap"
        n_hit_all = sum(1 for p in noms if nearest_is_trap(p))
        n_hit_el = sum(1 for p in eligible if nearest_is_trap(p))
        bait_all = 100.0 * n_hit_all / len(noms)
        bait_el = (100.0 * n_hit_el / len(eligible)) if eligible else 0.0

        # ── write CSV (drop-in 7-column format) ──
        with open(os.path.join(OUT, f"stones_trapv3_L{lvl}.csv"), "w") as f:
            f.write("idx,ix,iy,cx,cy,size,top_z\n")
            for s in stones:
                f.write(f"{s['idx']},{s['ix']},{s['iy']},{s['cx']:.10g},"
                        f"{s['cy']:.10g},{s['size']:.10g},{s['top_z']:.10g}\n")

        max_diag = max(gold_diags) if gold_diags else 0.0
        print(f"{lvl:>3} {len(stones):>4} {len(chains['R']):>5} "
              f"{len(chains['L']):>5} {len(entries):>5} {len(traps):>5} "
              f"{n_f:>5} {skipped:>4} {bait_el:>6.1f} {tgt:>5.1f} "
              f"{max(gold_dzs):>7.4f} {max_diag:>7.4f} {B:>5.3f} "
              f"{entry_gap(lvl):>5.3f} {size_lvl:>5.3f} {'OK':>8}")
        meta_levels.append(dict(
            level=lvl, n_stones=len(stones),
            n_gold=len(chains["R"]) + len(chains["L"]),
            n_gold_R=len(chains["R"]), n_gold_L=len(chains["L"]),
            n_entry=len(entries), n_trap=len(traps), n_filler=n_f,
            n_bait_skipped=skipped,
            stride_budget=round(B, 4), entry_gap=round(entry_gap(lvl), 4),
            weave_amp_hi=round(amp_band(lvl)[1], 4),
            trap_x_min=round(xmin, 2),
            bait_target_pct=round(tgt, 1),
            bait_rate_eligible_pct=round(bait_el, 1),
            bait_rate_all_pct=round(bait_all, 1),
            n_nominals=len(noms), n_nominals_eligible=len(eligible),
            n_bait_hit=n_hit_all,
            gold_max_dz=float(max(gold_dzs)),
            gold_max_diag=round(max_diag, 4),
            size_lvl=size_lvl, trap_size=float(np.clip(0.50 * size_lvl,
                                                       0.09, 0.14)),
            profile_range=float(hzs.max() - hzs.min()),
            path_idx=sorted(s["idx"] for s in stones if s["role"] == "gold"),
            trap_idx=sorted(s["idx"] for s in stones if s["role"] == "trap"),
            entry_idx=sorted(s["idx"] for s in stones
                             if s["role"] == "entry"),
            near_decoy_idx=sorted(s["idx"] for s in stones
                                  if s["role"] in ("fill", "entry"))))

    with open(os.path.join(OUT, "trapv3_meta.json"), "w") as f:
        json.dump(dict(
            kind="stepping_stone_trap_fields_v3_cruise_budget", seed=SEED,
            frame=("lane-local, irregular sparse layout; spawn strip x<0.75 "
                   "top z=0.15; stones x in [0.75,4.25], y in [-1,1]; void "
                   "plane z=0"),
            golden_path=("per-side weaving chain, STRICT offset-sign "
                         "alternation around the hip track y=+-0.142, "
                         "amplitude ramp |off| in [~0.06,0.10] (L0) -> "
                         "[0.10,0.25] (L9); consecutive diag step "
                         "sqrt(dx^2+dy^2) <= budget(lvl)=0.32->0.40 (cruise-"
                         "stride, NOT the 0.53 geometric reach), dx<=0.45, "
                         "|dz|<=0.06 via shared h(x) in [0.10,0.26]; golden "
                         "stones may overlap each other (benign <=0.06 "
                         "seams); coverage to x>=4.0"),
            eased_entry=("x in [0.75,2.0]: per-hip-track support walk with "
                         "edge-to-edge gap <= 0.20+0.035*lvl, entry fillers "
                         "|cy| in [0.10,0.25], support |dz| <= 0.10 — a "
                         "0.15-0.25 m trot stride can physically enter"),
            traps=("v2 bait mechanism: placed NEARER than any other stone to "
                   "a ramping fraction (L2 25% -> L9 90%) of ELIGIBLE naive "
                   "nominal footprints (ADV=0.24 march, footfalls_L0 "
                   "anchors); eligible = x>2.0 for L2-4 (outside eased "
                   "entry), anywhere for L5+; top_z isolated >0.16 vs every "
                   "non-trap stone within radius 0.80 and vs the strip where "
                   "reachable; tiers sunken/high/higher; L0-1 trap-free"),
            fillers=("blue-noise sparse alternates at h(x)+-amp(lvl), "
                     "density ramping down 26->8, min spacing 0.50+0.02*lvl, "
                     "barred from stealing baited nominals and from trap "
                     "z-bands"),
            limits=dict(step_feasible=DZ_HARD, gold_dz_max=GOLD_DZ_MAX,
                        trap_dz_min=TRAP_DZ_MIN, dx_cap=DX_CAP,
                        dy_max=DY_MAX, l_max=L_MAX, z_lo=Z_LO, z_hi=Z_HI,
                        strip_z=STRIP_Z, cover_x=COVER_X,
                        stride_budget_l0=budget(0), stride_budget_l9=budget(9),
                        entry_x_end=X_ENTRY_END, entry_dz_max=ENTRY_DZ_MAX),
            levels=meta_levels), f, indent=1)
    print("wrote stones_trapv3_L{0..9}.csv + trapv3_meta.json")


if __name__ == "__main__":
    main()
