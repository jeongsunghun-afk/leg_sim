#!/usr/bin/env python3
"""stepping_go2/astar_select_v2.py — multi-step stone-SELECTION planner (A* graph
search) for the v2 IRREGULAR TRAP stepping-stone fields
(stones_trapv3_L{0..9}.csv). v2 adaptation of astar_select.py: the fields have
no grid, so the reach/lateral edge bands are the fixed v2 design bands
(MAX_REACH=0.53, DY_MAX=0.55 — the golden chains are built inside them) and
the goal is coverage (cx+size/2 >= min(4.0, field max)) instead of a last
grid column. Edge rules otherwise identical: HARD |dz| <= 0.10, surface gap
<= l_max 0.45, side lateral band via plan_meta y_band.

Per level, per leg SIDE (mirroring the TAMOLS plan convention: left legs use
stones whose reachable surface intersects the left lateral foot band, right
legs the right band — plan_meta y_band [0.04,0.34] mirrored, on_stone_tol):
  nodes  = stones (plus a virtual START on the spawn strip, x=0.75, z=0.15)
  edges  = forward progress 0 < dx <= MAX_REACH (1.35x the column pitch, the
           plan-calibrated chain step band), |dy| <= row pitch (one-row lateral
           shift, plans realized <= 0.373), surface-to-surface gap <= l_max
           0.45, and |dz| <= 0.10 HARD (Go2 feasible step; strip z included).
  cost   = W1*edge length + W2*|dz| + W3*edge-margin penalty
           (penalty = gap/l_max + |dy|/dy_max + (1 - size/size_max): prefer
           aligned stone centers, small gaps, larger stones)
  goal   = any usable stone in the last column (corridor end); A* heuristic
           h = W1 * remaining forward distance (admissible).
The two side routes are interleaved into the per-leg footfall sequence in the
plan_L*.csv gait order walk(HR,FR,HL,FL): front leg leads on route[k], hind
follows on route[k-1] (front ends on the last column, hind one stone behind —
same as the verified TAMOLS plans).

Output: plan_astarv3_L{0..9}.csv with columns order,leg,stone_idx,cx,cy,top_z.
Validation: route found at every level; every consecutive same-side stone
transition |dz| <= 0.10 (strip->first included); route reaches the last
column with coverage cx+size/2 >= min(4.0, field max coverage — the grid of
some levels ends before x=4.0, e.g. L2 max 3.833, exactly as in the TAMOLS
plans whose criterion was coverage_front_last_col).
Report per level: route found, n stones used, max |dz| used, golden-path
overlap %, traps used (must be 0).
"""
import csv
import heapq
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NUM_LEVELS = 10

# ── hard limits / calibration (trap_meta limits + plan_meta bands) ──────────
DZ_MAX = 0.10            # HARD Go2 feasible step up/down between supports
STRIP_Z = 0.15           # spawn strip top z (x < 0.75)
STRIP_X = 0.75           # strip front edge (start of stone region)
DX_CAP = 0.45            # hard forward-step cap (budget <= 0.40 binds first)
DY_MAX = 0.55            # lateral band (diag budget binds first)
L_MAX = 0.45             # plan_meta l_max: max foot travel (surface gap bound)
ON_STONE_TOL = 0.02      # plan_meta on_stone_tol: foot may overhang this much
Y_BAND = (0.04, 0.34)    # plan_meta y_band: |foot y| band; L=+, R=- (mirror)
COVER_X = 4.0            # corridor-end coverage target (capped at field max)

# ── cost weights ────────────────────────────────────────────────────────────
W1 = 1.0                 # path length (m)
W2 = 2.0                 # sum |dz| (m)
W3 = 0.3                 # edge-margin penalty (dimensionless)


def load_stones(lvl):
    rows = []
    with open(os.path.join(HERE, f"stones_trapv3_L{lvl}.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(dict(idx=int(r["idx"]), ix=int(r["ix"]), iy=int(r["iy"]),
                             cx=float(r["cx"]), cy=float(r["cy"]),
                             size=float(r["size"]), top_z=float(r["top_z"])))
    return rows


def side_usable(stone, side):
    """Mirror of the TAMOLS plan lateral convention: the stone's reachable
    y-interval (surface +- on_stone_tol overhang) must intersect the side's
    foot band (y_band, sign-mirrored). L9's left chain on the cy=0 row stands
    at foot y 0.04..0.06 exactly via this tolerance."""
    lo = stone["cy"] - stone["size"] / 2 - ON_STONE_TOL
    hi = stone["cy"] + stone["size"] / 2 + ON_STONE_TOL
    band = Y_BAND if side == "L" else (-Y_BAND[1], -Y_BAND[0])
    return max(lo, band[0]) <= min(hi, band[1])


def edge_feasible(a, b, max_reach, dy_max):
    """a -> b: forward, within reach/lateral band, surface gap, HARD |dz|."""
    dx = b["cx"] - a["cx"]
    if not (0.0 < dx <= max_reach + 1e-9):
        return None
    dy = abs(b["cy"] - a["cy"])
    if dy > dy_max + 1e-9:
        return None
    dz = abs(b["top_z"] - a["top_z"])
    if dz > DZ_MAX + 1e-12:
        return None
    if a["idx"] != -1 and math.hypot(dx, dy) > max_reach + 1e-9:
        return None          # cruise-stride diagonal budget (v3 fix 2)
    half = (a["size"] + b["size"]) / 2
    gap = math.hypot(max(0.0, dx - half), max(0.0, dy - half))
    if gap > L_MAX + 1e-9:
        return None
    return dx, dy, dz, gap


def edge_cost(a, b, dx, dy, dz, gap, dy_max, size_max):
    length = math.hypot(dx, dy)
    margin_pen = gap / L_MAX + dy / dy_max + (1.0 - b["size"] / size_max)
    return W1 * length + W2 * dz + W3 * margin_pen


def astar_side(stones, side, max_reach, dy_max, size_max, cov_target):
    """Min-cost stone route strip -> corridor-end coverage for one leg side.
    Returns list of stone dicts, or None."""
    usable = [s for s in stones if side_usable(s, side)]
    start = dict(idx=-1, ix=-1, iy=-1, cx=STRIP_X, cy=0.0,
                 size=0.0, top_z=STRIP_Z)   # virtual strip node

    def succ(node):
        for s in usable:
            f = edge_feasible(node, s, max_reach, dy_max)
            if f is None:
                continue
            if node["idx"] == -1:
                # from the strip the foot stands anywhere on it: no lateral
                # jump constraint, only reach-in-x, |dz| vs strip z, side band
                dx, dy, dz, gap = f
                gap = max(0.0, dx - s["size"] / 2)   # x-gap past strip edge
                if gap > L_MAX:
                    continue
                yield s, edge_cost(node, s, dx, 0.0, dz, gap, dy_max, size_max)
            else:
                yield s, edge_cost(node, s, *f, dy_max, size_max)

    def h(node):                                  # admissible: any goal stone
        return W1 * max(0.0, cov_target - size_max / 2 - node["cx"])

    openq = [(h(start), -1)]
    g = {-1: 0.0}
    came = {}
    nodes = {-1: start, **{s["idx"]: s for s in usable}}
    closed = set()
    while openq:
        fcost, i = heapq.heappop(openq)
        if i in closed:
            continue
        closed.add(i)
        node = nodes[i]
        if i != -1 and node["cx"] + node["size"] / 2 >= cov_target - 1e-9:
            route, j = [], i
            while j != -1:
                route.append(nodes[j])
                j = came[j]
            return route[::-1]
        for s, c in succ(node):
            ng = g[i] + c
            if ng < g.get(s["idx"], float("inf")) - 1e-12:
                g[s["idx"]] = ng
                came[s["idx"]] = i
                heapq.heappush(openq, (ng + h(s), s["idx"]))
    return None


def interleave(route_R, route_L):
    """Per-leg footfall sequence, gait order walk(HR,FR,HL,FL) per cycle:
    front leads on route[k], hind follows on route[k-1] (plan convention)."""
    rows, order = [], 0
    n_cyc = max(len(route_R), len(route_L))
    for k in range(n_cyc):
        for leg, route in (("HR", route_R), ("FR", route_R),
                           ("HL", route_L), ("FL", route_L)):
            j = k - 1 if leg[0] == "H" else k
            if 0 <= j < len(route):
                s = route[j]
                rows.append((order, leg, s["idx"], s["cx"], s["cy"], s["top_z"]))
                order += 1
    return rows


def main():
    with open(os.path.join(HERE, "trapv3_meta.json")) as f:
        meta = {m["level"]: m for m in json.load(f)["levels"]}

    per_level = []
    print(f"{'lvl':>3} {'found':>5} {'nsteps':>6} {'nstones':>7} {'maxdz':>7} "
          f"{'overlapR':>8} {'overlapL':>8} {'overlap':>8} {'traps':>5} "
          f"{'cov_x':>6} {'fieldmax':>8} {'cov_ok':>6}")
    for lvl in range(NUM_LEVELS):
        stones = load_stones(lvl)
        size_max = max(s["size"] for s in stones)
        max_reach = min(DX_CAP, meta[lvl]["stride_budget"])
        dy_max = DY_MAX
        field_max_cov = max(s["cx"] + s["size"] / 2 for s in stones)
        cov_target = min(COVER_X, field_max_cov)

        routes = {}
        for side in ("R", "L"):
            r = astar_side(stones, side, max_reach, dy_max, size_max,
                           cov_target)
            assert r is not None, (
                f"L{lvl} side {side}: A* found no route — bug (golden path "
                f"guarantees existence)")
            routes[side] = r

        # ── validation ──
        max_dz = 0.0
        for side in ("R", "L"):
            zs = [STRIP_Z] + [s["top_z"] for s in routes[side]]
            for a, b in zip(zs, zs[1:]):
                max_dz = max(max_dz, abs(b - a))
                assert abs(b - a) <= DZ_MAX + 1e-12, \
                    f"L{lvl} side {side}: |dz|={abs(b-a):.3f} > {DZ_MAX}"
        cov_x = min(r[-1]["cx"] + r[-1]["size"] / 2
                    for r in routes.values())
        assert cov_x >= cov_target - 1e-9, \
            f"L{lvl} coverage {cov_x:.3f} < {cov_target:.3f}"

        used = {s["idx"] for r in routes.values() for s in r}
        path_set = set(meta[lvl]["path_idx"])
        trap_set = set(meta[lvl]["trap_idx"])
        ovl = {side: 100.0 * sum(s["idx"] in path_set for s in routes[side])
               / len(routes[side]) for side in ("R", "L")}
        ovl_all = 100.0 * len(used & path_set) / len(used)
        n_trap_used = len(used & trap_set)

        rows = interleave(routes["R"], routes["L"])
        out = os.path.join(HERE, f"plan_astarv3_L{lvl}.csv")
        with open(out, "w") as f:
            f.write("order,leg,stone_idx,cx,cy,top_z\n")
            for (o, leg, idx, cx, cy, tz) in rows:
                f.write(f"{o},{leg},{idx},{cx:.10g},{cy:.10g},{tz:.10g}\n")

        print(f"{lvl:>3} {'yes':>5} {len(rows):>6} {len(used):>7} "
              f"{max_dz:>7.4f} {ovl['R']:>7.1f}% {ovl['L']:>7.1f}% "
              f"{ovl_all:>7.1f}% {n_trap_used:>5} {cov_x:>6.3f} "
              f"{field_max_cov:>8.3f} {'OK' if cov_x >= cov_target else 'NO':>6}")
        per_level.append(dict(
            level=lvl, route_found=True, n_footfalls=len(rows),
            n_stones_used=len(used), route_len_R=len(routes["R"]),
            route_len_L=len(routes["L"]), max_abs_dz=round(max_dz, 4),
            overlap_R_pct=round(ovl["R"], 1), overlap_L_pct=round(ovl["L"], 1),
            overlap_golden_pct=round(ovl_all, 1), n_trap_stones_used=n_trap_used,
            coverage_x=round(cov_x, 3), field_max_coverage=round(field_max_cov, 3),
            coverage_ok=cov_x >= cov_target - 1e-9))
    return per_level


if __name__ == "__main__":
    main()
