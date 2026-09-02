#!/usr/bin/env python3
"""stepping_go2/gen_stones.py — per-level stone tables for the Go2 varied-height
stepping-stone field (GO2_STONE_HVAR=0.05), lane-local coords.

Mirrors _build_stepping_terrain_curriculum (go2_wtw_env.py) EXACTLY:
  * cfg: gap_depth=0.15, stone 0.40->0.10, gap 0.02->0.18, num_levels=10,
    corridor_len=3.5, lane_pitch=3.0 -> lane_width=max(1.2, 3.0-1.0)=2.0
  * layout: spawn strip x in [-0.75, 0.75] (top z=depth), stones begin at
    x_stone0=0.75; per level: pitch=size+gap, nx=int(3.5//pitch),
    ny=int(2.0//pitch), cx=0.75+(ix+0.5)*pitch, cy=y0+iy*pitch with
    y0 = -0.5*(ny-1)*pitch (LANE-LOCAL: y relative to the lane center y_lane;
    world y = cy + lvl*lane_pitch)
  * heights [HVAR]: ONE np.random.RandomState(0); one uniform(-a,+a) draw per
    stone in (level, ix, iy) order (drawn even when a=0); a = max_amp*lvl/9;
    stone top z = depth + dz.  Same RNG stream as the env builder AND as the
    verified /tmp/hvar_replica.py on the server.

Output: stones_L{0..9}.csv  (idx,ix,iy,cx,cy,size,top_z)  + stones_meta.json.
Prints per-level stats; L0 must be flat 0.15 and L9 must show z 0.1046..0.1994
with max grid-adjacent dz 0.0845 (replica ground truth).
"""
import json
import os

import numpy as np

MAX_AMP = 0.05          # GO2_STONE_HVAR
DEPTH = 0.15            # gap_depth = nominal stone-top surface z (void plane = z 0)
SIZE_MAX, SIZE_MIN = 0.40, 0.10
GAP_MIN, GAP_MAX = 0.02, 0.18
NUM_LEVELS = 10
CORRIDOR_LEN = 3.5
LANE_PITCH = 3.0
LANE_WIDTH = max(1.2, LANE_PITCH - 1.0)   # 2.0
PLAT_LEN = 1.5
X_PLAT0 = -0.75                           # spawn strip [-0.75, 0.75], spawn at x=0
X_STONE0 = X_PLAT0 + PLAT_LEN             # 0.75
X_STONE_END = X_STONE0 + CORRIDOR_LEN     # 4.25

OUT = os.path.dirname(os.path.abspath(__file__))

rng = np.random.RandomState(0)            # one stream shared across ALL levels
rows_per_level = {}
meta_levels = []
print(f"{'lvl':>3} {'size':>5} {'gap':>5} {'pitch':>6} {'nx*ny':>6} {'n':>4} "
      f"{'z_min':>7} {'z_max':>7} {'adj_dx':>7} {'adj_dy':>7} {'adj_max':>8}")
for lvl in range(NUM_LEVELS):
    frac = lvl / (NUM_LEVELS - 1)
    size = SIZE_MAX + (SIZE_MIN - SIZE_MAX) * frac
    gap = GAP_MIN + (GAP_MAX - GAP_MIN) * frac
    pitch = size + gap
    amp = MAX_AMP * frac
    assert gap >= 1e-3, "all 10 levels have discrete stones with these cfg values"
    nx = max(1, int(CORRIDOR_LEN // pitch))
    ny = max(1, int(LANE_WIDTH // pitch))
    y0 = -0.5 * (ny - 1) * pitch          # lane-local (y_lane subtracted)
    rows = []
    tops = np.empty((nx, ny))
    for ix in range(nx):                  # (level, ix, iy) RNG order = builder order
        cx = X_STONE0 + (ix + 0.5) * pitch
        for iy in range(ny):
            cy = y0 + iy * pitch
            dz = float(rng.uniform(-amp, amp))   # drawn even when amp == 0 (lvl 0)
            top = DEPTH + dz
            tops[ix, iy] = top
            rows.append((len(rows), ix, iy, cx, cy, size, top))
    rows_per_level[lvl] = rows
    dx = float(np.abs(np.diff(tops, axis=0)).max()) if nx > 1 else 0.0
    dy = float(np.abs(np.diff(tops, axis=1)).max()) if ny > 1 else 0.0
    print(f"{lvl:>3} {size:5.2f} {gap:5.2f} {pitch:6.3f} {nx:>2}x{ny:<3} {nx*ny:>4} "
          f"{tops.min():7.4f} {tops.max():7.4f} {dx:7.4f} {dy:7.4f} {max(dx, dy):8.4f}")
    with open(os.path.join(OUT, f"stones_L{lvl}.csv"), "w") as f:
        f.write("idx,ix,iy,cx,cy,size,top_z\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.10g},{r[4]:.10g},{r[5]:.10g},{r[6]:.10g}\n")
    meta_levels.append(dict(level=lvl, size=size, gap=gap, pitch=pitch, nx=nx, ny=ny,
                            amp=amp, n_stones=nx * ny,
                            z_min=float(tops.min()), z_max=float(tops.max()),
                            max_adj_dz=max(dx, dy)))

with open(os.path.join(OUT, "stones_meta.json"), "w") as f:
    json.dump(dict(
        frame=("lane-local: x = corridor coord (spawn strip [-0.75,0.75] top z=0.15, "
               "spawn at x=0, stones span [0.75,4.25]); y relative to lane center "
               "(world y = cy + level*lane_pitch); z absolute (void plane z=0, "
               "stone top z = 0.15 + dz)"),
        max_amp=MAX_AMP, depth=DEPTH, size_max=SIZE_MAX, size_min=SIZE_MIN,
        gap_min=GAP_MIN, gap_max=GAP_MAX, num_levels=NUM_LEVELS,
        corridor_len=CORRIDOR_LEN, lane_pitch=LANE_PITCH, lane_width=LANE_WIDTH,
        plat_len=PLAT_LEN, x_plat0=X_PLAT0, x_stone0=X_STONE0, x_stone_end=X_STONE_END,
        rng="np.random.RandomState(0), one uniform(-a,+a) per stone, (level,ix,iy) order",
        levels=meta_levels), f, indent=1)

# ── ground-truth check vs the verified replica (same RNG stream re-derived) ──
rng2 = np.random.RandomState(0)
ok = True
for lvl in range(NUM_LEVELS):
    frac = lvl / (NUM_LEVELS - 1)
    size = SIZE_MAX + (SIZE_MIN - SIZE_MAX) * frac
    gap = GAP_MIN + (GAP_MAX - GAP_MIN) * frac
    pitch = size + gap
    amp = MAX_AMP * frac
    nx, ny = max(1, int(CORRIDOR_LEN // pitch)), max(1, int(LANE_WIDTH // pitch))
    ref = np.array([[DEPTH + float(rng2.uniform(-amp, amp)) for _ in range(ny)]
                    for _ in range(nx)])
    got = np.array([r[6] for r in rows_per_level[lvl]]).reshape(nx, ny)
    if not np.array_equal(ref, got):
        ok = False
        print(f"  MISMATCH at level {lvl}")
print("replica RNG-stream check:", "IDENTICAL" if ok else "MISMATCH")
