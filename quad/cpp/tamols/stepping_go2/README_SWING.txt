README_SWING.txt — per-footfall SWING TRAJECTORY references (stepping-stone fields)
generator: ../cache_gen_go2_stepping_swing.cpp  (build/run cmd in its header comment)

FILES
  swing_L{0..9}.csv         safe varied fields (stones_L*).  Footfall source = footfalls_L{lvl}.csv
                            (TAMOLS receding-horizon walk sweep, per cycle x per leg).
  swing_trapv3_L{0..9}.csv  trap v3 fields (stones_trapv3_L*).  Footfall source = plan_astarv3_L{lvl}.csv
                            (A* stone chains; each row = one footfall, per-leg sequence).
  swing_meta.json           machine-readable version of this contract.

FORMAT (CSV)
  level, leg, footfall_order, s, x, y, z
  - leg            FL | FR | HL | HR
  - footfall_order per-leg 0-based swing index (the k-th swing that leg performs)
  - s              swing phase 0..1, K=11 samples (0, 0.1, ..., 1.0) per footfall
  - x,y,z          swing foot reference at phase s

FRAME
  lane-local, identical to plan_meta.json / stones CSVs:
  x = corridor coordinate (spawn strip [-0.75, 0.75] top z=0.15; stones x in [0.75, 4.25])
  y = relative to lane center (world y = y + level*3.0)
  z = absolute (void plane = 0)

LIFTOFF / TOUCHDOWN RULE
  swing k of leg L: liftoff = that leg's foothold k-1, touchdown = foothold k.
  k=0 liftoff = spawn stance: hip nominal (+-0.1934, +-0.142), z=0.15 (strip).
  Endpoints of the sampled swing match the planned footholds exactly (<1e-9).
  Caveat (trap set only): the A* chains contain no strip approach steps, so
  footfall_order 0 (especially hind legs) spans from the spawn stance to the first
  stone and can be long (~1 m). The env should treat order-0 as an entry reference
  or re-anchor its own approach gait onto footfall_order >= 1.

HOW THE ENV CONSUMES (spatial, NOT time-clocked)
  Per leg, keep a footfall counter n_L (increment at each touchdown / swing start).
  During swing, take the env's own gait-clock swing progress s in [0,1] and linearly
  interpolate the K=11 points of (leg, footfall_order=n_L) at that s -> foot pos ref.
  No reference clock is exported on purpose: references are indexed by footfall order,
  so the policy's own timing cannot drift against them (prior reference-clock fragility).

WHAT IS TAMOLS-NATIVE vs APPROXIMATED
  native:
   - footholds + sequence: the already-validated plans (TAMOLS sweep / A* chains).
   - terrain layers: h_s2 "virtual floor" from terrain_proc.hpp process_height_maps
     (median -> edge mask -> dilate -> 3x3 local max -> gaussian sigma2; sigma1=1,
     sigma2=2 cells, cell=0.02 m) — the TAMOLS layer that makes swings collision-free
     without explicit constraints.
   - xy shape: linear liftoff->touchdown interpolation (tamols_track.hpp swing form).
  approximated (the TAMOLS QP does NOT output swing splines — its decision variables
  are base spline + footholds + GIAC slack only; the stock tracker uses a fixed-height
  parabola, not terrain-aware):
   - vertical profile: two-segment cubic Hermite (zero slope at both ends and at apex);
     apex z = max(liftoff_z, touchdown_z, path floor max) + 0.08;
     apex phase = 0.5 - 0.25*tanh(dz/0.10) clamped to [0.3,0.7], snapped to the 0.1
     sample grid (step-up -> earlier apex, step-down -> later apex);
     interior samples clipped up to max(exact terrain, h_s2) + 0.025 (virtual-floor
     clearance; floor(s) evaluated along the xy line).

VALIDATION (all 20 files, violations = 0)
  - every interior sample (s=0.1..0.9): z >= exact terrain height at its xy + 0.02
    (endpoints are ON the terrain by definition, so they are exempt)
  - apex >= max(liftoff_z, touchdown_z) + 0.05  AND  >= max exact terrain along the
    xy path + 0.05 (tall stones crossed mid-path are cleared, e.g. trapv3 L6 apex
    0.63 over a 0.55 trap stone)
  - endpoints match planned footholds exactly (max err ~1e-16)
  - monotone horizontal progress along s (linear xy => exact)
