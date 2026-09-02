#!/usr/bin/env python3
"""Convert a D1 PLAN_EXPORT CSV into a clean, RL-consumable DTC reference.

Usage: make_dtc_ref.py <in.csv> <out_prefix> <stair_rise> <lane_y> [primary=fk|region] [source_tag]

Input CSV columns (from test02legMujoco PLAN_EXPORT):
  t,mode,bx,by,bz,byaw,bpitch,broll, then per foot[FL,FR,HL,HR]:
  fx_,fy_,fz_(OCS2 FK foot, world) , st_(stance 0/1) , regx_,regy_,regz_(region-snapped target, world) , valid_

Two foothold channels are ALWAYS emitted:
  fk_*     = OCS2-optimized FK foot (dynamically-feasible foot trajectory from the NMPC plan)
  region_* = region-snapped target (terrain-feasible tread-top; geometric sanity)
`primary` selects which becomes foot_world/foot_rel (the RL's main target):
  fk     -> for OCS2_ROLLOUT refs (dynamically feasible)  [DEFAULT]
  region -> for KIN_DRIVE refs (geometric)

Frames:
  world   : MuJoCo world (x forward, z up)
  base-rel: yaw-only base frame  rel = Rz(-yaw)*(foot_world_xy - base_xy),  rel_z = foot_world_z - base_z
"""
import csv, sys, math
import numpy as np

def main():
    inp, out = sys.argv[1], sys.argv[2]
    rise   = float(sys.argv[3]) if len(sys.argv) > 3 else float('nan')
    lane_y = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    primary = sys.argv[5] if len(sys.argv) > 5 else "fk"
    source = sys.argv[6] if len(sys.argv) > 6 else "ocs2_rollout"
    comH = 0.50
    legs = ["FL", "FR", "HL", "HR"]
    rows = list(csv.DictReader(open(inp)))
    f = lambda r, k: float(r[k])

    # keep strictly-advancing base_x (drop dups/pre-start)
    recs, lastx = [], -1e9
    for r in rows:
        if r.get('valid_HR') is None or r.get('bx') is None:
            continue  # skip any partially-written trailing line
        bx = f(r, 'bx')
        if bx <= lastx + 1e-5:
            continue
        lastx = bx; recs.append(r)
    N = len(recs)

    base_x=np.zeros(N); base_y=np.zeros(N); base_z=np.zeros(N)
    base_pitch=np.zeros(N); base_yaw=np.zeros(N)
    fk_world=np.zeros((N,4,3)); fk_rel=np.zeros((N,4,3))
    region_world=np.zeros((N,4,3)); region_rel=np.zeros((N,4,3))
    foot_stance=np.zeros((N,4),dtype=np.int8); foot_valid=np.zeros((N,4),dtype=np.int8)

    for i, r in enumerate(recs):
        bx,by,bz = f(r,'bx'),f(r,'by'),f(r,'bz')
        yaw,pitch = f(r,'byaw'),f(r,'bpitch')
        base_x[i],base_y[i],base_z[i]=bx,by,bz
        base_pitch[i],base_yaw[i]=pitch,yaw
        c,s = math.cos(-yaw), math.sin(-yaw)
        for j,lg in enumerate(legs):
            foot_stance[i,j]=int(r[f'st_{lg}']); foot_valid[i,j]=int(r[f'valid_{lg}'])
            fkw=(f(r,f'fx_{lg}'),f(r,f'fy_{lg}'),f(r,f'fz_{lg}'))
            rgw=(f(r,f'regx_{lg}'),f(r,f'regy_{lg}'),f(r,f'regz_{lg}')) if foot_valid[i,j] else fkw
            for (arrW,arrR,w) in [(fk_world,fk_rel,fkw),(region_world,region_rel,rgw)]:
                arrW[i,j]=w
                dx,dy=w[0]-bx,w[1]-by
                arrR[i,j]=(c*dx-s*dy, s*dx+c*dy, w[2]-bz)

    foot_world = fk_world if primary=="fk" else region_world
    foot_rel   = fk_rel   if primary=="fk" else region_rel

    np.savez(out+"_ref.npz",
             base_x=base_x,base_y=base_y,base_z=base_z,base_pitch=base_pitch,base_yaw=base_yaw,
             foot_world=foot_world,foot_rel=foot_rel,
             fk_world=fk_world,fk_rel=fk_rel,region_world=region_world,region_rel=region_rel,
             foot_stance=foot_stance,foot_valid=foot_valid,
             legs=np.array(legs),stair_rise=rise,lane_y=lane_y,com_height=comH,
             primary_channel=primary,source=source)

    with open(out+"_ref.csv","w",newline="") as fo:
        fo.write(f"# D1 OCS2 DTC reference | source={source} primary={primary} stair_rise={rise} lane_y={lane_y} comH={comH}\n")
        fo.write("# frames: world=MuJoCo(x fwd,z up); rel=yaw-only base (Rz(-yaw)*(foot-base), z=foot_z-base_z)\n")
        fo.write("# primary foothold = OCS2 FK foot (dynamically feasible) if primary=fk; region_*=terrain-snap sanity; st=stance valid=region_valid\n")
        w=csv.writer(fo)
        hdr=["base_x","base_y","base_z","base_pitch","base_yaw"]
        for lg in legs:
            hdr+=[f"{lg}_wx",f"{lg}_wy",f"{lg}_wz",f"{lg}_rx",f"{lg}_ry",f"{lg}_rz",
                  f"{lg}_fkx",f"{lg}_fky",f"{lg}_fkz",f"{lg}_regx",f"{lg}_regy",f"{lg}_regz",f"{lg}_st",f"{lg}_valid"]
        w.writerow(hdr)
        for i in range(N):
            row=[f"{base_x[i]:.4f}",f"{base_y[i]:.4f}",f"{base_z[i]:.4f}",f"{base_pitch[i]:.4f}",f"{base_yaw[i]:.4f}"]
            for j in range(4):
                row+=[f"{foot_world[i,j,0]:.4f}",f"{foot_world[i,j,1]:.4f}",f"{foot_world[i,j,2]:.4f}",
                      f"{foot_rel[i,j,0]:.4f}",f"{foot_rel[i,j,1]:.4f}",f"{foot_rel[i,j,2]:.4f}",
                      f"{fk_world[i,j,0]:.4f}",f"{fk_world[i,j,1]:.4f}",f"{fk_world[i,j,2]:.4f}",
                      f"{region_world[i,j,0]:.4f}",f"{region_world[i,j,1]:.4f}",f"{region_world[i,j,2]:.4f}",
                      int(foot_stance[i,j]),int(foot_valid[i,j])]
            w.writerow(row)

    # verification
    kpk=int(np.argmax(base_z))
    dz_asc=np.diff(base_z[:kpk+1]) if kpk>1 else np.array([0.0])
    mono_asc=float((dz_asc>=-0.003).mean())*100
    treads=sorted(set(round(float(region_world[i,j,2]),3) for i in range(N) for j in range(4) if foot_valid[i,j]))
    print(f"[{out}] primary={primary} rows={N} base_x {base_x[0]:.2f}->{base_x[-1]:.2f} "
          f"base_z {base_z.min():.3f}->{base_z.max():.3f}(peak@x={base_x[kpk]:.2f}) pitch[min={base_pitch.min()*57.3:.0f} max={base_pitch.max()*57.3:.0f}]deg")
    print(f"  ascent base_z monotonic={mono_asc:.1f}%  region tread tops hit={treads}")
    print(f"  saved {out}_ref.csv , {out}_ref.npz")

if __name__=="__main__":
    main()
