#!/usr/bin/env python3
# 지형 테스트 씬 생성기 — 로봇 MJCF(quad_real_17dof_waist_sphere.mjcf)를 <include>로 재사용하고
#   지형(계단/험지/마찰)만 얹는다. trot_view/trot_sim 에 이 씬 경로를 argv1로 주면 로드.
#   로봇은 원점(0,0)서 +x로 전진 → 지형은 +x 앞쪽에 배치.
import math, os
ROBOT = "quad_real_17dof_waist_sphere.mjcf"
HDR = f'<mujoco model="{{name}}">\n  <include file="{ROBOT}"/>\n  <worldbody>\n'
FTR = "  </worldbody>\n</mujoco>\n"

def box(name, sx, sy, sz, x, y, z, rgba, fric=None, euler=None):
    f = f' friction="{fric}"' if fric else ""
    e = f' euler="{euler}"' if euler else ""
    return (f'    <geom name="{name}" type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" '
            f'pos="{x:.3f} {y:.3f} {z:.3f}" rgba="{rgba}"{f}{e}/>\n')

# ── 1) 계단 (오름 → 랜딩 → 내림) ─────────────────────────────
def stairs():
    s = HDR.format(name="terrain_stairs"); rise=0.04; depth=0.28; wid=1.4; x0=1.2; N=6
    col1="0.55 0.45 0.38 1"; col2="0.62 0.52 0.44 1"
    x=x0
    for i in range(1,N+1):   # 오름: 각 계단=바닥~i*rise 솔리드 블록
        s+=box(f"up{i}", depth/2, wid/2, i*rise/2, x+depth/2, 0, i*rise/2, col1 if i%2 else col2)
        x+=depth
    top=N*rise
    s+=box("landing", 0.6, wid/2, top/2, x+0.6, 0, top/2, "0.50 0.42 0.36 1"); x+=1.2
    for i in range(1,N+1):   # 내림
        h=(N-i)*rise if (N-i)>0 else 0.001
        s+=box(f"dn{i}", depth/2, wid/2, h/2, x+depth/2, 0, h/2, col2 if i%2 else col1)
        x+=depth
    return s+FTR

# ── 2) 험지 (불규칙 높이 블록 필드) ─────────────────────────
def rough():
    s = HDR.format(name="terrain_rough")
    x0,x1,dx = 1.0, 4.2, 0.26
    y0,y1,dy = -1.0, 1.0, 0.26
    bs = 0.12   # 블록 반경(간격 있게)
    nx = int((x1-x0)/dx); ny=int((y1-y0)/dy)
    for i in range(nx):
        for j in range(ny):
            x=x0+i*dx; y=y0+j*dy
            # 결정론적 의사난수 높이 0.01~0.10 (관통 방지 위해 바닥~top 솔리드)
            h=0.055+0.045*math.sin(i*1.7+0.6)*math.cos(j*2.3+1.1)
            h=max(0.012,h)
            g=0.40+0.25*(h/0.10)   # 높이에 따라 밝기
            s+=box(f"r{i}_{j}", bs, bs, h/2, x, y, h/2, f"{g:.2f} {g*0.9:.2f} {g*0.8:.2f} 1")
    return s+FTR

# ── 3) 마찰 지형 (저마찰 ice / 보통 / 고마찰 레인) ───────────
def friction():
    s = HDR.format(name="terrain_friction")
    wid=1.6
    lanes=[  # (x중심, 길이반경, μtan, rgba, 라벨)
        (1.6, 0.5, "0.20 0.02 0.001", "0.45 0.70 0.95 1"),   # ice(저마찰) 하늘색
        (2.7, 0.5, "1.30 0.02 0.001", "0.55 0.57 0.62 1"),   # 보통 회색
        (3.8, 0.5, "2.50 0.05 0.002", "0.90 0.55 0.30 1"),   # 고마찰 주황
    ]
    for k,(x,lx,mu,rgba) in enumerate(lanes):
        s+=box(f"fric{k}", lx, wid/2, 0.012, x, 0, 0.012, rgba, fric=mu)
    return s+FTR

out=os.path.dirname(os.path.abspath(__file__))
for fn,gen in [("quad_terrain_stairs.mjcf",stairs),
               ("quad_terrain_rough.mjcf",rough),
               ("quad_terrain_friction.mjcf",friction)]:
    p=os.path.join(out,fn); open(p,"w").write(gen())
    print(f"생성: {fn} ({open(p).read().count('<geom')} geoms)")
