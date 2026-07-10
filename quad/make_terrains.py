#!/usr/bin/env python3
# 지형 테스트 씬 생성기 — 로봇 MJCF(quad_real_17dof_waist_sphere.mjcf)를 <include>로 재사용하고
#   지형(계단/험지/마찰)만 얹는다. trot_view/trot_sim 에 이 씬 경로를 argv1로 주면 로드.
#   로봇은 원점(0,0)서 +x로 전진 → 지형은 +x 앞쪽에 배치.
#   개별 씬(stairs/rough/friction) + 종합코스(course: 마찰→험지→계단 순차).
import math, os
ROBOT = "quad_real_17dof_waist_sphere.mjcf"
HDR = f'<mujoco model="{{name}}">\n  <include file="{ROBOT}"/>\n  <worldbody>\n'
FTR = "  </worldbody>\n</mujoco>\n"

def box(name, sx, sy, sz, x, y, z, rgba, fric=None, prio=None, sref=None, simp=None):
    f = f' friction="{fric}"' if fric else ""
    p = f' priority="{prio}"' if prio is not None else ""   # ★priority 지정 시 이 geom friction이 접촉을 지배(max규칙 무시)
    r = f' solref="{sref}"' if sref else ""                 # ★solref/solimp=접촉 강성(물렁한 지형=발 침하)
    m = f' solimp="{simp}"' if simp else ""
    return (f'    <geom name="{name}" type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" '
            f'pos="{x:.3f} {y:.3f} {z:.3f}" rgba="{rgba}"{f}{p}{r}{m}/>\n')

# ── 지형 빌더(geom 문자열만 반환, x0=시작 x, 접두=이름 유일화) ──────────
def g_stairs(x0, pfx="s", rise=0.15, depth=0.35, yc=0.0):
    wid=1.4; N=6   # ★계단씬 기본 15cm(가파름=점프용). course는 rise=0.05로 완만(walk 등반)
    c1="0.55 0.45 0.38 1"; c2="0.62 0.52 0.44 1"; s=""; x=x0
    for i in range(1,N+1):
        s+=box(f"{pfx}up{i}", depth/2, wid/2, i*rise/2, x+depth/2, yc, i*rise/2, c1 if i%2 else c2); x+=depth
    top=N*rise
    s+=box(f"{pfx}land", 0.6, wid/2, top/2, x+0.6, yc, top/2, "0.50 0.42 0.36 1"); x+=1.2
    for i in range(1,N+1):
        h=max(0.001,(N-i)*rise)
        s+=box(f"{pfx}dn{i}", depth/2, wid/2, h/2, x+depth/2, yc, h/2, c2 if i%2 else c1); x+=depth
    return s, x   # x=지형 끝

def g_rough(x0, span=2.4, pfx="r", yc=0.0):
    dx=0.26; dy=0.26; y0,y1=-1.0,1.0; bs=0.12; s=""
    nx=int(span/dx); ny=int((y1-y0)/dy)
    for i in range(nx):
        for j in range(ny):
            x=x0+i*dx; y=yc+y0+j*dy
            h=max(0.012, 0.055+0.045*math.sin(i*1.7+0.6)*math.cos(j*2.3+1.1))
            g=0.40+0.25*(h/0.10)
            s+=box(f"{pfx}{i}_{j}", bs, bs, h/2, x, y, h/2, f"{g:.2f} {g*0.9:.2f} {g*0.8:.2f} 1")
    return s, x0+span

def g_friction(x0, pfx="f", yc=0.0):
    # ★발 sphere friction=1.3인데 MuJoCo는 접촉 friction=두 geom의 max → 지형 priority=1로 지형값이 지배(안 그러면 얼음도 1.3라 안 미끄러움)
    # 기본바닥 μ=1.0. 만나는 순서=1.0(바닥)→0.5→0.3→0.1(점점 미끄러움).
    wid=1.6; llen=0.6; s=""; x=x0+llen
    lanes=[("f2","0.50 0.02 0.001",   "0.60 0.62 0.66 1"),  # μ0.5(보통, 회색)
           ("f1","0.30 0.01 0.0005",  "0.62 0.72 0.85 1"),  # μ0.3(미끌, 연파랑)
           ("f0","0.10 0.005 0.0001", "0.45 0.80 0.98 1")]  # μ0.1(빙판, 파랑)
    for nm,mu,rgba in lanes:
        s+=box(nm, llen, wid/2, 0.012, x, yc, 0.012, rgba, fric=mu, prio=1); x+=2*llen
    return s, x

def g_soft(x0, pfx="d", yc=0.0):
    # 저강성(매트리스/이불/모래 근사) — soft contact(solref 큰 time_const·solimp 낮음)로 발이 눌려 침하.
    # ★탄성 복원(스프링)이라 소성 자국유지는 아님. "물렁·꿀렁" 균형 강건성 테스트용. 발 관통 방지 위해 pad 두껍게(0.16).
    wid=1.6; L=1.0; h=0.05; s=""; x=x0+L
    pads=[("d0","0.04 1","0.6 0.85 0.02", "0.72 0.55 0.85 1"),  # 약간 물렁(스펀지)
          ("d1","0.12 1","0.3 0.65 0.03", "0.85 0.45 0.72 1")]  # 매우 물렁(푹 빠짐)
    for nm,sref,simp,rgba in pads:
        s+=box(nm, L, wid/2, h, x, yc, h, rgba, sref=sref, simp=simp); x+=2*L
    return s, x

def g_gap(x0, pfx="g", yc=0.0):
    # 갭 건너기: 높은 발판 사이 빈 공간(아래 바닥으로 드롭). 제어기 실패 허용 난이도.
    wid=1.4; h=0.16; plat=0.7; gap=0.32; N=3; s=""; x=x0
    c1="0.42 0.48 0.58 1"; c2="0.48 0.54 0.64 1"
    for i in range(N):
        s+=box(f"{pfx}p{i}", plat/2, wid/2, h/2, x+plat/2, yc, h/2, c1 if i%2 else c2); x+=plat
        if i<N-1: x+=gap                       # 갭(발판 없음)
    return s, x

def g_stepping(x0, pfx="t", yc=0.0):
    # 스테핑 스톤: 좌우 교대 디딤돌(사이 빈 공간). 정밀 발배치 요구, 실패 허용.
    h=0.13; sz=0.13; dx=0.44; N=6; s=""; x=x0
    c="0.38 0.55 0.44 1"
    for i in range(N):
        s+=box(f"{pfx}{i}L", sz, sz, h/2, x, yc+0.22, h/2, c)   # 좌 디딤돌
        s+=box(f"{pfx}{i}R", sz, sz, h/2, x, yc-0.22, h/2, c)   # 우 디딤돌
        x+=dx
    return s, x+0.2

def scene(name, *parts):
    return HDR.format(name=name)+"".join(parts)+FTR

out=os.path.dirname(os.path.abspath(__file__))
files={}
# 개별 씬
files["quad_terrain_stairs.mjcf"]   = scene("terrain_stairs",   g_stairs(1.2)[0])
files["quad_terrain_rough.mjcf"]    = scene("terrain_rough",    g_rough(1.0, span=3.2)[0])
files["quad_terrain_friction.mjcf"] = scene("terrain_friction", g_friction(1.1)[0])
files["quad_terrain_gap.mjcf"]      = scene("terrain_gap",      g_gap(1.2)[0])
files["quad_terrain_stepping.mjcf"] = scene("terrain_stepping", g_stepping(1.2)[0])
files["quad_terrain_soft.mjcf"]     = scene("terrain_soft",     g_soft(1.2)[0])
# 종합 코스: 저강성·마찰·험지·갭·스테핑·계단을 y축 병렬 레인(각 x0=1.2서 시작) → 조향해 골라 진입(뒤까지 안 걸어도 됨)
X0=1.2; LY=2.6   # 레인 y간격(폭 겹침 없이)
gd,xd = g_soft(X0,                 yc=-3*LY)   # 저강성(매트리스), 마찰 옆
gf,xf = g_friction(X0,             yc=-2*LY)
gr,xr = g_rough(X0, span=2.4,      yc=-1*LY)
gg,xg = g_gap(X0,                  yc= 0*LY)
gt,xt = g_stepping(X0,             yc=+1*LY)
gs,xs = g_stairs(X0, rise=0.05, depth=0.28, yc=+2*LY)
files["quad_terrain_course.mjcf"]   = scene("terrain_course", gd, gf, gr, gg, gt, gs)

for fn,txt in files.items():
    p=os.path.join(out,fn); open(p,"w").write(txt)
    print(f"생성: {fn} ({txt.count('<geom')} geoms)")
print(f"★ course 병렬 레인(y): 저강성 {-3*LY:+.1f} · 마찰 {-2*LY:+.1f} · 험지 {-1*LY:+.1f} · 갭 0.0 · 스테핑 {+1*LY:+.1f} · 계단 {+2*LY:+.1f} (모두 x={X0}서 시작, 조향 진입)")
