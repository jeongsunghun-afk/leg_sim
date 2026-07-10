#!/usr/bin/env python3
# J0.1 — 점프 타당성 상한 계산 (RPET_JUMP_MPC.md §1.1) · 우리 실제 모델(38kg) 기준
#   물리: 토크제한 수직 GRF → 추진 스트로크 에너지 → 이륙속도 v_z → 높이 h=v_z²/2g.
#   ankle ω 한계도 병목으로 반영. crouch z0 × 신전 스트로크 그리드.
#   ★go/no-go 게이트: 상한 h<0.05m면 하드웨어 선행(문서 §1.3).
import mujoco, numpy as np, os

QUAD = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # offline/jump/ → quad/
MJCF = os.path.join(QUAD, 'mjcf', 'quad_real_17dof_waist_sphere.mjcf')
m = mujoco.MjModel.from_xml_path(MJCF); d = mujoco.MjData(m)
g = 9.81; M = float(m.body_subtreemass[0]); mg = M*g

# 관절 peak 토크(=actuatorfrcrange, 8:1 재기어는 foot만 96)·속도한계 w_limit
GEAR = {'hip':7.,'thigh':7.,'calf':10.5,'foot':14.*0.5714}   # foot 8:1
def jinfo(name):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return m.jnt_range[jid], m.jnt_dofadr[jid]
# peak 토크: MJCF actuatorfrcrange (foot는 재기어로 96)
PEAK = {'hip':84.,'thigh':84.,'calf':126.,'foot':168.*0.5714}   # =96
WLIM = {k: 207./GEAR[k] for k in GEAR}                          # 무부하 속도한계[rad/s]

fgid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f+'_sphere') for f in ['HL','HR','FL','FR']]
foot_r = [m.geom_size[gid][0] for gid in fgid]
legs = ['HL','HR','FL','FR']
def leg_dofs(L):  # 다리 L의 관절 dof adr + 종류
    out=[]
    for jn in ['hip','thigh','calf','foot']:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f'{L}_{jn}_joint')
        if jid>=0: out.append((m.jnt_dofadr[jid], jn))
    return out

def crouch_pose(base_z):
    """base_z 높이 crouch: 발 XY 유지 무릎굽힘 IK (crouch_home 축약)."""
    if m.nkey>0: mujoco.mj_resetDataKeyframe(m,d,0)
    else: d.qpos[:]=0; d.qpos[3]=1
    d.qpos[2]=0.55; mujoco.mj_forward(m,d)
    foot_xy=[d.geom_xpos[fgid[i]][:2].copy() for i in range(4)]
    d.qpos[2]=base_z
    for _ in range(200):
        mujoco.mj_kinematics(m,d); mujoco.mj_comPos(m,d)
        for i in range(4):
            p=d.geom_xpos[fgid[i]].copy(); p[2]-=foot_r[i]
            e=np.array([foot_xy[i][0],foot_xy[i][1],0.0])-p
            jp=np.zeros((3,m.nv)); mujoco.mj_jac(m,d,jp,None,d.geom_xpos[fgid[i]],
                mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,f'{legs[i]}_calf_link') if False else fgid[i]*0+ m.geom_bodyid[fgid[i]])
            cols=[da for da,_ in leg_dofs(legs[i])]
            J=jp[:,cols]
            dq=0.5*J.T@np.linalg.solve(J@J.T+1e-4*np.eye(3), e)
            for k,da in enumerate(cols): d.qpos[7+da-6+ (da-da)] = d.qpos[m.jnt_qposadr[0]]  # noop guard
            qadr=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'{legs[i]}_{jn}_joint')] for _,jn in leg_dofs(legs[i])]
            for k,qa in enumerate(qadr): d.qpos[qa]+=dq[k]
    mujoco.mj_forward(m,d)

def max_vertical_grf():
    """현 자세에서 4발이 낼 수 있는 총 수직 GRF (관절토크 한계)."""
    Ftot=0.0
    for i in range(4):
        jp=np.zeros((3,m.nv)); mujoco.mj_jac(m,d,jp,None,d.geom_xpos[fgid[i]],m.geom_bodyid[fgid[i]])
        dofs=leg_dofs(legs[i]); cols=[da for da,_ in dofs]; kinds=[jn for _,jn in dofs]
        Jz=jp[2,cols]   # 수직력→관절토크 매핑행: τ_i = Jz_i * Fz
        # Fz 최대 = min_i(peak_i/|Jz_i|)
        lim=[]
        for k,jn in enumerate(kinds):
            if abs(Jz[k])>1e-6: lim.append(PEAK[jn]/abs(Jz[k]))
        if lim: Ftot += min(lim)
    return Ftot

def max_vz_by_omega(base_z):
    """신전 속도 상한: 관절 ω한계로 낼 수 있는 수직 발끝속도(=이륙 v_z 상한)."""
    vmax=1e9
    for i in range(4):
        jp=np.zeros((3,m.nv)); mujoco.mj_jac(m,d,jp,None,d.geom_xpos[fgid[i]],m.geom_bodyid[fgid[i]])
        dofs=leg_dofs(legs[i]); cols=[da for da,_ in dofs]; kinds=[jn for _,jn in dofs]
        Jz=jp[2,cols]
        # 각 관절 최대속도로 낼 수 있는 수직속도 합 (동시 최대 가정=낙관)
        vz=sum(abs(Jz[k])*WLIM[kinds[k]] for k in range(len(kinds)))
        vmax=min(vmax,vz)
    return vmax

print(f"=== J0.1 점프 타당성 (실제 모델) ===  M={M:.1f}kg  mg={mg:.0f}N")
print(f"peak τ: hip/thigh 84 · calf 126 · foot 96(8:1) Nm | ω한계 foot {WLIM['foot']:.1f} calf {WLIM['calf']:.1f}")
print(f"{'crouch z0':>9} {'스트로크Δz':>9} {'F_grf[N]':>9} {'초과/mg':>8} {'v_z(에너지)':>10} {'v_z(ω한계)':>10} {'v_z*':>7} {'h[m]':>7}")
Z_STAND=0.52; best_h=0
for z0 in [0.26,0.29,0.32,0.36,0.40]:
    crouch_pose(z0)
    F=max_vertical_grf(); dz=Z_STAND-z0
    W=max(0.0,(F-mg))*dz                         # CoM에 한 순일(중력 제외)
    vz_e=np.sqrt(2*W/M) if W>0 else 0.0          # 에너지법
    vz_w=max_vz_by_omega(z0)                       # ω 병목
    vz=min(vz_e,vz_w); h=vz**2/(2*g)
    best_h=max(best_h,h)
    print(f"{z0:>9.2f} {dz:>9.2f} {F:>9.0f} {F/mg:>8.2f} {vz_e:>10.2f} {vz_w:>10.2f} {vz:>7.2f} {h:>7.3f}")
print(f"\n★ 도달 가능 최대 높이 상한 ≈ {best_h:.3f} m")
print(f"★ go/no-go: {'GO (h≥0.05m, 점프 성립)' if best_h>=0.05 else 'NO-GO (h<0.05m → 하드웨어 선행)'}")
print("주의: 정적 토크근사·다리관성 미보정·soft contact 미반영 = 낙관적 상한. 실측은 이보다 낮음.")
