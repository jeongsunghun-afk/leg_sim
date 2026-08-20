#!/usr/bin/env python3
"""모델 총질량·바디별 질량과 **정적 stand 자세에서 축별 중력토크**를 뽑는다.
   "주저앉는다" 가 질량 탓인지 토크 실현 탓인지 가르려면 먼저 기준값이 필요하다."""
import sys, os
import mujoco, numpy as np

mjcf = sys.argv[1]
m = mujoco.MjModel.from_xml_path(mjcf)
d = mujoco.MjData(m)

print(f"모델: {os.path.basename(mjcf)}   nq={m.nq} nv={m.nv} nu={m.nu}")
print(f"★총질량 = {m.body_mass.sum():.4f} kg\n")
print(f"  {'body':<28}{'mass[kg]':>10}")
for i in range(m.nbody):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "(world)"
    if m.body_mass[i] > 0:
        print(f"  {nm:<28}{m.body_mass[i]:>10.4f}")

# 평발 stand 자세에서 중력토크 (Qflat8, biped_control.hpp:28)
QFLAT = [0, 0.064256, -0.416657, -1.043858, 0, 0.064256, -0.416657, -1.043858]
NJ = m.nq - 7
if NJ == len(QFLAT):
    d.qpos[:] = 0; d.qpos[3] = 1
    d.qpos[7:] = QFLAT
    d.qpos[2] = 0.7
    mujoco.mj_forward(m, d)
    # 발 최저점을 지면에 맞춤
    zmin = 1e9
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "sphere" in nm:
            zmin = min(zmin, d.geom_xpos[g][2] - m.geom_size[g][0])
    d.qpos[2] -= zmin
    d.qvel[:] = 0; d.qacc[:] = 0
    mujoco.mj_inverse(m, d)      # 정지 상태 유지에 필요한 일반화력
    names = ["HL_hip","HL_thigh","HL_calf","HL_foot","HR_hip","HR_thigh","HR_calf","HR_foot"]
    print(f"\n  평발 stand 자세 · **공중(접촉 없음)** 중력 유지토크")
    print(f"  {'joint':<12}{'tau[Nm]':>10}")
    for j, nm in enumerate(names):
        print(f"  {nm:<12}{d.qfrc_inverse[6+j]:>10.3f}")
    print(f"\n  base z = {d.qpos[2]:.4f} m   (발 접지 기준)")
    print(f"  ⚠접촉이 없으므로 이건 **다리를 공중에 든 채** 유지하는 토크다.")
    print(f"    실제 stand 는 지면반력이 체중을 받으므로 관절토크는 이보다 작다.")
