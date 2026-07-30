import pinocchio as pin, numpy as np, sys
URDF = "/home/jsh/문서/jsh/simulation/quad/ocs2_02leg/urdf/02leg_ocs2.urdf"
TARGET_BASE_Z = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
model = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
data = model.createData()
feet = ["FL_foot_contact_link", "FR_foot_contact_link", "HL_foot_contact_link", "HR_foot_contact_link"]
fid = [model.getFrameId(f) for f in feet]
# 시드 자세(현 nominal, task.info 순서). 발목 앞-0.5·뒤-0.3.
seed = {"FL_hip_joint":0.0,"FL_thigh_joint":0.180,"FL_calf_joint":-0.285,"FL_foot_joint":-0.5,
        "FR_hip_joint":0.0,"FR_thigh_joint":0.180,"FR_calf_joint":-0.285,"FR_foot_joint":-0.5,
        "HL_hip_joint":0.0,"HL_thigh_joint":-0.212,"HL_calf_joint":0.332,"HL_foot_joint":-0.3,
        "HR_hip_joint":0.0,"HR_thigh_joint":-0.212,"HR_calf_joint":0.332,"HR_foot_joint":-0.3}
order = ["FL_hip_joint","FL_thigh_joint","FL_calf_joint","FL_foot_joint",
         "FR_hip_joint","FR_thigh_joint","FR_calf_joint","FR_foot_joint",
         "HL_hip_joint","HL_thigh_joint","HL_calf_joint","HL_foot_joint",
         "HR_hip_joint","HR_thigh_joint","HR_calf_joint","HR_foot_joint"]
qidx = {jn: model.joints[model.getJointId(jn)].idx_q for jn in order}
vidx = {jn: model.joints[model.getJointId(jn)].idx_v for jn in order}
ank = {jn: (-0.5 if "F" in jn.split("_")[0] else -0.3) for jn in order if "foot" in jn}  # 발목 posture 목표
q = pin.neutral(model)
for jn, v in seed.items(): q[qidx[jn]] = v
# 발 xy 목표 = 현 base 0.45 config의 발위치, z=0(지면)
q[2] = 0.45; pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
tgt = [data.oMf[fid[i]].translation.copy() for i in range(4)]
for t in tgt: t[2] = 0.0
# base를 목표높이로, ★발목 고정(-0.5/-0.3), hip/thigh/calf 3관절만 IK(여분 없음)
q[2] = TARGET_BASE_Z
solveJ = [jn for jn in order if "foot" not in jn]  # 12관절(발목 제외)
jv = [vidx[jn] for jn in solveJ]
for it in range(300):
    pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
    err = np.concatenate([data.oMf[fid[i]].translation - tgt[i] for i in range(4)])  # 12
    if np.linalg.norm(err) < 1e-6: break
    Jrows = []
    for i in range(4):
        J = pin.computeFrameJacobian(model, data, q, fid[i], pin.LOCAL_WORLD_ALIGNED)[:3]
        Jrows.append(J[:, jv])  # 3 x 12
    Jj = np.vstack(Jrows)  # 12 x 12
    Jpinv = Jj.T @ np.linalg.inv(Jj @ Jj.T + 1e-6*np.eye(12))
    dq = -0.3 * (Jpinv @ err)  # 작은 스텝(DLS)=seed 브랜치 유지
    for k, jn in enumerate(solveJ): q[qidx[jn]] += dq[k]  # 발목은 고정(업데이트 안함)
pin.forwardKinematics(model, data, q); pin.updateFramePlacements(model, data)
print(f"# base_z={TARGET_BASE_Z}, 수렴 |err|={np.linalg.norm(np.concatenate([data.oMf[fid[i]].translation-tgt[i] for i in range(4)])):.2e}")
print("# 발 z(지면=0):", [f"{data.oMf[fid[i]].translation[2]:+.4f}" for i in range(4)])
for jn in order:
    print(f"  {jn:16s} {q[qidx[jn]]:+.4f}")
