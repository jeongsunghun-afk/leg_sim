"""biped_flatfoot.mjcf → biped_gen.urdf 변환 (crocoddyl 내장 actuation 호환용).

pinocchio가 MJCF를 직접 로드하면 free-flyer를 crocoddyl이 오인식(nu=nv-1)하는 문제 →
동일 관성/운동학의 URDF를 생성해 buildModelFromUrdf 경로 사용(ci_mpc go2/02_Leg처럼 정상).
링크 관성만 필요(visual/collision 생략) — dynamics 모델 전용.

관성 변환: MuJoCo는 주관성 diag(body_inertia)+body_iquat(주축 방향) → URDF는 body 프레임
전체 텐서 필요. I_body = R(iquat)·diag·R(iquat)ᵀ, inertial origin rpy=0.
"""
import mujoco, numpy as np

MJCF = "/home/jsh/문서/jsh/simulation/biped/biped_flatfoot.mjcf"
OUT = "/home/jsh/문서/jsh/simulation/biped/ocp/biped_gen.urdf"
# 관절별 effort(peak 토크, MJCF actuatorfrcrange)·velocity 한계
EFFORT = {"hip": 84.0, "thigh": 84.0, "calf": 126.0, "foot": 168.0}
VEL = 20.0


def quat2mat(q):  # MuJoCo wxyz
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def main():
    m = mujoco.MjModel.from_xml_path(MJCF)
    name = lambda t, i: mujoco.mj_id2name(m, t, i)
    # body별 joint 매핑
    body_joint = {}
    for j in range(m.njnt):
        body_joint.setdefault(int(m.jnt_bodyid[j]), []).append(j)

    L = ['<?xml version="1.0"?>', '<robot name="biped">']
    for i in range(1, m.nbody):                     # world 제외
        bn = name(mujoco.mjtObj.mjOBJ_BODY, i)
        mass = m.body_mass[i]
        ip = m.body_ipos[i]
        R = quat2mat(m.body_iquat[i])
        Ib = R @ np.diag(m.body_inertia[i]) @ R.T
        # link
        L.append(f'  <link name="{bn}">')
        L.append(f'    <inertial>')
        L.append(f'      <origin xyz="{ip[0]:.6f} {ip[1]:.6f} {ip[2]:.6f}" rpy="0 0 0"/>')
        L.append(f'      <mass value="{mass:.6f}"/>')
        L.append(f'      <inertia ixx="{Ib[0,0]:.6e}" ixy="{Ib[0,1]:.6e}" ixz="{Ib[0,2]:.6e}"'
                 f' iyy="{Ib[1,1]:.6e}" iyz="{Ib[1,2]:.6e}" izz="{Ib[2,2]:.6e}"/>')
        L.append(f'    </inertial>')
        L.append(f'  </link>')

    for i in range(1, m.nbody):
        bn = name(mujoco.mjtObj.mjOBJ_BODY, i)
        pid = int(m.body_parentid[i])
        pn = name(mujoco.mjtObj.mjOBJ_BODY, pid)
        if pid == 0:
            continue                                # torso=root(부모=world) → 관절 없음(free-flyer는 pinocchio가 추가)
        pos = m.body_pos[i]
        jts = body_joint.get(i, [])
        if not jts:                                 # 관절 없는 body(foot_contact) = fixed
            L.append(f'  <joint name="{bn}_fixed" type="fixed">')
            L.append(f'    <origin xyz="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" rpy="0 0 0"/>')
            L.append(f'    <parent link="{pn}"/>'); L.append(f'    <child link="{bn}"/>')
            L.append(f'  </joint>')
        else:
            j = jts[0]
            jn = name(mujoco.mjtObj.mjOBJ_JOINT, j)
            ax = m.jnt_axis[j]
            lo, hi = m.jnt_range[j]
            eff = next((v for k, v in EFFORT.items() if k in jn), 84.0)
            L.append(f'  <joint name="{jn}" type="revolute">')
            L.append(f'    <origin xyz="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" rpy="0 0 0"/>')
            L.append(f'    <parent link="{pn}"/>'); L.append(f'    <child link="{bn}"/>')
            L.append(f'    <axis xyz="{ax[0]:.4f} {ax[1]:.4f} {ax[2]:.4f}"/>')
            L.append(f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" effort="{eff}" velocity="{VEL}"/>')
            L.append(f'  </joint>')
    L.append('</robot>')
    open(OUT, "w").write("\n".join(L))
    print(f"생성: {OUT}  ({m.nbody-1} links)")


if __name__ == "__main__":
    main()
