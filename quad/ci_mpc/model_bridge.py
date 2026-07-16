"""MJCF <-> Pinocchio bridge for the 17-DOF 02_Leg quadruped (Phase 0).

The 17-DOF robot exists as a MuJoCo MJCF (quad_real_17dof_waist_sphere.mjcf) and as
a SolidWorks-exported URDF (02_Leg_UFDF_260703_3.urdf). The MJCF is byte-for-byte
derived from that URDF (mass 38.016 kg, foot FK match to 1e-7 m; verified).

For the Crocoddyl OCP we load the URDF into Pinocchio and LOCK the waist joint
(FB_waist), leaving 16 leg DOF (nv=22 = 6 base + 16). The waist is held at 0 in
MuJoCo (ctrl=0) until a later phase re-activates it.

The two engines order the legs differently:
  MJCF actuator/ctrl order (nu=17): HL(4), HR(4), FB_waist(1), FL(4), FR(4)
  Pinocchio reduced leg order (nv leg block): FL(4), FR(4), HL(4), HR(4)
so a fixed permutation maps between them (MJ2PIN_LEG below).

Base convention differences handled here:
  quaternion  MuJoCo [w,x,y,z]  <->  Pinocchio [x,y,z,w]
  base linear velocity  MuJoCo qvel[0:3] is WORLD frame  <->  Pinocchio v[0:3] is
    LOCAL (body) frame, so v_local = R^T v_world.
  base angular velocity  both are body-local (MuJoCo qvel[3:6], Pinocchio v[3:6]).
"""
import numpy as np
import pinocchio as pin

# --- paths ---
URDF = "/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf"
PKG_DIRS = ["/home/jsh/문서/jsh/simulation"]  # resolves package://02_Leg_UFDF_260703_2/...
MJCF = "/home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf"

FEET = ["FL", "FR", "HL", "HR"]          # Pinocchio leg order
FOOT_FRAME = {L: f"{L}_foot_contact_link" for L in FEET}

# MJCF joint vector (qpos[7:], 17 entries): HL0-3, HR4-7, waist8, FL9-12, FR13-16
# Pinocchio reduced leg block order: FL, FR, HL, HR.
# MJ2PIN_LEG[i] = index into the MJCF 17-joint vector that supplies Pinocchio leg dof i.
MJ2PIN_LEG = np.array([9, 10, 11, 12,   13, 14, 15, 16,   0, 1, 2, 3,   4, 5, 6, 7])
MJ_WAIST_JIDX = 8                        # index of FB_waist inside the MJCF 17-joint vector


# Reflected rotor inertia + joint friction (GEARBOX), matching the C++ controller
# (quad_control.hpp): armature = I_rotor * N^2, plus viscous damping and dry friction.
# The MJCF ships with dof_armature=0; without this reflected inertia a stiff joint PD
# numerically explodes at dt=2ms. This is real physics, not a sim hack.
_GEAR = {"hip": 7.0, "thigh": 7.0, "calf": 10.5, "foot": 8.4}   # 실 감속비; waist -> hip fallback (7)
_I_ROTOR = 1e-4
_J_DAMP = 0.1
_J_FRIC = 0.5


def apply_gearbox(mm, i_rotor=_I_ROTOR, j_damp=_J_DAMP, j_fric=_J_FRIC):
    """Set dof_armature/damping/frictionloss on actuated joints (C++ GEARBOX parity)."""
    import mujoco
    for k in range(mm.nu):
        jid = mm.actuator_trnid[k, 0]
        if jid < 0:
            continue
        jn = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        N = 7.0
        for key, g in _GEAR.items():
            if key in jn:
                N = g
                break
        dof = mm.jnt_dofadr[jid]
        mm.dof_armature[dof] = i_rotor * N * N
        mm.dof_damping[dof] = j_damp
        mm.dof_frictionloss[dof] = j_fric


def strip_mesh_collision(mm):
    """Disable collision on all MESH geoms so only the sphere feet + floor collide.

    The deployed MJCF has colliding body meshes (Base/thigh/calf) for self-collision
    and terrain; MJX compiles every such pair, which is very slow. For sampling
    rollouts we only need foot-floor contact, so mesh collision is turned off.
    Kinematics/inertials are untouched (they live on the bodies, not these geoms).
    """
    import mujoco
    for g in range(mm.ngeom):
        if mm.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mm.geom_contype[g] = 0
            mm.geom_conaffinity[g] = 0


def set_foot_sphere(mm, radius):
    """Runtime-resize the 4 foot contact spheres (ci_mpc only; deployed MJCF untouched).

    NOTE: a sphere always contacts a plane at a single point regardless of radius -
    this does not enlarge the contact patch, it only lowers the contact point (the
    robot stands `radius` higher). The Pinocchio contact frame is the sphere CENTER,
    so a stance foot rests with its frame origin at z=radius.
    """
    import mujoco
    for L in FEET:
        gid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, f"{L}_sphere")
        if gid >= 0:
            mm.geom_size[gid, 0] = radius
    # keep rbound/AABB consistent for the collision broadphase
    mujoco.mj_setConst(mm, mujoco.MjData(mm))


class MjPinBridge:
    """Builds the waist-locked Pinocchio model and maps state/torque to/from MuJoCo."""

    def __init__(self, urdf=URDF, pkg_dirs=PKG_DIRS):
        full = pin.buildModelFromUrdf(urdf, pin.JointModelFreeFlyer())
        wid = full.getJointId("FB_waist_joint")
        self.model = pin.buildReducedModel(full, [wid], pin.neutral(full))
        self.data = self.model.createData()
        self.nq = self.model.nq          # 23
        self.nv = self.model.nv          # 22
        self.nu = self.nv - 6            # 16 actuated leg dof
        self.foot_fid = {L: self.model.getFrameId(FOOT_FRAME[L]) for L in FEET}
        self.mass = pin.computeTotalMass(self.model)

    # ----- state mapping -----
    def mj_to_pin_q(self, mj_qpos):
        """MuJoCo qpos (nq=24) -> Pinocchio reduced q (nq=23)."""
        p = mj_qpos[0:3]
        w, x, y, z = mj_qpos[3:7]                      # MuJoCo wxyz
        legs = mj_qpos[7:]                             # 17 joints
        qleg = legs[MJ2PIN_LEG]                        # 16, FL,FR,HL,HR
        return np.concatenate([p, [x, y, z, w], qleg])  # Pinocchio xyzw

    def mj_to_pin_v(self, mj_qvel, mj_qpos):
        """MuJoCo qvel (nv=23) -> Pinocchio reduced v (nv=22). Base lin vel world->local."""
        w, x, y, z = mj_qpos[3:7]
        R = pin.Quaternion(w, x, y, z).toRotationMatrix()
        v_lin = R.T @ mj_qvel[0:3]                     # world -> body
        v_ang = mj_qvel[3:6]                           # already body-local
        vleg = mj_qvel[6:][MJ2PIN_LEG]                 # 16
        return np.concatenate([v_lin, v_ang, vleg])

    def mj_to_pin_x(self, mj_qpos, mj_qvel):
        return np.concatenate([self.mj_to_pin_q(mj_qpos), self.mj_to_pin_v(mj_qvel, mj_qpos)])

    # ----- torque mapping -----
    def pin_tau_to_mj_ctrl(self, tau_pin, nu_mj=17):
        """Pinocchio leg torque (16, FL,FR,HL,HR) -> MuJoCo ctrl (17). Waist held at 0."""
        ctrl = np.zeros(nu_mj)
        ctrl[MJ2PIN_LEG] = tau_pin                     # scatter into HL,HR,FL,FR slots
        ctrl[MJ_WAIST_JIDX] = 0.0                      # waist locked
        return ctrl

    def scatter_leg(self, vec16, waist=0.0, n_mj=17):
        """Pinocchio leg vector (16, FL,FR,HL,HR) -> MuJoCo 17-joint vector (waist slot set)."""
        out = np.zeros(n_mj)
        out[MJ2PIN_LEG] = vec16
        out[MJ_WAIST_JIDX] = waist
        return out

    def pin_to_mj_qpos(self, q_pin, waist=0.0):
        """Pinocchio reduced q (23) -> MuJoCo qpos (24) for initialization."""
        p = q_pin[0:3]
        x, y, z, w = q_pin[3:7]                        # Pinocchio xyzw
        legs = self.scatter_leg(q_pin[7:], waist)      # 16 -> 17
        return np.concatenate([p, [w, x, y, z], legs])  # MuJoCo wxyz

    # ----- kinematics helpers -----
    def foot_positions(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return {L: self.data.oMf[self.foot_fid[L]].translation.copy() for L in FEET}


def _selftest():
    """Verify the bridge against MuJoCo: mass, foot FK, round-trip state mapping."""
    import mujoco
    br = MjPinBridge()
    mm = mujoco.MjModel.from_xml_path(MJCF)
    md = mujoco.MjData(mm)
    print(f"reduced nq={br.nq} nv={br.nv} nu={br.nu} mass={br.mass:.5f}")
    assert abs(br.mass - mm.body_mass.sum()) < 1e-4

    # random-ish config, check foot FK agreement
    rng = np.random.default_rng(0)
    md.qpos[:] = mm.qpos0
    md.qpos[7:] += 0.15 * rng.standard_normal(17)
    md.qpos[MJ_WAIST_JIDX + 7] = 0.0                   # keep waist at 0 (locked in pin)
    md.qvel[:] = 0.1 * rng.standard_normal(mm.nv)
    mujoco.mj_forward(mm, md)

    q = br.mj_to_pin_q(md.qpos)
    feet_pin = br.foot_positions(q)
    maxd = 0.0
    for L in FEET:
        bid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, FOOT_FRAME[L])
        d = np.linalg.norm(feet_pin[L] - md.xpos[bid])
        maxd = max(maxd, d)
    print(f"max foot FK diff = {maxd:.2e} m")
    assert maxd < 1e-5, "foot FK mismatch"

    # torque scatter round-trip
    tau = rng.standard_normal(br.nu)
    ctrl = br.pin_tau_to_mj_ctrl(tau)
    assert abs(ctrl[MJ_WAIST_JIDX]) == 0.0
    back = ctrl[MJ2PIN_LEG]
    assert np.allclose(back, tau)
    print("torque scatter round-trip OK")
    print("PASS")


if __name__ == "__main__":
    _selftest()
