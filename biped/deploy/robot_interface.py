"""biped 하드웨어 추상화 계층 (Unitree/RBQ SDK 규약: LowCmd / LowState, kp·kd·tau).

컨트롤러(biped_mpc_wbic)는 MuJoCo 모델로 M·자코비안·bias를 계산하고 **순수 토크**(d.ctrl)를 출력한다.
배포 = 이 '플랜트'만 교체한다 — sim=mj_step / 실HW=모터 read+write + 상태추정.

  loop:  st = iface.read()             # 관절 q/dq + IMU(quat/gyro) (+ 발 접촉)
         iface.apply_state(d, st)      # 측정값을 컨트롤러의 MuJoCo d(모델)에 주입
         c.control(dt)                 # WBIC → d.ctrl (feedforward tau)
         iface.write(LowCmd(tau=...))  # sim=mj_step / 실HW=setTorqueRef

백엔드(SimInterface↔HardwareInterface)만 갈아끼우면 컨트롤러·게인·GUI JSON 채널은 그대로.
실모터 SDK(CAN/EtherCAT/DDS)는 HardwareInterface의 TODO에 채운다.

참조: 메모리 rbq-sdk-reference(SDK 규약)·02leg-motor-spec(모터 스펙)·sim2real-checklist-17dof(갭).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

# 관절 순서 = MuJoCo qpos[7:] / actuator 순서와 동일 (biped_from_quad.mjcf)
JOINT_NAMES = ['HL_hip', 'HL_thigh', 'HL_calf', 'HL_foot',
               'HR_hip', 'HR_thigh', 'HR_calf', 'HR_foot']
N_JOINT = 8
FOOT_NAMES = ['HL_sphere', 'HR_sphere']


@dataclass
class LowState:
    """센서 피드백(실HW=모터 인코더 + IMU + 접촉센서 / sim=MuJoCo d)."""
    q:    np.ndarray = field(default_factory=lambda: np.zeros(N_JOINT))   # 관절각[rad]
    dq:   np.ndarray = field(default_factory=lambda: np.zeros(N_JOINT))   # 관절속도[rad/s]
    tau:  np.ndarray = field(default_factory=lambda: np.zeros(N_JOINT))   # 추정토크[Nm] (모니터)
    quat: np.ndarray = field(default_factory=lambda: np.array([1.,0,0,0]))# base 자세(IMU) [w,x,y,z]
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))         # base 각속도(IMU) [rad/s] body-frame
    acc:  np.ndarray = field(default_factory=lambda: np.zeros(3))         # base 선가속(IMU) [m/s²]
    foot_contact: np.ndarray = field(default_factory=lambda: np.zeros(2, bool))  # [HL,HR] 접촉
    # ★base 위치/선속도는 직접 측정 불가 → 상태추정기가 채움(sim2real 최대 갭, checklist C)
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class LowCmd:
    """액추에이터 명령. 실모터 = tau_out = kp·(q_des−q) + kd·(dq_des−dq) + tau.
       WBIC 배포 = 순수 토크(kp=kd=0, tau=feedforward). damp/hold = kp·kd만."""
    tau:    np.ndarray = field(default_factory=lambda: np.zeros(N_JOINT))  # feedforward 토크[Nm]
    q_des:  np.ndarray = field(default_factory=lambda: np.zeros(N_JOINT))
    dq_des: np.ndarray = field(default_factory=lambda: np.zeros(N_JOINT))
    kp:     np.ndarray = field(default_factory=lambda: np.zeros(N_JOINT))
    kd:     np.ndarray = field(default_factory=lambda: np.zeros(N_JOINT))


class StateEstimator:
    """leg-odometry 상태추정 — IMU(자세·각속도) + 관절엔코더(q,dq) + 접촉 → base pose/vel.
    ★절대 base(GPS/모캡) 안 씀 = 실로봇과 동일. **센서만으로 full state 복원**.

    원리(stance 발 정지 가정): 접촉발이 지면서 미끄러지지 않으면 그 발의 world 속도=0.
      다리 운동학으로 '몸통 기준 발 속도'를 알면 역산: v_base = −(ω×R·p_foot + R·v_foot).
      여러 stance 발 평균 + 저역통과(alpha) → base 선속도, 적분 → base 위치(드리프트=실기와 동일).
      base 자세=IMU quat 직접, base 각속도=gyro 직접(추정 불필요).
    state_estimator.hpp(quad C++) 계보 + ★강건화(점발 biped 폐루프용):
      ① 접촉 dwell 게이팅 — 착지/이지 과도(velocity 스파이크=폐루프 붕괴 주범) 제거, 안정 접촉발만 사용.
      ② 접촉 앵커 위치보정 — 안정 접촉발의 world 위치 고정 → 드리프트 제거(적분만 하던 위치를 stance마다 앵커).
      ③ IMU accel 융합(선택) — 예측(연속성) + leg-odom 보정(complementary). 발 개수 무관."""

    def __init__(self, m, foot_geom, foot_rad, dt,
                 alpha=0.4, dwell_steps=0, k_anchor=0.0, use_accel=False):
        import mujoco
        self._mj = mujoco
        self.m = m
        self.ed = mujoco.MjData(m)          # 상대 운동학용 scratch(base 원점·단위자세)
        self.foot_geom = list(foot_geom); self.foot_rad = list(foot_rad); self.dt = dt
        self.alpha = alpha; self.dwell_steps = dwell_steps
        self.k_anchor = k_anchor; self.use_accel = use_accel
        self.g_world = np.array([0.0, 0.0, -9.81])
        self.p = np.zeros(3); self.v = np.zeros(3)
        self.dwell = [0] * len(foot_geom)   # 발별 연속 접촉 스텝 수
        self.anchor = [None] * len(foot_geom)  # 발별 world 앵커(안정 접촉 시 고정)

    def reset(self, p0):
        self.p = np.array(p0, float); self.v = np.zeros(3)
        self.dwell = [0] * len(self.foot_geom); self.anchor = [None] * len(self.foot_geom)

    def hold(self):
        """정지(미보행): 속도 0 유지(드리프트 억제)."""
        self.v[:] = 0.0

    def estimate(self, qj, dqj, quat_wxyz, gyro, contacts, acc=None):
        mj, m, ed = self._mj, self.m, self.ed
        ed.qpos[:] = 0.0; ed.qpos[3] = 1.0          # base 원점·단위자세 → 순수 다리 운동학
        ed.qpos[7:] = qj
        ed.qvel[:] = 0.0; ed.qvel[6:] = dqj
        mj.mj_forward(m, ed)
        Rm = np.zeros(9); mj.mju_quat2Mat(Rm, np.asarray(quat_wxyz, float)); R = Rm.reshape(3, 3)
        omw = R @ np.asarray(gyro, float)           # 동체 각속도 → world
        # ── ① dwell 갱신: 착지=카운트↑, 이지=리셋. 안정 접촉(dwell≥임계)만 신뢰 ──
        jacp = np.zeros((3, m.nv)); vbs = []; pfw_solid = []
        for k, g in enumerate(self.foot_geom):
            if contacts[k]:
                self.dwell[k] += 1
            else:
                self.dwell[k] = 0; self.anchor[k] = None
            if not contacts[k] or self.dwell[k] < self.dwell_steps:
                continue                             # 과도(touchdown/liftoff) 스킵 = 스파이크 제거
            gp = ed.geom_xpos[g].copy()
            pfb = gp.copy(); pfb[2] -= self.foot_rad[k]          # 발 접촉점(base frame, 반경 보정)
            mj.mj_jac(m, ed, jacp, None, gp, m.geom_bodyid[g])
            vfb = jacp @ ed.qvel                                 # 발 속도(관절 기여, base frame)
            pfw = R @ pfb                                        # 발 오프셋(world)
            vbs.append(-(np.cross(omw, pfw) + R @ vfb))          # 발 정지 → base world 속도
            pfw_solid.append((k, pfw))
        # ── ③ 속도: (선택)accel 예측 + dwell-gated leg-odom 보정 ──
        if self.use_accel and acc is not None:
            self.v = self.v + (R @ np.asarray(acc, float) + self.g_world) * self.dt   # IMU 예측
        if vbs:
            self.v = (1 - self.alpha) * self.v + self.alpha * np.mean(vbs, axis=0)     # 보정
        # ── ② 위치: 적분 + 접촉 앵커 보정(드리프트 제거) ──
        self.p = self.p + self.v * self.dt
        for k, pfw in pfw_solid:
            if self.anchor[k] is None:
                self.anchor[k] = self.p + pfw            # 안정 접촉 시작 = 발 world 위치 고정
            else:
                p_meas = self.anchor[k] - pfw            # 앵커 기준 base 위치(비드리프트)
                self.p = self.p + self.k_anchor * (p_meas - self.p)   # 서서히 보정
        return self.p.copy(), self.v.copy()


class RobotInterface(ABC):
    """플랜트 추상화. 컨트롤러↔모터 사이 경계(SDK LowCmd/LowState)."""
    JOINT_NAMES = JOINT_NAMES
    N_JOINT = N_JOINT

    @abstractmethod
    def read(self) -> LowState:
        """센서 → LowState (관절 q/dq + IMU + 접촉)."""

    @abstractmethod
    def apply_state(self, d, st: LowState) -> None:
        """측정 상태를 컨트롤러 MuJoCo d(=동역학 모델)에 주입."""

    @abstractmethod
    def write(self, cmd: LowCmd) -> None:
        """토크 명령을 플랜트로 전송(sim=mj_step / 실HW=setTorqueRef)."""

    @abstractmethod
    def enable_motors(self, on: bool) -> None:
        """★모터 전원 on/off. off=torque disable(limp). GUI 'Off 전원'과 연결."""

    def dt(self) -> float:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class SimInterface(RobotInterface):
    """MuJoCo 시뮬 백엔드 — 플랜트=컨트롤러 자신의 (m,d). 배포 API 검증·회귀용.
    read()=현재 d 스냅샷 · apply_state=no-op(d가 곧 진실) · write=ctrl 세팅+mj_step."""

    def __init__(self, m, d):
        import mujoco
        self._mj = mujoco
        self.m, self.d = m, d
        self.motors_on = True
        self.sph = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
        self.fbody = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                      for b in ['HL_foot_contact_link', 'HR_foot_contact_link']]
        # IMU/GT 센서 주소(있으면 sensordata 사용=진짜 센서만, 없으면 d 직접 fallback)
        def _sadr(nm):
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, nm)
            return int(m.sensor_adr[i]) if i >= 0 else -1
        self._s = {k: _sadr(k) for k in ('imu_gyro', 'imu_acc', 'imu_quat', 'gt_pos', 'gt_vel')}

    def dt(self):
        return self.m.opt.timestep

    def read(self) -> LowState:
        d, m = self.d, self.m
        st = LowState()
        st.q[:]  = d.qpos[7:]                       # 엔코더(실센서)
        st.dq[:] = d.qvel[6:]
        st.tau[:] = d.ctrl[:]
        s = self._s
        if s['imu_quat'] >= 0:                      # ★진짜 IMU 센서(sensordata) 사용
            st.quat[:] = d.sensordata[s['imu_quat']:s['imu_quat']+4]
            st.gyro[:] = d.sensordata[s['imu_gyro']:s['imu_gyro']+3]
            st.acc[:]  = d.sensordata[s['imu_acc']:s['imu_acc']+3]
            st.base_pos[:] = d.sensordata[s['gt_pos']:s['gt_pos']+3]   # GT(검증 전용)
            st.base_vel[:] = d.sensordata[s['gt_vel']:s['gt_vel']+3]
        else:                                        # 센서 없는 옛 모델 fallback
            st.quat[:] = d.qpos[3:7]; st.gyro[:] = d.qvel[3:6]
            st.base_pos[:] = d.qpos[0:3]; st.base_vel[:] = d.qvel[0:3]
        # 발 접촉(수직력>임계) — 실기=힘센서/추정, sim=접촉 진실
        for i in range(2):
            f = 0.0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if self.sph[i] in (c.geom1, c.geom2):
                    f = 1.0; break
            st.foot_contact[i] = f > 0.5
        return st

    def apply_state(self, d, st: LowState) -> None:
        pass  # sim: d가 곧 진실. HW에서만 주입 필요.

    def write(self, cmd: LowCmd) -> None:
        d, m = self.d, self.m
        if self.motors_on:
            tau = cmd.tau.copy()
            if np.any(cmd.kp) or np.any(cmd.kd):   # 선택적 관절 PD(hold/damp)
                tau = tau + cmd.kp*(cmd.q_des - d.qpos[7:]) + cmd.kd*(cmd.dq_des - d.qvel[6:])
            d.ctrl[:] = tau
        else:
            d.ctrl[:] = 0.0                        # 전원 off = limp
        self._mj.mj_step(m, d)

    def enable_motors(self, on: bool) -> None:
        self.motors_on = bool(on)


# ─────────────────────────────────────────────────────────────────────────────
class HardwareInterface(RobotInterface):
    """실모터 백엔드 스텁 — RGA/RBQ 저수준 SDK(setTorqueRef/getLowState)에 연결.
    ★TODO: 각 메서드의 driver 호출을 실제 SDK로 채운다. 순서=JOINT_NAMES.

    체크리스트(deploy/README.md 상세):
      A. 모터 버스 초기화(CAN/EtherCAT), 방향·부호·오프셋 캘리브레이션
      B. IMU 스트림(quat·gyro·acc), 좌표계=base body-frame 정렬
      C. ★base 위치/선속도 상태추정기(leg-odometry + IMU EKF) = 최대 갭
      D. 안전: 토크 slew-rate·관절한계·E-stop·통신 워치독
    """

    def __init__(self, mjcf, dt=0.002):
        import mujoco
        self._mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(mjcf)   # 동역학 모델(상태추정 FK·WBIC 재사용)
        self._dt = dt
        self.motors_on = False
        sph = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f) for f in FOOT_NAMES]
        rad = [float(self.m.geom_size[g][0]) for g in sph]
        self.est = StateEstimator(self.m, sph, rad, dt)   # ★센서만으로 base 복원(leg-odometry)
        # self.bus = MotorBus(...)      # TODO: 실 SDK 핸들(CAN/EtherCAT/DDS)
        # self.imu = ImuDriver(...)
        raise NotImplementedError(
            "HardwareInterface: 실모터 SDK 연결 필요. deploy/README.md의 A~D 채우기.\n"
            "  - read(): 모터 인코더 q/dq + IMU quat/gyro + 접촉 → LowState (아래 TODO)\n"
            "  - write(): tau → setTorqueRef (slew-rate·한계 클램프)\n"
            "  - enable_motors(): 모터 enable/disable (Off 전원)\n"
            "  ★base 위치/선속도는 apply_state의 StateEstimator가 센서만으로 복원(구현 완료).")

    def dt(self):
        return self._dt

    def read(self) -> LowState:            # TODO: bus/imu 폴링 → LowState(q,dq,quat,gyro,foot_contact)
        raise NotImplementedError

    def apply_state(self, d, st: LowState) -> None:
        # ① 측정 관절 + IMU 자세/각속도를 모델에 주입
        d.qpos[7:] = st.q
        d.qvel[6:] = st.dq
        d.qpos[3:7] = st.quat
        d.qvel[3:6] = st.gyro
        # ② ★base 위치/선속도 = leg-odometry 추정(센서만, 절대 base 불필요). WBIC CoM/속도 task가 사용.
        self.est.estimate(st.q, st.dq, st.quat, st.gyro, st.foot_contact, acc=st.acc)
        d.qpos[0:3] = self.est.p
        d.qvel[0:3] = self.est.v
        self._mj.mj_forward(self.m, d)

    def write(self, cmd: LowCmd) -> None:  # TODO: setTorqueRef(cmd.tau), slew-rate·한계 클램프
        raise NotImplementedError

    def enable_motors(self, on: bool) -> None:  # TODO: motor enable/disable
        self.motors_on = bool(on)
