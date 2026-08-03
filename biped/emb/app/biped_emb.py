"""app/biped_emb.py — biped 실기(Emb) 배포 메인 루프.

데이터흐름:  Backend.read → HwInterface(매핑·IMU변환) → [모드 디스패치] → Backend.write
  off   : 모터 limp
  jog   : per-axis 저속 위치 검증  ← 첫 딜리버러블(각축 확인)
  hold  : 현재자세 임피던스 홀드
  stand : 모델기반 균형 서기(MPC+WBIC)   ← jog 검증 후
  walk  : 모델기반 보행                    ← jog 검증 후
GUI(gui/teleop_emb.py)와 JSON 채널(/tmp/biped_cmd.json)로 디커플. 상태는 /tmp/biped_state.json 발행.

실행:
  MOCK=1 python app/biped_emb.py         # 데스크톱 데모(SHM 없이 jog 루프 검증)
  python app/biped_emb.py                # 실기(Pi): ShmBackend(libbipedshm.so)
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))            # emb/app
EMB  = os.path.dirname(HERE)                                 # emb
BIPED = os.path.dirname(EMB)                                 # simulation/biped
for d in ("hal", "interface", "control"):
    sys.path.insert(0, os.path.join(EMB, d))

import yaml
from backend import RawState
from mock_backend import MockBackend
from joint_map import JointMap, R2D, D2R
from hw_interface import HwInterface
from jog import Jogger
import mode_fsm as FSM

CMD_PATH   = os.environ.get("QUAD_CMD",   "/tmp/biped_cmd.json")
STATE_PATH = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def make_backend(cfg, force_mock: bool):
    n = int(cfg["shm"]["n_channel"])
    dt = 1.0 / float(cfg["meta"]["ctrl_hz"])
    if force_mock:
        return MockBackend(n, dt=dt), "mock"
    try:                                                     # 실기: SHM. 실패 시 mock 폴백.
        from shm_backend import ShmBackend
        be = ShmBackend(cfg["shm"]["lib"], n, int(cfg["shm"]["recv_wait_ms"]))
        return be, "shm"
    except OSError as e:
        print(f"[biped_emb] ⚠ SHM 라이브러리 로드 실패 ({e}) → MockBackend 폴백. "
              f"Pi에서 hal/build_bridge.sh 먼저 실행.")
        return MockBackend(n, dt=dt), "mock"


def read_cmd():
    try:
        with open(CMD_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def axis_health(raw, jm):
    """축별 상태(임베디드 보고): dead(무통신) / fault(통신 O·ucStatus≠0) / ok(통신 O·정상).
       ★ucStatus 의미는 모터/펌웨어 정의 → 지금은 ≠0 을 에러로 간주(실기 확정 후 세분화)."""
    conn, stat = raw.connected, raw.status
    out = []
    for ch in jm.ch:
        if conn.size <= ch or not conn[ch]:
            out.append("dead")
        elif stat.size > ch and int(stat[ch]) != 0:
            out.append("fault")
        else:
            out.append("ok")
    return out


def publish_state(mode, q_leg_deg, rpy_deg, loop_hz, motors_on, backend, extra=None):
    st = {"mode": mode, "q_leg_deg": [round(float(x), 2) for x in q_leg_deg],
          "rpy_deg": [round(float(x), 2) for x in rpy_deg],
          "tilt_deg": round(float(np.hypot(rpy_deg[0], rpy_deg[1])), 2),
          "loop_hz": round(float(loop_hz), 1), "motors_on": bool(motors_on),
          "backend": backend}
    if extra:
        st.update(extra)
    try:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, STATE_PATH)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(EMB, "config", "biped_emb.yaml"))
    ap.add_argument("--mock", action="store_true", help="SHM 없이 데스크톱 데모")
    ap.add_argument("--T", type=float, default=1e12, help="최대 실행시간[s] (검증용)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    force_mock = args.mock or os.environ.get("MOCK") == "1"
    jm = JointMap(cfg)
    backend, be_name = make_backend(cfg, force_mock)
    hw = HwInterface(backend, jm, imu_deg=bool(cfg["shm"].get("imu_deg", True)))
    hw.init()

    cfg_dt = 1.0 / float(cfg["meta"]["ctrl_hz"])
    jogger = Jogger(jm, cfg_dt, float(cfg["jog"]["max_speed_dps"]))
    fsm = FSM.ModeFSM(FSM.OFF)
    settle = float(cfg["jog"]["settle_deg"])
    tilt_estop = float(cfg["safety"]["tilt_estop_deg"])
    watchdog_s = float(cfg["safety"]["watchdog_ms"]) / 1000.0
    tau_frac = float(cfg["safety"]["tau_max_frac"])

    jog_goal = np.zeros(jm.n_leg)            # GUI 축별 목표각[deg]
    hold_ch  = np.zeros(jm.n_channel)        # hold 목표(채널)
    walk_cmd = {"v": 0.0, "vy": 0.0, "w": 0.0, "body_h": 0.38}
    model = None                             # 모델기반 래퍼(lazy)

    print(f"[biped_emb] backend={be_name} · ctrl_hz={cfg['meta']['ctrl_hz']} · CMD={CMD_PATH}")
    print("            모드: off/jog/hold(=지금) · stand/walk(=jog 검증 후). GUI로 조종.")

    hw.enable(False)
    t0 = time.perf_counter(); k = 0; last_cmd_t = t0; last_pub = 0.0
    prev_loop_t = t0
    hz_ema = float(cfg["meta"]["ctrl_hz"])

    while time.perf_counter() - t0 < args.T:
        loop_t = time.perf_counter()
        raw = hw.read()
        q_leg = hw.q_leg_deg()
        rpy = raw.imu_rpy_deg * (1.0 if cfg["shm"].get("imu_deg", True) else R2D)

        # ── 명령 폴링(~50Hz) ──
        if k % max(1, int(0.02 / cfg_dt)) == 0:
            cmd = read_cmd()
            if cmd:
                last_cmd_t = loop_t
                new_mode = cmd.get("mode", fsm.mode)
                if new_mode == "reset":
                    if fsm.mode in FSM.MODEL_BASED and model is not None:
                        model.reset(walk_cmd["body_h"])
                    jogger.reset(q_leg); new_mode = FSM.HOLD
                changed = fsm.set(new_mode)
                jog_goal = np.asarray(cmd.get("jog_deg", jog_goal), float)[: jm.n_leg]
                for key in ("v", "vy", "w", "body_h"):
                    if key in cmd:
                        walk_cmd[key] = float(cmd[key])
                # ── 전이 진입 부작용 ──
                if changed:
                    if fsm.mode == FSM.OFF:
                        hw.enable(False)
                    else:
                        hw.enable(True)
                    if fsm.entered(FSM.JOG):
                        jogger.reset(q_leg)
                    if fsm.entered(FSM.HOLD):
                        hold_ch = raw.q_deg.copy()
                    if fsm.is_model_based():
                        if model is None:
                            try:
                                sys.path.insert(0, os.path.join(EMB, "control"))
                                from model_ctrl import ModelController
                                print("[biped_emb] 모델기반 컨트롤러 로드 중(mujoco+qpsolvers)…")
                                model = ModelController(BIPED, os.path.join(BIPED, "deploy"), tau_frac)
                            except Exception as e:
                                print(f"[biped_emb] ❌ 모델 로드 실패 ({e}) → hold 폴백")
                                fsm.set(FSM.HOLD); hold_ch = raw.q_deg.copy()
                        if model is not None and fsm.is_model_based():
                            model.reset(walk_cmd["body_h"])

        # ── 워치독: 명령 끊기면 안전(limp) ──
        if fsm.mode != FSM.OFF and (loop_t - last_cmd_t) > watchdog_s:
            hw.enable(False)
        elif fsm.mode != FSM.OFF:
            hw.enable(True)

        # ── 기울기 E-stop ──
        if fsm.mode in FSM.MODEL_BASED and np.hypot(rpy[0], rpy[1]) > tilt_estop:
            print(f"[biped_emb] ⚠ tilt {np.hypot(rpy[0],rpy[1]):.0f}° > {tilt_estop:.0f}° → hold")
            fsm.set(FSM.HOLD); hold_ch = raw.q_deg.copy(); hw.enable(True)

        # ── 축별 health = 임베디드 보고 반영(통신+ucStatus). ok/fault/dead. (제어 아님, 모니터) ──
        health = axis_health(raw, jm)
        extra = {"health": health,
                 "n_ok": health.count("ok"), "n_fault": health.count("fault"),
                 "n_dead": health.count("dead")}

        # ── 모드 디스패치 (전 채널 명령; 미배선/죽은 축은 임베디드가 흡수) ──
        if fsm.mode == FSM.OFF:
            hw.write_jog(jm.q_ctrl_to_ch(np.zeros(jm.n_leg)))     # enable=False → 브리지 0 토크
        elif fsm.mode == FSM.JOG:
            hw.write_jog(jogger.step(jog_goal))
            extra["jog_at_goal"] = jogger.at_goal(jog_goal, settle)
        elif fsm.mode == FSM.HOLD:
            hw.write_hold(hold_ch)
        elif fsm.mode in FSM.MODEL_BASED and model is not None:
            q, dq, quat, gyro, acc, contact = hw.ctrl_state()
            model.set_cmd(walk_cmd["v"], walk_cmd["vy"], walk_cmd["w"],
                          walk_cmd["body_h"], walking=(fsm.mode == FSM.WALK))
            tau = model.step(q, dq, quat, gyro, acc, contact)
            hw.write_torque(q, dq, tau)
            ep, ev = model.est_state
            extra["est_x"] = round(float(ep[0]), 3); extra["est_z"] = round(float(ep[2]), 3)

        # ── 상태 발행(~20Hz) + HUD hz (실제 루프 주기) ──
        period = loop_t - prev_loop_t; prev_loop_t = loop_t
        if period > 0:
            hz_ema = 0.98 * hz_ema + 0.02 * (1.0 / period)
        if loop_t - last_pub > 0.05:
            publish_state(fsm.mode, q_leg, rpy, hz_ema, fsm.mode != FSM.OFF, be_name, extra)
            last_pub = loop_t

        # ── 실시간 페이싱 ──
        k += 1
        lag = t0 + k * cfg_dt - time.perf_counter()
        if lag > 0:
            time.sleep(lag)

    hw.enable(False)
    hw.close()
    print("[biped_emb] 종료(limp).")


if __name__ == "__main__":
    main()
