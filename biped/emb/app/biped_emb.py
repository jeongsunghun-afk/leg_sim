"""app/biped_emb.py — biped 실기(Emb) 배포 메인 루프.

데이터흐름:  Backend.read → HwInterface(매핑·IMU변환) → [모드 디스패치] → Backend.write
  off   : 모터 limp
  jog   : per-axis 저속 위치 검증  ← 첫 딜리버러블(각축 확인)
  home  : 정해진 홈 자세로 S-curve 복귀(control/home.py) → 도달 후 그 자세 유지
  hold  : 현재자세 임피던스 홀드
  stand : 모델기반 균형 서기(MPC+WBIC)   ← jog 검증 후
  walk  : 모델기반 보행                    ← jog 검증 후
GUI(gui/teleop_emb.py)와 JSON 채널(/tmp/biped_cmd.json)로 디커플. 상태는 /tmp/biped_state.json 발행.

실행:
  MOCK=1 python app/biped_emb.py         # 데스크톱 데모(SHM 없이 jog 루프 검증)
  python app/biped_emb.py                # 실기(Pi): ShmBackend(libbipedshm.so)
"""
from __future__ import annotations
import os, sys, json, time, signal, argparse
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
from home import HomeTrajectory
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


def safe_shutdown(hw, jm):
    """실제로 무여자 상태로 만들고 끝낸다.

    ★기존 종료는 `hw.enable(False); hw.close()` 뒤에 "종료(limp)" 를 찍었지만
      **거짓이었다.** bridge_enable(0) 은 g_enabled 플래그만 바꿀 뿐 SHM 명령버퍼를
      건드리지 않는다(shm_bridge.cpp:115). 마지막에 쓴 kp40 명령이 SHM 에 그대로 남고
      Emb 는 그것을 1kHz 로 **영원히 재전송**한다. 즉 앱을 껐는데 모터는 계속 잡고 있다.
      "명령을 안 쓰는 것" 은 정지가 아니다 — Kp=Kd=0 을 **실제로 써야** 한다.
    ⇒ enable(False) 후 **반복 기록**해서 무여자 명령이 확실히 SHM 에 들어가게 한다.
    """
    try:
        hw.enable(False)
        z = jm.q_ctrl_to_ch(np.zeros(jm.n_leg))
        for _ in range(25):                     # enable=False → 브리지가 kp=kd=0 으로 기록
            hw.write_jog(z)
            time.sleep(0.002)
        print("[biped_emb] 종료 — 무여자(Kp=Kd=0) 명령 25회 기록 완료.")
    except Exception as e:
        print(f"[biped_emb] ⚠⚠ 종료 중 무여자 기록 실패({e}) — "
              f"Emb 가 마지막 명령을 계속 재전송한다. **모터 전원을 차단할 것**.")
    finally:
        try:
            hw.close()
        except Exception:
            pass


def read_cmd():
    try:
        with open(CMD_PATH) as f:
            return json.load(f)
    except Exception:
        return None


_last_cmd_repr = None


def read_cmd_fresh():
    """명령을 읽되 **내용이 실제로 바뀌었는지**를 함께 반환한다.

    ★기존 워치독은 데드코드였다. read_cmd() 가 파싱만 성공하면 last_cmd_t 를 갱신했는데,
      명령 파일은 정적이라 20ms 마다 항상 파싱에 성공한다 → 경과시간이 watchdog_ms(100)
      를 넘을 수가 없어 **워치독이 한 번도 동작하지 않는다.**
      GUI 가 죽어도, 릴레이가 끊겨도 로봇은 마지막 명령을 계속 수행한다.
    ⇒ 파일이 읽히는 것과 명령이 살아있는 것은 다르다. 내용 변화로 판정한다.
    ⚠ GUI 가 주기적으로 재발행하지 않으면(정적 파일) 이제 워치독이 정상적으로 트립한다.
      teleop_gui_biped 는 이벤트 시에만 파일을 쓰므로, 발행측에 20Hz 하트비트(seq 증가)를
      넣기 전에는 watchdog_ms 를 넉넉히(500ms+) 두어야 jog 램프 중 오작동하지 않는다.
    """
    global _last_cmd_repr
    c = read_cmd()
    if c is None:
        return None, False
    r = repr(sorted(c.items()))
    fresh = (r != _last_cmd_repr)
    _last_cmd_repr = r
    return c, fresh


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
    hcfg = cfg.get("home", {})
    homer = HomeTrajectory(jm, cfg_dt,
                           hcfg.get("q_deg", [0.0] * jm.n_leg),
                           float(hcfg.get("max_speed_dps", 15.0)),
                           float(hcfg.get("max_acc_dps2", 30.0)),
                           float(hcfg.get("min_time_s", 0.6)))
    home_settle = float(hcfg.get("settle_deg", 0.5))
    # ★홈 목표가 jog 안전한계에 잘렸으면 조용히 넘어가지 않는다 — "홈에 갔다" 는 보고와
    #   실제 자세가 어긋나게 되고, 그 어긋남은 다음 모드의 시작자세가 된다.
    for nm, want, got in homer.clamped:
        print(f"[biped_emb] ⚠ home.q_deg[{nm}] {want:+.1f}° → {got:+.1f}° 로 클램프 "
              f"(jog 안전한계 = 관절한계×{jm.jog_frac}). 이 자세로는 홈에 도달하지 못한다.")
    settle = float(cfg["jog"]["settle_deg"])
    tilt_estop = float(cfg["safety"]["tilt_estop_deg"])
    watchdog_s = float(cfg["safety"]["watchdog_ms"]) / 1000.0
    tau_frac = float(cfg["safety"]["tau_max_frac"])
    # ★토크/속도 트립(시험 하네스 emb/pace/hwio.py 에서 승격). 미설정이면 하네스 기본값.
    tau_trip_nm  = float(cfg["safety"].get("tau_trip_nm",  8.0))
    tau_trip_ms  = float(cfg["safety"].get("tau_trip_ms",  50))
    vel_trip_dps = float(cfg["safety"].get("vel_trip_dps", 200.0))
    tau_over_t0  = None                      # 토크 연속초과 시작시각(None=정상)

    jog_goal = np.zeros(jm.n_leg)            # GUI 축별 목표각[deg]
    hold_ch  = np.zeros(jm.n_channel)        # hold 목표(채널)
    walk_cmd = {"v": 0.0, "vy": 0.0, "w": 0.0, "body_h": 0.38}
    model = None                             # 모델기반 래퍼(lazy)

    print(f"[biped_emb] backend={be_name} · ctrl_hz={cfg['meta']['ctrl_hz']} · CMD={CMD_PATH}")
    print("            모드: off/jog/hold(=지금) · stand/walk(=jog 검증 후). GUI로 조종.")

    hw.enable(False)
    t0 = time.perf_counter(); k = 0; last_cmd_t = t0; last_pub = 0.0
    estop_latched = False
    wd_tripped = False
    # ★tilt E-stop 은 IMU 가 있어야 동작한다. 이 로봇은 현재 SHM IMU 가 전부 0 이라
    #   tilt 가 항상 0 으로 계산되어 **E-stop 이 사실상 비활성**이다. 조용히 넘어가면
    #   보호장치가 있다고 착각하게 되므로 기동 시 명시적으로 경고한다.
    _imu0 = hw.read().imu_rpy_deg
    if float(np.max(np.abs(_imu0))) < 1e-9:
        print("[biped_emb] ⚠⚠ IMU 값이 전부 0 → **tilt E-stop 비활성**. "
              "IMU 배선/펌웨어 확인 전까지 기울기 보호가 없다고 간주할 것.")
    prev_loop_t = t0
    hz_ema = float(cfg["meta"]["ctrl_hz"])

    # ★어떤 경로로 죽어도 무여자로 빠지게 한다. 기존엔 정상 종료(args.T 만료) 경로에만
    #   종료 처리가 있어서 Ctrl-C·예외·SIGTERM 이면 모터가 잡힌 채로 앱만 사라졌다.
    def _sig(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")
    for _s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_s, _sig)
        except (ValueError, OSError):
            pass

    try:
      while time.perf_counter() - t0 < args.T:
          loop_t = time.perf_counter()
          raw = hw.read()
          q_leg = hw.q_leg_deg()
          rpy = raw.imu_rpy_deg * (1.0 if cfg["shm"].get("imu_deg", True) else R2D)

          # ── 명령 폴링(~50Hz) ──
          if k % max(1, int(0.02 / cfg_dt)) == 0:
              cmd, cmd_fresh = read_cmd_fresh()
              if cmd:
                  if cmd_fresh:
                      last_cmd_t = loop_t          # ★내용이 바뀐 경우에만 갱신(위 주석 참조)
                  new_mode = cmd.get("mode", fsm.mode)
                  if new_mode == "reset":
                      if fsm.mode in FSM.MODEL_BASED and model is not None:
                          model.reset(walk_cmd["body_h"])
                      jogger.reset(q_leg); new_mode = FSM.HOLD
                  # ★E-stop 래치 해제는 명시적 off 명령으로만. 그 전까지 모드변경 무시.
                  if estop_latched:
                      if new_mode == FSM.OFF:
                          estop_latched = False
                          print("[biped_emb] E-stop 래치 해제(off 수신) — 재무장 가능")
                      else:
                          new_mode = FSM.OFF
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
                      if fsm.entered(FSM.HOME):
                          # ★측정각에서 궤적을 만든다(명령각이 아니라). 부하로 처진 상태에서
                          #   명령각을 기점으로 잡으면 첫 틱에 그 편차만큼 계단이 나간다.
                          T = homer.start(q_leg)
                          print(f"[biped_emb] home 복귀 시작 — {T:.2f}s "
                                f"(v≤{hcfg.get('max_speed_dps', 15.0)}dps, "
                                f"a≤{hcfg.get('max_acc_dps2', 30.0)}dps²)  "
                                + "  ".join(f"{jm.names[i]}{q_leg[i]:+.1f}→{homer.q_home[i]:+.1f}"
                                            for i in range(jm.n_leg)), flush=True)
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
          #   ★전이를 출력한다. 조용히 enable(False) 만 하면 워치독이 도는지 운용 중에도
          #     시험 중에도 알 수 없다 — 실제로 데드코드인 것을 오래 못 봤다.
          wd_trip = (fsm.mode != FSM.OFF) and (loop_t - last_cmd_t) > watchdog_s
          if wd_trip != wd_tripped:
              wd_tripped = wd_trip
              if wd_trip:
                  print(f"[biped_emb] 워치독 트립 — 명령 두절 "
                        f"{loop_t - last_cmd_t:.2f}s > {watchdog_s:.2f}s → limp", flush=True)
              else:
                  print("[biped_emb] 워치독 해제 — 명령 복귀", flush=True)
          if fsm.mode != FSM.OFF:
              hw.enable(not wd_trip)

          # ── 기울기 E-stop ──────────────────────────────────────────────────
          #   ★기존 구현의 결함 셋을 모두 고쳤다:
          #     (a) 게이트가 MODEL_BASED 뿐이라 JOG/HOLD 에는 보호가 없었다 → 전 모드로 확대
          #     (b) `hw.enable(True)` 였다 — E-stop 인데 **인가 상태**로 두는 것이라
          #         yaml 주석의 "limp" 와 정반대였다 → enable(False)
          #     (c) **래치가 없어** 20ms 뒤 명령파일이 여전히 stand 면 곧바로 재무장했다.
          #         게다가 hold_ch 를 매 틱 재캡처해 목표가 낙하를 따라가 복원토크가 0 이 됐다.
          #         → 래치 + 1회만 캡처. 해제는 명령파일이 명시적으로 off 를 보낼 때만.
          tilt_now = float(np.hypot(rpy[0], rpy[1]))
          if (not estop_latched) and fsm.mode != FSM.OFF and tilt_now > tilt_estop:
              print(f"[biped_emb] ⛔ E-STOP: tilt {tilt_now:.0f}° > {tilt_estop:.0f}° → limp·래치")
              estop_latched = True
              fsm.set(FSM.OFF); hw.enable(False)
          # ── 토크/속도 트립 ────────────────────────────────────────────────
          #   ★2026-08-05 추가. 이 임계값들은 emb/pace/hwio.py 에 이미 있었고 실측으로
          #     확정돼 있었는데 배포 앱은 하나도 쓰지 않았다 — raw.tau_nm / raw.dq_dps 를
          #     매 틱 읽어놓고 안전 판정에 안 썼다. tilt E-stop 이 IMU 부재로 무력하므로
          #     실질 런타임 보호가 워치독뿐이었다. 시험 하네스의 검증값을 승격한다.
          #   ⚠OFF 모드에선 검사하지 않는다 — 무여자 상태의 외력(사람이 다리를 미는 등)까지
          #     트립으로 잡으면 재기동이 불가능해진다. 여자 중일 때만 의미가 있다.
          if (not estop_latched) and fsm.mode != FSM.OFF:
              tau_pk = float(np.max(np.abs(raw.tau_nm))) if raw.tau_nm.size else 0.0
              vel_pk = float(np.max(np.abs(raw.dq_dps))) if raw.dq_dps.size else 0.0
              # 토크는 **연속 초과**만 트립(착지 충격 같은 순간 스파이크를 살린다)
              if tau_pk > tau_trip_nm:
                  if tau_over_t0 is None:
                      tau_over_t0 = loop_t
                  elif (loop_t - tau_over_t0) * 1000.0 >= tau_trip_ms:
                      ch = int(np.argmax(np.abs(raw.tau_nm)))
                      print(f"[biped_emb] ⛔ E-STOP: ch{ch} 토크 {tau_pk:.2f}Nm > {tau_trip_nm}Nm "
                            f"가 {tau_trip_ms}ms 연속 → limp·래치")
                      estop_latched = True; fsm.set(FSM.OFF); hw.enable(False)
              else:
                  tau_over_t0 = None                # 한 틱이라도 정상이면 타이머 리셋
              # 속도는 즉시 트립(폭주는 지연시킬 이유가 없다)
              if (not estop_latched) and vel_pk > vel_trip_dps:
                  ch = int(np.argmax(np.abs(raw.dq_dps)))
                  print(f"[biped_emb] ⛔ E-STOP: ch{ch} 속도 {vel_pk:.0f}dps > {vel_trip_dps}dps → limp·래치")
                  estop_latched = True; fsm.set(FSM.OFF); hw.enable(False)
          else:
              tau_over_t0 = None

          if estop_latched:
              hw.enable(False)                    # 래치 동안 계속 무여자 강제

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
          elif fsm.mode == FSM.HOME:
              # ★워치독 트립 중에는 궤적을 현재 측정각으로 계속 재기준한다.
              #   트립 = 무여자(limp) 라 다리가 중력으로 처지는데, 그동안 궤적 시간만
              #   흘려보내면 명령 복귀 순간 "처진 실제 위치" 와 "그새 진행된 목표" 사이의
              #   편차가 통째로 계단 입력이 된다(kp40 → 1° 당 0.7Nm). 재기준해 두면
              #   복귀 시 처진 자리에서 홈까지 궤적을 처음부터 다시 그린다.
              if wd_trip:
                  homer.start(q_leg)
              hw.write_jog(homer.step())
              extra["home_progress"] = round(homer.progress, 3)
              extra["home_done"] = homer.done
              extra["home_at_goal"] = homer.at_goal(q_leg, home_settle)
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


    except KeyboardInterrupt as e:
        print(f"\n[biped_emb] 중단 요청({e}) → 안전종료")
    except Exception as e:
        print(f"\n[biped_emb] ❌ 예외({type(e).__name__}: {e}) → 안전종료")
        raise
    finally:
        safe_shutdown(hw, jm)


if __name__ == "__main__":
    main()
