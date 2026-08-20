"""app/biped_emb.py — biped 실기(Emb) 배포 메인 루프.

데이터흐름:  Backend.read → HwInterface(매핑·IMU변환) → [모드 디스패치] → Backend.write
  off   : 모터 limp
  jog   : per-axis 저속 위치 검증  ← 첫 딜리버러블(각축 확인)
  home  : 정해진 홈 자세로 S-curve 복귀(control/home.py) → 도달 후 그 자세 유지
  hold  : 현재자세 임피던스 홀드
  ★stand/walk 는 **이 앱이 처리하지 않는다** — 실기 배포는 C++ 기준이다.
    cpp/build/biped_deploy 가 담당한다(emb/NEXT_HW.md §9). 여기서 받으면 hold 로 되돌린다.
    ⚠모터 명령 writer 는 한 번에 하나만 — 둘을 동시에 띄우지 말 것.
GUI(../teleop_gui_biped.py — 실행은 ../run_gui_only.sh)와 JSON 채널(/tmp/biped_cmd.json)로
디커플. 상태는 /tmp/biped_state.json 발행.
  ★종전 주석의 `gui/teleop_emb.py` 는 **없는 파일**이었다 — 커밋 9454912 에서 각축 JOG 패널을
    teleop_gui_biped 로 통합하며 teleop_emb 를 제거했는데 참조가 안 고쳐져 있었다.

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
# STATE_PATH / publish_state 는 interface/state_pub.py 로 이관(2026-08-11).
# PACE 하니스도 같은 구현을 써야 뷰어 스키마가 갈라지지 않는다.
from state_pub import STATE_PATH, publish_state  # noqa: E402


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
    """명령 버퍼를 무여자(Kp=Kd=τ=0)로 만들고 끝낸다.

    ⚠⚠**이것이 "모터를 끈다" 는 뜻은 아니다** (2026-08-14 벤더 확인).
      SHM 에는 드라이브를 **실제로 끄는 경로가 없다** — MD80 의 `MOTOR_ENABLE(0x00)`·
      `CONTROL_SELECT(IDLE)` 은 CAN 프로토콜에는 있지만 **MCU 에 구현돼 있지 않다.**
      `ucMode`/`ucCommand` 필드는 있으나 의미가 정의돼 있지 않다(ucStatus 와 같은 상황).
      ⇒ 여기서 보장하는 것은 **"우리가 보내는 명령토크가 0"** 까지다.
        드라이브는 여전히 여자(energized) 상태이고, 종료 뒤에도 축이 안 풀릴 수 있다.
      ⇒ **확실한 해제는 물리 리셋/전원 차단뿐이다.** 실기 시험 전에 그 스위치 위치를
        손 닿는 곳에 확인해 둘 것. (벤더가 나중에 추가할 수도 있다고 함)

    ★기존 종료는 `hw.enable(False); hw.close()` 뒤에 "종료(limp)" 를 찍었지만
      **거짓이었다.** bridge_enable(0) 은 g_enabled 플래그만 바꿀 뿐 SHM 명령버퍼를
      건드리지 않는다(shm_bridge.cpp:115). 마지막에 쓴 kp40 명령이 SHM 에 그대로 남고
      Emb 는 그것을 1kHz 로 **영원히 재전송**한다. 즉 앱을 껐는데 모터는 계속 잡고 있다.
      "명령을 안 쓰는 것" 은 정지가 아니다 — Kp=Kd=0 을 **실제로 써야** 한다.
    ⇒ enable(False) 후 **반복 기록**해서 무여자 명령이 확실히 SHM 에 들어가게 한다.
    """
    try:
        hw.enable(False)
        for _ in range(25):                     # enable=False → 브리지가 kp=kd=0 으로 기록
            hw.write_limp()
            time.sleep(0.002)
        print("[biped_emb] 종료 — 명령토크 0(Kp=Kd=τ=0) 25회 기록 완료.\n"
              "           ⚠드라이브는 **여전히 여자 상태**다(SHM 에 disable 경로 없음).\n"
              "             축이 안 풀리면 물리 리셋/전원 차단이 유일한 해제 수단이다.")
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
    """축별 상태: absent(미장착) / dead(무통신) / fault(통신 O·ucStatus≠0) / ok(통신 O·정상).
       ★ucStatus 의미는 모터/펌웨어 정의 → 지금은 ≠0 을 에러로 간주(실기 확정 후 세분화).

    ★absent 를 왜 따로 두나: **Emb 는 모터가 없어도 8채널 전부 connected=1·ucStatus=0 을
      보고한다.** 그래서 미장착 6축이 전부 `ok` 로 잡혀 GUI 에 초록 LED 로 떴다 —
      **없는 모터가 "정상"으로 보이는 것**이라, 진짜 축이 죽었을 때 "8개 중 8개 정상"
      이라는 거짓 안심을 준다. 통신으로는 구분이 불가능하므로 config 선언으로 가른다.
    """
    conn, stat = raw.connected, raw.status
    out = []
    for i, ch in enumerate(jm.ch):
        if not bool(jm.installed[i]):
            out.append("absent")
        elif conn.size <= ch or not conn[ch]:
            out.append("dead")
        elif stat.size > ch and int(stat[ch]) != 0:
            out.append("fault")
        else:
            out.append("ok")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(EMB, "config", "biped_emb.yaml"))
    ap.add_argument("--mock", action="store_true", help="SHM 없이 데스크톱 데모")
    ap.add_argument("--T", type=float, default=1e12, help="최대 실행시간[s] (검증용)")
    # ★시작 모드 — 기본 hold. 근거는 아래 "인계" 주석 참조.
    ap.add_argument("--force", action="store_true",
                    help="다른 writer 가 떠 있어도 강행(권장하지 않음)")
    ap.add_argument("--start-mode", choices=["hold", "off"], default="hold",
                    help="시작 모드. hold=Emb 가 잡고 있던 자세를 그대로 인계(기본) · off=무여자")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    force_mock = args.mock or os.environ.get("MOCK") == "1"

    # ── ★중복 writer 차단 (2026-08-10 실기 사고) ─────────────────────────────
    #   모터 명령 writer 가 둘이면 **같은 SHM 에 1kHz 로 서로 다른 명령을 번갈아 쓴다.**
    #   실제로 코드 버전이 다른 두 인스턴스가 떠서 sign=−1 축이 +18° ↔ −20° 로 진동했다.
    #   문서에 "writer 는 하나만" 이라고 적어두는 것만으로는 못 막는다 — 여기서 거부한다.
    #   ⚠mock 은 SHM 을 안 쓰므로 예외(데스크톱에서 여러 개 띄워 시험할 수 있어야 한다).
    if not force_mock:
        import subprocess as _sp
        me = os.getpid()
        # 자기 자신과 **조상 프로세스 전부**를 제외한다 — 이 앱을 띄운 셸의 커맨드라인에
        # "app/biped_emb.py" 가 들어 있어 오탐이 난다(2026-08-10 실측).
        _anc, _p = set(), me
        for _ in range(24):
            try:
                with open(f"/proc/{_p}/stat") as f:
                    _p = int(f.read().split(") ", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                break
            if _p <= 1:
                break
            _anc.add(_p)
        others = []
        for pat in ("app/biped_emb.py", "biped_deploy", "mot_test", "actuator_test.py"):
            r = _sp.run(["pgrep", "-f", pat], capture_output=True, text=True)
            for pid in r.stdout.split():
                pid = int(pid)
                if pid == me or pid in _anc:
                    continue
                try:                                   # 자기 자신·pgrep 자체를 세지 않도록 확인
                    with open(f"/proc/{pid}/cmdline") as f:
                        cl = f.read().replace("\0", " ")
                except OSError:
                    continue
                if "pgrep" in cl or "grep" in cl:
                    continue
                # ★★**빌드 프로세스를 writer 로 세지 않는다** (2026-08-14).
                #   pgrep -f 는 명령줄 문자열로 찾으므로 `cmake --build --target biped_deploy`,
                #   `gmake … biped_deploy.dir`, `cc1plus … biped_deploy.cpp` 가 전부 걸린다.
                #   실기에서 실제로 그랬다 — biped_emb.py 를 못 띄우게 막았는데
                #   정작 writer 는 하나도 안 돌고 있었고, 사용자는 "pkill 이 안 먹었나" 로 읽었다.
                #   **가짜 경보는 진짜 경보를 무디게 한다.** 이 가드는 실기 사고
                #   (2026-08-10, 관절 +18°↔−20° 진동)를 막으려고 있는 것이라 정확해야 한다.
                _tool = ("cmake", "gmake", "make", "cc1plus", "c++", "g++", "ld", "ninja",
                         "sh", "bash", "/bin/sh")
                _exe0 = (cl.strip().split(" ", 1)[0] or "").split("/")[-1]
                if _exe0 in _tool:
                    continue
                if pat.split("/")[-1] in cl:
                    others.append((pid, cl.strip()[:90]))
        if others:
            print("✗ **모터 명령 writer 가 이미 실행 중이다.** 둘이 뜨면 SHM 에 서로 다른 명령을\n"
                  "  1kHz 로 번갈아 써서 관절이 진동한다(2026-08-10 실기 사고: +18° ↔ −20°).")
            for pid, cl in others:
                print(f"    PID {pid}: {cl}")
            print("  → 먼저 종료할 것:  kill " + " ".join(str(p) for p, _ in others))
            print("  (의도적으로 강행하려면 --force. 권장하지 않는다.)")
            if not args.force:
                return 1
            print("  ⚠ --force 로 강행한다. 명령이 꼬일 수 있다.")
    jm = JointMap(cfg)
    if getattr(jm, "cpl_dst", None):
        print("[biped_emb] 기구 커플링 보정 활성: "
              + ", ".join(f"{jm.names[d]} ← {jm.names[s]} × {c:+g}"
                          for d, s, c in zip(jm.cpl_dst, jm.cpl_src, jm.cpl_coef)))
    # ★Emb 는 ±180 을 넘는 명령을 **클램프가 아니라 래핑**한다(halGait.cpp:666-671).
    #   한계에서 멈추는 게 아니라 반대편으로 날아간다. jog 범위는 JointMap 이 기동 시
    #   예외로 막지만, 관절한계 박스까지 넓게 쓰면 넘는 축은 여기서 알린다.
    if getattr(jm, "wrap_warn", None):
        print("[biped_emb] ⚠ 관절한계 전 범위를 쓰면 채널각이 ±180 을 넘는 축이 있다 — "
              "Emb 가 래핑해 **반대편으로 튄다**:\n    "
              + ", ".join(f"{n} 최대 {v:.0f}°" for n, v in jm.wrap_warn)
              + "\n    지금 jog 범위 안에서는 안전하다. stand/walk 로 넓게 쓰기 전에 "
                "foot 한계를 좁히거나 offset 을 다시 잡을 것.")
    backend, be_name = make_backend(cfg, force_mock)
    hw = HwInterface(backend, jm, imu_deg=bool(cfg["shm"].get("imu_deg", True)))
    hw.init()

    cfg_dt = 1.0 / float(cfg["meta"]["ctrl_hz"])
    jogger = Jogger(jm, cfg_dt, float(cfg["jog"]["max_speed_dps"]))
    fsm = FSM.ModeFSM(FSM.HOLD if args.start_mode == 'hold' else FSM.OFF)
    hcfg = cfg.get("home", {})
    homer = HomeTrajectory(jm, cfg_dt,
                           hcfg.get("q_deg", [0.0] * jm.n_leg),
                           float(hcfg.get("max_speed_dps", 15.0)),
                           float(hcfg.get("max_acc_dps2", 30.0)),
                           float(hcfg.get("min_time_s", 0.6)))
    home_settle = float(hcfg.get("settle_deg", 0.5))
    home_warned = False        # ★HOME 도달실패 경고를 진입당 한 번만 낸다
    # ★홈 목표가 jog 안전한계에 잘렸으면 조용히 넘어가지 않는다 — "홈에 갔다" 는 보고와
    #   실제 자세가 어긋나게 되고, 그 어긋남은 다음 모드의 시작자세가 된다.
    for nm, want, got in homer.clamped:
        print(f"[biped_emb] ⚠ home.q_deg[{nm}] {want:+.1f}° → {got:+.1f}° 로 클램프 "
              f"(jog 안전한계 = 관절한계×{jm.jog_frac}). 이 자세로는 홈에 도달하지 못한다.")
    settle = float(cfg["jog"]["settle_deg"])
    tilt_estop = float(cfg["safety"]["tilt_estop_deg"])
    watchdog_s = float(cfg["safety"]["watchdog_ms"]) / 1000.0
    _es = cfg["safety"]
    estop_auto_max = int(_es.get("estop_auto_max", 3))          # 창 안 허용 자동해제 횟수
    estop_auto_window_s = float(_es.get("estop_auto_window_s", 30.0))
    estop_cooldown_s = float(_es.get("estop_cooldown_s", 1.0))  # 트립 직후 연타 방지
    vel_trip_ms = float(_es.get("vel_trip_ms", 20.0))           # 속도 트립 디바운스
    tau_frac = float(cfg["safety"]["tau_max_frac"])
    # ★토크/속도 트립(시험 하네스 emb/pace/hwio.py 에서 승격). 미설정이면 하네스 기본값.
    tau_trip_nm  = float(cfg["safety"].get("tau_trip_nm",  8.0))
    tau_trip_ms  = float(cfg["safety"].get("tau_trip_ms",  50))
    vel_trip_dps = float(cfg["safety"].get("vel_trip_dps", 200.0))
    tau_over_t0  = None                      # 토크 연속초과 시작시각(None=정상)

    jog_goal = np.zeros(jm.n_leg)            # GUI 축별 목표각[deg]
    hold_leg = np.zeros(jm.n_leg)            # hold 목표 [**모델각 deg**]
    walk_cmd = {"v": 0.0, "vy": 0.0, "w": 0.0, "body_h": 0.38}
    # ★model_ctrl(Python 모델기반)은 더 이상 쓰지 않는다 — 배포는 C++ 기준(§9).
    model_warned = False                     # stand/walk 거부 안내를 한 번만 출력

    print(f"[biped_emb] backend={be_name} · ctrl_hz={cfg['meta']['ctrl_hz']} · CMD={CMD_PATH}")
    print("            모드: off/jog/hold(=지금) · stand/walk(=jog 검증 후). GUI로 조종.")

    # ── ★Emb → 이 앱 인계 (2026-08-07) ────────────────────────────────────
    #   Emb 는 기동 시 4.5초 램프로 전 관절을 0°로 보낸 뒤 **그 자세를 잡고 있다**
    #   (halGait.cpp:694-711, kp 100/50/50/20·kd 5). 그 상태에서 이 앱이 뜬다.
    #
    #   ⚠종전엔 무조건 off 로 시작했다 — 그러면 브리지가 kp=kd=0 을 SHM 에 쓰고
    #     Emb 가 그걸 **클램프 없이 통과**시켜(commGait.cpp:190 memcpy) 모터가 풀린다.
    #     즉 우리 앱이 **능동적으로 잡고 있던 걸 놓아 다리를 떨어뜨렸다.**
    #     다리 미장착 시절엔 무해했지만(잡을 게 없었다) 조립 후에는 hip 중력토크
    #     4.96 Nm 로 실제 낙하한다.
    #
    #   ★원칙이 뒤집힌 지점: "사람이 버튼을 누르기 전에 전류가 흐르면 위험" 이라는
    #     기존 근거는 **Emb 가 이미 전류를 흘리고 있는 상황**에선 성립하지 않는다.
    #     인계 시 움직임이 0 인 쪽(hold)이 더 안전하다 — 우리가 푸는 쪽이 오히려
    #     예상 못 한 움직임을 만든다.
    #   ⇒ 기본을 hold 로 바꾸고, 무여자로 시작하려면 --start-mode off 를 쓴다.
    _raw0 = hw.read()
    # ── ★installed_channels 불일치 감지 (2026-08-07) ──────────────────────────
    #   `installed_channels` 는 사람이 선언하는 값이라 **조용히 낡는다** — 실제로
    #   8축을 다 연결하고도 config 가 [0,4] 로 남아 LED 가 안 바뀌는 일이 있었다.
    #   Emb 는 모터가 없어도 connected=1 을 주므로 그걸로는 판정할 수 없다. 대신
    #   **엔코더 값**을 본다: 미장착 채널은 0.000 에서 꿈쩍도 안 하고, 장착 채널은
    #   비영값이거나 노이즈로 미세하게 흔들린다.
    #   ⚠휴리스틱이다(진짜로 0.000 에 정지한 축은 오판할 수 있다) → **자동으로 고치지 않고
    #     경고만** 한다. 선언은 사람이 하는 게 맞다.
    try:
        _s0 = hw.read().q_deg.copy()
        time.sleep(0.15)
        _s1 = hw.read().q_deg.copy()
        _sus = []
        for _i, _ch in enumerate(jm.ch):
            if bool(jm.installed[_i]):
                continue
            _live = abs(float(_s1[_ch])) > 1e-6 or abs(float(_s1[_ch]) - float(_s0[_ch])) > 1e-9
            if _live:
                _sus.append(f"{jm.names[_i]}(ch{_ch}, {_s1[_ch]:+.2f}°)")
        if _sus:
            print(f"[biped_emb] ⚠ installed_channels 가 낡았을 수 있다 — 미장착으로 선언됐는데 "
                  f"**엔코더가 살아 있는** 축: {', '.join(_sus)}\n"
                  f"    config/biped_emb.yaml 과 pace/spec.yaml 의 meta.installed_channels 를 갱신할 것.\n"
                  f"    (그 전까지 GUI LED 는 어두운 채로 남고 ok 카운트에도 안 잡힌다)")
    except Exception:
        pass

    if fsm.mode == FSM.HOLD:
        hold_leg = jm.ch_to_q_joint(_raw0.q_deg)   # ★측정각 래치(모델각) — 인계 순간 움직임 0
        hw.enable(True)
        # ★`q_leg=` 라고 쓰고 채널각을 찍고 있었다(2026-08-10 수정). offset 이 전부 0 이던
        #   동안은 두 값이 같아 드러나지 않았지만, 영점 캘리브레이션 후에는 완전히 다른
        #   숫자다(예 HL_foot 채널 −84.8° = 모델 0°). 인계 자세를 눈으로 확인하는 자리라
        #   여기서 틀리면 "홈 자세가 이상하다" 는 오진으로 바로 이어진다. 둘 다 찍는다.
        print(f"[biped_emb] 인계: hold 로 시작 — Emb 가 잡고 있던 자세를 그대로 유지한다.\n"
              f"            q_leg(모델각)={np.round(hold_leg, 2).tolist()} deg\n"
              f"            q_ch (채널각)={np.round(_raw0.q_deg[jm.ch], 2).tolist()} deg\n"
              f"            (무여자로 시작하려면 --start-mode off)")
    else:
        hw.enable(False)
        print("[biped_emb] off 로 시작 — ⚠Emb 가 잡고 있던 자세가 풀려 다리가 떨어진다.")

    t0 = time.perf_counter(); k = 0; last_cmd_t = t0; last_pub = 0.0
    estop_latched = False
    wd_tripped = False
    estop_reason = None        # ★래치 사유를 남긴다 — 상태로 발행해 GUI 가 보여준다
    estop_hist = []            # 최근 트립 시각들(자동해제 남용 차단용)
    estop_sticky = False       # ★반복 트립으로 **자동해제를 거둬들인** 상태(명시 OFF 필요)
    vel_over_t0 = None         # ★속도 초과 시작시각(디바운스용)
    estop_log = []             # ★최근 트립 (t, 사유). **해제해도 지우지 않는다** —
                               #   지우면 원인을 볼 방법이 사라진다(2026-08-12 실수).
    # ★tilt E-stop 은 IMU 가 있어야 동작한다. 이 로봇은 현재 SHM IMU 가 전부 0 이라
    #   tilt 가 항상 0 으로 계산되어 **E-stop 이 사실상 비활성**이다. 조용히 넘어가면
    #   보호장치가 있다고 착각하게 되므로 기동 시 명시적으로 경고한다.
    _imu0 = hw.read().imu_rpy_deg
    if float(np.max(np.abs(_imu0))) < 1e-9:
        print("[biped_emb] ⚠⚠ IMU 값이 전부 0 → **tilt E-stop 비활성**. "
              "IMU 배선/펌웨어 확인 전까지 기울기 보호가 없다고 간주할 것.")
    prev_loop_t = t0
    pace_warned = False          # 루프 밀림 경고 1회만
    hz_ema = float(cfg["meta"]["ctrl_hz"])
    dt_buf = []          # 루프 주기 표본(지터 통계용)

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
                      jogger.reset(q_leg); new_mode = FSM.HOLD
                  # ★stand/walk 는 이 앱의 역할이 아니다 — **실기 배포는 C++ 기준**(NEXT_HW.md §9).
                  #   ⚠종전엔 model_ctrl(Python) 분기가 살아 있었고, Pi 에 mujoco·qpsolvers 가
                  #     없어 import 실패로 hold 에 떨어졌다. 그건 **설계된 차단이 아니라 우연**이라
                  #     그 둘을 설치하는 순간 열린다. 의존성에 안전을 맡기지 않고 여기서 막는다.
                  #   ★FSM 진입 **이전**에 치환한다. 진입 후에 되돌리면 GUI 가 20ms 마다
                  #     stand 를 재전송할 때 hold→stand→hold 재전이가 무한 반복돼 로그가 폭주한다.
                  if new_mode in FSM.MODEL_BASED:
                      if not model_warned:
                          model_warned = True
                          print(f"[biped_emb] ⛔ '{new_mode}' 는 이 앱이 처리하지 않는다 → hold 유지.\n"
                                f"    실기 stand/walk 는 C++ 배포 바이너리가 담당한다:\n"
                                f"      cd {os.path.join(BIPED, 'cpp')} && "
                                f"LD_LIBRARY_PATH=$HOME/mujoco/lib ./build/biped_deploy\n"
                                f"    ⚠ 모터 명령 writer 는 한 번에 하나만 — 이 앱을 먼저 종료할 것.\n"
                                f"    ※지연보상은 그 바이너리에서 **기본 켜짐**(8.4ms 실측·운동학 외삽).\n"
                                f"      기동 로그의 '지연보상' 줄로 확인할 것. 끄려면 LAT_COMP_MS=0.",
                                flush=True)
                      new_mode = FSM.HOLD
                  elif new_mode != FSM.HOLD:
                      model_warned = False        # 다른 모드로 갔다 오면 다시 한 번 알린다
                  # ★E-stop 래치 해제는 명시적 off 명령으로만. 그 전까지 모드변경 무시.
                  # ── ★래치 해제 정책 (2026-08-12, 사용자 결정: "②로 진행") ─────
                  #   종전: **명시적 OFF 로만** 해제. 그래서 트립이 한 번 걸리면
                  #     off↔home 을 두 번 눌러야 했고, 그 사실이 어디에도 안 보였다.
                  #   지금: **조건이 실제로 사라졌으면** 다음 모드명령에서 자동 해제한다.
                  #     "OFF 를 눌렀다" 는 사실보다 "결함이 없다" 를 확인하는 쪽이 맞다.
                  #   ⚠단 **반복 트립은 막는다** — 결함이 계속 재발하는데 사람이 계속
                  #     누르면 여자/무여자를 반복하며 다리가 떨린다. 창 안에서 N회를
                  #     넘기면 자동해제를 거둬들이고(sticky) 명시적 OFF 를 요구한다.
                  if estop_latched:
                      _now = loop_t
                      estop_hist[:] = [t_ for t_ in estop_hist
                                       if _now - t_ <= estop_auto_window_s]
                      if new_mode == FSM.OFF:
                          estop_latched = False; estop_sticky = False
                          print(f"[biped_emb] E-stop 래치 해제(off 수신) — 재무장 가능"
                                + (f"  [사유였던 것: {estop_reason}]" if estop_reason else ""))
                      elif estop_sticky:
                          if new_mode != fsm.mode:
                              print(f"[biped_emb] ⚠ '{new_mode}' 무시 — **반복 트립으로 자동해제가 "
                                    f"꺼졌다**({len(estop_hist)}회/{estop_auto_window_s:.0f}s). "
                                    f"원인({estop_reason})을 확인하고 OFF 를 누를 것.", flush=True)
                          new_mode = FSM.OFF
                      else:
                          # 조건이 지금도 살아 있는가 — 여자 전이므로 토크는 볼 수 없고
                          # **속도**만 유효하다. 정지해 있으면 재무장을 허용한다.
                          _v = float(np.max(np.abs(raw.dq_dps))) if raw.dq_dps.size else 0.0
                          _cool = (_now - estop_hist[-1]) if estop_hist else 1e9
                          if _v > vel_trip_dps * 0.25:
                              if new_mode != fsm.mode:
                                  print(f"[biped_emb] ⚠ '{new_mode}' 보류 — 아직 움직인다"
                                        f"({_v:.0f}dps). 멈춘 뒤 다시 누를 것.", flush=True)
                              new_mode = FSM.OFF
                          elif _cool < estop_cooldown_s:
                              new_mode = FSM.OFF        # 트립 직후 연타 방지(조용히)
                          else:
                              estop_latched = False
                              print(f"[biped_emb] E-stop 자동해제 — 결함조건 소멸"
                                  f"(속도 {_v:.0f}dps, 트립 후 {_cool:.1f}s). "
                                  f"[사유였던 것: {estop_reason}]  누적 {len(estop_hist)}"
                                  f"/{estop_auto_max}회", flush=True)
                              if len(estop_hist) >= estop_auto_max:
                                  estop_sticky = True
                                  print(f"[biped_emb] ⛔ {estop_auto_window_s:.0f}초 안에 "
                                        f"{len(estop_hist)}회 트립 — **자동해제를 끈다**. "
                                        f"다음부터는 OFF 를 눌러야 풀린다. 원인을 볼 것.",
                                        flush=True)
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
                          home_warned = False
                          # ★측정각에서 궤적을 만든다(명령각이 아니라). 부하로 처진 상태에서
                          #   명령각을 기점으로 잡으면 첫 틱에 그 편차만큼 계단이 나간다.
                          T = homer.start(q_leg)
                          print(f"[biped_emb] home 복귀 시작 — {T:.2f}s "
                                f"(v≤{hcfg.get('max_speed_dps', 15.0)}dps, "
                                f"a≤{hcfg.get('max_acc_dps2', 30.0)}dps²)  "
                                + "  ".join(f"{jm.names[i]}{q_leg[i]:+.1f}→{homer.q_home[i]:+.1f}"
                                            for i in range(jm.n_leg)), flush=True)
                      if fsm.entered(FSM.HOLD):
                          hold_leg = q_leg.copy()          # 모델각

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
          #         게다가 hold 목표를 매 틱 재캡처해 목표가 낙하를 따라가 복원토크가 0 이 됐다.
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
                      estop_reason = (f"토크 ch{ch} {tau_pk:.2f}>{tau_trip_nm}Nm "
                                      f"{tau_trip_ms}ms 연속")
                      estop_log.append((round(loop_t - t0, 2), estop_reason))
                      del estop_log[:-10]
                      print(f"[biped_emb] ⛔ E-STOP: {estop_reason} → limp·래치")
                      print( "            ★래치 중에는 OFF 외의 모드요구가 전부 무시된다. "
                             "OFF 를 눌러 해제할 것.")
                      estop_latched = True; estop_hist.append(loop_t); fsm.set(FSM.OFF); hw.enable(False)
              else:
                  tau_over_t0 = None                # 한 틱이라도 정상이면 타이머 리셋
              # ── 속도 트립 (2026-08-12: **디바운스 추가**) ─────────────────
              #   종전엔 "폭주는 지연시킬 이유가 없다" 며 **한 샘플**로 트립했다.
              #   그런데 실측 속도잡음이 15dps RMS 이고, **foot 채널은 특히 취약하다**:
              #       dq_ch_foot = (dq_foot + coef·dq_calf)·sign·k   (k=1.2)
              #     두 축의 잡음이 합쳐진 뒤 1.2배 된다.
              #   실제로 그렇게 걸렸다 — ch3 209dps 트립. 그런데 홈복귀의 **공칭** 첨두
              #   채널속도는 22dps 로 한계(200)와 거리가 멀다 ⇒ 궤적이 아니라 순간 튐이다.
              #   ⇒ 토크와 같은 방식으로 **연속 초과**만 트립한다. 진짜 폭주는 지속되고
              #     스파이크는 안 지속된다. vel_trip_ms 는 짧게(20ms) 둬서 응답성을 지킨다.
              if vel_pk > vel_trip_dps:
                  if vel_over_t0 is None:
                      vel_over_t0 = loop_t
              else:
                  vel_over_t0 = None
              if ((not estop_latched) and vel_over_t0 is not None
                      and (loop_t - vel_over_t0) * 1000.0 >= vel_trip_ms):
                  ch = int(np.argmax(np.abs(raw.dq_dps)))
                  estop_reason = (f"속도 ch{ch} {vel_pk:.0f}>{vel_trip_dps}dps "
                                  f"{vel_trip_ms:.0f}ms 연속")
                  estop_log.append((round(loop_t - t0, 2), estop_reason))
                  del estop_log[:-10]
                  print(f"[biped_emb] ⛔ E-STOP: {estop_reason} → limp·래치")
                  print( "            ★래치 중에는 OFF 외의 모드요구가 전부 무시된다. "
                         "OFF 를 눌러 해제할 것.")
                  estop_latched = True; estop_hist.append(loop_t); fsm.set(FSM.OFF); hw.enable(False)
          else:
              tau_over_t0 = None; vel_over_t0 = None

          if estop_latched:
              hw.enable(False)                    # 래치 동안 계속 무여자 강제

          # ── 축별 health = 임베디드 보고 반영(통신+ucStatus). ok/fault/dead. (제어 아님, 모니터) ──
          health = axis_health(raw, jm)
          # ★n_ok/n_fault/n_dead 의 분모는 **실장축**이다. 미장착을 분모에 넣으면
          #   "8개 중 2개 정상" 처럼 보여서 정상 상태가 고장으로 읽힌다.
          extra = {"health": health, "installed": [bool(x) for x in jm.installed],
                   "n_ok": health.count("ok"), "n_fault": health.count("fault"),
                   "n_dead": health.count("dead"), "n_absent": health.count("absent"),
                   "n_installed": int(jm.installed.sum()),
                   # ★raw 채널각(=드라이버 채널각, sign·offset 적용 **전**). 관절 순서로 정렬.
                   #   영점 캘리브레이션은 이 값이 있어야 한다 — offset 은 채널각 단위이고
                   #   모델각만 보고 있으면 offset≠0 이 된 뒤로는 역산이 헷갈린다.
                   #   diag/calib_zero.py 가 이 필드를 쓴다.
                   "q_ch_deg": [round(float(raw.q_deg[c]), 2) for c in jm.ch]}

          # ★실측 루프주기 — jog/home 의 속도·궤적 제한을 **호출 횟수가 아니라 실제 시간**
          #   기준으로 걸기 위해 넘긴다(캐치업 폭주로 한계가 뚫리는 것 방지).
          dt_meas = loop_t - prev_loop_t if prev_loop_t else cfg_dt
          if dt_meas <= 0:
              dt_meas = cfg_dt

          # ── 모드 디스패치 (전 채널 명령; 미배선/죽은 축은 임베디드가 흡수) ──
          if fsm.mode == FSM.OFF:
              hw.write_limp()          # enable=False → 브리지가 kp=kd=0 기록. 위치는 측정각 유지
          elif fsm.mode == FSM.JOG:
              # ★q_leg 를 함께 넘긴다 — 늘어진 자세(범위 밖)에서 진입해도 계단이 안 나간다
              hw.write_jog(jogger.step(jog_goal, dt_meas), q_leg)
              extra["jog_at_goal"] = jogger.at_goal(jog_goal, settle)
          elif fsm.mode == FSM.HOME:
              # ★워치독 트립 중에는 궤적을 현재 측정각으로 계속 재기준한다.
              #   트립 = 무여자(limp) 라 다리가 중력으로 처지는데, 그동안 궤적 시간만
              #   흘려보내면 명령 복귀 순간 "처진 실제 위치" 와 "그새 진행된 목표" 사이의
              #   편차가 통째로 계단 입력이 된다(kp40 → 1° 당 0.7Nm). 재기준해 두면
              #   복귀 시 처진 자리에서 홈까지 궤적을 처음부터 다시 그린다.
              if wd_trip:
                  homer.start(q_leg)
              # ★write_jog 가 아니라 write_home — jog 클램프가 계단을 만든다(2026-08-12)
              hw.write_ramped(homer.step(dt_meas), q_leg)
              extra["home_progress"] = round(homer.progress, 3)
              extra["home_done"] = homer.done
              _at = homer.at_goal(q_leg, home_settle)
              extra["home_at_goal"] = _at
              # ★궤적이 끝났는데 도달 못 했으면 **한 번 크게 알린다** (2026-08-11).
              #   종전엔 at_goal 을 계산해 발행만 하고 아무도 안 봤다. 그래서 홈복귀가
              #   2° 못 맞추고 끝나도 조용했고, 그 자세에서 영점을 잡으면 그 오차가
              #   그대로 offset 으로 박혔다(커플링 때문에 foot 은 calf 오차까지 함께).
              if homer.done and not _at and not home_warned:
                  home_warned = True
                  _e = q_leg[:jm.n_leg] - homer.q_home
                  _w = [f"{jm.names[i]}{_e[i]:+.2f}" for i in range(jm.n_leg)
                        if abs(_e[i]) > home_settle]
                  print(f"[biped_emb] ⚠ HOME 궤적 종료 — **도달 실패**(허용 {home_settle}°): "
                        + " ".join(_w))
                  print( "            게인 부족·마찰·기구 간섭 중 하나다. "
                         "★이 상태로 영점을 잡으면 오차가 offset 에 박힌다.")
              extra["home_miss"] = [round(float(v), 2) for v in
                                    (q_leg[:jm.n_leg] - homer.q_home)]
          elif fsm.mode == FSM.HOLD:
              hw.write_hold(hold_leg)
          # ★stand/walk 디스패치는 없다 — 진입에서 hold 로 되돌린다(위 참조).
          #   실기 모델기반 제어는 cpp/build/biped_deploy 담당(NEXT_HW.md §9).

          # ── 상태 발행(~20Hz) + 루프 주기 통계 ──
          #   ★종전엔 hz_ema = EMA(1/period) 를 발행했는데 이건 **편향된 지표**다.
          #     주기가 들쭉날쭉하면 1/period 의 평균은 실제 평균 주파수보다 항상 높게 나온다
          #     (산술평균 ≥ 조화평균). 짧은 틱 하나가 긴 틱 하나보다 지표를 더 크게 끌어올린다.
          #     그래서 목표 500Hz 인데 화면에 700Hz 가 떠 "CPU 때문에 빨라졌나?" 로 오해를 샀다.
          #   ⇒ 진짜 평균 = 표본수/총경과시간 을 쓰고, 지터는 **주기 분위수**로 따로 낸다.
          #     평균만 보면 지터가 안 보이고, 지터는 제어 품질에 직접 영향을 준다.
          period = loop_t - prev_loop_t; prev_loop_t = loop_t
          if period > 0:
              dt_buf.append(period)
              if len(dt_buf) > 2000:            # 최근 ~4초(500Hz 기준)
                  del dt_buf[:len(dt_buf) - 2000]
          # ★발행 주기 0.05(20Hz) → 0.02(50Hz). 표시 지연의 병목이 여기였다:
          #   명령은 내부 500Hz 로 실행되는데 화면은 20Hz 라 "로봇은 빠른데 표시가 느리다".
          #   ★실측 — 지터 비용은 없다(안정 후 30초, 2000샘플):
          #       20Hz: p50 1.999 · p95 2.691 · max 11.276 ms · 평균 500.0Hz
          #       50Hz: p50 1.999 · p95 2.501 · max  7.884 ms · 평균 500.0Hz
          #     /tmp 가 tmpfs(RAM)라 json.dump+os.replace 가 충분히 싸다.
          #   ⚠기동 직후 ~15초는 과도구간이라 max 가 300ms 까지 튄다(SHM init). 그때 재면
          #     "50Hz 가 지터를 만든다" 는 오판을 하게 된다 — 반드시 안정 후에 잴 것.
          #   ⚠더 올릴 거면(>100Hz) 발행을 별도 스레드로 빼라. 루프 안 I/O 는 언젠가 문제가 된다.
          if loop_t - last_pub > 0.02:
              if dt_buf:
                  sdt = sorted(dt_buf)
                  n = len(sdt)
                  hz_true = n / sum(sdt)                       # ★편향 없는 실제 평균 주파수
                  extra["dt_ms_p50"] = round(sdt[n // 2] * 1e3, 3)
                  extra["dt_ms_p95"] = round(sdt[min(n - 1, int(n * 0.95))] * 1e3, 3)
                  extra["dt_ms_max"] = round(sdt[-1] * 1e3, 3)
                  extra["dt_ms_nom"] = round(cfg_dt * 1e3, 3)
                  hz_ema = hz_true
              extra["write_fail"] = int(getattr(hw, "n_write_fail", 0))
              # ★모니터링: 측정 vs 명령 (2026-08-13). 종전엔 **위치 실측 하나만** 나갔다.
              #   그래서 "명령대로 따라오는가" 를 화면에서 볼 방법이 아예 없었다 —
              #   속도·토크는 SHM 에서 이미 읽고 있었는데 발행만 안 하고 있었다.
              #   단위 통일: 전부 **모델각(deg·deg/s)** · **관절토크(Nm)**. 채널각 아님.
              #   ⚠명령값은 클램프·램프를 **거친 뒤**의 값이다(hw._log_cmd). 상위 목표를
              #     그대로 쓰면 "안 따라온다" 는 오진이 난다 — 잘린 건 하드웨어 탓이 아니다.
              try:
                  extra["dq_leg_dps"] = [round(float(v), 2) for v in hw.dq_leg_dps()]
                  extra["tau_leg_nm"] = [round(float(v), 3) for v in hw.tau_leg_nm()]
                  extra["q_cmd_deg"]  = [round(float(v), 2) for v in hw.cmd_q_deg]
                  extra["dq_cmd_dps"] = [round(float(v), 2) for v in hw.cmd_dq_dps]
                  extra["tau_cmd_nm"] = [round(float(v), 3) for v in hw.cmd_tau_nm]
                  extra["kp_leg"]     = [round(float(v), 1) for v in hw.cmd_kp]
                  extra["kd_leg"]     = [round(float(v), 2) for v in hw.cmd_kd]
                  # ★ucStatus 원값 (2026-08-13). 종전엔 `health` 문자열로만 나가서
                  #   **"fault" 라는 것만 알고 왜인지는 못 봤다.**
                  #   ucStatus = MD80 DEFAULT_RESPONSE 의 **ERROR VECTOR 하위 8bit**
                  #   (벤더 확인 2026-08-14. 아직 정제 전이라 원값 그대로 실린다).
                  #   ⚠상위 8bit 는 MCU 에서 잘려 안 온다 — 거기 있는 비트는 못 본다.
                  #   래치오프 순간의 이 값이 원인 판별의 유일한 단서다. 반드시 원값으로 남긴다.
                  _rs = getattr(hw, "_raw", None)
                  if _rs is not None and len(getattr(_rs, "status", [])) >= jm.n_leg:
                      extra["stt_raw"] = [int(_rs.status[c]) for c in jm.ch]
              except Exception:
                  pass          # 발행 실패가 제어를 멈추면 안 된다(state_pub 와 같은 원칙)
              # ★래치·워치독을 **밖으로 드러낸다** (2026-08-12).
              #   래치되면 OFF 외의 모든 모드요구가 OFF 로 되돌려진다. 그런데 그 사실이
              #   어디에도 안 나가서, 사용자에겐 "HOME 을 눌러도 안 바뀐다" 로만 보였다.
              #   실제로 그 증상을 보고받았고, 원인을 찾는 데 한참 걸렸다.
              extra["estop_latched"] = bool(estop_latched)
              extra["estop_reason"] = estop_reason
              extra["wd_trip"] = bool(wd_tripped)
              extra["estop_sticky"] = bool(estop_sticky)
              extra["estop_recent"] = len(estop_hist)
              extra["estop_log"] = estop_log[-5:]      # ★해제돼도 남는다(원인 추적용)
              publish_state(fsm.mode, q_leg, rpy, hz_ema, fsm.mode != FSM.OFF, be_name, extra)
              last_pub = loop_t

          # ── 실시간 페이싱 ──
          #   ★캐치업 폭주 방지 (2026-08-07 실측으로 발견).
          #     종전엔 `lag>0 이면 sleep` 뿐이라, 루프가 밀리면(Pi load 5.14/4코어 관측)
          #     그 다음 수십 틱이 **sleep 없이 연속 실행**됐다. loop_hz 가 500 → 4452 로
          #     튄 게 그 흔적이다(실제로 빨리 돈 게 아니라 몰아서 돈 것).
          #   ⚠이게 표시 문제가 아닌 이유: jog 속도제한·home S-curve 는 "호출이 dt 간격으로
          #     온다"는 전제로 걸려 있다. 25틱을 2ms 에 몰아 돌면 jog 가 20dps → **500dps** 로
          #     뚫리고 home 궤적은 빨리감기 된다. 즉 안전한계가 조용히 무력화된다.
          #   ⇒ 일정 이상 밀리면 스케줄을 현재로 재동기해 폭주 구간 자체를 없앤다.
          #     (남은 영향은 "가끔 한 틱이 느려짐" = 명령이 느려지는 쪽이라 안전하다)
          k += 1
          now = time.perf_counter()
          lag = t0 + k * cfg_dt - now
          if lag < -10 * cfg_dt:                    # 10틱(20ms) 이상 밀림 → 재동기
              if not pace_warned:
                  pace_warned = True
                  print(f"[biped_emb] ⚠ 루프가 {-lag*1e3:.0f}ms 밀렸다 — {cfg['meta']['ctrl_hz']}Hz "
                        f"를 못 지키고 있다. CPU 부하 확인(현재 Pi 4코어).\n"
                        f"    캐치업 폭주는 막았으나, 지속되면 ctrl_hz 를 낮추는 게 낫다.", flush=True)
              t0 = now - k * cfg_dt
          elif lag > 0:
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
