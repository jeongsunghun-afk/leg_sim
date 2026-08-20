#!/usr/bin/env python3
"""biped_deploy 가 내는 것과 **같은 모양**의 상태 파일을 20Hz 로 계속 쓴다.
   MockHw 로는 상태 발행이 안 되므로(이 변경 이전부터), 모니터 쪽만 따로 검증하기 위한 것.
   토크 값은 sim 기준표(15.0kg)에서 가져왔다 — 눈으로 볼 때 실제와 비슷하게 보이도록."""
import json, math, os, sys, time

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/biped_state.json"
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

# sim 기준표(TORSO_ADD_KG=1.1, 발행창 0.05s)
MU  = [-3.202, 3.354, -2.705, 5.660, 3.217, 3.429, -2.734, 5.690]
SD  = [ 0.028, 0.071,  0.169, 0.202, 0.029, 0.069,  0.164, 0.206]
LO  = [-3.597, 2.222, -5.388, 2.969, 2.776, 2.338, -5.152, 2.539]
HI  = [-2.722, 4.287, -0.124, 9.103, 3.587, 4.438, -0.044, 8.652]
QF  = [0.0, 3.68, -23.87, -59.81, 0.0, 3.68, -23.87, -59.81]

t0 = time.time()
n = 0
while time.time() - t0 < DUR:
    t = time.time() - t0
    w = math.sin(2 * math.pi * 0.4 * t)
    st = {
        "mode": "stand", "backend": "fake",
        "q_leg_deg":  [round(q + 0.2 * w, 3) for q in QF],
        "q_ch_deg":   [0.0] * 8,
        "dq_leg_dps": [round(2.0 * w, 3)] * 8,
        "tau_leg_nm": [round(m + 0.5 * s * w, 3) for m, s in zip(MU, SD)],
        "tau_cmd_nm": [round(m, 3) for m in MU],
        "tau_std_nm": [round(s, 3) for s in SD],
        "tau_min_nm": [round(v, 3) for v in LO],
        "tau_max_nm": [round(v, 3) for v in HI],
        "tau_win_n": 25,
        "kp_leg": [0.0] * 8, "kd_leg": [0.0] * 8,
        "rpy_deg": [0.0, 0.0, 0.0], "tilt_deg": 0.2, "loop_hz": 500.0,
        "motors_on": True,
        "health": ["ok"] * 8, "installed": [True] * 8,
        "n_ok": 8, "n_fault": 0, "n_dead": 0, "n_absent": 0, "n_installed": 8,
        "est_x": 0.002, "est_z": 0.444, "estop": False, "tilt_estop_ok": True,
        "qp_fail_pct": 0.0, "qp_K": 4, "qp_cerr": [0.006, 0.0, 0.0],
        "lat_comp_ms": 8.4, "lc_skip_pct": 0.0, "home_progress": 0.0,
        "err": [0] * 8,
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, OUT)          # 원자적 교체 — 모니터가 반쪽 파일을 읽지 않게
    n += 1
    time.sleep(0.05)              # 20Hz — biped_deploy 와 같은 발행 주기
print(f"발행 {n} 회 → {OUT}")
