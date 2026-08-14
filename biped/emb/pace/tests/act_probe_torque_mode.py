#!/usr/bin/env python3
"""act_probe_torque_mode.py — 드라이버가 **순수 토크모드**(Kp=Kd=0, fTorque만)를 받는가?

★왜 중요한가: WBIC/MPC 는 관절 **토크**를 출력한다. 드라이버가 순수 토크를 받으면
  컨트롤러 출력을 그대로 넣을 수 있다. 안 받으면 위치+게인으로 변환해야 하는데,
  그건 근본적으로 다른(그리고 열등한) 인터페이스다 — 임피던스 뒤에 토크를 숨기는 꼴이라
  접촉힘 제어의 권한이 드라이버로 넘어간다.

측정 방법 — 토크 램프로 파단(breakaway)을 찾는다:
  Kp=Kd=0 으로 두고 tau_ff 를 **중력 바이어스**부터 아주 천천히 올린다.

  ★왜 0 이 아니라 바이어스인가 (2026-08-12)
    0 부터 올리면 **중력이 큰 축은 아예 못 잰다**:
      hip  중력 5.25 Nm(채널) — 파단 0.65 를 재려면 +dir 에 5.90 Nm 이 필요한데
           tau_max 가 1.4 라 도달 자체가 불가능하다. 자세를 바꿔도 안 준다
           (다리를 좌우로 여는 방향이라 thigh·calf 각과 무관).
      calf 중력 0.81 Nm > 파단 0.65 — **kp=kd=0 을 주는 순간 이미 미끄러진다.**
           τ_ff=0 인데 파단이 잡히니 측정이 성립하지 않고, 하위 링크가 떨어진다.
    ⇒ 램프 시작점을 그 축의 **중력토크**로 옮긴다. 그러면 양방향 모두
        τ_break± = bias ± τ_friction  이고 마찰은 (τ⁺−τ⁻)/2 로 깨끗이 분리된다.
      부수효과: 지금까지 +dir/−dir 비대칭에 섞여 있던 중력이 사라진다.
    ⚠바이어스는 **먼저 램프로 올린 뒤** 정착시키고 프로브를 시작한다. 계단으로 주면
      그 자체가 축을 움직여 파단으로 오검출된다.
    ⚠모델 중력이 틀리면 축이 표류한다. 바이어스 인가 후 이동량을 확인하고,
      move_thresh 를 넘으면 **바이어스가 틀린 것**이라 보고 중단한다.
    · 어느 지점에서 관절이 움직이기 시작하면 → **순수 토크모드 지원**.
      그 파단토크는 위치모드로 잰 정지마찰(τ_s)과 일치해야 한다 → 교차검증까지 된다.
    · tau_max 까지 올려도 미동이면 → fTorque 가 무시된다(= 순수 토크모드 미지원).

⚠ 안전 — 위치+게인 모드와 근본적으로 다르다:
  임피던스 모드는 토크가 `Kp·err` 로 자기제한되지만, 순수 토크는 **제한이 없다**.
  마찰(정지 0.71 Nm)을 넘는 순간 관절이 계속 가속한다. 다리 미장착 관성 0.0375 kg·m²
  에서 1 Nm 면 α=26.7 rad/s². 그래서:
    · tau_max 를 정지마찰의 2배 이내로 (기본 1.4 Nm)
    · 움직임 감지 즉시 토크 0 (한 틱 = 5 ms 안에)
    · 위치·속도 한계는 hwio._check 가 매 틱 강제
    · 어떤 경로로 끝나든 limp
  **다리가 없는 지금이 이 시험을 하기에 가장 안전한 시점이다.**
"""
from __future__ import annotations

import time

import numpy as np

from hwio import DEG


def probe_torque_mode(hw, spec, joint, log=print) -> dict:
    ch = int(joint["ch"])
    name = joint["name"]
    cfg = spec.get("torque_mode", {})
    ramp = float(cfg.get("ramp_nm_per_s", 0.25))
    move_deg = float(cfg.get("move_thresh_deg", 0.30))
    trials = int(cfg.get("trials", 2))
    # ★중력 바이어스 — 램프 시작점. 채널토크[Nm], 부호 포함.
    #   ★**상수가 아니라 위치의 함수**다 (2026-08-12). 게인을 놓으면 축이 자기 평형점으로
    #     흘러가는데 거기서는 중력이 이미 다르다:
    #       HL_hip 0° 에서 5.25 Nm → 실제로 멈춘 −11° 에서는 4.09 Nm (1.16 Nm 차이)
    #     그 차이 때문에 τ_break 이 마찰이 아닌 값이 됐다(리포트 0.454/0.584 은 마찰이
    #     아니라 '어긋난 bias 에서 파단까지의 거리' 였다).
    #   ⇒ 표(tools/gen_grav_table.py 생성)를 **채널각으로** 보간한다. 런타임 변환 없음.
    _tbl = (cfg.get("tau_grav_table") or {}).get(ch)
    _fallback = float((cfg.get("tau_bias_by_ch") or {}).get(ch, 0.0))
    if _tbl:
        _qs = np.asarray(_tbl["q_ch"], float)
        _ts = np.asarray(_tbl["tau"], float)
    # ★**hw.grav_fn 이 있으면 그걸 쓴다** (2026-08-12). actuator_test 가 홈복귀 **전에**
    #   중력을 실측해 표에 상수 보정을 얹고 hw.grav_fn 에 넣어 둔다. 여기서 표를 새로
    #   읽으면 그 보정이 사라진다 — 마찰시험은 이미 고쳤는데 **토크 프로브만 빠져 있었다.**
    #   ⚠이게 왜 치명적인가: 이 프로브는 τ_break = |τ_cmd − bias| 를 양방향 평균해서
    #     마찰을 낸다. bias 오차 Δ 는 s⁺=Δ+f · s⁻=−Δ+f 로 갈리는데, **|Δ| < f 일 때만**
    #     평균에서 상쇄된다. --solo 의 thigh 는 표 오차가 **1.27Nm** 이고 마찰이 0.67 이라
    #     |Δ| > f — 한쪽이 즉시 움직여 평균이 깨진다. 보정 없이는 thigh 를 못 잰다.
    _gfn = getattr(hw, "grav_fn", None)

    def grav_at(q):
        """채널각 q 에서의 중력토크[Nm]. 표 밖은 끝값으로 고정(np.interp 기본)."""
        if _gfn is not None:
            return float(_gfn(ch, q))
        return float(np.interp(q, _qs, _ts)) if _tbl else _fallback
    swing = float((cfg.get("swing_by_ch") or {}).get(ch, cfg.get("tau_max_nm", 1.4)))
    # 절대 상한 — 보정이 걸려 있으면 그만큼 올라간다(안 그러면 램프가 상한에 먼저 걸린다)
    _tbase = float(np.max(np.abs(_ts))) if _tbl else abs(_fallback)
    if _gfn is not None and _tbl:
        _tbase = max(_tbase, abs(grav_at(hw.read(ch)[0])) + abs(
            float(np.interp(hw.read(ch)[0], _qs, _ts)) - grav_at(hw.read(ch)[0])))
    tau_max = _tbase + swing

    bias = grav_at(hw.read(ch)[0])          # 시작 위치의 중력. 시행마다 갱신된다.
    log(f"  [{name}] 순수 토크모드 프로브 — tau G(q)±{swing} Nm @ {ramp} Nm/s, "
        f"방향당 {trials}회"
        + (f"  (중력 바이어스 {bias:+.3f} @ 현재위치 — **시행마다 갱신**)" if _tbl
           else (f"  (중력 바이어스 {bias:+.3f} 고정)" if bias else "")))
    log(f"           (위치모드로 잰 정지마찰 τ_s 와 비교하면 교차검증이 된다)")

    results, raw = [], []
    for direction in (+1.0, -1.0):
        for k in range(trials):
            # ★중력이 큰 축은 **게인을 먼저 놓으면 안 된다** — 바이어스를 올리기 전에
            #   떨어진다(드라이런: hip 207dps 트립, calf 1.58° 표류).
            #   ⇒ 홀드게인으로 잡고 시작해 kp·kd→0 / τ→bias 로 **핸드오프**한다.
            bias = grav_at(hw.read(ch)[0])     # 지금 위치의 중력
            handoff = abs(bias) > 0.05
            kp_h, kd_h = (hw._hold_gain_of(ch) if handoff else (0.0, 0.0))
            hw.arm(ch, kp_h, kd_h)
            time.sleep(0.2)
            q_latch = float(hw._q_cmd[ch])   # arm 이 래치한 명령각
            q_pre = hw.read(ch)[0]
            if bias != 0.0:
                # ★목표각은 **arm 이 래치한 명령각 그대로** 다.
                #   arm 직후 축은 이미 처져서 kp·(q_cmd−q) = bias 로 균형을 이루고 있다.
                #   핸드오프는 그 균형을 (1−r) 비율로 토크에 넘기는 것뿐이라,
                #   q_cmd 를 유지하면 kp·err 가 저절로 (1−r)·bias 가 되어 축이 안 움직인다.
                #   ⚠측정각 q_pre 에 처짐분을 더하면 **이중 계산**이다 —
                #     드라이런에서 hip 이 err 6.02° → 10.46 Nm 로 토크 트립했다.
                nb = max(1, int(0.6 / hw.dt))
                for b in range(nb):
                    r = (b + 1) / nb
                    hw.step_torque(ch, bias * r, tau_max,
                                   kp=kp_h * (1 - r), kd=kd_h * (1 - r), q_des=q_latch)
                    time.sleep(hw.dt)
                # ⚠정착은 stale_ms(150) 보다 짧게. 실기는 잡음(σ_q 0.0044°)이 있어 값이
                #   늘 변하지만, 완전히 정지한 구간이 길면 stale 이 오탐한다.
                for _ in range(int(0.10 / hw.dt)):
                    hw.step_torque(ch, bias, tau_max)
                    time.sleep(hw.dt)
                # ★중력**추종** 홀드로 평형까지 데려간다 (2026-08-12).
                #   매 틱 τ = G(q_now) 를 다시 계산해 실으면, 축은 중력이 아니라
                #   **마찰 불균형만큼만** 움직이다가 스스로 선다. 상수 bias 로 두면
                #   축이 흘러가는 내내 중력과 어긋난 채라 파단값이 오염된다.
                #   HL_hip 시행0 이 그 사례다 — 혼자 시작각이 3.2° 달랐고(−7.81 vs −11.00)
                #   그 흐름이 파단으로 오검출돼 f<0.084 라는 모순된 값이 나왔다.
                t_end = time.monotonic() + 2.5
                q_w, t_w = hw.read(ch)[0], time.monotonic()
                while time.monotonic() < t_end:
                    q_now = hw.read(ch)[0]
                    hw.step_torque(ch, grav_at(q_now), tau_max)
                    if time.monotonic() - t_w > 0.15:
                        if abs(q_now - q_w) < 0.05:
                            break                      # 0.15s 동안 0.05° 미만 → 정착
                        q_w, t_w = q_now, time.monotonic()
                    time.sleep(hw.dt)
                bias = grav_at(hw.read(ch)[0])         # ★최종 bias = **정착 위치**의 중력

                # ★판정 기준은 "**계속 흐르는가**" 지 "움직였나" 가 아니다.
                #   핸드오프 중 축은 그 축의 정상 처짐(bias/kp_h — hip 3.0° · calf 0.58°)
                #   만큼 내려앉는다. 그건 **무해**하다 — 끝나면 τ=bias 가 중력과 정확히
                #   상쇄해 그 자리에 선다. 그걸 오류로 보면 hip·calf 를 영영 못 잰다.
                #   바이어스가 **틀렸을 때만** 정착 후에도 계속 흐른다.
                # ★**멈출 때까지 기다린다**. 한 번만 보고 판정하면 안 된다 (2026-08-12).
                #   hip 은 8축 중 가장 무거워(중력 5.25Nm) 0.10s 로는 정착이 안 끝난다.
                #   실기: 정착 뒤 0.15s 에 +0.234° 로 걸렸는데, 그건 잔여 감쇠였지
                #   영구 표류가 아니었다. 느리게 서는 축은 정상이고, **끝내 안 서는 축**만
                #   바이어스가 틀린 것이다.
                #   ⇒ 창을 반복해 보다가 한 번이라도 문턱 아래면 통과. 전부 넘으면 중단.
                #   ⚠창마다 표류가 **줄고 있는지**도 함께 본다 — 줄지 않으면 기다려도 소용없다.
                creep, hist = None, []
                for _w in range(int(2.0 / 0.15)):
                    q_s = hw.read(ch)[0]
                    for _ in range(int(0.15 / hw.dt)):
                        hw.step_torque(ch, bias, tau_max)
                        time.sleep(hw.dt)
                    creep = hw.read(ch)[0] - q_s
                    hist.append(creep)
                    if abs(creep) <= move_deg * 0.5:
                        break
                    # 3창 연속 안 줄면 영구 표류로 본다(기다려도 소용없다)
                    if len(hist) >= 3 and abs(hist[-1]) > 0.7 * abs(hist[-3]):
                        break
                if len(hist) > 1:
                    log(f"    정착 대기 {len(hist)}창 — 표류 "
                        + " → ".join(f"{v:+.3f}" for v in hist) + " °/0.15s")
                settle = hw.read(ch)[0] - q_pre       # 핸드오프 총 이동(정보용)
                drift = creep
                if abs(creep) > move_deg * 0.5:
                    hw.release_test_axis(ch)
                    gk = float(joint.get("gear_k", 1.0))
                    raise RuntimeError(
                        f"{name}: 중력 바이어스 {bias:+.3f} Nm 인가 후에도 축이 **계속 흐른다** "
                        f"— {len(hist)}창({len(hist)*0.15:.2f}s) 기다려도 표류가 안 멎는다: "
                        f"{' → '.join(f'{v:+.3f}' for v in hist)} °/0.15s (한계 {move_deg*0.5:.2f}). "
                        f"핸드오프 총 이동은 {settle:+.3f}° (정상 처짐 "
                        f"{np.rad2deg(bias/kp_h) if kp_h else 0:+.2f}° 와 비교할 것).\n"
                        f"    **바이어스가 틀렸다** — 이 상태로 재면 파단토크가 오염된다.\n"
                        f"    표류 부호가 바이어스와 **같으면 과다**, 반대면 **부족**이다.\n"
                        f"    토크 규약 후보:  채널 {bias:+.3f}  ·  관절 {bias * gk:+.3f} Nm "
                        f"(gear_k={gk})\n"
                        f"    spec.torque_mode.tau_bias_by_ch[{ch}] 를 고칠 것.")
            q0 = hw.read(ch)[0]
            t0 = time.monotonic()
            tau_at_move, tau_peak, moved = None, 0.0, False
            traj = []
            # ★중력 바이어스를 **매 틱 현재각에서 다시 잡는다** (2026-08-14).
            #   종전엔 시행 시작 때 한 번 잡고 램프 내내 고정이었다. 중력이 완만한 축
            #   (foot |τ_g| 0.09~0.24Nm)에서는 통했지만 **thigh 에서 무너졌다**:
            #     파단 0.172Nm(τ_s 실측 0.95 의 1/5) → 상자 이탈 q=41.97 ∉ [−60, 40]
            #   thigh 는 기울기가 ~0.1Nm/° 라 축이 몇 도만 움직여도 FF 가 그만큼 틀어지고,
            #   그 오차가 곧 순토크라 가속이 붙는다. 파단이 비정상적으로 낮게 잡히는 것도
            #   같은 이유다 — 스윙이 아니라 **FF 오차**가 축을 밀고 있었다.
            #   ⇒ grav_at(현재각) 으로 추종하면 스윙만 남아 램프가 자기안정된다.
            #   ⚠추종에도 상한을 둔다: 보간이 틀리면 추종이 오히려 폭주를 만든다.
            #     시작 바이어스에서 ±_BCAP 을 넘지 않게 자른다.
            _BCAP = 1.5                          # [Nm] 바이어스 추종 허용폭
            _qs = q0
            while True:
                t = time.monotonic() - t0
                _sw = min(ramp * t, swing)
                _b = float(np.clip(grav_at(_qs), bias - _BCAP, bias + _BCAP))
                tau_cmd = _b + direction * _sw
                if ramp * t >= swing and t > swing / ramp + 1.5:
                    break                       # 상한에서 1.5초 더 버텨보고 종료
                s = hw.step_torque(ch, tau_cmd, tau_max)
                _qs = s.q_deg
                traj.append((t, tau_cmd, s.q_deg, s.dq_dps, s.tau))
                tau_peak = max(tau_peak, _sw)
                if (s.q_deg - q0) * direction > move_deg:
                    # ★기록은 **스윙분** 이다 = 순수 마찰분. 바이어스가 매 틱 바뀌므로
                    #   종전처럼 `tau_cmd - bias` 로 빼면 추종분이 섞인다.
                    tau_at_move = _sw; moved = True
                    break                       # ★즉시 중단 → 다음 줄에서 토크 0
                time.sleep(hw.dt)
            # ★파단 뒤에는 **놓지 말고 세운다** (2026-08-14). 종전엔 시험축을 무여자로
            #   풀고 0.4초 기다렸는데, 그 사이 축이 **중력으로 자유낙하**한다.
            #   중력이 작은 축(foot 0.09~0.24Nm)에서는 티가 안 났지만 thigh 는 그 자리
            #   중력이 **−3.06Nm** 이라 0.4초에 80° 를 간다(I≈0.17 ⇒ 17rad/s²).
            #   실측: 파단 검출 즉시 break 했는데도 +1.4° → **+41.97°** 로 40° 를 갔고
            #   상자 [−60, +40] 를 넘어 SafetyAbort. 두 번 연속 **같은 값**이 나왔다.
            #   ⇒ hwio.brake() 로 그 자리에 붙든다. 이건 2026-08-12 에 스윕이 같은 이유로
            #     축을 날려서 만든 함수인데(calf 21.8° 관성주행) 여기만 안 쓰고 있었다.
            #   ⚠tau_ff 로 중력을 계속 태워야 게인이 낮아도 버틴다.
            # ⚠제동 게인은 **따로** 잡는다. kp_h 는 handoff 일 때만 채워지므로
            #   (중력 ≤0.05Nm 축은 0) 그대로 쓰면 저중력 축에서 제동이 안 걸린다.
            _bkp, _bkd = hw._hold_gain_of(ch)
            try:
                hw.brake(ch, _bkp, _bkd, hold_s=0.4,
                         tau_ff_fn=lambda q, _c=ch: grav_at(q))
            except Exception as _e:              # 제동 실패해도 시행 결과는 살린다
                log(f"    ⚠제동 실패({type(_e).__name__}: {_e}) — 시험축을 푼다")
                hw.release_test_axis(ch)
                time.sleep(0.4)
            a = np.array(traj) if traj else np.zeros((1, 5))
            raw.append(a)                       # ★원시 궤적 보존(아래서 npz 로 저장)
            results.append({"dir": direction, "moved": moved, "tau_break": tau_at_move,
                            "tau_peak": tau_peak, "dq_max": float(np.abs(a[:, 3]).max()),
                            "dq_end": float(a[-1, 2] - q0), "n": len(traj)})
            log(f"    {'+' if direction > 0 else '−'}dir #{k}: "
                + (f"파단 {tau_at_move:.3f} Nm 에서 움직임" if moved
                   else f"{tau_peak:.3f} Nm 까지 올려도 미동(이동 {a[-1,2]-q0:+.3f}°)"))

    moved_any = [r for r in results if r["moved"]]
    supported = len(moved_any) >= 2            # 양방향 최소 1회씩은 움직여야 인정
    out = {"supported": supported, "trials": results, "ch": ch, "name": name,
           # ★기어정보를 결과에 실어 보낸다 — 파단토크는 **채널토크**라 리포트에서
           #   관절토크(τ_ch·k)·모터축토크(÷N)로 환산해야 축끼리 비교가 된다.
           "gear_k": float(joint.get("gear_k", 1.0)),
           "gear": float(joint.get("gear", 7.0)),
           "tau_break_mean": (float(np.mean([r["tau_break"] for r in moved_any]))
                              if moved_any else None)}

    # ★원시 궤적을 npz 로 남긴다(2026-08-11). 종전엔 통계만 남기고 버려서, 사후에
    #   "파단 직후 가속도가 얼마였나" 같은 걸 확인할 방법이 없었다.
    try:
        import os
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "results")
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, f"torque_probe_ch{ch:02d}.npz"),
                 **{f"trial{i}": a for i, a in enumerate(raw)},
                 cols=np.array(["t", "tau_cmd", "q_deg", "dq_dps", "tau_meas"]),
                 tau_break=np.array([r["tau_break"] if r["tau_break"] else np.nan
                                     for r in results]),
                 dirs=np.array([r["dir"] for r in results]),
                 gear_k=float(joint.get("gear_k", 1.0)), gear=float(joint.get("gear", 7.0)))
        log(f"  원시 궤적 저장: results/torque_probe_ch{ch:02d}.npz ({len(raw)} 시행)")
    except Exception as e:
        log(f"  ⚠ 원시 궤적 저장 실패({type(e).__name__}: {e}) — 측정 자체는 유효하다")

    log("")
    if supported:
        tb = out["tau_break_mean"]
        log(f"  ✅ 순수 토크모드 **지원됨** — 평균 파단토크 {tb:.3f} Nm")
        log(f"     → WBIC/MPC 토크를 fTorque 로 직접 넣을 수 있다(Kp=Kd=0).")
    elif moved_any:
        log(f"  ⚠ 한쪽 방향만 움직였다 — 중력/편향 의심. 재시험 필요")
    else:
        log(f"  ❌ 순수 토크모드 **미지원** — {tau_max} Nm 까지 올려도 미동.")
        log(f"     fTorque 가 무시되는 것으로 보인다. 배포는 위치+게인 경로로 가야 하며,")
        log(f"     WBIC 토크를 임피던스로 변환하는 계층이 필요하다.")
    return out
