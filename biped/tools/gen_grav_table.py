#!/usr/bin/env python3
"""gen_grav_table.py — 축별 **중력토크 곡선**을 뽑아 spec.yaml 에 넣는다.

★왜 필요한가 (2026-08-12)
  순수토크 프로브의 램프 시작점(bias)을 0° 기준 **상수**로 쓰면 틀린다.
  게인을 놓으면 축이 자기 평형점으로 흘러가는데, 거기서는 중력이 이미 다르다:
      HL_hip  0° 에서 5.25 Nm → 축이 실제로 멈춘 −11° 에서는 **4.09 Nm**
  그 1.16 Nm 차이 때문에 τ_break 이 마찰이 아닌 값이 됐다
  (리포트 0.454/0.584 Nm 은 마찰이 아니라 '어긋난 bias 에서 파단까지의 거리' 였다).
  사후에 위치별 중력으로 다시 맞춰보니 hip 마찰은 **1.40 / 1.54 Nm** 이었다 —
  다리 미장착 시절 0.65 의 2배 이상(다리 ~3kg 이 베어링에 매달린 결과).

⇒ 각 축의 **채널각 → 채널토크** 곡선을 미리 뽑아 둔다. 프로브는 그때그때 위치를 읽어
  보간해 쓴다. Pi 의 시스템 python 에 mujoco 가 없으므로 표를 미리 만들어 두는 방식이다.

★표는 **채널각**으로 색인한다 — 프로브가 재는 값이 그것이라 런타임 변환이 없다
  (변환식을 한 군데 더 복사하면 갈라진다. 오늘만 그 부류 버그가 여섯 번 나왔다).

⚠다른 관절은 hold_pose.neutral_deg 에 고정한 채 해당 축만 쓸었다. 자세를 바꾸면
  표도 다시 뽑아야 한다. spec 에 pose 를 같이 적어 둔다.

사용:  ~/.venv-mujoco/bin/python3 tools/gen_grav_table.py [--apply]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(HERE)
sys.path[:0] = [os.path.join(BIPED, "emb", "pace"), os.path.join(BIPED, "emb", "interface")]

N_PT = 21          # 축당 표본수. 곡선이 완만해서(hip 12° 에 1.3Nm) 이 정도면 선형보간 오차 <1%.
MARGIN = 0.90      # 관절한계의 이 비율까지만 — 한계구속 반력이 섞이면 값이 50Nm 로 튄다.


def build(mjcf: str, cfg: dict) -> dict:
    import mujoco
    from pace_cmaes import load_fixed_base

    m = load_fixed_base(mjcf)
    d = mujoco.MjData(m)
    js = sorted(cfg["joints"], key=lambda x: int(x["channel"]))
    pose = list(cfg["hold_pose"]["neutral_deg"])

    # ★커플링 축의 **원천축**을 쓸 때는 종동 관절이 따라 움직인다 (2026-08-12).
    #   couple_from 은 "종동 채널각이 원천 관절각에도 의존한다" 는 뜻이다:
    #       θ_foot = (q_foot + coef·q_calf)·s·k + o
    #   프로브에서 종동축(foot)은 **홀드축이라 채널각이 잠긴다**. 그러면 원천축(calf)이
    #   δ 돌 때 q_foot = −coef·δ 로 **발목이 되돌아 돈다**. 표를 '다른 관절 고정' 으로
    #   뽑으면 그 되돌아 도는 몫이 빠져 중력이 틀린다 — calf 에서 최대 0.11 Nm(15%대).
    #   ⚠반대 방향은 문제없다. foot 을 구동할 때 calf 채널은 잠겨 있고 **calf 관절은
    #     실제로 안 움직인다** → 고정 가정이 맞다. 그래서 원천축일 때만 보정한다.
    driven = {}                      # {원천 ch: [(종동 ch, coef), ...]}
    for j in js:
        src = j.get("couple_from")
        if src:
            sc = next(int(x["channel"]) for x in js if x["name"] == src)
            driven.setdefault(sc, []).append((int(j["channel"]), float(j["couple_coef"])))

    def tau_at(ch: int, qj: float) -> float:
        q = list(pose)
        q[ch] = qj
        for dch, coef in driven.get(ch, []):
            q[dch] = pose[dch] - coef * (qj - pose[ch])
        d.qpos[:] = 0
        for i, v in enumerate(q):
            d.qpos[i] = np.deg2rad(v)
        d.qvel[:] = 0
        d.qacc[:] = 0
        mujoco.mj_inverse(m, d)
        return float(d.qfrc_inverse[ch])

    out = {}
    for j in js:
        ch = int(j["channel"])
        sg, k = float(j["sign"]), float(j.get("gear_k", 1.0))
        off = float(j["offset_deg"])
        lo, hi = float(j["min_deg"]) * MARGIN, float(j["max_deg"]) * MARGIN
        qj = np.linspace(lo, hi, N_PT)
        # 채널각·채널토크. 각도변환 q_ch = q_joint·sign·k + offset 의 전치가 토크변환이다.
        q_ch = qj * sg * k + off
        t_ch = np.array([tau_at(ch, float(v)) for v in qj]) * sg / k
        o = np.argsort(q_ch)                      # np.interp 는 오름차순을 요구한다
        out[ch] = {"q_ch": [round(float(v), 3) for v in q_ch[o]],
                   "tau": [round(float(v), 4) for v in t_ch[o]]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default=os.path.join(BIPED, "biped_flatfoot.mjcf"))
    ap.add_argument("--apply", action="store_true", help="spec.yaml 에 기록")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(BIPED, "emb", "config", "biped_emb.yaml")))
    tbl = build(a.mjcf, cfg)
    names = {int(j["channel"]): j["name"] for j in cfg["joints"]}
    print(f"■ 중력토크 표 — {os.path.basename(a.mjcf)} · 자세 neutral · {N_PT}점/축")
    print(f"{'ch':<3}{'축':<10}{'채널각 범위':>22}{'τ_ch 범위':>20}{'변화폭':>9}")
    for ch, v in sorted(tbl.items()):
        q, t = v["q_ch"], v["tau"]
        print(f"{ch:<3}{names[ch]:<10}[{q[0]:+8.2f}, {q[-1]:+8.2f}]°"
              f"  [{min(t):+7.3f}, {max(t):+7.3f}] Nm{max(t)-min(t):>9.3f}")
    if not a.apply:
        print("\n  (--apply 를 주면 spec.yaml 에 기록한다)")
        return 0

    p = os.path.join(BIPED, "emb", "pace", "spec.yaml")
    s = open(p).read()
    blob = yaml.safe_dump({"tau_grav_table": tbl}, default_flow_style=None,
                          allow_unicode=True, sort_keys=True, width=100)
    blob = "".join("  " + ln + "\n" for ln in blob.rstrip().split("\n"))
    head = ("  # ★중력토크 표 — **tools/gen_grav_table.py 가 생성**. 손으로 고치지 말 것.\n"
            "  #   채널각[deg] → 채널토크[Nm]. 프로브가 그때그때 위치를 읽어 보간해 bias 로 쓴다.\n"
            "  #   자세는 hold_pose.neutral_deg 고정, 해당 축만 쓸었다. 자세를 바꾸면 재생성할 것.\n"
            "  #   유래·필요성은 tools/gen_grav_table.py 의 독스트링 참조.\n")
    # ★기존 블록 제거는 **줄 단위**로 한다 (2026-08-12).
    #   종전엔 정규식 `# ★중력토크 표.*?(?=\n  [a-z_]+:)` 를 썼는데, 그 lookahead 가
    #   **블록 자신의 키 `  tau_grav_table:`** 에 먼저 걸려 주석만 지우고 표는 남겼다.
    #   그래서 새 표가 **추가**되고 키가 2개가 됐다 — YAML 은 뒤엣것(옛 표)을 쓴다.
    #   ⇒ 프로브가 offset 변경 전 표로 bias 를 계산했다. ch3 에서 MuJoCo 직접값
    #     −0.034 대신 표 상한 **+0.142** 를 썼다(부호까지 반대).
    lines, out, skip = s.split("\n"), [], False
    for ln in lines:
        if ln.startswith("  # ★중력토크 표") or ln.startswith("  tau_grav_table:"):
            skip = True
            continue
        if skip:
            # 블록에 속한 줄 = 더 깊게 들여썼거나 주석. 그 외를 만나면 블록 끝.
            if ln.strip() == "" or ln.startswith("   ") or ln.startswith("  #"):
                continue
            skip = False
        out.append(ln)
    s = "\n".join(out)

    import re
    m = re.search(r"^torque_mode:\s*\n", s, re.M)
    assert m, "spec.yaml 에 torque_mode: 가 없다"
    s = s[:m.end()] + head + blob + s[m.end():]
    open(p, "w").write(s)
    # ★검증 — 키가 **정확히 1개**인지까지 본다. 중복이면 YAML 이 조용히 뒤엣것을 쓴다.
    n_key = sum(1 for ln in open(p) if ln.startswith("  tau_grav_table:"))
    assert n_key == 1, f"tau_grav_table 키가 {n_key}개 — 중복이면 옛 표가 쓰인다"
    got = yaml.safe_load(open(p))["torque_mode"]["tau_grav_table"]
    assert set(got) == set(tbl), "기록 후 재파싱 불일치"
    for ch in tbl:
        assert got[ch]["q_ch"] == tbl[ch]["q_ch"], f"ch{ch} q_ch 불일치 — 옛 표가 남았다"
    print(f"\n  ✓ {p} 기록 — {len(got)}축")
    return 0


if __name__ == "__main__":
    sys.exit(main())
