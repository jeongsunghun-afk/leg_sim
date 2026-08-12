#!/usr/bin/env python3
"""pace_cmaes.py — PACE 본절차. MuJoCo 롤아웃 + CMA-ES 로 **궤적 재현매칭**.

목적함수:  Σ_t Σ_i w_i·(q_sim,i(t) − q_real,i(t))²      ← 드라이버 τ 를 **쓰지 않는다**
  ⇒ 우리를 괴롭히던 순환 문제(보고 τ 가 kp·err 로 재구성됨)가 **구조적으로 소멸**한다.
    실기가 위치+게인으로 돌았으니 시뮬도 **같은 제어기·같은 게인**으로 돌린다.

탐색 파라미터 (MJCF dof 속성):
      dof_armature      = ROTOR_I · N²      ← 로터 반사관성
      dof_damping       = JDAMP             ← 점성
      dof_frictionloss  = JFRIC             ← 쿨롱마찰
  기본은 **축종류별 묶음**(hip/thigh/calf/foot) — 8축이 같은 모터라 ROTOR_I 는 하나,
  마찰·감쇠는 기구가 다르니 4개씩. 총 1+4+4 = **9개**.
  `--per-axis` 로 축별(1+8+8=17)도 가능하나 sloppy 해진다.

★왜 창 단위로 재초기화하나
  30초를 개루프로 적분하면 편향이 누적돼 발산한다 — 모델 결함이 아니라 적분의 성질이다.
  `--window` 마다 **실측 상태로 재초기화**해 국소 정합도와 드리프트를 분리한다
  (pace_validate.py 와 같은 처리).

★base 는 **고정**하고 **바닥을 없앤다** — 실기가 매달린(거치) 상태이기 때문이다.
  ⚠freejoint 만 빼면 torso 가 `pos="0 0 0.5257"` 에 고정되는데, 그 높이는 **발이 바닥에
    닿도록** 계산된 값이다. 그대로 두면 시뮬 로봇이 **바닥을 딛고 선다** —
    2026-08-11 실제로 이걸 놓쳐서 "8축 전부 트립" 이라는 **가짜 결론**을 냈다.
    가진 4초 내내 두 발이 100% 접촉 중이었고, 지면이 다리를 붙잡아 추종오차가
    13~32° 로 폭증했다. 실기에는 없는 힘이다.
  ⇒ floor geom 을 제거한다. 확인: 롤아웃 중 `d.ncon == 0` 이어야 한다.

⚠**PACE 는 강체 부분이 맞다는 걸 전제한다.** MJCF 질량·관성이 틀리면 CMA-ES 가 그 오차를
  armature/damping/friction 으로 흡수해 "잘 맞는데 물리적으로 틀린" 값을 낸다.
  → 그래서 축별로 I_link 를 먼저 검증했다(foot 예측 대비 −1.0%, 2026-08-11).

═══ 2026-08-12 검증 기록 — 원문 파라미터(bias·delay)를 넣은 뒤 ═══════════════

원문 PACE(arXiv:2509.06342)는 p=[I_a, d, τ_f, q̃_b, T_d] 를 **한꺼번에** 탐색한다.
우리도 그렇게 바꿨다(9 → 18 모수: 1+4+4 에 축별 bias 8 + 전역 지연 1).
셀프테스트(합성·참값 기지)로 실제로 되찾는지 확인한 결과:

  ★새로 넣은 둘은 **잘 복원된다**
      bias  최대오차 **0.024~0.031°**  (기준 0.3°)      — 전 조건에서 일관
      delay 오차     **0.54 ms**       (기준 2 ms, 경계를 실측으로 묶은 뒤)

  ★★그러나 **모수를 늘린 대가가 있다** — 같은 조건(T=4·120세대) 대조:
      | | 9모수(종전) | 18모수 |
      | 최종 RMS   | **0.0207°**(참값 수준) | 0.0449~0.0615° |
      | ROTOR_I    | −0.0%  | +1.5 ~ −6.7% |
      | JFRIC      | ≤0.6%  | −2.4 ~ −63% |
    120세대에서 RMS 가 **아직 내려가는 중**이다(0.0607→0.0515→0.0449).
    ⇒ **미수렴이지 버그가 아니다.** 원문은 N=4096 병렬환경을 쓰는데 우리는 popsize 10 이다.
      그래서 위 미수렴 가드를 넣었다 — 이 경고가 뜨면 그 값을 결론으로 쓰지 말 것.

  ★지연을 자유변수로 두면 **모델오차를 흡수한다** — 이게 결정적이었다
      경계 2~16ms  → 추정 **12.83 ms**(실측 8.39±0.79 **밖**) · RMS 0.0615 · ROTOR_I −6.7%
      경계 6.8~10ms → 추정 **9.46 ms** · RMS **0.0449**(−27%) · ROTOR_I **+1.5%**(4.5배 개선)
    ⇒ 원문이 T_d 를 자유롭게 두는 건 그 값을 따로 재지 않아서다. **우리는 쟀으니 묶는다.**
      실측이 있는 값을 자유변수로 두면 나머지가 그리로 흘러간다.

  ★남은 약한 방향은 여전히 JDAMP·JFRIC 다 (−66 ~ +64%).
    design_excitation.py 가 설계 단계에서 이미 짚었다: JDAMP 감도가 ROTOR_I 의 1/16~1/87,
    JDAMP.foot↔JFRIC.foot r=+0.93. **궤적으로는 못 고친다**(dual·f0scale·f1scale 전부 실패).
    ⇒ 한쪽을 축별 마찰-속도 곡선으로 못박고 고정하는 것이 정공법이다(NEXT_HW §B).

사용:
    ~/.venv-mujoco/bin/python pace_cmaes.py results/pace_multichirp.npz
    ~/.venv-mujoco/bin/python pace_cmaes.py --selftest        # 하드웨어·실측 불필요
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(os.path.dirname(HERE))
DEG = np.pi / 180.0

# 축종류별 묶음 — 8축이 같은 모터(RO100)라 ROTOR_I 는 공유, 나머지는 기구별
KINDS = ("hip", "thigh", "calf", "foot")


def kind_of(name: str) -> str:
    for k in KINDS:
        if k in name:
            return k
    raise ValueError(name)


def load_fixed_base(mjcf_path: str):
    """freejoint 를 뺀 모델을 만들어 로드. 거치(매단) 상태와 일치시킨다."""
    import mujoco
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    for parent in root.iter():
        for fj in list(parent.findall("freejoint")):
            parent.remove(fj)
        # ★바닥 제거 — 매단 상태를 재현한다(위 주석 참조). 안 지우면 발이 지면에 눌린다.
        for g in list(parent.findall("geom")):
            if g.get("name") == "floor":
                parent.remove(g)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=os.path.dirname(mjcf_path),
                                     delete=False) as f:
        tree.write(f, encoding="unicode")
        tmp = f.name
    try:
        m = mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.unlink(tmp)
    return m


def joint_index(m, names):
    """npz 의 축 순서 → MJCF dof/qpos/actuator 인덱스. **이름으로** 맞춘다(순서 가정 금지)."""
    import mujoco
    out = []
    for nm in names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{nm}_joint")
        if jid < 0:
            raise SystemExit(f"✗ MJCF 에 관절 {nm}_joint 가 없다")
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
        if aid < 0:
            raise SystemExit(f"✗ MJCF 에 actuator {nm} 가 없다")
        out.append((jid, int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]), aid))
    return out


def apply_params(m, idx, gear_n, p, per_axis: bool, names):
    """탐색 벡터 → MJCF dof 속성. p 는 log 공간이 아니라 실공간(양수 제약은 clip)."""
    n = len(idx)
    if per_axis:
        rot = p[0:1].repeat(n)
        dmp, frc = p[1:1 + n], p[1 + n:1 + 2 * n]
    else:
        rot = p[0:1].repeat(n)
        kd_ = {k: p[1 + i] for i, k in enumerate(KINDS)}
        kf_ = {k: p[5 + i] for i, k in enumerate(KINDS)}
        dmp = np.array([kd_[kind_of(x)] for x in names])
        frc = np.array([kf_[kind_of(x)] for x in names])
    for i, (_, _, dof, _) in enumerate(idx):
        m.dof_armature[dof] = max(rot[i], 1e-9) * gear_n[i] ** 2
        m.dof_damping[dof] = max(dmp[i], 0.0)
        m.dof_frictionloss[dof] = max(frc[i], 0.0)
    foot_rotor_to_tendon(m, idx, gear_n, rot, names)


_TENDON_WARNED = []


def foot_rotor_to_tendon(m, idx, gear_n, rot, names):
    """★foot 로터 반사관성을 dof_armature 에서 **tendon 으로 옮긴다**(calf→foot 커플링).

    foot 로터는 관절각이 아니라 raw 각으로 돈다(실기 coef=+1, biped_emb.yaml):
        raw_foot = q_foot + coef·q_calf
    ⇒ 로터 KE = ½·I_rot·N²·(q̇_foot + coef·q̇_calf)² 라 반사관성이 (calf,foot) **비대각**이다.
      `dof_armature` 는 M 의 대각뿐이라 표현할 수 없다 → fixed tendon 의 armature.

    ★**PACE 에서 특히 중요하다.** 축별 측정에서는 이 항이 죽어 있었다(타축 고정 ⇒ q̇_calf=0).
      전축 동시 처프는 calf·foot 이 같이 움직이므로 살아난다. 이게 없으면 CMA-ES 가
      실기에 있고 시뮬에 없는 관성(calf 대각 기준 +46%)을 armature/damping 으로 흡수하려
      들고, 구조가 다르니 깨끗하게 흡수되지 않는다.
    ⚠CMA-ES 가 매 평가마다 ROTOR_I 를 흔들므로 tendon_armature 도 **같이** 갱신해야 한다.
      한 번만 넣어 두면 탐색이 foot 로터를 못 흔든다.
    ⚠**옮기는** 것이지 더하는 게 아니다 — dof_armature[foot] 을 0 으로 안 두면 이중 계상.
    """
    import mujoco
    tid = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, f'{s}_foot_rotor')
           for s in ('HL', 'HR')}
    if any(t < 0 for t in tid.values()):
        if not _TENDON_WARNED:                       # 매 평가 호출이라 1회만 경고한다
            _TENDON_WARNED.append(1)
            print('  ⚠MJCF 에 *_foot_rotor tendon 이 없다 — calf↔foot 커플 반사관성 누락 상태로 적합한다')
        return False
    for i, (_, _, dof, _) in enumerate(idx):
        if kind_of(names[i]) != 'foot':
            continue
        m.tendon_armature[tid[names[i][:2]]] = max(rot[i], 1e-9) * gear_n[i] ** 2
        m.dof_armature[dof] = 0.0                    # ★대각에서 뺀다(tendon 으로 이전)
    return True


def rollout(m, d, idx, q_real, dq_real, q_cmd, kp, kd, dt, win_steps,
            bias=None, delay_s=0.0):
    """창 단위 재초기화 롤아웃. 실기와 **같은 제어법칙**을 시뮬 안에서 돌린다.

    ★bias·delay 는 PACE 원문 파라미터다 (arXiv:2509.06342, p = [I_a, d, τ_f, q̃_b, T_d]).
      · bias q̃_b [deg] — 엔코더 영점 오차. 실기는 `q_enc = q_true + bias` 를 읽으므로
        시뮬이 맞춰야 할 진짜 각은 **q_real − bias** 다. 재초기화·비교 둘 다 그 값을 쓴다.
        ⚠지그 영점 후에도 모델각 기준 0.5~2.3° 가 남아 있었다(2026-08-11) — 무시 못 한다.
      · delay T_d [s] — 명령이 늦게 도달한다. 시각 t 의 유효명령은 q_cmd[t − T_d/dt].
        실측 왕복지연 8.39±0.79 ms 를 초기값·경계의 근거로 쓴다.
    ⚠안 넣으면 이 둘의 오차가 **armature/마찰로 흡수된다** — 잘 맞는데 물리적으로 틀린 값.
    """
    import mujoco
    N, n = q_real.shape
    q_sim = np.empty_like(q_real)
    q_ref = q_real if bias is None else (q_real - bias)      # 진짜 관절각
    sh = int(round(float(delay_s) / dt))                     # 지연[샘플]
    m.opt.timestep = dt
    for s in range(0, N, win_steps):
        e = min(s + win_steps, N)
        # ★창 시작마다 실측 상태로 재초기화 — 개루프 적분 발산과 모델오차를 분리한다
        for i, (_, qa, dofa, _) in enumerate(idx):
            d.qpos[qa] = q_ref[s, i] * DEG
            d.qvel[dofa] = dq_real[s, i] * DEG
        mujoco.mj_forward(m, d)
        for t in range(s, e):
            tc = t - sh if t - sh >= 0 else 0                # 지연된 명령
            for i, (_, qa, dofa, aid) in enumerate(idx):
                q_sim[t, i] = d.qpos[qa] / DEG
                err = (q_cmd[tc, i] - d.qpos[qa] / DEG) * DEG
                d.ctrl[aid] = kp[i] * err - kd[i] * d.qvel[dofa]
            mujoco.mj_step(m, d)
    return q_sim


def cost_of(q_sim, q_real, w=None):
    r = (q_sim - q_real)
    if w is not None:
        r = r * w
    return float(np.sqrt(np.mean(r ** 2)))          # RMS[deg]


def load_data(path):
    z = np.load(path, allow_pickle=True)
    return dict(t=z["t"], q=z["q"], q_cmd=z["q_cmd"], dq=z["dq"],
                kp=z["kp_joint"], kd=z["kd_joint"], gear_n=z["gear_n"],
                names=[str(x) for x in z["names"]], dt=float(z["dt"]))


def param_labels(names, per_axis: bool) -> list:
    """탐색벡터의 **라벨**. init_bounds 와 짝이며 길이가 반드시 같다.

    ★단일 출처로 두는 이유 (2026-08-12): design_excitation 이 라벨을 자기가 만들고 있었는데,
      init_bounds 에 bias·delay 를 추가하자 **x0 18개 vs 라벨 9개**로 어긋났다.
      그러면 (a) plabels[9] 에서 IndexError 로 죽거나 (b) 더 나쁘게, 라벨이 밀려
      **엉뚱한 파라미터 이름으로 보고**된다. 라벨과 벡터는 같은 곳에서 나와야 한다.
    """
    # ★per_axis=True 가 **깨져 있었다** (2026-08-12). 이 함수는 21개를 냈는데
    #   init_bounds 는 26개, split_params 는 nd=1+2n=17 을 가정했다. apply_params 도
    #   `rot = p[0:1]` 로 ROTOR_I 를 **하나만** 쓴다 — 라벨만 kind별 4개로 적혀 있었다.
    #   바로 위 독스트링이 경고한 그 사고를 per_axis 경로에서 다시 낸 것이다
    #   (non-per_axis 만 고치고 per_axis 는 안 봤다).
    #   ⇒ apply_params/split_params 에 맞춘다: ROTOR_I 1개 + JDAMP·JFRIC 축별 n개씩.
    dyn = (["ROTOR_I"] + [f"JDAMP.{x}" for x in names] + [f"JFRIC.{x}" for x in names]
           if per_axis else
           ["ROTOR_I"] + [f"JDAMP.{k}" for k in KINDS] + [f"JFRIC.{k}" for k in KINDS])
    return dyn + [f"bias.{n}" for n in names] + ["delay"]


def init_bounds(spec_path, names, per_axis):
    """초기값·탐색범위 — **축별 측정값**에서 온다. 이게 sloppy 를 줄이는 핵심이다."""
    import yaml
    sp = yaml.safe_load(open(spec_path, encoding="utf-8"))
    rot0 = 7.327e-4          # 2026-08-11 τ_ff 경로 실측(foot 좌우 평균)
    d0 = {"hip": 0.09, "thigh": 0.09, "calf": 0.09, "foot": 0.02}   # JDAMP (hip 외삽/foot 실측)
    # ★JFRIC 은 **실측을 읽어 못박는다** (2026-08-12). 이 파일이 스스로 적어 둔 결론이다:
    #   "JDAMP↔JFRIC r=+0.93 — 궤적으로는 못 고친다. 한쪽을 축별 마찰-속도 곡선으로
    #    못박는 게 정공법". 지연 T_d 를 실측해 묶었더니 ROTOR_I 오차가 4.5배 좋아진 것과
    #   같은 처방이다 — **실측이 있는 값을 자유변수로 두면 나머지가 그리로 흘러간다.**
    #   ⚠spec 값은 **채널토크**다. dof_frictionloss 는 관절토크라 × gear_k 한다.
    #     이 변환은 여기 한 곳에서만 한다 — 두 곳에서 하면 반드시 갈라진다.
    _mc = (sp.get("friction") or {}).get("measured_coulomb_ch")
    if _mc:
        _gk = {int(j["ch"]): float(j.get("gear_k", 1.0)) for j in sp["joints"]}
        _by_kind = {}
        for c, v in _mc.items():
            c = int(c)
            if c < len(names):
                _by_kind.setdefault(kind_of(names[c]), []).append(
                    float(v) * _gk.get(c, 1.0))
        f0 = {k: float(np.mean(v)) for k, v in _by_kind.items()}
        JFRIC_SPAN[0] = 0.20        # 실측이 있으니 ±20% 로 조인다(종전 ×[0.3,3.0])
    else:
        f0 = {"hip": 0.38, "thigh": 0.38, "calf": 0.38, "foot": 0.44}   # 구 추정치
        JFRIC_SPAN[0] = None
    if per_axis:
        x0 = np.concatenate([[rot0], [d0[kind_of(x)] for x in names],
                             [f0[kind_of(x)] for x in names]])
    else:
        x0 = np.concatenate([[rot0], [d0[k] for k in KINDS], [f0[k] for k in KINDS]])
    # ★PACE 원문 파라미터를 마저 넣는다: 관절 bias q̃_b(축별) + 전역 지연 T_d.
    #   원문은 4n+1. 우리는 8축이 **같은 모터**라 ROTOR_I 를 공유하고 마찰·감쇠를
    #   kind별로 묶으므로 (1+4+4) + n(bias) + 1(delay) = 18 이다.
    #   ⚠bias 는 축별이어야 한다 — 엔코더 영점은 물리량이 아니라 축마다 따로다.
    lo = x0 * 0.3
    hi = x0 * 3.0
    # JFRIC 구간만 실측 기반으로 조인다. 벡터 순서는 [ROTOR_I, JDAMP…, JFRIC…] 이다.
    if JFRIC_SPAN[0] is not None:
        nj = len(names) if per_axis else len(KINDS)
        sl = slice(1 + nj, 1 + 2 * nj)
        lo[sl] = x0[sl] * (1.0 - JFRIC_SPAN[0])
        hi[sl] = x0[sl] * (1.0 + JFRIC_SPAN[0])
    nb = len(names)
    x0 = np.concatenate([x0, np.zeros(nb), [DELAY0]])
    lo = np.concatenate([lo, np.full(nb, -BIAS_MAX), [DELAY_LO]])
    hi = np.concatenate([hi, np.full(nb, +BIAS_MAX), [DELAY_HI]])
    return x0, lo, hi


# bias·delay 경계 — 실측 근거로 잡는다(임의값이 아니다)
JFRIC_SPAN = [None]   # init_bounds 가 채운다(실측 있으면 ±비율)
BIAS_MAX = 3.0        # [deg] 지그 영점 후 잔차가 모델각 0.5~2.3° 였다(2026-08-11)
# ★지연은 **직접 실측**했다: 8.39 ± 0.79 ms (act_measure_latency.py).
#   원문(PACE)은 T_d 를 자유롭게 탐색하는데, 그건 그 값을 따로 재지 않았기 때문이다.
#   우리는 있으니 **±2σ 로 묶는다**(6.8~10.0 ms).
#   ⚠근거: 넓게(2~16ms) 뒀더니 추정이 **12.83 ms** 로 실측 범위 밖까지 나갔다
#     (셀프테스트 T=4·120세대). 지연이 모델오차를 **흡수**한 것이고, 그 대가로
#     JFRIC 이 −13~−63% 로 무너졌다. 실측이 있는 값을 자유변수로 두면 이렇게 된다.
DELAY0, DELAY_LO, DELAY_HI = 0.00839, 0.0068, 0.0100


def to_z(x, lo, hi):
    """★탐색을 z∈[0,1] 로 정규화한다.

    CMA-ES 는 sigma **하나**를 전 차원에 쓴다. 그런데 우리 파라미터는
    ROTOR_I 7e−4 · JFRIC 0.44 · bias ±3.0 · delay 0.008 로 스케일이 4자리 넘게 벌어진다.
    원공간에서 sigma 0.3 을 쓰면 ROTOR_I 는 400배 과도, delay 는 37배 과도로 흔들린다.
    ⇒ 경계로 정규화해 전 차원을 같은 크기로 만든다. (bias 는 초기값이 0 이라
      종전의 '초기값 비율' 방식 자체가 성립하지 않기도 한다.)
    """
    return (np.asarray(x, float) - lo) / np.where(hi - lo > 0, hi - lo, 1.0)


def from_z(z, lo, hi):
    return lo + np.clip(np.asarray(z, float), 0.0, 1.0) * (hi - lo)


def split_params(p, n, per_axis):
    """탐색벡터 → (동역학, bias[deg], delay[s]). 순서는 init_bounds 와 짝이다."""
    nd = 1 + 2 * n if per_axis else 9
    p = np.asarray(p, float)
    return p[:nd], p[nd:nd + n], float(p[nd + n])


def report(names, x, per_axis, log=print):
    log(f"  {'파라미터':<16}{'값':>12}")
    log(f"  {'ROTOR_I':<16}{x[0]:>12.4e}")
    if per_axis:
        n = len(names)
        for i, nm in enumerate(names):
            log(f"  {'JDAMP.'+nm:<16}{x[1+i]:>12.4f}")
        for i, nm in enumerate(names):
            log(f"  {'JFRIC.'+nm:<16}{x[1+n+i]:>12.4f}")
    else:
        for i, k in enumerate(KINDS):
            log(f"  {'JDAMP.'+k:<16}{x[1+i]:>12.4f}")
        for i, k in enumerate(KINDS):
            log(f"  {'JFRIC.'+k:<16}{x[5+i]:>12.4f}")
    n = len(names)
    nd = 1 + 2 * n if per_axis else 9
    if len(x) >= nd + n + 1:                       # bias·delay 가 있으면 같이 찍는다
        b = np.asarray(x[nd:nd + n], float)
        log(f"  {'bias[deg]':<16}" + " ".join(f"{v:+.3f}" for v in b))
        log(f"  {'delay[ms]':<16}{float(x[nd+n])*1e3:>12.2f}   (실측 8.39±0.79)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", help="collect_multichirp.py 산출물")
    ap.add_argument("--mjcf", default=os.path.join(BIPED, "biped_flatfoot.mjcf"))
    ap.add_argument("--spec", default=os.path.join(HERE, "spec.yaml"))
    ap.add_argument("--window", type=float, default=0.5, help="재초기화 창[s]")
    ap.add_argument("--per-axis", action="store_true", help="축별 17모수(기본은 묶음 9모수)")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="뒤쪽 이 비율은 **적합에 안 쓰고** 검증에만 쓴다")
    ap.add_argument("--validate", default=None, metavar="npz",
                    help="★unseen PD gains 검증 (PACE arXiv:2509.06342). 다른 게인으로 수집한 "
                         "npz 를 주면, 적합된 θ 를 **그 데이터**에 걸어 RMS 를 낸다. "
                         "게인이 바뀌어도 같은 θ 가 맞으면 순환·과적합이 아니다")
    ap.add_argument("--eval-only", action="store_true", help="초기값만 평가(CMA-ES 생략)")
    ap.add_argument("--st-T", type=float, default=4.0, help="셀프테스트 길이[s]")
    ap.add_argument("--st-dt", type=float, default=0.002,
                    help="셀프테스트 스텝[s]. ★Pi 는 느리다 — 1ms×12s 는 세대당 수십초다")
    ap.add_argument("--selftest", action="store_true",
                    help="알려진 파라미터로 합성데이터를 만들어 되찾는지 확인")
    a = ap.parse_args()

    try:
        import mujoco  # noqa: F401
    except ImportError:
        raise SystemExit("✗ mujoco 가 없다. ~/.venv-mujoco/bin/python 으로 실행할 것.")

    import mujoco
    m = load_fixed_base(a.mjcf)
    print(f"■ 모델 {os.path.basename(a.mjcf)} (base 고정) — nq={m.nq} nv={m.nv} nu={m.nu}")

    if a.selftest:
        return selftest(m, a)

    if not a.npz:
        raise SystemExit("npz 를 주거나 --selftest 를 쓸 것")
    D = load_data(a.npz)
    idx = joint_index(m, D["names"])
    d = mujoco.MjData(m)
    N = len(D["t"])
    win = max(1, int(round(a.window / D["dt"])))
    ncut = int(N * (1 - a.holdout))
    print(f"■ 데이터 {os.path.basename(a.npz)} — {N} 표본 · dt {D['dt']*1000:.1f}ms · "
          f"창 {a.window}s({win} 스텝)")
    print(f"  적합 구간 0~{ncut} · **hold-out {ncut}~{N}** ({a.holdout:.0%})")

    x0, lo, hi = init_bounds(a.spec, D["names"], a.per_axis)
    print(f"■ 초기값 — {len(x0)} 모수 "
          f"(동역학 {len(x0)-len(D['names'])-1} + bias {len(D['names'])} + delay 1). "
          f"동역학 ×0.3~3.0 · bias ±{BIAS_MAX}° · delay {DELAY_LO*1e3:.0f}~{DELAY_HI*1e3:.0f}ms")
    report(D["names"], x0, a.per_axis)

    def evaluate(p, s, e):
        dyn, bias, dly = split_params(p, len(D["names"]), a.per_axis)
        apply_params(m, idx, D["gear_n"], dyn, a.per_axis, D["names"])
        qs = rollout(m, d, idx, D["q"][s:e], D["dq"][s:e], D["q_cmd"][s:e],
                     D["kp"], D["kd"], D["dt"], win, bias=bias, delay_s=dly)
        return cost_of(qs, D["q"][s:e] - bias), qs      # ★진짜 각(q_enc − bias)과 비교

    c0, _ = evaluate(x0, 0, ncut)
    h0, _ = evaluate(x0, ncut, N)
    print(f"\n■ 초기값 RMS — 적합 {c0:.4f}° · hold-out {h0:.4f}°")
    if a.eval_only:
        return 0

    try:
        import cma
    except ImportError:
        raise SystemExit("✗ cma 가 없다: ~/.venv-mujoco/bin/pip install cma")
    # ★z∈[0,1] 정규화 공간에서 탐색한다(to_z 주석 참조). 원공간은 스케일이 4자리 벌어진다.
    es = cma.CMAEvolutionStrategy(list(to_z(x0, lo, hi)), 0.25, {
        "popsize": a.popsize, "maxiter": a.iters,
        "bounds": [0.0, 1.0], "verbose": -9})       # ★z 공간이므로 경계도 [0,1] 이다
    print(f"\n■ CMA-ES — popsize {a.popsize} · 최대 {a.iters} 세대 (z∈[0,1] 정규화 탐색)")
    best, bestc = np.array(x0), c0
    hist = []
    it = 0
    while not es.stop():
        Z = es.ask()
        X = [from_z(z, lo, hi) for z in Z]
        F = [evaluate(x, 0, ncut)[0] for x in X]
        es.tell(Z, F)
        it += 1
        if min(F) < bestc:
            bestc = float(min(F)); best = np.array(X[int(np.argmin(F))])
        hist.append(bestc)
        if it % 10 == 0 or it == 1:
            print(f"    세대 {it:>3}  최량 RMS {bestc:.4f}°")
    # ★미수렴 가드 — 마지막 20% 구간에서도 계속 좋아지고 있으면 **아직 안 끝난 것**이다.
    #   모수를 9→18 로 늘리면서 기본 120세대로는 부족해졌다(아래 docstring 검증 기록 참조).
    if len(hist) >= 10:
        tail_gain = 1.0 - hist[-1] / hist[max(0, int(len(hist) * 0.8))]
        if tail_gain > 0.05:
            print(f"  ★**미수렴** — 마지막 20% 세대에서 RMS 가 {tail_gain*100:.0f}% 더 내려갔다."
                  f" --iters 를 올릴 것(현재 {a.iters}). 이 값을 결론으로 쓰지 말 것")
    hb, qs = evaluate(best, ncut, N)
    print(f"\n■ 결과 — 적합 RMS {bestc:.4f}° · **hold-out RMS {hb:.4f}°** "
          f"(초기 {c0:.4f}/{h0:.4f})")
    print(f"  개선 적합 {(1-bestc/c0)*100:+.1f}% · hold-out {(1-hb/h0)*100:+.1f}%")
    if hb > bestc * 1.5:
        print("  ⚠hold-out 이 적합보다 크게 나쁘다 — **과적합**이다. 모수를 줄이거나"
              " 데이터를 늘릴 것")
    report(D["names"], best, a.per_axis)

    # ── ★unseen PD gains 검증 (원문의 주 검증) ──────────────────────────────
    #   hold-out 궤적보다 강하다: 게인이 바뀌면 kp·err 순환이나 과적합이 바로 드러난다.
    #   ⚠게인은 **그 데이터셋의 것**을 쓴다(npz 에 kp_joint/kd_joint 가 들어 있다).
    if a.validate:
        V = load_data(a.validate)
        if list(V["names"]) != list(D["names"]):
            raise SystemExit("✗ 검증셋의 축 순서가 다르다")
        vi = joint_index(m, V["names"])
        vw = max(1, int(round(a.window / V["dt"])))
        print(f"\n■ unseen PD gains 검증 — {os.path.basename(a.validate)}")
        print(f"  적합셋 kp {np.round(D['kp'], 1)}")
        print(f"  검증셋 kp {np.round(V['kp'], 1)}   ← 다른 게인이어야 의미가 있다")
        if np.allclose(D["kp"], V["kp"]):
            print("  ⚠게인이 같다 — 이건 unseen gains 검증이 아니다(궤적 hold-out 일 뿐)")
        rows = []
        for lab, px in (("초기값", x0), ("적합 θ", best)):
            dyn, bias, dly = split_params(px, len(V["names"]), a.per_axis)
            apply_params(m, idx, V["gear_n"], dyn, a.per_axis, V["names"])
            qs = rollout(m, d, vi, V["q"], V["dq"], V["q_cmd"], V["kp"], V["kd"],
                         V["dt"], vw, bias=bias, delay_s=dly)
            rows.append((lab, cost_of(qs, V["q"] - bias)))
        for lab, c in rows:
            print(f"    {lab:<8}RMS {c:.4f}°")
        imp = (1 - rows[1][1] / rows[0][1]) * 100
        print(f"  개선 {imp:+.1f}%")
        if rows[1][1] > bestc * 2.0:
            print("  ★검증 RMS 가 적합의 2배를 넘는다 — **게인을 바꾸면 안 맞는다**."
                  " θ 가 게인에 얹혀 있다는 뜻이라 그대로 배포하면 안 된다")
        elif imp < 0:
            print("  ★적합 θ 가 초기값보다 **나쁘다** — 과적합이다")
        else:
            print("  ✅게인을 바꿔도 개선이 유지된다")

    out = os.path.splitext(a.npz)[0] + "_cmaes.npz"
    np.savez(out, x=best, x0=x0, rms_fit=bestc, rms_holdout=hb,
             per_axis=a.per_axis, names=np.array(D["names"]))
    print(f"\n  ✓ 저장: {out}")
    return 0


def selftest(m, a) -> int:
    """★알려진 파라미터로 합성데이터를 만들어 CMA-ES 가 되찾는지 확인.

    실측을 믿기 전에 **추정기 자체**를 분리 검증한다. 이번 세션에서 여러 번 배운 것:
    합성검증은 반드시 **실제 조건**(같은 궤적·같은 창·같은 노이즈)으로 해야 한다.
    """
    import mujoco
    import yaml
    names = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot",
             "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
    idx = joint_index(m, names)
    d = mujoco.MjData(m)
    sp = yaml.safe_load(open(a.spec, encoding="utf-8"))
    mc = sp["pace_multi"]
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(HERE), "config",
                                           "biped_emb.yaml"), encoding="utf-8"))
    J = {j["name"]: j for j in cfg["joints"]}
    kp = np.array([J[n]["kp"] * J[n]["gear_k"] ** 2 for n in names])
    kd = np.array([J[n]["kd"] * J[n]["gear_k"] ** 2 for n in names])
    gear_n = np.array([float([x for x in sp["joints"]
                              if x["name"] == n][0]["gear"]) for n in names])

    T, dt = float(a.st_T), float(a.st_dt)
    tt = np.arange(0, T, dt)
    amps = np.array(mc["amp_deg"], float)
    f0 = np.array(mc["f_start_hz"], float)
    k_ = (np.array(mc["f_end_hz"], float) - f0) / T
    phi = (np.arange(8) * 0.6180339887) % 1.0 * 2 * np.pi
    q_cmd = np.array([amps * np.sin(2 * np.pi * (f0 * t + 0.5 * k_ * t * t) + phi)
                      for t in tt])

    # ★bias·delay 도 **0 이 아닌 참값**을 심는다. 0 으로 두면 "안 움직였다" 와
    #   "되찾았다" 가 구분되지 않아 검증이 되지 않는다.
    bias_true = np.array([0.8, -1.2, 0.5, -0.3, -0.6, 0.9, -0.4, 0.7])
    delay_true = 0.010
    x_true = np.concatenate([[7.327e-4], [0.11, 0.07, 0.13, 0.025],
                             [0.42, 0.31, 0.50, 0.44], bias_true, [delay_true]])
    apply_params(m, idx, gear_n, split_params(x_true, len(names), False)[0], False, names)
    win = max(1, int(round(a.window / dt)))
    # 참 궤적: 실측 자리에 시뮬을 넣고 창 재초기화 없이 한 번에 굴린다
    q0 = np.zeros((len(tt), 8)); dq0 = np.zeros_like(q0)
    q_true = rollout(m, d, idx, q0, dq0, q_cmd, kp, kd, dt, len(tt), delay_s=delay_true)
    dq_true = np.vstack([np.zeros(8), np.diff(q_true, axis=0) / dt])
    rng = np.random.default_rng(0)
    # ★엔코더는 q_true + bias 를 읽는다 — 추정기는 이 bias 를 되찾아야 한다
    q_meas = q_true + bias_true + rng.normal(0, 0.02, q_true.shape)
    print(f"■ 셀프테스트 — 합성 {T:.0f}s · dt {dt*1000:.0f}ms · 측정잡음 0.02°")

    def ev(p):
        dyn, bias, dly = split_params(p, len(names), False)
        apply_params(m, idx, gear_n, dyn, False, names)
        qs = rollout(m, d, idx, q_meas, dq_true, q_cmd, kp, kd, dt, win,
                     bias=bias, delay_s=dly)
        return cost_of(qs, q_meas - bias)

    x0, lo, hi = init_bounds(a.spec, names, False)
    print(f"  초기 RMS {ev(x0):.4f}°  ·  참값 RMS {ev(x_true):.4f}°")
    try:
        import cma
    except ImportError:
        raise SystemExit("✗ cma 가 없다: ~/.venv-mujoco/bin/pip install cma")
    # ★z∈[0,1] 정규화 공간에서 탐색한다(to_z 주석 참조). 원공간은 스케일이 4자리 벌어진다.
    es = cma.CMAEvolutionStrategy(list(to_z(x0, lo, hi)), 0.25, {
        "popsize": a.popsize, "maxiter": a.iters, "bounds": [0.0, 1.0], "verbose": -9})
    best, bestc = np.array(x0), ev(x0)
    it = 0
    while not es.stop():
        Z = es.ask(); X = [from_z(z, lo, hi) for z in Z]
        F = [ev(x) for x in X]; es.tell(Z, F); it += 1
        if min(F) < bestc:
            bestc = float(min(F)); best = np.array(X[int(np.argmin(F))])
        if it % 10 == 0:
            print(f"    세대 {it:>3}  최량 RMS {bestc:.4f}°")
    print(f"\n  {'파라미터':<16}{'참값':>12}{'추정':>12}{'오차':>10}")
    lab = ["ROTOR_I"] + [f"JDAMP.{k}" for k in KINDS] + [f"JFRIC.{k}" for k in KINDS]
    ok = True
    for i, l in enumerate(lab):
        e = (best[i] / x_true[i] - 1) * 100
        good = abs(e) < 30
        ok &= good
        print(f"  {l:<16}{x_true[i]:>12.4g}{best[i]:>12.4g}{e:>9.1f}%"
              + ("" if good else "  ★"))
    # ★bias·delay 는 **절대오차**로 본다 — 참값이 0 근처일 수 있어 비율이 무의미하다
    nb = len(names)
    be = np.abs(best[9:9 + nb] - x_true[9:9 + nb])
    de = abs(best[9 + nb] - x_true[9 + nb]) * 1e3
    bg, dg = be.max() < 0.3, de < 2.0                 # 0.3° · 2ms 이내면 통과
    ok &= bool(bg and dg)
    print(f"  {'bias[deg] 최대오차':<16}{'':>12}{be.max():>12.3f}{'':>9}"
          + ("" if bg else "  ★(0.3° 초과)"))
    print(f"  {'delay[ms] 오차':<16}{x_true[9+nb]*1e3:>12.2f}{best[9+nb]*1e3:>12.2f}"
          f"{de:>8.2f}ms" + ("" if dg else "  ★(2ms 초과)"))
    print(f"\n  RMS {bestc:.4f}° · 셀프테스트 {'통과' if ok else '실패(30% 초과 항목)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
