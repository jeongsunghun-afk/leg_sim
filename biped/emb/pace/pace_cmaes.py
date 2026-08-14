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

═══ 2026-08-14 **해결** — foot PD 는 raw각으로 잰다 ═══════════════════════

궤적 따로 뺀 구간 은 네 번 다 관절각을 골랐는데, **unseen PD gains 검증이 뒤집었다.**

  ★결정적 실험 — 게인만 2배로 바꿔 수집한 둘째 데이터셋(`--gains validate`)
      적합셋 kp [100  50  56.2 31.7 ...]  ·  검증셋 kp [100 100 112.5 63.4 ...]
                          A(raw)     B(joint)
      적합                0.4481     0.4313    ← B 가 4% 낫다
      따로 뺀 구간            0.4727     0.4567    ← B 가 3% 낫다
      **검증(게인 2배)**   **0.2806**  0.3281    ← **A 가 14.5% 낫다**
      검증 개선           +28.2%     +20.1%
    같은 게인에서는 B 가 근소하게 앞서지만 **게인을 바꾸면 A 가 확실히 이긴다.**
    틀린 구조는 훈련 게인에 얹혀 맞추고 게인이 바뀌면 무너진다 — 원문 PACE 가
    이 검증을 주 검증으로 쓰는 이유다.

  ★구조 증거 네 갈래도 전부 raw 였다 — 이제 숫자와 일치한다
      ① 관성 비 무릎 자유/고정 = 1.06 (예측 tendon 0.99 · 커플링없음 0.64)
      ② 엔코더가 커플링을 안 본다(무릎 143°에 기울기 <2e-4, 좌우 모두) = 모터축 = raw
      ③ MJCF tendon 길이 = q_calf+q_foot = 엔코더 좌표와 같다
      ④ `joint_map.q_joint_to_ch` 가 명령에 coef 를 되먹인다 → 목표도 raw

  ★왜 궤적 따로 뺀 구간 이 속였나
    raw 법칙은 calf 추종오차를 foot 토크에 실으므로 **calf 모델오차가 foot 으로
    증폭**된다. 관절 법칙은 둘을 떼어 그 오염을 막는다 — 그래서 같은 게인에서는
    더 잘 맞는다. 하지만 그건 물리가 아니라 **오차 차단**이다.
    ⇒ **적합이 좋은 쪽이 물리적으로 맞는 쪽이라는 보장이 없다.** 이 사례가 그 증거다.
    ⚠교훈: 궤적 따로 뺀 구간 만으로 **모델 구조**를 고르지 말 것. 구조는 구조 증거로
      고르고, 숫자로 가리려면 **조건을 바꾼 데이터**(게인·자세·궤적)가 필요하다.

  ⚠아직 남은 것: JFRIC.calf·JFRIC.foot 이 양쪽 모두 탐색범위 하한에 박힌다. calf 가
    전혀 안 정해지는 것도 그대로다(JDAMP.calf 가 법칙에 따라 0.011↔0.477 로 뛴다).
    실측 JFRIC 이 처프 속도대에서 과대평가라는 뜻일 수 있다(Stribeck).

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


def set_coupling_coef(m, c: float) -> None:
    """calf→foot 커플링 계수를 tendon wrap 에 쓴다. **힘·관성 쪽**을 바꾼다.

    tendon 길이 = c·q_calf + 1·q_foot = q_raw 가 되도록 한다.
    로터 KE = ½·I·N²·(q̇_foot + c·q̇_calf)² 이므로 tendon_armature 는 그대로다.
    ⚠**기구학 쪽(데이터)도 같이 바꿔야 한다** — `retarget_coupling` 참조.
    """
    import mujoco
    for s_ in ("HL", "HR"):
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, f"{s_}_foot_rotor")
        if tid < 0:
            continue
        for k in range(m.tendon_adr[tid], m.tendon_adr[tid] + m.tendon_num[tid]):
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.wrap_objid[k])
            if jn and "calf" in jn:
                m.wrap_prm[k] = float(c)


def retarget_coupling(arr, names, c: float):
    """저장된 모델각(coef=1 가정)을 **coef=c 기준**으로 다시 쓴다.

    ★왜 필요한가 (2026-08-14)
      npz 의 q·q_cmd·dq 는 `jm.ch_to_q_joint` 가 만든 값이고 그 함수는 **coef=1 을
      하드코딩**한다. 즉 데이터 자체에 가정이 박혀 있다. c 를 탐색하려면 그걸 풀어야 한다.

      엔코더가 실제로 재는 것은 raw각이고 그건 가정과 무관하게 정확하다:
          q_raw = q_foot_stored + 1·q_calf          (저장값에서 정확히 복원된다)
      참 모델각은 c 로 다시 푼다:
          q_foot_true = q_raw − c·q_calf = q_foot_stored + (1−c)·q_calf
      calf 는 커플링이 없으므로 그대로다.
    ⚠속도도 같은 선형관계다(offset 만 빠진다).
    ⚠c=1 이면 항등이다 — 그래서 기존 결과가 그대로 재현되는지로 검증할 수 있다.
    """
    if c == 1.0:
        return arr
    out = np.array(arr, float, copy=True)
    for i, n in enumerate(names):
        if kind_of(n) != "foot":
            continue
        j = [k for k, o in enumerate(names)
             if o[:2] == n[:2] and kind_of(o) == "calf"]
        if j:
            out[:, i] = arr[:, i] + (1.0 - c) * arr[:, j[0]]
    return out


def actuator_wrap(m, idx, names, ctrl_space="tendon"):
    """액추에이터별 **PD 가 무엇을 오차로 재는가** 를 MJCF 에서 읽어 만든다.

    ★2026-08-14 발견 — foot 만 **힘 쪽만 옮기고 오차 쪽을 안 옮겼다.**
      실기 드라이버는 **채널각**으로 PD 를 건다. foot 의 채널각은
          q_ch = (q_foot + coef·q_calf)·sign·k + offset      (coef=+1, biped_emb.yaml)
      이므로 드라이버가 보는 오차는 q_foot 이 아니라 **(q_foot + q_calf)** 의 오차다.
      kp_joint = kp_ch·k² 도 그 raw 오차에 곱해지는 값이다(τ_joint = k²·kp_ch·Δraw).
      MJCF 는 이미 foot 액추에이터를 `<fixed>` tendon(coef 1,1)으로 옮겨 뒀는데
      — **힘을 주는 쪽만** 옮겼고 `rollout` 의 오차는 관절각에 남아 있었다.

    ⚠증거(f0.4_dqfix, 적합구간, foot RMS):
        적합 안 된 x0 에서   관절각 1.0451° → **raw각 0.7072°**  (−32%)
        적합된 θ 에서        관절각 0.6866° → raw각 0.9752°      (+42%)
      뒤집힘 자체가 진단이다 — 틀린 법칙으로 적합하면 CMA-ES 가 그 오차를
      armature·마찰로 **흡수**하고, 그 상태에서 법칙만 바꾸면 당연히 나빠진다.
      이 파일 독스트링이 경고해 둔 "잘 맞는데 물리적으로 틀린 값" 그 자체다.
      ⇒ 기본값을 tendon 으로 둔다. `--ctrl-space joint` 로 옛 동작을 재현할 수 있다.

    반환: 축별 튜플 (col, qpos_adr, dof_adr, coef) 들, 커플 없으면 None.
    """
    import mujoco
    if ctrl_space == "joint":
        return [None] * len(idx)
    qadr = {n: idx[i][1] for i, n in enumerate(names)}
    dadr = {n: idx[i][2] for i, n in enumerate(names)}
    col = {n: i for i, n in enumerate(names)}
    out = []
    for i, (_, qa, dofa, aid) in enumerate(idx):
        tid = m.actuator_trnid[aid, 0]
        if m.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_TENDON:
            out.append(None)
            continue
        # fixed tendon 의 wrap 목록을 그대로 읽는다 — MJCF 가 바뀌어도 따라간다
        w = []
        for k in range(m.tendon_adr[tid], m.tendon_adr[tid] + m.tendon_num[tid]):
            jid = m.wrap_objid[k]
            jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            hit = [n for n in names if jn == n + "_joint" or jn.startswith(n)]
            if not hit:                       # 데이터에 없는 관절이면 무시한다
                continue
            w.append((col[hit[0]], qadr[hit[0]], dadr[hit[0]], float(m.wrap_prm[k])))
        out.append(tuple(w) if w else None)
    return out


def rollout(m, d, idx, q_real, dq_real, q_cmd, kp, kd, dt, win_steps,
            bias=None, delay_s=0.0, wrap=None):
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
                w = None if wrap is None else wrap[i]
                if w is None:                                # raw각 == 모델각 (hip·thigh·calf)
                    err = (q_cmd[tc, i] - d.qpos[qa] / DEG) * DEG
                    vel = d.qvel[dofa]
                else:
                    # ★foot — 드라이버는 **채널각**(= raw 의 선형사상)으로 PD 를 건다.
                    #   q_raw_foot = q_foot + coef·q_calf 이므로 오차도 그 합이다.
                    #   RL_INTERFACE.md §0·§1 · d.ctrl 이 드라이브 토크인 것은 §6-d.
                    err = vel = 0.0
                    for c_, qa_, da_, cf_ in w:
                        err += cf_ * (q_cmd[tc, c_] - d.qpos[qa_] / DEG) * DEG
                        vel += cf_ * d.qvel[da_]
                d.ctrl[aid] = kp[i] * err - kd[i] * vel
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
    return dyn + [f"bias.{n}" for n in names] + ["delay", "coef"]


def init_bounds(spec_path, names, per_axis, pin=()):
    """초기값·탐색범위 — **축별 측정값**에서 온다. 이게 sloppy 를 줄이는 핵심이다.

    ★pin — 그 kind 의 JDAMP·JFRIC 을 **탐색에서 뺀다**(값은 x0 에 고정).
      2026-08-14 f0.4_dqfix 적합에서 hip 이 탐색범위 **모서리**로 갔다:
          JFRIC.hip 상한 99% (1.069 / 상한 1.075) · JDAMP.hip 하한 1%
      이건 JDAMP↔JFRIC 서로 맞바꿈(r=+0.93)의 평탄방향을 따라 미끄러진 것이다. 왜 hip 이냐면
      발끝 충돌 때문에 진폭을 5°(thigh 17.4° 의 1/3.6)로 줄였고, 그 결과 hip 이
      **비용의 4% 밖에 안 되기 때문**이다 — 아무 데나 밀어도 손해가 없다.
      ⇒ 데이터가 못 보는 축에 2모수를 낭비하지 않는다. 실측 JFRIC 으로 고정한다.
      ⚠JDAMP 는 실측이 **없다**(외삽값 0.09). 고정하는 건 '측정했다' 가 아니라
        '이 데이터로는 정할 수 없으니 흔들지 않는다' 는 뜻이다."""
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

        def _kindmean(d):
            """채널 dict → kind별 평균 [관절 Nm]. spec 은 채널토크라 ×gear_k."""
            b = {}
            for c, v in (d or {}).items():
                c = int(c)
                if c < len(names):
                    b.setdefault(kind_of(names[c]), []).append(float(v) * _gk.get(c, 1.0))
            return {k: float(np.mean(v)) for k, v in b.items()}

        _sw = _kindmean(_mc)                                        # 위치모드 스윕
        _dy = _kindmean((sp.get("friction") or {}).get("measured_coulomb_dyn_ch"))
        # ★두 실측이 **최대 2배 벌어진다**. 어느 하나를 고르지 않고 **둘 다 담는다**
        #   (2026-08-14).
        #     kind    스윕     토크절편   비
        #     thigh  0.670    0.409    1.64
        #     calf   1.001    0.761    1.32
        #     foot   0.745    0.359    2.08
        #   갈리는 이유는 안다 — 스윕은 τ_c 와 b·q̇ 를 **뭉쳐** 재고, 토크절편은 q̇→0 으로
        #   외삽해 τ_c 만 낸다. MuJoCo 의 frictionloss 정의에는 후자가 맞지만, thigh 의
        #   절편은 방향편차 15.8% 로 게이트를 못 넘은 런에서 나왔다 — 못 믿는다.
        #   ⇒ **초기값은 기하평균, 탐색범위는 둘을 감싸게** 잡는다. 고르는 건 적합에 맡긴다.
        #     한쪽만 있으면(hip) 종전대로 그 값 ±비율이다.
        f0, _span_lo, _span_hi = {}, {}, {}
        for k in set(_sw) | set(_dy):
            a, b = _sw.get(k), _dy.get(k)
            if a and b:
                f0[k] = float(np.sqrt(a * b))
                _span_lo[k] = min(a, b) * (1.0 - 0.30)
                _span_hi[k] = max(a, b) * (1.0 + 0.30)
            else:
                f0[k] = float(a or b)
        # ★±20% → **±30%** (2026-08-12, 실기 데이터로 스캔 후).
        #   깨끗한 수집본(500Hz·dt 오차 0%)에서 배율 스캔을 하니 최소가 **×0.8** 이었다:
        #     ×0.6 0.6689 · **×0.8 0.6628** · ×1.0 0.6780 · ×1.2 0.7103 · ×1.4 0.7541 (따로 뺀 구간)
        #   ±20% 면 최적이 경계에 딱 붙어 탐색이 벽을 민다 — 내부에 두려면 ±30% 가 필요하다.
        #   ⚠그래도 여전히 **못박는 것**이다: 종전 ×[0.3, 3.0](폭 10배) → ×[0.7, 1.3](1.9배).
        #     JDAMP↔JFRIC 맞바꿈을 깨는 목적은 그대로 달성된다.
        #   ★실측 ×1.0 이 구 추정치(0.38/0.44)보다 낫다 — 적합 0.6603 vs 0.6689 ·
        #     따로 뺀 구간 0.6780 vs 0.6889. **오늘 잰 마찰이 유효하다는 증거다.**
        #     최적이 0.8 인 방향은 설명된다: 우리는 ≤20dps 평탄부에서 쟀는데 처프는
        #     채널 90dps 까지 간다(Stribeck 이면 고속에서 낮다).
        JFRIC_SPAN[0] = 0.30
    else:
        f0 = {"hip": 0.38, "thigh": 0.38, "calf": 0.38, "foot": 0.44}   # 구 추정치
        JFRIC_SPAN[0] = None
    # ★JDAMP 실측이 있으면 초기값을 그걸로 덮는다 (2026-08-14).
    #   토크시험의 q̇_ref 훑기가 HL_foot 에서 b_joint ≈ 0.20 을 냈다(무릎 자유·고정
    #   두 조건에서 0.214/0.202 로 재현). 그런데 종전 x0 은 0.02 였고 탐색범위가
    #   ×0.1~10 = [0.002, 0.2] 라 **실측이 천장에 딱 걸려 있었다.**
    #   그 상태로 적합하면 CMA-ES 가 하한 0.002 로 내려가 버린다(A′·B′ 둘 다 그랬다).
    #   ⚠이 값은 **괄호지 값이 아니다** — I 가 속도마다 21~24% 흔들려 가드가 b 를
    #     거부했다. 그래도 "0.002 냐 0.2 냐" 는 100배 차이라 초기값·탐색범위를 잡는 데는
    #     충분하고, 안 넣으면 탐색범위 밖에 최적이 있는 상태로 계속 돈다.
    _md = (sp.get("friction") or {}).get("measured_damping_joint") or {}
    if _md:
        # ⚠kind 별 **평균**을 쓴다. 종전엔 마지막 채널 값이 이겼다 — foot 은 좌우가
        #   0.12/0.10 인데 0.10 만 반영됐다(dict 순서에 결과가 달렸다는 뜻이다).
        _by = {}
        for c, v in _md.items():
            c = int(c)
            if c < len(names):
                _by.setdefault(kind_of(names[c]), []).append(float(v))
        for k, vs in _by.items():
            d0[k] = float(np.mean(vs))
    if per_axis:
        x0 = np.concatenate([[rot0], [d0[kind_of(x)] for x in names],
                             [f0[kind_of(x)] for x in names]])
    else:
        x0 = np.concatenate([[rot0], [d0[k] for k in KINDS], [f0[k] for k in KINDS]])
    # ★PACE 원문 파라미터를 마저 넣는다: 관절 bias q̃_b(축별) + 전역 지연 T_d.
    #   원문은 4n+1. 우리는 8축이 **같은 모터**라 ROTOR_I 를 공유하고 마찰·감쇠를
    #   kind별로 묶으므로 (1+4+4) + n(bias) + 1(delay) = 18 이다.
    #   ⚠bias 는 축별이어야 한다 — 엔코더 영점은 물리량이 아니라 축마다 따로다.
    # ★탐색범위 — 파라미터마다 **근거가 다르므로** 폭도 다르다 (2026-08-14 재조정).
    #   종전은 전부 ×0.3~3.0 이었는데, f0.4_dqfix 적합에서 둘이 탐색범위 끝까지 밀렸다:
    #     ROTOR_I    하한 1%  (×0.332) — 최적이 탐색범위 **밖 아래**
    #     JDAMP.foot 상한 95% (×2.88)  — 최적이 탐색범위 **밖 위**
    #   탐색범위 끝에 붙은 값은 "거기가 벽이었다" 는 말이지 식별된 값이 아니다.
    #   JDAMP 는 **실측이 하나도 없다**(각축 8축 전부 nan/≈0). 실측으로 좁힌 JFRIC 과
    #   달리 좁힐 근거가 없으니 넓게 연다. ROTOR_I 도 하한만 연다.
    lo = x0 * 0.3
    hi = x0 * 3.0
    nk = len(names) if per_axis else len(KINDS)
    lo[0] = x0[0] * ROTOR_SPAN[0]                      # ROTOR_I 하한만 확장
    hi[0] = x0[0] * ROTOR_SPAN[1]
    lo[1:1 + nk] = x0[1:1 + nk] * JDAMP_SPAN[0]        # JDAMP 양쪽 확장
    hi[1:1 + nk] = x0[1:1 + nk] * JDAMP_SPAN[1]
    # JFRIC 구간만 실측 기반으로 조인다. 벡터 순서는 [ROTOR_I, JDAMP…, JFRIC…] 이다.
    if JFRIC_SPAN[0] is not None:
        nj = len(names) if per_axis else len(KINDS)
        sl = slice(1 + nj, 1 + 2 * nj)
        # ★탐색범위를 **비대칭**으로 연다 (2026-08-14). 아래쪽을 더 넓게.
        #   실측 JFRIC 은 **≤20dps 평탄부**에서 쟀는데 처프는 40~90dps 로 돈다.
        #   토크시험이 그 속도대의 운동마찰을 따로 쟀고 **정지마찰의 54~89%** 였다
        #   (Stribeck). 즉 실측값은 처프 속도대에서 **과대평가**다.
        #   실제로 네 번의 적합에서 JFRIC.calf·JFRIC.foot 이 **매번 하한에 박혔다** —
        #   최적이 탐색범위 밖 아래라는 뜻이다. 대칭 ±30% 로는 못 담는다.
        #   ⇒ 아래 −50% · 위 +30%. 위쪽은 넓힐 이유가 없다(실측보다 큰 마찰은
        #     물리적 근거가 없고, 넓히면 JDAMP 와의 맞바꿈만 키운다).
        lo[sl] = x0[sl] * (1.0 - JFRIC_SPAN_DN[0])
        hi[sl] = x0[sl] * (1.0 + JFRIC_SPAN[0])
        # 두 실측이 다 있는 kind 는 **둘을 감싸는** 탐색범위로 덮어쓴다(위 주석)
        _kk = names if per_axis else KINDS
        for i, k in enumerate(_kk):
            kk = kind_of(k) if per_axis else k
            if kk in _span_lo:
                lo[1 + nj + i] = _span_lo[kk]
                hi[1 + nj + i] = _span_hi[kk]
    nb = len(names)
    # ★coef — calf→foot 커플링 계수 (2026-08-14 추가).
    #   저장소에 남은 **유일한 미측정 구조 파라미터**다. RL_INTERFACE:
    #   "부호는 [실측], 크기 1.0 은 [미확정], 근거는 육안 일치뿐".
    #
    #   ★넣게 된 동기는 **틀렸고**, 결과는 맞았다 — 기록해 둔다.
    #     동기: "foot 이 잔차의 63~72% 이고 coef 3~5% 오차면 0.44~0.73° 가 나온다".
    #     그런데 실제로 c 를 낮춰 보니 개선이 **calf 에서** 나왔다(−22%/−18%,
    #     foot 은 −3%). 즉 coef 는 foot 위치오차가 아니라, foot 액추에이터의 반력·
    #     회전자 관성이 tendon 을 타고 calf 로 실리는 양을 고친다.
    #     ⇒ 가설의 논거는 폐기하되, 파라미터 자체는 남긴다.
    #
    #   ★★결론 — **자유롭게 풀 값이 아니다. c=1.0 으로 고정하는 게 맞다** (2026-08-14).
    #     x0 에서 1-D 로 훑었을 때 최소가 0.76 이었고 따로 뺀 구간도 −7.6% 로 따라와서
    #     "실물이 1 이 아니다" 로 읽었다. **틀렸다.** 전체 적합을 돌리면:
    #         fit_v2 (coef 고정 1.000)  적합 0.4050 · 따로 뺀 구간 0.3967
    #         fit_v3 (coef 자유→1.064)  적합 0.4058 · 따로 뺀 구간 0.3983
    #     자유로 풀면 **오히려 나빠진다** — 차원만 늘고 얻는 게 없다.
    #
    #   ⚠⚠맞바꿈 판정(tests/scan_coef_jdamp.py)이 **왜 틀렸는지** 남긴다.
    #     격자에서 각 행의 JDAMP.calf 최소가 전부 ×0.5 였고 그걸 "coef 와 무관 = 독립"
    #     으로 읽었다. 그런데 ×0.5 는 **훑은 범위의 끝**이었다. 진짜 최적은
    #     JDAMP.calf = 0.0097 (×0.108) 로 격자 밖이다.
    #     ⇒ 이 파일이 box_report 로 경고하는 그 실수(벽에 박힌 값을 식별된 값으로 읽기)를
    #       진단 스크립트 안에서 냈다. JDAMP.calf 를 충분히 낮출 수 있으면 calf 잔차가
    #       그쪽으로 흡수되고 coef 는 할 일이 없어진다.
    #     ⇒ 격자 진단은 **행 최소가 격자 경계면 결론을 내지 않는다**(가드 추가함).
    #
    #   ★탐색범위는 0.70~1.15 로 남긴다. 기본은 고정(--pin coef)이지만, 재수집 데이터로
    #     다시 볼 때 범위가 좁아 못 보는 일은 없어야 한다. 1.064 는 벽이 아니었다(81%).
    #     ⚠0.80 = k_foot/k_calf(1.2/1.5) 는 **우연이다**. coef 는 joint_map.py:238 대로
    #       이미 감속비 이후 공간의 계수이고, 채널공간과 혼동했을 때 나오는 값은
    #       0.8 이 아니라 **1.25**(과보상)다 — 방향이 반대라 그 설명은 성립하지 않는다.
    #   ⚠⚠couple_check.py 로는 크기를 못 잰다. 2026-08-14 실기 결과가 **(A)** 다
    #     (HL 115.9°→기울기 +0.0001 · HR 148.9°→+0.0000). 엔코더가 모터축에 있어
    #     커플링을 아예 못 보므로 크기 정보가 원리적으로 없다.
    #     ⇒ 결정적 측정은 **발목을 채널각 고정으로 잡고 무릎을 크게 돌린 뒤 발바닥
    #       절대각을 경사계로 재는 것**이다. Δ발_절대각 = Δq_calf·(1−c) 이므로
    #       100° 스윙에서 c=1.00 → 0° · c=0.76 → 24°.
    x0 = np.concatenate([x0, np.zeros(nb), [DELAY0], [1.0]])
    lo = np.concatenate([lo, np.full(nb, -BIAS_MAX), [DELAY_LO], [COEF_LO]])
    hi = np.concatenate([hi, np.full(nb, +BIAS_MAX), [DELAY_HI], [COEF_HI]])
    # ★고정축은 **탐색벡터에서 뺀다**(경계를 붙이는 게 아니라 차원을 없앤다).
    #   lo==hi 로 눌러도 되지만 그러면 CMA-ES 가 죽은 차원을 계속 흔든다 —
    #   popsize 10 에 18차원이라 2차원 낭비가 작지 않다.
    free = np.ones(len(x0), bool)
    if pin:
        lab = param_labels(names, per_axis)
        # ★"rotor" — ROTOR_I 를 **실측으로 못박는다** (2026-08-14).
        #   두 독립 측정이 만났다:
        #       foot τ_ff 경로 (2026-08-11)      7.327e-4
        #       calf 공통속도법 (2026-08-14)     7.340e-4   (+0.17%)
        #   감속비(8.4 vs 10.5)도 방법도 다른 두 축이 0.17% 로 일치한다.
        #   calf 쪽 근거: I_joint 실측 0.11192(MJCF 예측 −0.6% · 방향간 편차 9.4% ·
        #   R² 0.997/0.986) 에서 I_link 0.0310 을 빼고 N²=110.25 로 나눈 값이다.
        #   ⚠f0.4_dqfix 적합은 2.436e-4(−66.8%)를 냈고 **탐색범위 하한 1%** 에 박혀 있었다.
        #     그건 foot 제어법칙 버그를 armature 로 흡수한 결과다 — 실측을 믿는다.
        if "rotor" in pin:
            free[0] = False
        if "coef" in pin:
            free[-1] = False
        for i, L in enumerate(lab):
            # kind 로 지정(hip → JDAMP.hip·JFRIC.hip 둘 다) 또는 **개별 라벨**로 지정
            #   (JDAMP.foot → 그 하나만). 실측이 한쪽만 있을 때 개별 지정이 필요하다.
            if L in pin:
                free[i] = False
            elif L.startswith(("JDAMP.", "JFRIC.")) and kind_of(L.split(".", 1)[1]) in pin:
                free[i] = False
    return x0, lo, hi, free


# bias·delay 경계 — 실측 근거로 잡는다(임의값이 아니다)
JFRIC_SPAN = [None]   # init_bounds 가 채운다(실측 있으면 +비율)
JFRIC_SPAN_DN = [0.75]  # ★아래쪽은 더 넓게 — Stribeck 으로 실측이 과대평가다(위 주석)
#   2026-08-14: 0.50 → 0.75. 0.50 이어도 **JFRIC.foot 이 매번 바닥(0%)** 이다
#   (fit_v2·fit_v3 둘 다 0.2516). JDAMP.foot 은 실측 0.11 로 못박혀 있으니 이건
#   JDAMP 와의 맞바꿈이 아니라 "발목 마찰이 실측보다 더 낮다" 는 한 방향 신호다.
ROTOR_SPAN = (0.10, 3.0)   # 실측 초기값이 있으나 τ_ff 경로 1점 — 하한을 연다
JDAMP_SPAN = (0.02, 10.0)  # **실측이 없다**(각축 전부 nan). 좁힐 근거가 없다
#   2026-08-14: 하한 0.10→0.02. JDAMP.calf 가 fit_v2·fit_v3 에서 **둘 다 바닥**
#   (0.0097 = x0×0.108)에 박혔다. 점성감쇠가 0 으로 가고 싶다는 뜻인데 막혀 확인이
#   안 됐다. JFRIC.calf 는 범위 안(7%)이라 둘이 같이 미끄러지는 맞바꿈이 아니라
#   **calf 손실이 쿨롱 지배적**이라는 신호다 — 그 방향을 열어 준다.
BIAS_MAX = 3.0        # [deg] 지그 영점 후 잔차가 모델각 0.5~2.3° 였다(2026-08-11)
# ★지연은 **직접 실측**했다: 8.39 ± 0.79 ms (act_measure_latency.py).
#   원문(PACE)은 T_d 를 자유롭게 탐색하는데, 그건 그 값을 따로 재지 않았기 때문이다.
#   우리는 있으니 **±2σ 로 묶는다**(6.8~10.0 ms).
#   ⚠근거: 넓게(2~16ms) 뒀더니 추정이 **12.83 ms** 로 실측 범위 밖까지 나갔다
#     (셀프테스트 T=4·120세대). 지연이 모델오차를 **흡수**한 것이고, 그 대가로
#     JFRIC 이 −13~−63% 로 무너졌다. 실측이 있는 값을 자유변수로 두면 이렇게 된다.
DELAY0, DELAY_LO, DELAY_HI = 0.00839, 0.0068, 0.0100
COEF_LO, COEF_HI = 0.70, 1.15   # ★커플링 계수 탐색범위 — 아래만 열었다 (init_bounds 주석)


def split_segments(N, win, holdout, mode="tail"):
    """적합/따로 뺀 구간 구간을 나눈다. 반환은 (s,e) 목록 두 개.

    ★왜 `interleave` 가 필요한가 (2026-08-14)
      종전엔 **뒤쪽 연속 20%** 만 따로 뺀 구간 이었다. 그런데 우리 데이터는 **처프**다 —
      뒤쪽은 곧 **고주파**다(적합구간 |dq|95% 40~46dps · 따로 뺀 구간 42~51dps).
      그러면 따로 뺀 구간 은 일반화 시험이 아니라 **"안 배운 주파수로의 외삽"** 시험이 된다.
      실제로 foot PD 법칙 A/B 비교에서 이게 결론을 뒤집었다:
          A(raw)   적합 0.3568 · 따로 뺀 구간 0.7711   ← 저주파에 강하고 고주파에 약함
          B(joint) 적합 0.3782 · 따로 뺀 구간 0.5000
      A 는 저주파에서 3배 좋은데 꼬리(고주파)만 보는 따로 뺀 구간 에서는 진다.
      ⇒ 창 단위로 **번갈아** 떼면 양쪽이 같은 주파수 범위를 본다. 그래야 비교가 된다.
    ⚠창(win) 경계로 자른다. 롤아웃이 어차피 창마다 실측으로 재초기화하므로
      창을 쪼개지 않는 한 의미가 바뀌지 않는다.
    """
    if mode == "tail":
        ncut = int(N * (1 - holdout))
        return [(0, ncut)], [(ncut, N)]
    blocks = [(s_, min(s_ + win, N)) for s_ in range(0, N, win)]
    k = max(2, int(round(1.0 / max(holdout, 1e-9))))    # k 개마다 1개를 따로 뺀 구간
    fit_s = [b for i, b in enumerate(blocks) if i % k != k - 1]
    hold_s = [b for i, b in enumerate(blocks) if i % k == k - 1]
    return fit_s, hold_s


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
    coef = float(p[nd + n + 1]) if len(p) > nd + n + 1 else 1.0
    return p[:nd], p[nd:nd + n], float(p[nd + n]), coef


def box_report(labels, x, x0, lo, hi, free=None, log=print) -> list:
    """탐색값이 **탐색범위 어디에 있는지** 찍는다. 탐색범위 끝에 붙은 항목 라벨을 반환한다.

    ★왜 매번 찍나 (2026-08-14)
      f0.4_dqfix 적합에서 ROTOR_I·JDAMP.foot·JFRIC.hip 셋이 탐색범위 끝에 붙어 있었는데,
      출력에는 값만 있어서 **한참 뒤에야** 알았다. 탐색범위 끝에 붙은 값은 "최적이 탐색범위 밖" 이라는
      뜻이지 식별된 값이 아니다 — 그걸 결론으로 쓰면 안 된다. 그러니 값과 **같이** 찍는다.
    """
    log(f"\n  {'파라미터':<16}{'하한':>11}{'값':>12}{'상한':>11}{'범위내':>8}  판정")
    wall = []
    for i, L in enumerate(labels):
        if L.startswith("bias"):
            continue
        if free is not None and not free[i]:
            log(f"  {L:<16}{'':>11}{x[i]:>12.4g}{'':>11}{'고정':>8}  — 탐색 제외")
            continue
        a, b = float(lo[i]), float(hi[i])
        u = (x[i] - a) / (b - a) if b > a else float("nan")
        if u >= 0.95 or u <= 0.05:
            v = "★범위 끝까지 밀렸다 — 최적이 범위 밖이다"
            wall.append(L)
        elif u >= 0.85 or u <= 0.15:
            v = "△범위 끝 근처"
        else:
            v = "✓범위 안에서 정해졌다"
        log(f"  {L:<16}{a:>11.4g}{x[i]:>12.4g}{b:>11.4g}{u:>7.0%}  {v}")
    return wall


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
    if len(x) >= nd + n + 2:
        log(f"  {'coef':<16}{float(x[nd+n+1]):>12.4f}   "
            f"(커플링 계수 — 1.0 은 **가정**이지 실측이 아니다)")


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
                    help="이 비율은 **적합에 안 쓰고** 검증에만 쓴다")
    ap.add_argument("--holdout-mode", default="tail", choices=("tail", "interleave"),
                    help="tail=뒤쪽 연속(종전) · interleave=창 단위로 번갈아. "
                         "처프 데이터에서 tail 은 **고주파 외삽 시험**이 된다 — "
                         "모델 비교에는 interleave 를 쓸 것(split_segments 독스트링 참조)")
    ap.add_argument("--validate", default=None, metavar="npz",
                    help="★unseen PD gains 검증 (PACE arXiv:2509.06342). 다른 게인으로 수집한 "
                         "npz 를 주면, 적합된 θ 를 **그 데이터**에 걸어 RMS 를 낸다. "
                         "게인이 바뀌어도 같은 θ 가 맞으면 순환·과적합이 아니다")
    ap.add_argument("--pin", default="", metavar="KIND[,KIND]",
                    help="그 kind 의 JDAMP·JFRIC 을 **탐색에서 뺀다**(x0 에 고정). "
                         "예: --pin hip — 데이터가 hip 을 4%%밖에 안 보므로 탐색범위 모서리로 간다. "
                         "`rotor` 는 ROTOR_I 를 실측(7.33e-4)에 못박는다")
    ap.add_argument("--ctrl-space", default="tendon", choices=("tendon", "joint"),
                    help="foot PD 가 재는 오차. tendon=raw각(q_foot+q_calf, **실기**) · "
                         "joint=관절각(2026-08-14 이전 동작). actuator_wrap 독스트링 참조")
    ap.add_argument("--out", default=None, metavar="npz",
                    help="산출물 경로. 기본은 <입력>_cmaes.npz 인데, 조건을 바꿔 여러 번 "
                         "돌리면 **서로 덮어쓴다** — 통제군 비교를 하려면 반드시 나눌 것")
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
    fit_segs, hold_segs = split_segments(N, win, a.holdout, a.holdout_mode)
    print(f"■ 데이터 {os.path.basename(a.npz)} — {N} 표본 · dt {D['dt']*1000:.1f}ms · "
          f"창 {a.window}s({win} 스텝)")
    _nf = sum(e - s_ for s_, e in fit_segs); _nh = sum(e - s_ for s_, e in hold_segs)
    if a.holdout_mode == "tail":
        print(f"  적합 구간 0~{fit_segs[0][1]} · **따로 뺀 구간 {hold_segs[0][0]}~{N}** "
              f"({a.holdout:.0%})")
        print(f"  ⚠처프에서 뒤쪽은 **고주파**다 — 이건 일반화가 아니라 **외삽** 시험이다."
              f" 모델 비교에는 `--holdout-mode interleave` 를 쓸 것")
    else:
        print(f"  **창 단위 교차분할** — 적합 {len(fit_segs)}창({_nf}표본) · "
              f"따로 뺀 구간 {len(hold_segs)}창({_nh}표본, {_nh/N:.0%})")
        print(f"  ⇒ 양쪽이 같은 주파수 범위를 본다(처프의 tail 편향 제거)")

    pin = tuple(k.strip() for k in a.pin.split(",") if k.strip())
    _lab_all = param_labels(D["names"], a.per_axis)
    for k in pin:
        if k not in KINDS + ("rotor", "coef") and k not in _lab_all:
            raise SystemExit(f"✗ --pin {k} — {KINDS + ('rotor',)} 또는 개별 라벨"
                             f"(예: JDAMP.foot) 이어야 한다")
    x0, lo, hi, free = init_bounds(a.spec, D["names"], a.per_axis, pin)
    wrap = actuator_wrap(m, idx, D["names"], a.ctrl_space)
    _nw = sum(1 for w in wrap if w)
    print(f"■ 제어공간 — foot PD 오차 = "
          + ("**raw각**(q_foot+q_calf, 실기 드라이버와 같다)" if a.ctrl_space == "tendon"
             else "관절각(2026-08-14 이전 동작 — 실기와 다르다)")
          + f"  · 커플 액추에이터 {_nw}개")
    print(f"■ 초기값 — {len(x0)} 모수 "
          f"(동역학 {len(x0)-len(D['names'])-1} + bias {len(D['names'])} + delay 1). "
          f"ROTOR_I ×{ROTOR_SPAN[0]}~{ROTOR_SPAN[1]} · JDAMP ×{JDAMP_SPAN[0]}~{JDAMP_SPAN[1]} · "
          f"JFRIC ×{1-JFRIC_SPAN_DN[0]:.2g}~{1+JFRIC_SPAN[0]:.2g} · "
          f"bias ±{BIAS_MAX}° · delay {DELAY_LO*1e3:.0f}~{DELAY_HI*1e3:.0f}ms")
    report(D["names"], x0, a.per_axis)
    plab = param_labels(D["names"], a.per_axis)
    if not free.all():
        # ★여기서 찍는다 — `--eval-only` 로 **고정이 걸렸는지 먼저 확인**할 수 있어야 한다.
        print("\n■ 고정(탐색 제외) — 실측이 있거나 이 데이터로는 정할 수 없다")
        for i in np.flatnonzero(~free):
            print(f"    {plab[i]:<16}{x0[i]:>12.4g}")
        print(f"    자유차원 {int(free.sum())}/{len(x0)}")

    def evaluate(p, segs):
        """구간 **목록**에 대해 RMS 를 낸다. 표본수로 가중해야 창 길이가 달라도 맞다."""
        dyn, bias, dly, cf = split_params(p, len(D["names"]), a.per_axis)
        apply_params(m, idx, D["gear_n"], dyn, a.per_axis, D["names"])
        # ★coef 는 **두 곳**을 같이 바꿔야 한다 — MJCF 의 힘·관성 쪽(tendon wrap)과
        #   데이터의 기구학 쪽(저장된 모델각은 coef=1 가정으로 만들어졌다).
        #   한쪽만 바꾸면 물리적으로 앞뒤가 안 맞는 모델이 된다.
        set_coupling_coef(m, cf)
        Q  = retarget_coupling(D["q"], D["names"], cf)
        QC = retarget_coupling(D["q_cmd"], D["names"], cf)
        DQ = retarget_coupling(D["dq"], D["names"], cf)
        ss = 0.0; cnt = 0; last = None
        for s_, e in segs:
            qs = rollout(m, d, idx, Q[s_:e], DQ[s_:e], QC[s_:e],
                         D["kp"], D["kd"], D["dt"], win, bias=bias, delay_s=dly, wrap=wrap)
            r = qs - (Q[s_:e] - bias)
            ss += float(np.sum(r ** 2)); cnt += r.size; last = qs
        return float(np.sqrt(ss / max(cnt, 1))), last

    c0, _ = evaluate(x0, fit_segs)
    h0, _ = evaluate(x0, hold_segs)
    print(f"\n■ 초기값 RMS — 적합 {c0:.4f}° · 따로 뺀 구간 {h0:.4f}°")
    if a.eval_only:
        return 0

    try:
        import cma
    except ImportError:
        raise SystemExit("✗ cma 가 없다: ~/.venv-mujoco/bin/pip install cma")
    # ★z∈[0,1] 정규화 공간에서 탐색한다(to_z 주석 참조). 원공간은 스케일이 4자리 벌어진다.
    z0 = to_z(x0, lo, hi)

    def expand(zf):
        """자유차원 z → **전체** 실공간 벡터. 고정축은 언제나 x0 값이다."""
        z = z0.copy()
        z[free] = zf
        return from_z(z, lo, hi)

    es = cma.CMAEvolutionStrategy(list(z0[free]), 0.25, {
        "popsize": a.popsize, "maxiter": a.iters,
        "bounds": [0.0, 1.0], "verbose": -9})       # ★z 공간이므로 경계도 [0,1] 이다
    print(f"\n■ CMA-ES — popsize {a.popsize} · 최대 {a.iters} 세대 "
          f"(z∈[0,1] 정규화 · 자유차원 {int(free.sum())}/{len(x0)})")
    best, bestc = np.array(x0), c0
    hist = []
    it = 0
    while not es.stop():
        Z = es.ask()
        X = [expand(z) for z in Z]
        F = [evaluate(x, fit_segs)[0] for x in X]
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
    hb, qs = evaluate(best, hold_segs)
    print(f"\n■ 결과 — 적합 RMS {bestc:.4f}° · **따로 뺀 구간 RMS {hb:.4f}°** "
          f"(초기 {c0:.4f}/{h0:.4f})")
    print(f"  개선 적합 {(1-bestc/c0)*100:+.1f}% · 따로 뺀 구간 {(1-hb/h0)*100:+.1f}%")
    if hb > bestc * 1.5:
        print("  ⚠따로 뺀 구간 이 적합보다 크게 나쁘다 — **과적합**이다. 모수를 줄이거나"
              " 데이터를 늘릴 것")
    report(D["names"], best, a.per_axis)
    _wall = box_report(plab, best, x0, lo, hi, free)
    if _wall:
        print("  ★**탐색범위 끝에 붙은 값은 식별된 값이 아니다** — 최적이 탐색범위 밖이라는 뜻이다."
              " 경계를 넓혀 다시 돌리거나, 실측으로 고정할 것")

    # ── ★unseen PD gains 검증 (원문의 주 검증) ──────────────────────────────
    #   따로 뺀 구간 궤적보다 강하다: 게인이 바뀌면 kp·err 순환이나 과적합이 바로 드러난다.
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
            print("  ⚠게인이 같다 — 이건 unseen gains 검증이 아니다(궤적 따로 뺀 구간 일 뿐)")
        rows = []
        for lab, px in (("초기값", x0), ("적합 θ", best)):
            dyn, bias, dly, cf = split_params(px, len(V["names"]), a.per_axis)
            apply_params(m, idx, V["gear_n"], dyn, a.per_axis, V["names"])
            set_coupling_coef(m, cf)
            VQ  = retarget_coupling(V["q"], V["names"], cf)
            VQC = retarget_coupling(V["q_cmd"], V["names"], cf)
            VDQ = retarget_coupling(V["dq"], V["names"], cf)
            qs = rollout(m, d, vi, VQ, VDQ, VQC, V["kp"], V["kd"],
                         V["dt"], vw, bias=bias, delay_s=dly,
                         wrap=actuator_wrap(m, vi, V["names"], a.ctrl_space))
            rows.append((lab, cost_of(qs, VQ - bias)))
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

    # ★조건별로 파일을 나눈다 (2026-08-14). 종전엔 입력 이름에서만 만들어서
    #   `--ctrl-space joint` 통제군을 돌리면 앞 결과를 **말없이 덮어썼다**.
    #   비교하려고 돌린 두 실행이 한 파일을 쓰면 비교 자체가 불가능하다.
    out = a.out or (os.path.splitext(a.npz)[0] + "_cmaes.npz")
    np.savez(out, x=best, x0=x0, rms_fit=bestc, rms_holdout=hb,
             per_axis=a.per_axis, names=np.array(D["names"]),
             # ★탐색범위를 같이 남긴다 (2026-08-14). 안 남겨서 "탐색범위 끝에 붙었는지" 를
             #   나중에 init_bounds 를 다시 불러 손으로 계산해야 했다.
             lo=lo, hi=hi, free=free, labels=np.array(plab),
             ctrl_space=a.ctrl_space, pin=np.array(pin))
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
    # ★참값은 **탐색범위 안에서** 만든다 (2026-08-14 수정).
    #   종전엔 JFRIC 참값을 [0.42, 0.31, 0.50, 0.44] 로 **하드코딩**했는데, 2026-08-12 에
    #   JFRIC 탐색범위를 실측 ×[0.7,1.3] 으로 조인 뒤로 **네 개 전부 탐색범위 밖**이 됐다:
    #       hip 0.42 ∉ [0.579,1.075] · thigh 0.31 ∉ [0.469,0.871]
    #       calf 0.50 ∉ [0.700,1.301] · foot 0.44 ∉ [0.521,0.968]
    #   ⇒ 도달 불가능한 값을 되찾으라고 시킨 셈이라 **구조적으로 항상 실패**했다.
    #     "셀프테스트 실패" 가 상시화되면 진짜 회귀를 못 알아본다.
    #   x0 에 배율을 곱해 만든다 — 탐색범위가 바뀌어도 따라가고, x0 와 충분히 달라
    #   "안 움직였다" 와 "되찾았다" 가 구분된다.
    _x0s, _lo_s, _hi_s, _ = init_bounds(a.spec, names, False)
    _mul = np.array([1.35] + [1.6, 0.55, 1.9, 2.4] + [0.86, 1.14, 0.90, 1.10])
    x_true = np.concatenate([_x0s[:9] * _mul, bias_true, [delay_true], [1.0]])
    _out = [(l, v, lo_, hi_) for l, v, lo_, hi_ in
            zip(param_labels(names, False)[:9], x_true[:9], _lo_s[:9], _hi_s[:9])
            if not (lo_ <= v <= hi_)]
    if _out:                       # 탐색범위가 또 좁아지면 여기서 바로 잡힌다
        raise SystemExit("✗ 셀프테스트 참값이 탐색범위 밖이다 — _mul 을 조정할 것: "
                         + " · ".join(f"{l} {v:.4g}∉[{a_:.4g},{b_:.4g}]"
                                      for l, v, a_, b_ in _out))
    apply_params(m, idx, gear_n, split_params(x_true, len(names), False)[0], False, names)
    win = max(1, int(round(a.window / dt)))
    # 참 궤적: 실측 자리에 시뮬을 넣고 창 재초기화 없이 한 번에 굴린다
    q0 = np.zeros((len(tt), 8)); dq0 = np.zeros_like(q0)
    # ★합성·적합 **둘 다** 실전 제어법칙으로 돈다. 여기만 옛 법칙이면 셀프테스트가
    #   production 경로를 안 지키게 된다(2026-08-14 foot PD 를 raw각으로 바꾸며 추가).
    wrap = actuator_wrap(m, idx, names, a.ctrl_space)
    q_true = rollout(m, d, idx, q0, dq0, q_cmd, kp, kd, dt, len(tt),
                     delay_s=delay_true, wrap=wrap)
    dq_true = np.vstack([np.zeros(8), np.diff(q_true, axis=0) / dt])
    rng = np.random.default_rng(0)
    # ★엔코더는 q_true + bias 를 읽는다 — 추정기는 이 bias 를 되찾아야 한다
    q_meas = q_true + bias_true + rng.normal(0, 0.02, q_true.shape)
    print(f"■ 셀프테스트 — 합성 {T:.0f}s · dt {dt*1000:.0f}ms · 측정잡음 0.02°")

    def ev(p):
        dyn, bias, dly, _cf = split_params(p, len(names), False)
        apply_params(m, idx, gear_n, dyn, False, names)
        qs = rollout(m, d, idx, q_meas, dq_true, q_cmd, kp, kd, dt, win,
                     bias=bias, delay_s=dly, wrap=wrap)
        return cost_of(qs, q_meas - bias)

    x0, lo, hi, _free = init_bounds(a.spec, names, False)
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
    # ★JDAMP 는 **합격 판정에서 뺀다** (2026-08-14).
    #   궤적데이터로 JDAMP 를 못 얻는 건 이 파일이 이미 문서화한 성질이다
    #   (JDAMP↔JFRIC r=+0.93 평탄방향 · design_excitation 이 설계 단계에서 짚음).
    #   그걸 합격 조건에 넣으면 셀프테스트가 **영구 실패**가 되고, 그러면 진짜 회귀를
    #   못 알아본다 — 참값을 탐색범위 밖에 두어 늘 실패하던 것과 같은 병이다.
    #   ⇒ 값은 찍되 게이트는 ROTOR_I·JFRIC·bias·delay 가 진다. JDAMP 는 **토크시험**
    #     (act_measure_inertia_torque 의 q̇_ref 훑기)에서 괄호로 받아 온다.
    ok = True
    for i, l in enumerate(lab):
        e = (best[i] / x_true[i] - 1) * 100
        good = abs(e) < 30
        gate = not l.startswith("JDAMP.")
        ok &= (good or not gate)
        print(f"  {l:<16}{x_true[i]:>12.4g}{best[i]:>12.4g}{e:>9.1f}%"
              + ("" if good else ("  ★" if gate else "  (판정제외 — 궤적으로는 못 얻는다)")))
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
    print(f"\n  RMS {bestc:.4f}° · 셀프테스트 "
          + ("통과" if ok else "실패(30% 초과 — JDAMP 는 판정에서 제외됨)"))
    if bestc > 0.05:
        print(f"  ⚠RMS {bestc:.4f}° 는 잡음바닥(0.02°)보다 한참 위다 — **미수렴**이다."
              f" 게이트로 쓰려면 `--st-T 4 --iters 120` 이상으로 돌릴 것")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
