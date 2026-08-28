#!/usr/bin/env python3
"""에어워크 — 1점 walk 궤적을 매달린 로봇에서 감속 재생(jog 스트리밍)하는 리허설 + 계측.

왜 (2026-08-28, 사용자 요청): walk 실기 전에 ①궤적·속도·트립 여유를 실물에서 확인하고
②sim-real 간극을 **숫자로** 얻는다. 0.25배속이면 관성항이 1/16 이라 기대토크 ≈ G(q)
(중력만) 로 떨어져, 준정적 캠페인(g*·push·r(G))과 같은 잣대의 **동적 대조**가 된다.

얻는 값 (--analyze):
  · 축별 추종오차 rms/max — 구동 건강/처짐
  · 이력폭(반전 시 오차 점프) — **백래쉬+컴플라이언스 정량화** (08-28 calf 발견의 수치화)
  · r̂ = G_model/τ_echo (부호별) — 전달비의 동적판. 캠페인 값(hip 0.84·thigh 0.8·
    calf 0.82·foot r(G)) 과 대조 = sim-real 갭
  · 채널속도 max → 1× 외삽 vs walk 트립(900dps) 실측 검증

절차:
  [노트북] --gen                          → biped/data/airwalk/traj_*.json (커밋→Pi pull)
  [Pi]     배포기를 JOG_SPEED_DPS=140 로 기동 (run_deploy_hw.sh 앞에 env)
  [Pi]     --play <traj.json>             → 같은 폴더에 log_*.json (즉시저장)
  [노트북] --analyze <traj.json> <log.json> → 표 + summary
안전: 크레인 매달림 전제 · 시작 매달림 자세 저장 → 종료 시 jog 서행 복귀 → off (v3 규약)
     · E-stop 래치 감시 · 재생 전 채널속도 사전검사(170dps 초과 시 거부).
"""
import os, sys, json, time, argparse, subprocess

NJ = 8
CMD = "/tmp/biped_cmd.json"
STT = "/tmp/biped_state.json"
BIPED = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTD = os.path.join(BIPED, "data", "airwalk")
NAMES = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot", "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]

# ── 명령/상태 (float_gstar 규약 재사용) ──────────────────────────────────────
_seq = [0]

def send(**kw):
    _seq[0] += 1
    c = {"v": 0.0, "vy": 0.0, "w": 0.0, "body_h": 0.38,
         "jog_deg": [0.0] * NJ, "pos_kp_scale": 1.0, "seq": _seq[0]}
    c.update(kw)
    tmp = "%s.%d.tmp" % (CMD, os.getpid())
    with open(tmp, "w") as f:
        json.dump(c, f)
    os.replace(tmp, CMD)

def state():
    try:
        return json.load(open(STT))
    except Exception:
        return {}

def estopped():
    st = state()
    for k in ("estop", "estop_latched"):
        if st.get(k):
            return st.get("estop_reason") or k
    return None

def q_now():
    v = state().get("q_leg_deg")
    return [float(x) for x in v[:NJ]] if isinstance(v, list) and len(v) >= NJ else None

def hold(mode, secs, hz=20, **kw):
    t0 = time.time(); last = None
    while time.time() - t0 < secs:
        send(mode=mode, **kw); time.sleep(1.0 / hz)
        last = q_now() or last
    return last

# ── 안전 종료 (float_gstar v3/멱등 규약) ─────────────────────────────────────
_QSTART = [None]; _SHUT = [False]

def safe_shutdown():
    if _SHUT[0] or _QSTART[0] is None:
        return
    _SHUT[0] = True
    qs = _QSTART[0]
    try:
        latched = False
        print("\n  ■ 안전 종료 — 시작 매달림 자세로 jog 서행 복귀 후 무여자.")
        t0 = time.time()
        while time.time() - t0 < 30.0:
            if estopped():
                latched = True; break
            cur = hold("jog", 0.5, jog_deg=list(qs))
            if cur and max(abs(x - y) for x, y in zip(cur, qs)) < 1.5:
                break
        if latched:
            print("  ⛔ E-stop 래치 — jog 무시됨. 로봇 위치를 눈으로 확인하고 배포기 재기동.")
        else:
            hold("jog", 1.5, jog_deg=list(qs))
        send(mode="off")
        print("  ✅ 안전 종료 완료(무여자).")
    except Exception as e:
        print(f"  ⚠종료 정리 실패({type(e).__name__}) — GUI 로 수동 off")

# ── 채널속도 (트립 잣대: 발목 = (calf+foot)×1.2 합산) ────────────────────────
def ch_speeds(dq8):
    out = []
    for leg in range(2):
        b = 4 * leg
        out += [abs(dq8[b]), abs(dq8[b + 1]), abs(dq8[b + 2]) * 1.5,
                abs(dq8[b + 2] + dq8[b + 3]) * 1.2]
    return out

# ═══ --gen: sim 롤아웃 → 궤적 (노트북 · mujoco 필요) ═══════════════════════
def gen(a):
    os.makedirs(OUTD, exist_ok=True)
    env = dict(os.environ, ALPHA_AXIS="0.85", FOOT_FRIC_EXTRA="0.36", FRIC_COMP="0",
               FOOT_COMP_NM="0", T_STEP="0.30")   # 실측 플랜트 · 상속 오염 차단
    code = f'''
import os, sys, json
sys.path.insert(0, {BIPED!r})
import numpy as np, mujoco
import biped_mpc_wbic as BM
c = BM.BipedMPCWBIC(mjcf=os.path.join({BIPED!r}, "biped_flatfoot.mjcf"))
c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
m, d = c.m, c.d; dt = m.opt.timestep
dec = max(1, int(round(0.02/dt)))                 # 50Hz 프레임
T0, DUR = {a.warmup}, {a.dur}
frames = []
for k in range(int((T0+DUR)/dt)+1):
    t = k*dt
    c.vx_cmd = {a.vx} if t > 2.0 else 0.0
    c.vy_cmd = c.wz_cmd = 0.0
    c.control(dt); mujoco.mj_step(m, d)
    if d.qpos[2] < 0.2: print("RESULT " + json.dumps(dict(error="낙상 t=%.2f"%t))); sys.exit()
    if t >= T0 and k % dec == 0:
        frames.append([round(float(x),4) for x in np.rad2deg(d.qpos[7:7+8])])
# ★프레임별 고정베이스 중력토크 — float 모드와 같은 계산(매달림 기대토크)
taus = []
for q in frames:
    d.qpos[0]=d.qpos[1]=0.0; d.qpos[2]=0.5
    d.qpos[3]=1.0; d.qpos[4]=d.qpos[5]=d.qpos[6]=0.0
    d.qpos[7:7+8] = np.deg2rad(q); d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    taus.append([round(float(x),4) for x in d.qfrc_bias[6:6+8]])
print("RESULT " + json.dumps(dict(frames=frames, tau_g=taus)))
'''
    print(f"■ sim 롤아웃 — 1점 walk vx={a.vx} · 정상부 {a.warmup}~{a.warmup+a.dur}s · 실측 플랜트")
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=1800)
    res = None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if not res or "error" in res:
        print("✗ 롤아웃 실패:", (res or {}).get("error") or (r.stderr or r.stdout)[-300:]); return 1
    fr = res["frames"]
    # 관절속도(사전검사용) — 중앙차분
    dqmax = [0.0] * NJ
    for i in range(1, len(fr) - 1):
        for j in range(NJ):
            dqmax[j] = max(dqmax[j], abs(fr[i + 1][j] - fr[i - 1][j]) / 0.04)
    path = os.path.join(OUTD, f"traj_vx{a.vx:g}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    json.dump(dict(meta=dict(vx=a.vx, t_step=0.30, hz=50, plant="alpha0.85+foot0.36",
                             warmup=a.warmup, dur=a.dur, dqmax_dps=dqmax),
                   frames=fr, tau_g=res["tau_g"]), open(path, "w"))
    print(f"  {len(fr)} 프레임 저장 → {path}")
    print("  관절속도 max[dps]: " + " ".join(f"{NAMES[j].split('_')[1][:2]}{v:.0f}" for j, v in enumerate(dqmax)))
    for sp in (0.25, 0.5):
        chm = max(max(ch_speeds([v * sp for v in dqmax[:4]] * 2)),
                  max(ch_speeds([v * sp for v in dqmax[4:]] * 2)))
        print(f"  배속 {sp}: 채널속도 max ≈ {chm:.0f} dps (트립 200 대비)")
    print("  → 커밋 후 Pi 에서 --play 로 재생")
    return 0

# ═══ retau: 기존 궤적의 기대토크를 다른(무게추) 모델로 재계산 (노트북) ═══════
def retau(a):
    """무게추 에어워크용 — frames 는 그대로 두고 tau_g 만 주어진 MJCF 로 재계산.

    양발 무게추 모델은 make_weighted_mjcf 2회 체인으로 만든다(커밋 금지 규약):
      make_weighted_mjcf --leg HL --mass-g 2080 --at toe --x -6.3 --y 0 --z 35.5 --base flat
      make_weighted_mjcf --leg HR (동일 인자) --base biped_flatfoot_wHL.mjcf
    (오프셋 = 발끝→발목 선상 36 mm — 08-27 캠페인 부착점)
    """
    import numpy as np, mujoco
    traj = json.load(open(a.traj))
    m = mujoco.MjModel.from_xml_path(a.mjcf); d = mujoco.MjData(m)
    taus = []
    for q in traj["frames"]:
        d.qpos[:] = 0; d.qpos[2] = 0.5; d.qpos[3] = 1.0
        d.qpos[7:7 + NJ] = np.deg2rad(q); d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        taus.append([round(float(x), 4) for x in d.qfrc_bias[6:6 + NJ]])
    traj["tau_g"] = taus
    traj["meta"]["tau_mjcf"] = os.path.basename(a.mjcf)
    out = a.traj.replace(".json", f"_{a.tag}.json")
    json.dump(traj, open(out, "w"))
    mx = [max(abs(t[j]) for t in taus) for j in range(NJ)]
    print(f"  tau_g 재계산({os.path.basename(a.mjcf)}) → {out}")
    print("  |G|max[Nm]: " + " ".join(f"{v:.2f}" for v in mx))
    return 0

# ═══ rhat: 전달비·마찰 동시적합 (노트북 · 정밀판) ═══════════════════════════
#   analyze 의 r̂ 은 **명령 자세**의 G 를 쓰고 kd 항이 섞여 있어 참고치였다.
#   여기서는 세 가지를 고친다(2026-08-28):
#     ①G 를 **실측 자세**에서 계산 — 매달림 PD 처짐이 5~13° 라 명령 자세 G 는 그만큼 틀리다
#     ②명령토크에서 **kd 항을 분리** — τ_ch = kp·err + kd·(0−dq̇) 중 뒤엣것은 중력과 무관.
#       채널→관절 환산: τ_kd_raw = kd_ch·gear_k²·dq̇_raw · 발목 커플링(raw_foot = foot+calf)
#       · τ_joint = drive_to_tau(τ_raw)  (calf 관절 = calf raw + foot raw)
#     ③비의 중앙값이 아니라 **동시적합** c = (G + τ_f·sign(q̇))/α  →  push_solve 와 같은 꼴.
#       미지수 둘(1/α, τ_f/α)이라 마찰과 전달비가 함께 나온다(벤치 τ_c 와 대조 가능).
KD_CH   = [6.0, 4.0, 3.5, 2.0] * 2
GEAR_K  = [1.0, 1.0, 1.5, 1.2] * 2

def _kd_joint(dq_dps):
    """관절속도[dps] → kd 항이 만드는 **관절토크**[Nm] (부호: 속도를 거스름)."""
    import numpy as np
    w = np.deg2rad(np.asarray(dq_dps, float))
    out = np.zeros(NJ)
    for leg in range(2):
        b = 4 * leg
        raw_h = KD_CH[b + 0] * GEAR_K[b + 0] ** 2 * w[b + 0]
        raw_t = KD_CH[b + 1] * GEAR_K[b + 1] ** 2 * w[b + 1]
        raw_c = KD_CH[b + 2] * GEAR_K[b + 2] ** 2 * w[b + 2]
        raw_f = KD_CH[b + 3] * GEAR_K[b + 3] ** 2 * (w[b + 2] + w[b + 3])   # 커플링 coef +1
        out[b + 0] = -raw_h; out[b + 1] = -raw_t
        out[b + 2] = -(raw_c + raw_f)          # drive_to_tau: calf 관절 = calf + foot
        out[b + 3] = -raw_f
    return out

def rhat(a):
    import numpy as np, mujoco
    log = json.load(open(a.log))
    m = mujoco.MjModel.from_xml_path(a.mjcf); d = mujoco.MjData(m)
    rows = [r for r in log["rows"] if r.get("q") and r.get("tau") and r.get("dq")]
    G = np.zeros((len(rows), NJ)); C = np.zeros((len(rows), NJ)); S = np.zeros((len(rows), NJ))
    DQ = np.zeros((len(rows), NJ))
    for i, r in enumerate(rows):
        d.qpos[:] = 0; d.qpos[2] = 0.5; d.qpos[3] = 1.0
        d.qpos[7:7 + NJ] = np.deg2rad(r["q"]); d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        G[i] = d.qfrc_bias[6:6 + NJ]
        C[i] = np.asarray(r["tau"], float) - _kd_joint(r["dq"])   # 중력 담당분만
        DQ[i] = r["dq"]
        S[i] = np.sign(r["dq"])
    W = np.deg2rad(DQ)
    AC = np.zeros_like(W)                      # 중앙차분 각가속도 [rad/s²]
    ts = np.array([r["t"] for r in rows])
    for i in range(1, len(rows) - 1):
        dt2 = ts[i + 1] - ts[i - 1]
        if 1e-3 < dt2 < 0.3:
            AC[i] = (W[i + 1] - W[i - 1]) / dt2
    print(f"■ 전달비·마찰 동시적합 — {len(rows)}점 · {os.path.basename(a.log)}")
    print(f"  모델 {os.path.basename(a.mjcf)} · c=(G+τ_f·sign(q̇))/α · |q̇|>{a.vmin}dps·|G|>{a.gmin}Nm")
    print(f"  {'축':9s}{'α̂':>7s}{'±':>6s}{'τ̂_f':>7s}{'b_v':>7s}{'관성':>8s}{'R²':>6s}{'n':>6s}{'α_qs':>8s}  벤치τ_c")
    BENCH = [0.724, 0.604, 0.871, 0.639] * 2
    out = {}
    for j in range(NJ):
        sel = (np.abs(DQ[:, j]) > a.vmin) & (np.abs(G[:, j]) > a.gmin)
        n = int(sel.sum())
        if n < 40:
            print(f"  {NAMES[j]:9s}{'—':>7s}{'':>6s}{'—':>7s}{'':>7s}{'':>8s}{'':>6s}{n:6d}"
                  f"{'—':>8s}   (신호 부족 · |G|max {np.abs(G[:,j]).max():.2f})")
            out[NAMES[j]] = None; continue
        # ★회귀항 (08-28 2차): 쿨롱마찰만으로는 **추종 지연 토크**가 sign 항에 섞여
        #   τ_f 가 벤치의 2~5배로 부풀고 α 가 1 을 넘는다(비물리). 지연은 속도에 비례하고
        #   가속 구간에는 관성항이 실리므로 넷을 함께 푼다:
        #     c·α = G + τ_f·sign(q̇) + b_v·q̇ + I·q̈
        A = np.column_stack([G[sel, j], S[sel, j], W[sel, j], AC[sel, j]]); y = C[sel, j]
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ sol
        r2 = 1.0 - res.var() / y.var() if y.var() > 0 else 0.0
        cov = np.linalg.pinv(A.T @ A) * (res @ res) / max(1, n - len(sol))
        a_hat, b_hat, e_hat, f_hat = sol
        alpha = 1.0 / a_hat if abs(a_hat) > 1e-9 else float("nan")
        da = np.sqrt(max(cov[0, 0], 0)) * alpha ** 2
        tf, bv, inr = b_hat * alpha, e_hat * alpha, f_hat * alpha
        # ★교차검증: 준정적(|q̇| 하위 구간)만으로 G/c 중앙값 — 회귀와 어긋나면 모델 부적합
        q_sel = sel & (np.abs(W[:, j]) < 0.08)
        qs = float(np.median(G[q_sel, j] / C[q_sel, j])) if q_sel.sum() > 30 else float("nan")
        out[NAMES[j]] = dict(alpha=round(float(alpha), 3), d_alpha=round(float(da), 3),
                             tau_f=round(float(tf), 3), visc=round(float(bv), 3),
                             inertia=round(float(inr), 4), r2=round(float(r2), 3),
                             n=n, alpha_qs=None if qs != qs else round(qs, 3))
        print(f"  {NAMES[j]:9s}{alpha:7.3f}{da:6.3f}{tf:7.3f}{bv:7.2f}{inr:8.3f}{r2:6.3f}{n:6d}"
              + (f"{qs:8.2f}" if qs == qs else f"{'—':>8s}") + f"   {BENCH[j]:.2f}")
    print("  판독: α̂ = 경로 전달비(캠페인 hip 0.84·thigh 0.8·calf 0.82) · τ̂_f = 조립 마찰(벤치 대조)")
    print("        R² 낮으면 그 축은 동적항/백래쉬가 커서 준정적 모델이 안 맞는다는 뜻")
    o = a.log.replace("log_", "rhat_")
    json.dump(out, open(o, "w"), indent=1, ensure_ascii=False)
    print(f"  → {o}")
    return 0

# ═══ --play: 스트리밍 재생 + 기록 (Pi · stdlib) ═══════════════════════════
def play(a):
    traj = json.load(open(a.traj))
    fr = traj["frames"]; hz = traj["meta"]["hz"]
    # ── 사전검사: 채널속도 (배속 반영) ──
    chmax = [0.0] * NJ
    for i in range(1, len(fr) - 1):
        dq = [(fr[i + 1][j] - fr[i - 1][j]) / (2.0 / hz) * a.speed for j in range(NJ)]
        cs = ch_speeds(dq)
        for j in range(NJ):
            chmax[j] = max(chmax[j], cs[j])
    jmax = max(traj["meta"]["dqmax_dps"]) * a.speed
    print(f"■ 사전검사 — 배속 {a.speed} · 프레임 {len(fr)} · 루프 {a.loop}")
    print("  채널속도 max[dps]: " + " ".join(f"{v:.0f}" for v in chmax) + "  (거부 한계 170)")
    print(f"  필요한 배포기 env: JOG_SPEED_DPS≥{jmax * 1.2:.0f} (지금 세션에 설정했는지 확인)")
    if max(chmax) > 170.0:
        print("✗ 채널속도가 트립 여유(170dps)를 넘는다 — --speed 를 낮출 것"); return 1
    st = state()
    if not st or st.get("q_leg_deg") is None:
        print("✗ 상태 파일이 없다/비었다 — 배포기가 떠 있나?"); return 1

    os.makedirs(OUTD, exist_ok=True)
    log_path = os.path.join(OUTD, f"log_{os.path.basename(a.traj).replace('traj_','').replace('.json','')}"
                                  f"_x{a.speed:g}_{time.strftime('%H%M%S')}.json")
    rows = []
    def _save():
        json.dump(dict(traj=os.path.basename(a.traj), speed=a.speed, rows=rows),
                  open(log_path, "w"))
    # ── GUI 경쟁 검사 (float_gstar 규약) — 같이 뜨면 20ms 마다 명령을 서로 덮는다 ──
    try:
        r = subprocess.run(["pgrep", "-af", "teleop_gui_biped"],
                           capture_output=True, text=True, timeout=3)
        others = [l for l in r.stdout.splitlines() if "pgrep" not in l]
    except Exception:
        others = []
    if others:
        print("✗ teleop GUI 가 떠 있다 — 명령이 20ms 마다 덮인다. GUI 를 닫고 재실행:")
        for l in others:
            print("    " + l[:100])
        return 1
    try:
        _QSTART[0] = q_now()
        if _QSTART[0] is None:
            print("✗ 관절각을 못 읽는다"); return 1
        # ★off 선발행(2026-08-28 실기 1차 실패 반영) — 두 마리를 한 번에 잡는다:
        #   ①E-stop 래치 해제 규약(float_gstar 와 동일) ②기동 잔류명령 잠금(boot_mode) 해소 —
        #   배포기 기동 시 명령파일에 "jog" 가 남아 있었으면 jog 가 영구 무시된다(:1036 규약).
        #   off 는 boot_mode 와 다른 모드라 잠금을 풀고, 그 다음 jog 가 정상 수신된다.
        send(mode="off"); time.sleep(0.4)
        hold("off", 0.6)
        print("  jog 진입(점프 방지: 현재각 시드) → 첫 프레임 정렬…")
        hold("jog", 1.0, jog_deg=list(_QSTART[0]))
        # ★모드 에코 검사 — 배포기가 정말 jog 에 들어갔는지 상태로 확인(자가진단)
        dm = state().get("mode")
        if dm != "jog":
            print(f"✗ 배포기 모드가 '{dm}' (jog 아님) — 명령이 안 먹힌다. 배포기 터미널에서")
            print("  '명령 잠금'/'jog 진입' 출력을 확인할 것. (GUI 경쟁·estop·워치독 잠금 순으로 의심)")
            return 1
        # ★정렬 판정 = **수렴**(2026-08-28 실기 2차 반영): 매달림에선 PD 처짐이 hip ~3.8°·
        #   thigh ~3.2° 로 남는 게 정상이다(문서값 hip 5.2Nm/kp100=3.0° 와 정합) — 고정 2° 는
        #   영원히 못 넘는 문턱이었다. 잔차가 2s 동안 0.3° 미만으로 안 변하면 정착으로 본다.
        #   절대 가드 8°: 그 이상 남으면 처짐이 아니라 고장(트립/무구동)이다.
        t0 = time.time()
        cur = None; hist = []
        ok = False
        while time.time() - t0 < 40.0:
            if estopped():
                print("⛔ E-stop 래치 — 중단"); return 1
            cur = hold("jog", 0.5, jog_deg=list(fr[0]))
            if not cur:
                continue
            emax = max(abs(x - y) for x, y in zip(cur, fr[0]))
            hist.append((time.time(), list(cur)))
            hist = [(t, v) for t, v in hist if time.time() - t <= 2.0]
            settled = (len(hist) >= 3 and time.time() - hist[0][0] > 1.5 and
                       max(abs(a2 - b2) for a2, b2 in zip(hist[0][1], hist[-1][1])) < 0.3)
            if emax < 2.0 or (settled and emax < 8.0):
                if emax >= 2.0:
                    print(f"  정착 판정 — 잔차 max {emax:.1f}° 는 매달림 PD 처짐(정상). 재생으로 간다.")
                ok = True; break
        if not ok:
            print("✗ 첫 프레임 정렬 실패(40s) — 축별 잔차(측정−목표):")
            if cur:
                for j in range(NJ):
                    e = cur[j] - fr[0][j]
                    print(f"    {NAMES[j]:9s} {cur[j]:+7.1f} → {fr[0][j]:+7.1f}  잔차 {e:+6.1f}°"
                          + ("  ⚠" if abs(e) >= 2.0 else ""))
            print(f"  배포기 모드: '{state().get('mode')}' · JOG_SPEED_DPS 로그·트립 여부를 확인할 것")
            return 1
        print(f"  재생 시작 — {len(fr)/hz/a.speed:.1f}s × {a.loop}회. Ctrl+C = 안전 종료.")
        for lp in range(a.loop):
            tstart = time.time()
            while True:
                tw = time.time() - tstart
                ti = tw * a.speed                     # 궤적 시간
                idx = ti * hz
                i0 = int(idx)
                if i0 >= len(fr) - 1:
                    break
                w = idx - i0
                tgt = [fr[i0][j] * (1 - w) + fr[i0 + 1][j] * w for j in range(NJ)]
                send(mode="jog", jog_deg=tgt)
                s = state()
                rows.append(dict(t=round(tw, 3), ti=round(ti, 3), tgt=[round(x, 3) for x in tgt],
                                 q=s.get("q_leg_deg"), dq=s.get("dq_leg_dps"),
                                 tau=s.get("tau_leg_nm")))
                if len(rows) % 100 == 0:
                    _save()
                es = estopped()
                if es:
                    print(f"\n⛔ E-stop ({es}) — 재생 중단, 수집 {len(rows)}점은 저장"); _save()
                    return 1
                time.sleep(0.02)               # 목표 인덱스가 벽시계 기반이라 드리프트 무해
            print(f"  루프 {lp + 1}/{a.loop} 완료 ({len(rows)}점)")
        _save()
        print(f"  ✅ 기록 → {log_path}  (노트북에서 --analyze)")
        return 0
    except KeyboardInterrupt:
        print(f"\n  사용자 중단 — 수집 {len(rows)}점 저장"); _save()
        return 130
    finally:
        safe_shutdown()

# ═══ --analyze: 궤적 vs 기록 → sim-real 갭 표 (노트북) ═══════════════════════
def analyze(a):
    traj = json.load(open(a.traj)); log = json.load(open(a.log))
    fr, tg, hz = traj["frames"], traj["tau_g"], traj["meta"]["hz"]
    sp = log["speed"]
    rows = [r for r in log["rows"] if r.get("q") and r.get("tau")]
    if len(rows) < 50:
        print(f"✗ 유효 표본 {len(rows)}점 — 부족"); return 1
    print(f"■ 에어워크 분석 — {len(rows)}점 · 배속 {sp} · 궤적 {log['traj']}")
    print(f"  {'축':9s}{'추종rms':>8s}{'max':>6s}{'이력폭°':>8s}{'r̂↑':>6s}{'r̂↓':>6s}{'|dq|max':>8s}{'1×외삽':>7s}")
    summary = {}
    camp = [0.84, 0.80, 0.82, None, 0.84, 0.80, 0.82, None]     # 캠페인 준정적 r (foot 은 r(G))
    for j in range(NJ):
        errs, taus_m, taus_g, vs, dqs = [], [], [], [], []
        for r in rows:
            i0 = min(int(r["ti"] * hz), len(fr) - 2)
            errs.append(r["tgt"][j] - r["q"][j])
            taus_m.append(r["tau"][j])
            taus_g.append(tg[i0][j])
            vs.append((fr[i0 + 1][j] - fr[i0][j]) * hz * sp)     # 목표속도 부호용
            dqs.append(abs(r["dq"][j]) if r.get("dq") else 0.0)
        n = len(errs)
        rms = (sum(e * e for e in errs) / n) ** 0.5
        emax = max(abs(e) for e in errs)
        up = [e for e, v in zip(errs, vs) if v > 3.0]
        dn = [e for e, v in zip(errs, vs) if v < -3.0]
        hyst = (sum(up) / len(up) - sum(dn) / len(dn)) if (len(up) > 10 and len(dn) > 10) else float("nan")
        # r̂ = G_model / τ_echo — |G| 신호 있는 표본만, 목표속도 부호별(마찰 ±)
        r_up = sorted(g / t for g, t, v in zip(taus_g, taus_m, vs)
                      if abs(g) > 0.8 and abs(t) > 0.2 and v > 3.0 and 0.05 < g / t < 3.0)
        r_dn = sorted(g / t for g, t, v in zip(taus_g, taus_m, vs)
                      if abs(g) > 0.8 and abs(t) > 0.2 and v < -3.0 and 0.05 < g / t < 3.0)
        med = lambda x: x[len(x) // 2] if x else float("nan")
        dmax = max(dqs) if dqs else 0.0
        summary[NAMES[j]] = dict(rms=round(rms, 2), emax=round(emax, 2),
                                 hyst_deg=None if hyst != hyst else round(hyst, 2),
                                 r_up=None if med(r_up) != med(r_up) else round(med(r_up), 3),
                                 r_dn=None if med(r_dn) != med(r_dn) else round(med(r_dn), 3),
                                 dq_max=round(dmax, 1), dq_1x=round(dmax / sp, 0))
        cmp_s = f" (캠페인 {camp[j]})" if camp[j] and r_up else ""
        print(f"  {NAMES[j]:9s}{rms:8.2f}{emax:6.1f}"
              + (f"{hyst:8.2f}" if hyst == hyst else f"{'—':>8s}")
              + (f"{med(r_up):6.2f}" if r_up else f"{'—':>6s}")
              + (f"{med(r_dn):6.2f}" if r_dn else f"{'—':>6s}")
              + f"{dmax:8.0f}{dmax / sp:7.0f}" + cmp_s)
    print("\n  판독: 이력폭° = 반전 시 오차 점프(백래쉬+컴플라이언스) — calf HL vs HR 대조가 핵심")
    print("        r̂↑/↓ 사이가 마찰 밴드, 평균이 전달비 — 캠페인(0.84/0.8/0.82) 과 대조")
    print("        1×외삽 dps 는 walk 트립 900 대비 여유 확인용")
    out = a.log.replace("log_", "summary_")
    json.dump(summary, open(out, "w"), indent=1)
    print(f"  요약 → {out}")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen", help="sim 롤아웃 → 궤적 (노트북)")
    g.add_argument("--vx", type=float, default=0.05)
    g.add_argument("--warmup", type=float, default=4.0)
    g.add_argument("--dur", type=float, default=3.6, help="정상부 기록 길이[s] (0.6s 주기 배수)")
    p = sub.add_parser("play", help="스트리밍 재생+기록 (Pi · 매달림)")
    p.add_argument("traj")
    p.add_argument("--speed", type=float, default=0.25)
    p.add_argument("--loop", type=int, default=2)
    n = sub.add_parser("analyze", help="궤적 vs 기록 → 갭 표 (노트북)")
    n.add_argument("traj"); n.add_argument("log")
    rh = sub.add_parser("rhat", help="전달비·마찰 동시적합 — 실측자세 G·kd 분리 (노트북)")
    rh.add_argument("log"); rh.add_argument("mjcf")
    rh.add_argument("--vmin", type=float, default=2.0, help="분류에 쓸 최소 관절속도[dps]")
    rh.add_argument("--gmin", type=float, default=0.8, help="최소 중력토크 신호[Nm]")
    t = sub.add_parser("retau", help="궤적 tau_g 를 다른(무게추) MJCF 로 재계산 (노트북)")
    t.add_argument("traj"); t.add_argument("mjcf")
    t.add_argument("--tag", default="w2080")
    a = ap.parse_args()
    try:
        rc = {"gen": gen, "play": play, "analyze": analyze, "retau": retau, "rhat": rhat}[a.cmd](a)
    except KeyboardInterrupt:
        print("\n⛔ 사용자 중단"); rc = 130
    finally:
        safe_shutdown()
    sys.exit(rc)
