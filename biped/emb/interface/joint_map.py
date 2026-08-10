"""interface/joint_map.py — biped 컨트롤러(8-DOF, rad) ↔ Gait 채널(10, deg) 변환.

★★용어 표준 (이 저장소 전체. emb/RL_INTERFACE.md §0 과 동일) — 각도는 **네 층**이다:
    모터축각 θ_m  : 실제 모터 샤프트 회전각 = 모델각 × 실제감속비. 직접 관측 불가.
    채널각   q_ch : 드라이버가 SHM 으로 주고받는 값 = θ_m/7 (드라이버가 전축 7:1 가정)
    raw각    q_raw: (q_ch − offset)/(sign·k). 감속비·부호·영점 제거, **커플링 미해제**
    모델각   q_joint: MJCF qpos. 커플링까지 해제. GUI·뷰어·정책이 쓰는 유일한 단위
  커플링이 없는 축은 raw각 == 모델각(foot 만 다르다).
  ⚠폐기: `보고각`→채널각, `joint 각도`→모델각.
  ⚠`관절각` 은 **각도 단위로 쓰지 말 것**(어느 층인지 모호). 관절토크·관절한계처럼
    물리량 수식어로만 쓴다. `모터각` 도 모호하므로 모터축각/채널각/raw각 중 하나로 쓴다.

규약:  q_ch_deg[ch] = sign·k · rad2deg(q_ctrl_rad[i]) + offset_deg    (다리 0~7)
       q_ctrl_rad[i] = deg2rad( (q_ch_deg[ch] − offset_deg) / (sign·k) )
속도는 오프셋 없이 sign·k 만.  ★토크만 반대로 **나눈다**(τ_ch = τ_관절·sign/k) —
  드라이버가 감속비를 7 로 착각하면 보고토크 = 실제관절토크/k 이기 때문이다.
허리 채널(8~9)은 hold_deg 로 고정.
sign·offset·gear_k·limit 는 config(biped_emb.yaml) = 실기 캘리브레이션 값.

⚠⚠ 알려진 미해결 — **드라이버 감속비 오설정**(2026-08-10):
  드라이버가 전 축을 7:1 로 가정해 보고하는 것으로 보인다(실제 calf 10.5 · foot 8.4).
  그래서 그 두 축은 채널각이 실제 관절각의 1.5 / 1.2 배다.
  ★2026-08-10 **소프트 보정 도입** — config 의 `gear_k`(= 실제감속비/7)로 나눈다.
    종전엔 "근본은 드라이버 설정이니 소프트로 가리지 말자" 로 넣지 않았는데, 영점을
    잡고 나니 offset 은 **기준자세 한 점**만 맞추고 기울기를 못 고친다는 게 드러났다
    (calf 실제 10° → 뷰어 15°). 뷰어·GUI 가 쓸 수 없는 상태라 보정을 넣는다.
    ⚠대가는 그대로다 — RobotTestGait·mot_test 등 **다른 도구는 여전히 보정이 없어**
      같은 축의 각도가 도구마다 다르다. 드라이버 설정이 고쳐지면 gear_k 를 전부 1.0 으로
      되돌리고 offset 만 다시 잡을 것(그때 다시 일치한다).
"""
from __future__ import annotations
import numpy as np

R2D = 180.0 / np.pi
D2R = np.pi / 180.0


class JointMap:
    def __init__(self, cfg: dict):
        js = cfg["joints"]
        self.names   = [j["name"] for j in js]
        self.n_leg   = len(js)                                   # 8
        self.ch      = np.array([j["channel"] for j in js], int)
        self.sign    = np.array([j["sign"] for j in js], float)
        self.offset  = np.array([j["offset_deg"] for j in js], float)
        # ★k = 드라이버 감속비 오설정 보정(실제감속비/7). 미지정이면 1.0 = 종전 동작.
        #   sk = sign·k 를 미리 만들어 둔다 — 두 값을 따로 곱하다 한쪽을 빠뜨리는 실수를 막는다.
        self.k       = np.array([float(j.get("gear_k", 1.0)) for j in js], float)
        if np.any(self.k <= 0):
            raise ValueError(f"gear_k 는 양수여야 한다: {self.k.tolist()}")
        self.sk      = self.sign * self.k
        self._couple(js)                       # 기구 커플링(calf→foot) 파싱. 아래 참조
        self.min_deg = np.array([j["min_deg"] for j in js], float)
        self.max_deg = np.array([j["max_deg"] for j in js], float)
        self.kp_leg  = np.array([j["kp"] for j in js], float)
        self.kd_leg  = np.array([j["kd"] for j in js], float)
        self.n_channel = int(cfg["shm"]["n_channel"])

        w = cfg.get("waist", {})
        self.waist_ch   = np.array(w.get("channels", []), int)
        self.waist_hold = np.array(w.get("hold_deg", []), float)
        self.waist_kp   = float(w.get("kp", 10.0))
        self.waist_kd   = float(w.get("kd", 0.5))

        jg = cfg.get("jog", {})
        self.jog_frac = float(jg.get("range_frac", 0.5))
        # jog 안전 한계 = 관절한계를 0쪽으로 range_frac 축소(중립 근처만 허용)
        self.jog_min = self.min_deg * self.jog_frac
        self.jog_max = self.max_deg * self.jog_frac
        # ★축별 예외 (2026-08-10). range_frac 은 "0 근처만 허용" 이라는 전제인데, 영점을
        #   잡고 나면 **정지 자세가 0 이 아닌 축**이 생긴다(calf 는 구조적 한계인 −55° 에
        #   서 있다). 그런 축까지 0.5배로 좁히면 JOG 에 들어가는 순간 한계까지 끌려간다
        #   (calf 는 27.5°). 그건 안전이 아니라 의도치 않은 동작이다.
        #   ⇒ 그런 축만 config 에서 jog_min_deg / jog_max_deg 로 직접 지정한다.
        for i, j in enumerate(js):
            if "jog_min_deg" in j:
                self.jog_min[i] = float(j["jog_min_deg"])
            if "jog_max_deg" in j:
                self.jog_max[i] = float(j["jog_max_deg"])
        bad = [(self.names[i], self.jog_min[i], self.jog_max[i], self.min_deg[i], self.max_deg[i])
               for i in range(self.n_leg)
               if self.jog_min[i] < self.min_deg[i] - 1e-9 or self.jog_max[i] > self.max_deg[i] + 1e-9
               or self.jog_min[i] > self.jog_max[i]]
        if bad:
            raise ValueError("jog 한계가 관절한계 밖이거나 뒤집혔다: "
                             + "; ".join(f"{n} jog[{a:+.1f},{b:+.1f}] vs limit[{c:+.1f},{d:+.1f}]"
                                         for n, a, b, c, d in bad))

        # ── 실장 여부 (meta.installed_channels) ────────────────────────────
        #   ★Emb 는 모터가 물리적으로 없어도 8채널 전부 connected=1·ucStatus=0 으로
        #     보고한다. 그래서 통신 보고만으로는 장착 여부를 알 수 없고, 미장착 축이
        #     `ok` 로 판정돼 왔다(GUI LED 초록). 여기서 사람이 선언한 값으로 가른다.
        #   미선언이면 전부 실장으로 간주 = 종전 동작 유지.
        inst = cfg.get("meta", {}).get("installed_channels")
        if inst is None:
            self.installed = np.ones(self.n_leg, bool)
        else:
            s = {int(c) for c in inst}
            self.installed = np.array([int(c) in s for c in self.ch], bool)

        self.n_saturated = 0
        self.last_saturated = []
        self._check_wrap()

    # ── ±180 래핑 가드 ────────────────────────────────────────────────────
    #   ★Emb 는 우리 명령을 ±180 으로 **래핑한다**(halGait.cpp:666-671):
    #       while(fPosition > 180) fPosition -= 360;
    #     그래서 채널각 232.8° 를 보내면 −127.2° 로 뒤집혀 **완전히 다른 자세로 튄다.**
    #     클램프가 아니라 래핑이라 "한계에서 멈춘다" 가 아니고 반대편으로 날아간다.
    #   ⚠커플링을 넣으면서 실제로 위험해졌다 — foot 명령이 calf 각도만큼 더해지기 때문에
    #     (raw_foot = φ_foot − φ_calf) 두 축 한계의 **조합**에서 ±180 을 넘는다.
    #   ⇒ jog 박스(운전자가 실제로 쓰는 범위)는 **기동 시 예외로 막고**,
    #     관절한계 박스는 도달 불가 조합일 수 있으므로 경고만 한다.
    WRAP_DEG = 180.0

    def _ch_extremes(self, lo, hi):
        """[lo,hi] 박스의 꼭짓점을 훑어 축별 채널각 최대 절대값을 구한다.
        커플링 때문에 축별 독립 최악값이 아니라 **조합**을 봐야 한다(소스축과 함께 움직임)."""
        out = np.zeros(self.n_leg)
        pair = {d: s for d, s in zip(self.cpl_dst, self.cpl_src)}
        for i in range(self.n_leg):
            cand = []
            for a in (lo[i], hi[i]):
                if i in pair:
                    s = pair[i]
                    for b in (lo[s], hi[s]):
                        q = (lo + hi) / 2.0
                        q[i] = a; q[s] = b
                        cand.append(self.q_joint_to_ch(q)[self.ch[i]])
                else:
                    q = (lo + hi) / 2.0
                    q[i] = a
                    cand.append(self.q_joint_to_ch(q)[self.ch[i]])
            out[i] = max(abs(v) for v in cand)
        return out

    def _check_wrap(self):
        # ★예외가 아니라 경고다(2026-08-10 정정). q_joint_to_ch 가 ±180 으로 포화시키므로
        #   래핑 사고는 이미 막혀 있다. 그리고 한계의 **모양**이 박스가 아니다 — 제약은
        #   발목 raw각(q_raw = foot + coef·calf)에 걸리므로, 박스 꼭짓점 하나가 넘는다고
        #   foot 단독 한계를 좁히면 무릎이 펴져 있을 때 쓸 범위까지 불필요하게 잘린다.
        jog = self._ch_extremes(self.jog_min, self.jog_max)
        self.jog_wrap_warn = [(self.names[i], jog[i]) for i in range(self.n_leg)
                              if jog[i] > self.WRAP_DEG]
        lim = self._ch_extremes(self.min_deg, self.max_deg)
        self.wrap_warn = [(self.names[i], lim[i]) for i in range(self.n_leg)
                          if lim[i] > self.WRAP_DEG]

    # ── 컨트롤러(rad) → 채널(deg) ──────────────────────────────────────────
    #   ★deg 경로(q_joint_to_ch)에 **위임**한다. 종전엔 같은 식을 두 벌 썼는데, 커플링처럼
    #     규약이 늘어날 때마다 한쪽을 빠뜨릴 위험이 커진다(실제로 GUI 복제본에서 당했다).
    def q_ctrl_to_ch(self, q_rad) -> np.ndarray:
        return self.q_joint_to_ch(np.asarray(q_rad, float) * R2D)

    def dq_ctrl_to_ch(self, dq_rad) -> np.ndarray:
        # 속도도 위치와 **같은 선형관계**를 따른다(상수항 offset 만 빠진다).
        dq = np.asarray(dq_rad, float) * R2D
        raw = dq.copy()
        for d, s, c in zip(self.cpl_dst, self.cpl_src, self.cpl_coef):
            raw[d] = dq[d] + c * dq[s]
        out = np.zeros(self.n_channel)
        out[self.ch] = self.sk * raw
        return out

    def tau_ctrl_to_ch(self, tau_nm) -> np.ndarray:
        """관절토크[Nm] → 채널 토크.

        ★두 가지가 각도와 **다르게** 들어간다:
          (1) k 로 **나눈다**(각도는 곱한다). 드라이버가 감속비를 7 로 착각하면
              보고토크 = 실제관절토크/k 이므로, τ 를 내려면 τ/k 를 보낸다.
          (2) 커플링은 **전치(transpose)** 로 들어간다. q_true=(I−C)·q_raw 이면
              일률 보존에서 τ_raw=(I−C)ᵀ·τ_true → 소스축(calf)이 목적축(foot) 토크를
              **떠안는다**: τ_raw_calf = τ_calf − c·τ_foot.
              (각도처럼 foot 쪽에 더하면 틀린다 — 방향이 반대다)
        """
        t = np.asarray(tau_nm, float)
        raw = t.copy()
        for d, s, c in zip(self.cpl_dst, self.cpl_src, self.cpl_coef):
            raw[s] = raw[s] - c * t[d]
        out = np.zeros(self.n_channel)
        out[self.ch] = self.sign * raw / self.k
        return out

    # ── 채널(deg) → 컨트롤러(rad) ──────────────────────────────────────────
    def ch_to_q_ctrl(self, q_ch_deg) -> np.ndarray:
        return self.ch_to_q_joint(q_ch_deg) * D2R

    def ch_to_dq_ctrl(self, dq_ch_dps) -> np.ndarray:
        dq = np.asarray(dq_ch_dps, float)
        raw = dq[self.ch] / self.sk
        out = raw.copy()
        for d, s, c in zip(self.cpl_dst, self.cpl_src, self.cpl_coef):
            out[d] = raw[d] - c * raw[s]
        return out * D2R

    def ch_to_tau_joint(self, tau_ch_nm) -> np.ndarray:
        """보고 토크(채널) → **실제 관절토크**[Nm]. 실제 = 보고·k, 커플링은 (I+C)ᵀ."""
        t = np.asarray(tau_ch_nm, float)
        raw = t[self.ch] * self.k / self.sign
        out = raw.copy()
        for d, s, c in zip(self.cpl_dst, self.cpl_src, self.cpl_coef):
            out[s] = raw[s] + c * raw[d]
        return out

    # ── 모델각[deg] ↔ 채널각[deg] ────────────────────────────────────────
    #   ★★단위 규약 (2026-08-10 확정):
    #     GUI·jog·home·hold·한계·상태발행·뷰어는 **전부 모델각[deg]** 으로 일한다.
    #     채널각(= 드라이버가 보고/수신하는 값)은 **SHM 경계에서만** 쓴다. 변환은 여기 두 함수뿐.
    #
    #       모델각 = (채널각 − offset) / (sign · k)
    #       채널각 =  모델각 · sign · k + offset
    #
    #     sign  : 모터 + 방향이 모델좌표의 어느 방향인가 (모터부호는 8축 전부 + 로 검증됨)
    #     offset: 기계적 0점 보정[deg] — **채널각 단위**(sign·k 로 나누기 전에 뺀다)
    #     k     : 드라이버 감속비 오설정 보정 = 실제감속비/드라이버가정(7).
    #             hip·thigh 1.0 · calf 10.5/7=1.5 · foot 8.4/7=1.2
    #
    #     ★k 를 넣기 전에는 offset 하나로 **한 점**(기준자세)만 맞았고 기울기가 틀렸다.
    #       calf 는 실제 10° 움직이면 15° 로, foot 은 12° 로 표시됐다. offset 은 위치를
    #       맞출 뿐 배율을 못 고친다 — 그래서 영점을 잡고 나서야 이 오차가 드러났다.
    #     ★2026-08-10 **실기 검증됨** — k 적용 후 뷰어와 실기의 회전비율이 일치함을
    #       육안 확인(사용자). 처음엔 감속비에서 역산한 값이었으나 이제 근거가 있다.
    #       ★★이 관찰은 **링키지 가설을 배제**한다: 4절 링키지라면 배율이 자세마다
    #         달라져 **상수** k 로는 전 구간이 맞을 수 없다. 상수 하나로 맞았다는 것이
    #         곧 "드라이버가 고정 7:1 로 나눈다" 의 서명이다.
    #       ⚠남은 정밀도: 각도기 실측(diag/chain_check.py --axis N --physical)은 아직이라
    #         1.50/1.20 의 소수점 아래는 미확정이다. 큰 각도 동작에서 재확인할 것.
    #     ⚠근본 해결은 여전히 드라이버 설정이다(RGA). 고쳐지면 k 를 전부 1.0 으로 되돌리고
    #       offset 만 다시 잡으면 된다 — 그때 RobotTestGait 등 다른 도구와도 다시 일치한다.
    # ── ★기구 커플링 (2026-08-10 발견) ───────────────────────────────────
    #   calf 를 움직이면 foot 이 종속적으로 따라 움직인다(발목이 링키지 구동).
    #   MJCF 는 4관절을 **독립**으로 모델링하므로 여기서 풀어 준다.
    #
    #       모델각_foot = raw_foot − coef · 모델각_calf
    #       raw_foot    = 모델각_foot + coef · 모델각_calf
    #     (raw_x = (채널각_x − offset_x)/(sign_x·k_x) — 커플링 풀기 **전** 값)
    #
    #   ★★coef 는 **raw각·모델각 공간(= 감속비 이후)** 의 계수다. 채널각 공간이 아니다.
    #     raw 가 이미 sign·k 로 나눠진 값이므로 자동으로 그렇게 된다.
    #     의미: "calf 관절이 30° 돌면 foot 관절도 30° 돈다" = coef 크기 1
    #           (calf 10.5:1, foot 8.4:1 로 감속비가 달라도 **관절각**끼리 1:1)
    #     ⇒ 그 결과 **채널각** 비율은 1:1 이 **아니다**: k_foot/k_calf = 1.2/1.5 = 0.8.
    #       실측 확인 — calf 관절 +30° 명령 시 채널각은 HL calf−45/foot−36 ·
    #       HR calf+45/foot+36. **다리 안에서 두 축 부호는 같다**(sign 이 같아서).
    #     ⚠여기를 채널각 공간으로 착각해 1:1 로 넣으면(Δch_foot = Δch_calf) raw 보상이
    #       30° 가 아니라 37.5° 가 된다 — k_calf/k_foot = 1.25 배 과보상.
    #       감속비가 고쳐져 k 가 전부 1.0 이 되면 두 공간이 같아져 이 구분이 사라진다 —
    #       그때 무심코 옮겨 적지 말 것.
    #
    #   ★삼각(triangular) 구조라 역변환이 닫힌 형태로 존재한다: 소스(calf)가 스스로
    #     커플링되지 않으므로 calf 를 먼저 풀고 그 값을 foot 에 쓰면 된다.
    #
    #   ★★2026-08-10 **실기 확정** — coef = +1 로 커플링이 정확히 상쇄됨을 실기에서 확인.
    #     (부호를 −1 로 잘못 넣었을 때는 보상이 상쇄가 아니라 **가중**되어 발목이 무릎을 더
    #      따라갔다. 뒤집으니 일치.)
    #     ⚠판정 근거는 **실기 물리 관측뿐**이다. JOG 중 엔코더 회귀는 쓸 수 없다 —
    #       발목 모터가 우리 보상 명령을 그대로 추종하므로 자기 명령을 되재게 된다
    #       (실제로 그 함정에 빠져 "부호 −1 이 맞다" 고 잘못 보고했다). 기구 자체를 재려면
    #       off(무여자)로 두고 무릎을 손으로 돌릴 것 — diag/couple_check.py.
    #   ⚠소스가 또 커플링된 축이면(체인) 이 구현은 틀린다 — 지금은 금지하고 검출한다.
    def _couple(self, cfg_js):
        self.cpl_dst, self.cpl_src, self.cpl_coef = [], [], []
        for i, j in enumerate(cfg_js):
            src = j.get("couple_from")
            if not src:
                continue
            if src not in self.names:
                raise ValueError(f"{j['name']}.couple_from='{src}' 를 joints 에서 못 찾았다")
            s = self.names.index(src)
            if cfg_js[s].get("couple_from"):
                raise ValueError(f"커플링 체인은 지원하지 않는다: {j['name']} ← {src} ← ...")
            if s == i:
                raise ValueError(f"{j['name']} 이 자기 자신에 커플링됐다")
            self.cpl_dst.append(i); self.cpl_src.append(s)
            self.cpl_coef.append(float(j.get("couple_coef", 1.0)))
        self.has_couple = bool(self.cpl_dst)

    def ch_to_q_joint(self, q_ch_deg) -> np.ndarray:
        """채널각 → 모델각. 상태 읽기 경로. 커플링을 **푼다**."""
        q = np.asarray(q_ch_deg, float)[self.ch]
        raw = (q - self.offset) / self.sk
        if not self.has_couple:
            return raw
        out = raw.copy()
        for d, s, c in zip(self.cpl_dst, self.cpl_src, self.cpl_coef):
            out[d] = raw[d] - c * raw[s]        # 소스는 커플링 없음 → raw[s] = 모델각
        return out

    def q_joint_to_ch(self, q_joint_deg) -> np.ndarray:
        """모델각 → 채널각. 명령 쓰기 경로. 커플링을 **되먹인다**. 허리는 hold_deg 고정.

        ★마지막에 채널각을 ±180 으로 **포화(saturate)** 시킨다.
          Emb 는 초과분을 클램프가 아니라 **래핑**하므로(halGait.cpp:666-671) 181° 가
          −179° 로 뒤집혀 반대편으로 날아간다. 포화는 "한계에서 멈춤" 이라 훨씬 안전하다.
        ★한계의 **모양**이 중요하다: 물리적 제약은 발목 **raw각**(q_raw = foot + coef·calf)에
          걸린다. foot 단독 한계로 좁히면 무릎이 펴져 있을 때 쓸 수 있는 범위까지 잘린다.
          그래서 모델각 한계는 그대로 두고 여기 raw/채널 공간에서만 막는다.
        ⚠포화가 실제로 걸리면 명령이 조용히 왜곡되므로 횟수를 세어 앱이 알릴 수 있게 한다.
        """
        qj = np.asarray(q_joint_deg, float)
        raw = qj.copy()
        for d, s, c in zip(self.cpl_dst, self.cpl_src, self.cpl_coef):
            raw[d] = qj[d] + c * qj[s]
        out = np.zeros(self.n_channel)
        v = raw * self.sk + self.offset
        n_sat = int(np.count_nonzero(np.abs(v) > self.WRAP_DEG))
        if n_sat:
            self.n_saturated += n_sat
            self.last_saturated = [self.names[i] for i in range(self.n_leg)
                                   if abs(v[i]) > self.WRAP_DEG]
            v = np.clip(v, -self.WRAP_DEG, self.WRAP_DEG)
        out[self.ch] = v
        for c_, h in zip(self.waist_ch, self.waist_hold):
            out[c_] = h
        return out

    # ── 한계 클램프 (전부 **모델각** 입력/출력) ───────────────────────────
    #   ★min_deg/max_deg 는 MJCF range 에서 온 **모델각** 한계다. 종전엔 채널각에
    #     그대로 걸어서 sign=−1 축의 허용범위가 거울처럼 뒤집혔다
    #     (HR_thigh 는 물리한계를 2.5° 넘었다). 모델각에 걸면 그 문제가 사라진다.
    def clamp_joint(self, q_joint_deg) -> np.ndarray:
        return np.clip(np.asarray(q_joint_deg, float), self.min_deg, self.max_deg)

    def clamp_jog_joint(self, q_joint_deg) -> np.ndarray:
        return np.clip(np.asarray(q_joint_deg, float), self.jog_min, self.jog_max)

    # ── 게인·한계 (채널 배열) ──────────────────────────────────────────────
    def kp_ch(self, leg_scale=1.0) -> np.ndarray:
        out = np.zeros(self.n_channel)
        out[self.ch] = self.kp_leg * leg_scale
        for c in self.waist_ch: out[c] = self.waist_kp
        return out

    def kd_ch(self, leg_scale=1.0) -> np.ndarray:
        out = np.zeros(self.n_channel)
        out[self.ch] = self.kd_leg * leg_scale
        for c in self.waist_ch: out[c] = self.waist_kd
        return out

    # ★clamp_ch / clamp_jog 삭제(2026-08-10) — **모델각 한계를 채널각에 거는**
    #   함수였다. 호출자는 이미 0 이었지만(전수 grep) 남겨 두면 같은 오용이
    #   재생산된다. 실제로 C++ 미러가 그걸 쓰다가 hold 진입 21.4° 점프를 만들었다.
    #   클램프는 반드시 **모델각 공간**에서: clamp_joint / clamp_jog_joint.
