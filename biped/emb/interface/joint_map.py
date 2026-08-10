"""interface/joint_map.py — biped 컨트롤러(8-DOF, rad) ↔ Gait 채널(10, deg) 변환.

규약:  q_ch_deg[ch] = sign·k · rad2deg(q_ctrl_rad[i]) + offset_deg    (다리 0~7)
       q_ctrl_rad[i] = deg2rad( (q_ch_deg[ch] − offset_deg) / (sign·k) )
속도는 오프셋 없이 sign·k 만.  ★토크만 반대로 **나눈다**(τ_ch = τ_관절·sign/k) —
  드라이버가 감속비를 7 로 착각하면 보고토크 = 실제관절토크/k 이기 때문이다.
허리 채널(8~9)은 hold_deg 로 고정.
sign·offset·gear_k·limit 는 config(biped_emb.yaml) = 실기 캘리브레이션 값.

⚠⚠ 알려진 미해결 — **드라이버 감속비 오설정**(2026-08-10):
  드라이버가 전 축을 7:1 로 가정해 보고하는 것으로 보인다(실제 calf 10.5 · foot 8.4).
  그래서 그 두 축은 보고각이 실제 관절각의 1.5 / 1.2 배다.
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

    # ── 컨트롤러(rad) → 채널(deg) ──────────────────────────────────────────
    def q_ctrl_to_ch(self, q_rad) -> np.ndarray:
        out = np.zeros(self.n_channel)
        out[self.ch] = self.sk * (np.asarray(q_rad) * R2D) + self.offset
        for c, h in zip(self.waist_ch, self.waist_hold):
            out[c] = h
        return out

    def dq_ctrl_to_ch(self, dq_rad) -> np.ndarray:
        out = np.zeros(self.n_channel)
        out[self.ch] = self.sk * (np.asarray(dq_rad) * R2D)
        return out

    def tau_ctrl_to_ch(self, tau_nm) -> np.ndarray:
        # ★토크는 k 로 **나눈다**(각도는 곱한다). 드라이버가 감속비를 7 로 착각하면
        #   보고 토크 = 실제 관절토크 / k 이므로, 관절토크 τ 를 내려면 τ/k 를 보내야 한다.
        out = np.zeros(self.n_channel)
        out[self.ch] = self.sign * np.asarray(tau_nm) / self.k
        return out

    # ── 채널(deg) → 컨트롤러(rad) ──────────────────────────────────────────
    def ch_to_q_ctrl(self, q_ch_deg) -> np.ndarray:
        q = np.asarray(q_ch_deg, float)
        return ((q[self.ch] - self.offset) / self.sk) * D2R

    def ch_to_dq_ctrl(self, dq_ch_dps) -> np.ndarray:
        dq = np.asarray(dq_ch_dps, float)
        return (dq[self.ch] / self.sk) * D2R

    def ch_to_tau_joint(self, tau_ch_nm) -> np.ndarray:
        """보고 토크(채널) → **실제 관절토크**[Nm]. 실제 = 보고 · k."""
        t = np.asarray(tau_ch_nm, float)
        return t[self.ch] * self.k / self.sign

    # ── 모델각[deg] ↔ 채널각[deg] ────────────────────────────────────────
    #   ★★단위 규약 (2026-08-10 확정):
    #     GUI·jog·home·hold·한계·상태발행·뷰어는 **전부 모델각[deg]** 으로 일한다.
    #     채널각(=드라이버 보고각)은 **SHM 경계에서만** 쓴다. 변환은 여기 두 함수뿐.
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
    #     ⚠k 는 **아직 실측이 아니라 감속비에서 나온 계산값**이다. 자세에 따라 배율이
    #       변하면 감속비가 아니라 링키지 문제이고 이 보정이 틀린 것이다.
    #       diag/chain_check.py --axis N --physical 로 **두 개 이상 자세에서** 확인할 것.
    #     ⚠근본 해결은 여전히 드라이버 설정이다(RGA). 고쳐지면 k 를 전부 1.0 으로 되돌리고
    #       offset 만 다시 잡으면 된다 — 그때 RobotTestGait 등 다른 도구와도 다시 일치한다.
    def ch_to_q_joint(self, q_ch_deg) -> np.ndarray:
        """채널각(보고각) → 모델각. 상태 읽기 경로."""
        q = np.asarray(q_ch_deg, float)[self.ch]
        return (q - self.offset) / self.sk

    def q_joint_to_ch(self, q_joint_deg) -> np.ndarray:
        """모델각 → 채널각. 명령 쓰기 경로. 허리는 hold_deg 로 고정."""
        out = np.zeros(self.n_channel)
        out[self.ch] = np.asarray(q_joint_deg, float) * self.sk + self.offset
        for c, h in zip(self.waist_ch, self.waist_hold):
            out[c] = h
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

    def clamp_ch(self, q_ch_deg) -> np.ndarray:
        out = np.asarray(q_ch_deg, float).copy()
        out[self.ch] = np.clip(out[self.ch], self.min_deg, self.max_deg)
        for c, h in zip(self.waist_ch, self.waist_hold):
            out[c] = h
        return out

    def clamp_jog(self, q_ch_deg) -> np.ndarray:
        out = np.asarray(q_ch_deg, float).copy()
        out[self.ch] = np.clip(out[self.ch], self.jog_min, self.jog_max)
        for c, h in zip(self.waist_ch, self.waist_hold):
            out[c] = h
        return out
