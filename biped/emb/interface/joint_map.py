"""interface/joint_map.py — biped 컨트롤러(8-DOF, rad) ↔ Gait 채널(10, deg) 변환.

규약(★2026-08-10 scale 추가):
       q_ch_deg[ch] = ( sign · rad2deg(q_ctrl_rad[i]) + offset_deg ) · scale
       q_ctrl_rad[i] = deg2rad( ( q_ch_deg[ch]/scale − offset_deg ) / sign )
속도/토크는 오프셋 없이 sign·scale 만.  허리 채널(8~9)은 hold_deg 로 고정.
sign·offset·scale·limit 는 config(biped_emb.yaml) = 실기 jog 캘리브레이션 값.

★scale 이 왜 필요한가 (2026-08-10 발견):
  드라이버는 **모터각을 감속비로 나눠서** 관절각으로 보고한다(2026-08-05 hip 검증:
  36° 명령 → 출력축 36° 회전. 모터축 기준이면 5.1° 였어야 함).
  그런데 그 감속비가 **전 축 7:1 로 고정**돼 있는 것으로 보인다. 실제 감속비는
  hip/thigh 7 · calf 10.5 · foot 8.4 이므로:
      보고각 = 실제 관절각 × (실제감속비 / 7)
      calf 1.50배 · foot 1.20배 · hip/thigh 1.00배
  → JOG 중 "calf 가 뷰어에서 실제보다 훨씬 많이 움직인다" 는 관찰이 이것이다.

  ⚠**뷰어만 고치면 안 된다.** 같은 배율이 명령 경로에도 걸린다 — +10° 를 보내면 실제
    관절은 6.7° 만 움직인다. 그래서 배율을 여기(매핑)에 넣어 **경계에서 한 번만** 적용한다.
    그러면 이 위(제어기·jog 한계·home 목표·뷰어·PACE)는 전부 **실제 관절각** 으로 통일된다.
  ⚠scale 은 **측정으로 확정할 것**(diag/scale_check.py). 확정 전 기본값 1.0 = 종전 동작.
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
        self.min_deg = np.array([j["min_deg"] for j in js], float)
        self.max_deg = np.array([j["max_deg"] for j in js], float)
        # ★scale: 보고각 / 실제 관절각. 미설정이면 1.0(종전 동작)
        self.scale   = np.array([float(j.get("scale", 1.0)) for j in js], float)
        if np.any(self.scale <= 0):
            raise ValueError("joints[].scale 은 0 보다 커야 한다")
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
        out[self.ch] = (self.sign * (np.asarray(q_rad) * R2D) + self.offset) * self.scale
        for c, h in zip(self.waist_ch, self.waist_hold):
            out[c] = h
        return out

    def dq_ctrl_to_ch(self, dq_rad) -> np.ndarray:
        out = np.zeros(self.n_channel)
        out[self.ch] = self.sign * (np.asarray(dq_rad) * R2D) * self.scale
        return out

    def tau_ctrl_to_ch(self, tau_nm) -> np.ndarray:
        # ⚠토크는 각도와 **반대로** 스케일된다. 드라이버가 감속비 7 을 가정하면
        #   관절토크 = 모터토크×실제감속비 인데 보고는 모터토크×7 이므로 보고 = 실제/scale.
        #   따라서 실제 관절토크 T 를 내려면 보고 단위로는 T/scale 을 써야 한다.
        out = np.zeros(self.n_channel)
        out[self.ch] = self.sign * np.asarray(tau_nm) / self.scale
        return out

    # ── 채널(deg) → 컨트롤러(rad) ──────────────────────────────────────────
    def ch_to_q_ctrl(self, q_ch_deg) -> np.ndarray:
        q = np.asarray(q_ch_deg, float)
        return ((q[self.ch] / self.scale - self.offset) / self.sign) * D2R

    def ch_to_dq_ctrl(self, dq_ch_dps) -> np.ndarray:
        dq = np.asarray(dq_ch_dps, float)
        return (dq[self.ch] / self.scale / self.sign) * D2R

    # ── 모델각[deg] ↔ 채널각[deg] ────────────────────────────────────────
    #   ★★단위 규약 (2026-08-10 확정):
    #     GUI·jog·home·hold·한계·상태발행·뷰어는 **전부 모델각[deg]** 으로 일한다.
    #     채널각(=드라이버 보고각)은 **SHM 경계에서만** 쓴다. 변환은 여기 두 함수뿐.
    #
    #       모델각 = (채널각 / scale − offset) / sign
    #       채널각 = (모델각 · sign + offset) · scale
    #
    #     sign  : 모터 + 방향이 모델좌표의 어느 방향인가 (모터부호는 8축 전부 + 로 검증됨)
    #     scale : 보고각 / 실제 관절각. 드라이버가 전 축을 7:1 로 가정하므로
    #             실제감속비/7 = hip·thigh 1.0 · calf 1.5 · foot 1.2
    #     offset: 기계적 0점 보정[모델각 deg]
    #
    #     ⚠순서 주의 — scale 을 먼저 풀고 offset·sign 을 적용한다. offset 은 **모델각 단위**다
    #       (사람이 각도기로 재는 값이므로 실제 관절각이어야 한다).
    def ch_to_q_joint(self, q_ch_deg) -> np.ndarray:
        """채널각(보고각) → 모델각. 상태 읽기 경로."""
        q = np.asarray(q_ch_deg, float)[self.ch]
        return (q / self.scale - self.offset) / self.sign

    def q_joint_to_ch(self, q_joint_deg) -> np.ndarray:
        """모델각 → 채널각. 명령 쓰기 경로. 허리는 hold_deg 로 고정."""
        out = np.zeros(self.n_channel)
        out[self.ch] = (np.asarray(q_joint_deg, float) * self.sign + self.offset) * self.scale
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
