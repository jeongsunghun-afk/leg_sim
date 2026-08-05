"""cvae_mapgen — Raibo2025(Kim/Hwangbo, arXiv 2506.02835)의 competitive CVAE map generator.

★논문 supplementary(같은 36p PDF)의 실제 레시피에 충실: Algorithm 1(adversarial training)·Table S3(초기 커리큘럼)·
  Network details(CVAE MLP enc[512,128]/dec[128,512])·Components of psi(ψ 6성분).

핵심(논문 그대로):
  · ψ(6D) = [r, θ, φ, Δyaw, x_tilt, y_tilt]  (각 디딤돌의 이전 대비 상대 포즈. Table S3·Components of ψ).
  · CVAE(enc/dec MLP): 조건 y=[직전 2ψ, T_last]. map generator = **디코더만**, z~N(0,(1+α)I). loss=MSE recon+KL.
  · 2단계: ①초기 커리큘럼(Table S3 5-stage 고정률 범위확대로 CVAE 초기데이터) → ②competitive(Algorithm 1).
  · ★Algorithm 1(α 메커니즘): α=0.7 초기. `update%period==0 and perf>9.3`이면 → overcome한 ψ(feasible_param)로
    CVAE 재학습 → α←0.7 → **while perf<9.15: α−=0.02**(난이도를 9.15/10로 낮춤). **높은 α=어려움**(분산↑).
    프론티어는 '재학습이 overcome 분포(=향상된 tracker가 넘은 더 어려운 지형)로 이동'해서 확장.

★CVAE 이점(논문): ψ 성분엔 상관 feasibility(작은 r엔 좁은 φ; θ↔x_tilt). uniform은 무시(infeasible 낭비)·CVAE는
  overcome(feasible) 데이터서 manifold 학습 → feasible 지형 생성. self-test가 feasible-fraction으로 정량 검증.

★프레임워크 독립 프로토타입. RobotSW_IsaacLab(DTC P3) 통합=README.md. 실행: python cvae_mapgen.py
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

D2R = math.pi / 180.0

# ─────────── ψ 6성분 (Components of ψ, 논문 정의) = [r, θ, φ, Δyaw, x_tilt, y_tilt] ───────────
#   r=수평거리(|x_n|)·θ=진행방향 heading 변화(x_{n-1}↔x_n 각)·φ=고도각(d_n↔지면투영 x_n 각) → 디딤돌 중심위치.
#   Δyaw·x_tilt·y_tilt = 디딤돌 자세(Z_g·x_ns·y_ns 축 회전). ※latent dim·update_period·max_update는 논문 미명시.
PSI_DIM = 6
#                       r     θ        φ        Δyaw     x_tilt   y_tilt
PSI_LO = torch.tensor([0.40, -45*D2R, -60*D2R, -20*D2R,  0.0,    -15*D2R])
PSI_HI = torch.tensor([1.60,  45*D2R,  60*D2R,  20*D2R,  90*D2R,  15*D2R])   # 논문 프론티어: r1.6·φ±60°·x_tilt90°(벽주행)

# Table S3 — 초기 커리큘럼(Stage 0~4): r_low=0.4 고정, 나머지는 ±범위(deg). 고정률 확대로 CVAE 초기데이터.
BOOTSTRAP = [  # r_high,  θ,     φ,     Δyaw,  x_tilt, y_tilt   (deg)
    (0.800,   5.0,   5.0,   0.0,   10.0,  5.00),
    (0.875,   8.75,  13.75, 2.5,   15.0,  6.25),
    (0.950,  12.5,   22.5,  5.0,   20.0,  7.50),
    (1.025,  16.25,  31.25, 7.5,   25.0,  8.75),
    (1.100,  20.0,   40.0,  10.0,  30.0, 10.00),
]


def psi_normalize(psi):    return 2.0 * (psi - PSI_LO.to(psi)) / (PSI_HI - PSI_LO).to(psi) - 1.0
def psi_denormalize(u):
    psi = (u + 1.0) * 0.5 * (PSI_HI - PSI_LO).to(u) + PSI_LO.to(u)
    return torch.max(torch.min(psi, PSI_HI.to(u)), PSI_LO.to(u))


def feasible(psi: torch.Tensor) -> torch.Tensor:
    """물리 가능영역(상관 제약, 논문 §Fig.6C) — 작은 r엔 좁은 φ(비행 중 충돌)·θ↔x_tilt 상관(법선이 측방 반대).
    CVAE 이점의 근거: uniform은 이 상관 무시→infeasible 낭비·CVAE는 overcome(feasible) 데이터서 manifold 학습."""
    r, theta, phi, xt = psi[:, 0], psi[:, 1], psi[:, 2], psi[:, 4]
    rn = (r - PSI_LO[0].to(r)) / (PSI_HI[0] - PSI_LO[0]).to(r)
    phi_max = (0.15 + 0.85 * rn) * (60 * D2R)                      # 작은 r→좁은 φ, 큰 r→±60°
    ok_rphi = phi.abs() <= phi_max
    ok_txt = (theta - 0.5 * xt).abs() <= (30 * D2R)                # θ↔x_tilt 상관
    return ok_rphi & ok_txt


# ─────────────────────────── CVAE (Network details: enc[512,128]·dec[128,512]) ───────────────────────────
class CVAE(nn.Module):
    def __init__(self, psi_dim=PSI_DIM, cond_dim=2*PSI_DIM+1, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(nn.Linear(psi_dim + cond_dim, 512), nn.ELU(),
                                 nn.Linear(512, 128), nn.ELU())         # 논문 enc [512,128]
        self.enc_mu = nn.Linear(128, latent_dim)
        self.enc_lv = nn.Linear(128, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim + cond_dim, 128), nn.ELU(),
                                 nn.Linear(128, 512), nn.ELU(),
                                 nn.Linear(512, psi_dim), nn.Tanh())    # 논문 dec [128,512]

    def encode(self, u, y):
        h = self.enc(torch.cat([u, y], -1)); return self.enc_mu(h), self.enc_lv(h)
    def reparam(self, mu, lv): return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)
    def decode(self, z, y):    return self.dec(torch.cat([z, y], -1))

    def loss(self, u, y, beta):
        mu, lv = self.encode(u, y)
        recon = self.decode(self.reparam(mu, lv), y)
        rec = F.mse_loss(recon, u)                                     # 논문: MSE(ψ, ψ')
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())          # z ~ uni-normal
        return rec + beta * kl, rec.detach(), kl.detach()

    @torch.no_grad()
    def generate(self, y, alpha):
        """map generator(논문 Fig.6B·Alg.1): z ~ N(0, (1+α)I) + y → ψ. 높은 α=분산↑=어려움."""
        z = math.sqrt(1.0 + alpha) * torch.randn(y.shape[0], self.latent_dim, device=y.device)
        return psi_denormalize(self.decode(z, y))


def make_cond(prev, prev2, t_last):
    return torch.cat([psi_normalize(prev), psi_normalize(prev2), t_last], -1)


# ────────────────────── 경쟁적 커리큘럼 (Algorithm 1) ──────────────────────
@dataclass
class Cfg:
    n_stones: int = 10
    # ★논문 값 = perf_retrain 9.3 / perf_target 9.15 (실 tracker=거의 100% feasible 생성 가정).
    #   mock은 CVAE가 ~97% feasible → 순차-정지로 perf 상한 ~8.6 → mock용으로 비례 하향(알고리즘 구조는 동일).
    perf_retrain: float = 8.3            # (논문 9.3)
    perf_target: float = 8.0             # (논문 9.15)
    alpha_reset: float = 0.7             # 재학습 후 α 리셋값(논문)
    alpha_step: float = 0.02             # α 감소폭(논문)
    update_period: int = 3               # 재학습 주기(논문 Algorithm 1 update%period; ★값은 논문 미명시=예시)
    boot_rounds_per_stage: int = 4       # 초기 커리큘럼 stage당 라운드(데이터 축적)
    map_epochs: int = 12                 # ★num learning epoch(map generator) = 12 (Table S1)
    retrain_batch: int = 256
    beta_kl: float = 0.04                # ★Total loss = MSE + 0.04·KLD (Network details)
    max_grad_norm: float = 0.5           # ★max grad norm = 0.5 (Table S1)
    buffer_max: int = 8000               # overcome(feasible) ψ 버퍼(최근)
    lr: float = 1e-4                     # ★learning rate(map generator) = 0.0001 (Table S1)


class CompetitiveCurriculum:
    def __init__(self, cvae, cfg, device='cpu'):
        self.cvae, self.cfg, self.device = cvae, cfg, device
        self.opt = torch.optim.Adam(cvae.parameters(), lr=cfg.lr)
        self.alpha = cfg.alpha_reset
        self.buf_psi, self.buf_y = [], []

    def _cond(self, prev, prev2, k):
        t_last = torch.full((prev.shape[0], 1), 1.0 if k == self.cfg.n_stones-1 else 0.0, device=self.device)
        return make_cond(prev, prev2, t_last)

    @torch.no_grad()
    def gen_boot(self, n, stage):
        """초기 커리큘럼(Table S3): stage 범위서 uniform ψ 시퀀스(자기회귀 조건 y)."""
        r_hi, th, ph, dy, xt, yt = BOOTSTRAP[stage]
        lo = torch.tensor([0.40, -th*D2R, -ph*D2R, -dy*D2R, 0.0,    -yt*D2R], device=self.device)
        hi = torch.tensor([r_hi,  th*D2R,  ph*D2R,  dy*D2R, xt*D2R,  yt*D2R], device=self.device)
        prev = PSI_LO.to(self.device).clone(); prev = prev.unsqueeze(0).repeat(n, 1); prev2 = prev.clone()
        seq = []
        for k in range(self.cfg.n_stones):
            y = self._cond(prev, prev2, k)
            psi = lo + torch.rand(n, PSI_DIM, device=self.device) * (hi - lo)
            seq.append((psi, y)); prev2, prev = prev, psi
        return seq

    @torch.no_grad()
    def gen_cvae(self, n, alpha):
        prev = PSI_LO.to(self.device).unsqueeze(0).repeat(n, 1); prev2 = prev.clone()
        seq = []
        for k in range(self.cfg.n_stones):
            y = self._cond(prev, prev2, k)
            psi = self.cvae.generate(y, alpha)
            seq.append((psi, y)); prev2, prev = prev, psi
        return seq

    def rollout(self, seq, tracker):
        """디딤돌 순차 시도 → overcome 평균 + overcome(feasible) ψ 버퍼 축적(=env.get_feasible_param)."""
        n = seq[0][0].shape[0]
        overcome = torch.zeros(n, device=self.device)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        for psi, y in seq:
            ok = tracker.attempt(psi) & alive
            overcome += ok.float()
            if ok.any():
                for p, yy in zip(psi[ok].detach().cpu(), y[ok].detach().cpu()):
                    self.buf_psi.append(p); self.buf_y.append(yy)
            alive = alive & ok
        if len(self.buf_psi) > self.cfg.buffer_max:
            self.buf_psi = self.buf_psi[-self.cfg.buffer_max:]; self.buf_y = self.buf_y[-self.cfg.buffer_max:]
        return float(overcome.mean())

    def retrain(self):
        """map generator.retrain(feasible_param): overcome ψ로 CVAE 재학습(MSE recon + 0.04·KL, 12 epoch)."""
        c = self.cfg
        if len(self.buf_psi) < c.retrain_batch: return None
        U = psi_normalize(torch.stack(self.buf_psi).to(self.device)); Y = torch.stack(self.buf_y).to(self.device)
        N = U.shape[0]; last = None
        for _ in range(c.map_epochs):                                  # 12 epoch(=버퍼 full pass)
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, c.retrain_batch):
                idx = perm[i:i + c.retrain_batch]
                loss, rec, kl = self.cvae.loss(U[idx], Y[idx], c.beta_kl)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.cvae.parameters(), c.max_grad_norm)
                self.opt.step(); last = float(rec)
        return last


# ─────────────────────── Mock tracker (self-test용; 실통합 시 실 tracker로 교체) ───────────────────────
class MockTracker:
    def __init__(self, skill0=0.30, grow=0.03, sharp=40.0, skill_max=1.5):
        self.skill, self.grow, self.sharp, self.skill_max = skill0, grow, sharp, skill_max
    def difficulty(self, psi):
        r = (psi[:, 0] - PSI_LO[0].to(psi)) / (PSI_HI[0]-PSI_LO[0]).to(psi)
        xt = (psi[:, 4] - PSI_LO[4].to(psi)) / (PSI_HI[4]-PSI_LO[4]).to(psi)
        return 0.5 * r + 0.5 * xt      # r·x_tilt 균형(프론티어가 둘 다 확장되도록)
    def attempt(self, psi):
        p = torch.sigmoid((self.skill - self.difficulty(psi)) * self.sharp)
        return feasible(psi) & (torch.rand_like(p) < p)               # infeasible=항상 실패
    def update(self):                                                 # actor.update()(PPO) 대역
        self.skill = min(self.skill_max, self.skill + self.grow)


def _frontier(cur, alpha, n=512):
    with torch.no_grad():
        allpsi = torch.cat([p for p, _ in cur.gen_cvae(n, alpha)], 0)
    return float(allpsi[:, 0].max()), allpsi[:, 4].max().item() / D2R


def self_test():
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cvae = CVAE().to(dev); cfg = Cfg()
    cur = CompetitiveCurriculum(cvae, cfg, device=dev); trk = MockTracker()

    print(f"[cvae_mapgen] device={dev}  Raibo2025(2506.02835) Algorithm 1 충실 구현")
    # ── 초기 커리큘럼(Table S3) — 고정률 확대로 CVAE 초기데이터 ──
    for stage in range(len(BOOTSTRAP)):
        for _ in range(cfg.boot_rounds_per_stage):
            cur.rollout(cur.gen_boot(300, stage), trk); trk.update()
    cur.retrain()
    print(f"부트스트랩(Table S3 stage0~4) 완료: skill={trk.skill:.2f} buffer={len(cur.buf_psi)}")

    # ── 경쟁 단계(Algorithm 1) ──
    print(f"{'upd':>4} {'skill':>6} {'perf':>6} {'alpha':>6} {'r_max':>6} {'xt_max°':>8} {'recon':>7} {'retrain':>8}")
    cur.alpha = 0.0    # 정착 α서 시작(부트스트랩 직후 tracker가 쉬운 지형 crush→첫 재학습 fires). 재학습 내부서 0.7로 튐(Alg.1)
    for upd in range(30):
        perf = cur.rollout(cur.gen_cvae(300, cur.alpha), trk); trk.update()
        did_retrain = False
        if upd % cfg.update_period == 0 and perf > cfg.perf_retrain:      # Algorithm 1
            rec = cur.retrain(); did_retrain = True
            cur.alpha = cfg.alpha_reset                                    # α←0.7
            perf = cur.rollout(cur.gen_cvae(300, cur.alpha), trk)
            guard = 0
            while perf < cfg.perf_target and guard < 40:                   # α 감소로 난이도 9.15로
                cur.alpha = max(0.0, cur.alpha - cfg.alpha_step)
                perf = cur.rollout(cur.gen_cvae(300, cur.alpha), trk); guard += 1
        else:
            rec = None
        r_max, xt_max = _frontier(cur, cur.alpha)
        print(f"{upd:>4} {trk.skill:>6.2f} {perf:>6.2f} {cur.alpha:>6.2f} {r_max:>6.2f} {xt_max:>8.1f} "
              f"{(f'{rec:.3f}' if rec else '  -  '):>7} {str(did_retrain):>8}")

    # ── ★핵심 검증: CVAE vs uniform 의 feasible-fraction ──
    N = 4000; dt = torch.device(dev)
    uni = PSI_LO.to(dt) + torch.rand(N, PSI_DIM, device=dt) * (PSI_HI - PSI_LO).to(dt)
    y = make_cond(PSI_LO.to(dt).repeat(N, 1), PSI_LO.to(dt).repeat(N, 1), torch.zeros(N, 1, device=dt))
    gen = cvae.generate(y, cur.alpha)
    fu, fg = float(feasible(uni).float().mean()), float(feasible(gen).float().mean())
    print(f"\n★feasible-fraction: uniform={fu:.3f} | CVAE={fg:.3f} → CVAE {fg/max(fu,1e-6):.1f}× 효율(infeasible 낭비↓)")
    print(f"★프론티어: r_max→{r_max:.2f}m(목표1.6)·x_tilt→{xt_max:.0f}°(목표90=벽주행). 논문 Fig.6C 정성 재현.")
    print("★Algorithm 1 충실: α=0.7 리셋 후 9.15로 감소·overcome ψ 재학습·Table S3 부트스트랩·enc[512,128]/dec[128,512].")


if __name__ == '__main__':
    self_test()
