"""cvae_mapgen — Raibo2025(Kim/Hwangbo, arXiv 2506.02835)의 competitive CVAE map generator 독립 프로토타입.

논문 요지(Terrain generation §, Fig.2/6):
  · ψ(6D) = 각 디딤돌의 이전 대비 상대 포즈 [r, φ, θ, x_tilt, y_tilt, h]. (정확 6성분=supplementary "Components of psi")
  · CVAE(encoder/decoder MLP) 가 ψ 분포를 표현. 조건 y = 직전 2개 ψ + T_last(다음이 마지막 타깃인지).
  · map generator = **디코더만** 사용: z~N(0, α·I) 샘플 + y → ψ 생성. α = 난이도(분산) 조절 knob.
  · 경쟁적(adversarial) 커리큘럼: tracker가 성공(10개 중 >9.3개)하면 CVAE를 tracker가 성공한 (ψ,y)로 재학습 →
    "성공 가능 지형 분포"를 학습 → α를 키워 프론티어를 물리적 가능영역 안에서 확장(r 0.4→1.6m·tilt→90° 벽주행).
  · 손실 = reconstruction + KL.

★이 파일은 프레임워크 독립(순수 PyTorch) 프로토타입. RobotSW_IsaacLab(DTC P3) 통합 시:
  - MockTracker → 실제 tracker 성공신호(에피소드당 넘은 디딤돌 수).
  - decode된 ψ → 실제 지형 빌더(stepping-stone 배치)로 변환(ψ→월드 포즈).
  - 자세한 통합 인터페이스=README.md.

실행(torch 환경, 예: GPU 서버 isaac-5.1): python cvae_mapgen.py
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── ψ 규약 ───────────────────────────
# 6D ψ = [r, phi, theta, x_tilt, y_tilt, h].  물리 가능영역(대략, 논문 Fig.6C + 우리 지형).
#   r      : 다음 디딤돌까지 거리 [m]      (0.30 ~ 1.60)
#   phi    : 방위각(측방) [rad]            (-60° ~ 60°)
#   theta  : 진행방향 경사(피치) [rad]     (-45° ~ 45°)
#   x_tilt : 표면법선 종tilt [rad]         (0 ~ 90° 벽주행)
#   y_tilt : 표면법선 횡tilt [rad]         (-45° ~ 45°)
#   h      : 높이차 [m]                    (-0.20 ~ 0.40)
PSI_DIM = 6
PSI_LO = torch.tensor([0.30, -math.pi/3, -math.pi/4, 0.0,        -math.pi/4, -0.20])
PSI_HI = torch.tensor([1.60,  math.pi/3,  math.pi/4, math.pi/2,   math.pi/4,  0.40])


def psi_normalize(psi: torch.Tensor) -> torch.Tensor:
    """물리 ψ → [-1,1] (신경망 학습 안정화)."""
    return 2.0 * (psi - PSI_LO.to(psi)) / (PSI_HI.to(psi) - PSI_LO.to(psi)) - 1.0


def psi_denormalize(u: torch.Tensor) -> torch.Tensor:
    """[-1,1] → 물리 ψ (물리 가능영역으로 clamp = '물리적 가능영역 포착')."""
    psi = (u + 1.0) * 0.5 * (PSI_HI.to(u) - PSI_LO.to(u)) + PSI_LO.to(u)
    return torch.max(torch.min(psi, PSI_HI.to(u)), PSI_LO.to(u))


# ─────────────────────────── CVAE ───────────────────────────
class CVAE(nn.Module):
    """조건부 VAE. 조건 y = [prev ψ, prev2 ψ, T_last] (정규화된 ψ 사용). encoder/decoder = MLP."""

    def __init__(self, psi_dim=PSI_DIM, cond_dim=2*PSI_DIM+1, latent_dim=8, hidden=128):
        super().__init__()
        self.psi_dim, self.cond_dim, self.latent_dim = psi_dim, cond_dim, latent_dim
        # encoder: [ψ(정규화) ; y] → (μ, logvar)
        self.enc = nn.Sequential(nn.Linear(psi_dim + cond_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU())
        self.enc_mu = nn.Linear(hidden, latent_dim)
        self.enc_lv = nn.Linear(hidden, latent_dim)
        # decoder(=map generator): [z ; y] → ψ(정규화, tanh로 [-1,1])
        self.dec = nn.Sequential(nn.Linear(latent_dim + cond_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, psi_dim), nn.Tanh())

    def encode(self, u_psi, y):
        h = self.enc(torch.cat([u_psi, y], dim=-1))
        return self.enc_mu(h), self.enc_lv(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y):
        return self.dec(torch.cat([z, y], dim=-1))   # 정규화 ψ([-1,1])

    def forward(self, u_psi, y):
        mu, lv = self.encode(u_psi, y)
        z = self.reparam(mu, lv)
        return self.decode(z, y), mu, lv

    def loss(self, u_psi, y, beta=1.0):
        recon, mu, lv = self.forward(u_psi, y)
        rec = F.mse_loss(recon, u_psi, reduction='mean')
        kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        return rec + beta * kl, rec.detach(), kl.detach()

    @torch.no_grad()
    def generate(self, y, alpha=1.0):
        """map generator: z~N(0, α·I) 샘플 + 조건 y → 물리 ψ. α=난이도(분산)."""
        z = alpha * torch.randn(y.shape[0], self.latent_dim, device=y.device)
        return psi_denormalize(self.decode(z, y))


def make_cond(prev_psi: torch.Tensor, prev2_psi: torch.Tensor, t_last: torch.Tensor) -> torch.Tensor:
    """조건 y 구성: 직전 2개 ψ(정규화) + T_last(0/1). prev/prev2 = 물리 ψ."""
    return torch.cat([psi_normalize(prev_psi), psi_normalize(prev2_psi), t_last], dim=-1)


# ────────────────────── 경쟁적 커리큘럼 루프 ──────────────────────
@dataclass
class CurriculumCfg:
    success_thresh: float = 9.3          # 10개 중 넘은 디딤돌 평균 > 이 값 → 난이도 확장(논문 9.3/10)
    n_stones: int = 10                   # 에피소드당 디딤돌 수
    # Stage 1(부트스트랩): 물리영역의 frac 비율 안에서 uniform ψ. 성공하면 frac 확대(고정률). CVAE 초기 데이터 축적.
    frac0: float = 0.03                  # 초기 난이도 범위(쉬움)
    frac_grow: float = 1.15              # 마스터마다 frac ×
    stage2_buffer: int = 4000            # frac 포화 + 이만큼 성공데이터 → Stage 2(CVAE) 전환
    # Stage 2(경쟁): CVAE 디코더 + α(latent 분산=난이도). 성공하면 재학습 + α 확대.
    alpha0: float = 1.0
    alpha_grow: float = 1.1
    alpha_max: float = 3.0
    retrain_epochs: int = 120
    retrain_batch: int = 256
    beta_kl: float = 0.5
    buffer_max: int = 40000
    lr: float = 1e-3


class CompetitiveCurriculum:
    """map generator ⇄ tracker 경쟁 루프(논문 2단계).
    Stage1(부트스트랩): 쉬운 범위서 시작→성공하면 범위 확대, 성공 ψ로 CVAE 초기학습.
    Stage2(경쟁): CVAE 디코더+α로 생성→성공하면 성공 ψ 재학습+α 확대(프론티어 밀기)."""

    def __init__(self, cvae: CVAE, cfg: CurriculumCfg, device='cpu'):
        self.cvae, self.cfg, self.device = cvae, cfg, device
        self.opt = torch.optim.Adam(cvae.parameters(), lr=cfg.lr)
        self.frac = cfg.frac0
        self.alpha = cfg.alpha0
        self.stage = 1
        self.buffer_psi: list[torch.Tensor] = []   # 성공한 ψ — CVAE 학습 데이터
        self.buffer_y: list[torch.Tensor] = []

    @torch.no_grad()
    def gen_episode(self, n_envs: int):
        """n_stones개 디딤돌 ψ 시퀀스를 자기회귀(y=직전2ψ) 생성. Stage1=범위확대 uniform·Stage2=CVAE."""
        c = self.cfg
        prev = PSI_LO.to(self.device).unsqueeze(0).repeat(n_envs, 1)
        prev2 = prev.clone()
        lo, span = PSI_LO.to(self.device), (PSI_HI - PSI_LO).to(self.device)
        seq = []
        for k in range(c.n_stones):
            t_last = torch.full((n_envs, 1), 1.0 if k == c.n_stones - 1 else 0.0, device=self.device)
            y = make_cond(prev, prev2, t_last)
            if self.stage == 1:
                psi = lo + torch.rand(n_envs, PSI_DIM, device=self.device) * self.frac * span   # 부트스트랩
            else:
                psi = self.cvae.generate(y, alpha=self.alpha)                                    # CVAE
            seq.append((psi, y))
            prev2, prev = prev, psi
        return seq

    def retrain(self):
        """성공 버퍼로 CVAE 학습(recon+KL) = tracker가 넘은 지형 분포 학습."""
        c = self.cfg
        if len(self.buffer_psi) < c.retrain_batch:
            return None
        U = psi_normalize(torch.stack(self.buffer_psi).to(self.device))
        Y = torch.stack(self.buffer_y).to(self.device)
        N = U.shape[0]; last = None
        for _ in range(c.retrain_epochs):
            idx = torch.randint(0, N, (c.retrain_batch,), device=self.device)
            loss, rec, kl = self.cvae.loss(U[idx], Y[idx], beta=c.beta_kl)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            last = (float(rec), float(kl))
        return last

    def add_success(self, psi: torch.Tensor, y: torch.Tensor):
        for p, yy in zip(psi.detach().cpu(), y.detach().cpu()):
            self.buffer_psi.append(p); self.buffer_y.append(yy)
        if len(self.buffer_psi) > self.cfg.buffer_max:
            self.buffer_psi = self.buffer_psi[-self.cfg.buffer_max:]
            self.buffer_y = self.buffer_y[-self.cfg.buffer_max:]

    def step_round(self, tracker, n_envs=300):
        """1 라운드: 생성→시도(성공 ψ 수집)→마스터(>9.3/10) 시 CVAE 재학습 + 난이도 확대."""
        c = self.cfg
        seq = self.gen_episode(n_envs)
        overcome = torch.zeros(n_envs, device=self.device)
        alive = torch.ones(n_envs, dtype=torch.bool, device=self.device)
        for psi, y in seq:
            ok = tracker.attempt(psi) & alive          # 이 디딤돌 성공?(순차: 실패 시 이후 못감)
            overcome += ok.float()
            if ok.any():
                self.add_success(psi[ok], y[ok])       # 성공 ψ만 버퍼에
            alive = alive & ok
        mo = float(overcome.mean())
        info = {'overcome': mo, 'stage': self.stage, 'frac': self.frac, 'alpha': self.alpha,
                'buffer': len(self.buffer_psi), 'expanded': False, 'rk': None}
        if mo > c.success_thresh:                       # 현 난이도 마스터 → CVAE 학습 + 확장
            info['rk'] = self.retrain()                 # 성공 분포 학습
            if self.stage == 1:
                self.frac = min(1.0, self.frac * c.frac_grow)
                if self.frac >= 0.999 and len(self.buffer_psi) >= c.stage2_buffer:
                    self.stage = 2                      # CVAE로 전환(부트스트랩 완료)
            else:
                self.alpha = min(c.alpha_max, self.alpha * c.alpha_grow)
            info['expanded'] = True
        return info


# ─────────────────────── Mock tracker (self-test용) ───────────────────────
class MockTracker:
    """실 tracker 대역. skill이 라운드마다 성장(RL 연속학습). ψ 난이도(r·x_tilt 지배)가 skill보다 낮으면 성공.
    → 난이도 확대 ⇄ tracker 성장 의 경쟁 동역학을 재현(생성기 로직 격리 검증용).
    ★실통합 시 이 클래스를 실제 tracker 에피소드 성공신호로 교체."""
    def __init__(self, skill0=0.15, grow=0.03, skill_max=1.0, sharp=40.0):
        self.skill, self.grow, self.skill_max, self.sharp = skill0, grow, skill_max, sharp

    def difficulty(self, psi):
        r = (psi[:, 0] - PSI_LO[0].to(psi)) / (PSI_HI[0] - PSI_LO[0]).to(psi)     # 거리(프론티어 지배)
        xt = (psi[:, 3] - PSI_LO[3].to(psi)) / (PSI_HI[3] - PSI_LO[3]).to(psi)    # 종tilt(벽주행)
        return 0.6 * r + 0.4 * xt

    def attempt(self, psi):
        d = self.difficulty(psi)
        p = torch.sigmoid((self.skill - d) * self.sharp)          # skill≫난이도라야 9.3/10(순차) 가능
        return torch.rand_like(p) < p

    def learn(self):
        self.skill = min(self.skill_max, self.skill + self.grow)  # 연속 성장


def self_test():
    """mock tracker와 경쟁 루프를 돌려, 생성 ψ 프론티어(r·x_tilt)가 확장·CVAE가 학습되는지 확인(논문 Fig.6C 정성)."""
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cvae = CVAE().to(dev)
    cur = CompetitiveCurriculum(cvae, CurriculumCfg(), device=dev)
    tracker = MockTracker()

    print(f"[cvae_mapgen self-test] device={dev}  (Raibo2025 competitive CVAE map generator)")
    print(f"{'rnd':>4} {'stage':>5} {'skill':>6} {'overcome':>8} {'frac':>5} {'alpha':>5} "
          f"{'r_max':>6} {'xt_max°':>7} {'recon':>7} {'buf':>6}")
    for rnd in range(45):
        info = cur.step_round(tracker, n_envs=300)
        tracker.learn()                                  # tracker 연속 성장(RL)
        with torch.no_grad():                            # 진단: 현 생성 분포 프론티어
            allpsi = torch.cat([p for p, _ in cur.gen_episode(512)], 0)
            r_max = float(allpsi[:, 0].max()); xt_max = math.degrees(float(allpsi[:, 3].max()))
        rec = f"{info['rk'][0]:.3f}" if info['rk'] else "  -  "
        print(f"{rnd:>4} {info['stage']:>5} {tracker.skill:>6.2f} {info['overcome']:>8.2f} "
              f"{info['frac']:>5.2f} {info['alpha']:>5.2f} {r_max:>6.2f} {xt_max:>7.1f} {rec:>7} {info['buffer']:>6}")
    print("기대: skill↑ + frac/α↑ + r_max→~1.6·xt_max→~90°(벽주행) = 프론티어 확장 & CVAE 학습(recon↓). 논문 Fig.6C 정성 재현.")


if __name__ == '__main__':
    self_test()
