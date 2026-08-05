"""terrain_demo — end-to-end: CVAE 학습 → ψ 생성(easy/hard) → 디딤돌 포즈 → MJCF 지형 export.

CVAE map generator의 출력을 **실제 로드 가능한 MuJoCo 지형**으로 만드는 완결 파이프라인:
  CVAE.generate(ψ, α) → psi_to_stones(ψ) → stones_to_mjcf → terrain_{easy,hard}.mjcf
α로 난이도 조절(easy=낮은 α·hard=높은 α) → 커리큘럼 진화를 지형 파일로 확인(논문 Fig.6D 정성).

실행(torch 환경): python terrain_demo.py   → terrain_easy.mjcf, terrain_hard.mjcf 생성.
"""
import torch
from cvae_mapgen import CVAE, CompetitiveCurriculum, Cfg, MockTracker, BOOTSTRAP
from psi_to_stones import psi_to_stones, stones_to_mjcf


def main():
    dev = 'cpu'; torch.manual_seed(0)
    cvae = CVAE().to(dev); cfg = Cfg(); cur = CompetitiveCurriculum(cvae, cfg, dev); trk = MockTracker()

    # 학습: 부트스트랩(Table S3) + 경쟁 몇 update
    for st in range(len(BOOTSTRAP)):
        for _ in range(cfg.boot_rounds_per_stage):
            cur.rollout(cur.gen_boot(300, st), trk); trk.update()
    cur.retrain(); cur.alpha = 0.0
    for upd in range(30):                                      # 경쟁(프론티어까지)
        p = cur.rollout(cur.gen_cvae(300, cur.alpha), trk); trk.update()
        if upd % cfg.update_period == 0 and p > cfg.perf_retrain:
            cur.retrain(); cur.alpha = cfg.alpha_reset
    print(f"학습 완료(skill={trk.skill:.2f}). CVAE map generator → 지형 export:")

    # easy=초기 커리큘럼 stage0(Table S3, r 0.4~0.8) · hard=학습된 CVAE 프론티어(α=1.5)
    plans = [("easy", lambda: cur.gen_boot(1, 0)), ("hard", lambda: cur.gen_cvae(1, 1.5))]
    for label, gen in plans:
        psi = torch.cat([p for p, _ in gen()], 0)              # (K,6)
        centers, Rs = psi_to_stones(psi)
        open(f"terrain_{label}.mjcf", "w").write(stones_to_mjcf(centers, Rs, name=f"cvae_{label}"))
        rmin, rmax = float(psi[:, 0].min()), float(psi[:, 0].max())
        xtmax = float(psi[:, 4].max()) * 57.2958
        print(f"  {label:>4}: {psi.shape[0]} 디딤돌  r∈[{rmin:.2f},{rmax:.2f}]m  x_tilt_max={xtmax:.0f}°  → terrain_{label}.mjcf")
    print("→ MuJoCo에 직접 로드 가능(우리 스택). RobotSW_IsaacLab 통합 시 이 ψ→지형을 env 빌더로.")


if __name__ == '__main__':
    main()
