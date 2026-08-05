"""psi_to_stones — Raibo2025 ψ → stepping-stone 월드 포즈. CVAE 출력을 실제 지형으로 변환하는 브리지.

논문 Components of ψ (기하 정의):
  d_n = (n-1)→n 디딤돌 중심 벡터 · x_n = d_n의 지면 투영 · Z_g = 중력 반대(위).
  r = |x_n|(수평거리) · θ = x_{n-1}↔x_n 각(heading 변화) · φ = d_n↔x_n 각(고도각) → 중심 위치.
  디딤돌 자세 = x_ns를 x_n·z_ns를 Z_g에 정렬 후, Δyaw(Z_g)·x_tilt(x_ns)·y_tilt(y_ns) 순차 회전.

역할: cvae_mapgen.CVAE.generate(ψ) → 이 함수 → (중심, 회전) → RobotSW_IsaacLab 지형 빌더(_build_gap_terrain 대체).
실행(검증): python psi_to_stones.py   (round-trip: ψ→포즈→ψ 복원 일치 확인)
"""
from __future__ import annotations
import math
import torch


def _Rz(a):  # scalar angle → (3,3)
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], device=a.device)
def _Rx(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[1., 0., 0.], [0., c, -s], [0., s, c]], device=a.device)
def _Ry(a):
    c, s = torch.cos(a), torch.sin(a)
    return torch.tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]], device=a.device)


def psi_to_stones(psi: torch.Tensor, C0: torch.Tensor | None = None, heading0: float = 0.0):
    """psi: (K,6) [r, θ, φ, Δyaw, x_tilt, y_tilt] 물리단위(rad·m). → centers (K,3), R (K,3,3).
    단일 env. 배치(N env)는 이 함수를 반복/vmap(지형 빌드는 CPU 1회성이라 충분)."""
    K = psi.shape[0]; dev = psi.device
    up = torch.tensor([0., 0., 1.], device=dev)
    C = torch.zeros(3, device=dev) if C0 is None else C0.clone()
    h_yaw = torch.tensor(float(heading0), device=dev)
    centers = torch.zeros(K, 3, device=dev); Rs = torch.zeros(K, 3, 3, device=dev)
    for k in range(K):
        r, th, phi, dyaw, xt, yt = (psi[k, i] for i in range(6))
        h_yaw = h_yaw + th                                        # heading 변화 θ
        h = torch.stack([torch.cos(h_yaw), torch.sin(h_yaw), torch.zeros((), device=dev)])  # 수평 heading 단위
        x_n = r * h                                              # 수평 변위(|x_n|=r)
        d_n = x_n + up * (r * torch.tan(phi))                    # 3D 변위(고도각 φ: 수직=r·tanφ)
        C = C + d_n
        centers[k] = C
        y_axis = torch.linalg.cross(up, h)                       # y_ns=좌측(up×heading)
        R_align = torch.stack([h, y_axis, up], dim=1)            # 열=[x_ns, y_ns, z_ns]
        R = R_align @ _Rz(dyaw) @ _Rx(xt) @ _Ry(yt)             # Δyaw(Z_g)·x_tilt(x_ns)·y_tilt(y_ns) 순차
        Rs[k] = R
    return centers, Rs


def recover_psi(centers: torch.Tensor, C0: torch.Tensor | None = None, heading0: float = 0.0):
    """centers (K,3) → (r, θ, φ) 복원(중심 기하 검증용). 자세(Δyaw/tilt)는 R 필요."""
    K = centers.shape[0]; dev = centers.device
    prev_c = torch.zeros(3, device=dev) if C0 is None else C0.clone()
    prev_yaw = float(heading0)
    out = torch.zeros(K, 3, device=dev)
    for k in range(K):
        d = centers[k] - prev_c
        xy = d[:2]; r = torch.linalg.norm(xy)
        yaw = math.atan2(float(xy[1]), float(xy[0]))
        th = yaw - prev_yaw
        th = math.atan2(math.sin(th), math.cos(th))              # wrap
        phi = math.atan2(float(d[2]), float(r))                  # 고도각
        out[k, 0] = r; out[k, 1] = th; out[k, 2] = phi
        prev_c = centers[k]; prev_yaw = yaw
    return out


def _self_test():
    torch.manual_seed(0)
    from cvae_mapgen import PSI_LO, PSI_HI, feasible
    # feasible ψ 시퀀스 샘플
    psi = PSI_LO + torch.rand(10, 6) * (PSI_HI - PSI_LO)
    psi = psi[feasible(psi)][:6]                                 # feasible 6개
    centers, Rs = psi_to_stones(psi)
    rec = recover_psi(centers)
    err = (rec[:, :3] - psi[:, :3]).abs().max().item()
    print("[psi_to_stones self-test] Raibo2025 Components-of-ψ 기하")
    print(f"  입력 ψ(r,θ°,φ°):")
    for k in range(psi.shape[0]):
        print(f"    stone{k}: r={psi[k,0]:.2f} θ={math.degrees(psi[k,1]):+.1f}° φ={math.degrees(psi[k,2]):+.1f}° "
              f"→ 중심=({centers[k,0]:+.2f},{centers[k,1]:+.2f},{centers[k,2]:+.2f})")
    print(f"  ★round-trip (r,θ,φ) 복원 최대오차 = {err:.2e}  (≈0 → 중심 기하 정확)")
    # 회전 정규직교 확인
    orth = (Rs[0] @ Rs[0].T - torch.eye(3)).abs().max().item()
    print(f"  ★디딤돌 회전 정규직교 오차 = {orth:.2e}")
    print("  → CVAE.generate(ψ) → psi_to_stones → 지형 빌더(RobotSW_IsaacLab _build_gap_terrain 대체)")


if __name__ == '__main__':
    _self_test()
