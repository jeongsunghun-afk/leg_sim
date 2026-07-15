"""TAMOLS 02_Leg 적응 — tamols-rl(Go2) 정식화에 02_Leg 파라미터 + 갭 heightmap 주입.
실행: cd go2-hrl && PYTHONPATH=fetch/tamols drake_venv/bin/python -m fetch.tamols.tamols_02leg
"""
import numpy as np
from .tamols import TAMOLSState, setup_variables
from .test import setup_costs_and_constraints, run_single_optimization, save_optimal_solutions
from .map_processing import process_height_maps
from .helpers import evaluate_spline_position, evaluate_spline_velocity


def export_trajectory(tmls, filepath, dt=0.005):
    """★C++ WBIC 추종용 궤적 export. base pose/vel(6) + 다리별 접촉상태(4) 샘플 + 발판(4×3).
       프레임=로봇중심(base 시작 0,0,0.52). C++가 1kHz로 보간·추종(wbic_jump식 base=GRF).
       포맷: 헤더 'N dt' → 'FOOTHOLDS' 4행(x y z) → 'SAMPLES' N행(t·pose6·vel6·c0..c3)."""
    coeffs = tmls.optimal_spline_coeffs                 # [phase][4×order]
    Td = tmls.phase_durations                           # phase별 길이
    pt = tmls.gait_pattern['phase_timing']              # 누적 경계 [0,..,T]
    cs = tmls.gait_pattern['contact_states']            # [phase][4]
    T_total = pt[-1]
    n = int(round(T_total / dt))
    rows = []
    for k in range(n + 1):
        t = k * dt
        ph = min(int(np.searchsorted(pt, t, side='right')) - 1, len(Td) - 1)
        ph = max(ph, 0)
        tau = t - pt[ph]
        pos = evaluate_spline_position(tmls, coeffs[ph], tau)   # [x,y,z,roll,pitch,yaw]
        vel = evaluate_spline_velocity(tmls, coeffs[ph], tau)   # [vx,vy,vz,wr,wp,wy]
        c = cs[ph]
        rows.append([t, *pos, *vel, *[int(x) for x in c]])
    fh = tmls.optimal_footsteps
    with open(filepath, 'w') as f:
        f.write(f"{len(rows)} {dt}\n")
        f.write("FOOTHOLDS\n")
        for L in range(4):
            f.write(f"{fh[L,0]:.6f} {fh[L,1]:.6f} {fh[L,2]:.6f}\n")
        f.write("SAMPLES\n")
        for r in rows:
            f.write(" ".join(f"{v:.6f}" for v in r[:13]) + " " + " ".join(str(int(v)) for v in r[13:]) + "\n")
    print(f"→ {filepath} export (N={len(rows)} dt={dt} T={T_total})")


def add_base_bounds(tmls, zlo=0.45, zhi=0.60, rp_max=0.20, yaw_max=0.15, nsamp=4):
    """★하드 base bound: 스플라인 z∈[zlo,zhi]·|roll|,|pitch|≤rp_max·|yaw|≤yaw_max 강제(오버슈트/틸트/스핀 원천차단).
       tracking cost가 전진 담당, 이 bound가 자세/높이 담당 → 소프트 가중 tension 해소.
       ★yaw 미구속 시 갭 전진 강제서 base가 yaw 스핀(100°)으로 퇴화해 도피 → yaw 구속 필수."""
    for phase in range(len(tmls.phase_durations)):
        a_k = tmls.spline_coeffs[phase]; T_k = tmls.phase_durations[phase]
        for tau in np.linspace(0, T_k, nsamp + 1)[1:]:
            s = evaluate_spline_position(tmls, a_k, tau)
            z = s[2]; roll = s[3]; pitch = s[4]; yaw = s[5]
            tmls.prog.AddConstraint(z >= zlo); tmls.prog.AddConstraint(z <= zhi)
            tmls.prog.AddConstraint(roll >= -rp_max); tmls.prog.AddConstraint(roll <= rp_max)
            tmls.prog.AddConstraint(pitch >= -rp_max); tmls.prog.AddConstraint(pitch <= rp_max)
            tmls.prog.AddConstraint(yaw >= -yaw_max); tmls.prog.AddConstraint(yaw <= yaw_max)


def add_base_forward_target(tmls, x_target=0.45):
    """★갭 크로싱 유도: 최종 base x가 갭 넘어까지 전진하도록 강제(갭 페널티 때문에 옵티마이저가 '안 건너기'를 택하는 것 방지).
       tracking(ref_vel)만으론 갭 앞에서 멈춤 → 터미널 전진 목표로 건너기 커밋."""
    phase = len(tmls.phase_durations) - 1
    a_k = tmls.spline_coeffs[phase]; T_k = tmls.phase_durations[phase]
    s = evaluate_spline_position(tmls, a_k, T_k)
    tmls.prog.AddConstraint(s[0] >= x_target)


def add_foot_y_bounds(tmls, ymin=0.10, ymax=0.22):
    """★대칭 stance 강제: 좌측발(FL0·RL2) y∈[+ymin,+ymax]·우측발(FR1·RR3) y∈[−ymax,−ymin].
       뒷발 비대칭/중앙쏠림(RR y≈−0.04·RL y≈+0.24) 차단 → 측방 안정 stance."""
    for L in (0, 2):
        tmls.prog.AddConstraint(tmls.p[L, 1] >= ymin); tmls.prog.AddConstraint(tmls.p[L, 1] <= ymax)
    for R in (1, 3):
        tmls.prog.AddConstraint(tmls.p[R, 1] <= -ymin); tmls.prog.AddConstraint(tmls.p[R, 1] >= -ymax)


def setup_02leg_state(tmls: TAMOLSState):
    # ── 02_Leg 로봇 파라미터 (Go2 기본값 → 02_Leg 17dof) ──
    tmls.mass = 37.9                 # Go2 6.921 → 02_Leg 17dof
    tmls.mu = 0.6                    # 배포 MU=0.6
    tmls.foot_radius = 0.018         # sphere 발 반경
    tmls.nominal_height = 0.52       # 배포 base_z0≈0.5234
    tmls.desired_height = 0.52
    tmls.h_des = 0.52                # ★★base 정렬이 실제 쓰는 목표높이(Go2 기본0.25→base가 0.25로 끌려 요동한 원인)
    tmls.base_pose_sampling_rate = 3 # ★스플라인 전체 높이구속(기본1=phase시작만 → 중간 요동)
    tmls.l_min = 0.12                # 다리 최소/최대 reach (크라우치~신전)
    tmls.l_max = 0.80
    tmls.min_foot_distance = 0.10
    # hip 오프셋(base 기준 hip joint 근사) — x=±0.225, y=±0.14(명목 stance)
    tmls.hip_offsets = np.array([
        [ 0.225,  0.14, 0],   # FL
        [ 0.225, -0.14, 0],   # FR
        [-0.225,  0.14, 0],   # RL
        [-0.225, -0.14, 0],   # RR
    ])

    # ── 초기 상태 ──
    tmls.base_pose = np.array([0, 0, 0.52, 0, 0, 0])
    tmls.base_vel = np.array([0, 0, 0, 0, 0, 0])
    tmls.p_meas = np.array([
        [ 0.225,  0.14, 0],   # FL
        [ 0.225, -0.14, 0],   # FR
        [-0.225,  0.14, 0],   # RL
        [-0.225, -0.14, 0],   # RR
    ])

    # ── heightmap: 원점 중심, 앞쪽(+x)에 갭(trench) ──
    #   map 중심=원점(로봇). x∈[-off,+off]. 갭=z낮춤(빠짐). 나머지=평지 z=0.
    tmls.cell_size = 0.05
    tmls.map_size = 27                              # Go2 동일(NaN 격리)
    import os as _os
    if _os.environ.get("GAP", "0") != "0":
        off = tmls.cell_size * tmls.map_size / 2.0
        N = tmls.map_size
        elev = np.zeros((N, N), dtype=float)
        for i in range(N):
            x = i * tmls.cell_size - off
            if 0.20 <= x <= 0.40:                  # 0.20m 갭
                elev[i, :] = -0.5
    else:
        import manual_heightmaps as mhm
        elev = mhm.get_heightmap_with_holes(tmls)  # Go2 작동 맵(격리 테스트)
    tmls.h = elev

    h_s1, h_s2, gradients = process_height_maps(elev)
    tmls.h_s1 = h_s1; tmls.h_s2 = h_s2
    tmls.h_grad_x, tmls.h_grad_y = gradients['h']
    tmls.h_s1_grad_x, tmls.h_s1_grad_y = gradients['h_s1']
    tmls.h_s2_grad_x, tmls.h_s2_grad_y = gradients['h_s2']

    tmls.ref_vel = np.array([0.4, 0, 0])
    tmls.ref_angular_momentum = np.array([0, 0, 0])

    # ── 게이트(trot-like, tamols-rl 동일) ──
    tmls.gait_pattern = {
        'phase_timing': [0, 0.4, 0.8, 1.2, 1.6, 2.0],
        'contact_states': [
            [1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1],
        ],
        'at_des_position': [
            [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1], [1, 1, 1, 1],
        ],
    }


if __name__ == "__main__":
    tmls = TAMOLSState()
    setup_02leg_state(tmls)
    setup_variables(tmls)
    setup_costs_and_constraints(tmls)
    add_base_bounds(tmls, zlo=0.45, zhi=0.60, rp_max=0.20)   # ★하드 자세/높이 bound
    add_foot_y_bounds(tmls, ymin=0.10, ymax=0.22)            # ★하드 foot-y bound(대칭 stance)
    import os as _o
    if _o.environ.get("GAP", "0") != "0":
        add_base_forward_target(tmls, x_target=0.45)         # ★갭이면 전진 강제(갭 넘어까지)
    print("\n===== 02_Leg TAMOLS solve =====", flush=True)
    ok = run_single_optimization(tmls)
    print("solve 성공:", ok, flush=True)
    if ok:
        save_optimal_solutions(tmls, filepath='out/02leg_solution.txt')
        print("→ out/02leg_solution.txt 저장")
        export_trajectory(tmls, filepath='out/02leg_traj.txt', dt=0.005)   # ★C++ WBIC 추종용
