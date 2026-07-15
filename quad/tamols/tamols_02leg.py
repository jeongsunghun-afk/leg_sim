"""TAMOLS 02_Leg 적응 — tamols-rl(Go2) 정식화에 02_Leg 파라미터 + 갭 heightmap 주입.
실행: cd go2-hrl && PYTHONPATH=fetch/tamols drake_venv/bin/python -m fetch.tamols.tamols_02leg
"""
import numpy as np
from .tamols import TAMOLSState, setup_variables
from .test import setup_costs_and_constraints, run_single_optimization, save_optimal_solutions
from .map_processing import process_height_maps
from .helpers import evaluate_spline_position


def add_base_bounds(tmls, zlo=0.45, zhi=0.60, rp_max=0.20, nsamp=4):
    """★하드 base bound: 스플라인 z∈[zlo,zhi]·|roll|,|pitch|≤rp_max 강제(오버슈트/틸트 원천차단).
       tracking cost가 전진 담당, 이 bound가 자세/높이 담당 → 소프트 가중 tension 해소."""
    for phase in range(len(tmls.phase_durations)):
        a_k = tmls.spline_coeffs[phase]; T_k = tmls.phase_durations[phase]
        for tau in np.linspace(0, T_k, nsamp + 1)[1:]:
            s = evaluate_spline_position(tmls, a_k, tau)
            z = s[2]; roll = s[3]; pitch = s[4]
            tmls.prog.AddConstraint(z >= zlo); tmls.prog.AddConstraint(z <= zhi)
            tmls.prog.AddConstraint(roll >= -rp_max); tmls.prog.AddConstraint(roll <= rp_max)
            tmls.prog.AddConstraint(pitch >= -rp_max); tmls.prog.AddConstraint(pitch <= rp_max)


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
    print("\n===== 02_Leg TAMOLS solve =====", flush=True)
    ok = run_single_optimization(tmls)
    print("solve 성공:", ok, flush=True)
    if ok:
        save_optimal_solutions(tmls, filepath='out/02leg_solution.txt')
        print("→ out/02leg_solution.txt 저장")
