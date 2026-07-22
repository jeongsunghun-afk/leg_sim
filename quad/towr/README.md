# TOWR-in-CasADi — 모델기반 지형 궤적최적화(오프라인) + 추종

TOWR(Winkler 2018) phase-based 궤적최적화를 **CasADi+IPOPT로 우리 스택에 재구현**.
C++ TOWR(ifopt/ROS/catkin) 빌드 없이, casadi 3.7 번들 IPOPT 사용. A/B/D1이 못하는
**footholds를 지형 위 결정변수로 최적화**(갭 회피·계단 base상승)가 핵심.

## 구성
- `towr_cd.py` — Phase0 궤적최적화. SRBD 동역학 + footholds(지형 위) + 마찰콘 + ROM.
  게이트 `trot`(동적 대각쌍) / `crawl`(한발씩, 3발지지 정적안정). 지형 `flat/step/gap/platgap`.
  출력=base pos/ori, 발 위치, 접촉력 궤적 JSON.
- `towr_track.py` — Phase1 추종. 오프라인 IK로 관절궤적 변환 + **SRBD 힘-기반 균형 추종
  (WBIC-lite)**: base pose 피드백→보정렌치→지지발 GRF 분배 + 중력보상 + 스윙발 PD.

## 실행 (경로 무관·풀경로)
```bash
PIX=/home/jsh/simple-mpc/.pixi/envs/default/bin/python   # casadi+IPOPT+pinocchio
TD=/home/jsh/문서/jsh/simulation/quad/towr
MJ=/home/jsh/문서/jsh/simulation/quad/mjcf

# ── STEP 0.10m 크로싱(★검증됨: base 0.50→0.64 등반, falls=0, tilt7°) ──
env TERRAIN=step GAIT=crawl N=80 DT=0.02 XGOAL=0.9 X0=0.5 H=0.10 TG=0.80 DUTY=0.8 \
    OUT=$TD/traj_crawl_step.json $PIX $TD/towr_cd.py
env TRAJ=$TD/traj_crawl_step.json MJCF=$MJ/quad_terrain_step.mjcf VIEW=1 $PIX $TD/towr_track.py

# ── 평지 crawl(sanity) ──
env TERRAIN=flat GAIT=crawl N=60 XGOAL=0.5 OUT=$TD/traj_crawl_flat.json $PIX $TD/towr_cd.py
env TRAJ=$TD/traj_crawl_flat.json VIEW=1 $PIX $TD/towr_track.py

# ── 오프라인 planning만(trot, 지형인지 궤적) ──
env TERRAIN=gap N=50 XGOAL=1.0 X0=0.55 X1=0.80 DEPTH=0.30 $PIX $TD/towr_cd.py
```

## 상태 (2026-07-21)
- **Phase0 planning 작동**: flat/step/gap 궤적 solve. footholds가 지형 정합·갭 회피
  (깊은갭 착지 0개), base 계단서 상승, SRBD 동역학 잔차 ~1e-6.
- **Phase1 추종 작동(STEP)**: ★TOWR STEP 궤적 → WBIC-lite 추종 → 로봇이 0.10m 단 위로
  **base 0.50→0.64 상승하며 완주(falls=0, tilt7°)**. 엔드투엔드 모델기반 지형 크로싱 실증.
  안정 게인=KP_R 1600(강 자세권한)·KP_P 300(부드러운 위치)·KP_J 20(약 스윙반력).
### GAP 크로싱 진행 (2026-07-22)
- **★GAP PLANNING 해결**: 핵심=**짧은 stance**. 긴 crawl stance(Tg0.80·0.64s)는 발판 하나가
  base 0.19m 범위를 커버해야 해 유효ROM을 조여 갭 근처 infeasible. **Tg≤0.40(stance≤0.28s)로
  줄이면** 발이 갭 밖으로 빨리 재배치→feasible+**갭 회피 완벽**(갭내착지 0, cadence 스윕 검증).
  - 전역 위상오프셋 8개 스윕=**전부 infeasible**(정렬 아닌 stance 지속시간 문제 확인).
  - 부수: base높이 참조 bhref(갭 위 base가 갭바닥으로 안 빠지게 지지면 레벨), platgap 지형·phase_off.
- **GAP TRACKING=풀 WBIC 필요(미해결)**: WBIC-lite(준정적 gravity+GRF)는 **slow crawl(Tg0.80,
  step)은 추종하나 fast cadence(Tg0.40)는 못 잡음** — 갭 전 평지서 이미 발산(빠른 스윙 다리반력을
  준정적 힘균형이 미보상). computed-torque(rnea 전신역동역학) 시도=프레임규약 버그로 실패(제거).
  ★게인스윕 "완주 tilt6.7"은 거짓양성(텀블링 후 끝점만 우연히 정립)→낙상감지에 tilt>50 추가.
  **결론: 갭 추종은 full-dynamics WBIC(QP: M q̈+h=Sᵀτ+J_cᵀf, 접촉·마찰콘)로 승격 필요.**

### QP-WBIC 프로토타입 (2026-07-22, towr_wbic.py)
풀 QP-WBIC 구현(proxsuite): 변수 [q̈, f], base 동역학 하드등식 + 접촉 소프트task + 마찰콘,
MuJoCo M/bias/Jac 사용, τ 복원. **부분 작동**:
- ✅ 정지 안정(tilt<3°), 첫 스윙 통과(소프트접촉+STIFF), 일부 config는 목표 x 도달.
- ❌ **지속 보행 미달**: 스윙 몇 회 후 자세 발산(tilt↑)·base 전진 부족. 원인 후보=접촉전이
  임팩트 처리·정확한 J̇q̇(유한차분은 노이즈)·계층 null-space 부재·soft/rigid 접촉 잔차.
- 결론: **성숙 WBIC 엔지니어링 필요**. from-scratch보다 **B(simple_mpc)의 검증된 C++ WBIC 재사용이
  신뢰 경로**. B WBIC는 접촉전이·soft접촉·null-space를 이미 해결(STIFF, [[b-elevation-tamols-towr-track]]).
실행: `env TRAJ=traj_crawl_flat.json VIEW=1 $PIX towr_wbic.py` (게인 env: KP_R·W_BR·W_C·KP_S…)

## 다음
1. **★B의 WBIC 재사용**(최우선): TOWR 궤적을 B(quad_centroidal_17dof)의 WBIC 참조로 브리지
   → 접촉전이·soft접촉 이미 해결된 추종. QP-WBIC 프로토타입은 참고/폴백.
2. 변동 phase timing → slow-on-platform + short-at-gap(트래킹 부담↓) + 넓은 갭.
3. 실제 지형맵(heightmap) → 지형함수 자동생성(perceptive 연동).
