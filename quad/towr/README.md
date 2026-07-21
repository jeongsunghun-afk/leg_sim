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
- **GAP 크로싱=변동 timing 필요(미구현)**: 고정 phase timing은 갭>2·ROM_x(0.26m)서
  발이 어느 플랫폼도 못 닿는 사각구간→infeasible. 긴 crawl stance도 유효ROM을 조임.
  **TOWR 핵심 기능=phase duration을 결정변수화**하면 touchdown을 갭 밖으로 옮겨 해결.
  = 다음 개발과제(Phase0c). trot로는 standalone gap solve되나 동적이라 WBIC-lite 추종 난이.

## 다음
1. **변동 phase timing** → 넓은 갭 크로싱(TOWR 시그니처).
2. 추종을 B의 풀 WBIC로 승격(동적 trot 궤적도 추종 → 고속 지형).
3. 실제 지형맵(heightmap) → 지형함수 자동생성(perceptive 연동).
