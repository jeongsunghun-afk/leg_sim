# TAMOLS ① (faithful, Drake/GIAC) — 02_Leg 적응

full TAMOLS(base+발판 동시최적화, GIAC 안정성) = ③(발판선택)이 못 깨는 넓은 갭용.
tamols-rl(github ianpedroza, Go2)의 Drake 정식화를 02_Leg로 적응.

## 셋업 (scratchpad은 세션한정 → 재생성 필요)
```bash
git clone https://github.com/ianpedroza/tamols-rl.git
python3.11 -m venv drake_venv && drake_venv/bin/pip install drake numpy plotly scipy
cp tamols_02leg.py tamols-rl/go2-hrl/fetch/tamols/
cd tamols-rl/go2-hrl && mkdir -p out
PYTHONPATH=fetch/tamols ../drake_venv/bin/python -m fetch.tamols.tamols_02leg   # GAP=1로 갭맵
```

## 상태 (2026-07-14)
- ✅ 02_Leg 파라미터로 feasible solve(mass37.9·nominal0.52·hip_offsets·l_min0.12/l_max0.80).
- ⚠️ 솔루션 품질 나쁨: base z 요동(0.52→0.03→0.43) — 비용가중 튜닝 필요(base_alignment·smoothness).
- ⚠️ reach 물리값(0.25/0.60)은 infeasible → 넓혀야 함(초기추정 개선 필요).
- ⚠️ 갭 heightmap NaN(급격 rows) → 매끄러운 갭 필요.

## 다음
비용가중 튜닝(base 안정화) → reach 물리복귀 → 갭맵 매끄럽게 → base+발판 export → C++ 17dof WBIC(wbic_jump식) 추종.

## 참조
- 정식화: tamols-rl fetch/tamols/{tamols.py, constraints.py(GIAC), costs.py, map_processing.py}
- 논문: Jenelten et al. TAMOLS, T-RO 2022, arXiv 2206.14049
- 메모리: perceptive-nav-tamols

## 품질 튜닝 (2026-07-14, 이어서)
- ★★**핵심 버그 = `h_des`(Go2 기본 0.25)**: base_pose_alignment_cost가 nominal/desired_height가 아니라 `tmls.h_des`를 씀 → base가 0.25로 끌려 요동(0.52→0.03). **`tmls.h_des=0.52`로 수정 → base 추락 해결(0.5~0.8 유지·rpy~0 레벨).**
- base_pose_sampling_rate 1→3(스플라인 전체 높이구속), costs.py base_alignment 가중 0.01→0.1(★ws costs.py 직접수정, 프로젝트엔 미반영=재적용 필요).
- ⚠️ **남은 균형문제**: base_alignment 가중↑(0.3)=base 안정하나 전진 안 함 / 가중↓(0.05)=전진하나 z 오버슈트(0.77)+틸트. 비선형 NLP 국소최적이라 노이지. **tracking↔regulation 균형 튜닝 or 하드 높이/자세 bound or 더 나은 초기추정 필요**(후속).
- ★재적용 시 costs.py 수정: `add_base_pose_alignment_cost`의 `weight = 0.01*...` → `0.1*...`

## 품질 튜닝 2 (하드 bound — base 궤적 해결)
- ★★**base 궤적 품질 해결**: `add_base_bounds()`(tamols_02leg.py 신규) = 스플라인 z∈[0.45,0.60]·|roll|,|pitch|≤0.20 **하드 제약** → 오버슈트/틸트 원천차단, tracking이 전진 담당(균형 tension 해소). 결과: base z 0.52 안정·전진 x→0.64·레벨. (소프트 base_alignment 가중 0.03로 낮춤=costs.py 직접수정.)
- costs.py 재적용: `add_base_pose_alignment_cost` weight `0.01`→`0.03`. (nominal_kinematic weight 20 유지=100은 infeasible.)
- ⚠️ **남은 후속 = 발판 stance 좁음**(발판 y~0으로 몰림=측방 불안정). nominal_kinematic 가중↑(100)=infeasible → 하드 foot-y bound(±0.14) or 게이트/초기추정 개선 필요.
- 실행 지속위치=`/home/jsh/tamols_ws`(tamols-rl+drake_venv 영구).
