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
