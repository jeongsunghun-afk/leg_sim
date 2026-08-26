# 무게추 foot 브래킷 실험 — Qhome8 결손(0.68)의 원인 확증 (2026-08-26 준비)

**원리**: float 브래킷이 calf/foot 을 못 쟀던 이유 = 중력 ≪ 마찰. 발에 **실측 질량**을
달면 foot 중력토크가 마찰을 넘고, 그 질량은 저울 실측이라 G비≈1 →
**브래킷 1/g\* ≈ r_foot 단독** (CAD 게이지 밖). 조건 A/B 로 발목 **관절각** 창을
옮겨가며 재면 "foot 전동의 발목각 의존 손실" 가설을 직격한다.

예측: A(관절 0° 창) r_foot ≈ 0.4~0.5 · B(−50° 창) ≈ 0.8 이면 가설 확증.
둘 다 0.8 이면 foot 무죄 → Qhome8 결손은 thigh/기하 쪽으로.

## 준비물
- 추 (권장 **2 kg**, 걸이 포함 총질량을 주방저울로 실측 · ±10 g)
- 부착: 발끝 부근에 **단단히** (진자 금지 — 짧게/테이프). 부착점 위치를
  발끝 구 중심 또는 발목축 기준 mm 로 기록 (±5 mm)

## 절차 (다리당 ~20분 · HL 예시)

```bash
# 0. 모델 생성 (Pi 에서 · mujoco 불필요) — 실측값으로 채울 것:
cd /home/rpetubt/simulation/biped && python3 tools/make_weighted_mjcf.py \
    --leg HL --mass-g <실측질량g> --at toe --x <mm> --y <mm> --z <mm>
#   → biped_from_quad_wHL.mjcf 생성. (--at ankle 로 발목 기준 지정도 가능)

# 1. 조건 A — foot 관절각 0° 창 (home = Qhome8 그대로)
cd /home/rpetubt/simulation/biped && ./run_deploy_hw.sh biped_from_quad_wHL.mjcf
#   (GUI 끄기) 브래킷:
cd /home/rpetubt/simulation/biped && python3 tools/float_gstar.py --axis HL_foot \
    --lo 0.3 --hi 1.7 --step 0.1 --hold "1.20,1.10,1.22,1.00,1.18,1.10,1.22,1.00"

# 2. 조건 B — foot 관절각 −50° 창: biped_emb.yaml home.q_deg 를
#    [0, 30, -20, -50, 0, 30, -20, -50] 로 임시 변경 후 배포기 재기동 → 같은 브래킷
#    ⚠끝나면 home 을 Qhome8 로 원복!

# 3. HR 반복: 추를 HR 로 옮겨 달고 --leg HR 로 모델 재생성 (모델은 단 다리만 반영)
```

⚠ 추가 안전: 추 낙하 주의 · 브래킷 중 발이 위아래로 표류 — 주변 간섭물 제거 ·
crane 줄 팽팽 · 축사망 시 포렌식 배너 복사.
⚠ \*_w\*.mjcf 는 실험용 임시 파일 — 커밋 금지, 실험 후 삭제.

기대 신호(2 kg 발끝 기준, mujoco 로 검증): A −1.83 Nm · B −2.34 Nm (마찰 0.64 의 3~4배 ✅)
