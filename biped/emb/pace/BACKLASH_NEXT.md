# 백래시/유격 측정 — 다음 라운드 (매크로 OFF 재빌드 후)

절대경로. 로봇 **매달린 채**. deploy 는 모터 writer 하나 — 스크립트 전 `pkill -f build/biped_deploy`.

## 0. 매크로 OFF 검증 (CPU·로그 정상화 확인)
```bash
cd /home/rpetubt/simulation/biped/emb && diag/emb_ctl.sh start
ls -lh /tmp/emb.log ; df -h /tmp          # 로그 증가 멈췄나
top -bn1 | grep RobotEmbedded             # CPU 90%→정상?
```

## 1. calf 컴플라이언스 확정 — 홈 추 on/off (같은 자세, 위치 상쇄)
```bash
AUX_MODE=1 bash /home/rpetubt/simulation/biped/run_deploy_hw.sh /home/rpetubt/simulation/biped/biped_pointfoot_payload.mjcf
bash /home/rpetubt/simulation/biped/run_hw.sh home
bash /home/rpetubt/simulation/biped/run_hw.sh hold
bash /home/rpetubt/simulation/biped/run_hw.sh enc     # ① 추 없이 aux-q_ch
#  → 발끝 2kg
bash /home/rpetubt/simulation/biped/run_hw.sh enc     # ② 추 달고 aux-q_ch
#  ②-① 크다(>0.5°)=감속단이 하중에 휨 / 작다(~0.05°)=견고→움직임은 벨트
#  ⚠ status 로 calf τ 가 몇 Nm 실렸는지 확인(작으면 종아리를 수평 자세로 jog 후 재시도)
```

## 2. 나머지 축 백래시 (감속단)
```bash
pkill -f build/biped_deploy
for CH in 0 4 3 7; do   # HL_hip HR_hip HL_foot HR_foot
  AUX_MODE=1 python3 /home/rpetubt/simulation/biped/emb/pace/backlash_sweep.py --ch $CH --f0 0.02 --f1 0.15 --amp 8
done
```

## 3. 벨트 유격 — 외부 실측 (엔코더 사각)
```bash
bash /home/rpetubt/simulation/biped/run_hw.sh hold          # 모터 고정
bash /home/rpetubt/simulation/biped/run_hw.sh enc           # 손으로 종아리 흔드는 동안
#  → 다이얼게이지/각도기를 프레임↔링크 에 대고 유격 측정. HL vs HR 비교.
#  q_ch·aux 둘 다 고정인데 링크만 움직이면 = 벨트(또는 마운트) 확정.
```

## 해석 요약
- 감속단 백래시: 측정 셋 다 ≤0.1°(분해능 0.0625° 바닥) = 기어 교체 불필요, **단 벨트는 별개**.
- calf 체감 유격 = 벨트(aux 하류) 유력 → 벨트 재텐션/교체.
- 부하 컴플라이언스: 홈 추 on/off 로 확정.
