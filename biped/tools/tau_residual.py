#!/usr/bin/env python3
"""fTorque 가 **실측 토크**인가 **명령 에코**인가 — 트레이스 잔차로 가른다.

★왜 (2026-09-01): 발 힘센서가 없어서 접촉력을 관절토크로 역추정(proprioceptive force
  estimation)하고 싶은데, 그 전제가 "fTorque 가 측정이다" 이다. 그런데 우리 코드는
  `biped_deploy.cpp:475` 에서 **"명령 에코"라고 단정**해 놓고 한 번도 확인한 적이 없다.
  (확인된 것은 `fTorque ≡ fCurrent` 비트 동일뿐 — 같은 슬롯이라는 뜻일 뿐이다.)

원리:
  드라이브는 임피던스 모드에서  τ = kp·(q_des − q) + kd·(q̇_des − q̇) + τ_ff  를 낸다.
  **hold 0%** 는 τ_ff = 0 이고 q̇_des = 0 이므로, 에코라면 보고 토크가 우변과 정확히 같다.
      r = τ_보고 − [ kp·(q_cmd − q) + kd·(−q̇) ]
    · |r| ≈ 0 (float16 정밀도)        → **명령 에코**. 역추정 불가
    · |r| ~ 0.6~0.9 N·m, 부호가 −sign(q̇) → **실측**. 그 잔차가 곧 쿨롱마찰이고 역추정 가능
    · |r| 가 크고 부호가 무관           → 제3의 항(포화·보상·단위오류). 아래 배율도 같이 본다

⚠트레이스는 2026-08-28 에 고친 형식이어야 한다(열 6개/채널: q,dq,tau,cmd,kp,kd).
  그 전 파일은 `cmd` 열이 **직전 틱 측정각**이라 이 계산이 통째로 무의미하다 — 자동 거부한다.

사용:
  /usr/bin/python3 /home/rpetubt/simulation/biped/tools/tau_residual.py /tmp/arm_trace.csv
  (노트북: /home/jsh/simulation/biped/tools/tau_residual.py)
"""
import csv
import sys

import numpy as np

CH = ['HL_hip', 'HL_thigh', 'HL_calf', 'HL_foot',
      'HR_hip', 'HR_thigh', 'HR_calf', 'HR_foot']
D2R = np.pi / 180.0


def main(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    head = rows[0]
    ncol = len(head) - 1
    if ncol % 6 != 0 or 'kp0' not in head:
        print('✗ 옛 트레이스 형식이다 (열 %d개, kp 열 없음).' % ncol)
        print('  2026-08-28 이전 파일은 `cmd` 열이 명령각이 아니라 **직전 틱 측정각**이라')
        print('  이 분석이 성립하지 않는다. 고친 배포기로 다시 받아야 한다.')
        return 2
    nch = ncol // 6
    d = np.array([[float(x) for x in r] for r in rows[1:] if len(r) == len(head)])
    t = d[:, 0]
    fs = 1.0 / np.median(np.diff(t))
    print('■ %s — 채널 %d · 표본 %d · %.1f Hz\n' % (path, nch, len(d), fs))

    # ★퇴화 방어 — 목업 백엔드는 토크를 0 으로 보고한다. 그대로 계산하면 |r|/τ 가
    #   1e8 로 터지고 R²=1.000·a=0 이 나와 "완전선형 = 에코" 라고 **거짓 판정**한다.
    #   (실제로 그렇게 찍혀서 이 가드를 넣었다 — 2026-09-01)
    tall = np.concatenate([np.abs(d[:, 3 + 6 * i]) for i in range(min(nch, 8))])
    if tall.mean() < 0.05:
        print('✗ 토크 에코가 전 축 ~0 이다 (평균 %.4f N·m).' % tall.mean())
        print('  목업(--mock) 이거나 무여자 구간이다 — 이 시험은 **실기에서 토크가 실린 상태**로만')
        print('  유효하다. hold 0% 로 접지·가압한 뒤 3초 트레이스를 다시 받아라.')
        return 3

    print('  %-9s %8s %8s %8s %8s %8s   %s' %
          ('축', '|r|평균', '|r|최대', 'τ규모', '|r|/τ', '상관(r,-sgn q̇)', '판정'))
    verdicts = []
    for i in range(min(nch, 8)):
        q, dq, tau = d[:, 1 + 6 * i], d[:, 2 + 6 * i], d[:, 3 + 6 * i]
        cmd, kp, kd = d[:, 4 + 6 * i], d[:, 5 + 6 * i], d[:, 6 + 6 * i]
        if np.all(kp == 0) and np.all(kd == 0):
            print('  %-9s (게인 0 — 순수 토크 구간이라 이 시험이 성립하지 않는다)' % CH[i])
            continue
        pd = kp * (cmd - q) * D2R + kd * (-dq) * D2R      # 드라이브가 낼 PD 토크
        r = tau - pd
        scale = np.abs(tau).mean()
        # 마찰이면 잔차 부호가 −sign(q̇) 를 따른다
        mv = np.abs(dq) > 3.0                              # 정지구간은 부호가 무의미
        corr = 0.0
        if mv.sum() > 20:
            a, b = r[mv], -np.sign(dq[mv])
            if a.std() > 1e-9:
                corr = float(np.corrcoef(a, b)[0, 1])
        ra = np.abs(r).mean()
        if ra < 0.05:
            v = '★에코 (역추정 불가)'
        elif 0.2 < ra < 2.0 and corr > 0.3:
            v = '★실측 — 잔차=마찰'
        elif ra >= 2.0:
            v = '⚠제3의 항 (배율표 확인)'
        else:
            v = '판정보류'
        verdicts.append(v)
        print('  %-9s %8.3f %8.3f %8.3f %8.3f %13.2f   %s' %
              (CH[i], ra, np.abs(r).max(), scale, ra / max(scale, 1e-9), corr, v))

    # 단위오류(kt 5배) 가설 — τ 와 PD 가 **비례**하면 에코이되 스케일이 다른 것이다
    print('\n  ── 배율 검정: τ_보고 = a·τ_PD 로 최소자승 (a≈1 이면 같은 단위, a≈0.2 면 kt 단위오류) ──')
    for i in range(min(nch, 8)):
        q, dq, tau = d[:, 1 + 6 * i], d[:, 2 + 6 * i], d[:, 3 + 6 * i]
        cmd, kp, kd = d[:, 4 + 6 * i], d[:, 5 + 6 * i], d[:, 6 + 6 * i]
        pd = kp * (cmd - q) * D2R + kd * (-dq) * D2R
        den = float(pd @ pd)
        if den < 1e-9:
            continue
        a = float(pd @ tau) / den
        tvar = float((tau - tau.mean()) @ (tau - tau.mean()))
        if tvar < 1e-6:                      # τ 가 상수면 R² 이 정의되지 않는다
            print('    %-9s a = %6.3f   R² =    n/a   (τ 가 상수 — 판정 불가)' % (CH[i], a))
            continue
        resid = tau - a * pd
        r2 = 1.0 - float(resid @ resid) / tvar
        print('    %-9s a = %6.3f   R² = %6.3f   %s' %
              (CH[i], a, r2,
               '완전선형 = 에코' if r2 > 0.999 else ('선형성 낮음 = 독립성분 있음' if r2 < 0.95 else '')))

    print('\n  읽는 법:')
    print('   · 전 축 |r|≈0 이고 R²>0.999  → **명령 에코 확정.** 접촉력 역추정은 불가하고,')
    print('     접촉 판정은 속도 급감·QP 실패율 같은 간접 신호로 가야 한다.')
    print('   · |r| 가 0.6~0.9 N·m 이고 부호가 −sign(q̇) 를 따름 → **실측 확정.**')
    print('     그 잔차가 곧 쿨롱마찰이고, F = (Jᵀ)⁺(τ − τ_g − τ_i) 역추정이 성립한다.')
    print('   · a 가 1 에서 크게 벗어나면 **단위 문제**다(kt=0.2 면 a≈0.2). RGA 확인 필요.')
    print('   ⚠이 시험은 **hold 0%** 처럼 τ_ff=0 인 구간에서만 유효하다.')
    print('     stand/walk 는 τ_ff 가 있어 잔차가 곧 τ_ff 라 마찰과 구분이 안 된다.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/arm_trace.csv'))
