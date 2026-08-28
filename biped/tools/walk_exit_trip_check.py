#!/usr/bin/env python3
"""walk 이탈(→stand/hold) 순간의 잔류 스윙속도가 cfg 속도트립(200dps/20ms)에
걸리는지 실측 플랜트 sim 으로 재현한다.

Phase A: 실측 플랜트 walk (walk_demand_check.py 와 동일 env) 를 돌리며
         t=3.0~3.9s(3 보행주기) 구간에서 5ms 마다 (qpos,qvel) 스냅샷.
         walk 중 채널속도 900dps(walk 트립) 초과 여부도 같이 확인(디바운스 이월 전제 검증).
Phase B: 각 스냅샷에서 stand 진입(bs=0)과 동일한 제어로 전환:
         - 목표 = 전환 순간 측정 채널각 래치(stand_hold), dq_des=0
         - kp_ch/kd_ch 전량(비emb: hip100/6 thigh50/4 calf80/3.5 foot30/2)
         - WBIC 토크 0 (bs=0 → tau_ch*=0)
         - 드라이브 한계 x tau_max_frac 0.6 클램프
         120ms 롤아웃, 2ms(500Hz) 마다 vel_pk=max|dq_ch| / tau_pk=max|tau_drv| 를 재서
         200dps 연속 20ms(→vel 트립) / 15Nm 연속 50ms(→tau 트립) 판정.
"""
import os, sys, json
import numpy as np

BIPED = '/home/jsh/simulation/biped'
sys.path.insert(0, BIPED)
os.environ.update(ALPHA_AXIS='0.85', FOOT_FRIC_EXTRA='0.36', FRIC_COMP='0',
                  FOOT_COMP_NM='0', T_STEP='0.30')

import mujoco
import biped_mpc_wbic as BM
from biped_wbic import base_rpy

GEARK = [1.0, 1.0, 1.5, 1.2]
KP_CH = [100.0, 50.0, 80.0, 30.0]
KD_CH = [6.0, 4.0, 3.5, 2.0]
ALPHA = 0.85  # actuator_gear 에 주입돼 있으므로 ctrl 은 명령값 그대로 쓰면 된다
VEL_TRIP, VEL_MS = 200.0, 20.0
TAU_TRIP, TAU_MS = 15.0, 50.0
TAU_FRAC = 0.6

def ch_state(qj, dqj):
    """관절(8) → 채널각/채널속도(8). 채널: hip,thigh,calf,foot x 2다리"""
    ch_q = np.zeros(8); ch_dq = np.zeros(8)
    for leg in range(2):
        b = 4*leg
        ch_q[b+0] = qj[b+0]*GEARK[0]; ch_dq[b+0] = dqj[b+0]*GEARK[0]
        ch_q[b+1] = qj[b+1]*GEARK[1]; ch_dq[b+1] = dqj[b+1]*GEARK[1]
        ch_q[b+2] = qj[b+2]*GEARK[2]; ch_dq[b+2] = dqj[b+2]*GEARK[2]
        ch_q[b+3] = (qj[b+2]+qj[b+3])*GEARK[3]; ch_dq[b+3] = (dqj[b+2]+dqj[b+3])*GEARK[3]
    return ch_q, ch_dq

def pd_ctrl(m, ch_latch, qj, dqj):
    """드라이버 MIT PD (채널공간, dq_des=0, tau_ff=0) → actuator ctrl(8)"""
    ch_q, ch_dq = ch_state(qj, dqj)
    u = np.zeros(8); tau_drv = np.zeros(8)
    for leg in range(2):
        b = 4*leg
        for k in range(4):
            i = b+k
            td = KP_CH[k]*(ch_latch[i]-ch_q[i]) + KD_CH[k]*(0.0-ch_dq[i])  # 드라이브측 Nm
            tau_drv[i] = td
            u[i] = td*GEARK[k]   # actuator 좌표: hip/thigh=관절, calf=관절(x1.5), foot=tendon(x1.2)
    # 드라이브 한계 x 0.6 (C++ 와 동일: ctrlrange x tau_max_frac). ctrlrange 는 actuator 좌표.
    for i in range(8):
        lim = m.actuator_ctrlrange[i,1]*TAU_FRAC
        u[i] = np.clip(u[i], -lim, lim)
        tau_drv[i] = u[i]/GEARK[i%4]   # 클램프 반영한 드라이브측 보고토크
    return u, tau_drv

def main():
    c = BM.BipedMPCWBIC(mjcf=os.path.join(BIPED, 'biped_flatfoot.mjcf'))
    c.set_contact_mode('1pt'); c.reset(); c.setup_mpc()
    m, d = c.m, c.d
    dt = m.opt.timestep
    snaps = []          # (t, qpos, qvel)
    walk_vel_pk_max = 0.0
    fell = None
    T_END = 3.9
    for k in range(int(T_END/dt)+1):
        t = k*dt
        c.vx_cmd = 0.10 if t > 2.0 else 0.0
        c.wz_cmd = c.vy_cmd = 0.0
        c.control(dt)
        mujoco.mj_step(m, d)
        tilt = float(np.hypot(*base_rpy(d.qpos[3:7])[:2]))
        if d.qpos[2] < 0.2 or tilt > 45:
            fell = t; break
        if t > 2.5:
            _, ch_dq = ch_state(np.array(d.qpos[7:15]), np.array(d.qvel[6:14]))
            walk_vel_pk_max = max(walk_vel_pk_max, float(np.max(np.abs(np.rad2deg(ch_dq)))))
        if t >= 3.0 and (k % max(1,int(0.005/dt)) == 0):
            snaps.append((t, d.qpos.copy(), d.qvel.copy()))
    if fell:
        print(f"walk 자체가 낙상 t={fell:.2f}s — 재현 불가"); return
    print(f"Phase A: walk 완주 · 스냅샷 {len(snaps)}개 · walk 중 채널속도 max {walk_vel_pk_max:.0f} dps"
          f" ({'900 초과 있음(디바운스 이월 가능)' if walk_vel_pk_max>900 else '900 미만 — 이월 없음, 유예는 전체 20ms'})")

    d2 = mujoco.MjData(m)
    results = []
    for (t0, qp, qv) in snaps:
        d2.qpos[:] = qp; d2.qvel[:] = qv
        d2.ctrl[:] = 0; d2.time = 0
        mujoco.mj_forward(m, d2)
        qj = np.array(d2.qpos[7:15]); dqj = np.array(d2.qvel[6:14])
        ch_latch, ch_dq0 = ch_state(qj, dqj)
        v0 = float(np.max(np.abs(np.rad2deg(ch_dq0))))
        # 롤아웃 120ms — 500Hz(2ms) 마다 트립판정 샘플
        run_v = 0.0; run_v_max = 0.0
        run_t = 0.0; run_t_max = 0.0
        t_below = None
        n = int(0.120/dt)
        vel_hist = []
        for i in range(n):
            qj = np.array(d2.qpos[7:15]); dqj = np.array(d2.qvel[6:14])
            u, tau_drv = pd_ctrl(m, ch_latch, qj, dqj)
            d2.ctrl[:] = u
            mujoco.mj_step(m, d2)
            qj = np.array(d2.qpos[7:15]); dqj = np.array(d2.qvel[6:14])
            _, ch_dq = ch_state(qj, dqj)
            vel_pk = float(np.max(np.abs(np.rad2deg(ch_dq))))
            tau_pk = float(np.max(np.abs(tau_drv)))
            vel_hist.append(vel_pk)
            if vel_pk > VEL_TRIP: run_v += dt*1000
            else:
                if t_below is None and run_v > 0: t_below = (i+1)*dt*1000
                run_v = 0.0
            run_v_max = max(run_v_max, run_v)
            if tau_pk > TAU_TRIP: run_t += dt*1000
            else: run_t = 0.0
            run_t_max = max(run_t_max, run_t)
        vel_trip = run_v_max >= VEL_MS
        tau_trip = run_t_max >= TAU_MS
        results.append(dict(t0=t0, v0=v0, run_v=run_v_max, run_t=run_t_max,
                            vel_trip=vel_trip, tau_trip=tau_trip))
    n_over  = sum(1 for r in results if r['v0'] > VEL_TRIP)
    n_vtrip = sum(1 for r in results if r['vel_trip'])
    n_ttrip = sum(1 for r in results if r['tau_trip'])
    print(f"\nPhase B: 전환점 {len(results)}개 (5ms 간격, 3보행주기)")
    print(f"  전환 순간 vel_pk>200dps      : {n_over}/{len(results)} ({100*n_over/len(results):.0f}%)")
    print(f"  vel 트립(200dps 연속>=20ms)  : {n_vtrip}/{len(results)} ({100*n_vtrip/len(results):.0f}%)")
    print(f"  tau 트립(15Nm 연속>=50ms)    : {n_ttrip}/{len(results)}")
    worst = sorted(results, key=lambda r: -r['run_v'])[:8]
    print("\n  최악 전환점 (연속초과 상위):")
    print("   t0[s]   v0[dps]  vel연속초과[ms]  tau연속초과[ms]  판정")
    for r in worst:
        print(f"   {r['t0']:.3f}  {r['v0']:7.0f}  {r['run_v']:12.0f}  {r['run_t']:14.0f}"
              f"   {'VEL-TRIP' if r['vel_trip'] else ''}{' TAU-TRIP' if r['tau_trip'] else ''}")
    # 감속 프로파일 하나 출력(최악점)
    print(f"\n  판정 vel 트립율 = {100*n_vtrip/len(results):.0f}%  (전환점 무작위 가정)")

if __name__ == '__main__':
    main()
