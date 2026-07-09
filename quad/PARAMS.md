# 02_Leg 17-DOF C++ 제어기 — 파라미터 레퍼런스

`cpp/src/` 기준. env 오버라이드: **S**=trot_sim.cpp, **V**=trot_view.cpp, **A**=apply_env_gains(trot_controller.hpp).
실사용값은 멤버 기본값이며 env로 덮어쓸 수 있음. 뷰어(GUI)는 CMDFILE(JSON)로 일부를 런타임 갱신.

---

## ★ 자주 쓰는 튜닝 knob (요약)

| knob | 기본값 | env | 역할 |
|---|---|---|---|
| `REAR_ANKLE` | **−0.3** | REAR_ANKLE | 뒷발목 각(축별 τ·ω 최적. 발목 ω 병목 해소) |
| `FRONT_ANKLE` | −0.7 | FRONT_ANKLE | 앞발목 각 |
| `base_z0` / `body_h` | 0.5234 | BASE_Z0 / BODY_H | 서기 몸통 높이 |
| gait | trot | GAIT / GUI | trot·walk·run·gallop 프리셋 |
| `V` `VY` `WZ` | 0.3·0·0 | TROT_V·VY·WZ | 전진·측방·선회 명령 |
| `step_h` | 0.10 | TROT_STEPH / GUI | 스텝 발 높이(프리셋 위 override) |
| `raibert_k` | 0.5 | RAIBERT_K / GUI | 전방 reach(0.5=표준 중립) |
| `steer` | 0 | TROT_STEER / GUI | 자동차식 조향각 δ(허리 핸들) |
| GEARBOX | ON | GEARBOX=0 끔 | 반사관성·감쇠·마찰(발 flail 억제) |

---

## 기립자세 / posture

| 파라미터 | 기본값 | env | 의미 | 위치 |
|---|---|---|---|---|
| base_z0 | 0.52(17dof 런치 0.5234) | BASE_Z0 (A) | 서기 몸통 높이 | quad_control.hpp:27 |
| REAR_ANKLE | −0.3 | REAR_ANKLE (A) | 뒷발목 각. 스윕 최적(발목 ω 155%→100%·calf τ76%·falls0). REAR_ANKLE 지정 시 FRONT도 동일화 | quad_control.hpp:27 |
| FRONT_ANKLE | −0.7 | FRONT_ANKLE (A) | 앞발목 각 | quad_control.hpp:27 |
| body_h | 0.5234 | BODY_H (S) / GUI | 서기 높이 슬라이더(q_home 목표) | trot_controller.hpp:46 |

`crouch_home`(quad_control.hpp:114): z=base_z0서 발 XY 유지·무릎 굽힘 IK → q_home·com_ref·foot_hip_off·foot_gz0. 발목은 R/FRONT_ANKLE로 고정 후 hip/thigh/calf만 IK.

## 게이트 / gait — set_gait 프리셋 (trot_controller.hpp:96–99)

| 게이트 | gp_T | gp_SWF | off[HL,HR,FL,FR] | raibert_k | step_h |
|---|---|---|---|---|---|
| walk | 0.7 | 0.25 | 0.25,0.75,0.5,0 | 0.5 | 0.10 |
| trot(기본) | 0.5 | 0.5 | 0,0.5,0.5,0 | 0.5 | 0.10 |
| run(고속) | 0.40 | 0.5 | 0,0.5,0.5,0 | 0.5 | 0.08 |
| gallop | 0.35 | 0.55 | 0,0.05,0.55,0.5 | 0.8 | 0.10 |

override: `GAIT`(S)·`TROT_T`/`TROT_SWF`/`TROT_STEPH`/`RAIBERT_K`(S, 프리셋 뒤 적용). config 상수 TC_KCAP=0.16(Raibert 피드백)·TC_RAICLIP=0.25(발배치 클립)·TC_WARMUP=0.6·TC_ACC=0.6(명령 슬루).

## 발배치 · whip

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| auto_whip | true | AUTO_WHIP=0 / GUI | 속도트리거 whip on/off |
| whip_v0 / whip_v1 | 0.8 / 1.6 | — | whip 선형증가 속도구간 |
| whip_hi | 2.0 | — | 저속(매끈) swing_w |
| whip_lo_f / whip_lo_r | 0.1 / 0.6 | SWING_W_F / SWING_W_R (S,V) | 앞/뒤 고속 whip 목표(작을수록 강함) |
| POS_HOLD | true | POS_HOLD=0 / GUI | 정지 위치홀드 |
| ALIP | false | ALIP=0 | 각운동량 발배치 보정(무효 확인) |

## WBIC 게인

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| w_ori | 5.0(17dof 20) | W_ORI (A) | 자세(roll/pitch) task 가중 |
| w_yaw | 0.0 | W_YAW (A) | yaw 홀드(0=MPC 담당, 최적) |
| W_AM / KD_AM | 0.0 / 8.0 | W_AM / KD_AM (A) | 각운동량 보상 가중 / 감쇠 |
| STANCE_KD | 20.0 | STANCE_KD (S) | stance 접촉속도 감쇠(slip↓) |
| MU | 0.6 | MU (A) | 마찰콘 μ(MPC 동시) |
| motor_curve | false | MOTOR_CURVE (A) | 토크-속도 곡선 |

하드코딩 task PD(참고): stance CoM (120,120,200)/(20,20,25)·자세 150/20(w5)·posture 60/5. swing 2400/110(w90)·z 200/25(w150).

## MPC (Di Carlo SRBD, quad_control.hpp:243–247)

N=14, dt=0.02. Q대각(roll,pitch,yaw/px,py,pz/ω/v/g)=[200,200,100, 0,0,200, 0,0,1, 10,10,1, 0]. R=1e-6(per-foot GRF). μ=MU·0.707. λz∈[1, 2·mg].

## 기어박스 · 모터물리

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| gear[4] hip,thigh,calf,foot | 7.0, 7.0, 10.5, 14.0 | — | 감속비 N. ★허리(FB_waist)는 미매칭→gi=0=hip **7:1**(실제와 일치, 의도됨) |
| GEARBOX | ON | GEARBOX=0 끔 | 반사관성 armature=Irot·N² + 감쇠 + 마찰 |
| Irot | 1e-4 | ROTOR_I | 로터 관성(★실측 대기) |
| jdmp / jfrc | 0.1 / 0.5 | JDAMP / JFRIC | 기어 점성감쇠 / 마찰(★실측 대기) |
| gmul | 1.0 | GEAR_HIP/THIGH/CALF/FOOT | 재기어 배율. ★배포=GEAR_FOOT=0.5714(foot 14→8:1, peak168→96·ω→25.9) |
| w_limit | 207/(gear·gmul) | (파생) | 무부하 속도한계(MOTOR_CURVE용) |

모터 peak: hip/thigh 84, calf 126, **foot 96(8:1 재기어)** Nm. 상세=메모리 02leg-motor-spec.

## 조향 / steering

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| steer(δ) | 0 | TROT_STEER (S,V) / GUI | 자동차식 조향각. Ackermann R=축거/tanδ |
| wheelbase | ≈0.61(런타임) | — | 축거 L(앞뒤 힙 x거리) |
| yaw-rate 캡 | ±0.9 | — | understeer(고속 tight turn 낙상방지) |

## 허리 / waist (17-DOF)

| 파라미터 | 기본값 | env | 의미 |
|---|---|---|---|
| WAIST_W | 80.0 | WAIST_W (A) | 허리 홀드 가중 |
| WAIST_KP / WAIST_KD | 150 / 20 | WAIST_KP / WAIST_KD (A) | 허리 강홀드 PD |
| waist_steer | 0.4 | WAIST_STEER (S,V) | 허리 lean 보조 게인(핸들만 구동) |
| waist_cap | 0.20 | — | 허리 lean 캡 ±0.20rad(≈11°) |

## 앉기 · 기립 / sit · getup

**개-앉기(haunch) 시드** (quad_control.hpp:28–29): HAUNCH_THIGH −1.0·CALF 1.2·FOOT −0.3·HOCK_Z 0.019·FRONT_REACH −0.22·FOOT_LAND −1.2 (모두 S,V env).
**스케줄** (trot_controller.hpp): HAUNCH_Z 0.30·FOLD_RATE 0.60·UNFOLD_Z 0.40·SIT_POSTURE_W 40·HAUNCH_PITCH 0.50·SIT_KP 90·SIT_SLEW 0.6. 기립궤적 추종 GETUP_TRAJ_KP 120·KD 4(getup_dt 0.01).
**스크립트 기립(SGU_*)**: KICK_T 0.5·FB_THIGH −0.55·FB_CALF 1.20·SLEW 1.5·KP 120·GATHER_Z 0.24·DONE_TILT 22·WALKOUT_V 0.6·HANDOFF_Z 0.34 (모두 S env).
**모드 상태**: GROUND_Z 0.18·GETUP_TRIG 0.32·GETUP_DONE 0.40·JOINT_SLEW 1.5·HRATE 0.3.

## 실행 / 진단 env

| env | 기본 | 의미 |
|---|---|---|
| MODE | move | move/stand_up/stand_down/sit/off (S) |
| MODE2 + SWITCH_T | — | t>SWITCH_T서 1회 모드전환(getup 검증, S) |
| STEPS | 3000 | 헤드리스 스텝 수 (S, argv2) |
| RATE / CMDFILE / STATE_PUB | 1.0 / — / /tmp/quad_state.json | 뷰어 배속·GUI 명령·상태발행 (V) |
| JSTAT | — | 관절별 τ·ω peak/RMS(정착후) 출력 (S) |
| QHDBG·SITDBG·SLIPLOG·GRFLOG·DUMP_QPOS | — | 진단 로깅 (S) |

---

## 테스트 지형 씬 (make_terrains.py 생성)

`<include>`로 로봇 재사용 + 지형만 얹음. 로봇은 원점서 +x 전진 → 지형 진입.

| 씬 | 내용 | walk v0.5 결과 |
|---|---|---|
| quad_terrain_stairs.mjcf | 오름6·랜딩·내림6 계단(rise 4cm·depth 28cm) | 계단 올라탐 z0.55·tilt3.0°·falls0 |
| quad_terrain_rough.mjcf | 불규칙 높이 블록필드(84개, h 1~10cm) | tilt1.4°·falls0 |
| quad_terrain_friction.mjcf | ice(μ0.2)·보통(μ1.3)·고마찰(μ2.5) 레인 | ice 통과·tilt1.5°·falls0 |

실행: `GEAR_FOOT=0.5714 ./cpp/build/trot_view quad_terrain_stairs.mjcf` (또는 GUI 연동 시 CMDFILE 추가).
지형 조정: make_terrains.py 편집 후 재생성.
