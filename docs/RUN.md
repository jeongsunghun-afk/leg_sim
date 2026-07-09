# 02_Leg 실행 레시피

배포 = **17-DOF** (`quad_mpc_wbic_17dof` 허리 능동 + C++ 포팅). 물리 = convex MPC(GRF) + WBIC + 반응형 발배치.
파라미터 값 전체=[params.html](params.html) · 파이프라인=[pipeline.html](pipeline.html) · 실기 갭=[sim2real_checklist_17dof.html](sim2real_checklist_17dof.html) · 회귀검증=[MAINTENANCE.md](MAINTENANCE.md).

```
PXI=/home/jsh/miniforge3/envs/proxddp/bin/python   작업폴더: cd simulation/quad
GitHub: jeongsunghun-afk/simulation (master)
```

## 배포 물리 config (실모터, 항상)
```
CFG="MOTOR_CURVE=1 VEL_LIM=1 GEARBOX=1 GEAR_FOOT=0.5714"
```
- 기어 hip7/thigh7/calf10.5/foot8 → peak 84/84/126/**96**Nm · ω 29.6/29.6/19.7/25.9. foot만 8:1(GEAR_FOOT). 허리 7:1.
- GEARBOX=반사관성(발 flail 억제·필수). ROTOR_I=1e-4·JDAMP=0.1·JFRIC=0.5 = ★대략값, 실측 대기(sim2real 체크리스트).
- **★17dof 게인은 C++가 허리모델 자동감지로 기본 적용**(w_ori20·W_AM12·KD_AM24·FRONT_ANKLE−0.5·base_z0 0.5234) → env 불요. Python도 동일 기본.

## Python 배포 (GUI 텔레옵)
```bash
# 터미널1 (GUI)      cd simulation/quad && DISPLAY=:0 PXI teleop_gui_17dof.py
# 터미널2 (컨트롤러)  cd simulation/quad && env $CFG CMDFILE=/tmp/quad_cmd.json STATE_PUB=/tmp/quad_state.json \
                     PXI quad_mpc_wbic_17dof.py --robot ours_17dof_waist_sphere --mode trot
```
- 버튼: 전원(off)→눕기→앉기(haunch)→서기→보행. gait walk/trot/run 토글(속도·발높이 자동세팅).
- 조작(마우스1): 좌스틱 위 우클릭=고정(전진 유지)→"허리 핸들"로 조향(±68°, 오른쪽=우선회). 재우클릭/X=해제.
- 지형 씬: `MJCF=quad_terrain_course.mjcf`(또는 stairs/rough/friction) 추가 → perceptive 자동 ON.

## C++ 배포 (1kHz 실시간, cpp/)
```bash
cd simulation/quad/cpp && cmake -S . -B build && cmake --build build     # 빌드
# 헤드리스: GAIT=walk|trot|run  TROT_V  STEPS  (조향 TROT_STEER=δ · 선회 TROT_WZ)
GAIT=walk TROT_V=0.5 GEAR_FOOT=0.5714 STEPS=8000 ./build/trot_sim ../quad_real_17dof_waist_sphere.mjcf
# GLFW 뷰어 (★setsid 필수=SIGURG 회피 · 게인 자동감지라 env 최소)
export DISPLAY=:0
setsid bash -c 'cd simulation/quad/cpp; env GEAR_FOOT=0.5714 RATE=1.0 CMDFILE=/tmp/quad_cmd.json STATE_PUB=/tmp/quad_state.json \
  ./build/trot_view ../quad_real_17dof_waist_sphere.mjcf' </dev/null &
# 지형: 마지막 인자를 ../quad_terrain_course.mjcf 등으로 교체
```

## 회귀검증 / 데모
```bash
cd simulation/quad/cpp && ./verify.sh            # 표준 회귀 배터리(평지+지형, PASS/FAIL)
bash simulation/quad/record_demo.sh              # GUI+뷰어 NVENC 녹화(사용자 터미널서, 상세=RECORDING.md)
```

## 게이트·모드 요약
- **gait**: walk(순차 3~4발지지·정적안정·~0.6) / trot(대각 2지지·~1.4) / run(고속trot T0.40·~2.0).
- **선회 2채널**: TROT_WZ=yaw-rate 직접(제자리 스핀) / TROT_STEER=자동차식 δ(Ackermann, 전진해야 조향·허리 lean). yaw-rate 캡 0.9(understeer).
- **자세**: off(damp) / stand_down(눕기, wbic 저크라우치0.29) / sit(haunch 주저앉기) / stand_up(기립: 앉기→gather 궤적 추종). 상세=메모리.
- **perceptive**: PERCEPTIVE(기본 on) 계단/험지 착지·몸통높이 적응. 평지 무영향.

## 연구용(비배포) · 14-DOF
- 14-DOF: `env $CFG GEAR_CALF=1.0 TROT_V=1.0 PXI quad_mpc_wbic.py --robot ours_sphere --mode trot` (14dof는 calf 기본 8:1이라 GEAR_CALF=1.0로 10.5 복원). GUI=teleop_gui.py.
- 구조 B/C(FullDynamics·Centroidal, marginal): `simple_mpc/` (심링크). 반응형 발배치 거부라 본선 아님.
- 파일: A=quad_mpc_wbic.py(14) · **A'=quad_mpc_wbic_17dof.py + teleop_gui_17dof.py(배포)** · cpp/(C++) · biped/(초기 phase).
