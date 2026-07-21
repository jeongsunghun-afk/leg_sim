# 02_Leg 실행 레시피

배포 = **17-DOF** (`quad_mpc_wbic_17dof` 허리 능동 + C++ 포팅). 물리 = convex MPC(GRF) + WBIC + 반응형 발배치.
파라미터 값 전체=[params.html](params.html) · 파이프라인=[pipeline.html](pipeline.html) · 실기 갭=[sim2real_checklist_17dof.html](sim2real_checklist_17dof.html) · 회귀검증=[MAINTENANCE.md](MAINTENANCE.md).

```
# 경로 무관 실행용 절대경로 (아래 모든 레시피는 이 변수로 자기완결)
REPO=/home/jsh/문서/jsh/simulation
QUAD=$REPO/quad
PXI=/home/jsh/miniforge3/envs/proxddp/bin/python
GitHub: jeongsunghun-afk/simulation (master)
```
> 실행코드는 **어느 디렉토리에서도 그대로 동작**하도록 각 블록을 절대 `cd`로 시작하거나 절대경로로 명시함(상대 `../mjcf`·"먼저 cd" 가정 금지).

## 배포 물리 config (실모터, 항상)
```
CFG="MOTOR_CURVE=1 VEL_LIM=1 GEARBOX=1 GEAR_FOOT=0.5714"
```
- 기어 hip7/thigh7/calf10.5/foot8 → peak 84/84/126/**96**Nm · ω 29.6/29.6/19.7/25.9. foot만 8:1(GEAR_FOOT). 허리 7:1.
- GEARBOX=반사관성(발 flail 억제·필수). ROTOR_I=1e-4·JDAMP=0.1·JFRIC=0.5 = ★대략값, 실측 대기(sim2real 체크리스트).
- **★17dof 게인은 C++가 허리모델 자동감지로 기본 적용**(w_ori20·W_AM12·KD_AM24·FRONT_ANKLE−0.5·base_z0 0.5234) → env 불요. Python도 동일 기본.

## GUI 텔레옵 원샷 (권장) — 뷰어 + GUI 한번에
```bash
# 어디서든 실행 가능(절대 cd로 자기완결). QUAD=/home/jsh/문서/jsh/simulation/quad
(cd /home/jsh/문서/jsh/simulation/quad && bash run_gui.sh)        # ★C++ 실시간 배포(1kHz) — 기본 맵=종합코스
(cd /home/jsh/문서/jsh/simulation/quad && bash run_gui_py.sh)     # Python 레퍼런스(연구/디버그, 느림)
#   맵 선택: 두 스크립트 다 [course(기본)|flat|stairs|rough|friction|gap|stepping|soft] 인자
#   예: (cd /home/jsh/문서/jsh/simulation/quad && bash run_gui.sh flat)
```
- 종합코스 = **4라인 병렬**(2026-07-15 재구성): **y=+1.2 갭** / **y=−1.2 스테핑**(안쪽=도달 쉬움) · y=+3.6 마찰 / y=−3.6 완만험지. 조향해 라인 진입, 중앙(y=0)은 통로. ★**gap·stepping을 별도 라인에 분리**(같은 라인 연속은 간격 다른 정렬속도 요구라 불가) → 각 라인 **walk 0.5**로 건넘(FOOT_NUDGE).
- 두 스크립트 다 뷰어+GUI를 **CMDFILE/STATE_PUB 붙여** 띄움(GUI 명령·모니터 연동). ★이 env 누락 시 저절로 전진 → 반드시 스크립트로 실행.
- 버튼: 전원(off)→눕기→앉기(haunch)→서기→보행. gait walk/trot/run/stairs 토글(속도·발높이·base height 자동세팅).
- 조작(마우스1): 좌스틱 위 우클릭=고정(전진 유지)→"허리 핸들"로 조향(±68°, 오른쪽=우선회). 재우클릭/X=해제.
- perceptive 자동 ON(지형 씬). 평지만 원하면 `run_gui.sh flat`.
- **★강건 기립(2026-07-15)**: 앉기→서기 = **앉기→눕기→서기 라우팅**(haunch 직접상승은 측방 발산 → 대칭 저크라우치 거쳐 검증된 상승). 버튼 조작 동일.

## ★지형 crossing — ③ selectFoot (perceptive foothold, FOOT_NUDGE)
갭·스테핑을 건너려면 **footScore 기반 발판선택**을 켜야 함(perceptive 착지높이와 별개):
```bash
Q=/home/jsh/문서/jsh/simulation/quad
(cd $Q && FOOT_NUDGE=1 bash run_gui.sh course)     # 종합코스(레인3=갭→스테핑). 조향해 레인3(y≈+2.4)로 진입
(cd $Q && FOOT_NUDGE=1 bash run_gui.sh gap)        # 갭 전용(0.32m갭·0.70m발판)
(cd $Q && FOOT_NUDGE=1 bash run_gui.sh stepping)   # 스테핑스톤(0.44m간격)
```
- **★속도-공명 주의**: ③은 스텝길이가 발판간격에 **정렬된 속도에서만** 건넘. 이 지형들=**walk 0.5**(GUI walk버튼 기본 0.6은 실패!). 속도게이지를 0.5로. 지형 간격 바뀌면 정렬속도도 바뀜(로컬 nudge 한계, 강건 크로싱은 RL 트랙).
- **★보행 선택(지형별 강건성)**: **이산 정밀 발판(갭·stepping stone)=walk만**(high duty factor·정밀 디딤, trot/run은 전복). **연속·완만(험지 요철·평지)=walk/trot/run 다 OK**(trot·run이 빠름). 4라인 코스: 갭·스테핑 라인=walk 0.5 / 험지·계단 라인=trot/run 가능.
- **★perceptive 몸통높이**: 연속지형(계단·험지)은 base가 지형 따라 정상 상승(계단 z0.49→0.79). 갭(불연속)만 base 중심이 구멍 위라 발판만큼 안 오름(안정적 절충, 4-hip 강제는 출렁여 불안정). base높이는 MPC에만 공급(WBIC 공급은 tilt 악화).
- 헤드리스 검증: `(cd /home/jsh/문서/jsh/simulation/quad/cpp && FOOT_NUDGE=1 GAIT=walk TROT_V=0.5 STEPS=18000 ./build/trot_sim /home/jsh/문서/jsh/simulation/quad/mjcf/quad_terrain_gap.mjcf)`(falls=0 완주).
- **★★발 접촉구 크기가 지형 강건성 지배 레버(2026-07-21)**: 발 sphere `0.018→0.024`(실물 foot_contact_link 51×29mm 반폭 25.5mm에 부합)로 **gap 크로싱이 전-속도(0.3~0.6) falls=0**으로 해결(구 18mm=점접촉 과소근사→발판 엣지 트립→gap 전멸). 24mm=gap전속도·stepping0.45~0.55, 36mm(과대)=일부 실패. verify.sh 4/4 무회귀. 상세=메모리 [[perceptive-nav-tamols]].

## 상태추정 (sim2real, EST) — GT는 개발용·EST 성공이 배포 기준
```bash
EST_CTRL=1 ...   # 폐루프: 컨트롤러가 leg-odometry 추정상태로 계산(실기 조건). 기본=선형 접촉KF(IMU가속도 융합)
#   EST_ANCHOR=1 이면 stance-anchored(비교/폴백) · 노이즈/지연: ENCQ_N·GYRO_N·ACC_N·SENSE_LAT_MS·ACT_LAT_MS
```
- 뷰어 '모니터 표시(viz)' 체크박스 = 추정 고스트(초록 GT/주황 EST) + footScore 오버레이.
- 검증: 보행·선회·기립·점프 폐루프 falls=0. 지연예산 ≤2ms/각(앉기 posture는 3ms서 느린 드리프트).

## C++ 헤드리스 / 빌드
```bash
CPP=/home/jsh/문서/jsh/simulation/quad/cpp; MJCF=/home/jsh/문서/jsh/simulation/quad/mjcf
cmake -S $CPP -B $CPP/build && cmake --build $CPP/build     # 빌드(어디서든)
# 헤드리스: GAIT=walk|trot|run  TROT_V  STEPS  (조향 TROT_STEER=δ · 선회 TROT_WZ) · ★모든 mjcf는 quad/mjcf/
(cd $CPP && GAIT=walk TROT_V=0.5 GEAR_FOOT=0.5714 STEPS=8000 ./build/trot_sim $MJCF/quad_real_17dof_waist_sphere.mjcf)
# 뷰어 기동은 run_gui.sh 사용(setsid+CMDFILE/STATE_PUB 자동). 맵=인자로 지정.
```

## 회귀검증
```bash
(cd /home/jsh/문서/jsh/simulation/quad/cpp && ./verify.sh)      # 표준 회귀 배터리(평지+지형, PASS/FAIL)
```

## 데모 녹화 (2가지)
```bash
# ── (A) ffmpeg NVENC 원샷 — 뷰어+GUI 자동 기동 + 화면녹화, [Enter]로 종료 ──
bash /home/jsh/문서/jsh/simulation/quad/record_demo.sh    # 기본=종합코스. MAP=mjcf/… 로 맵 지정. 상세=RECORDING.md
#   결과 ~/Videos/Screencasts/quad_<stamp>.mp4. 배속예) ffmpeg -i in.mp4 -filter:v "setpts=0.5*PTS" out_2x.mp4

# ── (B) OBS 병행 — 뷰어+GUI 띄운 뒤 OBS로 녹화(수동 제어, 권장: 긴 데모·화면구성 자유) ──
bash /home/jsh/문서/jsh/simulation/quad/run_gui.sh course  # 1) 뷰어+GUI(견고화: 렌더검증+재시도)
DISPLAY=:0 obs &                                 # 2) OBS 실행 → Start Recording
#   OBS 설정: Output=Hybrid MP4(또는 MP4 fragmented)·NVENC, Source=Screen Capture(XSHM)→Ctrl+F(Fit)→Alt드래그 crop. 상세=RECORDING.md
```
- **★반드시 "본인 터미널"에서 실행** — Claude 도구 세션서 띄우면 프로세스가 회수될 수 있음.
- 데모 시퀀스 예: Walk→Trot→Run(속도게이트)·허리조향(원주행)·Ready→Sit·Ground→Sit(직행)·Sit→Ready·Jump.

## 점프 (offline OCP 궤적 추종)
```bash
(cd /home/jsh/문서/jsh/simulation/quad && bash gen_jump.sh 0.6)   # 전방 점프 궤적 생성(VX=이륙속도, 0=수직 제자리)
#   J1 OCP(crocoddyl)→J2 변환→/tmp/jump_traj.txt. C++ 점프모드가 crouch→재생(thrust/flight)→wbic 착지.
#   ★/tmp 휘발성 → 점프 안 될 때 재생성. VX=0.6→~0.14m 전방, apex 0.28m, falls=0.
```
- ★**선회 후 점프**: 몸통 돌린 뒤 점프해도 스핀 없이 **바라보는 방향으로** 뜀. 궤적을 발사 시점 현재 CoM·헤딩 프레임으로 re-anchor + wbic_jump 자세는 roll/pitch만 레벨·yaw 자유(걷기 360°선회 수정과 동일 원리). 직진 점프는 회귀 없음.

## sim2real 상태추정 비교 (뷰어 GT↔추정)
```bash
# GUI '모니터 표시' 체크박스 = 추정 base 고스트 표시(개루프: 제어는 GT라 보행 안정 · 초록=GT/주황=추정 + 헤딩 화살표)
#   추정=leg-odometry(IMU자세+엔코더+접촉, stance발 정지가정). 정적모드(서기/앉기/눕기)=ZUPT로 표류 차단.
(cd /home/jsh/문서/jsh/simulation/quad/cpp && EST_CTRL=1 GYRO_N=0.02 QUAT_N=0.01 ENCQ_N=0.001 ENCDQ_N=0.01 ./build/trot_sim /home/jsh/문서/jsh/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf 6000)
#   ★EST_CTRL=폐루프(제어에 추정상태 사용, 실기 동일구조) — 헤드리스 특성화용. 배포속도 walk0.6/trot1.2 falls=0, run2.0은 전복(안전속도~1.1).
#   센서노이즈 env: GYRO_N[rad/s]·QUAT_N[rad]·ENCQ_N[rad]·ENCDQ_N[rad/s] (0=완벽센서).
#   센서/구동 지연 env: SENSE_LAT_MS(센서→추정/제어)·ACT_LAT_MS(제어→구동). 뷰어 HUD·헤드리스 [LATENCY] 로그.
```
- ★**지연 특성화**(EST_CTRL 폐루프, 배포노이즈): **총 루프 예산 ~12ms** — 센서=구동 합산 시 각 **6ms까지 falls=0**(7ms↑ 붕괴), 단일 채널만이면 ~10ms. 넘으면 게인 재튜닝(강성↓)/지연보상 필요. 뷰어는 개루프라 지연=고스트 lag만(보행 안정 유지).

## 게이트·모드 요약
- **gait**: walk(순차 3~4발지지·정적안정·~0.6) / trot(대각 2지지·~1.4) / run(고속trot T0.40·~2.0).
- **선회 2채널**: TROT_WZ=yaw-rate 직접(제자리 스핀) / TROT_STEER=자동차식 δ(Ackermann, 전진해야 조향·허리 lean). yaw-rate 캡 0.9(understeer).
- **자세**: off(damp) / stand_down(눕기, wbic 저크라우치0.29) / sit(haunch 주저앉기) / stand_up(기립: 앉기→gather 궤적 추종). 상세=메모리.
- **perceptive**: PERCEPTIVE(기본 on) 계단/험지 착지·몸통높이 적응. 평지 무영향.

## 연구용(비배포) · 14-DOF
- 14-DOF: `env $CFG GEAR_CALF=1.0 TROT_V=1.0 PXI research/quad_mpc_wbic.py --robot ours_sphere --mode trot` (14dof는 calf 기본 8:1이라 GEAR_CALF=1.0로 10.5 복원). GUI=research/teleop_gui.py.
- 구조 B/C(FullDynamics·Centroidal, marginal): `simple_mpc/` (심링크). 반응형 발배치 거부라 본선 아님.
- **CI-MPC 제2트랙(그래디언트 CI iLQR, offline 연구·검증)**: 접촉타이밍 최적화. C-2=완성해도 실시간 불가(offline 교사 도구). 상세=`pipeline_ci_mpc.html`·`params_ci_mpc.html`·메모리 ci-mpc-track.
```bash
# gap 크로싱(경로 무관 자기완결). GAIT=crawl|trot · ILQR_ITERS=3 최적 · MJCF_PATH·PXI 절대경로
(cd /home/jsh/문서/jsh/simulation/quad/ci_mpc && XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MJCF_PATH=/home/jsh/문서/jsh/simulation/quad/mjcf/ci_mpc_gap.mjcf DISABLE_FLOOR=1 \
  GAIT=crawl VX=0.2 NCTRL=320 NH=12 ILQR_ITERS=3 STEP_H=0.08 \
  /home/jsh/miniforge3/envs/proxddp/bin/python -c "import mjx_ilqr; mjx_ilqr._mpc_run()")
#   평지 프로파일: 위에서 MJCF_PATH·DISABLE_FLOOR 빼고 PROFILE=1 NCTRL=40 (build_ref/ilqr/sim 분해)
```
- 파일: A=quad_mpc_wbic.py(14) · **A'=quad_mpc_wbic_17dof.py + teleop_gui_17dof.py(배포)** · cpp/(C++).
- **biped/**(뒷다리 2족, 별도 서브프로젝트): 추정 폐루프(leg-odom+접촉높이)+지연보상 C++ 배포·GUI. 전진 로버스트·측방 약함·선회 사실상 불가(점발 한계→RL). ★실행 레시피는 `biped/README.md`(run_gui_biped.sh·run_gui_cpp.sh·deploy/). 상세=메모리 biped-mpc-reimpl.
