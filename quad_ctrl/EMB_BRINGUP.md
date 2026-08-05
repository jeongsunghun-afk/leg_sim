# Emb(Pi) 브링업 가이드 — quad_ctrl 3단계 하드웨어 배선

> 이 문서를 인수하는 Claude(=Emb Claude)에게. 너의 역할 = **검증된 quad_ctrl 컨트롤러를 실기(Pi, RobotSharedMem)에 붙이는 하드웨어 배선**. 데스크톱 sim은 이미 검증 완료(아래) — 너는 HAL/관절맵/teleop만 채운다.

## ★★★핵심 불변 규칙 (먼저 읽어라)
1. **컨트롤러·estimator는 sim에서 검증됐고 불변이다.** `control/trot_bridge.hpp`(TrotCtrl wrap)·`estimator/ekf_estimator.hpp`(KF)·`quad/cpp/src/*`(WBIC/MPC)를 **고치지 마라**. 네가 채우는 건 오직: `hal/real_hal.hpp`의 read/write SHM 세부·`config/joint_map_17dof.hpp`(실측값)·`app/robot_main.cpp`의 teleop/RT.
2. **sim 회귀를 깨지 마라.** `quad_ctrl/verify.sh`가 9/9 PASS여야 한다(sim_bridge==trot_sim bit-동등). 데스크톱 빌드도 유지(real_hal/robot_main은 Pi 가드).
3. **비물리 sim2real 수단 금지**(상태 강제절단 등). 한계는 캡으로 숨기지 말고 정직히 드러내라.
4. **안전 우선**: 축별 JOG(저위험) → stand → walk 순. 모터 단품 → 다리 장착 순. tx_enabled(상태 100프레임 후)·통신두절 limp 이미 배선됨.

## 이미 확정된 것 (믿고 시작)
- **순수토크 가능(실기 검증)**: 드라이버가 Kp=Kd=0·fTorque 수용(0.45Nm ~1% 오차). → 컨트롤러 A(순수토크 tau_ff) 직결. real_hal write()가 곧 배포경로.
- **PACE 실측 액추에이터**(biped/emb/pace/RESULTS.md, HL/HR_hip N=7 모터단품): ROTOR_I=7.4e-4·JFRIC=0.38·JDAMP=0.096·왕복지연 T_rt=8.39ms. `config/deploy_17dof.yaml`에 반영됨. ⚠다리 미장착=낙관 하한, 다리장착 재측정 필요.
- **지연 예산 ~12ms > 실측 8.4ms** = 여유. sim서 실측물리+지연으로 walk/trot/run falls=0 확인.
- **확정 SHM API**(RobotTestGait): read=`RobotMemGait_{IsUpdatedMotorStatus16,GetMotorStatus16,IsUpdatedIMU,GetIMU}` · write=`RobotMemGait_SetMotorCommand16`(ch별) · 제어법칙 τ=Kp·(fPos−q)[rad]+Kd·(fVel−q̇)+fTorque·torque_frame=joint·단위 deg.

## 파일 지도 (네가 손댈 곳)
- `hal/real_hal.hpp` — read/write가 확정 SHM API로 완성됨. **TODO(코드에 명시)**: IMU gyro 인덱스(`IDX_OF_IMU_GYRO` 등 실 상수)·RPY→quat convention 부호/순서·accel 프레임(KF는 world 기대)·foot_force 접촉 소스.
- `config/joint_map_17dof.hpp` — 17-DOF `GaitJointCfg{chan,sign,zero,min,max,vel}` placeholder. **축별 JOG로 실측 채우기**(MJCF순서 HL·HR·waist·FL·FR).
- `app/robot_main.cpp` — RT 1kHz 루프(read→est→ctrl→write). **TODO**: HighCmd를 cmd_vel/조이스틱서 수신(현재 Stand 고정)·SCHED_FIFO/mlockall.
- `CMakeLists.txt` — robot_main 타깃(Pi 가드). **Pi 빌드 시 ENV_PREFIX(또는 MUJOCO/EIQP 경로)를 Pi의 MuJoCo(ARM)로 지정**.

## 참조
- `~/문서/jsh/RobotTestGait/src/main.cpp` — SHM read/write·IMU·RT timer·MotGeneral_t·tx_enable 실사용 패턴(수정 금지, 참조만).
- `biped/emb/` — 같은 SHM을 Python+C브리지로 이미 검증(관절맵·JOG·mock 절차의 실증 레퍼런스).
- `biped/emb/pace/` — 액추에이터 실측 도구(마찰·지연·PACE 식별). quad 다리장착 재측정에 재사용.
- `/usr/include/RobotSharedMem.h`(Pi) — 실 SDK 상수(IDX_OF_IMU_*·LEN_OF_IMU_DATA·ENUM_RESULT_*·MotorParam16_t·MAX_GaitMOT_CHAN).

## Pi 빌드
```bash
cmake -S quad_ctrl -B quad_ctrl/build -DENV_PREFIX=<Pi mujoco/eiquadprog prefix>
cmake --build quad_ctrl/build --target robot_main    # RobotSharedMem.h 있으니 활성
quad_ctrl/build/robot_main <pi>/simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf
```

## 순서 TODO (하드웨어)
1. **컴파일 정리**: real_hal의 IMU gyro 인덱스·convention을 `/usr/include/RobotSharedMem.h` 실 상수로 확정.
2. **축별 JOG**: joint_map placeholder(chan·sign·zero·min/max/vel)를 각 축 +명령→실회전 확인으로 실측. sign/zero 먼저, 한계는 보수적으로.
3. **stand**: wbic_stance로 서기(실모터). tilt·드리프트 확인.
4. **walk**: 저속부터. sim 대비 거동·falls·tilt 비교. 지연 실측(act_measure_latency 재사용)해 <12ms 확인.
5. **다리장착 재측정**: PACE로 마찰/관성 재측정 → config 갱신.
2차: 고토크 토크추종 정확도·teleop(cmd_vel) 배선·SCHED_FIFO.

## 경계·협업
- 브랜치 `fix/biped-quad-meshes`, 커밋 여기로(데스크톱 세션과 공유 — 서로 다른 파일: Pi=하드웨어 배선·데스크톱=sim). 커밋 메시지 끝에 Co-Authored-By.
- 막히면(SHM 상수·JOG 부호·빌드) 데스크톱 세션(sim 검증 담당)에 물어라. 상세 이관 명세=`quad_ctrl/MIGRATION.md`.

## 상태 요약
데스크톱: 이관 1/2/2b/3·3원칙·config·verify 9/9·실측파라미터 보행검증·순수토크 확정 = **완결**. 남은 건 위 하드웨어 배선뿐. 검증된 A가 그대로 실모터에 붙는다.
