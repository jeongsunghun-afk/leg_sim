# biped — 뒷다리 2족 보행 (MPC + WBIC, event-DCM)

quad 17-DOF 뒷다리(HL/HR)를 추출한 8-DOF biped를 **성숙 MPC+WBIC 스택**으로 제어.
점 발 동적 균형(capture-point)·event-based DCM 게이트·base-frame 발배치·heading-hold.
현황: **저속 양방향(±0.25 m/s) 25s+ 무낙상 직진 보행 + 실시간 GUI 조종.**

## 실행

```bash
# GUI 조종 (뷰어 + 조종 GUI 동시) — ★사용자 터미널에서
./run_gui_biped.sh

# 단독 뷰어 (낙상 자동리셋)
python3 biped_view.py            # 제자리
VX=0.15 python3 biped_view.py    # 저속 보행

# 헤드리스 (튜닝·검증)
VIEW=0 T=25 python3 biped_run.py

# OBS 녹화와 함께
./record_biped.sh                # GUI + OBS 동시 실행
```
GUI: 조이스틱 위/아래=전진/후진(저속·우클릭 고정) · 몸통높이(crouch) 슬라이더 · Stand/Walk/STOP.
env: `proxddp`(dearpygui) — 런처가 자동 사용.

## 파일 (활성)
| 파일 | 역할 |
|---|---|
| `biped_from_quad.mjcf` | 모델 (quad 뒷다리 + 최소몸통, sphere발, GEARBOX, 바닥강성) |
| `biped_wbic.py` | WBIC(양발 stance) + GEARBOX + 게인·상수 |
| `biped_step.py` | event-DCM 게이트 + **base-frame 발배치**(dcm_target) + swing 궤적 |
| `biped_mpc_wbic.py` | Di Carlo SRBD MPC(2발) + WBIC 추종 + heading-hold. **메인 컨트롤러** |
| `biped_run.py` | 실행기: 컨트롤러+뷰어 + JSON 명령채널 + 상태발행 + 실시간페이싱 |
| `biped_view.py` | 단독 뷰어(낙상 자동리셋) |
| `biped_ref_export.py` | RL 레퍼런스 궤적 생성기 → `ref_lib/` |
| `run_gui_biped.sh` · `record_biped.sh` | 원샷 런처 (GUI / GUI+OBS) |
| `teleop_gui_biped.py` | 슬림 GUI (dearpygui) — **`../quad/`에 위치** |

하위: `ref_lib/`(RL 레퍼런스 + 핸드오프) · `renders/`(시각화) · `archive/`(옛 biped 시도) · `meshes/`·`urdf/`(옛 자산).

## 제어 구조 (17-DOF quad와 동일 백본)
- **MPC**(50Hz, Di Carlo SRBD): 균형 GRF 계획. x_ref 속도=명령·yaw=heading목표(드리프트 보정).
- **WBIC**(500Hz): MPC GRF 추종 + 스윙발 + roll/pitch 레벨링 + posture. GEARBOX(반사관성).
- **게이트**: event-based(측방 DCM이 swing측 넘을 때 착지). 점 발이라 고정시간보다 안전.
- **발배치**: base-frame capture-point + SPREAD 벌림 + 스탠스기준 최소간격(다리 꼬임 차단).

상세 개발기록·튜닝 = 메모리 `biped-mpc-reimpl` · 전략 = `../docs/roadmap_hybrid.html`.
