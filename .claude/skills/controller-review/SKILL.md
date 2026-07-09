---
name: controller-review
description: 02_Leg 사족 제어기(C++ quad_mpc_wbic_17dof / Python) 변경을 검수·회귀검증할 때 사용. verify.sh 회귀 배터리 실행, Python↔C++ 파리티 확인, 문서(params·pipeline·sim2real) 동기화 점검을 매번 같은 절차로 수행해 품질을 일정하게 유지한다. "검수/회귀/품질확인/파리티체크"류 요청, 또는 컨트롤러 게인·게이트·perceptive·발목·기어 등을 바꾼 직후에 호출.
---

# controller-review — 컨트롤러 품질 검수 SOP

02_Leg 제어기 변경 후 **매번 동일한 절차**로 회귀·파리티·문서를 검수한다. 목적=품질 일정 유지.
기준값·문서맵은 `simulation/docs/MAINTENANCE.md` 참조.

## 절차 (순서대로)

### 1. 회귀 배터리 (하네스)
```
cd simulation/quad/cpp && ./verify.sh          # C++ 회귀 (~10초)
cd simulation/quad/cpp && ./verify.sh --python # + Python 파리티 스팟(느림, 파라미터/게인 바꿨을 때)
```
- 결과 표(✅/❌ · falls · tilt · x)를 그대로 사용자에게 보고.
- **판정=falls 0(하드) · max_tilt ≤ 한계 · x ≥ 최소.** 하나라도 ❌면 회귀.

### 2. 실패(❌) 시
- 실패 config의 falls/tilt/x 수치를 명시.
- **캡·상태절단 같은 비물리 수단으로 숨기지 말 것**([[no-unphysical-sim2real-hacks]]). 근본원인(게인·발배치·타이밍·프레임)부터.
- 필요 시 변수 격리 A/B(env 하나만 바꿔 비교)로 원인 규명 후 수정→재검증.

### 3. Python ↔ C++ 파리티
- 변경이 튜닝 파라미터/게인/게이트/perceptive를 건드렸으면 **양쪽이 일치하는지** 확인.
- C++는 17dof(허리)모델 자동감지로 게인(w_ori20·W_AM12·KD_AM24·FRONT_ANKLE−0.5)을 기본 적용 → 보통 자동 정합.
- 새 divergence가 생기면 `docs/params.html`·`PARAMS.md`의 **파리티 표**에 반영하거나 한쪽에 포팅.
- **C++가 배포 기준.** 값 상충 시 C++/canonical 실행코드를 정본으로.

### 4. 문서 동기화 (변경 성격별)
| 바뀐 것 | 갱신할 문서 |
|---|---|
| 파라미터 값·게인·게이트·발목·perceptive knob | `quad/PARAMS.md` + `docs/params.html` |
| 파이프라인·솔버 구조·MPC/WBIC I/O | `docs/pipeline.html` |
| 액추에이터 물리·sim2real 갭·운용한계 | `docs/sim2real_checklist_17dof.html` |
| 프로젝트 방향·진단·로드맵 | 메모리([[biped-wbic-mpc-project]] 등) |

문서는 **역할 분리** 유지: params=값 전체 / pipeline=수식·구조 / sim2real=실기 갭(값 중복 금지, params로 위임).

### 5. 커밋
- **논리 단위로 분리**(제어튜닝 / walk / 상태발행 / 지형 / 문서 / GUI 등), 각 커밋 1의도.
- 커밋 메시지 끝: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **커밋/푸쉬는 사용자가 요청할 때만.** 무관 파일(quad_centroidal.py·URDF·png 등) 스테이징 금지.

### 6. 보고
- PASS/FAIL 표 + 파리티 상태 + 문서 동기화 필요분을 요약. 미룬 항목은 정직히 표기.

## 임계값(회귀 판정) — verify.sh 내장
| config | falls | tilt≤ | x≥ |
|---|---|---|---|
| 평지 walk v0.6 | 0 | 8° | 2.0 |
| 평지 trot v1.2 | 0 | 8° | 3.5 |
| 평지 run v2.0 | 0 | 12° | 5.0 |
| course walk v0.5 (perceptive) | 0 | 10° | 13.0 |

임계값은 정상동작 여유값(회귀=falls 발생·tilt 급증만 잡음). 물리 한계 캡 아님. 정상 변화로 자주 ❌면 MAINTENANCE.md에서 조정.
