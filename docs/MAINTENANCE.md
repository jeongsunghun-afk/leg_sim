# 02_Leg 제어기 유지·품질 프로세스

이 파이프라인(C++ 배포 + Python 참조 + 문서)을 **일정한 품질로 유지**하기 위한 프로세스.
3계층: **하네스**(자동 회귀) · **스킬**(에이전트 검수 SOP) · **이 문서**(사람 기준).

## 1. 품질 게이트 — 하네스 `quad/cpp/verify.sh`

변경 후 **반드시** 회귀 배터리 통과 확인. 한 줄:
```bash
cd simulation/quad/cpp
./verify.sh              # C++ 회귀 (~10초)
./verify.sh --python     # + Python 파리티 스팟(느림; 게인/파라미터 바꿨을 때)
```
빌드 → 평지(walk/trot/run) + 지형(course, perceptive) 검증 → `✅ ALL PASS`(exit 0) / `❌ REGRESSION`(exit 1).

### 판정 임계값 (회귀만 잡는 여유값 · 물리 캡 아님)
| config | falls | max_tilt ≤ | 전진 x ≥ | 정상 실측 |
|---|---|---|---|---|
| 평지 walk v0.6 | **0** | 8° | 2.0 | tilt 1.7·x 2.5 |
| 평지 trot v1.2 | **0** | 8° | 3.5 | tilt 1.4·x 4.6 |
| 평지 run v2.0 | **0** | 12° | 5.0 | tilt 1.2·x 6.3 |
| course walk v0.5 (perceptive) | **0** | 10° | 13.0 | tilt 3.7·x 13.9 |

- `falls=0`은 하드. tilt/x는 여유값 → 정상 변동엔 통과, 회귀(falls 발생·tilt 급증·조기정지)만 ❌.
- 정상 변화로 자주 ❌면 이 표 + `verify.sh`의 인자를 함께 조정(둘을 항상 일치).

## 2. 에이전트 검수 SOP — 스킬 `/controller-review`

`.claude/skills/controller-review/SKILL.md`. Claude가 변경 검수 시 매번 동일 절차:
회귀(verify.sh) → 실패시 근본원인(비물리 캡 금지) → Python↔C++ 파리티 → 문서 동기화 → 논리단위 커밋 → 보고.
- 리포 내 스킬이라 **simulation을 프로젝트 루트로 열면** `/controller-review`로 호출. 상위(`/home/jsh/문서/jsh`)에서 작업하면 그쪽 `.claude/skills/`로 복사/심볼릭 필요.

## 3. 파리티 규칙 (Python ↔ C++)

- **C++가 배포 기준.** 값 상충 시 C++/canonical 실행코드를 정본.
- C++는 17dof(허리)모델 자동감지 → 게인(w_ori20·W_AM12·KD_AM24·FRONT_ANKLE−0.5·base_z0 0.5234)을 기본 적용 = Python 기본과 정합(env 불요).
- 남은 의도적 차이(perceptive 몸통높이 샘플·STANCE_KD·gallop·walk foot-lock)는 `docs/params.html` 파리티 표에 명시. 새 divergence 생기면 표 갱신 또는 포팅.

## 4. 문서 역할 분리 (중복 금지)

| 문서 | 역할 | 갱신 트리거 |
|---|---|---|
| `quad/PARAMS.md` · `docs/params.html` | 파라미터 값 **전체** + 파리티 + 튕김조절 | 값·게인·게이트·발목·perceptive knob 변경 |
| `docs/pipeline.html` | MPC+WBIC **수식·구조·I/O** | 파이프라인·솔버 문제 변경 |
| `docs/sim2real_checklist_17dof.html` | 실기 이식 **갭**(값은 params로 위임) | 액추에이터 물리·미모델·운용한계 변경 |
| 메모리(`~/.claude/.../memory`) | 프로젝트 방향·진단·로드맵 | 결정·통찰·상태 변화 |

값은 params에만 두고 다른 문서는 **포인터**로 위임 → 중복·불일치 방지.

## 5. 커밋 관례
- **논리 단위 분리**(제어튜닝 / walk / 상태발행 / 지형 / 문서 / GUI …), 각 1의도.
- 메시지 끝: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **커밋/푸쉬는 요청 시에만.** 무관 파일(`quad_centroidal.py`·URDF·png) 스테이징 금지.

## 6. 원칙
- **비물리 sim2real 수단 금지** — 상태절단·VEL_CLIP 등으로 한계를 숨기지 말 것. 물리적 수단(VEL_LIM·MOTOR_CURVE·기어·GEARBOX)만. 한계 못 지키면 정직히 드러낼 것.
- 아티팩트 재배포: `docs/*.html` 편집 → `Artifact` 도구로 각 URL에 갱신(params/pipeline은 고정 URL).
