# Ubuntu 22.04에서 GPU 가속 고화질 화면 녹화 가이드

## 1. 목적

Ubuntu 22.04 환경에서 **GPU(NVIDIA NVENC)를 사용해 고화질 영상으로 실험 화면을 기록하고 공유**하기 위한 방법을 정리한다.

- **녹화**: OBS Studio
- **후처리**: ffmpeg
- **주요 응용**: Gazebo + RViz 동시 실행 중 실험 영상 기록

---

## 2. 시스템 환경

본 가이드는 다음 환경에서 검증되었다.

| 항목 | 환경 |
| --- | --- |
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4050 |
| Encoder | NVENC |
| Resolution | 4K (3840 x 2160) |
| FPS | 30 |

Gazebo와 RViz를 동시에 실행하면서도 **실험 성능 저하 없이 녹화**가 가능하다.

---

## 3. OBS Studio 설치

```bash
sudo apt update
sudo apt install obs-studio
obs
```

---

## 4. OBS 초기 설정

OBS 최초 실행 시 **Auto Configuration Wizard**가 자동으로 뜬다.

**선택 옵션**: `Optimize just for recording`

스트리밍이 아닌 **로컬 녹화 전용 설정**이다.

---

## 5. Video 설정

`Settings → Video`

| 항목 | 값 |
| --- | --- |
| Base (Canvas) Resolution | 3840 x 2160 |
| Output Resolution | 3840 x 2160 |
| FPS | 30 |

**설명**

- 4K 모니터 사용 시 **4K 캔버스** 사용 권장
- Gazebo 실험 영상은 **30 FPS** 로 충분

---

## 6. Output 설정

`Settings → Output` → Recording 항목

| 항목 | 값 |
| --- | --- |
| Recording Path | `/home/<username>/Videos/Screencasts` |
| Recording Format | **Hybrid MP4** (`.mp4`) |
| Encoder | NVIDIA NVENC |
| Recording Quality | High Quality |

**MP4 바로 녹화 (mkv·remux 불필요)**

- **Recording Format 을 `Hybrid MP4` 로 선택**하면 처음부터 `.mp4` 로 녹화된다 → 별도 remux 단계 없음.
- Hybrid MP4 는 녹화 중 비정상 종료되어도 파일이 깨지지 않는다(기존 mp4의 finalize 손상 문제 해결). 즉 mkv 의 안전성 + mp4 의 호환성을 모두 가진다.
- OBS 30.1 이상에서 지원. 목록에 없으면 `MP4 (fragmented)` 를 선택해도 동일하게 직접 mp4·크래시 안전.
- (구버전에서 일반 `mp4` 만 있으면 크래시 손상 위험이 있으니 Hybrid/fragmented 를 우선 사용.)

---

## 7. 녹화 화면 설정

`Sources → +` 에서 다음을 선택한다.

```
Screen Capture (XSHM)
```

이후 화면을 다음 순서로 조정한다.

### 7-1. 화면 Scale (캔버스에 맞춤)

우클릭 메뉴:

```
Right Click → Transform → Fit to Screen
```

또는 단축키:

```
Ctrl + F
```

4K 화면을 OBS canvas에 맞게 스케일 조정한다.

### 7-2. 화면 Crop (불필요 영역 제거)

단축키:

```
Alt + Mouse Drag
```

**제거할 요소**

- Ubuntu dock
- 상단 시스템 바
- OBS UI

**최종 화면 예시**

```
+-----------------------------+
| Gazebo        | RViz        |
|               |             |
|               |             |
+-----------------------------+
```

---

## 8. Remux — 불필요 (Hybrid MP4 로 직접 녹화하므로 생략)

Section 6에서 **Hybrid MP4** 로 녹화하면 결과가 이미 `.mp4` 이므로 remux 단계가 필요 없다.

> 참고: 예전 방식(mkv 녹화)으로 만들어진 `.mkv` 파일이 남아 있다면 `File → Remux Recordings` 로 mp4 변환할 수 있다(재인코딩 없음, 화질 손실 없음). 신규 녹화에는 해당 없음.

---

## 9. ffmpeg 설치

```bash
sudo apt install ffmpeg
```

---

## 10. 영상 배속 처리

실험 영상은 종종 **배속 편집**이 필요하다.

### 2배속

```bash
ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" output.mp4
```

### 4배속

```bash
ffmpeg -i input.mp4 -filter:v "setpts=0.25*PTS" output.mp4
```

**참고 — setpts 값 규칙**

- `setpts=N*PTS`
- N < 1 이면 배속 (예: 0.5 = 2배속, 0.25 = 4배속)
- N > 1 이면 슬로우 (예: 2.0 = 0.5배속)

---

## 11. 편의를 위한 shell alias 함수

`~/.bashrc`에 다음 함수 추가:

```bash
faster2() {
    ffmpeg -i "$1" -filter:v "setpts=0.5*PTS" "$2"
}

faster4() {
    ffmpeg -i "$1" -filter:v "setpts=0.25*PTS" "$2"
}
```

적용:

```bash
source ~/.bashrc
```

사용 예시:

```bash
faster2 input.mp4 output.mp4
faster4 input.mp4 output.mp4
```

---

## 12. 권장 녹화 설정 요약

| 항목 | 설정 |
| --- | --- |
| Resolution | 4K (3840 x 2160) |
| FPS | 30 |
| Encoder | NVIDIA NVENC |
| Format | **Hybrid MP4 (.mp4 직접 녹화)** |
| Quality | High Quality |

---

## 13. 전체 파이프라인

```
Gazebo / RViz 실행
        ↓
OBS 녹화 (Hybrid MP4 → .mp4 직접)
        ↓
ffmpeg 배속 / 편집
        ↓
최종 영상
```

---

## 14. 트러블슈팅

- **NVENC가 목록에 안 보이면**: NVIDIA 드라이버 최신 버전 설치 확인 (`nvidia-smi`)
- **화면이 검게 녹화되면**: `Screen Capture (XSHM)` 대신 `PipeWire` 또는 `Screen Capture (XComposite)` 시도
- **오디오가 안 잡히면**: Audio Sources에서 Desktop Audio / Mic 별도 추가 확인
