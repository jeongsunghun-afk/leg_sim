#!/bin/bash
# lib_display.sh — GUI 런처 공용: 표시할 디스플레이를 자동으로 잡는다.
#   `source` 해서 쓴다. 성공 시 DISPLAY/XAUTHORITY 를 export 하고 0 반환.
#
# ★왜 필요한가 (2026-08-07 실측):
#   이 로봇 데스크톱은 **Wayland(GNOME)** 이고 Xwayland 는 **전용 auth 파일**을 쓴다
#     /run/user/1000/.mutter-Xwaylandauth.XXXXXX
#   ~/.Xauthority 는 낡은 쿠키라 안 맞는다. 그래서 SSH 에서 그냥 `DISPLAY=:0` 을 주면
#     Authorization required, but no authorization protocol specified
#     Glfw Error 65544: X11: Failed to open display :0
#   가 나고 dearpygui 가 assert 로 core dump 한다.
#   ⇒ mutter 쿠키를 찾아 XAUTHORITY 로 물려야 한다.
#
# 두 가지 표시 경로:
#   (A) 로봇 화면   — SSH 에서 로봇에 붙은 모니터에 띄운다. DISPLAY=:0 + mutter 쿠키.
#   (B) 노트북 화면 — `ssh -X`(또는 -Y) 로 접속하면 DISPLAY 가 자동 설정된다. 그걸 그대로 쓴다.
#   우선순위는 (B) → (A). 사용자가 -X 로 들어왔으면 그 의도를 존중한다.

_disp_probe() {                      # $1=DISPLAY $2=XAUTHORITY → 접속되면 0
  local d="$1" xa="$2"
  if command -v xdpyinfo >/dev/null 2>&1; then
    DISPLAY="$d" XAUTHORITY="$xa" timeout 5 xdpyinfo >/dev/null 2>&1
  else                               # xdpyinfo 없으면 소켓 존재로만 판단(약한 확인)
    [ -e "/tmp/.X11-unix/X${d#:}" ]
  fi
}

setup_display() {
  # (B) ssh -X 등으로 이미 잡혀 있으면 그대로 쓴다
  if [ -n "$DISPLAY" ] && _disp_probe "$DISPLAY" "${XAUTHORITY:-$HOME/.Xauthority}"; then
    export DISPLAY XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
    echo "display=$DISPLAY (기존 세션 — ssh -X 또는 로컬)"
    return 0
  fi

  # (A) 로컬 Wayland/Xwayland 세션에 붙는다
  local xa
  xa=$(ls -t /run/user/$(id -u)/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
  [ -z "$xa" ] && [ -f "$HOME/.Xauthority" ] && xa="$HOME/.Xauthority"
  local d
  for d in :0 :1 :2; do
    if [ -e "/tmp/.X11-unix/X${d#:}" ] && _disp_probe "$d" "$xa"; then
      export DISPLAY="$d" XAUTHORITY="$xa"
      echo "display=$d (로봇 화면 · auth=$(basename "$xa"))"
      echo "  ⚠ 창은 **로봇에 연결된 모니터**에 뜬다. 노트북에서 보려면 ssh -X 로 재접속할 것."
      return 0
    fi
  done

  cat >&2 <<'EOS'
❌ 표시할 디스플레이를 찾지 못했다.

  선택 1) 노트북 화면에서 보기 — SSH 를 X 포워딩으로 다시 연결
      (노트북에서)  ssh -X rpetubt@<로봇IP>
      ⚠ 노트북에 X 서버가 필요하다(리눅스=기본, macOS=XQuartz, Windows=VcXsrv/WSLg)

  선택 2) 로봇 화면에서 보기 — 로봇에 모니터가 붙어 있고 데스크톱이 로그인돼 있어야 한다
      확인:  ls /tmp/.X11-unix/ ; ls /run/user/$(id -u)/.mutter-Xwaylandauth.*

  선택 3) GUI 없이 진행 — CLI 로 명령 채널에 직접 쓴다(emb/NEXT_HW.md §4 참조)
      while :; do printf '{"mode":"jog","jog_deg":[0,0,0,0,0,0,0,0],"seq":%s}' $RANDOM \
        > /tmp/biped_cmd.json.t && mv /tmp/biped_cmd.json.t /tmp/biped_cmd.json; sleep 0.05; done
EOS
  return 1
}

# ★MuJoCo 뷰어가 뜰 수 있는 디스플레이인가 — 화면 크기가 0x0 이면 못 뜬다.
#   MuJoCo 뷰어는 **화면 크기로** 창을 잡으므로 0x0 이면 이렇게 죽는다:
#     GLFWError: (65540) Invalid window size 0x0 · ERROR: could not create window
#   (dearpygui 는 고정크기 700x800 이라 0x0 에서도 뜬다 — 그래서 GUI 만 뜨고 뷰어가 죽는다)
#
#   ⚠2026-08-07 정정: 처음엔 "rootless Xwayland 는 원래 0x0" 으로 진단했는데 **틀렸다.**
#     모니터가 깨어 있으면 1920x1280 로 정상 보고한다. 0x0 은 **일시적 상태**다
#     (모니터 절전 / 세션 잠금 등). 그래서 이 검사로 **막지 않고 경고만** 한다 —
#     일시적 판독으로 런처를 차단하면 멀쩡한 상황에서 못 뜨게 된다.
viewer_capable() {
  command -v xdpyinfo >/dev/null 2>&1 || return 0        # 확인 불가 → 통과시킨다
  local dim
  dim=$(DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" xdpyinfo 2>/dev/null | grep -m1 dimensions)
  case "$dim" in *"0x0 pixels"*) return 1;; esac
  return 0
}

warn_if_no_viewer() {                 # ★경고만 한다. 막지 않는다(위 정정 참조)
  viewer_capable && return 0
  cat >&2 <<'EOS'

⚠ 화면 크기가 0x0 으로 읽힌다 — 이대로면 MuJoCo 뷰어가 창을 못 만든다
  (GLFWError: Invalid window size 0x0). GUI 는 고정크기라 그래도 뜬다.

  가장 흔한 원인: **모니터가 절전 상태**다. 로봇 모니터를 깨우고 다시 실행해 볼 것.
  (마우스를 움직이거나 키를 한 번 누르면 된다)

  그래도 0x0 이면 sim 뷰어는 다른 화면에서 봐야 한다:
    ① 노트북에서 직접 실행 ← 가장 빠름
         (노트북)  cd ~/simulation && git pull && cd biped && ./run_gui_biped.sh
    ② ssh -X 로 노트북 화면에 포워딩
  GUI 만 필요하면(실기 JOG 검증) 뷰어 없이 ./run_gui_only.sh 를 쓸 것.

  → 일단 그대로 진행한다. 뷰어가 죽으면 위를 참고할 것.

EOS
  return 0                            # ★0 반환 = 런처를 멈추지 않는다
}
