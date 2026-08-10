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
