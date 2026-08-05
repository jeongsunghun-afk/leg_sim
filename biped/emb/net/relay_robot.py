#!/usr/bin/env python3
"""relay_robot.py — 로봇(Pi)측 릴레이. 노트북이 `ssh biped python3 .../relay_robot.py` 로 실행.
  stdin  ← 노트북 GUI 의 cmd JSON(한 줄) → /tmp/biped_cmd.json 로 기록 (app 이 소비)
  stdout → /tmp/biped_state.json 내용(한 줄) 을 ~30Hz 로 송신 (노트북이 수신)
NAT 뒤 노트북(WSL2)이 접속을 개시하므로 로봇→노트북 직접연결 불필요. 순수 stdlib.

★이 파일은 git 에 미추적 상태였다가 2026-08-05 클론 때 삭제되어 복원했다.
  로봇에는 dearpygui 가 없어 GUI 를 띄울 수 없으므로(teleop_gui_biped 는 노트북에서 실행),
  이 릴레이가 노트북 GUI ↔ 로봇 app 을 잇는 유일한 경로다. 커밋해 두는 것을 권장.
"""
import sys, os, json, time, threading

CMD   = os.environ.get("QUAD_CMD",   "/tmp/biped_cmd.json")
STATE = os.environ.get("QUAD_STATE", "/tmp/biped_state.json")

def rx_stdin():
    """노트북 → cmd 파일."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)                    # 유효성 검사
            tmp = CMD + ".tmp"
            with open(tmp, "w") as f:
                f.write(line)
            os.replace(tmp, CMD)
        except Exception:
            pass

def main():
    threading.Thread(target=rx_stdin, daemon=True).start()
    last = ""
    while True:
        try:
            with open(STATE) as f:
                s = f.read().strip()
            if s and s != last:
                sys.stdout.write(s.replace("\n", "") + "\n")
                sys.stdout.flush()
                last = s
        except Exception:
            pass
        time.sleep(0.03)                        # ~30Hz

if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
