#!/usr/bin/env python3
"""imu_peek.py — IMU 가 **살아 있는지**와 **어떻게 장착됐는지**를 본다. 읽기전용.

═══ 왜 필요한가 ═══
2026-09-02 벤더 업데이트가 halIMU.cpp 의 인덱싱 버그를 고쳤다:
    - memcpy(&m_fAccel_MpSEC2[0],          &fAccl[0], 3*sizeof(float));   // 항상 슬롯 0
    + memcpy(&m_fAccel_MpSEC2[unDevID][0], &fAccl[0], 3*sizeof(float));
배열은 [장치][3] 인데 쓰기는 늘 인덱스 0 이었고, 읽기는 DEF_DevID_IMU_Body = **4** 였다.
⇒ Body IMU 가 **영원히 0**. 그게 IMU_RECOVERY.md 의 "IMU 전부 0" 원인이다.
   그 0 때문에 WBIC orientation task 오차가 항상 0 이 되어 **균형이 구조적으로 불가능**했다.

⇒ 이 스크립트는 두 가지를 한 번에 답한다:
   ① **살아났는가** — |acc| 가 중력 9.81 근처인가. 0 이면 아직 죽은 것이다.
   ② **어떻게 붙어 있는가** — 정지 상태에서 중력이 **어느 축에** 실리는지가 곧
      장착 방향이다. 좌표변환을 짜려면 이걸 먼저 알아야 한다.

═══ 쓰는 법 ═══
  ① Emb 를 띄운다(제어기는 필요 없다):
       cd ~/simulation/biped/emb && diag/emb_ctl.sh start
  ② 로봇을 **평평하고 안정된 자세**로 두고(매달았으면 흔들림이 멎을 때까지 기다린다):
       python3 diag/imu_peek.py            # 3초 표본
       python3 diag/imu_peek.py --sec 10   # 더 길게
  ③ 축을 하나씩 기울여 가며 다시 찍으면 부호까지 확정된다.

⚠**아무것도 쓰지 않는다.** bridge_read 만 부른다 — 제어기와 같이 돌려도 안전하고,
  모터에 명령이 나가지 않는다(그래서 매달린 채로도 안전하다).
⚠단위: RPY 는 deg. 자이로는 **deg/s** 다 — 2026-09-02 업데이트에서 벤더가
  m_fGyro_RADpSEC → m_fGyro_DEGpSEC 로 이름을 고쳤다(이름만 틀렸던 것).
  우리 쪽 biped_deploy 는 config 의 `imu_deg: true` 를 보고 D2R 을 곱하므로 맞다.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hal"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "interface"))

G = 9.80665
AX = ("x", "y", "z")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=3.0, help="표본 시간[s]")
    ap.add_argument("--hz", type=float, default=50.0, help="표본율[Hz]")
    a = ap.parse_args()

    try:
        import numpy as np
        from shm_backend import ShmBackend
    except Exception as e:
        print(f"✗ 백엔드를 못 불러온다: {e}")
        print("  hal/build_bridge.sh 로 libbipedshm.so 를 먼저 만들 것.")
        return 2

    # ★ShmBackend 는 (lib_path, n_channel, recv_wait_ms) 를 받는다 — config 에서 읽는다
    #   (app/biped_emb.py:57 과 같은 방식. 여기만 다르면 조용히 갈라진다).
    try:
        import yaml
        _cfg = yaml.safe_load(open(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "config", "biped_emb.yaml")))
        _shm = _cfg["shm"]
        hw = ShmBackend(_shm["lib"], int(_shm["n_channel"]), int(_shm.get("recv_wait_ms", 2000)))
    except Exception as e:
        print(f"✗ SHM 연결 실패: {e}")
        print("  Emb 가 떠 있는지 확인:  pgrep -x RobotEmbedded")
        print("  기동:  cd ~/simulation/biped/emb && diag/emb_ctl.sh start")
        return 2

    n = max(2, int(a.sec * a.hz))
    rpy, acc, gyr, masks = [], [], [], []
    print(f"  표본 수집 {a.sec:.1f}s @ {a.hz:.0f}Hz …")
    for _ in range(n):
        s = hw.read()
        # ★필드명은 RawState 정의(hal/backend.py:20-25)를 따른다 — rpy/acc/gyr 이 아니다.
        rpy.append(list(s.imu_rpy_deg)); acc.append(list(s.imu_acc)); gyr.append(list(s.imu_gyro))
        masks.append(int(s.updated))
        time.sleep(1.0 / a.hz)

    rpy = np.array(rpy, float); acc = np.array(acc, float); gyr = np.array(gyr, float)

    print("\n" + "=" * 72)
    print("  IMU 실측  (읽기전용 · 아무것도 쓰지 않았다)")
    print("=" * 72)

    # ── ① 살아 있는가 ──────────────────────────────────────────────────────
    #   ★판정 기준은 "0 이 아님" 이 아니라 **중력이 잡히는가** 다.
    #     정지한 IMU 의 가속도계는 반드시 |a| ≈ 9.81 을 읽는다. 0 이면 죽은 것이고,
    #     9.81 이 아닌 다른 상수면 스케일이 틀린 것이다(둘은 조치가 다르다).
    amag = float(np.linalg.norm(acc.mean(axis=0)))
    alive = amag > 0.5
    print(f"\n  ① 생존판정 — |중력| = {amag:6.3f} m/s²   (기대 {G:.2f})")
    if not alive:
        print("     ⛔ **아직 죽어 있다.** 가속도 크기가 0 이다.")
        print("        · Emb 가 새 바이너리인지 확인(halIMU.cpp 의 [unDevID] 수정 포함)")
        print("        · UART IMU 배선/전원 확인 — emb/IMU_RECOVERY.md")
    elif abs(amag - G) > 1.5:
        print(f"     ⚠ 살아는 있으나 크기가 어긋난다({amag:.2f} vs {G:.2f}) — 스케일/단위 확인.")
    else:
        print("     ✅ **살아 있다.** 중력이 정상적으로 잡힌다.")

    # ── ② 어떻게 붙어 있는가 ───────────────────────────────────────────────
    #   정지 상태에서 중력이 실리는 축이 곧 **아래(−z_world)** 다.
    #   이게 좌표변환의 출발점이다 — 여기서 축 대응과 부호가 나온다.
    m = acc.mean(axis=0)
    print(f"\n  ② 장착 방향 — 정지 시 가속도 [m/s²]")
    for i in range(3):
        bar = "█" * int(abs(m[i]) / G * 20)
        print(f"       a{AX[i]} {m[i]:+7.3f}  {bar}")
    if alive:
        k = int(np.argmax(np.abs(m)))
        sgn = "+" if m[k] > 0 else "−"
        print(f"     ⇒ 중력이 **a{AX[k]}** 에 실린다({sgn}). 즉 센서의 {sgn}{AX[k]} 축이 "
              f"{'위' if m[k] > 0 else '아래'}를 향한다.")
        print( "        (일반적인 장착이면 az ≈ +9.81. 다르면 좌표변환이 필요하다)")
        off = [abs(m[i]) for i in range(3) if i != k]
        if max(off) > 1.0:
            tilt = np.degrees(np.arctan2(max(off), abs(m[k])))
            print(f"     ⚠ 다른 축에도 {max(off):.2f} 가 남아 있다 — 로봇이 {tilt:.1f}° 기울었거나")
            print( "        IMU 가 기울어 장착됐다. **평평한 기준자세에서 다시 찍어 가를 것.**")

    # ── ③ 값 표 ────────────────────────────────────────────────────────────
    print(f"\n  ③ 표본 {n}개 통계")
    print(f"       {'':6} {'평균':>10} {'최소':>10} {'최대':>10} {'표준편차':>10}")
    for nm, arr, unit in (("RPY", rpy, "deg"), ("ACC", acc, "m/s²"), ("GYR", gyr, "deg/s")):
        for i in range(3):
            v = arr[:, i]
            print(f"     {nm}.{AX[i]:1} {v.mean():10.3f} {v.min():10.3f} {v.max():10.3f} "
                  f"{v.std():10.4f}  {unit}")

    # ── ④ 동결 판정 ────────────────────────────────────────────────────────
    #   ⚠"0 이 아니다" 만으로는 부족하다. 마지막 값이 굳은 채 재발행될 수 있다
    #     (EtherCAT 동결과 같은 함정 — memory: emb-ethercat-freeze).
    #     실기 IMU 는 정지 중에도 반드시 미세하게 떨린다. 표준편차가 그 근거다.
    jitter = float(max(rpy.std(axis=0).max(), acc.std(axis=0).max(), gyr.std(axis=0).max()))
    print(f"\n  ④ 동결판정 — 전 채널 최대 표준편차 {jitter:.5f}")
    if jitter < 1e-6:
        print("     ⛔ **값이 굳어 있다.** 살아 있는 센서는 정지 중에도 떨린다.")
        print("        0 이 아니어도 '마지막 값 재발행' 일 수 있다 — 값만 보고 믿지 말 것.")
    else:
        print("     ✅ 값이 갱신되고 있다(떨림이 있다).")

    print(f"\n  updated=0x{masks[-1]:02X}   (0x10 비트 = IMU 유효. shm_bridge 가 |a|>0.5 일 때만 세운다)")
    print()
    return 0 if alive else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  중단")
        sys.exit(130)
