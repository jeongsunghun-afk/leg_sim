#!/usr/bin/env python3
"""render_traj.py — biped_sim 의 QPOS_LOG 궤적을 **EGL 오프스크린**으로 그려 GIF 로 낸다.

★왜 이 경로인가
  저장소 뷰어(`biped_view`)는 GLFW 라 libglfw3-dev 가 없으면 **빌드조차 안 된다.**
  WSLg 에서는 깔아도 인터랙티브 GLFW 가 검은 화면이 되는 이력이 있다(2026-08-04).
  ⇒ 창을 띄우지 않고 프레임버퍼에만 그린다. 화면 없는 서버·CI 에서도 그림이 나온다.

★sim 을 다시 돌리지 않는다 — 궤적을 **재생**만 한다
  제어는 C++(BipedControl)에 있어 파이썬으로 재현할 수 없다. 그래서 물리는 sim 이 풀고
  여기서는 qpos 를 넣어 `mj_forward` 로 자세만 세운다. 화면에 보이는 것은 **그 런의 실제
  거동**이지 재시뮬이 아니다.

사용:
  cd ~/simulation/biped/cpp
  QPOS_LOG=/tmp/stand.traj TORSO_ADD_KG=1.1 CONTACT=1 \
      ./build/biped_sim ../biped_flatfoot.mjcf 0.0 10
  python3 ../tools/render_traj.py ../biped_flatfoot.mjcf /tmp/stand.traj /tmp/stand.gif

옵션: --width --height --fps --cam-dist --cam-azim --cam-elev --every
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")      # ★import 전에 정해야 한다
# ★★소프트웨어 렌더를 **기본으로** 강제한다 (2026-08-21).
#   이 WSLg 기기의 GPU(D3D12) 경로는 깨져 있다 — 예외 없이 **노이즈**를 그린다.
#   ⚠검은 화면이 아니라 노이즈라서 "std > 0" 같은 검사는 **통과해 버린다.**
#     실제로 그렇게 통과시켰다가 노이즈 GIF 를 만들었다. std 로는 못 거른다.
#   llvmpipe 로 돌리면 정상이다. 느리지만 궤적 재생은 실시간일 필요가 없다.
#   GPU 가 멀쩡한 기기에선 GPU_RENDER=1 로 끌 수 있다.
if os.environ.get("GPU_RENDER", "0") != "1":
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

import numpy as np
import mujoco
from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mjcf")
    ap.add_argument("traj")
    ap.add_argument("out")
    # ★기본 448×336 — 이 WSLg/EGL 조합은 **150k 화소를 넘으면 조용히 검은 화면**이 된다.
    #   예외가 안 난다(448×336 ✅ / 480×360 ❌ · `D3D12: Removing Device`). 실측 경계다.
    #   더 키우려면 아래 검은화면 검사를 보고 통과하는 값을 찾을 것.
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--every", type=int, default=2, help="궤적 N 줄마다 한 프레임")
    ap.add_argument("--colors", type=int, default=64, help="GIF 팔레트 색 수(작을수록 가볍다)")
    ap.add_argument("--forces", action="store_true", help="접촉력 화살표(파일이 크게 늘어난다)")
    ap.add_argument("--follow", action="store_true", help="카메라가 몸통을 따라간다(보행용). 기본은 고정")
    ap.add_argument("--cam-dist", type=float, default=1.9)
    ap.add_argument("--cam-azim", type=float, default=135.0)
    ap.add_argument("--cam-elev", type=float, default=-12.0)
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.mjcf)
    d = mujoco.MjData(m)

    rows = []
    with open(a.traj) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = [float(x) for x in line.split()]
            if len(v) == m.nq + 1:
                rows.append(v)
    if not rows:
        print(f"✗ 궤적이 비었다: {a.traj}  (nq={m.nq} 와 열 수가 맞는지 확인)")
        return 1

    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = a.cam_dist, a.cam_azim, a.cam_elev
    opt = mujoco.MjvOption()
    # ★접촉점을 보이게 한다 — 2점 평발이 **네 점을 다 쓰는지**가 육안 확인의 핵심이다.
    #   실기에서 K=1~2 로 떨어졌던 항목이라, 시뮬에서 4점이 붙는 그림이 기준이 된다.
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    # ⚠접촉력 화살표는 기본 **끈다** — 매 프레임 길이가 흔들려 GIF 델타 압축을 무력화한다
    #   (실측: 켜면 14.1MB, 끄면 그 몇 분의 1). 힘 크기를 봐야 할 때만 --forces 로 켤 것.
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = a.forces

    # 요청 크기가 들어가도록 오프스크린 버퍼를 키운다(모델 기본은 640×480).
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, a.width)
    m.vis.global_.offheight = max(m.vis.global_.offheight, a.height)

    # ★카메라는 기본 **고정**이다(--follow 로 추종). 이유는 화질이 아니라 **파일 크기**다:
    #   따라가면 매 프레임 화면 전체가 미세하게 밀려 GIF 델타 압축이 통째로 무력해진다
    #   (실측 100프레임: 추종 11.0MB → 고정은 그 몇 분의 1). stand 는 제자리라 고정이 맞고,
    #   보행처럼 실제로 이동하는 런에서만 --follow 를 쓴다.
    mid = rows[len(rows) // 2]
    fixed = [mid[1], mid[2], max(0.25, mid[3] * 0.7)]

    frames, worst = [], 0.0
    with mujoco.Renderer(m, a.height, a.width) as r:
        for k, v in enumerate(rows):
            if k % a.every:
                continue
            d.qpos[:] = v[1:]
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            if a.follow:
                cam.lookat[:] = d.qpos[0:3]
                cam.lookat[2] = max(0.25, d.qpos[2] * 0.7)
            else:
                cam.lookat[:] = fixed
            r.update_scene(d, camera=cam, scene_option=opt)
            img = r.render()
            worst = max(worst, float(np.std(img)))
            frames.append(Image.fromarray(img))

    if not frames:
        print("✗ 프레임이 없다 — --every 를 줄일 것")
        return 1
    # ★★검은 화면 검사 — **이게 없으면 조용히 실패한다.**
    #   EGL 은 크기가 크면 예외 없이 전부 0 을 돌려준다(2026-08-21: 448×336 ✅ / 480×360 ❌).
    #   shape 만 보고 "렌더 성공" 으로 판단했다가 검은 GIF 를 만든 적이 있다.
    if worst < 5.0:
        print(f"✗ **전 프레임이 사실상 검은 화면**이다(최대 std={worst:.2f}).")
        print(f"  이 기기의 오프스크린 한계일 가능성이 크다 — --width/--height 를 줄여볼 것")
        print(f"  (실측: 448×336 까지 정상 · 480×360 부터 검은 화면)")
        return 2
    # ★팔레트 양자화 — 안 하면 GIF 가 프레임당 트루컬러로 저장돼 터진다(실측 250프레임
    #   448×336 이 **36.9 MB**). 첫 프레임의 적응 팔레트를 전 프레임이 공유하면
    #   장면이 고정된 stand 에서는 화질 손실이 거의 없다.
    pal = frames[0].convert("P", palette=Image.ADAPTIVE, colors=a.colors)
    qframes = [pal] + [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames[1:]]
    dur = max(20, int(1000.0 / a.fps))
    qframes[0].save(a.out, save_all=True, append_images=qframes[1:],
                    duration=dur, loop=0, optimize=True)
    mb = os.path.getsize(a.out) / 1e6
    print(f"✅ {a.out}  ({len(frames)} 프레임 · {dur}ms/f · {mb:.1f} MB)")
    print(f"   궤적 {len(rows)} 줄 · t {rows[0][0]:.2f}~{rows[-1][0]:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
