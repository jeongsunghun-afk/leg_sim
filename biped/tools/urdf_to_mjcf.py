#!/usr/bin/env python3
"""urdf_to_mjcf.py — SolidWorks URDF 내보내기 → MuJoCo MJCF.

★왜 스크립트인가 (2026-08-11)
  CAD 가 갱신될 때마다 손으로 XML 을 옮기면 반드시 어긋난다. 이번에 받은 URDF 만
  해도 질량 5개·관절범위 4개·링크원점 2개가 종전 MJCF 와 달랐다. 다음 갱신 때
  **같은 변환을 다시 돌릴 수 있어야** 한다.

무엇을 URDF 에서 가져오고 무엇을 안 가져오나:
  URDF 에서    — 링크 트리·관절 원점/축/범위·질량·**전관성(fullinertia)**·메시·effort
  MJCF 에 유지 — 바닥·조명·IMU site·actuator·sensor·option·default·**접촉구(sphere)**
                 (URDF 에는 이런 개념이 없다. 시뮬 설정은 사람이 정한 값이다)
  ⚠armature/damping/frictionloss 는 **넣지 않는다** — 런타임(apply_gearbox)이 주입한다.
    여기 적으면 이중적용된다. 파이프라인을 바꾸려면 그때 같이 바꿀 것.

★로드 검증 — Pi 에도 mujoco 를 깔았다(venv ~/.venv-mujoco, aarch64 휠 존재).
      ~/.venv-mujoco/bin/python -c "import mujoco;mujoco.MjModel.from_xml_path('x.mjcf')"
  XML 파싱만으로는 못 잡는 것이 실제로 있었다: thigh STL 이 263,287 면이라
  MuJoCo 상한(200,000)을 넘어 로드 실패했다 → tools/decimate_stl.py 로 감면.

사용:
    python3 tools/urdf_to_mjcf.py <urdf> -o <out.mjcf> [--ranges urdf|legacy|intersect]
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from math import degrees

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIPED = os.path.dirname(HERE)

# 종전 MJCF(quad 승계) 의 관절범위[rad] — --ranges legacy/intersect 에서 쓴다.
LEGACY_RANGE = {
    "hip":   (-0.6109, 0.6109),
    "thigh": (-2.356, 1.134),
    "calf":  (-0.959, 1.134),
    "foot":  (-1.396, 0.698),
}
SPHERE_R = 0.036          # 접촉구 반경(종전 MJCF 승계)
CONTACT = dict(friction="1.6 0.05 0.001", condim="3",
               solref="0.004 1", solimp="0.95 0.99 0.001")


def _f(s):
    return [float(x) for x in s.split()]


# ★MJCF body 이름은 컨트롤러와의 **계약**이다 — 아래 이름으로 mj_name2id 조회한다.
#   biped_wbic.py:74 · cpp/src/biped_control.hpp:95 (양쪽 동일)
REQUIRED_BODIES = ("HL_foot_contact_link", "HR_foot_contact_link")


def _norm(name):
    """링크명 정규화 — CAD 익스포터가 붙인 `_collision` 접미사를 뗀다.

    ⚠2026-08-12 실측: SolidWorks URDF 익스포터가 파트명에 `_collision` 을 붙여 내보내
      body 이름이 `HL_foot_contact_link_collision` 이 됐다. 컨트롤러의 조회가 **−1** 이
      되어 발 자코비안이 통째로 틀렸고 **15.00s 무낙상 → 0.32s 낙상**했다.
      C++·Python 이 같은 이름을 쓰므로 양쪽이 함께 죽었다.
    ⚠생성 시 이름 대조(551d7da)가 **관절·geom·actuator·sensor 만** 보고 body 를 안 봤다.
      → 아래 REQUIRED_BODIES 검증을 추가했다.
    CAD 파트명은 사람이 언제든 바꾼다. 이름 계약은 여기서 고정한다.
    """
    return name[:-len("_collision")] if name.endswith("_collision") else name


def load_urdf(path):
    r = ET.parse(path).getroot()
    links, joints = {}, {}
    for l in r.findall("link"):
        i = l.find("inertial")
        d = {"name": _norm(l.get("name")), "meshes": []}
        if i is not None:
            n = i.find("inertia")
            d.update(mass=float(i.find("mass").get("value")),
                     ipos=_f(i.find("origin").get("xyz")),
                     irpy=_f(i.find("origin").get("rpy")),
                     inertia=[float(n.get(k)) for k in
                              ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")])
        for v in l.findall("visual") + l.findall("collision"):
            m = v.find("geometry/mesh")
            if m is not None:
                d["meshes"].append(os.path.basename(m.get("filename")))
        links[d["name"]] = d
    for j in r.findall("joint"):
        o, a, lm = j.find("origin"), j.find("axis"), j.find("limit")
        joints[j.get("name")] = {
            "name": j.get("name"), "type": j.get("type"),
            "parent": _norm(j.find("parent").get("link")),
            "child": _norm(j.find("child").get("link")),
            "pos": _f(o.get("xyz")), "rpy": _f(o.get("rpy")),
            "axis": _f(a.get("xyz")) if a is not None else None,
            "range": (float(lm.get("lower")), float(lm.get("upper"))) if lm is not None else None,
            "effort": float(lm.get("effort")) if lm is not None else None,
        }
    return links, joints


def check_inertia(d, log):
    """관성텐서가 물리적으로 성립하는지. 양정부호 + 삼각부등식."""
    ixx, iyy, izz, ixy, ixz, iyz = d["inertia"]
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    ev = np.linalg.eigvalsh(I)
    bad = []
    if ev.min() <= 0:
        bad.append(f"양정부호 아님(최소 고윳값 {ev.min():.3e})")
    a, b, c = sorted(ev)
    if a + b < c * (1 - 1e-9):
        bad.append(f"삼각부등식 위반({a:.3e}+{b:.3e} < {c:.3e})")
    if bad:
        log(f"    ✗ {d['name']}: " + " · ".join(bad))
    return not bad, ev


def build(urdf_path, out_path, ranges_mode="urdf", base_link=None, foot="flat",
          meshdir_name="meshes", log=print):
    links, joints = load_urdf(urdf_path)
    kids = {}
    for j in joints.values():
        kids.setdefault(j["parent"], []).append(j)
    children = {j["child"] for j in joints.values()}
    roots = [n for n in links if n not in children]
    base = base_link or roots[0]
    log(f"  base link = {base}  (링크 {len(links)} · 관절 {len(joints)})")

    # ── 검증 1: 관성 ────────────────────────────────────────────────────
    log("  관성 검증(양정부호·삼각부등식):")
    ok = True
    for d in links.values():
        if "mass" in d:
            good, _ = check_inertia(d, log)
            ok &= good
    log("    ✓ 전 링크 통과" if ok else "    ★위 항목 확인 필요")

    # ── 검증 2: 메시 존재 ───────────────────────────────────────────────
    # ★MuJoCo 용 감면 메시를 쓴다. 원본(meshes/)은 CAD 비교용으로 보존한다.
    meshdir = os.path.join(os.path.dirname(os.path.dirname(urdf_path)), meshdir_name)
    names, missing = [], []
    for d in links.values():
        for m in d["meshes"]:
            if m not in names:
                names.append(m)
                if not os.path.exists(os.path.join(meshdir, m)):
                    missing.append(m)
    log(f"  메시 {len(names)}개 · 누락 {len(missing)}" + (f" ★{missing}" if missing else " ✓"))

    # ── 검증 3: ★body 이름 계약 ─────────────────────────────────────────
    # 종전 검증은 관절·geom·actuator·sensor 만 봤다. body 가 빠져 있었고 그래서
    # `_collision` 접미사가 조용히 통과해 보행이 죽었다(_norm 주석 참조).
    lost = [b for b in REQUIRED_BODIES if b not in links]
    if lost:
        raise SystemExit(
            f"  ✗ 컨트롤러가 찾는 body 가 URDF 에 없다: {lost}\n"
            f"    URDF 링크명: {sorted(links)}\n"
            f"    → CAD 파트명이 바뀌었다. _norm() 에 정규화 규칙을 추가하거나,\n"
            f"      이름 변경이 의도된 것이면 biped_wbic.py:74 · biped_control.hpp:95 를\n"
            f"      **함께** 고칠 것. 한쪽만 고치면 mj_name2id 가 −1 을 돌려주고\n"
            f"      발 자코비안이 틀려 즉시 낙상한다.")
    log(f"  body 이름 계약 {len(REQUIRED_BODIES)}개 ✓ ({', '.join(REQUIRED_BODIES)})")

    # ── 서기 높이 계산: 접촉구 최하점이 z=0 이 되도록 ────────────────────
    def chain_z(link, z=0.0):
        out = [z]
        for j in kids.get(link, []):
            out += chain_z(j["child"], z + j["pos"][2])
        return out
    z_low = min(chain_z(base)) - SPHERE_R
    base_z = round(-z_low, 4)
    log(f"  서기 높이 base z = {base_z}  (최하 접촉점이 바닥에 닿는 값)")

    # ── 본문 생성 ──────────────────────────────────────────────────────
    def key(nm):
        for k in ("hip", "thigh", "calf", "foot"):
            if f"_{k}_joint" in nm:
                return k
        return None

    def rng_of(j):
        k = key(j["name"])
        u = j["range"]
        if k is None or u is None:
            return u
        if ranges_mode == "urdf":
            return u
        if ranges_mode == "legacy":
            return LEGACY_RANGE[k]
        lo, hi = LEGACY_RANGE[k]
        return (max(lo, u[0]), min(hi, u[1]))          # intersect

    def emit(link, ind):
        d = links[link]
        s = []
        p = "  " * ind
        for j in kids.get(link, []):
            c = j["child"]
            s.append(f'{p}<body name="{c}" pos="{" ".join(f"{x:g}" for x in j["pos"])}">')
            cd = links[c]
            if "mass" in cd:
                ix = cd["inertia"]
                s.append(f'{p}  <inertial pos="{" ".join(f"{x:g}" for x in cd["ipos"])}" '
                         f'mass="{cd["mass"]:g}" '
                         f'fullinertia="{" ".join(f"{v:.6g}" for v in ix)}" />')
            if j["type"] != "fixed":
                r = rng_of(j)
                ax = " ".join(f"{x:g}" for x in j["axis"])
                s.append(f'{p}  <joint name="{j["name"]}" pos="0 0 0" axis="{ax}" '
                         f'range="{r[0]:g} {r[1]:g}" '
                         f'actuatorfrcrange="-{j["effort"]:g} {j["effort"]:g}" />')
            for m in cd["meshes"][:1]:
                # ★점발(배포)은 링크메시가 **충돌체**다. 평발은 시각 전용이고 접촉은 구가 맡는다.
                #   종전 두 MJCF 의 차이가 정확히 이것 + heel 구 유무였다.
                col = "" if foot == "point" else 'contype="0" conaffinity="0" '
                s.append(f'{p}  <geom type="mesh" {col}group="1" '
                         f'density="0" rgba="0.75294 0.75294 0.75294 1" '
                         f'mesh="{os.path.splitext(m)[0]}" />')
            # ★접촉구 — URDF 에 없는 시뮬 설정. 평발 2점 접촉(발목 heel + 발끝 toe).
            side = c.split("_")[0]
            if "_foot_link" in c and foot == "flat":     # heel 구는 평발에만
                s.append(f'{p}  <geom name="{side}_sphere2" type="sphere" size="{SPHERE_R}" '
                         f'pos="0 0 0" rgba="0.3 0.5 0.9 1" '
                         + " ".join(f'{k}="{v}"' for k, v in CONTACT.items()) + " />")
            if "_foot_contact_link" in c:
                s.append(f'{p}  <geom name="{side}_sphere" type="sphere" size="{SPHERE_R}" '
                         f'pos="0 0 0" rgba="0.9 0.3 0.3 1" '
                         + " ".join(f'{k}="{v}"' for k, v in CONTACT.items()) + " />")
            s += emit(c, ind + 1)
            s.append(f"{p}</body>")
        return s

    bd = links[base]
    ix = bd["inertia"]
    rel = os.path.relpath(meshdir, BIPED) + "/"
    body = "\n".join(emit(base, 4))
    assets = "\n".join(f'    <mesh name="{os.path.splitext(m)[0]}" content_type="model/stl" '
                       f'file="{m}" />' for m in names)
    act = "\n".join(f'    <motor joint="{n}" name="{n[:-6]}" />'
                    for n in joints if joints[n]["type"] != "fixed")
    foot_note = {"flat": "평발 — 링크메시는 시각 전용, 접촉은 발목(heel)+발끝(toe) **2점 구**",
                 "point": "점발(배포) — 링크메시가 **충돌체**, 접촉은 발끝 **1점 구**"}[foot]
    rng_note = {"urdf": "새 URDF 값 그대로",
                "legacy": "종전 MJCF(quad 승계) 값 유지",
                "intersect": "두 값의 **교집합**(보수적)"}[ranges_mode]

    xml = f'''<mujoco model="biped_cad_{os.path.basename(urdf_path).split("_")[-1][:6]}">
  <!--
    ★{os.path.basename(urdf_path)} 에서 tools/urdf_to_mjcf.py 로 생성. 손으로 고치지 말 것 —
      CAD 가 갱신되면 스크립트를 다시 돌린다.

    URDF 에서 온 것 : 링크 트리 · 관절 원점/축/범위 · 질량 · 전관성 · 메시 · effort
    사람이 정한 것   : 바닥 · 조명 · IMU site · actuator · sensor · option · default ·
                      **접촉구(sphere, r={SPHERE_R})** — URDF 에는 이런 개념이 없다

    ⚠발 구성: {foot_note}
    ⚠관절범위: {rng_note}
    ⚠armature/damping/frictionloss 는 **의도적으로 없다** — 런타임(apply_gearbox)이 주입한다.
      여기 적으면 이중적용된다.
    ⚠**MuJoCo 로드 검증 미실시** — 생성 환경(Pi)에 mujoco 가 없다. 노트북에서 반드시 확인할 것.
  -->
  <compiler angle="radian" meshdir="{rel}" />
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.33 0.36 0.40" rgb2="0.42 0.45 0.50" width="512" height="512" />
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="12 12" reflectance="0.12" />
{assets}
  </asset>
  <worldbody>
    <body name="torso" pos="0 0 {base_z}">
      <freejoint name="root" />
      <inertial pos="{" ".join(f"{x:g}" for x in bd["ipos"])}" mass="{bd["mass"]:g}" fullinertia="{" ".join(f"{v:.6g}" for v in ix)}" />
      <site name="imu" pos="0 0 0.05" size="0.01" />   <!-- ★IMU 장착점(실기 반영) -->
      <geom type="mesh" contype="0" conaffinity="0" group="1" density="0" rgba="0.75294 0.75294 0.75294 1" mesh="{os.path.splitext(bd["meshes"][0])[0]}" />
{body}
    </body>
    <geom name="floor" type="plane" size="0 0 0.1" material="floor_mat" friction="1.6 0.05 0.001" solref="0.004 1" solimp="0.95 0.99 0.001" />
    <light pos="0 0 3" dir="0 0 -1" />
  </worldbody>
  <option timestep="0.002" gravity="0 0 -9.81" />
  <default>
    <geom friction="1.3 0.02 0.001" condim="3" />
    <motor ctrllimited="true" ctrlrange="-200 200" />
  </default>
  <actuator>
{act}
  </actuator>
  <!-- ★IMU 센서(실기 반영): 상태추정기 입력. framepos/framelinvel 은 GT 비교 전용. -->
  <sensor>
    <gyro          site="imu" name="imu_gyro"  />
    <accelerometer site="imu" name="imu_acc"   />
    <framequat     objtype="site" objname="imu" name="imu_quat" />
    <framepos      objtype="site" objname="imu" name="gt_pos"  />
    <framelinvel   objtype="site" objname="imu" name="gt_vel"  />
  </sensor>
</mujoco>
'''
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    ET.parse(out_path)          # ★XML 구조 검증(로드 검증은 아니다)
    log(f"  ✓ 생성: {out_path}  (XML 파싱 통과 — MuJoCo 로드 검증은 별도)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--ranges", choices=["urdf", "legacy", "intersect"], default="urdf",
                    help="관절범위 출처. urdf=새 CAD · legacy=종전 유지 · intersect=교집합(보수적)")
    ap.add_argument("--foot", choices=["flat", "point"], default="flat",
                    help="flat=평발 2점 구(링크메시 시각전용) · point=점발 1점 구(링크메시 충돌체)")
    ap.add_argument("--meshdir", default="meshes",
                    help="URDF 패키지 안의 메시 디렉터리명(기본 meshes. MuJoCo 는 meshes_mjcf)")
    ap.add_argument("--base", default=None, help="base 링크 이름(기본: 부모 없는 링크)")
    a = ap.parse_args()
    print(f"URDF → MJCF  (범위: {a.ranges} · 발: {a.foot})")
    build(a.urdf, a.out, a.ranges, a.base, a.foot, a.meshdir)
