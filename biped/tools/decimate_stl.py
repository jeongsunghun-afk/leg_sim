#!/usr/bin/env python3
"""decimate_stl.py — 바이너리 STL 면수 감면(정점 클러스터링). 의존성 numpy 뿐.

★왜 필요한가 (2026-08-11)
  새 CAD 의 `*_collision.STL` 이 **충돌메시치고 터무니없이 조밀**하다:
      thigh 263,287 faces · base 183,352 · hip 92~97k
  MuJoCo 는 메시당 **200,000 면**이 상한이라 thigh 가 아예 로드에 실패했다.
  게다가 MuJoCo 의 메시 충돌은 **볼록껍질**로 계산하므로 원본 밀도는 대부분 낭비다
  (메모리·렌더 비용만 늘고 접촉 정확도는 안 오른다).

★원본을 고치지 않는다. 감면본을 **다른 디렉터리**에 쓰고 MJCF 가 그쪽을 본다.
  CAD 재수출 때 원본이 그대로 남아 있어야 비교가 된다.

방법 — 격자 정점 클러스터링:
  ① 정점을 한 변 d 인 격자로 양자화해 같은 칸끼리 묶고 무게중심으로 대표
  ② 세 정점이 같은 칸에 떨어진 삼각형은 퇴화 → 버림
  ③ 중복 삼각형 제거
  d 는 목표 면수를 맞추도록 이분탐색한다.
  ⚠단순한 방법이라 날카로운 모서리가 다소 뭉갠다. **충돌·시각용으로는 충분**하고,
    관성은 URDF 값을 쓰므로 메시 밀도와 무관하다(여기서 정확도가 안 깎인다).

사용: python3 tools/decimate_stl.py <in_dir> <out_dir> [--max-faces 60000]
"""
from __future__ import annotations

import argparse
import os
import struct

import numpy as np
from scipy.spatial import ConvexHull


def read_stl(path):
    with open(path, "rb") as f:
        head = f.read(84)
        if head[:5].lower() == b"solid" and b"facet" in open(path, "rb").read(512):
            raise ValueError("ASCII STL 은 지원하지 않는다")
        n = struct.unpack("<I", head[80:84])[0]
        buf = np.frombuffer(f.read(n * 50), dtype=np.uint8)
    if buf.size != n * 50:
        raise ValueError(f"크기 불일치: {buf.size} != {n*50}")
    rec = buf.reshape(n, 50)
    tri = rec[:, 12:48].copy().view(np.float32).reshape(n, 3, 3)
    return tri.astype(np.float64)


def write_stl(path, tri):
    n = tri.shape[0]
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    nrm = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, ln, out=np.zeros_like(nrm), where=ln > 0)
    rec = np.zeros((n, 50), np.uint8)
    blk = np.concatenate([nrm[:, None, :], tri], axis=1).astype(np.float32)
    rec[:, :48] = blk.reshape(n, 12).view(np.uint8).reshape(n, 48)
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())


def cluster(tri, d):
    """격자 d 로 정점을 묶어 삼각형을 줄인다."""
    lo = tri.reshape(-1, 3).min(0)
    key = np.floor((tri.reshape(-1, 3) - lo) / d).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    # 대표점 = 각 칸의 무게중심(격자 중심보다 형상 보존이 낫다)
    pts = tri.reshape(-1, 3)
    cen = np.zeros((uniq.shape[0], 3))
    cnt = np.bincount(inv, minlength=uniq.shape[0])[:, None]
    np.add.at(cen, inv, pts)
    cen /= np.maximum(cnt, 1)
    idx = inv.reshape(-1, 3)
    keep = (idx[:, 0] != idx[:, 1]) & (idx[:, 1] != idx[:, 2]) & (idx[:, 0] != idx[:, 2])
    idx = idx[keep]
    if idx.size == 0:
        return None
    srt = np.sort(idx, axis=1)
    _, u = np.unique(srt, axis=0, return_index=True)
    idx = idx[np.sort(u)]
    return cen[idx]


def decimate(tri, max_faces, log=print):
    if tri.shape[0] <= max_faces:
        return tri, None
    ext = tri.reshape(-1, 3).ptp(0).max()
    lo, hi = ext / 4000.0, ext / 4.0          # d 이분탐색 범위
    best = None
    for _ in range(40):
        d = (lo + hi) / 2
        out = cluster(tri, d)
        n = 0 if out is None else out.shape[0]
        if n <= max_faces and n > 0:
            best = (out, d)
            hi = d                            # 더 촘촘히(면수 늘리기) 시도
        else:
            lo = d
        if hi / lo < 1.001:
            break
    return (best[0], best[1]) if best else (tri, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-faces", type=int, default=60000)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)
    print(f"{'file':<40}{'원본':>10}{'감면':>10}{'비율':>8}  격자d")
    for f in sorted(os.listdir(a.src)):
        if not f.upper().endswith(".STL"):
            continue
        tri = read_stl(os.path.join(a.src, f))
        out, d = decimate(tri, a.max_faces)
        write_stl(os.path.join(a.dst, f), out)
        # ★**볼록껍질**로 검증한다 — MuJoCo 의 메시 충돌이 쓰는 게 바로 그것이다.
        #   ⚠부호부피(Σ v0·(v1×v2)/6)를 쓰면 안 된다. 그건 **닫힌 메시**에서만 유효한데
        #     클러스터링은 퇴화삼각형을 버려 구멍을 만든다. 그 값으로 재면 감면이
        #     멀쩡한데도 "+10~20% 부풀었다" 는 엉터리 숫자가 나온다(2026-08-11 실제 오진).
        pa, pb = tri.reshape(-1, 3), out.reshape(-1, 3)   # ★변수명 a 금지(argparse 와 충돌)
        ha, hb = ConvexHull(pa), ConvexHull(pb)
        dev = -1e9
        for i in range(0, len(pb), 20000):            # 메모리 절약(면 수십만이면 GB 단위)
            dev = max(dev, float((pb[i:i+20000] @ ha.equations[:, :3].T
                                  + ha.equations[:, 3]).max()))
        print(f"{f:<40}{tri.shape[0]:>10,}{out.shape[0]:>10,}"
              f"{out.shape[0]/tri.shape[0]*100:>7.1f}%  "
              f"{'—' if d is None else f'{d*1000:.2f}mm'}   "
              f"껍질부피 {(hb.volume/ha.volume-1)*100:+.2f}% · 최대편차 {dev*1000:+.2f}mm")


if __name__ == "__main__":
    main()
