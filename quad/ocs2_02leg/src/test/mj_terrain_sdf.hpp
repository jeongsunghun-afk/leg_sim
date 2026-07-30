// D1 Phase 3a: MuJoCo mj_ray heightmap → 2.5D 높이-SDF (ocs2_perceptive DistanceTransformInterface 구현).
//   발-지형 클리어런스(EndEffectorDistanceConstraint)용. SDF(p) = p.z − h(p.x,p.y) (지형면 위 높이).
//   스윙발이 지형면 위 clearance 유지 → 요철·계단 요철 추종. (갭 발판선택은 Phase 3b=convex FootPlacement.)
//   heightmap은 매 제어틱 update()로 mj_ray 래스터(=A TerrainMap 방식). 제약이 shared_ptr 보유→in-place 갱신.
#pragma once
#include <ocs2_perceptive/distance_transform/DistanceTransformInterface.h>
#include <mujoco/mujoco.h>
#include <vector>
#include <cmath>

class MjTerrainSdf : public ocs2::DistanceTransformInterface {
 public:
  using scalar_t = ocs2::scalar_t;
  using vector3_t = ocs2::DistanceTransformInterface::vector3_t;

  // 로봇중심 격자: nx×ny 셀, cell 간격, 원점(x0,y0)은 update마다 로봇 위치로 이동.
  MjTerrainSdf(int nx = 61, int ny = 41, scalar_t cell = 0.04) : nx_(nx), ny_(ny), cell_(cell), h_(nx * ny, 0.0) {}

  // ★매 제어틱: 로봇 중심 heightmap을 mj_ray로 래스터(group2 지형만). cx,cy=로봇 base xy.
  void update(const mjModel* m, mjData* d, scalar_t cx, scalar_t cy) {
    x0_ = cx - 0.5 * (nx_ - 1) * cell_;
    y0_ = cy - 0.5 * (ny_ - 1) * cell_;
    const mjtNum dir[3] = {0, 0, -1};
    for (int iy = 0; iy < ny_; ++iy)
      for (int ix = 0; ix < nx_; ++ix) {
        mjtNum pnt[3] = {x0_ + ix * cell_, y0_ + iy * cell_, 2.0};  // 위(2m)서 아래로 레이
        int geomid = -1;
        // group2(지형) geom만: geomgroup 마스크. (A와 동일 규약)
        mjtByte geomgroup[6] = {0, 0, 1, 0, 0, 0};
        mjtNum nrm[3];
        mjtNum dist = mj_ray(m, d, pnt, dir, geomgroup, 1, -1, &geomid, nrm);
        h_[iy * nx_ + ix] = (geomid >= 0) ? (2.0 - dist) : 0.0;  // 지면높이(없으면 0=평지)
      }
  }

  // 쌍선형 보간 높이 h(x,y)
  scalar_t height(scalar_t x, scalar_t y) const {
    scalar_t fx = (x - x0_) / cell_, fy = (y - y0_) / cell_;
    int ix = (int)std::floor(fx), iy = (int)std::floor(fy);
    if (ix < 0 || iy < 0 || ix >= nx_ - 1 || iy >= ny_ - 1) return 0.0;  // 격자 밖=평지
    scalar_t tx = fx - ix, ty = fy - iy;
    scalar_t h00 = h_[iy * nx_ + ix], h10 = h_[iy * nx_ + ix + 1];
    scalar_t h01 = h_[(iy + 1) * nx_ + ix], h11 = h_[(iy + 1) * nx_ + ix + 1];
    return (1 - tx) * (1 - ty) * h00 + tx * (1 - ty) * h10 + (1 - tx) * ty * h01 + tx * ty * h11;
  }

  // SDF = z − h(x,y): 지형면 위 높이(양수=위). 스윙발이 이 값 ≥ clearance 유지.
  scalar_t getValue(const vector3_t& p) const override { return p.z() - height(p.x(), p.y()); }

  // 지형면 위 투영점(같은 xy, z=지형높이)
  vector3_t getProjectedPoint(const vector3_t& p) const override { return {p.x(), p.y(), height(p.x(), p.y())}; }

  // (값, 그래디언트). ∇(z−h) = [−∂h/∂x, −∂h/∂y, 1]. 중앙차분.
  std::pair<scalar_t, vector3_t> getLinearApproximation(const vector3_t& p) const override {
    const scalar_t e = cell_;
    scalar_t dhdx = (height(p.x() + e, p.y()) - height(p.x() - e, p.y())) / (2 * e);
    scalar_t dhdy = (height(p.x(), p.y() + e) - height(p.x(), p.y() - e)) / (2 * e);
    return {p.z() - height(p.x(), p.y()), vector3_t(-dhdx, -dhdy, 1.0)};
  }

 private:
  int nx_, ny_;
  scalar_t cell_, x0_ = 0, y0_ = 0;
  std::vector<scalar_t> h_;
};
