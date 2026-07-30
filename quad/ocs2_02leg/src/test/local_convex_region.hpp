// D1 Phase 3b: CGAL 없는 로컬 convex 발판영역 자체생성 (legged_perceptive ConvexRegionSelector 대체).
//   MjTerrainSdf(height·∇h)만 질의. 발별: nominal 발판→유효셀 스냅→walkable 박스 성장→반평면 (A,b) 4행.
//   valid=무영역(스킵)·stanceEnd=게이팅 시각. 브리지가 매 스윙/틱 updateFoot() 호출.
//   walkable = 경사<SLOPE_MAX ∧ |h−hSeed|<STEP_TOL (같은 단=계단/갭 반대편 배제).
#pragma once
#include "mj_terrain_sdf.hpp"
#include <ocs2_core/Types.h>
#include <Eigen/Dense>
#include <array>
#include <memory>
#include <cmath>

struct FootRegionAB {  // 발판 반평면 A·p+b≥0 (nRow=4 박스, p=발 3D world)
  static constexpr int nRow = 4;
  Eigen::Matrix<double, nRow, 3> a;
  Eigen::Matrix<double, nRow, 1> b;
  FootRegionAB() { a.setZero(); b.setOnes(); }  // 기본=항상만족
};

class LocalConvexRegion {
 public:
  explicit LocalConvexRegion(std::shared_ptr<MjTerrainSdf> sdf) : sdf_(std::move(sdf)) {}

  // 파라미터 (제약이 읽음)
  const FootRegionAB& params(int i) const { return ab_[i]; }
  double stanceEnd(int i) const { return stanceEnd_[i]; }
  bool valid(int i) const { return valid_[i]; }

  // ★발 i의 발판영역 갱신: nominal(seedX,seedY) 근처 유효지형에 walkable 박스. stanceEndTime=게이팅.
  void updateFoot(int i, double seedX, double seedY, double stanceEndTime) {
    stanceEnd_[i] = stanceEndTime;
    double hSeed = sdf_->height(seedX, seedY);
    // 1) nominal이 무효면 최근접 유효셀로 스냅(나선탐색)
    if (!walkable(seedX, seedY, hSeed)) {
      bool found = false;
      for (int r = 1; r <= R_MAX && !found; ++r)
        for (int dy = -r; dy <= r && !found; ++dy)
          for (int dx = -r; dx <= r && !found; ++dx) {
            if (std::max(std::abs(dx), std::abs(dy)) != r) continue;  // 링만
            double cx = seedX + dx * CELL, cy = seedY + dy * CELL;
            if (walkable(cx, cy, sdf_->height(cx, cy))) { seedX = cx; seedY = cy; hSeed = sdf_->height(cx, cy); found = true; }
          }
      if (!found) { valid_[i] = false; return; }
    }
    // 2) 씨앗서 ±x/±y로 박스 성장(테두리 walkable일 때만)
    double hx = CELL, hy = CELL;
    while (hx < HALF_MAX && edgeWalkable(seedX, seedY, hx + CELL, hy, hSeed, true)) hx += CELL;
    while (hy < HALF_MAX && edgeWalkable(seedX, seedY, hx, hy + CELL, hSeed, false)) hy += CELL;
    setBox(i, seedX, seedY, hx, hy);
    valid_[i] = true;
  }
  void invalidate(int i) { valid_[i] = false; }

 private:
  // slope<SLOPE_MAX ∧ 같은 단
  bool walkable(double x, double y, double hSeed) const {
    auto vg = sdf_->getLinearApproximation({x, y, 0.0});  // grad=[-∂h/∂x,-∂h/∂y,1]
    double slope = std::hypot(vg.second.x(), vg.second.y());
    return slope < SLOPE_MAX && std::abs(sdf_->height(x, y) - hSeed) < STEP_TOL;
  }
  // 박스 확장 시 새 테두리(±x면 or ±y면) 셀들이 전부 walkable인지
  bool edgeWalkable(double cx, double cy, double hx, double hy, double hSeed, bool xEdge) const {
    if (xEdge) {  // x=±hx 두 변
      for (double y = cy - hy; y <= cy + hy + 1e-9; y += CELL)
        if (!walkable(cx + hx, y, hSeed) || !walkable(cx - hx, y, hSeed)) return false;
    } else {      // y=±hy 두 변
      for (double x = cx - hx; x <= cx + hx + 1e-9; x += CELL)
        if (!walkable(x, cy + hy, hSeed) || !walkable(x, cy - hy, hSeed)) return false;
    }
    return true;
  }
  void setBox(int i, double cx, double cy, double hx, double hy) {  // [cx±hx]×[cy±hy] → A·p+b≥0
    auto& r = ab_[i]; r.a.setZero();
    r.a(0, 0) = 1;  r.b(0) = hx - cx;   // p.x ≥ cx−hx
    r.a(1, 0) = -1; r.b(1) = hx + cx;   // p.x ≤ cx+hx
    r.a(2, 1) = 1;  r.b(2) = hy - cy;   // p.y ≥ cy−hy
    r.a(3, 1) = -1; r.b(3) = hy + cy;   // p.y ≤ cy+hy
  }

  std::shared_ptr<MjTerrainSdf> sdf_;
  std::array<FootRegionAB, 4> ab_;
  std::array<double, 4> stanceEnd_{0, 0, 0, 0};
  std::array<bool, 4> valid_{false, false, false, false};
  // 튜닝
  double CELL = 0.04, SLOPE_MAX = 0.47 /*tan25°*/, STEP_TOL = 0.06, HALF_MAX = 0.12;
  int R_MAX = 5;
};
