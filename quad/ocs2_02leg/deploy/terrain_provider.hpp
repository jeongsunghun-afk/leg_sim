#pragma once
// D1 배포: 추상 지형 제공자.
//   MjTerrainSdf의 유일한 MuJoCo 결합은 update()(mj_ray)뿐 — 나머지(height/getValue/
//   getLinearApproximation)는 내부 그리드 질의라 MuJoCo 무관. 그래서 매틱 SDF 그리드를
//   '채우는 소스'만 이 인터페이스로 추상화한다.
//     sim  : MujocoTerrain  → sdf.update(m,d,cx,cy)  (mj_ray group2 heightmap)
//     real : (미구현) 인지 elevation-map/pointcloud로 sdf 그리드 채움
//   D1Controller는 매 tick terrain.fill(sdf, cx, cy)만 호출하고 MuJoCo를 모른다.
#include "mj_terrain_sdf.hpp"
#include <functional>
#include <utility>

namespace d1 {

class TerrainProvider {
 public:
  virtual ~TerrainProvider() = default;
  // 로봇중심(cx,cy) 주변 heightmap을 sdf 그리드에 채운다.
  virtual void fill(MjTerrainSdf& sdf, double cx, double cy) = 0;
};

// ★실기 인지 heightmap 제공자 — height(x,y) 콜백(elevation map 질의)으로 SDF 그리드 채움.
//   real 배포 플러그: 인지스택(depth/pointcloud→elevation map)이 heightAt를 공급하면 컨트롤러 불변.
//   sim은 MujocoTerrain(mj_ray) 사용. 이건 실데이터 배선 지점(현재 콜백만·인지스택 미연결).
class HeightmapTerrainProvider : public TerrainProvider {
  std::function<double(double, double)> heightAt_;
 public:
  explicit HeightmapTerrainProvider(std::function<double(double, double)> heightAt) : heightAt_(std::move(heightAt)) {}
  void fill(MjTerrainSdf& sdf, double cx, double cy) override { sdf.setGrid(cx, cy, heightAt_); }
};

}  // namespace d1
