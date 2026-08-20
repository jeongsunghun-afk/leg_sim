#pragma once
// D1 배포: 추상 지형 제공자.
//   MjTerrainSdf의 유일한 MuJoCo 결합은 update()(mj_ray)뿐 — 나머지(height/getValue/
//   getLinearApproximation)는 내부 그리드 질의라 MuJoCo 무관. 그래서 매틱 SDF 그리드를
//   '채우는 소스'만 이 인터페이스로 추상화한다.
//     sim  : MujocoTerrain  → sdf.update(m,d,cx,cy)  (mj_ray group2 heightmap)
//     real : (미구현) 인지 elevation-map/pointcloud로 sdf 그리드 채움
//   D1Controller는 매 tick terrain.fill(sdf, cx, cy)만 호출하고 MuJoCo를 모른다.
#include "mj_terrain_sdf.hpp"

namespace d1 {

class TerrainProvider {
 public:
  virtual ~TerrainProvider() = default;
  // 로봇중심(cx,cy) 주변 heightmap을 sdf 그리드에 채운다.
  virtual void fill(MjTerrainSdf& sdf, double cx, double cy) = 0;
};

}  // namespace d1
