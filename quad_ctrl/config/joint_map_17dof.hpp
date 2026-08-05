#pragma once
// 17-DOF 관절맵 — MJCF(quad_real_17dof_waist_sphere) 관절순 ↔ 실모터 Gait SHM 채널.
//   ★★전 항목 실기 축별 JOG 실측 확정 필요(chan·sign·zero·min/max/vel). 아래는 구조/순서 스캐폴드(placeholder).
//   구조: GaitJointCfg{ chan, sign, zero_deg, min_deg, max_deg, vel_max_dps }.
//     chan     = Gait SHM 채널 인덱스(실배선 맵).
//     sign     = MJCF 관절 +방향 ↔ 실모터 +방향 (±1). JOG로 각 관절 +명령 시 실회전 방향 확인.
//     zero_deg = MJCF 0 자세일 때 실모터 절대각(deg). JOG로 기준자세 정렬 시 실엔코더값.
//     min/max  = 관절 한계(deg, 관절축). 안전 clip. 실기 기구한계 실측.
//     vel_max  = 최대 각속도(deg/s). 모터스펙([[02leg-motor-spec]]: hip/thigh 210·calf/foot 300급, 관절축 환산).
//   ※real_hal이 이 map으로 deg↔rad·부호·오프셋 변환. 미측정 값으로 구동 금지(안전).
#include "hal/real_hal.hpp"
#include <vector>

#if defined(QC_HAVE_ROBOT_SHM) || __has_include("/usr/include/RobotSharedMem.h")
namespace qc {

// MJCF nu=17 순서: HL(hip,thigh,calf,foot)·HR(4)·FB_waist·FL(4)·FR(4).
//                        chan sign zero  min   max   vel      // 관절            ★=JOG 실측 TODO
inline std::vector<GaitJointCfg> joint_map_17dof() { return {
  {  0, +1,  0.0, -180, 180, 300 },   // 0  HL_hip    ★
  {  1, +1,  0.0, -180, 180, 210 },   // 1  HL_thigh  ★
  {  2, +1,  0.0, -180, 180, 300 },   // 2  HL_calf   ★
  {  3, +1,  0.0, -180, 180, 300 },   // 3  HL_foot   ★
  {  4, +1,  0.0, -180, 180, 300 },   // 4  HR_hip    ★
  {  5, +1,  0.0, -180, 180, 210 },   // 5  HR_thigh  ★
  {  6, +1,  0.0, -180, 180, 300 },   // 6  HR_calf   ★
  {  7, +1,  0.0, -180, 180, 300 },   // 7  HR_foot   ★
  {  8, +1,  0.0, -180, 180, 210 },   // 8  FB_waist  ★ (조향 스파인)
  {  9, +1,  0.0, -180, 180, 300 },   // 9  FL_hip    ★
  { 10, +1,  0.0, -180, 180, 210 },   // 10 FL_thigh  ★
  { 11, +1,  0.0, -180, 180, 300 },   // 11 FL_calf   ★
  { 12, +1,  0.0, -180, 180, 300 },   // 12 FL_foot   ★
  { 13, +1,  0.0, -180, 180, 300 },   // 13 FR_hip    ★
  { 14, +1,  0.0, -180, 180, 210 },   // 14 FR_thigh  ★
  { 15, +1,  0.0, -180, 180, 300 },   // 15 FR_calf   ★
  { 16, +1,  0.0, -180, 180, 300 },   // 16 FR_foot   ★
}; }

}  // namespace qc
#endif
