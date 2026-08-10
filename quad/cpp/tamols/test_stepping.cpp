// test_stepping.cpp — TAMOLS(walk+COM_W)가 2D stepping-stone서 발판을 돌 위에 놓나 feasibility 확인.
//   돌=높음(z=0), gap=낮음(z=-0.5). online_replan 후 각 발판 xy의 지형높이로 "돌 위(0)냐 gap(-0.5)이냐" 판정.
//   walk 정식화(c.walk) + COM_W(마찰-sway 억제, GIAC 논의)로 균형포함 발판을 기대.
#include "tamols_online.hpp"
#include <cstdio>
#include <cmath>
#include <cstdlib>
using namespace tamols;

// (x,y)가 돌 위면 0, gap이면 -0.5. 돌 격자: pitch=stone+gap, 각 셀 [0,stone)이 돌.
static double stone_h(double x, double y, double stone, double gap) {
  double pitch = stone + gap;
  auto onstone = [&](double v){ double m = v - pitch*std::floor(v/pitch); return m < stone; };
  return (onstone(x) && onstone(y)) ? 0.0 : -0.5;
}

static void run(const char* name, bool walk, bool com, double stone, double gap) {
  if (com) setenv("COM_W","400",1); else unsetenv("COM_W");
  if (walk) setenv("WALK_QLEG","1",1); else unsetenv("WALK_QLEG");
  int N = 61; double cell = 0.04, off = cell*N/2.0; int ms = N;
  Grid h = Grid::Zero(N,N);
  for (int i=0;i<N;i++) for(int j=0;j<N;j++){ double x=i*cell-off, y=j*cell-off; h(i,j)=stone_h(x,y,stone,gap); }
  Params prm;
  OnlineCfg c; c.vadv=0.25; c.phase_dur = walk?0.175:0.2; c.walk=walk; c.straddle_init=false;
  double z0=0.52, base_x=0, vx0=0; TamolsState st;
  int nrep=6, on_stone=0, nfeet=0; double symax=0;
  for (int rp=0; rp<nrep; rp++){
    Eigen::Matrix<double,4,3> foot;
    for(int l=0;l<4;l++){ foot(l,0)=prm.hip_offsets(l,0); foot(l,1)=prm.hip_offsets(l,1);
      foot(l,2)=bilinear_height(h,cell,ms, base_x+prm.hip_offsets(l,0), prm.hip_offsets(l,1)); }
    c.warm=(rp>0); c.rti_iter=(rp>0)?5:60;
    QpResult r = online_replan(st,h,cell,ms,z0,0,vx0,0,foot,c);
    for(int k=0;k<st.num_phases();k++){ auto p=st.pos_at(k, st.gait[k].duration*0.5); if(std::abs(p(1))>symax) symax=std::abs(p(1)); }
    for(int l=0;l<4;l++){ double fx=base_x+st.p(l,0), fy=st.p(l,1);
      double terr = stone_h(fx,fy,stone,gap);  // 발판 xy의 실제 지형(돌0 / gap-0.5)
      if (terr > -0.25) on_stone++;   // 돌 위
      nfeet++; }
    vx0=std::min(c.vadv, vx0+0.08); base_x+=0.10;
  }
  std::printf("  [%-16s] 발판 돌위 비율 = %d/%d (%.0f%%)  max|y_drift|=%.3fm\n",
    name, on_stone, nfeet, 100.0*on_stone/nfeet, symax);
}

int main(){
  std::printf("== TAMOLS stepping 발판 feasibility (돌 위에 놓나) ==\n");
  std::printf("   돌위비율 높을수록 좋음(gap 회피) · walk+COM_W가 균형포함 발판 기대\n");
  struct { const char* nm; double s, g; } lv[] = {{"쉬움 돌0.5/gap0.1",0.5,0.1},{"중간 돌0.4/gap0.13",0.4,0.13},{"어려움 돌0.3/gap0.2",0.3,0.2}};
  for (auto& L : lv){
    std::printf("\n----- %s -----\n", L.nm);
    run("trot",        false, false, L.s, L.g);
    run("walk",        true,  false, L.s, L.g);
    run("walk+COM_W",  true,  true,  L.s, L.g);
  }
  return 0;
}
