#pragma once
// terrain_map.hpp — RPET_TERRAIN_MAP_INTEGRATION.md P0 구현 (sim 백엔드).
//   백엔드 무관 TerrainMap 계약(§1) + sim 백엔드 MjRayTerrainMap(mj_ray를 로봇중심 격자로 캐시).
//   파생 레이어(filtered/slope/roughness/edgeSDF/footScore, §3)는 update()에서만 계산(1 kHz 루프 밖, 원칙 2).
//   더블버퍼 + atomic 스왑으로 lock-free read(§4.2). 좌표는 항상 (x,y) odom 절대로 쿼리(인덱스 보관 금지, §4.1).
//   실기 백엔드(ElevationMapAdapter)는 §7 확정 후 같은 계약을 구현 → 소비자 코드 불변.
#include <mujoco/mujoco.h>
#include <vector>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <deque>

// 2D float 격자(row-major, i=x, j=y)
struct Grid {
  int nx=0, ny=0; std::vector<float> d;
  void resize(int NX,int NY){ nx=NX; ny=NY; d.assign((size_t)nx*ny,0.f); }
  inline float& at(int i,int j){ return d[(size_t)j*nx+i]; }
  inline float  at(int i,int j) const { return d[(size_t)j*nx+i]; }
};

// §1 계약의 로컬 서브맵(레이어 번들). z()/score()는 (x,y) odom로 bilinear.
struct Submap {
  double res=0.05, ox=0, oy=0; int nx=0, ny=0;
  Grid elevation, filtered, slope, roughness, edgeSDF, footScore;
  std::vector<uint8_t> valid;   // per-cell 유효(레이 hit)
  uint64_t stamp_ns=0;

  inline bool cell(double x,double y,double& fi,double& fj) const {
    fi=(x-ox)/res; fj=(y-oy)/res;
    return fi>=0 && fj>=0 && fi<=nx-1 && fj<=ny-1;
  }
  inline bool validAt(double x,double y) const {
    double fi,fj; if(!cell(x,y,fi,fj)) return false;
    int i=(int)(fi+0.5), j=(int)(fj+0.5);
    return valid[(size_t)j*nx+i]!=0;
  }
  inline float bilinear(const Grid& g,double x,double y,bool& ok) const {
    double fi,fj; if(!cell(x,y,fi,fj)){ ok=false; return 0.f; }
    int i0=(int)std::floor(fi), j0=(int)std::floor(fj);
    int i1=std::min(i0+1,nx-1), j1=std::min(j0+1,ny-1);
    double a=fi-i0, b=fj-j0; ok=true;
    return (float)((1-a)*(1-b)*g.at(i0,j0)+a*(1-b)*g.at(i1,j0)
                  +(1-a)*b*g.at(i0,j1)+a*b*g.at(i1,j1));
  }
  inline double z(double x,double y) const { bool ok; float v=bilinear(filtered,x,y,ok); return ok? v : -100.0; }
  inline float  score(double x,double y) const { bool ok; float v=bilinear(footScore,x,y,ok); return ok? v : 0.f; }
};

// sim 백엔드: mj_ray 하향캐스트를 로봇중심 격자로 래스터 → 레이어 계산 → 더블버퍼 공개.
struct MjRayTerrainMap {
  // ── config (튜닝 대상) ──
  double res=0.05, half=1.2;                     // 셀크기[m], 로봇중심 창 half-extent[m]
  double ray_z0=2.0, drop_thresh=0.05;           // 레이 시작높이, 엣지 판정 낙차[m]
  double slope_max=25.0*M_PI/180.0, rough_max=0.03;
  // ★점접촉(sphere 발) 주의: 02_Leg 발=점접촉(발바닥 패치 없음). footScore=패치 밀착이 아니라 "점 지지 안전".
  //   → margin = sphere 반경(foot_r) + 발배치 불확실성(σ_place). σ_place는 sim2real 특성화(latency/노이즈 발끝오차, RPET §6)서 근거.
  //   가중: edgeSDF 지배(굴러떨어짐/갭 회피), slope 보조(미끄러짐), roughness 저비중(패치 밀착 무의미, 뾰족돌기/구덩이만).
  double foot_r=0.05, placement_margin=0.03;     // margin = foot_r + placement_margin.
  // ★placement_margin=σ_place 근거화(§6, 2026-07-14): 배포노이즈+지연5ms EST_CTRL서 발배치오차 측정
  //   = base수평위치오차(~0.7cm) + vel오차×T_sw(~0.9cm) + 자세오차×reach(~0.3cm) ≈ 1.3cm RMS → 2σ≈0.03m.
  double w1=0.4, w2=0.2, w3=0.4;                 // (slope, roughness, edge) — 점접촉이라 edge/ slope 위주
  inline double margin() const { return foot_r + placement_margin; }
  int NX=0, NY=0;
  Submap buf[2]; std::atomic<Submap*> front{nullptr}; int back_idx=0;

  void init(){
    NX=(int)std::round(2*half/res)+1; NY=NX;
    for(int b=0;b<2;b++){ Submap& s=buf[b]; s.res=res; s.nx=NX; s.ny=NY;
      s.elevation.resize(NX,NY); s.filtered.resize(NX,NY); s.slope.resize(NX,NY);
      s.roughness.resize(NX,NY); s.edgeSDF.resize(NX,NY); s.footScore.resize(NX,NY);
      s.valid.assign((size_t)NX*NY,0); }
  }

  // ★맵 갱신 — 맵 rate에서만 호출(1 kHz 루프 밖). cx,cy=로봇 xy(odom), stamp_ns=맵 타임스탬프.
  void update(mjModel* m, mjData* d, double cx, double cy, uint64_t stamp_ns){
    if(NX==0) init();
    Submap& s=buf[back_idx];
    s.ox=cx-half; s.oy=cy-half; s.stamp_ns=stamp_ns;
    static const mjtByte gg[6]={0,0,1,0,0,0};   // group2(지형)만 — terrain_z와 동일 마스킹
    // ── 1) elevation 래스터 ──
    for(int j=0;j<NY;j++) for(int i=0;i<NX;i++){
      mjtNum pnt[3]={s.ox+i*res, s.oy+j*res, ray_z0}, vec[3]={0,0,-1}, nrm[3]; int gid=-1;
      mjtNum dist=mj_ray(m,d,pnt,vec,gg,1,-1,&gid,nrm);
      if(dist>=0 && gid>=0){ s.elevation.at(i,j)=(float)(ray_z0-dist); s.valid[(size_t)j*NX+i]=1; }
      else { s.elevation.at(i,j)=0.f; s.valid[(size_t)j*NX+i]=0; }
    }
    computeLayers(s);
    front.store(&s, std::memory_order_release);   // atomic 공개(§4.2)
    back_idx ^= 1;
  }

  // ── 계약 API (lock-free read) ──
  const Submap* map() const { return front.load(std::memory_order_acquire); }
  double z(double x,double y) const     { const Submap* s=map(); return s? s->z(x,y) : -100.0; }
  bool   valid(double x,double y) const { const Submap* s=map(); return s? s->validAt(x,y) : false; }
  float  footScore(double x,double y) const { const Submap* s=map(); return s? s->score(x,y) : 0.f; }

  // ★P1 nudge용: (x,y) 주변 반경 r 내 footScore 최고 셀 → (bx,by). 없으면 입력 유지.
  bool bestFoot(double x,double y,double r,double& bx,double& by) const {
    const Submap* s=map(); if(!s){ bx=x; by=y; return false; }
    double fi,fj; if(!s->cell(x,y,fi,fj)){ bx=x; by=y; return false; }
    int R=(int)std::ceil(r/s->res), ci=(int)(fi+0.5), cj=(int)(fj+0.5);
    float best=-1.f; bx=x; by=y;
    for(int dj=-R;dj<=R;dj++) for(int di=-R;di<=R;di++){
      int i=ci+di, j=cj+dj; if(i<0||j<0||i>=s->nx||j>=s->ny) continue;
      if(di*di+dj*dj>R*R) continue;
      if(!s->valid[(size_t)j*s->nx+i]) continue;
      float sc=s->footScore.at(i,j);
      if(sc>best){ best=sc; bx=s->ox+i*s->res; by=s->oy+j*s->res; }
    }
    return best>=0.f;
  }

private:
  inline float clamp01(float v){ return v<0.f?0.f:(v>1.f?1.f:v); }

  void computeLayers(Submap& s){
    const int nx=s.nx, ny=s.ny; const double res=s.res;
    auto vld=[&](int i,int j){ return i>=0&&j>=0&&i<nx&&j<ny && s.valid[(size_t)j*nx+i]; };
    // ── 2) filtered = 3×3 평균(유효셀만) ──
    for(int j=0;j<ny;j++) for(int i=0;i<nx;i++){
      if(!vld(i,j)){ s.filtered.at(i,j)=s.elevation.at(i,j); continue; }
      float sum=0; int n=0;
      for(int b=-1;b<=1;b++) for(int a=-1;a<=1;a++) if(vld(i+a,j+b)){ sum+=s.elevation.at(i+a,j+b); n++; }
      s.filtered.at(i,j) = n? sum/n : s.elevation.at(i,j);
    }
    // ── 3) slope(그래디언트→atan) + roughness(3×3 표준편차) ──
    for(int j=0;j<ny;j++) for(int i=0;i<nx;i++){
      int ip=std::min(i+1,nx-1), im=std::max(i-1,0), jp=std::min(j+1,ny-1), jm=std::max(j-1,0);
      double gx=(s.filtered.at(ip,j)-s.filtered.at(im,j))/((ip-im)*res+1e-9);
      double gy=(s.filtered.at(i,jp)-s.filtered.at(i,jm))/((jp-jm)*res+1e-9);
      s.slope.at(i,j)=(float)std::atan(std::hypot(gx,gy));
      float sum=0,sq=0; int n=0;
      for(int b=-1;b<=1;b++) for(int a=-1;a<=1;a++){ int ii=i+a,jj=j+b; if(ii<0||jj<0||ii>=nx||jj>=ny) continue;
        float v=s.filtered.at(ii,jj); sum+=v; sq+=v*v; n++; }
      float mean=n?sum/n:0; s.roughness.at(i,j)= n? (float)std::sqrt(std::max(0.f,sq/n-mean*mean)) : 0.f;
    }
    // ── 4) edgeSDF: 낙차>drop_thresh 셀 집합까지 BFS 거리(셀→×res). 엣지 셀=0 ──
    std::vector<int> distc((size_t)nx*ny, -1);
    std::deque<std::pair<int,int>> q;
    for(int j=0;j<ny;j++) for(int i=0;i<nx;i++){
      bool edge=false; float c=s.filtered.at(i,j);
      const int di4[4]={1,-1,0,0}, dj4[4]={0,0,1,-1};
      for(int k=0;k<4;k++){ int ii=i+di4[k],jj=j+dj4[k]; if(ii<0||jj<0||ii>=nx||jj>=ny) continue;
        if(std::fabs(c-s.filtered.at(ii,jj))>drop_thresh){ edge=true; break; } }
      if(edge){ distc[(size_t)j*nx+i]=0; q.push_back({i,j}); }
    }
    while(!q.empty()){ auto[ci,cj]=q.front(); q.pop_front(); int cd=distc[(size_t)cj*nx+ci];
      const int di8[8]={1,-1,0,0,1,1,-1,-1}, dj8[8]={0,0,1,-1,1,-1,1,-1};
      for(int k=0;k<8;k++){ int ii=ci+di8[k],jj=cj+dj8[k]; if(ii<0||jj<0||ii>=nx||jj>=ny) continue;
        if(distc[(size_t)jj*nx+ii]<0){ distc[(size_t)jj*nx+ii]=cd+1; q.push_back({ii,jj}); } } }
    bool anyEdge=!q.empty(); (void)anyEdge;
    for(int j=0;j<ny;j++) for(int i=0;i<nx;i++){ int dc=distc[(size_t)j*nx+i];
      s.edgeSDF.at(i,j)= dc<0? (float)(nx*res) : (float)(dc*res); }   // 엣지 없으면 멀다(안전)
    // ── 5) footScore(§3.2) = w1·flat + w2·smooth + w3·edge, × valid ──
    for(int j=0;j<ny;j++) for(int i=0;i<nx;i++){
      if(!s.valid[(size_t)j*nx+i]){ s.footScore.at(i,j)=0.f; continue; }
      float t1=clamp01(1.f - s.slope.at(i,j)/(float)slope_max);
      float t2=clamp01(1.f - s.roughness.at(i,j)/(float)rough_max);
      float t3=clamp01(s.edgeSDF.at(i,j)/(float)margin());
      s.footScore.at(i,j)=clamp01((float)(w1*t1+w2*t2+w3*t3));
    }
  }
};
