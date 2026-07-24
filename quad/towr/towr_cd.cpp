// TOWR-in-CasADi ★C++ 포팅 — towr_cd.py의 SRBD phase-based 궤적최적화를 CasADi C++로 이식.
//   목적: Python 오버헤드 제거 + C++ 스택 통합. IPOPT/CasADi 그대로. 결과 JSON은 towr_track_B.py 호환.
//   빌드: g++ -O2 towr_cd.cpp -I$ENV/include -L$ENV/lib -lcasadi -o towr_cd  (ENV=proxddp env)
//   실행: TERRAIN=flat GAIT=trot ./towr_cd  (env: TERRAIN GAIT N DT XGOAL TG DUTY X0 X1 H DEPTH PLAT TG_SLOW GAP_R OUT VERBOSE)
#include <casadi/casadi.hpp>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <cstdio>
using namespace casadi;
using std::string; using std::vector; using std::map;

// ───────── 로봇 SRBD 파라미터 (towr_cd.py와 동일) ─────────
static const double MASS=38.016, G=9.81, BASE_H=0.50, MU=0.6;
static const double INER[3]={0.941,2.521,2.236};
static const double ROM_DXY[2]={0.13,0.09};
static const double FOOT_Z_LO=-0.56, FOOT_Z_HI=-0.40;
static const double F_MAX=2.0*MASS*G;
static const vector<string> FEET={"FL","FR","HL","HR"};
static map<string,std::array<double,2>> P_NOM = {
  {"FL",{{ 0.30, 0.16}}},{"FR",{{ 0.30,-0.16}}},
  {"HL",{{-0.30, 0.16}}},{"HR",{{-0.30,-0.16}}}};

static double envd(const char*k,double d){const char*v=getenv(k);return v?atof(v):d;}
static int    envi(const char*k,int d){const char*v=getenv(k);return v?atoi(v):d;}
static string envs(const char*k,const string&d){const char*v=getenv(k);return v?string(v):d;}

// 지형 높이 h(x) — 심볼릭(MX). y 미사용(평지·terrain=x함수).
static MX terr(const MX& x, const string& kind, const map<string,double>& tk){
  if(kind=="step"){ double x0=tk.at("x0"),h=tk.at("h"),s=40.0; return h*0.5*(1+tanh(s*(x-x0))); }
  if(kind=="gap"){ double x0=tk.at("x0"),x1=tk.at("x1"),d=tk.at("depth"),s=60.0; return -d*0.5*(tanh(s*(x-x0))-tanh(s*(x-x1))); }
  if(kind=="platgap"){ double x0=tk.at("x0"),x1=tk.at("x1"),plat=tk.at("plat"),s=60.0; return plat-plat*0.5*(tanh(s*(x-x0))-tanh(s*(x-x1))); }
  return 0.0*x;   // flat
}
// 지형 높이 — 수치(double)
static double terrd(double x, const string& kind, const map<string,double>& tk){
  if(kind=="step"){ double x0=tk.at("x0"),h=tk.at("h"),s=40.0; return h*0.5*(1+std::tanh(s*(x-x0))); }
  if(kind=="gap"){ double x0=tk.at("x0"),x1=tk.at("x1"),d=tk.at("depth"),s=60.0; return -d*0.5*(std::tanh(s*(x-x0))-std::tanh(s*(x-x1))); }
  if(kind=="platgap"){ double x0=tk.at("x0"),x1=tk.at("x1"),plat=tk.at("plat"),s=60.0; return plat-plat*0.5*(std::tanh(s*(x-x0))-std::tanh(s*(x-x1))); }
  return 0.0;
}

// DM(3×M)을 JSON [[row0],[row1],[row2]] 로 (column-major get_elements: idx=k*3+i)
static string dm2json(const DM& M){
  vector<double> e=M.get_elements(); int nr=M.size1(), nc=M.size2();
  std::ostringstream o; o<<"[";
  for(int i=0;i<nr;i++){ o<<(i?",":"")<<"[";
    for(int k=0;k<nc;k++) o<<(k?",":"")<<e[k*nr+i];
    o<<"]"; }
  o<<"]"; return o.str();
}

int main(){
  string kind=envs("TERRAIN","flat"), gait=envs("GAIT","trot");
  int N=envi("N",40); double dt=envd("DT",0.02);
  double x_goal=envd("XGOAL",0.8);
  double Tg=envd("TG", gait=="crawl"?0.80:0.40);
  double duty=envd("DUTY", gait=="crawl"?0.8:0.5);
  double phase_off=0.0;
  map<string,double> tk;
  if(kind=="step"){ tk["x0"]=envd("X0",0.6); tk["h"]=envd("H",0.10); }
  if(kind=="gap"){ tk["x0"]=envd("X0",0.6); tk["x1"]=envd("X1",0.85); tk["depth"]=envd("DEPTH",0.30); }
  if(kind=="platgap"){ tk["x0"]=envd("X0",1.0); tk["x1"]=envd("X1",1.3); tk["plat"]=envd("PLAT",0.20);
                       tk["tg_slow"]=envd("TG_SLOW",envd("TG",0.40)); tk["gap_r"]=envd("GAP_R",0.45); }

  double T=N*dt, vx_des=x_goal/T;
  bool platgap = (kind=="platgap");
  double _bl = platgap ? tk["plat"] : NAN;
  auto bhref=[&](double x){ return platgap ? _bl : terrd(x,kind,tk); };

  // ── 접촉 스케줄(가변 cadence) ──
  vector<string> CRAWL={"FL","HR","FR","HL"};
  double tgs = tk.count("tg_slow")? tk["tg_slow"] : Tg;
  bool has_gapc = (tk.count("x0") && tgs>Tg);
  double gap_c = has_gapc? 0.5*(tk["x0"]+tk["x1"]) : 0.0;
  double gap_r = tk.count("gap_r")? tk["gap_r"] : 0.45;
  vector<double> ph_arr(N+1,0.0);
  for(int k=1;k<=N;k++){ double xk=x_goal*(k-0.5)/N;
    double tgl=(has_gapc && std::abs(xk-gap_c)<gap_r)? Tg : tgs;
    ph_arr[k]=ph_arr[k-1]+dt/tgl; }
  auto in_stance=[&](const string&foot,int k)->bool{
    double ph=std::fmod(ph_arr[k]+phase_off,1.0); if(ph<0)ph+=1;
    if(gait=="crawl"){ int idx=(int)(std::find(CRAWL.begin(),CRAWL.end(),foot)-CRAWL.begin());
      double win_lo=idx*0.25, sw=0.25*duty; return !(win_lo<=ph && ph<win_lo+sw); }
    double off=(foot=="FL"||foot=="HR")?0.0:0.5;
    double p=std::fmod(ph-off,1.0); if(p<0)p+=1; return p<duty; };
  map<string,vector<int>> contact;
  for(auto&f:FEET){ contact[f].resize(N+1); for(int k=0;k<=N;k++) contact[f][k]=in_stance(f,k)?1:0; }

  // ── Opti 문제 ──
  Opti opti;
  MX P=opti.variable(3,N+1), Th=opti.variable(3,N+1);
  map<string,MX> Ft,Fr;
  for(auto&f:FEET){ Ft[f]=opti.variable(3,N+1); Fr[f]=opti.variable(3,N+1); }
  DM g_vec=DM(vector<double>{0,0,-G});
  MX J=0;

  double z0=BASE_H+terrd(0.0,kind,tk);
  opti.subject_to(P(Slice(),0)==DM(vector<double>{0,0,z0}));
  opti.subject_to(Th(Slice(),0)==DM(vector<double>{0,0,0}));
  for(auto&f:FEET){
    opti.subject_to(Ft[f](0,0)==P_NOM[f][0]);
    opti.subject_to(Ft[f](1,0)==P_NOM[f][1]);
    opti.subject_to(Ft[f](2,0)==terrd(P_NOM[f][0],kind,tk));
  }

  for(int k=0;k<=N;k++){
    MX pk=P(Slice(),k), thk=Th(Slice(),k);
    MX Fsum=DM(vector<double>{0,0,0}), Msum=DM(vector<double>{0,0,0});
    for(auto&f:FEET){
      MX fk=Fr[f](Slice(),k), rk=Ft[f](Slice(),k);
      Fsum=Fsum+fk;
      Msum=Msum+cross(rk-pk,fk);
      if(contact[f][k]){
        opti.subject_to(fk(2)>=0);
        opti.subject_to(fk(2)<=F_MAX);
        opti.subject_to(fk(0)<=MU*fk(2)); opti.subject_to(-fk(0)<=MU*fk(2));
        opti.subject_to(fk(1)<=MU*fk(2)); opti.subject_to(-fk(1)<=MU*fk(2));
        opti.subject_to(rk(2)==terr(rk(0),kind,tk));
        if(k>0 && contact[f][k-1]){
          opti.subject_to(Ft[f](0,k)==Ft[f](0,k-1));
          opti.subject_to(Ft[f](1,k)==Ft[f](1,k-1));
        }
      } else {
        opti.subject_to(fk==DM(vector<double>{0,0,0}));
        opti.subject_to(rk(2)>=terr(rk(0),kind,tk));
      }
      MX dx=rk(0)-(pk(0)+P_NOM[f][0]), dy=rk(1)-(pk(1)+P_NOM[f][1]);
      opti.subject_to(dx<=ROM_DXY[0]); opti.subject_to(-dx<=ROM_DXY[0]);
      opti.subject_to(dy<=ROM_DXY[1]); opti.subject_to(-dy<=ROM_DXY[1]);
      opti.subject_to(rk(2)-pk(2)>=FOOT_Z_LO); opti.subject_to(rk(2)-pk(2)<=FOOT_Z_HI);
    }
    if(0<k && k<N){
      MX acc=(P(Slice(),k+1)-2*P(Slice(),k)+P(Slice(),k-1))/(dt*dt);
      opti.subject_to(MASS*acc==Fsum+MASS*g_vec);
      MX angacc=(Th(Slice(),k+1)-2*Th(Slice(),k)+Th(Slice(),k-1))/(dt*dt);
      for(int i=0;i<3;i++) opti.subject_to(INER[i]*angacc(i)==Msum(i));
    }
    if(k>0){ MX vx=(P(0,k)-P(0,k-1))/dt; J=J+5.0*pow(vx-vx_des,2); }
    J=J+50.0*(pow(thk(0),2)+pow(thk(1),2)+pow(thk(2),2));
    J=J+20.0*pow(P(1,k),2);
    MX bh = platgap? MX(_bl) : terr(P(0,k),kind,tk);
    J=J+80.0*pow(P(2,k)-(bh+BASE_H),2);
    if(0<k && k<N){
      J=J+2.0*sumsqr((P(Slice(),k+1)-P(Slice(),k-1))/(2*dt)-DM(vector<double>{vx_des,0,0}));
      J=J+5.0*sumsqr((Th(Slice(),k+1)-Th(Slice(),k-1))/(2*dt));
    }
    for(auto&f:FEET) J=J+1e-4*sumsqr(Fr[f](Slice(),k));
  }
  opti.minimize(J);

  // ── 초기추정 ──
  for(int k=0;k<=N;k++){
    double xk=x_goal*k/N;
    opti.set_initial(P(Slice(),k), DM(vector<double>{xk,0,BASE_H+bhref(xk)}));
    for(auto&f:FEET){
      double fx=xk+P_NOM[f][0], fz=terrd(fx,kind,tk);
      if(platgap && fz<_bl-1e-3) fz=_bl;
      opti.set_initial(Ft[f](Slice(),k), DM(vector<double>{fx,P_NOM[f][1],fz}));
      if(contact[f][k]) opti.set_initial(Fr[f](Slice(),k), DM(vector<double>{0,0,MASS*G/2}));
    }
  }

  Dict popt, sopt;
  popt["print_time"]=0;
  sopt["print_level"]= getenv("VERBOSE")?5:0;
  sopt["max_iter"]=800; sopt["tol"]=1e-4; sopt["acceptable_tol"]=1e-3;
  opti.solver("ipopt", popt, sopt);

  try{
    OptiSol sol=opti.solve();
    DM Pv=sol.value(P), Thv=sol.value(Th);
    map<string,DM> Ftv,Frv;
    for(auto&f:FEET){ Ftv[f]=sol.value(Ft[f]); Frv[f]=sol.value(Fr[f]); }
    vector<double> pe=Pv.get_elements();
    double x0=pe[0], xe=pe[N*3+0];
    double zmin=1e9,zmax=-1e9; for(int k=0;k<=N;k++){ double z=pe[k*3+2]; zmin=std::min(zmin,z); zmax=std::max(zmax,z);}
    printf("[TOWR-C++] ✅ solve 성공  x:%.3f→%.3f  z:%.3f~%.3f  총시간%.2fs  vx_avg=%.3f\n",
           x0,xe,zmin,zmax,T,(xe-x0)/T);

    // ── JSON 저장 (towr_track_B.py 호환: dt,N,P,Th,Ft,Fr,contact) ──
    string outf=envs("OUT","/home/jsh/문서/jsh/simulation/quad/towr/traj_"+kind+"_cpp.json");
    std::ostringstream o; o.precision(10);
    o<<"{\"dt\":"<<dt<<",\"N\":"<<N<<",\"kind\":\""<<kind<<"\",\"Tg\":"<<Tg<<",\"duty\":"<<duty
     <<",\"gait\":\""<<gait<<"\",\"tkw\":{}";
    o<<",\"P\":"<<dm2json(Pv)<<",\"Th\":"<<dm2json(Thv);
    o<<",\"Ft\":{"; bool first=true; for(auto&f:FEET){ o<<(first?"":",")<<"\""<<f<<"\":"<<dm2json(Ftv[f]); first=false;} o<<"}";
    o<<",\"Fr\":{"; first=true; for(auto&f:FEET){ o<<(first?"":",")<<"\""<<f<<"\":"<<dm2json(Frv[f]); first=false;} o<<"}";
    o<<",\"contact\":{"; first=true; for(auto&f:FEET){ o<<(first?"":",")<<"\""<<f<<"\":[";
      for(int k=0;k<=N;k++) o<<(k?",":"")<<contact[f][k]; o<<"]"; first=false;} o<<"}";
    o<<"}";
    std::ofstream of(outf); of<<o.str(); of.close();
    printf("[TOWR-C++] 궤적 저장: %s\n", outf.c_str());
  }catch(std::exception&e){
    printf("[TOWR-C++] solve 실패: %.200s\n", e.what());
    return 1;
  }
  return 0;
}
