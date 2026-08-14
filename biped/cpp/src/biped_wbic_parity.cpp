// biped WBIC C++ 파리티 — dump_biped_wbic.py 출력을 읽어 C++ WBIC와 Python tau 비교.
#include "biped_wbic.hpp"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
using namespace bipedwbic;

static std::map<std::string,std::vector<double>> D;
static std::vector<double>& G(const std::string&k){ return D[k]; }

int main(int argc,char**argv){
  std::string path=argc>1?argv[1]:"/tmp/biped_wbic_dump.txt";
  std::ifstream f(path); if(!f){ std::cerr<<"덤프 없음\n"; return 1; }
  std::string line;
  while(std::getline(f,line)){ std::istringstream ss(line); std::string k; ss>>k;
    std::vector<double> v; double x; while(ss>>x) v.push_back(x); D[k]=v; }
  WbicIn in;
  in.nv=(int)G("nv")[0]; in.nu=(int)G("nu")[0]; in.Kc=(int)G("Kc")[0];
  int nv=in.nv, nu=in.nu, Kc=in.Kc;
  auto mat=[&](const std::string&k,int r,int cc){ MatrixXd m(r,cc);
    for(int i=0;i<r;i++)for(int j=0;j<cc;j++) m(i,j)=G(k)[i*cc+j]; return m; };
  auto vec=[&](const std::string&k,int n){ VectorXd v(n); for(int i=0;i<n;i++) v[i]=G(k)[i]; return v; };
  in.M=mat("M",nv,nv); in.h=vec("h",nv); in.qv=vec("qv",nv);
  in.q=vec("q",nu); for(int i=0;i<4;i++) in.qc[i]=G("qc")[i];
  for(int i=0;i<3;i++) in.com[i]=G("com")[i]; in.zref=G("zref")[0];
  in.Jc=mat("Jc",3,nv);
  for(double x:G("contacts")) in.contacts.push_back((int)x);
  for(int k=0;k<Kc;k++){ in.cjac.push_back(mat("cjac"+std::to_string(k),3,nv));
    Vector3d l; for(int i=0;i<3;i++) l[i]=G("lam"+std::to_string(k))[i]; in.lam.push_back(l); }
  in.has_swing=true; in.swing_leg=(int)G("swing_leg")[0];
  in.Jsw=mat("Jsw",3,nv);
  for(int i=0;i<3;i++){ in.sw_pos[i]=G("sw_pos")[i]; in.sw_ptgt[i]=G("sw_ptgt")[i]; in.sw_vtgt[i]=G("sw_vtgt")[i]; }
  in.Qhome=vec("Qhome",nu); in.drv_peak=vec("drv_peak",nu);   // ★2026-08-13 개명(관절→드라이브 한계)
  for(double x:G("ankle")) in.ankle_idx.push_back((int)x);
  auto& gn=G("gains");
  in.SW_KP=gn[0]; in.SW_KD=gn[1]; in.W_ORI=gn[2]; in.W_ANKLE=gn[3]; in.W_POST=gn[4];
  in.W_LAM=gn[5]; in.STANCE_KD=gn[6]; in.MU_EFF=gn[7]; in.LAMZ_MIN=gn[8];

  VectorXd tau=wbic_track(in);
  VectorXd tau_py=vec("tau",nu);
  double maxdiff=(tau-tau_py).cwiseAbs().maxCoeff();
  std::cout<<"C++ tau: "<<tau.transpose()<<"\n";
  std::cout<<"Py  tau: "<<tau_py.transpose()<<"\n";
  std::cout<<"max |C++ - Py| = "<<maxdiff<<"\n";
  std::cout<<(maxdiff<1e-5 ? "✅ 파리티 OK (<1e-5)" : "❌ 불일치")<<"\n";
  return maxdiff<1e-5?0:2;
}
