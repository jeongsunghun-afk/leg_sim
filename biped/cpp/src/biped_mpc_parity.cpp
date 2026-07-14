// biped MPC C++ 파리티 검증 — dump_biped_mpc.py 출력을 읽어 C++ MPC와 Python lam 비교.
// 빌드: cmake로 · 실행: ./biped_mpc_parity /tmp/biped_mpc_dump.txt
#include "biped_mpc.hpp"
#include <fstream>
#include <sstream>
#include <map>
#include <iostream>
#include <string>
using namespace bipedmpc;

int main(int argc, char** argv){
  std::string path = argc>1 ? argv[1] : "/tmp/biped_mpc_dump.txt";
  std::ifstream f(path);
  if(!f){ std::cerr<<"덤프 파일 없음: "<<path<<"\n"; return 1; }
  std::map<std::string,std::vector<double>> D;
  std::string line;
  while(std::getline(f,line)){
    std::istringstream ss(line); std::string key; ss>>key;
    std::vector<double> v; double x; while(ss>>x) v.push_back(x);
    D[key]=v;
  }
  MpcCfg c;
  c.N=(int)D["N"][0]; c.DT=D["DT"][0]; c.TOTAL_MASS=D["MASS"][0]; c.G_ACC=9.81;
  c.MU=D["MU"][0]; c.LAMZ_MIN=D["LAMZ_MIN"][0]; c.LAMZ_MAX=D["LAMZ_MAX"][0];
  for(int i=0;i<9;i++) c.I_BODY(i/3,i%3)=D["I_BODY"][i];
  for(int i=0;i<13;i++) c.Qdiag[i]=D["QDIAG"][i];
  for(int i=0;i<3;i++) c.Rdiag[i]=D["RDIAG"][i];
  Matrix<double,13,1> x0,x_ref;
  for(int i=0;i<13;i++){ x0[i]=D["X0"][i]; x_ref[i]=D["XREF"][i]; }
  std::array<int,NF> cur; for(int i=0;i<NF;i++) cur[i]=(int)std::lround(D["CS"][i]);
  std::array<Vector3d,NF> frel;
  for(int i=0;i<NF;i++) frel[i]=Vector3d(D["FP"][i*3],D["FP"][i*3+1],D["FP"][i*3+2]);
  std::vector<std::array<int,NF>> cs(c.N,cur);
  std::vector<std::array<Vector3d,NF>> fp(c.N,frel);

  auto lam = mpc_qp_plan(c,x0,cs,fp,x_ref);
  Matrix<double,NF,3> lam_py;
  for(int i=0;i<NF;i++) for(int j=0;j<3;j++) lam_py(i,j)=D["LAM"][i*3+j];

  double maxdiff=(lam-lam_py).cwiseAbs().maxCoeff();
  std::cout<<"C++  lam:\n"<<lam<<"\n";
  std::cout<<"Py   lam:\n"<<lam_py<<"\n";
  std::cout<<"max |C++ - Py| = "<<maxdiff<<"\n";
  std::cout<<(maxdiff<1e-6 ? "✅ 파리티 OK (<1e-6)" : "❌ 불일치")<<"\n";
  return maxdiff<1e-6 ? 0 : 2;
}
