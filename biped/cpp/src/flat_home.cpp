// flat_home — 2점 평발 home 자세(Qflat8) 진단·재산출.
//
// ★왜 있나 (2026-08-13): 새 CAD(2026-08-12, 고관절 부착점 26cm 이동)로 `Qhome8`(점발)은
//   재산출됐는데 `Qflat8`(평발)이 구 CAD 값 그대로 남아 평발 자세유지가 1.29s 만에
//   낙상한다. 그 재산출을 눈대중이 아니라 **기하 조건으로** 풀기 위한 도구다.
//   (커밋 37a517e 가 Qhome8 에 한 것과 같은 성격의 작업. 그때 평발이 빠졌다.)
//
// 자세 파라미터: hip=0 고정, (thigh, calf, foot) 3개. 좌우 대칭.
// 잔차 3개 = 미지수 3개:
//   r1 = z_toe − z_heel                 밑창이 지면과 수평 (평발의 정의)
//   r2 = com_x − sole_center_x          CoM 이 지지다각형 전후 중심 (정적 안정 여유 최대)
//   r3 = com_z − target                 서는 높이
// ⚠r2 를 "0" 으로 두는 것이 핵심이다. 구 자세는 새 CAD 에서 이게 크게 벌어져 있고,
//   그 오차가 reset() 의 nominal_off 에 **스폰 시점에 굳어** 매 스텝 되풀이된다.
//
// 사용:
//   ./flat_home <mjcf>                       현재 Qflat8 진단 + 재산출값 출력
//   ./flat_home <mjcf> --eval t c f          임의 자세 진단
//   ./flat_home <mjcf> --comz Z              목표 CoM 높이 지정(기본=현 자세 유지)
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

static mjModel* m = nullptr;
static mjData*  d = nullptr;
static int g_toe[2], g_heel[2];

// 구 Qflat8 (biped_control.hpp) — 구 CAD 유래. 진단 기준점.
static const double QFLAT_OLD[3] = {0.25, -0.50, -1.14626};   // thigh, calf, foot

struct Pose { double toe_z, heel_z, com_x, com_z, sole_x, base_z; };

// (thigh,calf,foot) → 접지시킨 상태의 기하량. BipedControl::reset() 과 같은 순서로 내린다.
static Pose eval(double t, double c, double f){
  for(int i=0;i<m->nq;i++) d->qpos[i]=0;
  d->qpos[3]=1;
  const double q8[8]={0,t,c,f, 0,t,c,f};
  for(int j=0;j<8 && 7+j<m->nq;j++) d->qpos[7+j]=q8[j];
  d->qpos[2]=0.7;
  mj_forward(m,d);
  double zmin=1e9;                                   // 최저 접촉구 바닥이 지면에 닿게 내린다
  for(int l=0;l<2;l++){
    zmin=std::fmin(zmin, d->geom_xpos[g_toe[l]*3+2]  - m->geom_size[g_toe[l]*3]);
    zmin=std::fmin(zmin, d->geom_xpos[g_heel[l]*3+2] - m->geom_size[g_heel[l]*3]);
  }
  d->qpos[2]-=zmin;
  mj_forward(m,d);
  Pose p;
  p.toe_z  = 0.5*(d->geom_xpos[g_toe[0]*3+2]  + d->geom_xpos[g_toe[1]*3+2]);
  p.heel_z = 0.5*(d->geom_xpos[g_heel[0]*3+2] + d->geom_xpos[g_heel[1]*3+2]);
  double tx = 0.5*(d->geom_xpos[g_toe[0]*3]  + d->geom_xpos[g_toe[1]*3]);
  double hx = 0.5*(d->geom_xpos[g_heel[0]*3] + d->geom_xpos[g_heel[1]*3]);
  p.sole_x = 0.5*(tx+hx);
  p.com_x  = d->subtree_com[0];
  p.com_z  = d->subtree_com[2];
  p.base_z = d->qpos[2];
  return p;
}

static void residual(double t,double c,double f,double comz_tgt,double* r){
  Pose p=eval(t,c,f);
  r[0]=p.toe_z-p.heel_z;          // 밑창 수평
  r[1]=p.com_x-p.sole_x;          // CoM 을 밑창 중심에
  r[2]=p.com_z-comz_tgt;          // 높이
}

static void report(const char* tag,double t,double c,double f){
  Pose p=eval(t,c,f);
  double tx = 0.5*(d->geom_xpos[g_toe[0]*3]  + d->geom_xpos[g_toe[1]*3]);
  double hx = 0.5*(d->geom_xpos[g_heel[0]*3] + d->geom_xpos[g_heel[1]*3]);
  double half = std::fabs(tx-hx)*0.5;                      // 밑창 반길이(전후 여유의 이론상한)
  std::printf("\n  [%s]  thigh %+.6f · calf %+.6f · foot %+.6f\n", tag,t,c,f);
  std::printf("    밑창 기울기 (z_toe−z_heel) = %+8.5f m   %s\n", p.toe_z-p.heel_z,
              std::fabs(p.toe_z-p.heel_z)<1e-4?"✅수평":"❌기울어짐 = 평발이 아니다");
  std::printf("    CoM − 밑창중심 (x)         = %+8.5f m   (밑창 반길이 %.4f, 여유 %+.1f%%)\n",
              p.com_x-p.sole_x, half, half>1e-9?(1.0-std::fabs(p.com_x-p.sole_x)/half)*100.0:0.0);
  std::printf("    CoM z = %.4f · base z = %.4f · toe_x %+.4f · heel_x %+.4f\n",
              p.com_z,p.base_z,tx,hx);
}

int main(int argc,char**argv){
  const char* mjcf = argc>1?argv[1]:"../biped_flatfoot.mjcf";
  char err[1000]={0};
  m=mj_loadXML(mjcf,nullptr,err,1000);
  if(!m){ std::printf("모델 로드 실패: %s\n",err); return 1; }
  d=mj_makeData(m);
  g_toe[0] =mj_name2id(m,mjOBJ_GEOM,"HL_sphere");   g_toe[1] =mj_name2id(m,mjOBJ_GEOM,"HR_sphere");
  g_heel[0]=mj_name2id(m,mjOBJ_GEOM,"HL_sphere2");  g_heel[1]=mj_name2id(m,mjOBJ_GEOM,"HR_sphere2");
  if(g_heel[0]<0||g_heel[1]<0){
    std::printf("✗ heel 구(*_sphere2)가 없다 — 이 모델은 1점 점발이다. 평발 MJCF 를 줄 것.\n"); return 1; }

  double t=QFLAT_OLD[0], c=QFLAT_OLD[1], f=QFLAT_OLD[2];
  double comz_tgt=-1; bool eval_only=false;
  for(int i=2;i<argc;i++){
    if(!std::strcmp(argv[i],"--eval") && i+3<argc){
      t=atof(argv[i+1]); c=atof(argv[i+2]); f=atof(argv[i+3]); i+=3; eval_only=true; }
    else if(!std::strcmp(argv[i],"--comz") && i+1<argc){ comz_tgt=atof(argv[i+1]); i++; }
  }

  std::printf("모델: %s\n", mjcf);
  // ★--eval 은 **진단 전용**이다. 종전엔 여기서도 Newton 을 돌려서, 서지 못하는 자세를
  //   초기값으로 주면 349° 같은 발산값을 뱉었다 — 그걸 해로 오해하기 딱 좋다.
  if(eval_only){ report("진단(--eval)", t,c,f); mj_deleteData(d); mj_deleteModel(m); return 0; }
  report("현재 Qflat8", t,c,f);

  Pose p0=eval(t,c,f);
  if(comz_tgt<0) comz_tgt=p0.com_z;    // ★기본: 현 자세의 높이를 유지한다. 고치는 건 정렬뿐이다
  std::printf("\n  목표 CoM 높이 = %.4f m %s\n", comz_tgt,
              comz_tgt==p0.com_z?"(현 자세 유지 — 바꾸는 건 정렬뿐)":"(지정값)");

  // ── Newton (수치 야코비안) ─────────────────────────────────────────────
  double x[3]={t,c,f}, r[3];
  const double EPS=1e-6;
  for(int it=0; it<80; it++){
    residual(x[0],x[1],x[2],comz_tgt,r);
    double n=std::fmax(std::fabs(r[0]),std::fmax(std::fabs(r[1]),std::fabs(r[2])));
    if(n<1e-9){ std::printf("  Newton 수렴 %d회 · 잔차 %.2e\n",it,n); break; }
    double J[3][3], rp[3];
    for(int k=0;k<3;k++){
      double sav=x[k]; x[k]=sav+EPS;
      residual(x[0],x[1],x[2],comz_tgt,rp);
      x[k]=sav;
      for(int i=0;i<3;i++) J[i][k]=(rp[i]-r[i])/EPS;
    }
    // 3×3 가우스 소거 (부분 피벗)
    double A[3][4];
    for(int i=0;i<3;i++){ for(int j=0;j<3;j++) A[i][j]=J[i][j]; A[i][3]=-r[i]; }
    for(int col=0;col<3;col++){
      int piv=col; for(int i=col+1;i<3;i++) if(std::fabs(A[i][col])>std::fabs(A[piv][col])) piv=i;
      if(std::fabs(A[piv][col])<1e-12){ std::printf("  ✗ 야코비안 특이 — 해를 못 찾는다\n"); mj_deleteData(d); mj_deleteModel(m); return 1; }
      if(piv!=col) for(int j=0;j<4;j++) std::swap(A[col][j],A[piv][j]);
      for(int i=0;i<3;i++){ if(i==col) continue; double fct=A[i][col]/A[col][col];
        for(int j=col;j<4;j++) A[i][j]-=fct*A[col][j]; }
    }
    for(int i=0;i<3;i++) x[i]+=A[i][3]/A[i][i];
  }

  report("재산출 Qflat8", x[0],x[1],x[2]);
  std::printf("\n  ── 붙여넣을 값 ──────────────────────────────────────────────\n");
  std::printf("  C++  cpp/src/biped_control.hpp :\n");
  std::printf("    double Qflat8[8]={0,%.6f,%.6f,%.6f, 0,%.6f,%.6f,%.6f};\n",
              x[0],x[1],x[2],x[0],x[1],x[2]);
  std::printf("  Python  biped_wbic.py :\n");
  std::printf("    Q_HOME_FLAT = np.array([0.0, %.6f, %.6f, %.6f,  0.0, %.6f, %.6f, %.6f])\n",
              x[0],x[1],x[2],x[0],x[1],x[2]);
  std::printf("  ⚠양쪽을 **같이** 고칠 것. 한쪽만 고치면 파리티가 조용히 깨진다.\n");
  mj_deleteData(d); mj_deleteModel(m);
  return 0;
}
