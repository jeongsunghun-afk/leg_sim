// biped C++ 단독 sim — MJCF 로드 → BipedControl 제어루프 → mj_step. 헤드리스 폐루프 검증.
// 실행: ./biped_sim [mjcf] [vx] [T]   (기본 ../biped_from_quad.mjcf 0.15 15)
// ★EST_CTRL=1 : 추정 상태(leg-odom+접촉높이)로 폐루프 제어(물리는 GT). 배포 경로 검증. falls 카운트.
#include <mujoco/mujoco.h>
#include "biped_control.hpp"
#include "deploy_loop.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <vector>

static double tilt_deg(const double* q){
  double roll=std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
  double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
  return std::hypot(roll,pitch)*180/M_PI;
}

int main(int argc,char**argv){
  const char* mjcf = argc>1?argv[1]:"../biped_from_quad.mjcf";
  double vx = argc>2?atof(argv[2]):0.15;
  double T  = argc>3?atof(argv[3]):15.0;
  char err[1000]={0};
  mjModel* m=mj_loadXML(mjcf,nullptr,err,1000);
  if(!m){ printf("모델 로드 실패: %s\n",err); return 1; }
  // ★2026-08-20 **실기 탑재물 반영**. 저울 실측으로 모델 13.898 kg 은 맞다고 확인됐으나,
  //   MCU·PCB 를 얹은 실기는 약 15 kg 이다(+1.1 kg = **8%**). 중력보상이 8% 모자라면
  //   그것만으로 주저앉는다 ⇒ 실기와 토크를 대조할 땐 **이 차이를 먼저 없애고** 봐야
  //   한다. 안 그러면 전 축에 8% 부족이 깔려 어느 축이 진짜 문제인지 안 보인다.
  //   ⚠질량만 더한다(관성·CoM 은 그대로). 정적 stand 토크 비교가 목적이라 충분하다 —
  //     동적 거동까지 볼 거면 실측 CoM 으로 inertial 을 고쳐야 한다.
  if(const char* e=getenv("TORSO_ADD_KG")){
    int tb=mj_name2id(m,mjOBJ_BODY,"torso");
    if(tb>=0){
      m->body_mass[tb]+=atof(e);
      double tot=0; for(int i=0;i<m->nbody;i++) tot+=m->body_mass[i];
      std::printf("[sim] torso +%.3f kg → 총질량 %.4f kg (%.1f N)\n", atof(e), tot, tot*9.81);
    } else std::printf("[sim] ⚠torso 바디를 못 찾음 — TORSO_ADD_KG 무시\n");
  }
  mjData* d=mj_makeData(m);
  BipedControl c(m,d); c.reset();

  // ★TAU_DBG=<초> — 마지막 N초의 **축별 관절토크**와 **좌우 지면반력**을 낸다.
  //   실기 상태 JSON 의 `tau_leg_nm`(측정)·`tau_cmd_nm`(명령)과 **같은 좌표**다
  //   (deploy 는 ch_to_tau_joint 로 관절토크를 발행한다) ⇒ 변환 없이 그대로 대조된다.
  //   ★`qfrc_actuator` 를 쓰는 이유: 발목이 tendon 에 물려 있어 `ctrl[foot]` 은 모터축
  //     지령이고 calf·foot **두 DOF 에 같이** 걸린다. 축별 실제 관절토크는 일반화력이다.
  const double tau_win = getenv("TAU_DBG") ? atof(getenv("TAU_DBG")) : 0.0;
  std::vector<double> tsum(m->nu,0.0), tsq(m->nu,0.0);
  double grf[2]={0,0}; long nacc=0;
  if(getenv("CONTACT")) c.set_contact_mode(atoi(getenv("CONTACT")));   // ★0=1점 점발보행·1=2점 평발정적
  if(getenv("STAND_CZ")) c.com_ref_z=atof(getenv("STAND_CZ"));         // 정적 높이 테스트
  c.vx_cmd=vx;
  c.vy_cmd = getenv("VY")?atof(getenv("VY")):0.0;        // 측방/선회 테스트용 env
  c.wz_cmd = getenv("WZ")?atof(getenv("WZ")):0.0;
  double dt=m->opt.timestep; int steps=(int)(T/dt); double fell=-1;

  bool est_ctrl = getenv("EST_CTRL")!=nullptr;
  DeployLoop dl; int falls=0;
  if(est_ctrl){ dl.init(m,c); dl.reset(m,d); }

  bool do_switch=getenv("SWITCH")!=nullptr;    // ★중간 접촉모드 전환 검증(T/2에 토글)
  for(int i=0;i<steps;i++){
    if(do_switch && i==steps/2){ int nm=c.cmode==1?0:1;
      if(getenv("TRANS")){ c.transition_to(nm); std::printf("  [T/2] 굴림 전환 시작 → 목표 cmode=%d\n",nm); }
      else { c.set_contact_mode(nm); c.vx_cmd=(c.cmode==0?vx:0.0); if(est_ctrl) dl.reset(m,d);
             std::printf("  [T/2] 스냅 전환 → cmode=%d\n",c.cmode); } }
    if(est_ctrl) dl.step(m,d,c,dt);      // 추정+지연+보상 → d->ctrl (물리 d 불변)
    else c.control(dt);
    if(getenv("WALK_DBG") && i%25==0){ double* q=&d->qpos[3];
      double pitch=std::asin(std::max(-1.0,std::min(1.0,2*(q[0]*q[2]-q[3]*q[1]))));
      double roll=std::atan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]*q[1]+q[2]*q[2]));
      std::printf("  t%.2f com=(%+.3f,%+.3f,%.3f) v=(%+.2f,%+.2f) pitch%+.1f roll%+.1f sw=%d\n",
        i*dt,d->subtree_com[0],d->subtree_com[1],d->subtree_com[2],d->qvel[0],d->qvel[1],pitch*57.3,roll*57.3,c.swing); }
    mj_step(m,d);
    if(tau_win>0 && (T - i*dt) <= tau_win){                  // 정상상태 창만 집계
      for(int j=0;j<m->nu;j++){ double t=d->qfrc_actuator[6+j]; tsum[j]+=t; tsq[j]+=t*t; }
      for(int ci=0;ci<d->ncon;ci++){
        mjtNum f[6]; mj_contactForce(m,d,ci,f);              // f[0]=접촉 법선력
        int g1=d->contact[ci].geom1, g2=d->contact[ci].geom2;
        for(int l=0;l<2;l++)
          if(g1==c.sph[l]||g2==c.sph[l]||(c.has_heel&&(g1==c.sph2[l]||g2==c.sph2[l])))
            grf[l]+=f[0];
      }
      nacc++;
    }
    if(est_ctrl){                                           // 낙상 자동리셋 + 카운트(장시간 통계)
      if(d->qpos[2]<0.2 || tilt_deg(&d->qpos[3])>45){
        c.reset(); c.vx_cmd=vx; dl.reset(m,d); falls++;
      }
    } else if(d->qpos[2]<0.15 || tilt_deg(&d->qpos[3])>45){ fell=i*dt; break; }
    // ★tilt 판정 추가(2026-08-05). 기존엔 base 높이만 봐서 **기울어진 채 버티는 것을
    //   성공으로 셌다** — T_STEP=0.24/vx=0.30 이 tilt 81.9° 인데 "무낙상" 으로 집계됐다
    //   (0.28@0.25=50.4°, 0.28@0.30=46.2° 도 동일). 위 EST_CTRL 분기는 이미
    //   `qpos[2]<0.2 || tilt>45` 로 옳게 판정하고 있었으므로 임계를 그쪽에 맞췄다.
    //   ⚠ 45° 도 관대하다. 보행 품질 분석은 tilt 10° 이하만 진짜 성공으로 볼 것.
  }

  if(est_ctrl){
    printf("EST_CTRL vx=%.2f T=%.1fs · falls=%d · 추정 base=(%.2f,%.2f,%.3f) GT=(%.2f,%.2f,%.3f) tilt=%.1f°\n",
           vx, T, falls, dl.est.p[0],dl.est.p[1],dl.est.p[2], d->qpos[0],d->qpos[1],d->qpos[2], tilt_deg(&d->qpos[3]));
  } else {
    printf("vx=%.2f · 생존 %.2fs%s · base=(%.3f,%.3f,%.3f) tilt=%.1f°\n",
           vx, fell<0?T:fell, fell<0?"(무낙상)":"(낙상)",
           d->qpos[0],d->qpos[1],d->qpos[2], tilt_deg(&d->qpos[3]));
  }
  if(tau_win>0 && nacc>0){
    static const char* JN[8]={"HL_hip","HL_thigh","HL_calf","HL_foot",
                              "HR_hip","HR_thigh","HR_calf","HR_foot"};
    std::printf("\n== 정상상태 축별 관절토크 (마지막 %.0fs · %ld 샘플) ==\n", tau_win, nacc);
    std::printf("   실기 상태 JSON 의 tau_leg_nm(측정)·tau_cmd_nm(명령)과 **같은 좌표**다.\n");
    // ★좌우차는 **크기**로 뺀다. hip 은 좌우 부호가 거울이라(−2.59 vs +2.60) 그대로
    //   빼면 5.2 라는 유령이 찍힌다 — 실제 비대칭은 0.013 이다.
    std::printf("  %-10s %12s %10s   %10s\n", "축", "관절토크[Nm]", "표준편차", "|HR|−|HL|");
    for(int j=0;j<m->nu && j<8;j++){
      double mu=tsum[j]/nacc, sd=std::sqrt(std::max(0.0, tsq[j]/nacc-mu*mu));
      char dif[24]="";
      if(j>=4){ double ml=tsum[j-4]/nacc;
        std::snprintf(dif,sizeof dif,"%+.3f", std::fabs(mu)-std::fabs(ml)); }
      std::printf("  %-10s %+12.3f %10.3f   %10s\n", JN[j], mu, sd, dif);
    }
    double gl=grf[0]/nacc, gr=grf[1]/nacc, gt=gl+gr;
    double W=0; for(int i=0;i<m->nbody;i++) W+=m->body_mass[i]; W*=9.81;
    if(gt>1e-6)
      std::printf("  지면반력 Fz   HL %6.1f N (%4.1f%%) · HR %6.1f N (%4.1f%%)"
                  "   합 %6.1f N / 체중 %6.1f N\n", gl, 100*gl/gt, gr, 100*gr/gt, gt, W);
    else
      std::printf("  ⚠지면반력 0 — 접촉이 없다(공중이거나 낙상). 토크값도 의미 없다.\n");
  }
  mj_deleteData(d); mj_deleteModel(m); return 0;
}
