// biped_parity.cpp — JointMap(Python) ↔ JointMap(C++) 패리티 시험.
//   ★왜 필요한가 (2026-08-10): Python 에 ±180 포화를 넣고 C++ 미러를 빠뜨렸는데,
//     당시 패리티 확인이 **기준자세 하나**만 봐서 통과했다. 실제로는 관절한계
//     꼭짓점에서 60° 어긋나 있었다. 한 점 검증은 패리티 검증이 아니다.
//   ⇒ 관절한계 박스의 **모든 꼭짓점 + 무작위 내부점**을 훑어 stdout 으로 뱉고,
//     tools/parity_check.py 가 Python 결과와 비교한다.
#include "deploy_hw.hpp"
#include <cstdio>
#include <random>
using namespace bipedhw;
int main(int argc, char** argv){
  if(argc < 2){ std::fprintf(stderr, "usage: biped_parity <config.yaml> [n_random]\n"); return 2; }
  EmbCfg c; std::string err;
  if(!load_cfg(argv[1], c, err)){ std::fprintf(stderr, "load fail: %s\n", err.c_str()); return 1; }
  JointMap jm(c);
  const int N = jm.n_leg;
  const int nrand = (argc > 2) ? std::atoi(argv[2]) : 300;
  std::vector<double> q(N); std::vector<float> ch(jm.n_channel);
  // ★위치만 보면 안 된다 — 2026-08-10 에 q_joint_to_ch 만 검사하다가
  //   q_ctrl_to_ch 의 포화 누락(최대 54.9° 차)을 놓쳤다. **전 경로**를 뱉는다.
  std::vector<float> dch(jm.n_channel), tch(jm.n_channel);
  std::vector<double> qb(N), dqb(N);
  auto emit = [&](const char* tag){
    std::printf("%s", tag);
    for(int i=0;i<N;i++) std::printf(" %.6f", q[i]);
    std::printf(" |");
    jm.q_joint_to_ch(q.data(), ch.data());                    // ① 모델각 → 채널각
    for(int i=0;i<N;i++) std::printf(" %.6f", ch[c.joints[i].channel]);
    std::printf(" |");
    std::vector<double> qr(N);                                 // ② rad 경로
    for(int i=0;i<N;i++) qr[i] = q[i]*JointMap::D2R;
    jm.q_ctrl_to_ch(qr.data(), dch.data());
    for(int i=0;i<N;i++) std::printf(" %.6f", dch[c.joints[i].channel]);
    std::printf(" |");
    jm.dq_ctrl_to_ch(qr.data(), dch.data());                   // ③ 속도(같은 벡터 재사용)
    for(int i=0;i<N;i++) std::printf(" %.6f", dch[c.joints[i].channel]);
    std::printf(" |");
    jm.tau_ctrl_to_ch(q.data(), tch.data());                   // ④ 토크(q 를 Nm 로 재사용)
    for(int i=0;i<N;i++) std::printf(" %.6f", tch[c.joints[i].channel]);
    std::printf(" |");
    jm.ch_to_q_joint(ch.data(), qb.data());                    // ⑤ 역변환(왕복)
    jm.q_joint_to_ch(q.data(), ch.data());
    jm.ch_to_q_joint(ch.data(), qb.data());
    for(int i=0;i<N;i++) std::printf(" %.6f", qb[i]);
    std::printf("\n");
  };
  // ① 한계 박스 꼭짓점 전부 (2^8 = 256)
  for(int m=0; m < (1<<N); m++){
    for(int i=0;i<N;i++) q[i] = (m>>i & 1) ? c.joints[i].max_deg : c.joints[i].min_deg;
    emit("V");
  }
  // ② 박스 내부 무작위 (재현 위해 고정 시드)
  std::mt19937 rng(12345);
  for(int k=0;k<nrand;k++){
    for(int i=0;i<N;i++){
      std::uniform_real_distribution<double> u(c.joints[i].min_deg, c.joints[i].max_deg);
      q[i] = u(rng);
    }
    emit("R");
  }
  std::fprintf(stderr, "saturated=%ld last_ch=%d\n", jm.n_saturated, jm.last_saturated_ch);
  return 0;
}
