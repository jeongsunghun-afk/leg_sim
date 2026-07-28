#include "constraints.hpp"
#include <cstdio>
using namespace tamols;
int main(){
  TamolsState st;
  if(!load_solution("/tmp/tamols_sol.txt", st)){ std::printf("★ 해 없음\n"); return 1; }
  st.prm.l_min=0.12; st.prm.l_max=0.80; st.prm.mu=0.6; st.prm.mass=37.9;
  std::printf("[TAMOLS 제약 검증] P=%d  I=[%.4f %.4f %.4f]\n", st.num_phases(),
              st.prm.inertia_diag(0),st.prm.inertia_diag(1),st.prm.inertia_diag(2));
  double ei=initial_residual(st), ej=junction_residual(st), ef=friction_residual(st), ek=kinematic_residual(st), eg=giac_residual(st);
  std::printf("  초기          = %.2e  %s\n", ei, ei<1e-5?"OK":"★");
  std::printf("  위상연속       = %.2e  %s\n", ej, ej<1e-5?"OK":"★");
  std::printf("  friction       = %.2e  %s\n", ef, ef<1e-4?"OK":"★");
  std::printf("  kinematic      = %.2e  %s\n", ek, ek<1e-3?"OK":"★");
  std::printf("  ★GIAC(Eq17)   = %.2e  %s\n", eg, eg<1e-3?"OK":"★");
  bool ok=ei<1e-5&&ej<1e-5&&ef<1e-4&&ek<1e-3&&eg<1e-3;
  std::printf("  → 제약 5종 %s (Drake 정식화 완전 정합)\n", ok?"✓":"확인필요");
  return ok?0:1;
}
