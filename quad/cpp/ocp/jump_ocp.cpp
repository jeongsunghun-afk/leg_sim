// S1 — 점프 OCP C++ 포팅 (offline/jump/jump_ocp.py 대응). 실시간 OCP(§9)의 backbone.
//   [현재 단계] pinocchio 모델 로딩 = URDF freeflyer + 허리(FB_waist) lock → reduced 16 leg DOF.
//   Python parity 기준: nq=23 nv=22 다리DOF=16.
//   URDF 경로 = argv[1] (기본=배포 URDF 절대경로).
#include <iostream>
#include <vector>
#include <pinocchio/multibody/joint.hpp>       // JointModelFreeFlyer 등
#include <pinocchio/parsers/urdf.hpp>          // buildModel
#include <pinocchio/algorithm/model.hpp>       // buildReducedModel
#include <pinocchio/algorithm/joint-configuration.hpp>  // neutral

int main(int argc, char** argv) {
  const std::string URDF = argc > 1 ? argv[1]
      : "/home/jsh/문서/jsh/simulation/02_Leg_UFDF_260703_2/urdf/02_Leg_UFDF_260703_3.urdf";

  // freeflyer base로 전체 모델(운동학만; 메시/geometry 불요 → package 경로 불필요)
  pinocchio::Model full;
  pinocchio::urdf::buildModel(URDF, pinocchio::JointModelFreeFlyer(), full);

  // 허리(FB_waist_joint) lock → reduced 모델
  const pinocchio::JointIndex wj = full.getJointId("FB_waist_joint");
  Eigen::VectorXd q0 = pinocchio::neutral(full);
  std::vector<pinocchio::JointIndex> lock = {wj};
  pinocchio::Model model;
  pinocchio::buildReducedModel(full, lock, q0, model);

  // 발 프레임 id
  const std::vector<std::string> FEET = {
      "FL_foot_contact_link", "FR_foot_contact_link",
      "HL_foot_contact_link", "HR_foot_contact_link"};
  std::cout << "[ocp] reduced nq=" << model.nq << " nv=" << model.nv
            << " 다리DOF=" << (model.nv - 6) << " feet=[";
  for (size_t i = 0; i < FEET.size(); ++i)
    std::cout << (i ? "," : "") << model.getFrameId(FEET[i]);
  std::cout << "]" << std::endl;

  // parity 확인
  bool ok = (model.nq == 23 && model.nv == 22);
  std::cout << "[ocp] parity(nq23·nv22): " << (ok ? "OK" : "MISMATCH") << std::endl;
  return ok ? 0 : 1;
}
