#pragma once
// ================================================================================================
// D1 배포: MuJoCo 백엔드 (MuJoCo-as-model). 데스크톱 검증용 — 실기 real_hal이 이 자리에 들어간다.
//   D1MujocoHal : 상태읽기(qc::State) + 토크적용(관절12=WBC·발목4/허리1=0-hold PD) + 1스텝.
//   MujocoTerrain: TerrainProvider — sdf.update(m,d,cx,cy)(mj_ray group2 heightmap).
//   ★플랜트 물리(1kHz·solref·GEARBOX)는 test02legMujoco와 동일 → sim 재현으로 HAL 경계 검증.
//   관절순=OCS2 [FL,FR,HL,HR]×[hip,thigh,calf]. contact순=[FL,FR,HL,HR].
// ================================================================================================
#include <mujoco/mujoco.h>
#include <vector>
#include <array>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "estimator/state.hpp"       // qc::State
#include "terrain_provider.hpp"      // d1::TerrainProvider, MjTerrainSdf

namespace d1 {

// ── 지형: MuJoCo mj_ray로 SDF 그리드 채움 ──
class MujocoTerrain : public TerrainProvider {
  const mjModel* m_; mjData* d_;
 public:
  MujocoTerrain(const mjModel* m, mjData* d) : m_(m), d_(d) {}
  void fill(MjTerrainSdf& sdf, double cx, double cy) override { sdf.update(m_, d_, cx, cy); }
};

// ── HAL: MuJoCo 백엔드 ──
class D1MujocoHal {
 public:
  bool load(const std::string& mjcf) {
    char err[1024] = "";
    m_ = mj_loadXML(mjcf.c_str(), nullptr, err, sizeof(err));
    if (!m_) { std::fprintf(stderr, "[D1MujocoHal] mjcf 로드 실패: %s\n", err); return false; }
    d_ = mj_makeData(m_);
    return true;
  }

  // 플랜트 물리 (test02legMujoco 391-409): 1kHz·접촉강성·GEARBOX(이름기반). 컨트롤러 WBC와 정합.
  void setupPhysics() {
    m_->opt.timestep = std::getenv("TIMESTEP") ? atof(std::getenv("TIMESTEP")) : 0.001;
    double stiff = std::getenv("STIFF") ? atof(std::getenv("STIFF")) : 0.005;
    for (int g = 0; g < m_->ngeom; ++g) { m_->geom_solref[g * 2] = stiff; m_->geom_solref[g * 2 + 1] = 1.0; }
    const char* GN[4] = {"hip", "thigh", "calf", "foot"}; double gear[4] = {7.0, 7.0, 10.5, 8.4};
    bool gbx = !(std::getenv("GEARBOX") && !std::strcmp(std::getenv("GEARBOX"), "0"));
    double Irot = std::getenv("ROTOR_I") ? atof(std::getenv("ROTOR_I")) : 7.4e-4;
    double jdmp = std::getenv("JDAMP") ? atof(std::getenv("JDAMP")) : 0.099;
    double jfrc = std::getenv("JFRIC") ? atof(std::getenv("JFRIC")) : 0.38;
    if (gbx) for (int k = 0; k < m_->nu; ++k) { int jid = m_->actuator_trnid[k * 2]; if (jid < 0) continue;
      const char* jn = mj_id2name(m_, mjOBJ_JOINT, jid); if (!jn) continue;
      int gi = 0; for (int g = 0; g < 4; ++g) if (std::strstr(jn, GN[g])) gi = g;
      double N = gear[gi]; int dof = m_->jnt_dofadr[jid];
      m_->dof_armature[dof] = Irot * N * N; m_->dof_damping[dof] = jdmp; m_->dof_frictionloss[dof] = jfrc; }
  }

  // OCS2 관절순 매핑 + 발목/허리 홀드 + 발 geom.
  void buildMapping(const std::vector<std::string>& jNames) {
    nJ_ = (int)jNames.size();
    qadr_.resize(nJ_); vadr_.resize(nJ_); act_.resize(nJ_);
    auto actName = [](const std::string& jn) { return jn.substr(0, jn.size() - 6); };  // strip "_joint"
    for (int i = 0; i < nJ_; ++i) {
      int j = mj_name2id(m_, mjOBJ_JOINT, jNames[i].c_str());
      qadr_[i] = m_->jnt_qposadr[j]; vadr_[i] = m_->jnt_dofadr[j];
      act_[i] = mj_name2id(m_, mjOBJ_ACTUATOR, actName(jNames[i]).c_str());
    }
    // 홀드 발목(제어관절 아님) — 4발목 중 미제어분
    const char* kAnkle[4] = {"FL_foot_joint", "FR_foot_joint", "HL_foot_joint", "HR_foot_joint"};
    holdQ_.clear(); holdV_.clear(); holdA_.clear();
    for (int i = 0; i < 4; ++i) {
      bool controlled = false; for (int n = 0; n < nJ_; ++n) if (jNames[n] == kAnkle[i]) controlled = true;
      if (!controlled) { int j = mj_name2id(m_, mjOBJ_JOINT, kAnkle[i]);
        if (j >= 0) { holdQ_.push_back(m_->jnt_qposadr[j]); holdV_.push_back(m_->jnt_dofadr[j]);
                      holdA_.push_back(mj_name2id(m_, mjOBJ_ACTUATOR, actName(kAnkle[i]).c_str())); } }
    }
    const char* sph[4] = {"FL_sphere", "FR_sphere", "HL_sphere", "HR_sphere"};
    for (int i = 0; i < 4; ++i) footGeom_[i] = mj_name2id(m_, mjOBJ_GEOM, sph[i]);
    int wj = mj_name2id(m_, mjOBJ_JOINT, "FB_waist_joint");
    if (wj >= 0) { wq_ = m_->jnt_qposadr[wj]; wv_ = m_->jnt_dofadr[wj]; wact_ = mj_name2id(m_, mjOBJ_ACTUATOR, "FB_waist"); }
  }

  // 초기 포즈: base z + quat 단위 + nominal 관절. 리셋 스냅샷 저장.
  void setInitialPose(double baseZ, const std::vector<double>& jNom) {
    d_->qpos[2] = baseZ; d_->qpos[3] = 1; d_->qpos[4] = d_->qpos[5] = d_->qpos[6] = 0;
    for (int i = 0; i < nJ_ && i < (int)jNom.size(); ++i) d_->qpos[qadr_[i]] = jNom[i];
    mj_forward(m_, d_);
    qpos0_.assign(d_->qpos, d_->qpos + m_->nq);
  }
  void reset() {
    for (int q = 0; q < m_->nq; ++q) d_->qpos[q] = qpos0_[q];
    for (int v = 0; v < m_->nv; ++v) d_->qvel[v] = 0.0;
    mj_forward(m_, d_);
  }

  // 발-base nominal xy 오프셋(OCS2순) — geom_xpos 기반, 셋업 후 1회.
  void footOffsets(double off[4][2]) const {
    for (int i = 0; i < 4; ++i) { off[i][0] = d_->geom_xpos[3 * footGeom_[i] + 0] - d_->qpos[0];
                                  off[i][1] = d_->geom_xpos[3 * footGeom_[i] + 1] - d_->qpos[1]; }
  }
  // 현재 발 xy(리셋용).
  void footPositions(double pos[4][2]) const {
    for (int i = 0; i < 4; ++i) { pos[i][0] = d_->geom_xpos[3 * footGeom_[i] + 0];
                                  pos[i][1] = d_->geom_xpos[3 * footGeom_[i] + 1]; }
  }

  // ── 상태 읽기 → qc::State (sim GT). base_ang_vel=body(qvel[3:6]), base_lin_vel=world(qvel[0:3]). ──
  void readState(qc::State& s) {
    s.time = d_->time;
    s.base_pos << d_->qpos[0], d_->qpos[1], d_->qpos[2];
    s.base_quat << d_->qpos[3], d_->qpos[4], d_->qpos[5], d_->qpos[6];   // wxyz
    s.base_lin_vel << d_->qvel[0], d_->qvel[1], d_->qvel[2];             // world
    s.base_ang_vel << d_->qvel[3], d_->qvel[4], d_->qvel[5];             // body
    if (s.q.size() != nJ_) { s.q.resize(nJ_); s.dq.resize(nJ_); }
    for (int i = 0; i < nJ_; ++i) { s.q[i] = d_->qpos[qadr_[i]]; s.dq[i] = d_->qvel[vadr_[i]]; }
    // 접촉(OCS2순 [FL,FR,HL,HR]) — sphere geom 접촉
    for (int i = 0; i < 4; ++i) { bool con = false;
      for (int ci = 0; ci < d_->ncon; ++ci) { const auto& c = d_->contact[ci];
        if (c.geom1 == footGeom_[i] || c.geom2 == footGeom_[i]) { con = true; break; } }
      s.contact[i] = con ? 1.0 : 0.0; }
  }

  // ── 토크 적용: 제어관절(WBC τ) + 발목/허리 0-hold PD. 물리 진행은 step()에서. ──
  void applyTorque(const Eigen::VectorXd& tauJ) {
    for (int i = 0; i < nJ_; ++i) d_->ctrl[act_[i]] = tauJ(i);
    double KpA = std::getenv("ANKLE_KP") ? atof(std::getenv("ANKLE_KP")) : 40.0;
    double KdA = std::getenv("ANKLE_KD") ? atof(std::getenv("ANKLE_KD")) : 1.5;
    for (size_t i = 0; i < holdA_.size(); ++i)
      d_->ctrl[holdA_[i]] = KpA * (0.0 - d_->qpos[holdQ_[i]]) + KdA * (0.0 - d_->qvel[holdV_[i]]);
    if (wact_ >= 0) { double KpW = 300, KdW = 12;
      d_->ctrl[wact_] = KpW * (0.0 - d_->qpos[wq_]) + KdW * (0.0 - d_->qvel[wv_]); }
  }
  void step() { mj_step(m_, d_); }

  mjModel* model() { return m_; }
  mjData*  data()  { return d_; }
  int nJ() const { return nJ_; }
  ~D1MujocoHal() { if (d_) mj_deleteData(d_); if (m_) mj_deleteModel(m_); }

 private:
  mjModel* m_ = nullptr; mjData* d_ = nullptr;
  int nJ_ = 12;
  std::vector<int> qadr_, vadr_, act_, holdQ_, holdV_, holdA_;
  int footGeom_[4] = {-1, -1, -1, -1};
  int wq_ = -1, wv_ = -1, wact_ = -1;
  std::vector<double> qpos0_;
};

}  // namespace d1
