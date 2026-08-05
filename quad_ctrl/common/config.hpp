#pragma once
// 배포 config 로더(원칙③: 파라미터=config). flat `key: value` yaml-부분집합 → 프로세스 env로 주입.
//   ★비침습: quad/cpp의 apply_env_gains·inline getenv를 그대로 두고, 시작 시 config를 env로 채운다.
//   ★precedence: 실제 env-var가 이미 있으면 유지(override 우선), 없을 때만 config 적용(setenv overwrite=0).
//     → 배포=config가 baseline·실험=env로 즉석 override. config 미지정 시 코드 기본값(회귀).
//   파서: 한 줄 `key: value`(또는 `key = value`), `#` 주석·빈줄 무시, 값 앞뒤 공백/따옴표 제거.
//   ※의존성 없음(yaml-cpp 불요) — 배포 파라미터는 전부 flat scalar라 충분.
#include <cstdlib>
#include <cstdio>
#include <string>
#include <fstream>

namespace qc {

inline std::string cfg_trim_(const std::string& s) {
  size_t a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t\r\n");
  std::string t = s.substr(a, b - a + 1);
  if (t.size() >= 2 && (t.front() == '"' || t.front() == '\'') && t.back() == t.front()) t = t.substr(1, t.size() - 2);
  return t;
}

// path의 config를 env로 로드. 성공 반환값=적용 키 수(-1=열기 실패). env 우선(overwrite=0).
inline int load_config(const char* path) {
  std::ifstream f(path);
  if (!f) { std::fprintf(stderr, "[config] 열기 실패: %s\n", path); return -1; }
  int n = 0; std::string line;
  while (std::getline(f, line)) {
    size_t h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);  // 주석 제거
    if (cfg_trim_(line).empty()) continue;
    size_t sep = line.find(':'); if (sep == std::string::npos) sep = line.find('=');
    if (sep == std::string::npos) continue;
    std::string k = cfg_trim_(line.substr(0, sep)), v = cfg_trim_(line.substr(sep + 1));
    if (k.empty()) continue;
    setenv(k.c_str(), v.c_str(), 0);   // ★overwrite=0: 이미 있는 env(실험 override)는 유지
    ++n;
  }
  std::fprintf(stderr, "[config] %s 적용 %d키(env-var 우선)\n", path, n);
  return n;
}

// QC_CONFIG env가 가리키는 config를 로드(있을 때만). 시작 시 1회 호출.
inline void load_config_env() { if (const char* p = getenv("QC_CONFIG")) load_config(p); }

}  // namespace qc
