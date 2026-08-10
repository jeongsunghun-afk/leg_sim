# ethz-adrl/towr (진짜 TOWR) 빌드 노트 — 참조용

우리 `simulation/quad/towr/towr_cd.py/cpp`는 **단순화판**(고정 접촉 스케줄, phase-based 타이밍 최적화 없음).
이 `towr_ext/`가 **원조 TOWR**(Winkler 2018, phase-based로 게이트 시퀀스·스텝 타이밍을 연속변수 자동 최적화).

## 위치
- `ifopt/`  — NLP 인터페이스(ethz-adrl/ifopt)
- `towr/`   — TOWR 라이브러리(ethz-adrl/towr). `towr/towr/`=core, `towr_ros/`=ROS(미빌드)
- `install/` — ifopt 설치 프리픽스
- 빌드 산출물: `towr/towr/build/libtowr.so` · `towr/towr/build/towr-example`(데모)

## 빌드 재현 (proxddp env의 IPOPT/Eigen 사용)
```bash
ENV=/home/jsh/miniforge3/envs/proxddp
INSTALL=/home/jsh/문서/jsh/towr_ext/install
# 1) ifopt (테스트 링크실패 무시 — 라이브러리만)
cd ifopt && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL -DCMAKE_PREFIX_PATH=$ENV -DCMAKE_BUILD_TYPE=Release
make ifopt_core ifopt_ipopt -j4 && cmake --install .
#   ifopt_ipopt cmake export가 누락되면 수동복사:
cp build/ifopt_ipopt/CMakeFiles/Export/share/ifopt/cmake/ifopt_ipopt-targets*.cmake $INSTALL/share/ifopt/cmake/
cp ifopt_ipopt/include/ifopt/*.h $INSTALL/include/ifopt/
# 2) towr
cd towr/towr && mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$INSTALL;$ENV" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-include cassert" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib:$INSTALL/lib $ENV/lib/libstdc++.so"
make towr-example -j4
# 실행:
LD_LIBRARY_PATH=$ENV/lib:$INSTALL/lib ./towr-example
```

## 빌드서 만난 함정 (해결됨)
1. **ifopt 테스트 링크실패** — gtest 없음. 라이브러리 타겟만 빌드(`make ifopt_core ifopt_ipopt`).
2. **ifopt_ipopt cmake export 미설치** — Export 디렉토리서 수동복사.
3. **`assert` 미선언**(신 GCC) — `-include cassert` 전역 강제포함.
4. **C++ ABI 불일치**(`libspral.so: __cxa_call_terminate@CXXABI_1.3.15`) — proxddp env IPOPT가 신 libstdc++ 요구.
   `$ENV/lib/libstdc++.so`를 명시 링크(기존 quad/cpp와 동일 방식).

## 검증
- `towr-example`(monoped 지형 hopping) solve **0.21s** — base x:0→1.00·발 스텝·접촉/게이트 자동 창발.
- = phase-based 타이밍 최적화 작동(우리 재구현이 못하는 것).

## 다음
- 02_Leg **quadruped 모델**(SRBD 질량/관성·발위치·ROM) 설정 → 우리 로봇 궤적 생성.
- receding-horizon화(논문 2104.09078 방식) → 온라인.
