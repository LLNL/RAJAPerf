#include <random>
#include <iostream>

#include "benchmark/benchmark.h"

#include "RAJA/RAJA.hxx"

class MulFixture : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& st) {
      int n = st.range(0);
      x = new double[n];
      y = new double[n];
      z = new double[n];

      std::random_device rd;
      std::mt19937 e2(rd());
      std::uniform_real_distribution<> dist(0, 25);

      for (int i = 0; i < n; ++i) {
        x[i] = dist(e2);
        z[i] = dist(e2);
        y[i] = 0.0;
      }
    }

    void TearDown(const ::benchmark::State& st) {
      delete[] x;
      delete[] y;
      delete[] z;
    }

    double* x;
    double* y;
    double* z;
};

#define MUL y[i] = x[i] * z[i]

BENCHMARK_DEFINE_F(MulFixture, MulRaw)(benchmark::State& state)
{

  int N = state.range(0);

  while(state.KeepRunning()) {
    for( int i = 0; i < N; ++i) {
      MUL;
    }
  }
}
BENCHMARK_REGISTER_F(MulFixture, MulRaw)->Range(128, 8<<16);

BENCHMARK_DEFINE_F(MulFixture, MulRajaSeq)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::seq_exec>(0, N, [=] (int i) {
      MUL;
    });
  }
}
BENCHMARK_REGISTER_F(MulFixture, MulRajaSeq)->Range(128, 8<<16);

BENCHMARK_DEFINE_F(MulFixture, MulRajaSimd)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::simd_exec>(0, N, [=] (int i) {
      MUL;
    });
  }
}
BENCHMARK_REGISTER_F(MulFixture, MulRajaSimd)->Range(128, 8<<16);

#if defined(RAJA_ENABLE_OPENMP)
BENCHMARK_DEFINE_F(MulFixture, MulRajaOmp)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, N, [=] (int i) {
      MUL;
    });
  }
}

BENCHMARK_REGISTER_F(MulFixture, MulRajaOmp)->Range(128, 8<<16);
#endif

BENCHMARK_MAIN();
