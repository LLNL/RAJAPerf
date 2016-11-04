#include <random>
#include <iostream>

#include "benchmark/benchmark.h"

#include "RAJA/RAJA.hxx"

class CopyFixture : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& st) {
      int n = st.range(0);
      x = new double[n];
      y = new double[n];

      std::random_device rd;
      std::mt19937 e2(rd());
      std::uniform_real_distribution<> dist(0, 25);

      for (int i = 0; i < n; ++i) {
        x[i] = dist(e2);
        y[i] = 0.0;
      }
    }

    void TearDown(const ::benchmark::State& st) {
      delete[] x;
      delete[] y;
    }

    double* x;
    double* y;
};

#define COPY y[i] = x[i]

BENCHMARK_DEFINE_F(CopyFixture, CopyRaw)(benchmark::State& state)
{

  int N = state.range(0);

  while(state.KeepRunning()) {
    for( int i = 0; i < N; ++i) {
      COPY;
    }
  }
}
BENCHMARK_REGISTER_F(CopyFixture, CopyRaw)->Range(128, 8<<16);

BENCHMARK_DEFINE_F(CopyFixture, CopyRajaSeq)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::seq_exec>(0, N, [=] (int i) {
      COPY;
    });
  }
}
BENCHMARK_REGISTER_F(CopyFixture, CopyRajaSeq)->Range(128, 8<<16);

BENCHMARK_DEFINE_F(CopyFixture, CopyRajaSimd)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::simd_exec>(0, N, [=] (int i) {
      COPY;
    });
  }
}
BENCHMARK_REGISTER_F(CopyFixture, CopyRajaSimd)->Range(128, 8<<16);

#if defined(RAJA_ENABLE_OPENMP)
BENCHMARK_DEFINE_F(CopyFixture, CopyRajaOmp)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, N, [=] (int i) {
      COPY;
    });
  }
}

BENCHMARK_REGISTER_F(CopyFixture, CopyRajaOmp)->Range(128, 8<<16);
#endif

BENCHMARK_MAIN();
