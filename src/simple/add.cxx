#include <random>
#include <iostream>

#include "benchmark/benchmark.h"

#include "RAJA/RAJA.hxx"

class AddFixture : public ::benchmark::Fixture {
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
        y[i] = dist(e2);
        z[i] = 0.0;
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

#define ADD y[i] = x[i] + y[i]

BENCHMARK_DEFINE_F(AddFixture, AddRaw)(benchmark::State& state)
{

  int N = state.range(0);

  while(state.KeepRunning()) {
    for( int i = 0; i < N; ++i) {
      ADD;
    }
  }
}
BENCHMARK_REGISTER_F(AddFixture, AddRaw)->Range(128, 8<<16);

BENCHMARK_DEFINE_F(AddFixture, AddRajaSeq)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::seq_exec>(0, N, [=] (int i) {
      ADD;
    });
  }
}
BENCHMARK_REGISTER_F(AddFixture, AddRajaSeq)->Range(128, 8<<16);

BENCHMARK_DEFINE_F(AddFixture, AddRajaSimd)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::simd_exec>(0, N, [=] (int i) {
      ADD;
    });
  }
}
BENCHMARK_REGISTER_F(AddFixture, AddRajaSimd)->Range(128, 8<<16);

#if defined(RAJA_ENABLE_OPENMP)
BENCHMARK_DEFINE_F(AddFixture, AddRajaOmp)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, N, [=] (int i) {
      ADD;
    });
  }
}

BENCHMARK_REGISTER_F(AddFixture, AddRajaOmp)->Range(128, 8<<16);
#endif

BENCHMARK_MAIN();
