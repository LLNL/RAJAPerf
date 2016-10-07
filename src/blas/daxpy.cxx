#include <random>
#include <iostream>

#include "benchmark/benchmark.h"

#include "RAJA/RAJA.hxx"

class DaxpyFixture : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& st) {
      int n = st.range(0);
      double a = 2.0;
      x = new double[n];
      y = new double[n];

      std::random_device rd;

      std::mt19937 e2(rd());
      std::uniform_real_distribution<> dist(0, 25);

      for (int i = 0; i < n; ++i) {
        x[i] = dist(e2);
      }

      for (int i = 0; i < n; ++i) {
        y[i] = dist(e2);
      }
    }

    void TearDown(const ::benchmark::State& st) {
      delete[] x;
      delete[] y;
    }

    int a;
    double* x;
    double* y;
};

#define DAXPY y[i] = y[i] + a*x[i]

BENCHMARK_DEFINE_F(DaxpyFixture, DaxpyRaw)(benchmark::State& state)
{

  int N = state.range(0);

  while(state.KeepRunning()) {
    for( int i = 0; i < N; ++i) {
      DAXPY;
    }
  }
}
BENCHMARK_REGISTER_F(DaxpyFixture, DaxpyRaw)->Range(128, 8<<24);

BENCHMARK_DEFINE_F(DaxpyFixture, DaxpyRajaSeq)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::seq_exec>(0, N, [=] (int i) {
      DAXPY;
    });
  }
}
BENCHMARK_REGISTER_F(DaxpyFixture, DaxpyRajaSeq)->Range(128, 8<<24);

#if defined(RAJA_ENABLE_OPENMP)
BENCHMARK_DEFINE_F(DaxpyFixture, DaxpyRajaOmp)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, N, [=] (int i) {
      DAXPY;
    });
  }
}

BENCHMARK_REGISTER_F(DaxpyFixture, DaxpyRajaOmp)->Range(128, 8<<24);
#endif

BENCHMARK_MAIN();
