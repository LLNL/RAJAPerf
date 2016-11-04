#include <random>
#include <iostream>

#include "benchmark/benchmark.h"

#include "RAJA/RAJA.hxx"

class TriadFixture : public ::benchmark::Fixture {
  public:
    void SetUp(const ::benchmark::State& st) {
      int n = st.range(0);
      x = new double[n];
      y = new double[n];
      z = new double[n];

      a = 3.14;

      std::random_device rd;
      std::mt19937 e2(rd());
      std::uniform_real_distribution<> dist(0, 25);

      for (int i = 0; i < n; ++i) {
        x[i] = 0.0;
        y[i] = dist(e2);
        z[i] = dist(e2);
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

    double a;
};

#define TRIAD x[i] = y[i] + a*z[i]

BENCHMARK_DEFINE_F(TriadFixture, TriadRaw)(benchmark::State& state)
{

  int N = state.range(0);

  while(state.KeepRunning()) {
    for( int i = 0; i < N; ++i) {
      TRIAD;
    }
  }
}
BENCHMARK_REGISTER_F(TriadFixture, TriadRaw)->Range(128, 8<<16);

BENCHMARK_DEFINE_F(TriadFixture, TriadRajaSeq)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::seq_exec>(0, N, [=] (int i) {
      TRIAD;
    });
  }
}
BENCHMARK_REGISTER_F(TriadFixture, TriadRajaSeq)->Range(128, 8<<16);

BENCHMARK_DEFINE_F(TriadFixture, TriadRajaSimd)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::simd_exec>(0, N, [=] (int i) {
      TRIAD;
    });
  }
}
BENCHMARK_REGISTER_F(TriadFixture, TriadRajaSimd)->Range(128, 8<<16);

#if defined(RAJA_ENABLE_OPENMP)
BENCHMARK_DEFINE_F(TriadFixture, TriadRajaOmp)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, N, [=] (int i) {
      TRIAD;
    });
  }
}

BENCHMARK_REGISTER_F(TriadFixture, TriadRajaOmp)->Range(128, 8<<16);
#endif

BENCHMARK_MAIN();
