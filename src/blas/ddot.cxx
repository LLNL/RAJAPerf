#include <random>
#include <iostream>

#include "benchmark/benchmark.h"

#include "RAJA/RAJA.hxx"

class DdotFixture : public ::benchmark::Fixture {
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
      }

      for (int i = 0; i < n; ++i) {
        y[i] = dist(e2);
      }
    }

    void TearDown(const ::benchmark::State& st) {
      delete[] x;
      delete[] y;
    }

    double* x;
    double* y;
};

#define DDOT dot += (x[i] * y[i])

BENCHMARK_DEFINE_F(DdotFixture, DdotRaw)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    double dot = 0.0;
    for( int i = 0; i < N; ++i) {
      DDOT;
    }
    double dot_result;
    benchmark::DoNotOptimize(dot_result = dot);
  }
}
BENCHMARK_REGISTER_F(DdotFixture, DdotRaw)->Range(128, 8<<24);

BENCHMARK_DEFINE_F(DdotFixture, DdotRaja)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::ReduceSum<RAJA::seq_reduce, double> dot(0.0);
    RAJA::forall<RAJA::seq_exec>(0, N, [=] (int i) {
      DDOT;
    });
    double dot_result;
    benchmark::DoNotOptimize(dot_result = double(dot));
  }
}
BENCHMARK_REGISTER_F(DdotFixture, DdotRaja)->Range(128, 8<<24);

#if defined(RAJA_ENABLE_OPENMP)
BENCHMARK_DEFINE_F(DdotFixture, DdotRajaOmp)(benchmark::State& state)
{
  int N = state.range(0);

  while(state.KeepRunning()) {
    RAJA::ReduceSum<RAJA::omp_reduce, double> dot(0.0);
    RAJA::forall<RAJA::omp_parallel_for_exec>(0, N, [=] (int i) {
      DDOT;
    });
    double dot_result;
    benchmark::DoNotOptimize(dot_result = double(dot));
  }
}
BENCHMARK_REGISTER_F(DdotFixture, DdotRajaOmp)->Range(128, 8<<24);
#endif

// template <typename POL>
// void BM_Test(benchmark::State& state) {
//   POL policy;
//   while (state.keepRunning()) {
//     RAJA::forall<policy>(0, state.range(0), [=] (int i) {
//     });
//   }
// }
// BENCHMARK_TEMPLATE(BM_Test, RAJA::seq_exec)->Arg(8);

BENCHMARK_MAIN();
