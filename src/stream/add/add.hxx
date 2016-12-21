#include "common/Benchmark.hxx"

#include <random>

class AddBenchmark : public RAJA::Benchmark
{
public:
  AddBenchmark(size_t size, size_t numIterations);
  virtual ~AddBenchmark();

  virtual void setUp();
  virtual void executeBenchmark();
  virtual void tearDown();

protected:
  double* x;
  double* y;
};
