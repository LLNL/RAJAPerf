#ifndef INCLUDED_RAJA_BENCHMARK_H_
#define INCLUDED_RAJA_BENCHMARK_H_

#include <chrono>
#include <string>

namespace RAJA {

class Benchmark
{
  public:

  using clock = std::chrono::steady_clock;
  using TimeType = clock::time_point;
  using Duration = std::chrono::duration<double>;

  Benchmark(size_t size, unsigned iterations);

  virtual ~Benchmark();

  double getMinTime() { return min_time; }
  double getMaxTime() { return max_time; }
  double getAvgTime() { return avg_time; }

  void execute();

  //virtual bool checkCorrectness();

  virtual void setUp() = 0;
  virtual void executeBenchmark() = 0;
  virtual void tearDown() = 0;

  protected:
  size_t size;
  size_t numIterations;

  private:
  std::string name;

  double min_time;
  double max_time;
  double avg_time;
};

} // end namespace RAJA

#endif
