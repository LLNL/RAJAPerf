#include "Benchmark.hxx"

#include <iostream>

namespace RAJA {

Benchmark::Benchmark(size_t size, unsigned iterations):
  size(size), numIterations(iterations)
{
}

Benchmark::~Benchmark()
{
}

void Benchmark::execute() {
  for (size_t i = 0; i < numIterations; ++i) {
    this->tearDown();
    this->setUp();

    // start timer
    auto start = clock::now();

    this->executeBenchmark();

    // stop timer
    auto end = clock::now();
    Duration time = end - start;
    
    min_time = std::min(min_time, time.count());
    max_time = std::max(min_time, time.count());
    avg_time += time.count();
  }

  avg_time /= (double) numIterations;

  std::cout << "Avg time: " << avg_time << "s" << std::endl;
}

}
