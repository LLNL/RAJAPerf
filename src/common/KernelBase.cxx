#include "KernelBase.hxx"

#include <iostream>

namespace rajaperf {

KernelBase::KernelBase(size_t size, unsigned iterations):
  size(size), numIterations(iterations)
{
  for (size_t ivar = 0; ivar < NUM_VARIANTS; ++ivar) {
     min_time[ivar] = 0.0;
     max_time[ivar] = 0.0;
     avg_time[ivar] = 0.0;
  }
}

KernelBase::~KernelBase()
{
}

void KernelBase::execute() {
#if 0 // RDH
  for (size_t i = 0; i < numIterations; ++i) {
    this->tearDown();
    this->setUp();

    // start timer
    auto start = clock::now();

    this->executeKernel();

    // stop timer
    auto end = clock::now();
    Duration time = end - start;

    min_time = std::min(min_time, time.count());
    max_time = std::max(min_time, time.count());
    avg_time += time.count();
  }

  avg_time /= (double) numIterations;

  std::cout << "Avg time: " << avg_time << "s" << std::endl;
#endif
}

}  // closing brace for rajaperf namespace
