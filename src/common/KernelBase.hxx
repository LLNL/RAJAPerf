#ifndef RAJAPerfKernelBase_HXX

#include "common/RAJAPerfSuite.hxx"

#include <chrono>
#include <string>

namespace rajaperf {

class KernelBase
{
public:

  using clock = std::chrono::steady_clock;
  using TimeType = clock::time_point;
  using Duration = std::chrono::duration<double>;

  KernelBase(size_t size, unsigned iterations);

  virtual ~KernelBase();

  double getMinTime(VariantID vid) { return min_time[vid]; }
  double getMaxTime(VariantID vid) { return max_time[vid]; }
  double getAvgTime(VariantID vid) { return avg_time[vid]; }

  void execute();

  //virtual bool checkCorrectness();

  virtual void setUp() = 0;
  virtual void executeKernel() = 0;
  virtual void tearDown() = 0;

protected:
  size_t size;
  size_t numIterations;

private:
  std::string name;

  double min_time[NUM_VARIANTS];
  double max_time[NUM_VARIANTS];
  double avg_time[NUM_VARIANTS];
};

}  // closing brace for rajaperf namespace

#endif  // closing endif for header file include guard
