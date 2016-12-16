#ifndef RAJAPerf_LCALS_MULADDSUB_HXX
#define RAJAPerf_LCALS_MULADDSUB_HXX

#include "common/KernelBase.hxx"

namespace rajaperf 
{
namespace lcals
{

class MULADDSUB : public KernelBase
{
  public:

  MULADDSUB(size_t size, unsigned iterations); 

  ~MULADDSUB();

  void setUp();
  void executeKernel(); 
  void tearDown();

};

} // end namespace lcals
} // end namespace rajaperf

#endif // closing endif for header file include guard
