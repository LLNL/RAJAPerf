
#include "MULADDSUB.hxx"

#include "RAJA/RAJA.hxx"

namespace rajaperf 
{
namespace lcals
{

MULADDSUB::MULADDSUB(size_t size, unsigned iterations)
  : KernelBase(size, iterations) 
{
}

MULADDSUB::~MULADDSUB() 
{
}

void MULADDSUB::setUp()
{
}

void MULADDSUB::executeKernel()
{
   RAJA::Real_ptr out1;
   RAJA::Real_ptr out2;
   RAJA::Real_ptr out3;
   RAJA::Real_ptr in1;
   RAJA::Real_ptr in2;

   RAJA::forall<RAJA::seq_exec>(0, 100, [=](int i) {
      out1[i] = in1[i] * in2[i] ;
      out2[i] = in1[i] + in2[i] ;
      out3[i] = in1[i] - in2[i] ;
   }); 
}

void MULADDSUB::tearDown()
{
}

} // end namespace lcals
} // end namespace rajaperf
