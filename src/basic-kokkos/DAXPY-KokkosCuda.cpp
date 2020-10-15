//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf 
{
namespace basic
{

struct DaxpyCudaFunctor {
  Real_ptr x;
  Real_ptr y;
  Real_type a;
  DaxpyCudaFunctor(Real_ptr m_x, Real_ptr m_y, Real_type m_a) : DAXPY_FUNCTOR_CONSTRUCT {  }
  KOKKOS_FUNCTION void operator()(Index_type i) const { DAXPY_BODY; }
};

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define DAXPY_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, iend); \
  allocAndInitCudaDeviceData(y, m_y, iend);

#define DAXPY_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_y, y, iend); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y);


void DAXPY::runKokkosCudaVariant(VariantID vid)
{
#if defined(RUN_KOKKOS)
#if defined(RAJA_ENABLE_CUDA)
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getRunSize();

  DAXPY_DATA_SETUP;
  if ( vid == Kokkos_Functor_CUDA) {
    DAXPY_DATA_SETUP_CUDA;
    DaxpyCudaFunctor daxpy_functor_instance(y,x,a);                                

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep)    {

        Kokkos::parallel_for("perfsuite.kokkos.seq.functor", Kokkos::RangePolicy<Kokkos::Cuda>(ibegin, iend),
                             daxpy_functor_instance);

    }
    stopTimer();

    DAXPY_DATA_TEARDOWN_CUDA;

  } else if ( vid == Kokkos_Lambda_CUDA ) {

    DAXPY_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

	    Kokkos::parallel_for("perfsuite.kokkos.cuda.lambda",
        Kokkos::RangePolicy<Kokkos::Cuda>(ibegin, iend), [=] __device__ (Index_type i) {
        DAXPY_BODY;
      });

    }
    stopTimer();

    DAXPY_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  DAXPY : Unknown Cuda variant id = " << vid << std::endl;
  }
#endif // RAJA_ENABLE_CUDA
#endif // RUN_KOKKOS
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
