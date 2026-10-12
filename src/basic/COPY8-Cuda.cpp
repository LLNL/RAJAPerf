//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY8.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void copy8(Real_ptr y0, Real_ptr y1, Real_ptr y2, Real_ptr y3,
                      Real_ptr y4, Real_ptr y5, Real_ptr y6, Real_ptr y7,
                      Real_ptr x0, Real_ptr x1, Real_ptr x2, Real_ptr x3,
                      Real_ptr x4, Real_ptr x5, Real_ptr x6, Real_ptr x7,
                      Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     COPY8_BODY;
   }
}


template < size_t block_size >
void COPY8::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  COPY8_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("COPY8_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (copy8<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          y0, y1, y2, y3, y4, y5, y6, y7,
                          x0, x1, x2, x3, x4, x5, x6, x7,
                          iend );
      RP_CALI_SUBKERNEL_END("COPY8_1");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("COPY8_1");
      auto copy8_lambda = [=] __device__ (Index_type i) {
        COPY8_BODY;
      }; 

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(copy8_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, copy8_lambda );
      RP_CALI_SUBKERNEL_END("COPY8_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("COPY8_1");
      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        COPY8_BODY;
      });
      RP_CALI_SUBKERNEL_END("COPY8_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  COPY8 : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(COPY8, Cuda, Base_CUDA, Lambda_CUDA, RAJA_CUDA)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
