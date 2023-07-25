//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ARRAY_OF_PTRS.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void array_of_ptrs(Real_ptr y, ARRAY_OF_PTRS_Array x_array,
                      Index_type array_size,
                      Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ARRAY_OF_PTRS_BODY(x_array.array);
   }
}


template < size_t block_size >
void ARRAY_OF_PTRS::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  ARRAY_OF_PTRS_DATA_SETUP;

  if ( vid == Base_HIP ) {

    ARRAY_OF_PTRS_Array x_array = x;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      hipLaunchKernelGGL((array_of_ptrs<block_size>),dim3(grid_size), dim3(block_size), shmem, res.get_stream(),
          y, x_array, array_size, iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto array_of_ptrs_lambda = [=] __device__ (Index_type i) {
        ARRAY_OF_PTRS_BODY(x);
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(array_of_ptrs_lambda)>),
        grid_size, block_size, shmem, res.get_stream(), ibegin, iend, array_of_ptrs_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        ARRAY_OF_PTRS_BODY(x);
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  ARRAY_OF_PTRS : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(ARRAY_OF_PTRS, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
