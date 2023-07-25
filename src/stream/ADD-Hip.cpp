//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void add(Real_ptr c, Real_ptr a, Real_ptr b,
                     Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    ADD_BODY;
  }
}


template < size_t block_size >
void ADD::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  ADD_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      hipLaunchKernelGGL((add<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(),  c, a, b,
                                      iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto add_lambda = [=] __device__ (Index_type i) {
        ADD_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(add_lambda)>),
        grid_size, block_size, shmem, res.get_stream(), ibegin, iend, add_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        ADD_BODY;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  ADD : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(ADD, Hip)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

