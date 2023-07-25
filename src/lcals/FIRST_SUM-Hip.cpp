//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "FIRST_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void first_sum(Real_ptr x, Real_ptr y,
                          Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i > 0 && i < iend) {
     FIRST_SUM_BODY;
   }
}


template < size_t block_size >
void FIRST_SUM::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  FIRST_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       constexpr size_t shmem = 0;
       hipLaunchKernelGGL((first_sum<block_size>),grid_size, block_size, shmem, res.get_stream(), x, y,
                                              iend );
       hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         FIRST_SUM_BODY;
       });

    }
    stopTimer();

  } else {
     getCout() << "\n  FIRST_SUM : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(FIRST_SUM, Hip)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
