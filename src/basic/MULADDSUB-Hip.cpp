//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULADDSUB.hpp"

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
__global__ void muladdsub(Real_ptr out1, Real_ptr out2, Real_ptr out3,
                          Real_ptr in1, Real_ptr in2,
                          Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    MULADDSUB_BODY;
  }
}



template < size_t block_size >
void MULADDSUB::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  MULADDSUB_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MULADDSUB_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (muladdsub<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         out1, out2, out3,
                         in1, in2,
                         iend );
      RP_CALI_SUBKERNEL_END("MULADDSUB_1");

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MULADDSUB_1");
      auto muladdsub_lambda = [=] __device__ (Index_type i) {
        MULADDSUB_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (lambda_hip_forall<block_size,
                                            decltype(muladdsub_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         ibegin, iend, muladdsub_lambda );
      RP_CALI_SUBKERNEL_END("MULADDSUB_1");

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MULADDSUB_1");
      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        MULADDSUB_BODY;
      });
      RP_CALI_SUBKERNEL_END("MULADDSUB_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  MULADDSUB : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MULADDSUB, Hip, Base_HIP, Lambda_HIP, RAJA_HIP)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
