//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MUL.hpp"

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
__global__ void mul(Real_ptr b, Real_ptr c, Real_type alpha,
                    Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    MUL_BODY;
  }
}


template < size_t block_size >
void MUL::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  MUL_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MUL_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (mul<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         b, c, alpha, iend );
      RP_CALI_SUBKERNEL_END("MUL_1");

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MUL_1");
      auto mul_lambda = [=] __device__ (Index_type i) {
        MUL_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (lambda_hip_forall<block_size,
                                            decltype(mul_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         ibegin, iend, mul_lambda );
      RP_CALI_SUBKERNEL_END("MUL_1");

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MUL_1");
      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        MUL_BODY;
      });
      RP_CALI_SUBKERNEL_END("MUL_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  MUL : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MUL, Hip, Base_HIP, Lambda_HIP, RAJA_HIP)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
