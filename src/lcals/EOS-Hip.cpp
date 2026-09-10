//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EOS.hpp"

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
__global__ void eos(Real_ptr x, Real_ptr y, Real_ptr z, Real_ptr u,
                    Real_type q, Real_type r, Real_type t,
                    Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     EOS_BODY;
   }
}


template < size_t block_size >
void EOS::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  EOS_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("EOS_1");
       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       constexpr size_t shmem = 0;

       RPlaunchHipKernel( (eos<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          x, y, z, u,
                          q, r, t, 
                          iend );
       RP_CALI_SUBKERNEL_END("EOS_1");

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("EOS_1");
       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         EOS_BODY;
       });
       RP_CALI_SUBKERNEL_END("EOS_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  EOS : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(EOS, Hip, Base_HIP, RAJA_HIP)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
