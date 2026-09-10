//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

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
__global__ void tridiag_elim(Real_ptr xout, Real_ptr xin,
                             Real_ptr y, Real_ptr z,
                             Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i > 0 && i < N) {
     TRIDIAG_ELIM_BODY;
   }
}


template < size_t block_size >
void TRIDIAG_ELIM::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  auto res{getHipResource()};

  TRIDIAG_ELIM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("TRIDIAG_ELIM_1");
       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       constexpr size_t shmem = 0;

       RPlaunchHipKernel( (tridiag_elim<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          xout, xin,
                          y, z,
                          iend );
       RP_CALI_SUBKERNEL_END("TRIDIAG_ELIM_1");

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

       RP_CALI_SUBKERNEL_BEGIN("TRIDIAG_ELIM_1");
       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         TRIDIAG_ELIM_BODY;
       });
       RP_CALI_SUBKERNEL_END("TRIDIAG_ELIM_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  TRIDIAG_ELIM : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(TRIDIAG_ELIM, Hip, Base_HIP, RAJA_HIP)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
