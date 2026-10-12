//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void pressurecalc1(Real_ptr bvc, Real_ptr compression,
                              const Real_type cls,
                              Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     PRESSURE_BODY1;
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void pressurecalc2(Real_ptr p_new, Real_ptr bvc, Real_ptr e_old,
                              Real_ptr vnewc,
                              const Real_type p_cut, const Real_type eosvmax,
                              const Real_type pmin,
                              Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     PRESSURE_BODY2;
   }
}


template < size_t block_size >
void PRESSURE::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  PRESSURE_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
      RPlaunchHipKernel( (pressurecalc1<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         bvc, compression, cls, 
                         iend );
      RP_CALI_SUBKERNEL_END("PRESSURE_1");

      RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
      RPlaunchHipKernel( (pressurecalc2<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         p_new, bvc, e_old,
                         vnewc,
                         p_cut, eosvmax, pmin,
                         iend );
      RP_CALI_SUBKERNEL_END("PRESSURE_2");

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    const bool async = true;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RAJA::region<RAJA::seq_region>( [=]() {

        RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
        RAJA::forall< RAJA::hip_exec<block_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          PRESSURE_BODY1;
        });
        RP_CALI_SUBKERNEL_END("PRESSURE_1");
        RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
        RAJA::forall< RAJA::hip_exec<block_size, async> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          PRESSURE_BODY2;
        });
        RP_CALI_SUBKERNEL_END("PRESSURE_2");

      });  // end sequential region (for single-source code)

    }
    stopTimer();

  } else {
     getCout() << "\n  PRESSURE : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(PRESSURE, Hip, Base_HIP, RAJA_HIP)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
