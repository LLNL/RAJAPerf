//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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

#define EOS_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, m_array_length); \
  allocAndInitHipDeviceData(y, m_y, m_array_length); \
  allocAndInitHipDeviceData(z, m_z, m_array_length); \
  allocAndInitHipDeviceData(u, m_u, m_array_length);

#define EOS_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_x, x, m_array_length); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(z); \
  deallocHipDeviceData(u);

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
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  EOS_DATA_SETUP;

  if ( vid == Base_HIP ) {

    EOS_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hipLaunchKernelGGL((eos<block_size>), dim3(grid_size), dim3(block_size), 0, 0,  x, y, z, u,
                                       q, r, t,
                                       iend );
       hipErrchk( hipGetLastError() );

    }
    stopTimer();

    EOS_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    EOS_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         EOS_BODY;
       });

    }
    stopTimer();

    EOS_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  EOS : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(EOS, Hip)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
