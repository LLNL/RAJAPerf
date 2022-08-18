//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

#define TRIAD_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(a, m_a, iend); \
  allocAndInitHipDeviceData(b, m_b, iend); \
  allocAndInitHipDeviceData(c, m_c, iend);

#define TRIAD_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_a, a, iend); \
  deallocHipDeviceData(a); \
  deallocHipDeviceData(b); \
  deallocHipDeviceData(c);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void triad(Real_ptr a, Real_ptr b, Real_ptr c, Real_type alpha,
                      Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    TRIAD_BODY;
  }
}


template < size_t block_size >
void TRIAD::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIAD_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRIAD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((triad<block_size>), dim3(grid_size), dim3(block_size), 0, 0,  a, b, c, alpha,
                                        iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIAD_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    TRIAD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto triad_lambda = [=] __device__ (Index_type i) {
        TRIAD_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(triad_lambda)>),
        grid_size, block_size, 0, 0, ibegin, iend, triad_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIAD_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    TRIAD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        TRIAD_BODY;
      });

    }
    stopTimer();

    TRIAD_DATA_TEARDOWN_HIP;

  } else {
      getCout() << "\n  TRIAD : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(TRIAD, Hip)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
