//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DAXPY_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define DAXPY_ATOMIC_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend); \
  allocAndInitHipDeviceData(y, m_y, iend);

#define DAXPY_ATOMIC_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_y, y, iend); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void daxpy_atomic(Real_ptr y, Real_ptr x,
                      Real_type a,
                      Index_type iend)
{
   Index_type i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < iend) {
     DAXPY_ATOMIC_RAJA_BODY(RAJA::hip_atomic);
   }
}


template < size_t block_size >
void DAXPY_ATOMIC::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DAXPY_ATOMIC_DATA_SETUP;

  if ( vid == Base_HIP ) {

    DAXPY_ATOMIC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((daxpy_atomic<block_size>),dim3(grid_size), dim3(block_size), 0, 0, y, x, a,
                                        iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    DAXPY_ATOMIC_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    DAXPY_ATOMIC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto daxpy_atomic_lambda = [=] __device__ (Index_type i) {
        DAXPY_ATOMIC_RAJA_BODY(RAJA::hip_atomic);
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(daxpy_atomic_lambda)>),
        grid_size, block_size, 0, 0, ibegin, iend, daxpy_atomic_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    DAXPY_ATOMIC_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    DAXPY_ATOMIC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        DAXPY_ATOMIC_RAJA_BODY(RAJA::hip_atomic);
      });

    }
    stopTimer();

    DAXPY_ATOMIC_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  DAXPY_ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(DAXPY_ATOMIC, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
