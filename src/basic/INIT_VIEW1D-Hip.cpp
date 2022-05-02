//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define INIT_VIEW1D_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(a, m_a, iend);

#define INIT_VIEW1D_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_a, a, iend); \
  deallocHipDeviceData(a);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void initview1d(Real_ptr a,
                           Real_type v,
                           const Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    INIT_VIEW1D_BODY;
  }
}



template < size_t block_size >
void INIT_VIEW1D::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INIT_VIEW1D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    INIT_VIEW1D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((initview1d<block_size>), dim3(grid_size), dim3(block_size), 0, 0,
          a, v, iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    INIT_VIEW1D_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto initview1d_lambda = [=] __device__ (Index_type i) {
        INIT_VIEW1D_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(initview1d_lambda)>),
        grid_size, block_size, 0, 0, ibegin, iend, initview1d_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    INIT_VIEW1D_DATA_SETUP_HIP;

    INIT_VIEW1D_VIEW_RAJA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        INIT_VIEW1D_BODY_RAJA;
      });

    }
    stopTimer();

    INIT_VIEW1D_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  INIT_VIEW1D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(INIT_VIEW1D, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
