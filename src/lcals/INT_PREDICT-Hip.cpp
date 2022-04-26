//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INT_PREDICT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

#define INT_PREDICT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(px, m_px, m_array_length);

#define INT_PREDICT_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_px, px, m_array_length); \
  deallocHipDeviceData(px);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void int_predict(Real_ptr px,
                            Real_type dm22, Real_type dm23, Real_type dm24,
                            Real_type dm25, Real_type dm26, Real_type dm27,
                            Real_type dm28, Real_type c0,
                            const Index_type offset,
                            Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     INT_PREDICT_BODY;
   }
}


template < size_t block_size >
void INT_PREDICT::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  INT_PREDICT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    INT_PREDICT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hipLaunchKernelGGL((int_predict<block_size>), dim3(grid_size), dim3(block_size), 0, 0,  px,
                                               dm22, dm23, dm24, dm25,
                                               dm26, dm27, dm28, c0,
                                               offset,
                                               iend );
       hipErrchk( hipGetLastError() );

    }
    stopTimer();

    INT_PREDICT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    INT_PREDICT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         INT_PREDICT_BODY;
       });

    }
    stopTimer();

    INT_PREDICT_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  INT_PREDICT : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(INT_PREDICT, Hip)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
