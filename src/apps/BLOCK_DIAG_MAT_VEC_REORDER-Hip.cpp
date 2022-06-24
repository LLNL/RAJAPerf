//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "BLOCK_DIAG_MAT_VEC_REORDER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP_HIP                                          \
  const Index_type N = m_N;                                             \
  allocAndInitHipDeviceData(Me, m_Me, N);                                       \
  allocAndInitHipDeviceData(X, m_X, N);                                       \
  allocAndInitHipDeviceData(Y, m_Y, N);

#define BLOCK_DIAG_MAT_VEC_REORDER_DATA_TEARDOWN_HIP                                       \
  getHipDeviceData(m_Me, Me, N);                                                \
  getHipDeviceData(m_X, X, N);                                                \
  getHipDeviceData(m_Y, Y, N);                                                \
  deallocHipDeviceData(Me);                                                     \
  deallocHipDeviceData(X);                                                     \
  deallocHipDeviceData(Y);

template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void BLOCK_DIAG_MAT_VEC_REORDER(Index_type N, Real_ptr Me, Real_ptr X,
                               Real_ptr Y) {

}

template < size_t block_size >
void BLOCK_DIAG_MAT_VEC_REORDER::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP;

  if (vid == Base_HIP) {

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_TEARDOWN_HIP;

  } else if (vid == Lambda_HIP) {

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }  // loop over kernel reps
    stopTimer();

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_TEARDOWN_HIP;

  } else {
    getCout() << "\n  BLOCK_DIAG_MAT_VEC_REORDER : Unknown Hip variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(BLOCK_DIAG_MAT_VEC_REORDER, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
