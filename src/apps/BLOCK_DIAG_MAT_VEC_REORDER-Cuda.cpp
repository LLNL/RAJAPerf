//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "BLOCK_DIAG_MAT_VEC_REORDER.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP_CUDA                                         \
  const Index_type N = m_N;                                             \
  allocAndInitCudaDeviceData(Me, m_Me, N);                                      \
  allocAndInitCudaDeviceData(X, m_X, N);                                      \
  allocAndInitCudaDeviceData(Y, m_Y, N);

#define BLOCK_DIAG_MAT_VEC_REORDER_DATA_TEARDOWN_CUDA                                      \
  getCudaDeviceData(m_Me, Me, N);                                               \
  getCudaDeviceData(m_X, X, N);                                               \
  getCudaDeviceData(m_Y, Y, N);                                               \
  deallocCudaDeviceData(Me);                                                    \
  deallocCudaDeviceData(X);                                                    \
  deallocCudaDeviceData(Y);

template < Index_type block_size >
  __launch_bounds__(block_size)
__global__ void BLOCK_DIAG_MAT_VEC_REORDER(Index_type N, Real_ptr Me, Real_ptr X,
                               Real_ptr Y) {

}

template < size_t block_size >
void BLOCK_DIAG_MAT_VEC_REORDER::runCudaVariantImpl(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP;

  if (vid == Base_CUDA) {

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_TEARDOWN_CUDA;

  } else if (vid == Lambda_CUDA) {

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }  // loop over kernel reps
    stopTimer();

    BLOCK_DIAG_MAT_VEC_REORDER_DATA_TEARDOWN_CUDA;

  } else {
    getCout() << "\n  BLOCK_DIAG_MAT_VEC_REORDER : Unknown Cuda variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(BLOCK_DIAG_MAT_VEC_REORDER, Cuda)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
