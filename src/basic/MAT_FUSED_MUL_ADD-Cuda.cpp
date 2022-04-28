//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

#define MAT_FUSED_MUL_ADD_DATA_SETUP_CUDA           \
  const Index_type N = m_N;                         \
  const Index_type Ne = m_Ne;                       \
  allocAndInitCudaDeviceData(A, m_A, N);            \
  allocAndInitCudaDeviceData(B, m_B, N);            \
  allocAndInitCudaDeviceData(D, m_D, N);			

#define MAT_FUSED_MUL_ADD_DATA_TEARDOWN_CUDA        \
  getCudaDeviceData(m_A, A, N);                     \
  getCudaDeviceData(m_B, B, N);                     \
  getCudaDeviceData(m_D, D, N);                     \
  deallocCudaDeviceData(A);                         \
  deallocCudaDeviceData(B);                         \
  deallocCudaDeviceData(D);							

template < Index_type block_size >
  __launch_bounds__(block_size)
__global__ void MAT_FUSED_MUL_ADD(Index_type N, Real_ptr A, Real_ptr B,
                               Real_ptr D) {

}

template < size_t block_size >
void MAT_FUSED_MUL_ADD::runCudaVariantImpl(VariantID vid)
{

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  const Index_type Ne = m_Ne;

  MAT_FUSED_MUL_ADD_DATA_SETUP;

  if (vid == Base_CUDA) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_CUDA;

  } else if (vid == Lambda_CUDA) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }  // loop over kernel reps
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_CUDA;

  } else {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown Cuda variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MAT_FUSED_MUL_ADD, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
