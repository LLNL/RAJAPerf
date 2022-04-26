//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_FUSED_MUL_ADD.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

#define MAT_FUSED_MUL_ADD_DATA_SETUP_HIP           \
  const Index_type N = m_N;                        \
  const Index_type Ne = m_Ne;                      \
  allocAndInitHipDeviceData(A, m_A, N);            \
  allocAndInitHipDeviceData(B, m_B, N);            \
  allocAndInitHipDeviceData(D, m_D, N);			   \
  allocAndInitHipDeviceData(Ae, m_Ae, Ne);         \
  allocAndInitHipDeviceData(Be, m_Be, Ne);         \
  allocAndInitHipDeviceData(De, m_De, Ne);

#define MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP        \
  getHipDeviceData(m_A, A, N);                     \
  getHipDeviceData(m_B, B, N);                     \
  getHipDeviceData(m_D, D, N);                     \
  getHipDeviceData(m_Ae, Ae, Ne);                  \
  getHipDeviceData(m_Be, Be, Ne);                  \
  getHipDeviceData(m_De, De, Ne);                  \
  deallocHipDeviceData(A);                         \
  deallocHipDeviceData(B);                         \
  deallocHipDeviceData(D);						   \
  deallocHipDeviceData(Ae);                        \
  deallocHipDeviceData(Be);                        \
  deallocHipDeviceData(De);

template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void MAT_FUSED_MUL_ADD(Index_type N, Real_ptr A, Real_ptr B,
                               Real_ptr D) {

}

template < size_t block_size >
void MAT_FUSED_MUL_ADD::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;
  const Index_type Ne = m_Ne;

  MAT_FUSED_MUL_ADD_DATA_SETUP;

  if (vid == Base_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;

  } else if (vid == Lambda_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    MAT_FUSED_MUL_ADD_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
    }  // loop over kernel reps
    stopTimer();

    MAT_FUSED_MUL_ADD_DATA_TEARDOWN_HIP;

  } else {
    getCout() << "\n  MAT_FUSED_MUL_ADD : Unknown Hip variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MAT_FUSED_MUL_ADD, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
