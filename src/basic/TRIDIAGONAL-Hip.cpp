//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAGONAL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define TRIDIAGONAL_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(Aa_global, m_Aa_global, m_N*iend); \
  allocAndInitHipDeviceData(Ab_global, m_Ab_global, m_N*iend); \
  allocAndInitHipDeviceData(Ac_global, m_Ac_global, m_N*iend); \
  allocAndInitHipDeviceData(x_global, m_x_global, m_N*iend); \
  allocAndInitHipDeviceData(b_global, m_b_global, m_N*iend);

#define TRIDIAGONAL_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_x_global, x_global, m_N*iend); \
  deallocHipDeviceData(Aa_global); \
  deallocHipDeviceData(Ab_global); \
  deallocHipDeviceData(Ac_global); \
  deallocHipDeviceData(x_global); \
  deallocHipDeviceData(b_global); \
  deallocHipDeviceData(d_global);

#define TRIDIAGONAL_TEMP_DATA_SETUP_HIP \
  Real_ptr d_global; \
  allocHipDeviceData(d_global, m_N*iend);

#define TRIDIAGONAL_TEMP_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(d_global);

#define TRIDIAGONAL_LOCAL_DATA_SETUP_HIP \
  TRIDIAGONAL_LOCAL_DATA_SETUP; \
  Real_ptr d = d_global + TRIDIAGONAL_OFFSET(i);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void tridiagonal(Real_ptr Aa_global, Real_ptr Ab_global, Real_ptr Ac_global,
                            Real_ptr  x_global, Real_ptr  b_global, Real_ptr  d_global,
                            Index_type N, Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    TRIDIAGONAL_LOCAL_DATA_SETUP_HIP;
    TRIDIAGONAL_BODY_FORWARD_V2;
    TRIDIAGONAL_BODY_BACKWARD_V2;
  }
}


template < size_t block_size >
void TRIDIAGONAL::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIDIAGONAL_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRIDIAGONAL_DATA_SETUP_HIP;
    TRIDIAGONAL_TEMP_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((tridiagonal<block_size>), dim3(grid_size), dim3(block_size), 0, 0,
          Aa_global, Ab_global, Ac_global,
          x_global, b_global, d_global,
          N, iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIDIAGONAL_TEMP_DATA_TEARDOWN_HIP;
    TRIDIAGONAL_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    TRIDIAGONAL_DATA_SETUP_HIP;
    TRIDIAGONAL_TEMP_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto tridiagonal_lambda = [=] __device__ (Index_type i) {
        TRIDIAGONAL_LOCAL_DATA_SETUP_HIP;
        TRIDIAGONAL_BODY_FORWARD_V2;
        TRIDIAGONAL_BODY_BACKWARD_V2;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(tridiagonal_lambda)>),
        grid_size, block_size, 0, 0, ibegin, iend, tridiagonal_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIDIAGONAL_TEMP_DATA_TEARDOWN_HIP;
    TRIDIAGONAL_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    TRIDIAGONAL_DATA_SETUP_HIP;
    TRIDIAGONAL_TEMP_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        TRIDIAGONAL_LOCAL_DATA_SETUP_HIP;
        TRIDIAGONAL_BODY_FORWARD_V2;
        TRIDIAGONAL_BODY_BACKWARD_V2;
      });

    }
    stopTimer();

    TRIDIAGONAL_TEMP_DATA_TEARDOWN_HIP;
    TRIDIAGONAL_DATA_TEARDOWN_HIP;

  } else {
      getCout() << "\n  TRIDIAGONAL : Unknown Hip variant id = " << vid << std::endl;
  }
}

void TRIDIAGONAL::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tune_idx == t) {
        runHipVariantImpl<block_size>(vid);
      }
      t += 1;
    }
  });
}

void TRIDIAGONAL::setHipTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      addVariantTuningName(vid, "block_"+std::to_string(block_size));
    }
  });
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
