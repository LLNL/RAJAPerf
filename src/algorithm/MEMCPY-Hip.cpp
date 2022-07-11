//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMCPY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define MEMCPY_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend); \
  allocAndInitHipDeviceData(y, m_y, iend);

#define MEMCPY_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_y, y, iend); \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void memcpy(Real_ptr x, Real_ptr y,
                       Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if ( i < iend ) {
    MEMCPY_BODY;
  }
}


void MEMCPY::runHipVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  if ( vid == Base_HIP ) {

    MEMCPY_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk( hipMemcpyAsync(MEMCPY_STD_ARGS, hipMemcpyDefault, 0) );

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    MEMCPY_DATA_SETUP_HIP;

    camp::resources::Hip res = camp::resources::Hip::get_default();

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      res.memcpy(MEMCPY_STD_ARGS);

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  MEMCPY : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void MEMCPY::runHipVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  MEMCPY_DATA_SETUP;

  if ( vid == Base_HIP ) {

    MEMCPY_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (memcpy<block_size>),
          dim3(grid_size), dim3(block_size), 0, 0,
          x, y, iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    MEMCPY_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto memcpy_lambda = [=] __device__ (Index_type i) {
        MEMCPY_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(memcpy_lambda)>),
          grid_size, block_size, 0, 0,
          ibegin, iend, memcpy_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    MEMCPY_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          MEMCPY_BODY;
      });

    }
    stopTimer();

    MEMCPY_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  MEMCPY : Unknown Hip variant id = " << vid << std::endl;

  }

}

void MEMCPY::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if (vid == Base_HIP || vid == RAJA_HIP) {

    if (tune_idx == t) {

      runHipVariantLibrary(vid);

    }

    t += 1;

  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      if (tune_idx == t) {

        runHipVariantBlock<block_size>(vid);

      }

      t += 1;

    }

  });

}

void MEMCPY::setHipTuningDefinitions(VariantID vid)
{
  if (vid == Base_HIP || vid == RAJA_HIP) {
    addVariantTuningName(vid, "library");
  }

  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      addVariantTuningName(vid, "block_"+std::to_string(block_size));

    }

  });

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
