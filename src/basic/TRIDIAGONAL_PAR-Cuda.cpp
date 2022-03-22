//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAGONAL_PAR.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define TRIDIAGONAL_PAR_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(Aa_global, m_Aa_global, m_N*iend); \
  allocAndInitCudaDeviceData(Ab_global, m_Ab_global, m_N*iend); \
  allocAndInitCudaDeviceData(Ac_global, m_Ac_global, m_N*iend); \
  allocAndInitCudaDeviceData(x_global, m_x_global, m_N*iend); \
  allocAndInitCudaDeviceData(b_global, m_b_global, m_N*iend);

#define TRIDIAGONAL_PAR_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x_global, x_global, m_N*iend); \
  deallocCudaDeviceData(Aa_global); \
  deallocCudaDeviceData(Ab_global); \
  deallocCudaDeviceData(Ac_global); \
  deallocCudaDeviceData(x_global); \
  deallocCudaDeviceData(b_global); \
  deallocCudaDeviceData(d_global);

#define TRIDIAGONAL_PAR_TEMP_DATA_SETUP_CUDA \
  Real_ptr d_global; \
  allocCudaDeviceData(d_global, m_N*iend);

#define TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(d_global);

#define TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_CUDA \
  TRIDIAGONAL_PAR_LOCAL_DATA_SETUP; \
  Real_ptr d = d_global + TRIDIAGONAL_PAR_OFFSET(i);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void tridiagonal(Real_ptr Aa_global, Real_ptr Ab_global, Real_ptr Ac_global,
                           Real_ptr  x_global, Real_ptr  b_global, Real_ptr  d_global,
                           Index_type N, Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_CUDA;
    TRIDIAGONAL_PAR_BODY_FORWARD_V2;
    TRIDIAGONAL_PAR_BODY_BACKWARD_V2;
  }
}


template < size_t block_size >
void TRIDIAGONAL_PAR::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRIDIAGONAL_PAR_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    TRIDIAGONAL_PAR_DATA_SETUP_CUDA;
    TRIDIAGONAL_PAR_TEMP_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      tridiagonal<block_size><<<grid_size, block_size>>>(
          Aa_global, Ab_global, Ac_global,
          x_global, b_global, d_global,
          N, iend );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_CUDA;
    TRIDIAGONAL_PAR_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    TRIDIAGONAL_PAR_DATA_SETUP_CUDA;
    TRIDIAGONAL_PAR_TEMP_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda_forall<block_size><<<grid_size, block_size>>>(
        ibegin, iend, [=] __device__ (Index_type i) {
        TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_CUDA;
        TRIDIAGONAL_PAR_BODY_FORWARD_V2;
        TRIDIAGONAL_PAR_BODY_BACKWARD_V2;
      });
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_CUDA;
    TRIDIAGONAL_PAR_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    TRIDIAGONAL_PAR_DATA_SETUP_CUDA;
    TRIDIAGONAL_PAR_TEMP_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        TRIDIAGONAL_PAR_LOCAL_DATA_SETUP_CUDA;
        TRIDIAGONAL_PAR_BODY_FORWARD_V2;
        TRIDIAGONAL_PAR_BODY_BACKWARD_V2;
      });

    }
    stopTimer();

    TRIDIAGONAL_PAR_TEMP_DATA_TEARDOWN_CUDA;
    TRIDIAGONAL_PAR_DATA_TEARDOWN_CUDA;

  } else {
      getCout() << "\n  TRIDIAGONAL_PAR : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void TRIDIAGONAL_PAR::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tune_idx == t) {
        runCudaVariantImpl<block_size>(vid);
      }
      t += 1;
    }
  });
}

void TRIDIAGONAL_PAR::setCudaTuningDefinitions(VariantID vid)
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

#endif  // RAJA_ENABLE_CUDA
