//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIAD_PARTED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace stream
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void triad_parted(Real_ptr a, Real_ptr b, Real_ptr c, Real_type alpha,
                      Index_type ibegin, Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x + ibegin;
  if (i < iend) {
    TRIAD_PARTED_BODY;
  }
}


template < size_t block_size >
void TRIAD_PARTED::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  TRIAD_PARTED_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        triad_parted<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>( a, b, c, alpha,
                                          ibegin, iend );
        cudaErrchk( cudaGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend-ibegin, block_size);
        constexpr size_t shmem = 0;
        lambda_cuda_forall<block_size><<<grid_size, block_size, shmem, res.get_stream()>>>(
          ibegin, iend, [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
        cudaErrchk( cudaGetLastError() );
      }

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (size_t p = 1; p < parts.size(); ++p ) {
        const Index_type ibegin = parts[p-1];
        const Index_type iend = parts[p];

        RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
          RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          TRIAD_PARTED_BODY;
        });
      }

    }
    stopTimer();

  } else {
      getCout() << "\n  TRIAD_PARTED : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(TRIAD_PARTED, Cuda)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
