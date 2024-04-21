//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "COPY.hpp"

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
__global__ void copy(Real_ptr c, Real_ptr a,
                     Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    COPY_BODY;
  }
}


template < size_t block_size >
void COPY::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  COPY_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (copy<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          c, a, iend );

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto copy_lambda = [=] __device__ (Index_type i) {
        COPY_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size, 
                                              decltype(copy_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, copy_lambda );

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        COPY_BODY;
      });

    }
    stopTimer();

  } else {
      getCout() << "\n  COPY : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(COPY, Cuda)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
