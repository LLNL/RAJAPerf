//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MEMSET.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void memset(Real_ptr x, Real_type val,
                       Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if ( i < iend ) {
    MEMSET_BODY;
  }
}


void MEMSET::runCudaVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  MEMSET_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
      CAMP_CUDA_API_INVOKE_AND_CHECK( cudaMemsetAsync,
          MEMSET_STD_ARGS, res.get_stream() );
      RP_CALI_SUBKERNEL_END("MEMSET_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
      res.memset(MEMSET_STD_ARGS);
      RP_CALI_SUBKERNEL_END("MEMSET_1");

    }
    stopTimer();

  } else {

    getCout() << "\n  MEMSET : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void MEMSET::runCudaVariantBlock(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  MEMSET_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (memset<block_size>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          x, val, iend );
      RP_CALI_SUBKERNEL_END("MEMSET_1");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
      auto memset_lambda = [=] __device__ (Index_type i) {
        MEMSET_BODY;
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(memset_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, memset_lambda );
      RP_CALI_SUBKERNEL_END("MEMSET_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MEMSET_1");
      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          MEMSET_BODY;
      });
      RP_CALI_SUBKERNEL_END("MEMSET_1");

    }
    stopTimer();

  } else {

    getCout() << "\n  MEMSET : Unknown Cuda variant id = " << vid << std::endl;

  }

}


// _memset_define_cuda_start
void MEMSET::defineCudaVariantTunings()
{

  for (VariantID vid : {Base_CUDA, Lambda_CUDA, RAJA_CUDA}) {

    if (vid == Base_CUDA || vid == RAJA_CUDA) {

      addVariantTuning<&MEMSET::runCudaVariantLibrary>(
          vid, "library");

    }

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuning<&MEMSET::runCudaVariantBlock<block_size>>(
            vid, "block_"+std::to_string(block_size));

      }

    });

  }

}
// _memset_define_cuda_end

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
