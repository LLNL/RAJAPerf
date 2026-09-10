//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EMPTY.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void empty(Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    EMPTY_BODY;
  }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void empty_grid_stride(Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  Index_type grid_stride = gridDim.x * block_size;
  for ( ; i < iend; i += grid_stride) {
    EMPTY_BODY;
  }
}


template < size_t block_size, typename MappingHelper >
void EMPTY::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  EMPTY_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    auto func = MappingHelper::direct
        ? &empty<block_size>
        : &empty_grid_stride<block_size>;

    constexpr size_t shmem = 0;
    const size_t max_grid_size = RAJAPERF_CUDA_GET_MAX_BLOCKS(
        MappingHelper, func, block_size, shmem);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("EMPTY_1");
      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchCudaKernel( func,
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          iend );
      RP_CALI_SUBKERNEL_END("EMPTY_1");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    auto empty_lambda = [=] __device__ (Index_type i) {
      EMPTY_BODY;
    };

    auto func = MappingHelper::direct
        ? &lambda_cuda_forall<block_size, decltype(empty_lambda)>
        : &lambda_cuda_forall_grid_stride<block_size, decltype(empty_lambda)>;

    constexpr size_t shmem = 0;
    const size_t max_grid_size = RAJAPERF_CUDA_GET_MAX_BLOCKS(
        MappingHelper, func, block_size, shmem);

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("EMPTY_1");
      const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      const size_t grid_size = std::min(normal_grid_size, max_grid_size);

      RPlaunchCudaKernel( func,
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, empty_lambda );
      RP_CALI_SUBKERNEL_END("EMPTY_1");

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    using exec_policy = std::conditional_t<MappingHelper::direct,
        RAJA::cuda_exec<block_size, true /*async*/>,
        RAJA::cuda_exec_occ_calc<block_size, true /*async*/>>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("EMPTY_1");
      RAJA::forall< exec_policy >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        EMPTY_BODY;
      });
      RP_CALI_SUBKERNEL_END("EMPTY_1");

    }
    stopTimer();

  } else {
     getCout() << "\n  EMPTY : Unknown Cuda variant id = " << vid << std::endl;
  }
}


void EMPTY::defineCudaVariantTunings()
{

  for (VariantID vid : {Base_CUDA, Lambda_CUDA, RAJA_CUDA}) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::forall_helpers{}, [&](auto mapping_helper) {

            addVariantTuning<&EMPTY::runCudaVariantImpl<
                                 decltype(block_size){},
                                 decltype(mapping_helper)>>(
                vid, decltype(mapping_helper)::get_name()+"_"+
                     std::to_string(block_size));

        });

      }

    });

  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
