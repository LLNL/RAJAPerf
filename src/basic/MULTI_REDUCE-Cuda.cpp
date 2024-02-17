//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MULTI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

template < size_t block_size, size_t replication >
__launch_bounds__(block_size)
__global__ void multi_reduce(Real_ptr values,
                             Index_ptr bins,
                             Real_ptr data,
                             Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    MULTI_REDUCE_GPU_RAJA_BODY(RAJA::cuda_atomic);
  }
}



template < size_t block_size, size_t replication >
void MULTI_REDUCE::runCudaVariantReplicateGlobal(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  MULTI_REDUCE_GPU_DATA_SETUP;

  RAJAPERF_CUDA_REDUCER_SETUP(Real_ptr, values, hvalues, num_bins, replication);

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(values_init, values, hvalues, num_bins, replication);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (multi_reduce<block_size, replication>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          values,
                          bins,
                          data,
                          iend );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(values, hvalues, num_bins, replication);
      MULTI_REDUCE_GPU_FINALIZE_VALUES(hvalues, num_bins, replication);

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(values_init, values, hvalues, num_bins, replication);

      auto multi_reduce_lambda = [=] __device__ (Index_type i) {
        MULTI_REDUCE_GPU_RAJA_BODY(RAJA::cuda_atomic);
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(multi_reduce_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, multi_reduce_lambda );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(values, hvalues, num_bins, replication);
      MULTI_REDUCE_GPU_FINALIZE_VALUES(hvalues, num_bins, replication);

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(values_init, values, hvalues, num_bins, replication);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          MULTI_REDUCE_GPU_RAJA_BODY(RAJA::cuda_atomic);
      });

      RAJAPERF_CUDA_REDUCER_COPY_BACK(values, hvalues, num_bins, replication);
      MULTI_REDUCE_GPU_FINALIZE_VALUES(hvalues, num_bins, replication);

    }
    stopTimer();

  } else {
     getCout() << "\n  MULTI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }

  RAJAPERF_CUDA_REDUCER_TEARDOWN(values, hvalues);

}

void MULTI_REDUCE::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA || vid == Lambda_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(replication)) {

            if (tune_idx == t) {

              setBlockSize(block_size);
              runCudaVariantReplicateGlobal<decltype(block_size)::value, replication>(vid);

            }

            t += 1;

          }

        });

      }

    });

  } else {

    getCout() << "\n  MULTI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void MULTI_REDUCE::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA || vid == Lambda_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(replication)) {

            addVariantTuningName(vid, "replicate_"+std::to_string(replication)+
                                      "_global_"+std::to_string(block_size));

          }

        });

      }

    });

  }

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
