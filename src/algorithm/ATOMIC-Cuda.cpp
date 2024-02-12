//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

template < size_t block_size, size_t replication >
__launch_bounds__(block_size)
__global__ void atomic_replicate_thread(Real_ptr atomic,
                          Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ATOMIC_RAJA_BODY(RAJA::cuda_atomic, i);
   }
}

template < size_t block_size, size_t replication >
__launch_bounds__(block_size)
__global__ void atomic_replicate_block(Real_ptr atomic,
                          Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     ATOMIC_RAJA_BODY(RAJA::cuda_atomic, blockIdx.x);
   }
}


template < size_t block_size, size_t replication >
void ATOMIC::runCudaVariantReplicateGlobal(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  ATOMIC_DATA_SETUP(replication);

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (atomic_replicate_thread<block_size, replication>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         atomic,
                         iend );

    }
    stopTimer();

  } else  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall<RAJA::cuda_exec<block_size, true /*async*/>>(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          ATOMIC_RAJA_BODY(RAJA::cuda_atomic, i);
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown Cuda variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(replication);
}

template < size_t block_size, size_t replication >
void ATOMIC::runCudaVariantReplicateBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  ATOMIC_DATA_SETUP(replication);

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (atomic_replicate_block<block_size, replication>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         atomic,
                         iend );

    }
    stopTimer();

  } else {
     getCout() << "\n  ATOMIC : Unknown Cuda variant id = " << vid << std::endl;
  }

  ATOMIC_DATA_TEARDOWN(replication);
}

void ATOMIC::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(atomic_replications_type{}, [&](auto replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(replication)) {

            if (tune_idx == t) {

              setBlockSize(block_size);
              runCudaVariantReplicateGlobal<decltype(block_size)::value, replication>(vid);

            }

            t += 1;

          }

        });

        if ( vid == Base_CUDA ) {

          seq_for(atomic_replications_type{}, [&](auto replication) {

            if (run_params.numValidAtomicReplication() == 0u ||
                run_params.validAtomicReplication(replication)) {

              if (tune_idx == t) {

                setBlockSize(block_size);
                runCudaVariantReplicateBlock<decltype(block_size)::value, replication>(vid);

              }

              t += 1;

            }

          });

        }

      }

    });

  } else {

    getCout() << "\n  ATOMIC : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void ATOMIC::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(atomic_replications_type{}, [&](auto replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(replication)) {

            addVariantTuningName(vid, "replicate_"+std::to_string(replication)+
                                      "_global_"+std::to_string(block_size));

          }

        });

        if ( vid == Base_CUDA ) {

          seq_for(atomic_replications_type{}, [&](auto replication) {

            if (run_params.numValidAtomicReplication() == 0u ||
                run_params.validAtomicReplication(replication)) {

              addVariantTuningName(vid, "replicate_"+std::to_string(replication)+
                                        "_block_"+std::to_string(block_size));

            }

          });

        }

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
