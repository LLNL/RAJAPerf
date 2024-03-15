//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include "cub/device/device_reduce.cuh"
#include "cub/util_allocator.cuh"

#include <iostream>
#include <utility>
#include <type_traits>
#include <limits>


namespace rajaperf
{
namespace algorithm
{

static constexpr size_t atomic_stride =
    (detail::cuda::cache_line_size > sizeof(REDUCE_SUM::Data_type))
    ? detail::cuda::cache_line_size / sizeof(REDUCE_SUM::Data_type)
    : 1;

static constexpr size_t max_concurrent_atomics =
    (detail::cuda::max_concurrent_atomic_bytes > sizeof(REDUCE_SUM::Data_type))
    ? detail::cuda::max_concurrent_atomic_bytes / sizeof(REDUCE_SUM::Data_type)
    : 1;

using atomic_orderings = camp::list<
    detail::GetModReorderStatic,
    detail::GetStridedReorderStatic<atomic_stride, max_concurrent_atomics> >;


template < size_t block_size, typename AtomicOrdering >
__launch_bounds__(block_size)
__global__ void reduce_sum(REDUCE_SUM::Data_ptr x,
                           REDUCE_SUM::Data_ptr sum, REDUCE_SUM::Data_type sum_init,
                           Index_type iend,
                           AtomicOrdering atomic_ordering)
{
  extern __shared__ REDUCE_SUM::Data_type psum[ ];

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  psum[ threadIdx.x ] = sum_init;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    psum[ threadIdx.x ] += x[i];
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      psum[ threadIdx.x ] += psum[ threadIdx.x + i ];
    }
     __syncthreads();
  }

  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::cuda_atomic>( &sum[atomic_ordering(blockIdx.x)], psum[ 0 ] );
  }
}


void REDUCE_SUM::runCudaVariantCub(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    cudaStream_t stream = res.get_stream();

    int len = iend - ibegin;

    RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, sum, hsum, 1, 1);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                           temp_storage_bytes,
                                           x+ibegin,
                                           sum,
                                           len,
                                           ::cub::Sum(),
                                           m_sum_init,
                                           stream));

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::CudaDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
      cudaErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                             temp_storage_bytes,
                                             x+ibegin,
                                             sum,
                                             len,
                                             ::cub::Sum(),
                                             m_sum_init,
                                             stream));

      RAJAPERF_CUDA_REDUCER_COPY_BACK(sum, hsum, 1, 1);
      m_sum = hsum[0];

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::CudaDevice, temp_storage);
    RAJAPERF_CUDA_REDUCER_TEARDOWN(sum, hsum);

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size, typename MappingHelper, typename AtomicOrdering >
void REDUCE_SUM::runCudaVariantBase(VariantID vid, AtomicOrdering atomic_ordering)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    constexpr size_t shmem = sizeof(Data_type)*block_size;
    const size_t max_grid_size = RAJAPERF_CUDA_GET_MAX_BLOCKS(
        MappingHelper, (reduce_sum<block_size, AtomicOrdering>), block_size, shmem);
    const size_t normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
    const size_t grid_size = std::min(normal_grid_size, max_grid_size);
    const size_t allocation_size = atomic_ordering.range(grid_size);

    RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, sum, hsum, 1, allocation_size);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(&m_sum_init, sum, hsum, 1, allocation_size);

      RPlaunchCudaKernel( (reduce_sum<block_size, AtomicOrdering>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          x, sum, m_sum_init, iend, atomic_ordering );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(sum, hsum, 1, allocation_size);
      for (size_t r = 1; r < allocation_size; ++r) {
        hsum[0] += hsum[r];
      }
      m_sum = hsum[0];

    }
    stopTimer();

    RAJAPERF_CUDA_REDUCER_TEARDOWN(sum, hsum);

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size, typename AlgorithmHelper, typename MappingHelper >
void REDUCE_SUM::runCudaVariantRAJA(VariantID vid)
{
  using reduction_policy = std::conditional_t<AlgorithmHelper::atomic,
      RAJA::cuda_reduce_atomic,
      RAJA::cuda_reduce>;

  using exec_policy = std::conditional_t<MappingHelper::direct,
      RAJA::cuda_exec<block_size, true /*async*/>,
      RAJA::cuda_exec_occ_calc<block_size, true /*async*/>>;

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  REDUCE_SUM_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<reduction_policy, Data_type> sum(m_sum_init);

      RAJA::forall<exec_policy>( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void REDUCE_SUM::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA ) {

    if (tune_idx == t) {

      runCudaVariantCub(vid);

    }

    t += 1;

  }

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::types{}, [&](auto mapping_helper) {

          if ( vid == Base_CUDA ) {

            seq_for(atomic_orderings{}, [&](auto get_atomic_ordering) {

              seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

                if (run_params.numValidAtomicReplication() == 0u ||
                    run_params.validAtomicReplication(replication)) {

                  if (tune_idx == t) {

                    setBlockSize(block_size);
                    typename decltype(get_atomic_ordering)::template type<replication> atomic_ordering;
                    runCudaVariantBase<decltype(block_size){},
                                       decltype(mapping_helper)>(vid, atomic_ordering);

                  }

                  t += 1;

                }

              });

            });

          } else if ( vid == RAJA_CUDA ) {

            seq_for(gpu_algorithm::types{}, [&](auto algorithm_helper) {

              if (tune_idx == t) {

                setBlockSize(block_size);
                runCudaVariantRAJA<decltype(block_size){},
                                   decltype(algorithm_helper),
                                   decltype(mapping_helper)>(vid);

              }

              t += 1;

            });

          }

        });

      }

    });

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void REDUCE_SUM::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA ) {

    addVariantTuningName(vid, "cub");

  }

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::types{}, [&](auto mapping_helper) {

          if ( vid == Base_CUDA ) {

            seq_for(atomic_orderings{}, [&](auto get_atomic_ordering) {

              seq_for(gpu_atomic_replications_type{}, [&](auto replication) {

                if (run_params.numValidAtomicReplication() == 0u ||
                    run_params.validAtomicReplication(replication)) {

                  auto algorithm_helper = gpu_algorithm::block_atomic_helper{};

                  addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+
                                            "_"+decltype(mapping_helper)::get_name()+
                                            "_replicate_"+std::to_string(replication)+
                                            "_order_"+get_atomic_ordering.name()+
                                            "_"+std::to_string(block_size));

                }

              });

            });

          } else if ( vid == RAJA_CUDA ) {

            seq_for(gpu_algorithm::types{}, [&](auto algorithm_helper) {

              addVariantTuningName(vid, decltype(algorithm_helper)::get_name()+"_"+
                                        decltype(mapping_helper)::get_name()+"_"+
                                        std::to_string(block_size));

            });

          }

        });

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
