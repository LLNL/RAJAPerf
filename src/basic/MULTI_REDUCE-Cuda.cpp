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

constexpr Index_type warp_size = 32;

template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void multi_reduce_atomic_runtime(MULTI_REDUCE::Data_ptr global_values,
                                            MULTI_REDUCE::Data_ptr data,
                                            Index_ptr bins,
                                            Index_type iend,
                                            Index_type num_bins,
                                            Index_type shared_replication,
                                            Index_type global_replication)
{
  if (shared_replication > 0) {

    extern __shared__ MULTI_REDUCE::Data_type shared_values[];
    for (Index_type t = threadIdx.x;
         t < Index_type(num_bins * shared_replication);
         t += block_size) {
      shared_values[t] = MULTI_REDUCE::Data_type(0);
    }
    __syncthreads();

    {
      Index_type i = blockIdx.x * block_size + threadIdx.x;
      for ( ; i < iend ; i += gridDim.x * block_size ) {
        Index_type offset = bins[i] * shared_replication + RAJA::power_of_2_mod(Index_type{threadIdx.x}, shared_replication);
        RAJA::atomicAdd<RAJA::cuda_atomic>(&shared_values[offset], data[i]);
      }
    }

    __syncthreads();
    for (Index_type bin = threadIdx.x; bin < num_bins; bin += block_size) {
      auto block_sum = MULTI_REDUCE::Data_type(0);
      for (Index_type s = 0; s < shared_replication; ++s) {
        block_sum += shared_values[bin * shared_replication + RAJA::power_of_2_mod(s, shared_replication)];
      }
      if (block_sum != MULTI_REDUCE::Data_type(0)) {
        Index_type offset = bin + RAJA::power_of_2_mod(Index_type{blockIdx.x}, global_replication) * num_bins;
        RAJA::atomicAdd<RAJA::cuda_atomic>(&global_values[offset], block_sum);
      }
    }

  } else {

    Index_type i = blockIdx.x * block_size + threadIdx.x;
    Index_type warp = i / warp_size;
    for ( ; i < iend ; i += gridDim.x * block_size ) {
      Index_type offset = bins[i] + RAJA::power_of_2_mod(warp, global_replication) * num_bins;
      RAJA::atomicAdd<RAJA::cuda_atomic>(&global_values[offset], data[i]);
    }
  }
}

template < Index_type block_size,
           Index_type preferred_global_replication,
           Index_type preferred_shared_replication,
           typename MappingHelper >
void MULTI_REDUCE::runCudaVariantAtomicRuntime(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  MULTI_REDUCE_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    auto* func = &multi_reduce_atomic_runtime<block_size>;

    cudaFuncAttributes func_attr;
    cudaErrchk(cudaFuncGetAttributes(&func_attr, (const void*)func));
    const Index_type max_shmem_per_block_in_bytes = func_attr.maxDynamicSharedSizeBytes;
    const Index_type max_shared_replication = max_shmem_per_block_in_bytes / sizeof(Data_type) / num_bins;

    const Index_type shared_replication = RAJA::prev_pow2(std::min(preferred_shared_replication, max_shared_replication));
    const Index_type shmem = shared_replication * num_bins * sizeof(Data_type);

    const Index_type max_grid_size = RAJAPERF_CUDA_GET_MAX_BLOCKS(
        MappingHelper, func, block_size, shmem);
    const Index_type normal_grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
    const Index_type grid_size = std::min(normal_grid_size, max_grid_size);

    const Index_type global_replication = RAJA::next_pow2(std::min(preferred_global_replication, grid_size));

    RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, values, hvalues, num_bins, global_replication);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(values_init, values, hvalues, num_bins, global_replication);

      RPlaunchCudaKernel( func,
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          values,
                          data,
                          bins,
                          iend,
                          num_bins,
                          shared_replication,
                          global_replication );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(values, hvalues, num_bins, global_replication);
      for (Index_type bin = 0; bin < num_bins; ++bin) {
        Data_type value_final = Data_type(0);
        for (Index_type r = 0; r < global_replication; ++r) {
          Index_type offset = bin + RAJA::power_of_2_mod(r, global_replication) * num_bins;
          value_final += hvalues[offset];
        }
        values_final[bin] = value_final;
      }

    }
    stopTimer();

    RAJAPERF_CUDA_REDUCER_TEARDOWN(values, hvalues);

  } else if ( vid == RAJA_CUDA ) {

    using exec_policy = std::conditional_t<MappingHelper::direct,
        RAJA::cuda_exec<block_size, true /*async*/>,
        RAJA::cuda_exec_occ_calc<block_size, true /*async*/>>;

    using multi_reduce_policy = RAJA::policy::cuda::cuda_multi_reduce_policy<
        RAJA::cuda::MultiReduceTuning<
          RAJA::cuda::multi_reduce_algorithm::init_host_combine_block_atomic_then_grid_atomic,
          RAJA::cuda::AtomicReplicationTuning<
            RAJA::cuda::SharedAtomicReplicationMaxPow2Concretizer<
              RAJA::cuda::ConstantPreferredReplicationConcretizer<preferred_shared_replication>>,
            RAJA::cuda::thread_xyz<>,
            RAJA::GetOffsetRight<int>>,
          RAJA::cuda::AtomicReplicationTuning<
            RAJA::cuda::GlobalAtomicReplicationMinPow2Concretizer<
              RAJA::cuda::ConstantPreferredReplicationConcretizer<preferred_global_replication>>,
            RAJA::cuda::warp_global_xyz<>,
            RAJA::GetOffsetLeft<int>>>>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      MULTI_REDUCE_INIT_VALUES_RAJA(multi_reduce_policy);

      RAJA::forall<exec_policy>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=] __device__ (Index_type i) {
        MULTI_REDUCE_BODY;
      });

      MULTI_REDUCE_FINALIZE_VALUES_RAJA(multi_reduce_policy);

    }
    stopTimer();

  } else {
     getCout() << "\n  MULTI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }

}

void MULTI_REDUCE::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_mapping::reducer_helpers{}, [&](auto mapping_helper) {

          if (camp::size<cuda_atomic_global_replications_type>::value == 0 &&
              camp::size<cuda_atomic_shared_replications_type>::value == 0 ) {

            if (tune_idx == t) {

              setBlockSize(block_size);
              runCudaVariantAtomicRuntime<decltype(block_size)::value,
                                          default_cuda_atomic_global_replication,
                                          default_cuda_atomic_shared_replication,
                                          decltype(mapping_helper)>(vid);

            }

            t += 1;

          }

          seq_for(cuda_atomic_global_replications_type{}, [&](auto global_replication) {

            if (run_params.numValidAtomicReplication() == 0u ||
                run_params.validAtomicReplication(global_replication)) {

              seq_for(cuda_atomic_shared_replications_type{}, [&](auto shared_replication) {

                if (tune_idx == t) {

                  setBlockSize(block_size);
                  runCudaVariantAtomicRuntime<decltype(block_size)::value,
                                              decltype(global_replication)::value,
                                              decltype(shared_replication)::value,
                                              decltype(mapping_helper)>(vid);

                }

                t += 1;

              });

            }

          });

        });

      }

    });

  } else {

    getCout() << "\n  MULTI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void MULTI_REDUCE::setCudaTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {

      seq_for(gpu_mapping::reducer_helpers{}, [&](auto mapping_helper) {

        if (camp::size<cuda_atomic_global_replications_type>::value == 0 &&
            camp::size<cuda_atomic_shared_replications_type>::value == 0 ) {

          addVariantTuningName(vid, "atomic_"+
                                    decltype(mapping_helper)::get_name()+"_"+
                                    std::to_string(block_size));

        }

        seq_for(cuda_atomic_global_replications_type{}, [&](auto global_replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(global_replication)) {

            seq_for(cuda_atomic_shared_replications_type{}, [&](auto shared_replication) {

              addVariantTuningName(vid, "atomic_"
                                        "shared("+std::to_string(shared_replication)+")_"+
                                        "global("+std::to_string(global_replication)+")_"+
                                        decltype(mapping_helper)::get_name()+"_"+
                                        std::to_string(block_size));

            });

          }

        });

      });

    }

  });

}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
