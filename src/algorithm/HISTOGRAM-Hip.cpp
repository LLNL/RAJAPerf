//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HISTOGRAM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#if defined(__HIPCC__)
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_histogram.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_histogram.cuh"
#include "cub/util_allocator.cuh"
#endif

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

constexpr Index_type warp_size = 64;

constexpr Index_type default_shared_replication = 4;
constexpr Index_type default_global_replication = 16;


template < Index_type block_size >
__launch_bounds__(block_size)
__global__ void histogram_atomic_runtime(HISTOGRAM::Data_ptr global_counts,
                                         Index_ptr bins,
                                         Index_type iend,
                                         Index_type num_bins,
                                         Index_type shared_replication,
                                         Index_type global_replication)
{
  if (shared_replication > 0) {

    extern __shared__ HISTOGRAM::Data_type shared_counts[];
    for (Index_type t = threadIdx.x;
         t < Index_type(num_bins * shared_replication);
         t += block_size) {
      shared_counts[t] = HISTOGRAM::Data_type(0);
    }
    __syncthreads();

    {
      Index_type i = blockIdx.x * block_size + threadIdx.x;
      if (i < iend) {
        Index_type offset = bins[i] * shared_replication + RAJA::power_of_2_mod(threadIdx.x, shared_replication);
        RAJA::atomicAdd<RAJA::hip_atomic>(&shared_counts[offset], HISTOGRAM::Data_type(1));
      }
    }

    __syncthreads();
    for (Index_type bin = threadIdx.x; bin < num_bins; bin += block_size) {
      auto block_sum = HISTOGRAM::Data_type(0);
      for (Index_type s = 0; s < shared_replication; ++s) {
        block_sum += shared_counts[bin * shared_replication + RAJA::power_of_2_mod(s, shared_replication)];
      }
      if (block_sum != HISTOGRAM::Data_type(0)) {
        Index_type offset = bin + RAJA::power_of_2_mod(blockIdx.x, global_replication) * num_bins;
        RAJA::atomicAdd<RAJA::hip_atomic>(&global_counts[offset], block_sum);
      }
    }

  } else {

    Index_type i = blockIdx.x * block_size + threadIdx.x;
    Index_type warp = i / warp_size;
    if (i < iend) {
      Index_type offset = bins[i] + RAJA::power_of_2_mod(warp, global_replication) * num_bins;
      RAJA::atomicAdd<RAJA::hip_atomic>(&global_counts[offset], HISTOGRAM::Data_type(1));
    }
  }
}


void HISTOGRAM::runHipVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  HISTOGRAM_DATA_SETUP;

  RAJAPERF_HIP_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, 1);

  RAJAPERF_UNUSED_VAR(counts_init);

  if ( vid == Base_HIP ) {

    hipStream_t stream = res.get_stream();

    int len = iend - ibegin;

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
    hipErrchk(::rocprim::histogram_even(d_temp_storage,
                                        temp_storage_bytes,
                                        bins+ibegin,
                                        len,
                                        counts,
                                        static_cast<int>(num_bins+1),
                                        static_cast<Index_type>(0),
                                        num_bins,
                                        stream));
#elif defined(__CUDACC__)
    cudaErrchk(::cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                     temp_storage_bytes,
                                                     bins+ibegin,
                                                     counts,
                                                     static_cast<int>(num_bins+1),
                                                     static_cast<Index_type>(0),
                                                     num_bins,
                                                     len,
                                                     stream));
#endif

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::HipDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
#if defined(__HIPCC__)
      hipErrchk(::rocprim::histogram_even(d_temp_storage,
                                          temp_storage_bytes,
                                          bins+ibegin,
                                          len,
                                          counts,
                                          static_cast<int>(num_bins+1),
                                          static_cast<Index_type>(0),
                                          num_bins,
                                          stream));
#elif defined(__CUDACC__)
      cudaErrchk(::cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                       temp_storage_bytes,
                                                       bins+ibegin,
                                                       counts,
                                                       static_cast<int>(num_bins+1),
                                                       static_cast<Index_type>(0),
                                                       num_bins,
                                                       len,
                                                       stream));
#endif

      RAJAPERF_HIP_REDUCER_COPY_BACK(counts, hcounts, num_bins, 1);
      HISTOGRAM_GPU_FINALIZE_COUNTS(hcounts, num_bins, 1);

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::HipDevice, temp_storage);

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Hip variant id = " << vid << std::endl;
  }

  RAJAPERF_HIP_REDUCER_TEARDOWN(counts, hcounts);

}


template < Index_type block_size,
           Index_type preferred_global_replication,
           Index_type preferred_shared_replication >
void HISTOGRAM::runHipVariantAtomicRuntime(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  HISTOGRAM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    auto* func = &histogram_atomic_runtime<block_size>;

    hipFuncAttributes func_attr;
    hipErrchk(hipFuncGetAttributes(&func_attr, (const void*)func));
    const Index_type max_shmem_per_block_in_bytes = func_attr.maxDynamicSharedSizeBytes;
    const Index_type max_shared_replication = max_shmem_per_block_in_bytes / sizeof(Data_type) / num_bins;

    const Index_type grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

    const Index_type global_replication = RAJA::next_pow2(std::min(preferred_global_replication, grid_size));
    const Index_type shared_replication = RAJA::prev_pow2(std::min(preferred_shared_replication, max_shared_replication));

    const Index_type shmem = shared_replication * num_bins * sizeof(Data_type);

    RAJAPERF_HIP_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, global_replication);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_HIP_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, global_replication);

      RPlaunchHipKernel( func,
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         counts,
                         bins,
                         iend,
                         num_bins,
                         shared_replication,
                         global_replication );

      RAJAPERF_HIP_REDUCER_COPY_BACK(counts, hcounts, num_bins, global_replication);
      for (Index_type bin = 0; bin < num_bins; ++bin) {
        Data_type count_final = Data_type(0);
        for (Index_type r = 0; r < global_replication; ++r) {
          Index_type offset = bin + RAJA::power_of_2_mod(r, global_replication) * num_bins;
          count_final += hcounts[offset];
        }
        counts_final[bin] = count_final;
      }

    }
    stopTimer();

    RAJAPERF_HIP_REDUCER_TEARDOWN(counts, hcounts);

  } else if ( vid == RAJA_HIP ) {

    using multi_reduce_policy = RAJA::policy::hip::hip_multi_reduce_policy<
        RAJA::hip::MultiReduceTuning<
          RAJA::hip::multi_reduce_algorithm::init_host_combine_block_atomic_then_grid_atomic,
          RAJA::hip::AtomicReplicationTuning<
            RAJA::hip::SharedAtomicReplicationMaxPow2Concretizer<
              RAJA::hip::ConstantPreferredReplicationConcretizer<preferred_shared_replication>>,
            RAJA::hip::thread_xyz<>,
            RAJA::GetOffsetRight<int>>,
          RAJA::hip::AtomicReplicationTuning<
            RAJA::hip::GlobalAtomicReplicationMinPow2Concretizer<
              RAJA::hip::ConstantPreferredReplicationConcretizer<preferred_global_replication>>,
            RAJA::hip::warp_global_xyz<>,
            RAJA::GetOffsetLeft<int>>>>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      HISTOGRAM_INIT_COUNTS_RAJA(multi_reduce_policy);

      RAJA::forall<RAJA::hip_exec<block_size, true /*async*/>>( res,
          RAJA::RangeSegment(ibegin, iend),
          [=] __device__ (Index_type i) {
        HISTOGRAM_BODY;
      });

      HISTOGRAM_FINALIZE_COUNTS_RAJA(multi_reduce_policy);

    }
    stopTimer();

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Hip variant id = " << vid << std::endl;
  }

}


void HISTOGRAM::runHipVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_HIP ) {

    if (tune_idx == t) {

      runHipVariantLibrary(vid);

    }

    t += 1;

  }

  if ( vid == Base_HIP || vid == Lambda_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_global_replications_type{}, [&](auto global_replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(global_replication)) {

            seq_for(gpu_atomic_shared_replications_type{}, [&](auto shared_replication) {

              if (tune_idx == t) {

                setBlockSize(block_size);
                runHipVariantAtomicRuntime<decltype(block_size)::value,
                                           decltype(global_replication)::value,
                                           decltype(shared_replication)::value>(vid);

              }

              t += 1;

            });

          }

        });

      }

    });

  } else {

    getCout() << "\n  HISTOGRAM : Unknown Hip variant id = " << vid << std::endl;

  }

}

void HISTOGRAM::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP ) {

    addVariantTuningName(vid, "rocprim");

  }

  if ( vid == Base_HIP || vid == Lambda_HIP || vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_global_replications_type{}, [&](auto global_replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(global_replication)) {

            seq_for(gpu_atomic_shared_replications_type{}, [&](auto shared_replication) {

              addVariantTuningName(vid, "atomic_shared("+std::to_string(shared_replication)+
                                        ")_global("+std::to_string(global_replication)+
                                        ")_"+std::to_string(block_size));

            });

          }

        });

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
