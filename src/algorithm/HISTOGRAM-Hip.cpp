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

template < size_t block_size, size_t replication >
__launch_bounds__(block_size)
__global__ void histogram(HISTOGRAM::Data_ptr counts,
                          Index_ptr bins,
                          Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    HISTOGRAM_GPU_RAJA_BODY(RAJA::hip_atomic);
  }
}


void HISTOGRAM::runHipVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  HISTOGRAM_GPU_DATA_SETUP;

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
      HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, 1);

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::HipDevice, temp_storage);

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Hip variant id = " << vid << std::endl;
  }

  RAJAPERF_HIP_REDUCER_TEARDOWN(counts, hcounts);

}

template < size_t block_size, size_t replication >
void HISTOGRAM::runHipVariantAtomicGlobal(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getHipResource()};

  HISTOGRAM_GPU_DATA_SETUP;

  RAJAPERF_HIP_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, replication);

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_HIP_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, replication);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (histogram<block_size, replication>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         counts,
                         bins,
                         iend );

      RAJAPERF_HIP_REDUCER_COPY_BACK(counts, hcounts, num_bins, replication);
      HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, replication);

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_HIP_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, replication);

      auto histogram_lambda = [=] __device__ (Index_type i) {
        HISTOGRAM_GPU_RAJA_BODY(RAJA::hip_atomic);
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (lambda_hip_forall<block_size,
                                            decltype(histogram_lambda)>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         ibegin, iend, histogram_lambda );

      RAJAPERF_HIP_REDUCER_COPY_BACK(counts, hcounts, num_bins, replication);
      HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, replication);

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_HIP_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, replication);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          HISTOGRAM_GPU_RAJA_BODY(RAJA::hip_atomic);
      });

      RAJAPERF_HIP_REDUCER_COPY_BACK(counts, hcounts, num_bins, replication);
      HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, replication);

    }
    stopTimer();

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Hip variant id = " << vid << std::endl;
  }

  RAJAPERF_HIP_REDUCER_TEARDOWN(counts, hcounts);

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

            if (tune_idx == t) {

              setBlockSize(block_size);
              runHipVariantAtomicGlobal<decltype(block_size)::value, global_replication>(vid);

            }

            t += 1;

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

            addVariantTuningName(vid, "replicate_"+std::to_string(global_replication)+
                                      "_global_"+std::to_string(block_size));

          }

        });

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
