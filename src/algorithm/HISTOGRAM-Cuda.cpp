//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HISTOGRAM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "cub/device/device_histogram.cuh"
#include "cub/util_allocator.cuh"

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

template < Index_type block_size, Index_type global_replication >
__launch_bounds__(block_size)
__global__ void histogram_atomic_global(HISTOGRAM::Data_ptr counts,
                                        Index_ptr bins,
                                        Index_type iend)
{
  Index_type i = blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, counts, HISTOGRAM_GPU_BIN_INDEX(bins[i], i, global_replication), HISTOGRAM::Data_type(1));
  }
}


void HISTOGRAM::runCudaVariantLibrary(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  HISTOGRAM_GPU_DATA_SETUP;

  RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, 1);

  RAJAPERF_UNUSED_VAR(counts_init);

  if ( vid == Base_CUDA ) {

    cudaStream_t stream = res.get_stream();

    int len = iend - ibegin;

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cudaErrchk(::cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                     temp_storage_bytes,
                                                     bins+ibegin,
                                                     counts,
                                                     static_cast<int>(num_bins+1),
                                                     static_cast<Index_type>(0),
                                                     num_bins,
                                                     len,
                                                     stream));

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocData(DataSpace::CudaDevice, temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
      cudaErrchk(::cub::DeviceHistogram::HistogramEven(d_temp_storage,
                                                       temp_storage_bytes,
                                                       bins+ibegin,
                                                       counts,
                                                       static_cast<int>(num_bins+1),
                                                       static_cast<Index_type>(0),
                                                       num_bins,
                                                       len,
                                                       stream));

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, 1);
      HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, 1);

    }
    stopTimer();

    // Free temporary storage
    deallocData(DataSpace::CudaDevice, temp_storage);

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Cuda variant id = " << vid << std::endl;
  }

  RAJAPERF_CUDA_REDUCER_TEARDOWN(counts, hcounts);

}

template < size_t block_size, size_t global_replication >
void HISTOGRAM::runCudaVariantAtomicGlobal(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  auto res{getCudaResource()};

  HISTOGRAM_GPU_DATA_SETUP;

  RAJAPERF_CUDA_REDUCER_SETUP(Data_ptr, counts, hcounts, num_bins, global_replication);

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, global_replication);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (histogram_atomic_global<block_size, global_replication>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          counts,
                          bins,
                          iend );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, global_replication);
      HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, global_replication);

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, global_replication);

      auto histogram_lambda = [=] __device__ (Index_type i) {
        HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, counts, HISTOGRAM_GPU_BIN_INDEX(bins[i], i, global_replication), HISTOGRAM::Data_type(1));
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel( (lambda_cuda_forall<block_size,
                                              decltype(histogram_lambda)>),
                          grid_size, block_size,
                          shmem, res.get_stream(),
                          ibegin, iend, histogram_lambda );

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, global_replication);
      HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, global_replication);

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJAPERF_CUDA_REDUCER_INITIALIZE(counts_init, counts, hcounts, num_bins, global_replication);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          HISTOGRAM_GPU_RAJA_BODY(RAJA::cuda_atomic, counts, HISTOGRAM_GPU_BIN_INDEX(bins[i], i, global_replication), HISTOGRAM::Data_type(1));
      });

      RAJAPERF_CUDA_REDUCER_COPY_BACK(counts, hcounts, num_bins, global_replication);
      HISTOGRAM_GPU_FINALIZE_VALUES(hcounts, num_bins, global_replication);

    }
    stopTimer();

  } else {
     getCout() << "\n  HISTOGRAM : Unknown Cuda variant id = " << vid << std::endl;
  }

  RAJAPERF_CUDA_REDUCER_TEARDOWN(counts, hcounts);

}

void HISTOGRAM::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;

  if ( vid == Base_CUDA ) {

    if (tune_idx == t) {

      runCudaVariantLibrary(vid);

    }

    t += 1;

  }

  if ( vid == Base_CUDA || vid == Lambda_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_global_replications_type{}, [&](auto global_replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(global_replication)) {

            if (tune_idx == t) {

              setBlockSize(block_size);
              runCudaVariantAtomicGlobal<decltype(block_size)::value, global_replication>(vid);

            }

            t += 1;

          }

        });

      }

    });

  } else {

    getCout() << "\n  HISTOGRAM : Unknown Cuda variant id = " << vid << std::endl;

  }

}

void HISTOGRAM::setCudaTuningDefinitions(VariantID vid)
{
  if ( vid == Base_CUDA ) {

    addVariantTuningName(vid, "cub");

  }

  if ( vid == Base_CUDA || vid == Lambda_CUDA || vid == RAJA_CUDA ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        seq_for(gpu_atomic_global_replications_type{}, [&](auto global_replication) {

          if (run_params.numValidAtomicReplication() == 0u ||
              run_params.validAtomicReplication(global_replication)) {

            addVariantTuningName(vid, "atomic_global<"+std::to_string(global_replication)+
                                      ">_"+std::to_string(block_size));


          }

        });

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
