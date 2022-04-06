//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_SUM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#if defined(__HIPCC__)
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_reduce.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_reduce.cuh"
#include "cub/util_allocator.cuh"
#endif

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace algorithm
{

#define REDUCE_SUM_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend);

#define REDUCE_SUM_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(x);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_sum(Real_ptr x, Real_ptr dsum, Real_type sum_init,
                           Index_type iend)
{
  HIP_DYNAMIC_SHARED(Real_type, psum);

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

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    RAJA::atomicAdd<RAJA::hip_atomic>( dsum, psum[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *dsum += psum[ 0 ];
  }
#endif
}


void REDUCE_SUM::runHipVariantRocprim(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    hipStream_t stream = 0;

    int len = iend - ibegin;

    Real_type* sum_storage;
    allocHipPinnedData(sum_storage, 1);

    // Determine temporary device storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
#if defined(__HIPCC__)
    hipErrchk(::rocprim::reduce(d_temp_storage,
                                temp_storage_bytes,
                                x+ibegin,
                                sum_storage,
                                m_sum_init,
                                len,
                                rocprim::plus<Real_type>(),
                                stream));
#elif defined(__CUDACC__)
    hipErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                          temp_storage_bytes,
                                          x+ibegin,
                                          sum_storage,
                                          len,
                                          ::cub::Sum(),
                                          m_sum_init,
                                          stream));
#endif

    // Allocate temporary storage
    unsigned char* temp_storage;
    allocHipDeviceData(temp_storage, temp_storage_bytes);
    d_temp_storage = temp_storage;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Run
#if defined(__HIPCC__)
      hipErrchk(::rocprim::reduce(d_temp_storage,
                                  temp_storage_bytes,
                                  x+ibegin,
                                  sum_storage,
                                  m_sum_init,
                                  len,
                                  rocprim::plus<Real_type>(),
                                  stream));
#elif defined(__CUDACC__)
      hipErrchk(::cub::DeviceReduce::Reduce(d_temp_storage,
                                            temp_storage_bytes,
                                            x+ibegin,
                                            sum_storage,
                                            len,
                                            ::cub::Sum(),
                                            m_sum_init,
                                            stream));
#endif

      hipErrchk(hipStreamSynchronize(stream));
      m_sum = *sum_storage;

    }
    stopTimer();

    // Free temporary storage
    deallocHipDeviceData(temp_storage);
    deallocHipPinnedData(sum_storage);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void REDUCE_SUM::runHipVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_SUM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    Real_ptr dsum;
    allocHipDeviceData(dsum, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(dsum, &m_sum_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (reduce_sum<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          x, dsum, m_sum_init, iend );
      hipErrchk( hipGetLastError() );

      Real_type lsum;
      Real_ptr plsum = &lsum;
      getHipDeviceData(plsum, dsum, 1);

      m_sum = lsum;

    }
    stopTimer();

    deallocHipDeviceData(dsum);

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    REDUCE_SUM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> sum(m_sum_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_SUM_BODY;
      });

      m_sum = sum.get();

    }
    stopTimer();

    REDUCE_SUM_DATA_TEARDOWN_HIP;

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

void REDUCE_SUM::runHipVariant(VariantID vid, size_t tune_idx)
{
  if ( vid == Base_HIP ) {

    size_t t = 0;

    if (tune_idx == t) {

      runHipVariantRocprim(vid);

    }

    t += 1;

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runHipVariantBlock<block_size>(vid);

        }

        t += 1;

      }

    });

  } else if ( vid == RAJA_HIP ) {

    size_t t = 0;

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runHipVariantBlock<block_size>(vid);

        }

        t += 1;

      }

    });

  } else {

    getCout() << "\n  REDUCE_SUM : Unknown Hip variant id = " << vid << std::endl;

  }

}

void REDUCE_SUM::setHipTuningDefinitions(VariantID vid)
{
  if ( vid == Base_HIP ) {

#if defined(__HIPCC__)
    addVariantTuningName(vid, "rocprim");
#elif defined(__CUDACC__)
    addVariantTuningName(vid, "cub");
#endif

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuningName(vid, "block_"+std::to_string(block_size));

      }

    });

  } else if ( vid == RAJA_HIP ) {

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        addVariantTuningName(vid, "block_"+std::to_string(block_size));

      }

    });

  }

}

} // end namespace algorithm
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
