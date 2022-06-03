//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>


namespace rajaperf
{
namespace stream
{

#define DOT_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(a, m_a, iend); \
  allocAndInitCudaDeviceData(b, m_b, iend);

#define DOT_DATA_TEARDOWN_CUDA \
  deallocCudaDeviceData(a); \
  deallocCudaDeviceData(b);

#define DOT_BODY_CUDA(atomicAdd) \
  RAJAPERF_REDUCE_1_CUDA(Real_type, DOT_VAL, dprod, dot_init, RAJAPERF_ADD_OP, atomicAdd)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void dot(Real_ptr a, Real_ptr b,
                    Real_ptr dprod, Real_type dot_init,
                    Index_type iend)
{
  DOT_BODY_CUDA(::atomicAdd)
}


template < size_t block_size >
void DOT::runCudaVariantReduceAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    DOT_DATA_SETUP_CUDA;

    Real_ptr dprod;
    allocAndInitCudaDeviceData(dprod, &dot_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(dprod, &dot_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      dot<block_size><<<grid_size, block_size, sizeof(Real_type)*block_size>>>(
          a, b, dprod, dot_init, iend );
      cudaErrchk( cudaGetLastError() );

      Real_type lprod;
      Real_ptr plprod = &lprod;
      getCudaDeviceData(plprod, dprod, 1);
      m_dot += lprod;

    }
    stopTimer();

    DOT_DATA_TEARDOWN_CUDA;

    deallocCudaDeviceData(dprod);

  } else if ( vid == Lambda_CUDA ) {

    DOT_DATA_SETUP_CUDA;

    Real_ptr dprod;
    allocAndInitCudaDeviceData(dprod, &dot_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(dprod, &dot_init, 1);

      auto dot_lam = [=] __device__ () {
        DOT_BODY_CUDA(::atomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda<block_size><<<grid_size, block_size, sizeof(Real_type)*block_size>>>(
          dot_lam );
      cudaErrchk( cudaGetLastError() );

      Real_type lprod;
      Real_ptr plprod = &lprod;
      getCudaDeviceData(plprod, dprod, 1);
      m_dot += lprod;

    }
    stopTimer();

    DOT_DATA_TEARDOWN_CUDA;

    deallocCudaDeviceData(dprod);

  } else if ( vid == RAJA_CUDA ) {

    DOT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::ReduceSum<RAJA::cuda_reduce_atomic, Real_type> dot(dot_init);

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         DOT_BODY;
       });

       m_dot += static_cast<Real_type>(dot.get());

    }
    stopTimer();

    DOT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  DOT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void DOT::runCudaVariantReduce(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  if ( vid == RAJA_CUDA ) {

    DOT_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> dot(dot_init);

       RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         DOT_BODY;
       });

       m_dot += static_cast<Real_type>(dot.get());

    }
    stopTimer();

    DOT_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  DOT : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void DOT::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tune_idx == t) {
        runCudaVariantReduceAtomic<block_size>(vid);
      }
      t += 1;
    }
  });
  if ( vid == RAJA_CUDA ) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (tune_idx == t) {
          runCudaVariantReduce<block_size>(vid);
        }
        t += 1;
      }
    });
  }
}

void DOT::setCudaTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      addVariantTuningName(vid, "reduceAtomic_"+std::to_string(block_size));
    }
  });
  if ( vid == RAJA_CUDA ) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        addVariantTuningName(vid, "reduce_"+std::to_string(block_size));
      }
    });
  }
}

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
