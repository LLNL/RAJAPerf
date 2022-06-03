//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define PI_REDUCE_BODY_CUDA(atomicAdd) \
  RAJAPERF_REDUCE_1_CUDA(Real_type, PI_REDUCE_VAL, dpi, pi_init, RAJAPERF_ADD_OP, atomicAdd)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void pi_reduce(Real_type dx,
                          Real_ptr dpi, Real_type pi_init,
                          Index_type iend)
{
  PI_REDUCE_BODY_CUDA(::atomicAdd)
}


template < size_t block_size >
void PI_REDUCE::runCudaVariantAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    Real_ptr dpi;
    allocAndInitCudaDeviceData(dpi, &pi_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(dpi, &pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      pi_reduce<block_size><<<grid_size, block_size,
                  sizeof(Real_type)*block_size>>>( dx,
                                                   dpi, pi_init,
                                                   iend );
      cudaErrchk( cudaGetLastError() );

      Real_type lpi;
      Real_ptr plpi = &lpi;
      getCudaDeviceData(plpi, dpi, 1);

      m_pi = 4.0 * lpi;

    }
    stopTimer();

    deallocCudaDeviceData(dpi);

  } else if ( vid == Lambda_CUDA ) {

    Real_ptr dpi;
    allocAndInitCudaDeviceData(dpi, &pi_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initCudaDeviceData(dpi, &pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      lambda_cuda<block_size><<<grid_size, block_size,
                                sizeof(Real_type)*block_size>>>(
        [=] __device__ () {
          PI_REDUCE_BODY_CUDA(::atomicAdd)
      });
      cudaErrchk( cudaGetLastError() );

      Real_type lpi;
      Real_ptr plpi = &lpi;
      getCudaDeviceData(plpi, dpi, 1);

      m_pi = 4.0 * lpi;

    }
    stopTimer();

    deallocCudaDeviceData(dpi);

  } else if ( vid == RAJA_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::cuda_reduce, Real_type> pi(pi_init);

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PI_REDUCE_BODY;
       });

      m_pi = 4.0 * static_cast<Real_type>(pi.get());

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Cuda variant id = " << vid << std::endl;
  }
}

void PI_REDUCE::runCudaVariant(VariantID vid, size_t tune_idx)
{
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tune_idx == t) {
        runCudaVariantAtomic<block_size>(vid);
      }
      t += 1;
    }
  });
}

void PI_REDUCE::setCudaTuningDefinitions(VariantID vid)
{
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      addVariantTuningName(vid, "atomic_"+std::to_string(block_size));
    }
  });
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
