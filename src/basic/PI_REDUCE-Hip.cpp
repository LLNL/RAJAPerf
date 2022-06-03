//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_REDUCE.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define PI_REDUCE_BODY_HIP(atomicAdd) \
  RAJAPERF_REDUCE_1_HIP(Real_type, PI_REDUCE_VAL, dpi, pi_init, RAJAPERF_ADD_OP, atomicAdd)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void pi_reduce(Real_type dx,
                          Real_ptr dpi, Real_type pi_init,
                          Index_type iend)
{
  PI_REDUCE_BODY_HIP(::atomicAdd)
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void pi_reduce_unsafe(Real_type dx,
                          Real_ptr dpi, Real_type pi_init,
                          Index_type iend)
{
  PI_REDUCE_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd)
}


template < size_t block_size >
void PI_REDUCE::runHipVariantAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_HIP ) {

    Real_ptr dpi;
    allocAndInitHipDeviceData(dpi, &pi_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(dpi, &pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (pi_reduce<block_size>),
          grid_size, block_size, sizeof(Real_type)*block_size, 0,
          dx, dpi, pi_init, iend );
      hipErrchk( hipGetLastError() );

      Real_type lpi;
      Real_ptr plpi = &lpi;
      getHipDeviceData(plpi, dpi, 1);

      m_pi = 4.0 * lpi;

    }
    stopTimer();

    deallocHipDeviceData(dpi);

  } else if ( vid == Lambda_HIP ) {

    Real_ptr dpi;
    allocAndInitHipDeviceData(dpi, &pi_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(dpi, &pi_init, 1);

      auto reduce_pi_lambda = [=] __device__ () {
          PI_REDUCE_BODY_HIP(::atomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( ((lambda_hip<block_size, decltype(reduce_pi_lambda)>)),
          grid_size, block_size, sizeof(Real_type)*block_size, 0,
          reduce_pi_lambda );
      hipErrchk( hipGetLastError() );

      Real_type lpi;
      Real_ptr plpi = &lpi;
      getHipDeviceData(plpi, dpi, 1);

      m_pi = 4.0 * lpi;

    }
    stopTimer();

    deallocHipDeviceData(dpi);

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> pi(pi_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         PI_REDUCE_BODY;
       });

      m_pi = 4.0 * static_cast<Real_type>(pi.get());

    }
    stopTimer();

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void PI_REDUCE::runHipVariantUnsafeAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  PI_REDUCE_DATA_SETUP;

  if ( vid == Base_HIP ) {

    Real_ptr dpi;
    allocAndInitHipDeviceData(dpi, &pi_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(dpi, &pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( (pi_reduce_unsafe<block_size>), dim3(grid_size), dim3(block_size),
                          sizeof(Real_type)*block_size, 0,
                          dx, dpi, pi_init, iend );
      hipErrchk( hipGetLastError() );

      Real_type lpi;
      Real_ptr plpi = &lpi;
      getHipDeviceData(plpi, dpi, 1);

      m_pi = 4.0 * lpi;

    }
    stopTimer();

    deallocHipDeviceData(dpi);

  } else if ( vid == Lambda_HIP ) {

    Real_ptr dpi;
    allocAndInitHipDeviceData(dpi, &pi_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(dpi, &pi_init, 1);

      auto reduce_pi_lambda = [=] __device__ () {
          PI_REDUCE_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL( ((lambda_hip<block_size, decltype(reduce_pi_lambda)>)),
          grid_size, block_size, sizeof(Real_type)*block_size, 0,
          reduce_pi_lambda );
      hipErrchk( hipGetLastError() );

      Real_type lpi;
      Real_ptr plpi = &lpi;
      getHipDeviceData(plpi, dpi, 1);

      m_pi = 4.0 * lpi;

    }
    stopTimer();

    deallocHipDeviceData(dpi);

  } else {
     getCout() << "\n  PI_REDUCE : Unknown Hip variant id = " << vid << std::endl;
  }
}

void PI_REDUCE::runHipVariant(VariantID vid, size_t tune_idx)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tune_idx == t) {
        runHipVariantAtomic<block_size>(vid);
      }
      t += 1;
    }
  });
  if (vid == Base_HIP || vid == Lambda_HIP) {
    if (have_unsafe_atomics) {
      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {
          if (tune_idx == t) {
            runHipVariantUnsafeAtomic<block_size>(vid);
          }
          t += 1;
        }
      });
    }
  }
}

void PI_REDUCE::setHipTuningDefinitions(VariantID vid)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      addVariantTuningName(vid, "atomic_"+std::to_string(block_size));
    }
  });
  if (vid == Base_HIP || vid == Lambda_HIP) {
    if (have_unsafe_atomics) {
      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {
          addVariantTuningName(vid, "unsafeAtomic_"+std::to_string(block_size));
        }
      });
    }
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
