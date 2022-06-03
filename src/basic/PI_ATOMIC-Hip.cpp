//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PI_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

#define PI_ATOMIC_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(pi, m_pi, 1);

#define PI_ATOMIC_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(pi);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void atomic_pi(Real_ptr pi,
                          Real_type dx,
                          Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     PI_ATOMIC_BODY_ATOMIC(::atomicAdd)
   }
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void atomic_pi_unsafe(Real_ptr pi,
                          Real_type dx,
                          Index_type iend)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i < iend) {
     PI_ATOMIC_BODY_ATOMIC(RAJAPERF_HIP_unsafeAtomicAdd)
   }
}



template < size_t block_size >
void PI_ATOMIC::runHipVariantAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_ATOMIC_DATA_SETUP;

  if ( vid == Base_HIP ) {

    PI_ATOMIC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(pi, &m_pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((atomic_pi<block_size>),grid_size, block_size, 0, 0, pi, dx, iend );
      hipErrchk( hipGetLastError() );

      getHipDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    PI_ATOMIC_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    PI_ATOMIC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(pi, &m_pi_init, 1);

      auto atomic_pi_lambda = [=] __device__ (Index_type i) {
          PI_ATOMIC_BODY_ATOMIC(::atomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(atomic_pi_lambda)>),
          grid_size, block_size, 0, 0, ibegin, iend, atomic_pi_lambda);
      hipErrchk( hipGetLastError() );

      getHipDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    PI_ATOMIC_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    PI_ATOMIC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(pi, &m_pi_init, 1);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          PI_ATOMIC_BODY_ATOMIC(RAJA::atomicAdd<RAJA::hip_atomic>)
      });

      getHipDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    PI_ATOMIC_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  PI_ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void PI_ATOMIC::runHipVariantUnsafeAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PI_ATOMIC_DATA_SETUP;

  if ( vid == Base_HIP ) {

    PI_ATOMIC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(pi, &m_pi_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((atomic_pi_unsafe<block_size>),grid_size, block_size, 0, 0, pi, dx, iend );
      hipErrchk( hipGetLastError() );

      getHipDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    PI_ATOMIC_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    PI_ATOMIC_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(pi, &m_pi_init, 1);

      auto atomic_pi_lambda = [=] __device__ (Index_type i) {
          PI_ATOMIC_BODY_ATOMIC(RAJAPERF_HIP_unsafeAtomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(atomic_pi_lambda)>),
          grid_size, block_size, 0, 0, ibegin, iend, atomic_pi_lambda);
      hipErrchk( hipGetLastError() );

      getHipDeviceData(m_pi, pi, 1);
      *m_pi *= 4.0;

    }
    stopTimer();

    PI_ATOMIC_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  PI_ATOMIC : Unknown Hip variant id = " << vid << std::endl;
  }
}

void PI_ATOMIC::runHipVariant(VariantID vid, size_t tune_idx)
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

void PI_ATOMIC::setHipTuningDefinitions(VariantID vid)
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
