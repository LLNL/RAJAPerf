//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRAP_INT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{

//
// Function used in TRAP_INT loop.
//
RAJA_INLINE
RAJA_DEVICE
Real_type trap_int_func(Real_type x,
                        Real_type y,
                        Real_type xp,
                        Real_type yp)
{
   Real_type denom = (x - xp)*(x - xp) + (y - yp)*(y - yp);
   denom = 1.0/sqrt(denom);
   return denom;
}


#define TRAP_INT_DATA_SETUP_HIP // nothing to do here...

#define TRAP_INT_DATA_TEARDOWN_HIP // nothing to do here...

#define TRAP_INT_BODY_HIP(atomicAdd) \
  RAJAPERF_REDUCE_1_HIP(Real_type, TRAP_INT_VAL, sumx, 0.0, RAJAPERF_ADD_OP, atomicAdd)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void trapint(Real_type x0, Real_type xp,
                        Real_type y, Real_type yp,
                        Real_type h,
                        Real_ptr sumx,
                        Index_type iend)
{
  TRAP_INT_BODY_HIP(::atomicAdd)
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void trapint_unsafe(Real_type x0, Real_type xp,
                        Real_type y, Real_type yp,
                        Real_type h,
                        Real_ptr sumx,
                        Index_type iend)
{
  TRAP_INT_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd)
}


template < size_t block_size >
void TRAP_INT::runHipVariantAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  TRAP_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRAP_INT_DATA_SETUP_HIP;

    Real_ptr sumx;
    allocAndInitHipDeviceData(sumx, &sumx_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(sumx, &sumx_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((trapint<block_size>), dim3(grid_size), dim3(block_size), sizeof(Real_type)*block_size, 0, x0, xp,
                                                y, yp,
                                                h,
                                                sumx,
                                                iend);
      hipErrchk( hipGetLastError() );

      Real_type lsumx;
      Real_ptr plsumx = &lsumx;
      getHipDeviceData(plsumx, sumx, 1);
      m_sumx += lsumx * h;

    }
    stopTimer();

    deallocHipDeviceData(sumx);

    TRAP_INT_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    TRAP_INT_DATA_SETUP_HIP;

    Real_ptr sumx;
    allocAndInitHipDeviceData(sumx, &sumx_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(sumx, &sumx_init, 1);

      auto trapint_lam = [=] __device__ () {
        TRAP_INT_BODY_HIP(::atomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip<block_size, decltype(trapint_lam)>),
          dim3(grid_size), dim3(block_size), sizeof(Real_type)*block_size, 0,
          trapint_lam);
      hipErrchk( hipGetLastError() );

      Real_type lsumx;
      Real_ptr plsumx = &lsumx;
      getHipDeviceData(plsumx, sumx, 1);
      m_sumx += lsumx * h;

    }
    stopTimer();

    deallocHipDeviceData(sumx);

    TRAP_INT_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    TRAP_INT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> sumx(sumx_init);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        TRAP_INT_BODY;
      });

      m_sumx += static_cast<Real_type>(sumx.get()) * h;

    }
    stopTimer();

    TRAP_INT_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  TRAP_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

template < size_t block_size >
void TRAP_INT::runHipVariantUnsafeAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  TRAP_INT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRAP_INT_DATA_SETUP_HIP;

    Real_ptr sumx;
    allocAndInitHipDeviceData(sumx, &sumx_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(sumx, &sumx_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((trapint_unsafe<block_size>), dim3(grid_size), dim3(block_size), sizeof(Real_type)*block_size, 0, x0, xp,
                                                y, yp,
                                                h,
                                                sumx,
                                                iend);
      hipErrchk( hipGetLastError() );

      Real_type lsumx;
      Real_ptr plsumx = &lsumx;
      getHipDeviceData(plsumx, sumx, 1);
      m_sumx += lsumx * h;

    }
    stopTimer();

    deallocHipDeviceData(sumx);

    TRAP_INT_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    TRAP_INT_DATA_SETUP_HIP;

    Real_ptr sumx;
    allocAndInitHipDeviceData(sumx, &sumx_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(sumx, &sumx_init, 1);

      auto trapint_lam = [=] __device__ () {
        TRAP_INT_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((lambda_hip<block_size, decltype(trapint_lam)>),
          dim3(grid_size), dim3(block_size), sizeof(Real_type)*block_size, 0,
          trapint_lam);
      hipErrchk( hipGetLastError() );

      Real_type lsumx;
      Real_ptr plsumx = &lsumx;
      getHipDeviceData(plsumx, sumx, 1);
      m_sumx += lsumx * h;

    }
    stopTimer();

    deallocHipDeviceData(sumx);

    TRAP_INT_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  TRAP_INT : Unknown Hip variant id = " << vid << std::endl;
  }
}

void TRAP_INT::runHipVariant(VariantID vid, size_t tune_idx)
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

void TRAP_INT::setHipTuningDefinitions(VariantID vid)
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
