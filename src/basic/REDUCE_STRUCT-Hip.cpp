//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "REDUCE_STRUCT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


#define REDUCE_STRUCT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(x, m_x, iend); \
  allocAndInitHipDeviceData(y, m_y, iend); \
  points.x = x; \
  points.y = y;
  
#define REDUCE_STRUCT_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(x); \
  deallocHipDeviceData(y);

#define REDUCE_STRUCT_BODY_HIP(atomicAdd, atomicMin, atomicMax) \
  RAJAPERF_REDUCE_6_HIP(Real_type, REDUCE_STRUCT_VALS, xsum, init_sum, RAJAPERF_ADD_OP, atomicAdd, \
                                                       xmin, init_min, RAJAPERF_MIN_OP, atomicMin, \
                                                       xmax, init_max, RAJAPERF_MAX_OP, atomicMax, \
                                                       ysum, init_sum, RAJAPERF_ADD_OP, atomicAdd, \
                                                       ymin, init_min, RAJAPERF_MIN_OP, atomicMin, \
                                                       ymax, init_max, RAJAPERF_MAX_OP, atomicMax)

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_struct(Real_ptr x, Real_ptr y,
                              Real_ptr xsum, Real_ptr xmin, Real_ptr xmax, 
                              Real_ptr ysum, Real_ptr ymin, Real_ptr ymax, 
                              Real_type init_sum,
                              Real_type init_min,
                              Real_type init_max,
                              Index_type iend)
{
  REDUCE_STRUCT_BODY_HIP(::atomicAdd,
                         ::atomicMin,
                         ::atomicMax)
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_struct_unsafe(Real_ptr x, Real_ptr y,
                              Real_ptr xsum, Real_ptr xmin, Real_ptr xmax,
                              Real_ptr ysum, Real_ptr ymin, Real_ptr ymax,
                              Real_type init_sum,
                              Real_type init_min,
                              Real_type init_max,
                              Index_type iend)
{
  REDUCE_STRUCT_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd,
                         ::atomicMin,
                         ::atomicMax)
}


template < size_t block_size >
void REDUCE_STRUCT::runHipVariantReduceAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_STRUCT_DATA_SETUP_HIP;

    Real_ptr mem;
    allocHipDeviceData(mem,6);
    Real_ptr xsum = mem + 0;
    Real_ptr xmin = mem + 1;
    Real_ptr xmax = mem + 2;
    Real_ptr ysum = mem + 3;
    Real_ptr ymin = mem + 4;
    Real_ptr ymax = mem + 5;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk(hipMemsetAsync(mem, 0.0, 6*sizeof(Real_type)));

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((reduce_struct<block_size>), 
                         dim3(grid_size), dim3(block_size), 
                         6*sizeof(Real_type)*block_size, 0,
	                       x, y,
                         xsum, xmin, xmax,
                         ysum, ymin, ymax,
                         init_sum, init_min, init_max,
                         iend);
      hipErrchk( hipGetLastError() );

      Real_type lmem[6]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      Real_ptr plmem = &lmem[0];
      getHipDeviceData(plmem, mem, 6);

      points.SetCenter(lmem[0]/iend, lmem[3]/iend);
      points.SetXMin(lmem[1]);
      points.SetXMax(lmem[2]);
      points.SetYMin(lmem[4]);
      points.SetYMax(lmem[5]);
      m_points=points;

    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(mem);

  } else if ( vid == Lambda_HIP ) {

    REDUCE_STRUCT_DATA_SETUP_HIP;

    Real_ptr mem;
    allocHipDeviceData(mem,6);
    Real_ptr xsum = mem + 0;
    Real_ptr xmin = mem + 1;
    Real_ptr xmax = mem + 2;
    Real_ptr ysum = mem + 3;
    Real_ptr ymin = mem + 4;
    Real_ptr ymax = mem + 5;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk(hipMemsetAsync(mem, 0.0, 6*sizeof(Real_type)));

      auto reduce_struct_lambda = [=] __device__ () {
          REDUCE_STRUCT_BODY_HIP(::atomicAdd,
                                 ::atomicMin,
                                 ::atomicMax)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((lambda_hip<block_size, decltype(reduce_struct_lambda)>),
                         dim3(grid_size), dim3(block_size),
                         6*sizeof(Real_type)*block_size, 0,
                         reduce_struct_lambda);
      hipErrchk( hipGetLastError() );

      Real_type lmem[6]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      Real_ptr plmem = &lmem[0];
      getHipDeviceData(plmem, mem, 6);

      points.SetCenter(lmem[0]/iend, lmem[3]/iend);
      points.SetXMin(lmem[1]);
      points.SetXMax(lmem[2]);
      points.SetYMin(lmem[4]);
      points.SetYMax(lmem[5]);
      m_points=points;

    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(mem);

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown Hip variant id = " << vid << std::endl;
  }

}

template < size_t block_size >
void REDUCE_STRUCT::runHipVariantReduceUnsafeAtomic(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_STRUCT_DATA_SETUP_HIP;

    Real_ptr mem;
    allocHipDeviceData(mem,6);
    Real_ptr xsum = mem + 0;
    Real_ptr xmin = mem + 1;
    Real_ptr xmax = mem + 2;
    Real_ptr ysum = mem + 3;
    Real_ptr ymin = mem + 4;
    Real_ptr ymax = mem + 5;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk(hipMemsetAsync(mem, 0.0, 6*sizeof(Real_type)));

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((reduce_struct_unsafe<block_size>),
                         dim3(grid_size), dim3(block_size),
                         6*sizeof(Real_type)*block_size, 0,
                         x, y,
                         xsum, xmin, xmax,
                         ysum, ymin, ymax,
                         init_sum, init_min, init_max,
                         iend);
      hipErrchk( hipGetLastError() );

      Real_type lmem[6]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      Real_ptr plmem = &lmem[0];
      getHipDeviceData(plmem, mem, 6);

      points.SetCenter(lmem[0]/iend, lmem[3]/iend);
      points.SetXMin(lmem[1]);
      points.SetXMax(lmem[2]);
      points.SetYMin(lmem[4]);
      points.SetYMax(lmem[5]);
      m_points=points;

    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(mem);

  } else if ( vid == Lambda_HIP ) {

    REDUCE_STRUCT_DATA_SETUP_HIP;

    Real_ptr mem;
    allocHipDeviceData(mem,6);
    Real_ptr xsum = mem + 0;
    Real_ptr xmin = mem + 1;
    Real_ptr xmax = mem + 2;
    Real_ptr ysum = mem + 3;
    Real_ptr ymin = mem + 4;
    Real_ptr ymax = mem + 5;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk(hipMemsetAsync(mem, 0.0, 6*sizeof(Real_type)));

      auto reduce_struct_lambda = [=] __device__ () {
          REDUCE_STRUCT_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd,
                                 ::atomicMin,
                                 ::atomicMax)
      };

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((lambda_hip<block_size, decltype(reduce_struct_lambda)>),
                         dim3(grid_size), dim3(block_size),
                         6*sizeof(Real_type)*block_size, 0,
                         reduce_struct_lambda);
      hipErrchk( hipGetLastError() );

      Real_type lmem[6]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      Real_ptr plmem = &lmem[0];
      getHipDeviceData(plmem, mem, 6);

      points.SetCenter(lmem[0]/iend, lmem[3]/iend);
      points.SetXMin(lmem[1]);
      points.SetXMax(lmem[2]);
      points.SetYMin(lmem[4]);
      points.SetYMax(lmem[5]);
      m_points=points;

    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(mem);

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown Hip variant id = " << vid << std::endl;
  }

}

template < size_t block_size >
void REDUCE_STRUCT::runHipVariantReduce(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == RAJA_HIP ) {

    REDUCE_STRUCT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> xsum(init_sum);
      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> ysum(init_sum);
      RAJA::ReduceMin<RAJA::hip_reduce, Real_type> xmin(init_min);
      RAJA::ReduceMin<RAJA::hip_reduce, Real_type> ymin(init_min);
      RAJA::ReduceMax<RAJA::hip_reduce, Real_type> xmax(init_max);
      RAJA::ReduceMax<RAJA::hip_reduce, Real_type> ymax(init_max);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_STRUCT_BODY_RAJA;
      });

      points.SetCenter((xsum.get()/(iend)),
                       (ysum.get()/(iend)));
      points.SetXMin((xmin.get()));
      points.SetXMax((xmax.get()));
      points.SetYMin((ymin.get()));
      points.SetYMax((ymax.get()));
      m_points=points;

    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  REDUCE_STRUCT : Unknown Hip variant id = " << vid << std::endl;
  }

}

void REDUCE_STRUCT::runHipVariant(VariantID vid, size_t tune_idx)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();
  size_t t = 0;
  if (vid == Base_HIP || vid == Lambda_HIP) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (tune_idx == t) {
          runHipVariantReduceAtomic<block_size>(vid);
        }
        t += 1;
      }
    });
    if (have_unsafe_atomics) {
      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {
          if (tune_idx == t) {
            runHipVariantReduceUnsafeAtomic<block_size>(vid);
          }
          t += 1;
        }
      });
    }
  } else if ( vid == RAJA_HIP ) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        if (tune_idx == t) {
          runHipVariantReduce<block_size>(vid);
        }
        t += 1;
      }
    });
  }
}

void REDUCE_STRUCT::setHipTuningDefinitions(VariantID vid)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();
  if (vid == Base_HIP || vid == Lambda_HIP) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        addVariantTuningName(vid, "reduceAtomic_"+std::to_string(block_size));
      }
    });
    if (have_unsafe_atomics) {
      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {
          addVariantTuningName(vid, "reduceUnsafeAtomic_"+std::to_string(block_size));
        }
      });
    }
  } else if ( vid == RAJA_HIP ) {
    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {
        addVariantTuningName(vid, "reduce_"+std::to_string(block_size));
      }
    });
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
