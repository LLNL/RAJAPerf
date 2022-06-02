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
  allocAndInitHipDeviceData(points.x, m_x, points.N); \
  allocAndInitHipDeviceData(points.y, m_y, points.N); \
  
#define REDUCE_STRUCT_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(points.x); \
  deallocHipDeviceData(points.y);

#define REDUCE_STRUCT_BODY_HIP(atomicAdd, atomicMin, atomicMax) \
  \
  HIP_DYNAMIC_SHARED( Real_type, shared) \
  Real_type* pxsum = (Real_type*)&shared[ 0 * blockDim.x ]; \
  Real_type* pxmin = (Real_type*)&shared[ 1 * blockDim.x ]; \
  Real_type* pxmax = (Real_type*)&shared[ 2 * blockDim.x ]; \
  \
  Real_type* pysum = (Real_type*)&shared[ 3 * blockDim.x ]; \
  Real_type* pymin = (Real_type*)&shared[ 4 * blockDim.x ]; \
  Real_type* pymax = (Real_type*)&shared[ 5 * blockDim.x ]; \
  \
  Index_type i = blockIdx.x * blockDim.x + threadIdx.x; \
  \
  pxsum[ threadIdx.x ] = m_init_sum; \
  pxmin[ threadIdx.x ] = m_init_min; \
  pxmax[ threadIdx.x ] = m_init_max; \
  \
  pysum[ threadIdx.x ] = m_init_sum; \
  pymin[ threadIdx.x ] = m_init_min; \
  pymax[ threadIdx.x ] = m_init_max; \
  \
  for ( ; i < iend ; i += gridDim.x * blockDim.x ) { \
    pxsum[ threadIdx.x ] += x[ i ]; \
    pxmin[ threadIdx.x ] = RAJA_MIN( pxmin[ threadIdx.x ], x[ i ] ); \
    pxmax[ threadIdx.x ] = RAJA_MAX( pxmax[ threadIdx.x ], x[ i ] ); \
    \
    pysum[ threadIdx.x ] += y[ i ]; \
    pymin[ threadIdx.x ] = RAJA_MIN( pymin[ threadIdx.x ], y[ i ] ); \
    pymax[ threadIdx.x ] = RAJA_MAX( pymax[ threadIdx.x ], y[ i ] ); \
  } \
  __syncthreads(); \
  \
  for ( i = blockDim.x / 2; i > 0; i /= 2 ) { \
    if ( threadIdx.x < i ) { \
      pxsum[ threadIdx.x ] += pxsum[ threadIdx.x + i ]; \
      pxmin[ threadIdx.x ] = RAJA_MIN( pxmin[ threadIdx.x ], pxmin[ threadIdx.x + i ] ); \
      pxmax[ threadIdx.x ] = RAJA_MAX( pxmax[ threadIdx.x ], pxmax[ threadIdx.x + i ] ); \
      \
      pysum[ threadIdx.x ] += pysum[ threadIdx.x + i ]; \
      pymin[ threadIdx.x ] = RAJA_MIN( pymin[ threadIdx.x ], pymin[ threadIdx.x + i ] ); \
      pymax[ threadIdx.x ] = RAJA_MAX( pymax[ threadIdx.x ], pymax[ threadIdx.x + i ] ); \
    } \
     __syncthreads(); \
  } \
  \
  if ( threadIdx.x == 0 ) { \
    atomicAdd( xsum, pxsum[ 0 ] ); \
    atomicMin( xmin, pxmin[ 0 ] ); \
    atomicMax( xmax, pxmax[ 0 ] ); \
    \
    atomicAdd( ysum, pysum[ 0 ] ); \
    atomicMin( ymin, pymin[ 0 ] ); \
    atomicMax( ymax, pymax[ 0 ] ); \
  } \

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_struct(Real_ptr x, Real_ptr y,
                              Real_ptr xsum, Real_ptr xmin, Real_ptr xmax, 
                              Real_ptr ysum, Real_ptr ymin, Real_ptr ymax, 
                              Real_type m_init_sum,
                              Real_type m_init_min,
                              Real_type m_init_max,
                              Index_type iend)
{
  REDUCE_STRUCT_BODY_HIP(RAJA::atomicAdd<RAJA::hip_atomic>,
                         RAJA::atomicMin<RAJA::hip_atomic>,
                         RAJA::atomicMax<RAJA::hip_atomic>)
}

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void reduce_struct_unsafe(Real_ptr x, Real_ptr y,
                              Real_ptr xsum, Real_ptr xmin, Real_ptr xmax,
                              Real_ptr ysum, Real_ptr ymin, Real_ptr ymax,
                              Real_type m_init_sum,
                              Real_type m_init_min,
                              Real_type m_init_max,
                              Index_type iend)
{
  REDUCE_STRUCT_BODY_HIP(RAJAPERF_HIP_unsafeAtomicAdd,
                         ::atomicMin,
                         ::atomicMax)
}


template < size_t block_size >
void REDUCE_STRUCT::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_STRUCT_DATA_SETUP_HIP;

    Real_ptr mem; //xcenter,xmin,xmax,ycenter,ymin,ymax
    allocHipDeviceData(mem,6);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk(hipMemsetAsync(mem, 0.0, 6*sizeof(Real_type)));

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((reduce_struct<block_size>), 
                         dim3(grid_size), dim3(block_size), 
                         6*sizeof(Real_type)*block_size, 0,
	                 points.x, points.y,
                         mem, mem+1, mem+2,    // xcenter,xmin,xmax
                         mem+3, mem+4, mem+5,  // ycenter,ymin,ymax
                         m_init_sum, m_init_min, m_init_max,
                         points.N);
      hipErrchk( hipGetLastError() );

      Real_type lmem[6]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      Real_ptr plmem = &lmem[0];
      getHipDeviceData(plmem, mem, 6);

      points.SetCenter(lmem[0]/points.N, lmem[3]/points.N);
      points.SetXMin(lmem[1]);
      points.SetXMax(lmem[2]);
      points.SetYMin(lmem[4]);
      points.SetYMax(lmem[5]);
      m_points=points;

    }
    stopTimer();

    REDUCE_STRUCT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(mem);

  } else if ( vid == RAJA_HIP ) {

    REDUCE_STRUCT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> xsum(m_init_sum);
      RAJA::ReduceSum<RAJA::hip_reduce, Real_type> ysum(m_init_sum);
      RAJA::ReduceMin<RAJA::hip_reduce, Real_type> xmin(m_init_min);
      RAJA::ReduceMin<RAJA::hip_reduce, Real_type> ymin(m_init_min);
      RAJA::ReduceMax<RAJA::hip_reduce, Real_type> xmax(m_init_max);
      RAJA::ReduceMax<RAJA::hip_reduce, Real_type> ymax(m_init_max);

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
          REDUCE_STRUCT_BODY_RAJA;
      });

      points.SetCenter((xsum.get()/(points.N)),
                       (ysum.get()/(points.N)));
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

template < size_t block_size >
void REDUCE_STRUCT::runHipVariantUnsafe(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type iend = getActualProblemSize();

  REDUCE_STRUCT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    REDUCE_STRUCT_DATA_SETUP_HIP;

    Real_ptr mem; //xcenter,xmin,xmax,ycenter,ymin,ymax
    allocHipDeviceData(mem,6);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipErrchk(hipMemsetAsync(mem, 0.0, 6*sizeof(Real_type)));

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      hipLaunchKernelGGL((reduce_struct_unsafe<block_size>),
                         dim3(grid_size), dim3(block_size),
                         6*sizeof(Real_type)*block_size, 0,
                   points.x, points.y,
                         mem, mem+1, mem+2,    // xcenter,xmin,xmax
                         mem+3, mem+4, mem+5,  // ycenter,ymin,ymax
                         m_init_sum, m_init_min, m_init_max,
                         points.N);
      hipErrchk( hipGetLastError() );

      Real_type lmem[6]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      Real_ptr plmem = &lmem[0];
      getHipDeviceData(plmem, mem, 6);

      points.SetCenter(lmem[0]/points.N, lmem[3]/points.N);
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

void REDUCE_STRUCT::runHipVariant(VariantID vid, size_t tune_idx)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();
  size_t t = 0;
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      if (tune_idx == t) {
        runHipVariantImpl<block_size>(vid);
      }
      t += 1;
    }
  });
  if (vid == Base_HIP) {
    if (have_unsafe_atomics) {
      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {
          if (tune_idx == t) {
            runHipVariantUnsafe<block_size>(vid);
          }
          t += 1;
        }
      });
    }
  }
}

void REDUCE_STRUCT::setHipTuningDefinitions(VariantID vid)
{
  bool have_unsafe_atomics = haveHipUnsafeAtomics();
  seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
    if (run_params.numValidGPUBlockSize() == 0u ||
        run_params.validGPUBlockSize(block_size)) {
      addVariantTuningName(vid, "block_"+std::to_string(block_size));
    }
  });
  if (vid == Base_HIP) {
    if (have_unsafe_atomics) {
      seq_for(gpu_block_sizes_type{}, [&](auto block_size) {
        if (run_params.numValidGPUBlockSize() == 0u ||
            run_params.validGPUBlockSize(block_size)) {
          addVariantTuningName(vid, "unsafe_"+std::to_string(block_size));
        }
      });
    }
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
