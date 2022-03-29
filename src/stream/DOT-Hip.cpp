//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "DOT.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>


namespace rajaperf
{
namespace stream
{

#define DOT_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(a, m_a, iend); \
  allocAndInitHipDeviceData(b, m_b, iend);

#define DOT_DATA_TEARDOWN_HIP \
  deallocHipDeviceData(a); \
  deallocHipDeviceData(b);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void dot(Real_ptr a, Real_ptr b,
                    Real_ptr dprod, Real_type dprod_init,
                    Index_type iend)
{
  HIP_DYNAMIC_SHARED( Real_type, pdot)

  Index_type i = blockIdx.x * block_size + threadIdx.x;

  pdot[ threadIdx.x ] = dprod_init;
  for ( ; i < iend ; i += gridDim.x * block_size ) {
    pdot[ threadIdx.x ] += a[ i ] * b[i];
  }
  __syncthreads();

  for ( i = block_size / 2; i > 0; i /= 2 ) {
    if ( threadIdx.x < i ) {
      pdot[ threadIdx.x ] += pdot[ threadIdx.x + i ];
    }
     __syncthreads();
  }

#if 1 // serialized access to shared data;
  if ( threadIdx.x == 0 ) {
    //atomicAdd(dprod, pdot[ 0 ] );
    RAJA::atomicAdd(RAJA::hip_atomic{}, dprod, pdot[ 0 ] );
  }
#else // this doesn't work due to data races
  if ( threadIdx.x == 0 ) {
    *dprod += pdot[ 0 ];
  }
#endif

}


template < size_t block_size >
void DOT::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  DOT_DATA_SETUP;

  if ( vid == Base_HIP ) {

    DOT_DATA_SETUP_HIP;

    Real_ptr dprod;
    allocAndInitHipDeviceData(dprod, &m_dot_init, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      initHipDeviceData(dprod, &m_dot_init, 1);

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      hipLaunchKernelGGL((dot<block_size>), dim3(grid_size), dim3(block_size),
                                            sizeof(Real_type)*block_size, 0,
                         a, b, dprod, m_dot_init, iend );
      hipErrchk( hipGetLastError() );

      Real_type lprod;
      Real_ptr plprod = &lprod;
      getHipDeviceData(plprod, dprod, 1);
      m_dot += lprod;

    }
    stopTimer();

    DOT_DATA_TEARDOWN_HIP;

    deallocHipDeviceData(dprod);

  } else if ( vid == RAJA_HIP ) {

    DOT_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::ReduceSum<RAJA::hip_reduce, Real_type> dot(m_dot_init);

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         DOT_BODY;
       });

       m_dot += static_cast<Real_type>(dot.get());

    }
    stopTimer();

    DOT_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  DOT : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(DOT, Hip)

} // end namespace stream
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
