//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "TRIDIAG_ELIM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

#define TRIDIAG_ELIM_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(xout, m_xout, m_N); \
  allocAndInitHipDeviceData(xin, m_xin, m_N); \
  allocAndInitHipDeviceData(y, m_y, m_N); \
  allocAndInitHipDeviceData(z, m_z, m_N);

#define TRIDIAG_ELIM_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_xout, xout, m_N); \
  deallocHipDeviceData(xout); \
  deallocHipDeviceData(xin); \
  deallocHipDeviceData(y); \
  deallocHipDeviceData(z);

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void eos(Real_ptr xout, Real_ptr xin, Real_ptr y, Real_ptr z,
                    Index_type N)
{
   Index_type i = blockIdx.x * block_size + threadIdx.x;
   if (i > 0 && i < N) {
     TRIDIAG_ELIM_BODY;
   }
}


template < size_t block_size >
void TRIDIAG_ELIM::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = m_N;

  TRIDIAG_ELIM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    TRIDIAG_ELIM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
       hipLaunchKernelGGL((eos<block_size>), grid_size, block_size, 0, 0, xout, xin, y, z,
                                       iend );
       hipErrchk( hipGetLastError() );

    }
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    TRIDIAG_ELIM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

       RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >(
         RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
         TRIDIAG_ELIM_BODY;
       });

    }
    stopTimer();

    TRIDIAG_ELIM_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n  TRIDIAG_ELIM : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(TRIDIAG_ELIM, Hip)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
