//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EDGE3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

template < size_t block_size >
__launch_bounds__(block_size)
__global__ void edge3d(Real_ptr sum,
                       const Real_ptr x0, const Real_ptr x1,
                       const Real_ptr x2, const Real_ptr x3,
                       const Real_ptr x4, const Real_ptr x5,
                       const Real_ptr x6, const Real_ptr x7,
                       const Real_ptr y0, const Real_ptr y1,
                       const Real_ptr y2, const Real_ptr y3,
                       const Real_ptr y4, const Real_ptr y5,
                       const Real_ptr y6, const Real_ptr y7,
                       const Real_ptr z0, const Real_ptr z1,
                       const Real_ptr z2, const Real_ptr z3,
                       const Real_ptr z4, const Real_ptr z5,
                       const Real_ptr z6, const Real_ptr z7,
                       Index_type ibegin, Index_type iend)
{
  Index_type i = ibegin + blockIdx.x * block_size + threadIdx.x;
  if (i < iend) {
    EDGE3D_BODY;
  }
}


template < size_t block_size >
void EDGE3D::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  auto res{getHipResource()};

  EDGE3D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      hipLaunchKernelGGL((edge3d<block_size>), dim3(grid_size), dim3(block_size), shmem, res.get_stream(), sum,
                                       x0, x1, x2, x3, x4, x5, x6, x7,
                                       y0, y1, y2, y3, y4, y5, y6, y7,
                                       z0, z1, z2, z3, z4, z5, z6, z7,
                                       ibegin, iend);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      auto edge3d_lam = [=] __device__ (Index_type i) { EDGE3D_BODY; };

      hipLaunchKernelGGL((lambda_hip_forall<block_size, decltype(edge3d_lam)>),
        grid_size, block_size, shmem, res.get_stream(),
        ibegin, iend,  edge3d_lam);

      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        EDGE3D_BODY;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  EDGE3D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(EDGE3D, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
