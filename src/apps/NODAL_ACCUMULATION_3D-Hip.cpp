//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

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
__global__ void nodal_accumulation_3d(Real_ptr vol,
                      Real_ptr x0, Real_ptr x1,
                      Real_ptr x2, Real_ptr x3,
                      Real_ptr x4, Real_ptr x5,
                      Real_ptr x6, Real_ptr x7,
                      Index_ptr real_zones,
                      Index_type ibegin, Index_type iend)
{
   Index_type ii = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type i = ii + ibegin;
   if (i < iend) {
     NODAL_ACCUMULATION_3D_BODY_INDEX;
     NODAL_ACCUMULATION_3D_RAJA_ATOMIC_BODY(RAJA::hip_atomic);
   }
}


template < size_t block_size >
void NODAL_ACCUMULATION_3D::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  auto res{getHipResource()};

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);
      constexpr size_t shmem = 0;

      RPlaunchHipKernel( (nodal_accumulation_3d<block_size>),
                         grid_size, block_size,
                         shmem, res.get_stream(),
                         vol,
                         x0, x1, x2, x3, x4, x5, x6, x7,
                         real_zones,
                         ibegin, iend );
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_HIP ) {

    RAJA::TypedListSegment<Index_type> zones(real_zones, iend,
                                             res, RAJA::Unowned);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::hip_exec<block_size, true /*async*/> >( res,
        zones, [=] __device__ (Index_type i) {
          NODAL_ACCUMULATION_3D_RAJA_ATOMIC_BODY(RAJA::hip_atomic);
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  NODAL_ACCUMULATION_3D : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(NODAL_ACCUMULATION_3D, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
