//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "NODAL_ACCUMULATION_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

  //
  // Define thread block size for CUDA execution
  //
  const size_t block_size = 256;


#define NODAL_ACCUMULATION_3D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, m_nodal_array_length); \
  allocAndInitCudaDeviceData(vol, m_vol, m_zonal_array_length); \
  allocAndInitCudaDeviceData(real_zones, m_domain->real_zones, iend);

#define NODAL_ACCUMULATION_3D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_x, x, m_nodal_array_length); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(vol); \
  deallocCudaDeviceData(real_zones);

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
     NODAL_ACCUMULATION_3D_RAJA_ATOMIC_BODY(RAJA::cuda_atomic);
   }
}


void NODAL_ACCUMULATION_3D::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = m_domain->n_real_zones;

  NODAL_ACCUMULATION_3D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_CUDA;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      nodal_accumulation_3d<<<grid_size, block_size>>>(vol,
                                       x0, x1, x2, x3, x4, x5, x6, x7,
                                       real_zones,
                                       ibegin, iend);
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    NODAL_ACCUMULATION_3D_DATA_SETUP_CUDA;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;

    camp::resources::Resource working_res{camp::resources::Cuda()};
    RAJA::TypedListSegment<Index_type> zones(m_domain->real_zones,
                                             m_domain->n_real_zones,
                                             working_res);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        zones, [=] __device__ (Index_type i) {
          NODAL_ACCUMULATION_3D_RAJA_ATOMIC_BODY(RAJA::cuda_atomic);
      });

    }
    stopTimer();

    NODAL_ACCUMULATION_3D_DATA_TEARDOWN_CUDA;

  } else {
     std::cout << "\n  NODAL_ACCUMULATION_3D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
