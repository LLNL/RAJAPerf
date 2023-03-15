//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "EDGE3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

#define EDGE3D_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(x, m_x, m_array_length); \
  allocAndInitCudaDeviceData(y, m_y, m_array_length); \
  allocAndInitCudaDeviceData(z, m_z, m_array_length); \
  allocAndInitCudaDeviceData(sum, m_sum, m_array_length);

#define EDGE3D_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_sum, sum, m_array_length); \
  deallocCudaDeviceData(x); \
  deallocCudaDeviceData(y); \
  deallocCudaDeviceData(z); \
  deallocCudaDeviceData(sum);

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

void EDGE3D::runCudaVariant(VariantID vid, size_t tune_idx)
{
  if ( vid == Base_CUDA ) {

    size_t t = 0;

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runCudaVariantBlock<block_size>(vid);

        }

        t += 1;

      }

    });

  } else if ( vid == RAJA_CUDA ) {

    size_t t = 0;

    seq_for(gpu_block_sizes_type{}, [&](auto block_size) {

      if (run_params.numValidGPUBlockSize() == 0u ||
          run_params.validGPUBlockSize(block_size)) {

        if (tune_idx == t) {

          runCudaVariantBlock<block_size>(vid);

        }

        t += 1;

      }

    });

  } else {

    getCout() << "\n  EDGE3D : Unknown Cuda variant id = " << vid << std::endl;

  }

}

template < size_t block_size >
void EDGE3D::runCudaVariantBlock(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = m_domain->fpz;
  const Index_type iend = m_domain->lpz+1;

  EDGE3D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    EDGE3D_DATA_SETUP_CUDA;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      const size_t grid_size = RAJA_DIVIDE_CEILING_INT(iend, block_size);

      edge3d<block_size><<<grid_size, block_size>>>(sum,
                                       x0, x1, x2, x3, x4, x5, x6, x7,
                                       y0, y1, y2, y3, y4, y5, y6, y7,
                                       z0, z1, z2, z3, z4, z5, z6, z7,
                                       ibegin, iend);
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    EDGE3D_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    EDGE3D_DATA_SETUP_CUDA;

    NDPTRSET(m_domain->jp, m_domain->kp, x,x0,x1,x2,x3,x4,x5,x6,x7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, y,y0,y1,y2,y3,y4,y5,y6,y7) ;
    NDPTRSET(m_domain->jp, m_domain->kp, z,z0,z1,z2,z3,z4,z5,z6,z7) ;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::forall< RAJA::cuda_exec<block_size, true /*async*/> >(
        RAJA::RangeSegment(ibegin, iend), [=] __device__ (Index_type i) {
        EDGE3D_BODY;
      });

    }
    stopTimer();

    EDGE3D_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n  EDGE3D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(EDGE3D, Cuda)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
