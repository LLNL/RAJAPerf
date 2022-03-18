//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES_NOVIEW.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

//
// Define thread block size for CUDA execution
//
constexpr size_t z_block_sz = 2;
constexpr size_t g_block_sz = 4;
constexpr size_t m_block_sz = 32;

#define LTIMES_NOVIEW_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(m_block_sz, g_block_sz, z_block_sz);

#define LTIMES_NOVIEW_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_m, m_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_g, g_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_z, z_block_sz)));



#define LTIMES_NOVIEW_DATA_SETUP_CUDA \
  allocAndInitCudaDeviceData(phidat, m_phidat, m_philen); \
  allocAndInitCudaDeviceData(elldat, m_elldat, m_elllen); \
  allocAndInitCudaDeviceData(psidat, m_psidat, m_psilen);

#define LTIMES_NOVIEW_DATA_TEARDOWN_CUDA \
  getCudaDeviceData(m_phidat, phidat, m_philen); \
  deallocCudaDeviceData(phidat); \
  deallocCudaDeviceData(elldat); \
  deallocCudaDeviceData(psidat);

__global__ void ltimes_noview(Real_ptr phidat, Real_ptr elldat, Real_ptr psidat,
                              Index_type num_d,
                              Index_type num_m, Index_type num_g, Index_type num_z)
{
   Index_type m = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type g = blockIdx.y * blockDim.y + threadIdx.y;
   Index_type z = blockIdx.z * blockDim.z + threadIdx.z;

   if (m < num_m && g < num_g && z < num_z) {
     for (Index_type d = 0; d < num_d; ++d ) {
       LTIMES_NOVIEW_BODY;
     }
   }
}

template< typename Lambda >
__global__ void ltimes_noview_lam(Index_type num_m, Index_type num_g, Index_type num_z,
                                  Lambda body)
{
   Index_type m = blockIdx.x * blockDim.x + threadIdx.x;
   Index_type g = blockIdx.y * blockDim.y + threadIdx.y;
   Index_type z = blockIdx.z * blockDim.z + threadIdx.z;

   if (m < num_m && g < num_g && z < num_z) {
     body(z, g, m);
   }
}


void LTIMES_NOVIEW::runCudaVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_NOVIEW_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    LTIMES_NOVIEW_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      LTIMES_NOVIEW_THREADS_PER_BLOCK_CUDA;
      LTIMES_NOVIEW_NBLOCKS_CUDA;

      ltimes_noview<<<nblocks, nthreads_per_block>>>(phidat, elldat, psidat,
                                                     num_d,
                                                     num_m, num_g, num_z);
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    LTIMES_NOVIEW_DATA_TEARDOWN_CUDA;

  } else if ( vid == Lambda_CUDA ) {

    LTIMES_NOVIEW_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      LTIMES_NOVIEW_THREADS_PER_BLOCK_CUDA;
      LTIMES_NOVIEW_NBLOCKS_CUDA;

      ltimes_noview_lam<<<nblocks, nthreads_per_block>>>(num_m, num_g, num_z,
        [=] __device__ (Index_type z, Index_type g, Index_type m) {
          for (Index_type d = 0; d < num_d; ++d ) {
            LTIMES_NOVIEW_BODY;
          }
        }
      );
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

    LTIMES_NOVIEW_DATA_TEARDOWN_CUDA;

  } else if ( vid == RAJA_CUDA ) {

    LTIMES_NOVIEW_DATA_SETUP_CUDA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<m_block_sz*g_block_sz*z_block_sz,
          RAJA::statement::Tile<1, RAJA::tile_fixed<z_block_sz>,
                                   RAJA::cuda_block_z_direct,
            RAJA::statement::Tile<2, RAJA::tile_fixed<g_block_sz>,
                                     RAJA::cuda_block_y_direct,
              RAJA::statement::Tile<3, RAJA::tile_fixed<m_block_sz>,
                                       RAJA::cuda_block_x_direct,
                RAJA::statement::For<1, RAJA::cuda_thread_z_direct,     //z
                  RAJA::statement::For<2, RAJA::cuda_thread_y_direct,   //g
                    RAJA::statement::For<3, RAJA::cuda_thread_x_direct, //m
                      RAJA::statement::For<0, RAJA::seq_exec,           //d
                        RAJA::statement::Lambda<0>
                      >
                    >
                  >
                >
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(RAJA::RangeSegment(0, num_d),
                                               RAJA::RangeSegment(0, num_z),
                                               RAJA::RangeSegment(0, num_g),
                                               RAJA::RangeSegment(0, num_m)),
        [=] __device__ (Index_type d, Index_type z, Index_type g, Index_type m) {
        LTIMES_NOVIEW_BODY;
      });

    }
    stopTimer();

    LTIMES_NOVIEW_DATA_TEARDOWN_CUDA;

  } else {
     getCout() << "\n LTIMES_NOVIEW : Unknown Cuda variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
