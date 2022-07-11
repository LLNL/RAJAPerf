//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

//
// Define thread block shape for Hip execution
//
#define m_block_sz (32)
#define g_block_sz (gpu_block_size::greater_of_squarest_factor_pair(block_size/m_block_sz))
#define z_block_sz (gpu_block_size::lesser_of_squarest_factor_pair(block_size/m_block_sz))

#define LTIMES_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  m_block_sz, g_block_sz, z_block_sz

#define LTIMES_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(LTIMES_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP);

#define LTIMES_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_m, m_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_g, g_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_z, z_block_sz)));


#define LTIMES_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(phidat, m_phidat, m_philen); \
  allocAndInitHipDeviceData(elldat, m_elldat, m_elllen); \
  allocAndInitHipDeviceData(psidat, m_psidat, m_psilen);

#define LTIMES_DATA_TEARDOWN_HIP \
  getHipDeviceData(m_phidat, phidat, m_philen); \
  deallocHipDeviceData(phidat); \
  deallocHipDeviceData(elldat); \
  deallocHipDeviceData(psidat);

template < size_t m_block_size, size_t g_block_size, size_t z_block_size >
__launch_bounds__(m_block_size*g_block_size*z_block_size)
__global__ void ltimes(Real_ptr phidat, Real_ptr elldat, Real_ptr psidat,
                       Index_type num_d,
                       Index_type num_m, Index_type num_g, Index_type num_z)
{
   Index_type m = blockIdx.x * m_block_size + threadIdx.x;
   Index_type g = blockIdx.y * g_block_size + threadIdx.y;
   Index_type z = blockIdx.z * z_block_size + threadIdx.z;

   if (m < num_m && g < num_g && z < num_z) {
     for (Index_type d = 0; d < num_d; ++d ) {
       LTIMES_BODY;
     }
   }
}

template < size_t m_block_size, size_t g_block_size, size_t z_block_size, typename Lambda >
__launch_bounds__(m_block_size*g_block_size*z_block_size)
__global__ void ltimes_lam(Index_type num_m, Index_type num_g, Index_type num_z,
                           Lambda body)
{
   Index_type m = blockIdx.x * m_block_size + threadIdx.x;
   Index_type g = blockIdx.y * g_block_size + threadIdx.y;
   Index_type z = blockIdx.z * z_block_size + threadIdx.z;

   if (m < num_m && g < num_g && z < num_z) {
     body(z, g, m);
   }
}


template < size_t block_size >
void LTIMES::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  if ( vid == Base_HIP ) {

    LTIMES_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      LTIMES_THREADS_PER_BLOCK_HIP;
      LTIMES_NBLOCKS_HIP;

      hipLaunchKernelGGL((ltimes<LTIMES_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         phidat, elldat, psidat,
                         num_d,
                         num_m, num_g, num_z);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_HIP;

  } else if ( vid == Lambda_HIP ) {

    LTIMES_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      LTIMES_THREADS_PER_BLOCK_HIP;
      LTIMES_NBLOCKS_HIP;

      auto ltimes_lambda =
        [=] __device__ (Index_type z, Index_type g, Index_type m) {
          for (Index_type d = 0; d < num_d; ++d ) {
            LTIMES_BODY;
          }
        };

      hipLaunchKernelGGL((ltimes_lam<LTIMES_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(ltimes_lambda)>),
                         dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         num_m, num_g, num_z, ltimes_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_HIP;

  } else if ( vid == RAJA_HIP ) {

    LTIMES_DATA_SETUP_HIP;

    LTIMES_VIEWS_RANGES_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<m_block_sz*g_block_sz*z_block_sz,
          RAJA::statement::Tile<1, RAJA::tile_fixed<z_block_sz>,
                                   RAJA::hip_block_z_direct,
            RAJA::statement::Tile<2, RAJA::tile_fixed<g_block_sz>,
                                     RAJA::hip_block_y_direct,
              RAJA::statement::Tile<3, RAJA::tile_fixed<m_block_sz>,
                                       RAJA::hip_block_x_direct,
                RAJA::statement::For<1, RAJA::hip_thread_z_direct,     //z
                  RAJA::statement::For<2, RAJA::hip_thread_y_direct,   //g
                    RAJA::statement::For<3, RAJA::hip_thread_x_direct, //m
                      RAJA::statement::For<0, RAJA::seq_exec,          //d
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

      RAJA::kernel<EXEC_POL>( RAJA::make_tuple(IDRange(0, num_d),
                                               IZRange(0, num_z),
                                               IGRange(0, num_g),
                                               IMRange(0, num_m)),
        [=] __device__ (ID d, IZ z, IG g, IM m) {
        LTIMES_BODY_RAJA;
      });

    }
    stopTimer();

    LTIMES_DATA_TEARDOWN_HIP;

  } else {
     getCout() << "\n LTIMES : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(LTIMES, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP
