//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

//
// Define thread block shape for CUDA execution
//
#define m_block_sz (32)
#define g_block_sz (integer::greater_of_squarest_factor_pair(block_size/m_block_sz))
#define z_block_sz (integer::lesser_of_squarest_factor_pair(block_size/m_block_sz))

#define LTIMES_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA \
  m_block_sz, g_block_sz, z_block_sz

#define LTIMES_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(LTIMES_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA); \
  static_assert(m_block_sz*g_block_sz*z_block_sz == block_size, "Invalid block_size");

#define LTIMES_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_m, m_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_g, g_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(num_z, z_block_sz)));


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
void LTIMES::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  LTIMES_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      LTIMES_THREADS_PER_BLOCK_CUDA;
      LTIMES_NBLOCKS_CUDA;
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel(
        (ltimes<LTIMES_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        phidat, elldat, psidat,
        num_d, num_m, num_g, num_z );

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto ltimes_lambda = [=] __device__ (Index_type z, Index_type g, 
                                           Index_type m) {
        for (Index_type d = 0; d < num_d; ++d ) {
          LTIMES_BODY;
        }
      };

      LTIMES_THREADS_PER_BLOCK_CUDA;
      LTIMES_NBLOCKS_CUDA;
      constexpr size_t shmem = 0;

      RPlaunchCudaKernel(
        (ltimes_lam<LTIMES_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA,
                    decltype(ltimes_lambda)>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        num_m, num_g, num_z,
        ltimes_lambda );

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    LTIMES_VIEWS_RANGES_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<m_block_sz*g_block_sz*z_block_sz,
          RAJA::statement::For<1, RAJA::cuda_global_size_z_direct<z_block_sz>,     //z
            RAJA::statement::For<2, RAJA::cuda_global_size_y_direct<g_block_sz>,   //g
              RAJA::statement::For<3, RAJA::cuda_global_size_x_direct<m_block_sz>, //m
                RAJA::statement::For<0, RAJA::seq_exec,           //d
                  RAJA::statement::Lambda<0>
                >
              >
            >
          >
        >
      >;

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::kernel_resource<EXEC_POL>(
          RAJA::make_tuple(IDRange(0, num_d),
                           IZRange(0, num_z),
                           IGRange(0, num_g),
                           IMRange(0, num_m)),
          res,
          [=] __device__ (ID d, IZ z, IG g, IM m) {
            LTIMES_BODY_RAJA;
          }
        );

      }
      stopTimer();

  } else {
     getCout() << "\n LTIMES : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(LTIMES, Cuda)

} // end namespace apps
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
