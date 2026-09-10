//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_HEAT_3D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

  //
  // Define thread block shape for CUDA execution
  //
#define k_block_sz (32)
#define j_block_sz (block_size / k_block_sz)
#define i_block_sz (1)

#define HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA \
  k_block_sz, j_block_sz, i_block_sz

#define HEAT_3D_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA);

#define HEAT_3D_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, k_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N-2, i_block_sz)));


template < size_t k_block_size, size_t j_block_size, size_t i_block_size >
__launch_bounds__(k_block_size*j_block_size*i_block_size)
__global__ void poly_heat_3D_1(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = 1 + blockIdx.z;
   Index_type j = 1 + blockIdx.y * j_block_size + threadIdx.y;
   Index_type k = 1 + blockIdx.x * k_block_size + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     POLYBENCH_HEAT_3D_BODY1;
   }
}

template < size_t k_block_size, size_t j_block_size, size_t i_block_size >
__launch_bounds__(k_block_size*j_block_size*i_block_size)
__global__ void poly_heat_3D_2(Real_ptr A, Real_ptr B, Index_type N)
{
   Index_type i = 1 + blockIdx.z;
   Index_type j = 1 + blockIdx.y * j_block_size + threadIdx.y;
   Index_type k = 1 + blockIdx.x * k_block_size + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     POLYBENCH_HEAT_3D_BODY2;
   }
}

template< size_t k_block_size, size_t j_block_size, size_t i_block_size, typename Lambda >
__launch_bounds__(k_block_size*j_block_size*i_block_size)
__global__ void poly_heat_3D_lam(Index_type N, Lambda body)
{
   Index_type i = 1 + blockIdx.z;
   Index_type j = 1 + blockIdx.y * j_block_size + threadIdx.y;
   Index_type k = 1 + blockIdx.x * k_block_size + threadIdx.x;

   if (i < N-1 && j < N-1 && k < N-1) {
     body(i, j, k);
   }
}


template < size_t block_size >
void POLYBENCH_HEAT_3D::runCudaVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getCudaResource()};

  POLYBENCH_HEAT_3D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      HEAT_3D_THREADS_PER_BLOCK_CUDA;
      HEAT_3D_NBLOCKS_CUDA;
      constexpr size_t shmem = 0;

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_1");
      RPlaunchCudaKernel(
        (poly_heat_3D_1<HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        A, B, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_2");
      RPlaunchCudaKernel(
        (poly_heat_3D_2<HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        A, B, N );
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_2");

    }
    stopTimer();

  } else if ( vid == Lambda_CUDA ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      HEAT_3D_THREADS_PER_BLOCK_CUDA;
      HEAT_3D_NBLOCKS_CUDA;
      constexpr size_t shmem = 0;

      auto poly_heat_3D_1_lambda = [=] __device__ (Index_type i,
                                                   Index_type j,
                                                   Index_type k) {
        POLYBENCH_HEAT_3D_BODY1;
      };

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_1");
      RPlaunchCudaKernel(
        (poly_heat_3D_lam<HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA,
                          decltype(poly_heat_3D_1_lambda)>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        N, poly_heat_3D_1_lambda );
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_1");

      auto poly_heat_3D_2_lambda = [=] __device__ (Index_type i,
                                                   Index_type j,
                                                   Index_type k) {
        POLYBENCH_HEAT_3D_BODY2;
      };

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_2");
      RPlaunchCudaKernel(
        (poly_heat_3D_lam<HEAT_3D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA,
                          decltype(poly_heat_3D_2_lambda)>),
        nblocks, nthreads_per_block,
        shmem, res.get_stream(),
        N, poly_heat_3D_2_lambda );
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_2");

    }
    stopTimer();

  } else if (vid == RAJA_CUDA) {

    POLYBENCH_HEAT_3D_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<j_block_sz * k_block_sz,
          RAJA::statement::For<0, RAJA::cuda_block_z_direct,      // i
            RAJA::statement::For<1, RAJA::cuda_global_size_y_direct<j_block_sz>,   // j
              RAJA::statement::For<2, RAJA::cuda_global_size_x_direct<k_block_sz>, // k
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;


    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_1");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1}),
        res,
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_HEAT_3D_BODY1_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_1");

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_HEAT_3D_2");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1},
                         RAJA::RangeSegment{1, N-1}),
        res,
        [=] __device__ (Index_type i, Index_type j, Index_type k) {
          POLYBENCH_HEAT_3D_BODY2_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_HEAT_3D_2");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_HEAT_3D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_HEAT_3D, Cuda, Base_CUDA, Lambda_CUDA, RAJA_CUDA)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
