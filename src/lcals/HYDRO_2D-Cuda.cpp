//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_2D.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{

  //
  // Define thread block shape for CUDA execution
  //
#define j_block_sz (32)
#define k_block_sz (block_size / j_block_sz)

#define HYDRO_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA \
  j_block_sz, k_block_sz

#define HYDRO_2D_THREADS_PER_BLOCK_CUDA \
  dim3 nthreads_per_block(HYDRO_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA, 1); \
  static_assert(j_block_sz*k_block_sz == block_size, "Invalid block_size");

#define HYDRO_2D_NBLOCKS_CUDA \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(jn-2, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(kn-2, k_block_sz)), \
               static_cast<size_t>(1));


template < size_t j_block_size, size_t k_block_size >
__launch_bounds__(j_block_size*k_block_size)
__global__ void hydro_2d1(Real_ptr zadat, Real_ptr zbdat,
                          Real_ptr zpdat, Real_ptr zqdat,
                          Real_ptr zrdat, Real_ptr zmdat,
                          Index_type jn, Index_type kn)
{
   Index_type k = 1 + blockIdx.y * k_block_size + threadIdx.y;
   Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

   if (k < kn-1 && j < jn-1) {
     HYDRO_2D_BODY1;
   }
}

template < size_t j_block_size, size_t k_block_size >
__launch_bounds__(j_block_size*k_block_size)
__global__ void hydro_2d2(Real_ptr zudat, Real_ptr zvdat,
                          Real_ptr zadat, Real_ptr zbdat,
                          Real_ptr zzdat, Real_ptr zrdat,
                          Real_type s,
                          Index_type jn, Index_type kn)
{
   Index_type k = 1 + blockIdx.y * k_block_size + threadIdx.y;
   Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

   if (k < kn-1 && j < jn-1) {
     HYDRO_2D_BODY2;
   }
}

template < size_t j_block_size, size_t k_block_size >
__launch_bounds__(j_block_size*k_block_size)
__global__ void hydro_2d3(Real_ptr zroutdat, Real_ptr zzoutdat,
                          Real_ptr zrdat, Real_ptr zudat,
                          Real_ptr zzdat, Real_ptr zvdat,
                          Real_type t,
                          Index_type jn, Index_type kn)
{
   Index_type k = 1 + blockIdx.y * k_block_size + threadIdx.y;
   Index_type j = 1 + blockIdx.x * j_block_size + threadIdx.x;

   if (k < kn-1 && j < jn-1) {
     HYDRO_2D_BODY3;
   }
}


template < size_t block_size >
void HYDRO_2D::runCudaVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();
  const Index_type kbeg = 1;
  const Index_type kend = m_kn - 1;
  const Index_type jbeg = 1;
  const Index_type jend = m_jn - 1;

  auto res{getCudaResource()};

  HYDRO_2D_DATA_SETUP;

  if ( vid == Base_CUDA ) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      constexpr size_t shmem = 0;

      HYDRO_2D_THREADS_PER_BLOCK_CUDA;
      HYDRO_2D_NBLOCKS_CUDA;

      hydro_2d1<HYDRO_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
               <<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(zadat, zbdat,
                                                 zpdat, zqdat, zrdat, zmdat,
                                                 jn, kn);
      cudaErrchk( cudaGetLastError() );

      hydro_2d2<HYDRO_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
               <<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(zudat, zvdat,
                                                 zadat, zbdat, zzdat, zrdat,
                                                 s,
                                                 jn, kn);
      cudaErrchk( cudaGetLastError() );

      hydro_2d3<HYDRO_2D_THREADS_PER_BLOCK_TEMPLATE_PARAMS_CUDA>
               <<<nblocks, nthreads_per_block, shmem, res.get_stream()>>>(zroutdat, zzoutdat,
                                                 zrdat, zudat, zzdat, zvdat,
                                                 t,
                                                 jn, kn);
      cudaErrchk( cudaGetLastError() );

    }
    stopTimer();

  } else if ( vid == RAJA_CUDA ) {

    HYDRO_2D_VIEWS_RAJA;

    using EXECPOL =
      RAJA::KernelPolicy<
        RAJA::statement::CudaKernelFixedAsync<j_block_sz * k_block_sz,
          RAJA::statement::For<0, RAJA::cuda_global_size_y_direct<k_block_sz>,   // k
            RAJA::statement::For<1, RAJA::cuda_global_size_x_direct<j_block_sz>, // j
              RAJA::statement::Lambda<0>
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res,
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY1_RAJA;
      });

      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res,
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY2_RAJA;
      });

      RAJA::kernel_resource<EXECPOL>(
        RAJA::make_tuple( RAJA::RangeSegment(kbeg, kend),
                          RAJA::RangeSegment(jbeg, jend)),
        res,
        [=] __device__ (Index_type k, Index_type j) {
        HYDRO_2D_BODY3_RAJA;
      });

    }
    stopTimer();

  } else {
     getCout() << "\n  HYDRO_2D : Unknown Cuda variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(HYDRO_2D, Cuda)

} // end namespace lcals
} // end namespace rajaperf

#endif  // RAJA_ENABLE_CUDA
