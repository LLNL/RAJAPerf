//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_FLOYD_WARSHALL.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

namespace rajaperf
{
namespace polybench
{

//
// Define thread block shape for Hip execution
//
#define j_block_sz (32)
#define i_block_sz (block_size / j_block_sz)

#define POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  j_block_sz, i_block_sz

#define POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, 1);

#define POLY_FLOYD_WARSHALL_NBLOCKS_HIP \
  dim3 nblocks(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, j_block_sz)), \
               static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(N, i_block_sz)), \
               static_cast<size_t>(1));


template < size_t j_block_size, size_t i_block_size >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_floyd_warshall(Real_ptr pout, Real_ptr pin,
                                    Index_type k,
                                    Index_type N)
{
  Index_type i = blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = blockIdx.x * j_block_size + threadIdx.x;

  if ( i < N && j < N ) {
    POLYBENCH_FLOYD_WARSHALL_BODY;
  }
}

template < size_t j_block_size, size_t i_block_size, typename Lambda >
__launch_bounds__(j_block_size*i_block_size)
__global__ void poly_floyd_warshall_lam(Index_type N,
                                        Lambda body)
{
  Index_type i = blockIdx.y * i_block_size + threadIdx.y;
  Index_type j = blockIdx.x * j_block_size + threadIdx.x;

  if ( i < N && j < N ) {
    body(i, j);
  }
}


template < size_t block_size >
void POLYBENCH_FLOYD_WARSHALL::runHipVariantImpl(VariantID vid)
{
  setBlockSize(block_size);

  const Index_type run_reps = getRunReps();

  auto res{getHipResource()};

  POLYBENCH_FLOYD_WARSHALL_DATA_SETUP;

  if ( vid == Base_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      for (Index_type k = 0; k < N; ++k) {

        POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_HIP;
        POLY_FLOYD_WARSHALL_NBLOCKS_HIP;
        constexpr size_t shmem = 0;

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FLOYD_WARSHALL_k");
        RPlaunchHipKernel(
          (poly_floyd_warshall<POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
          nblocks, nthreads_per_block,
          shmem, res.get_stream(),
          pout, pin, k, N );
        RP_CALI_SUBKERNEL_END("POLYBENCH_FLOYD_WARSHALL_k");

      }

    }
    stopTimer();

  } else if ( vid == Lambda_HIP ) {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      for (Index_type k = 0; k < N; ++k) {

        POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_HIP;
        POLY_FLOYD_WARSHALL_NBLOCKS_HIP;
        constexpr size_t shmem = 0;

        auto poly_floyd_warshall_lambda = [=] __device__ (Index_type i,
                                                          Index_type j) {
          POLYBENCH_FLOYD_WARSHALL_BODY;
        };

        RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FLOYD_WARSHALL_k");
        RPlaunchHipKernel(
          (poly_floyd_warshall_lam<POLY_FLOYD_WARSHALL_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP,
                                   decltype(poly_floyd_warshall_lambda)>),
          nblocks, nthreads_per_block,
          shmem, res.get_stream(),
          N, poly_floyd_warshall_lambda );
        RP_CALI_SUBKERNEL_END("POLYBENCH_FLOYD_WARSHALL_k");

      }

    }
    stopTimer();

  } else if (vid == RAJA_HIP) {

    POLYBENCH_FLOYD_WARSHALL_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::HipKernelFixedAsync<i_block_sz * j_block_sz,
            RAJA::statement::For<1, RAJA::hip_global_size_y_direct<i_block_sz>,   // i
              RAJA::statement::For<2, RAJA::hip_global_size_x_direct<j_block_sz>, // j
                RAJA::statement::Lambda<0>
              >
            >
          >
        >
      >;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("POLYBENCH_FLOYD_WARSHALL_k");
      RAJA::kernel_resource<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N},
                         RAJA::RangeSegment{0, N}),
        res,
        [=] __device__ (Index_type k, Index_type i, Index_type j) {
          POLYBENCH_FLOYD_WARSHALL_BODY_RAJA;
        }
      );
      RP_CALI_SUBKERNEL_END("POLYBENCH_FLOYD_WARSHALL_k");

    }
    stopTimer();

  } else {
      getCout() << "\n  POLYBENCH_FLOYD_WARSHALL : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(POLYBENCH_FLOYD_WARSHALL, Hip, Base_HIP, Lambda_HIP, RAJA_HIP)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

