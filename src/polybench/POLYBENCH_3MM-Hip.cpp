//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "POLYBENCH_3MM.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf
{
namespace polybench
{

//
// Define thread block shape for Hip execution
//
#define in_block_sz (32)
#define out_block_sz (block_size / in_block_sz)

#define POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP \
  in_block_sz, out_block_sz

#define POLY_3MM_THREADS_PER_BLOCK_HIP \
  dim3 nthreads_per_block(POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, 1);

#define POLY_3MM_1_NBLOCKS_HIP \
  dim3 nblocks1(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, out_block_sz)), \
                static_cast<size_t>(1));

#define POLY_3MM_2_NBLOCKS_HIP \
  dim3 nblocks2(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nj, out_block_sz)), \
                static_cast<size_t>(1));

#define POLY_3MM_3_NBLOCKS_HIP \
  dim3 nblocks3(static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(nl, in_block_sz)), \
                static_cast<size_t>(RAJA_DIVIDE_CEILING_INT(ni, out_block_sz)), \
                static_cast<size_t>(1));


#define POLYBENCH_3MM_DATA_SETUP_HIP \
  allocAndInitHipDeviceData(A, m_A, m_ni * m_nk); \
  allocAndInitHipDeviceData(B, m_B, m_nk * m_nj); \
  allocAndInitHipDeviceData(C, m_C, m_nj * m_nm); \
  allocAndInitHipDeviceData(D, m_D, m_nm * m_nl); \
  allocAndInitHipDeviceData(E, m_E, m_ni * m_nj); \
  allocAndInitHipDeviceData(F, m_F, m_nj * m_nl); \
  allocAndInitHipDeviceData(G, m_G, m_ni * m_nl);


#define POLYBENCH_3MM_TEARDOWN_HIP \
  getHipDeviceData(m_G, G, m_ni * m_nl); \
  deallocHipDeviceData(A); \
  deallocHipDeviceData(B); \
  deallocHipDeviceData(C); \
  deallocHipDeviceData(D); \
  deallocHipDeviceData(E); \
  deallocHipDeviceData(F); \
  deallocHipDeviceData(G);

template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_1(Real_ptr E, Real_ptr A, Real_ptr B,
                           Index_type ni, Index_type nj, Index_type nk)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type j = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    POLYBENCH_3MM_BODY1;
    for (Index_type k=0; k < nk; ++k) {
      POLYBENCH_3MM_BODY2;
    }
    POLYBENCH_3MM_BODY3;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_1_lam(Index_type ni, Index_type nj,
                               Lambda body)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type j = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && j < nj ) {
    body(i, j);
  }
}

template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_2(Real_ptr F, Real_ptr C, Real_ptr D,
                           Index_type nj, Index_type nl, Index_type nm)
{
  Index_type j = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( j < nj && l < nl ) {
    POLYBENCH_3MM_BODY4;
    for (Index_type m=0; m < nm; ++m) {
      POLYBENCH_3MM_BODY5;
    }
    POLYBENCH_3MM_BODY6;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_2_lam(Index_type nj, Index_type nl,
                               Lambda body)
{
  Index_type j = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( j < nj && l < nl ) {
    body(j, l);
  }
}

template < size_t in_block_size, size_t out_block_size >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_3(Real_ptr G, Real_ptr E, Real_ptr F,
                           Index_type ni, Index_type nl, Index_type nj)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && l < nl ) {
    POLYBENCH_3MM_BODY7;
    for (Index_type j=0; j < nj; ++j) {
      POLYBENCH_3MM_BODY8;
    }
    POLYBENCH_3MM_BODY9;
  }
}

template < size_t in_block_size, size_t out_block_size, typename Lambda >
__launch_bounds__(in_block_size*out_block_size)
__global__ void poly_3mm_3_lam(Index_type ni, Index_type nl,
                               Lambda body)
{
  Index_type i = blockIdx.y * out_block_size + threadIdx.y;
  Index_type l = blockIdx.x * in_block_size + threadIdx.x;

  if ( i < ni && l < nl ) {
    body(i, l);
  }
}


template < size_t block_size >
void POLYBENCH_3MM::runHipVariantImpl(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  POLYBENCH_3MM_DATA_SETUP;

  if ( vid == Base_HIP ) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLY_3MM_THREADS_PER_BLOCK_HIP;

      POLY_3MM_1_NBLOCKS_HIP;
      hipLaunchKernelGGL((poly_3mm_1<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks1) , dim3(nthreads_per_block), 0, 0,
                         E, A, B,
                         ni, nj, nk);
      hipErrchk( hipGetLastError() );

      POLY_3MM_2_NBLOCKS_HIP;
      hipLaunchKernelGGL((poly_3mm_2<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks2), dim3(nthreads_per_block), 0, 0,
                         F, C, D,
                         nj, nl, nm);
      hipErrchk( hipGetLastError() );

      POLY_3MM_3_NBLOCKS_HIP;
      hipLaunchKernelGGL((poly_3mm_3<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP>),
                         dim3(nblocks3), dim3(nthreads_per_block), 0, 0,
                         G, E, F,
                         ni, nl, nj);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_HIP;

  } else if (vid == Lambda_HIP) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      POLY_3MM_THREADS_PER_BLOCK_HIP;

      auto poly_3mm_1_lambda = [=] __device__ (Index_type i, Index_type j) {
        POLYBENCH_3MM_BODY1;
        for (Index_type k=0; k < nk; ++k) {
          POLYBENCH_3MM_BODY2;
        }
        POLYBENCH_3MM_BODY3;
      };

      POLY_3MM_1_NBLOCKS_HIP;
      hipLaunchKernelGGL((poly_3mm_1_lam<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_3mm_1_lambda)>),
                         dim3(nblocks1), dim3(nthreads_per_block), 0, 0,
                         ni, nj, poly_3mm_1_lambda);
      hipErrchk( hipGetLastError() );

      auto poly_3mm_2_lambda = [=] __device__ (Index_type j, Index_type l) {
        POLYBENCH_3MM_BODY4;
        for (Index_type m=0; m < nm; ++m) {
          POLYBENCH_3MM_BODY5;
        }
        POLYBENCH_3MM_BODY6;
      };

      POLY_3MM_2_NBLOCKS_HIP;
      hipLaunchKernelGGL((poly_3mm_2_lam<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_3mm_2_lambda)>),
                         dim3(nblocks2), dim3(nthreads_per_block), 0, 0,
                         nj, nl, poly_3mm_2_lambda);
      hipErrchk( hipGetLastError() );

      auto poly_3mm_3_lambda = [=] __device__ (Index_type i, Index_type l) {
        POLYBENCH_3MM_BODY7;
        for (Index_type j=0; j < nj; ++j) {
          POLYBENCH_3MM_BODY8;
        }
        POLYBENCH_3MM_BODY9;
      };

      POLY_3MM_3_NBLOCKS_HIP;
      hipLaunchKernelGGL((poly_3mm_3_lam<POLY_3MM_THREADS_PER_BLOCK_TEMPLATE_PARAMS_HIP, decltype(poly_3mm_3_lambda)>),
                         dim3(nblocks3), dim3(nthreads_per_block), 0, 0,
                         ni, nl, poly_3mm_3_lambda);
      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    POLYBENCH_3MM_DATA_SETUP_HIP;

    POLYBENCH_3MM_VIEWS_RAJA;

    using EXEC_POL =
      RAJA::KernelPolicy<
        RAJA::statement::HipKernelFixedAsync<out_block_sz * in_block_sz,
          RAJA::statement::Tile<0, RAJA::tile_fixed<out_block_sz>,
                                   RAJA::hip_block_y_direct,
            RAJA::statement::Tile<1, RAJA::tile_fixed<in_block_sz>,
                                     RAJA::hip_block_x_direct,
              RAJA::statement::For<0, RAJA::hip_thread_y_direct,   // outer
                RAJA::statement::For<1, RAJA::hip_thread_x_direct, // inner
                  RAJA::statement::Lambda<0, RAJA::Params<0>>,
                  RAJA::statement::For<2, RAJA::seq_exec,
                    RAJA::statement::Lambda<1, RAJA::Segs<0,1,2>, RAJA::Params<0>>
                  >,
                  RAJA::statement::Lambda<2, RAJA::Segs<0,1>, RAJA::Params<0>>
                >
              >
            >
          >
        >
      >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nk}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_3MM_BODY1_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j, Index_type k,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY2_RAJA;
        },
        [=] __device__ (Index_type i, Index_type j,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY3_RAJA;
        }

      );

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, nj},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nm}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_3MM_BODY4_RAJA;
        },
        [=] __device__ (Index_type j, Index_type l, Index_type m,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY5_RAJA;
        },
        [=] __device__ (Index_type j, Index_type l,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY6_RAJA;
        }

      );

      RAJA::kernel_param<EXEC_POL>(
        RAJA::make_tuple(RAJA::RangeSegment{0, ni},
                         RAJA::RangeSegment{0, nl},
                         RAJA::RangeSegment{0, nj}),
        RAJA::tuple<Real_type>{0.0},

        [=] __device__ ( Real_type &dot) {
          POLYBENCH_3MM_BODY7_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l, Index_type j,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY8_RAJA;
        },
        [=] __device__ (Index_type i, Index_type l,
                        Real_type &dot) {
          POLYBENCH_3MM_BODY9_RAJA;
        }

      );

    }
    stopTimer();

    POLYBENCH_3MM_TEARDOWN_HIP;

  } else {
      getCout() << "\n  POLYBENCH_3MM : Unknown Hip variant id = " << vid << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(POLYBENCH_3MM, Hip)

} // end namespace polybench
} // end namespace rajaperf

#endif  // RAJA_ENABLE_HIP

