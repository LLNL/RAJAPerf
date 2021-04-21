//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_CUDA)

#include "common/CudaDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

template <typename BODY>
inline __device__ void block_x_direct(const Index_type st, const Index_type end,
                                      BODY const &body) {
  Index_type bx = st + blockIdx.x;
  if (bx < end)
    body(bx);
}

template <typename BODY>
inline __device__ void block_y_direct(const Index_type st, const Index_type end,
                                      BODY const &body) {
  Index_type by = st + blockIdx.y;
  if (by < end)
    body(by);
}

template <typename BODY>
inline __device__ void thread_x_direct(const Index_type st, const Index_type end,
                                       BODY const &body) {
  Index_type tx = st + threadIdx.x;
  if (tx < end)
    body(tx);
}

template <typename BODY>
inline __device__ void thread_y_direct(const Index_type st, const Index_type end,
                                       BODY const &body) {
  Index_type ty = st + threadIdx.y;
  if (ty < end)
    body(ty);
}

#define MAT_MAT_SHARED_DATA_SETUP_CUDA                                         \
  const Index_type NN = getRunSize()*getRunSize();                             \
  allocAndInitCudaDeviceData(A, m_A, NN);                                      \
  allocAndInitCudaDeviceData(B, m_B, NN);                                      \
  allocAndInitCudaDeviceData(C, m_C, NN);

#define MAT_MAT_SHARED_DATA_TEARDOWN_CUDA                                      \
  getCudaDeviceData(m_A, A, NN);                                               \
  getCudaDeviceData(m_B, B, NN);                                               \
  getCudaDeviceData(m_C, C, NN);                                               \
  deallocCudaDeviceData(A);                                                    \
  deallocCudaDeviceData(B);                                                    \
  deallocCudaDeviceData(C);

__global__ void mat_mat_shared(Index_type N, Real_ptr C, Real_ptr A, Real_ptr B) {

  Index_type tx = threadIdx.x;
  Index_type ty = threadIdx.y;
  Index_type bx = blockIdx.x;
  Index_type by = blockIdx.y;

  MAT_MAT_SHARED_BODY_0

  MAT_MAT_SHARED_BODY_1

  for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; k++) {

    MAT_MAT_SHARED_BODY_2

    __syncthreads();

    MAT_MAT_SHARED_BODY_3

    __syncthreads();
  }

  MAT_MAT_SHARED_BODY_4
}

void MAT_MAT_SHARED::runCudaVariant(VariantID vid) {

  const Index_type run_reps = getRunReps();
  const Index_type N = getRunSize();

  dim3 blockdim(TL_SZ, TL_SZ);
  dim3 griddim(RAJA_DIVIDE_CEILING_INT(N, blockdim.x),
               RAJA_DIVIDE_CEILING_INT(N, blockdim.y));

  const Index_type Nx = griddim.x;
  const Index_type Ny = griddim.y;

  MAT_MAT_SHARED_DATA_SETUP;

  if (vid == Base_CUDA) {

    MAT_MAT_SHARED_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      mat_mat_shared<<<griddim, blockdim>>>(N, A, B, C);
    }
    stopTimer();

    MAT_MAT_SHARED_DATA_TEARDOWN_CUDA;

  } else if (vid == Lambda_CUDA) {

    MAT_MAT_SHARED_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      lambda_cuda<<<griddim, blockdim>>>([=] __device__() {
        block_y_direct(0, Ny, [&](Index_type by) {
          block_x_direct(0, Nx, [&](Index_type bx) {
            MAT_MAT_SHARED_BODY_0

            thread_y_direct(0, TL_SZ, [&](Index_type ty) {
              thread_x_direct(0, TL_SZ, [&](Index_type tx) { MAT_MAT_SHARED_BODY_1 });
            });

            for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

              thread_y_direct(0, TL_SZ, [&](Index_type ty) {
                thread_x_direct(0, TL_SZ,
                                [&](Index_type tx) { MAT_MAT_SHARED_BODY_2 });
              });

              __syncthreads();
              thread_y_direct(0, TL_SZ, [&](Index_type ty) {
                thread_x_direct(0, TL_SZ,
                                [&](Index_type tx) { MAT_MAT_SHARED_BODY_3 });
              });

              __syncthreads();
            }

            thread_y_direct(0, TL_SZ, [&](Index_type ty) {
              thread_x_direct(0, TL_SZ, [&](Index_type tx) { MAT_MAT_SHARED_BODY_4 });
            });
          });
        });
      });
    }
    stopTimer();

    MAT_MAT_SHARED_DATA_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    MAT_MAT_SHARED_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<launch_policy>(
          RAJA::expt::DEVICE, RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                             RAJA::expt::Threads(TL_SZ, TL_SZ)),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

            RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny), [&](Index_type by) {
                RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nx), [&](Index_type bx) {

                        MAT_MAT_SHARED_BODY_0

                 RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type ty) {
                   RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type tx) {
                     MAT_MAT_SHARED_BODY_1
                   });
                 });

                 for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; k++) {

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type ty) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_2
                       });
                   });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type ty) {
                    RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type tx) {
                        MAT_MAT_SHARED_BODY_3
                    });
                  });

                  ctx.teamSync();
                 }

                 RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type ty) {
                   RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type tx) {
                       MAT_MAT_SHARED_BODY_4
                   });
                 });

              });
            });
          }); // kernel
    }
    stopTimer();

    MAT_MAT_SHARED_DATA_TEARDOWN_CUDA;

  } else {
    std::cout << "\n  MAT_MAT_SHARED : Unknown Cuda variant id = " << vid
              << std::endl;
  }
}

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
