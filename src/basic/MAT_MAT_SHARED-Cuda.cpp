//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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

#define MAT_MAT_SHARED_DATA_SETUP_CUDA                                         \
  const Index_type NN = m_N * m_N;                                             \
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

__global__ void mat_mat_shared(Index_type N, Real_ptr C, Real_ptr A,
                               Real_ptr B) {

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
  const Index_type N = m_N;

  dim3 block_size(TL_SZ, TL_SZ);
  dim3 grid_size(RAJA_DIVIDE_CEILING_INT(N, block_size.x),
               RAJA_DIVIDE_CEILING_INT(N, block_size.y));

  const Index_type Nx = grid_size.x;
  const Index_type Ny = grid_size.y;

  MAT_MAT_SHARED_DATA_SETUP;

  if (vid == Base_CUDA) {

    MAT_MAT_SHARED_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      mat_mat_shared<<<grid_size, block_size>>>(N, C, A, B);

      cudaErrchk( cudaGetLastError() );
    }
    stopTimer();

    MAT_MAT_SHARED_DATA_TEARDOWN_CUDA;

  } else if (vid == Lambda_CUDA) {

    MAT_MAT_SHARED_DATA_SETUP_CUDA;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      lambda_cuda<<<grid_size, block_size>>>([=] __device__() {
        auto outer_y = [&](Index_type by) {
          auto outer_x = [&](Index_type bx) {
            MAT_MAT_SHARED_BODY_0

            auto inner_y_1 = [&](Index_type ty) {
              auto inner_x_1 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_1 };

              {
                Index_type tx = threadIdx.x;
                if (tx < TL_SZ)
                  inner_x_1(tx);
              }
            };

            {
              Index_type ty = threadIdx.y;
              if (ty < TL_SZ)
                inner_y_1(ty);
            }

            for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

              auto inner_y_2 = [&](Index_type ty) {
                auto inner_x_2 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_2 };

                {
                  Index_type tx = threadIdx.x;
                  if (tx < TL_SZ)
                    inner_x_2(tx);
                }
              };

              {
                Index_type ty = threadIdx.y;
                if (ty < TL_SZ)
                  inner_y_2(ty);
              }

              __syncthreads();

              auto inner_y_3 = [&](Index_type ty) {
                auto inner_x_3 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_3 };

                {
                  Index_type tx = threadIdx.x;
                  if (tx < TL_SZ)
                    inner_x_3(tx);
                }
              };

              {
                Index_type ty = threadIdx.y;
                if (ty < TL_SZ)
                  inner_y_3(ty);
              }

              __syncthreads();
            }

            auto inner_y_4 = [&](Index_type ty) {
              auto inner_x_4 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_4 };

              {
                Index_type tx = threadIdx.x;
                if (tx < TL_SZ)
                  inner_x_4(tx);
              }
            };

            {
              Index_type ty = threadIdx.y;
              if (ty < TL_SZ)
                inner_y_4(ty);
            }
          }; // outer_x

          {
            Index_type bx = blockIdx.x;
            if(bx < Nx) outer_x(bx);
          }
        };

        {
          Index_type by = blockIdx.y;
          if(by < Ny) outer_y(by);
        }
      });

      cudaErrchk( cudaGetLastError() );
    }
    stopTimer();

    MAT_MAT_SHARED_DATA_TEARDOWN_CUDA;

  } else if (vid == RAJA_CUDA) {

    MAT_MAT_SHARED_DATA_SETUP_CUDA;

    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t
                                                   ,RAJA::expt::cuda_launch_t<true>
                                                   >;

    using teams_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                           ,RAJA::cuda_block_x_direct
                                           >;

    using teams_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                           ,RAJA::cuda_block_y_direct
                                           >;

    using threads_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                             ,RAJA::cuda_thread_x_direct
                                             >;

    using threads_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                             ,RAJA::cuda_thread_y_direct
                                             >;


    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<launch_policy>(
        RAJA::expt::DEVICE,
        RAJA::expt::Grid(RAJA::expt::Teams(Nx, Ny),
                         RAJA::expt::Threads(TL_SZ, TL_SZ)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny), 
            [&](Index_type by) {
              RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nx), 
                [&](Index_type bx) {

                  MAT_MAT_SHARED_BODY_0

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), 
                    [&](Index_type ty) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), 
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_1
                        }
                      );  // RAJA::expt::loop<threads_x>
                    } 
                  );  // RAJA::expt::loop<threads_y>

                  for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; k++) {
 
                    RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ),
                      [&](Index_type ty) {
                        RAJA::expt::loop<threads_x>(ctx, 
                                                    RAJA::RangeSegment(0, TL_SZ), 
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_2
                          }
                        ); // RAJA::expt::loop<threads_x>
                      }
                    );  // RAJA::expt::loop<threads_y>

                    ctx.teamSync();

                    RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ),
                      [&](Index_type ty) {
                        RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), 
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_3
                          }
                        );  // RAJA::expt::loop<threads_x>
                      }
                    );  // RAJA::expt::loop<threads_y>

                    ctx.teamSync();

                  }  // for (k)

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), 
                    [&](Index_type ty) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_4
                        }
                      );  // RAJA::expt::loop<threads_x>
                    }
                  );  // RAJA::expt::loop<threads_y>

                }  // lambda (bx)
              );  // RAJA::expt::loop<teams_x>
            }  // lambda (by)
          );  // RAJA::expt::loop<teams_y>

        }   // outer lambda (ctx)
      );  // RAJA::expt::launch

    }  // loop over kernel reps
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
