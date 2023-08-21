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

template < Index_type tile_size >
  __launch_bounds__(tile_size*tile_size)
__global__ void mat_mat_shared(Index_type N, Real_ptr C, Real_ptr A,
                               Real_ptr B) {

  Index_type tx = threadIdx.x;
  Index_type ty = threadIdx.y;
  Index_type bx = blockIdx.x;
  Index_type by = blockIdx.y;

  MAT_MAT_SHARED_BODY_0(tile_size)

  MAT_MAT_SHARED_BODY_1(tile_size)

  for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {

    MAT_MAT_SHARED_BODY_2(tile_size)

    __syncthreads();

    MAT_MAT_SHARED_BODY_3(tile_size)

    __syncthreads();
  }

  MAT_MAT_SHARED_BODY_4(tile_size)
}

template < size_t block_size >
void MAT_MAT_SHARED::runCudaVariantImpl(VariantID vid)
{
  constexpr Index_type tile_size = gpu_block_size::sqrt(block_size);
  static_assert(tile_size*tile_size == block_size, "Invalid block_size");

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  dim3 blockDim(tile_size, tile_size);
  dim3 gridDim(RAJA_DIVIDE_CEILING_INT(N, blockDim.x),
               RAJA_DIVIDE_CEILING_INT(N, blockDim.y));
  constexpr size_t shmem = 0;

  const Index_type Nx = gridDim.x;
  const Index_type Ny = gridDim.y;

  auto res{getCudaResource()};

  MAT_MAT_SHARED_DATA_SETUP;

  if (vid == Base_CUDA) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      mat_mat_shared<tile_size><<<gridDim, blockDim, shmem, res.get_stream()>>>(N, C, A, B);

      cudaErrchk( cudaGetLastError() );
    }
    stopTimer();

  } else if (vid == Lambda_CUDA) {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      lambda_cuda<tile_size*tile_size><<<gridDim, blockDim, shmem, res.get_stream()>>>([=] __device__() {
        auto outer_y = [&](Index_type by) {
          auto outer_x = [&](Index_type bx) {
            MAT_MAT_SHARED_BODY_0(tile_size)

            auto inner_y_1 = [&](Index_type ty) {
              auto inner_x_1 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_1(tile_size) };

              {
                Index_type tx = threadIdx.x;
                if (tx < tile_size)
                  inner_x_1(tx);
              }
            };

            {
              Index_type ty = threadIdx.y;
              if (ty < tile_size)
                inner_y_1(ty);
            }

            for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; ++k) {

              auto inner_y_2 = [&](Index_type ty) {
                auto inner_x_2 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_2(tile_size) };

                {
                  Index_type tx = threadIdx.x;
                  if (tx < tile_size)
                    inner_x_2(tx);
                }
              };

              {
                Index_type ty = threadIdx.y;
                if (ty < tile_size)
                  inner_y_2(ty);
              }

              __syncthreads();

              auto inner_y_3 = [&](Index_type ty) {
                auto inner_x_3 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_3(tile_size) };

                {
                  Index_type tx = threadIdx.x;
                  if (tx < tile_size)
                    inner_x_3(tx);
                }
              };

              {
                Index_type ty = threadIdx.y;
                if (ty < tile_size)
                  inner_y_3(ty);
              }

              __syncthreads();
            }

            auto inner_y_4 = [&](Index_type ty) {
              auto inner_x_4 = [&](Index_type tx) { MAT_MAT_SHARED_BODY_4(tile_size) };

              {
                Index_type tx = threadIdx.x;
                if (tx < tile_size)
                  inner_x_4(tx);
              }
            };

            {
              Index_type ty = threadIdx.y;
              if (ty < tile_size)
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

  } else if (vid == RAJA_CUDA) {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::cuda_launch_t<async, tile_size*tile_size>>;

    using teams_x = RAJA::LoopPolicy<RAJA::cuda_block_x_direct>;

    using teams_y = RAJA::LoopPolicy<RAJA::cuda_block_y_direct>;

    using threads_x = RAJA::LoopPolicy<RAJA::cuda_thread_size_x_direct<tile_size>>;

    using threads_y = RAJA::LoopPolicy<RAJA::cuda_thread_size_y_direct<tile_size>>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(RAJA::Teams(Nx, Ny),
                         RAJA::Threads(tile_size, tile_size)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny),
            [&](Index_type by) {
              RAJA::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nx),
                [&](Index_type bx) {

                  MAT_MAT_SHARED_BODY_0(tile_size)

                  RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                    [&](Index_type ty) {
                      RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_1(tile_size)
                        }
                      );  // RAJA::loop<threads_x>
                    }
                  );  // RAJA::loop<threads_y>

                  for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {

                    RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                      [&](Index_type ty) {
                        RAJA::loop<threads_x>(ctx,
                                                    RAJA::RangeSegment(0, tile_size),
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_2(tile_size)
                          }
                        ); // RAJA::loop<threads_x>
                      }
                    );  // RAJA::loop<threads_y>

                    ctx.teamSync();

                    RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                      [&](Index_type ty) {
                        RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_3(tile_size)
                          }
                        );  // RAJA::loop<threads_x>
                      }
                    );  // RAJA::loop<threads_y>

                    ctx.teamSync();

                  }  // for (k)

                  RAJA::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                    [&](Index_type ty) {
                      RAJA::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_4(tile_size)
                        }
                      );  // RAJA::loop<threads_x>
                    }
                  );  // RAJA::loop<threads_y>

                }  // lambda (bx)
              );  // RAJA::loop<teams_x>
            }  // lambda (by)
          );  // RAJA::loop<teams_y>

        }   // outer lambda (ctx)
      );  // RAJA::launch

    }  // loop over kernel reps
    stopTimer();

  } else {
    getCout() << "\n  MAT_MAT_SHARED : Unknown Cuda variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BOILERPLATE(MAT_MAT_SHARED, Cuda)

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_CUDA
