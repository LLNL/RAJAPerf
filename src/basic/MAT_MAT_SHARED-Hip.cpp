//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

#define MAT_MAT_SHARED_DATA_SETUP_HIP                                          \
  const Index_type NN = m_N * m_N;                                             \
  allocAndInitHipDeviceData(A, m_A, NN);                                       \
  allocAndInitHipDeviceData(B, m_B, NN);                                       \
  allocAndInitHipDeviceData(C, m_C, NN);

#define MAT_MAT_SHARED_DATA_TEARDOWN_HIP                                       \
  getHipDeviceData(m_A, A, NN);                                                \
  getHipDeviceData(m_B, B, NN);                                                \
  getHipDeviceData(m_C, C, NN);                                                \
  deallocHipDeviceData(A);                                                     \
  deallocHipDeviceData(B);                                                     \
  deallocHipDeviceData(C);

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
void MAT_MAT_SHARED::runHipVariantImpl(VariantID vid)
{
  constexpr Index_type tile_size = gpu_block_size::sqrt(block_size);
  static_assert(tile_size*tile_size == block_size, "Invalid block_size");

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  dim3 blockDim(tile_size, tile_size);
  dim3 gridDim(RAJA_DIVIDE_CEILING_INT(N, blockDim.x),
               RAJA_DIVIDE_CEILING_INT(N, blockDim.y));

  const Index_type Nx = gridDim.x;
  const Index_type Ny = gridDim.y;

  MAT_MAT_SHARED_DATA_SETUP;

  if (vid == Base_HIP) {

    MAT_MAT_SHARED_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipLaunchKernelGGL((mat_mat_shared<tile_size>), dim3(gridDim), dim3(blockDim), 0, 0,
                         N, C, A, B);

      hipErrchk( hipGetLastError() );
    }
    stopTimer();

    MAT_MAT_SHARED_DATA_TEARDOWN_HIP;

  } else if (vid == Lambda_HIP) {

    MAT_MAT_SHARED_DATA_SETUP_HIP;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      auto mat_mat_shared_lam = [=] __device__() {

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
      };

      hipLaunchKernelGGL((lambda_hip<tile_size*tile_size, decltype(mat_mat_shared_lam)>),
        gridDim, blockDim, 0, 0, mat_mat_shared_lam);

      hipErrchk( hipGetLastError() );
    }
    stopTimer();

    MAT_MAT_SHARED_DATA_TEARDOWN_HIP;

  } else if (vid == RAJA_HIP) {

    MAT_MAT_SHARED_DATA_SETUP_HIP;

    constexpr bool async = true;

    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::hip_launch_t<async, tile_size*tile_size>>;

    using teams_x = RAJA::expt::LoopPolicy<RAJA::hip_block_x_direct>;

    using teams_y = RAJA::expt::LoopPolicy<RAJA::hip_block_y_direct>;

    using threads_x = RAJA::expt::LoopPolicy<RAJA::hip_thread_x_direct>;

    using threads_y = RAJA::expt::LoopPolicy<RAJA::hip_thread_y_direct>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<launch_policy>(
        RAJA::expt::Grid(RAJA::expt::Teams(Nx, Ny),
                         RAJA::expt::Threads(tile_size, tile_size)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny),
            [&](Index_type by) {
              RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nx),
                [&](Index_type bx) {

                  MAT_MAT_SHARED_BODY_0(tile_size)

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                    [&](Index_type ty) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_1(tile_size)
                        }
                      );  // RAJA::expt::loop<threads_x>
                    }
                  );  // RAJA::expt::loop<threads_y>

                  for (Index_type k = 0; k < (tile_size + N - 1) / tile_size; k++) {

                    RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                      [&](Index_type ty) {
                        RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_2(tile_size)
                          }
                        );  // RAJA::expt::loop<threads_x>
                      }
                    );  // RAJA::expt::loop<threads_y>

                    ctx.teamSync();

                    RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                      [&](Index_type ty) {
                        RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                          [&](Index_type tx) {
                            MAT_MAT_SHARED_BODY_3(tile_size)
                          }
                        );  // RAJA::expt::loop<threads_x>
                      }
                    );  // RAJA::expt::loop<threads_y>

                    ctx.teamSync();

                  }  // for (k)

                  RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, tile_size),
                    [&](Index_type ty) {
                      RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, tile_size),
                        [&](Index_type tx) {
                          MAT_MAT_SHARED_BODY_4(tile_size)
                        }
                      );  // RAJA::expt::loop<threads_x>
                    }
                  );  // RAJA::expt::loop<threads_y>

                }  // lambda (bx)
              );  // RAJA::expt::loop<teams_x>
            }  // lambda (by)
          );  // RAJA::expt::loop<teams_y>

        }  // outer lambda (ctx)
      );  // RAJA::expt::launch

    }  // loop over kernel reps
    stopTimer();

    MAT_MAT_SHARED_DATA_TEARDOWN_HIP;

  } else {
    getCout() << "\n  MAT_MAT_SHARED : Unknown Hip variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MAT_MAT_SHARED, Hip)

} // end namespace basic
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
