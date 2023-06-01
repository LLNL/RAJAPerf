//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-23, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

template < size_t block_size >
  __launch_bounds__(block_size)
__global__ void Mass3DPA(const Real_ptr B, const Real_ptr Bt,
                         const Real_ptr D, const Real_ptr X, Real_ptr Y) {

  const int e = hipBlockIdx_x;

  MASS3DEA_0_GPU

  GPU_FOREACH_THREAD(dy, y, MPA_D1D) {
    GPU_FOREACH_THREAD(dx, x, MPA_D1D){
      MASS3DEA_1
    }
    GPU_FOREACH_THREAD(dx, x, MPA_Q1D) {
      MASS3DEA_2
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(dy, y, MPA_D1D) {
    GPU_FOREACH_THREAD(qx, x, MPA_Q1D) {
      MASS3DEA_3
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, MPA_Q1D) {
    GPU_FOREACH_THREAD(qx, x, MPA_Q1D) {
      MASS3DEA_4
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, MPA_Q1D) {
    GPU_FOREACH_THREAD(qx, x, MPA_Q1D) {
      MASS3DEA_5
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(d, y, MPA_D1D) {
    GPU_FOREACH_THREAD(q, x, MPA_Q1D) {
      MASS3DEA_6
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, MPA_Q1D) {
    GPU_FOREACH_THREAD(dx, x, MPA_D1D) {
      MASS3DEA_7
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dy, y, MPA_D1D) {
    GPU_FOREACH_THREAD(dx, x, MPA_D1D) {
      MASS3DEA_8
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(dy, y, MPA_D1D) {
    GPU_FOREACH_THREAD(dx, x, MPA_D1D) {
      MASS3DEA_9
    }
  }
}

template < size_t block_size >
void MASS3DEA::runHipVariantImpl(VariantID vid) {
  const Index_type run_reps = getRunReps();

  MASS3DEA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    dim3 nblocks(NE);
    dim3 nthreads_per_block(MPA_Q1D, MPA_Q1D, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipLaunchKernelGGL((Mass3DPA<block_size>), dim3(nblocks), dim3(nthreads_per_block), 0, 0,
                         B, Bt, D, X, Y);

      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    break;
  }

  case RAJA_HIP: {

    constexpr bool async = true;

    using launch_policy = RAJA::LaunchPolicy<RAJA::hip_launch_t<async, MPA_Q1D*MPA_Q1D>>;

    using outer_x = RAJA::LoopPolicy<RAJA::hip_block_x_direct>;

    using inner_x = RAJA::LoopPolicy<RAJA::hip_thread_x_loop>;

    using inner_y = RAJA::LoopPolicy<RAJA::hip_thread_y_loop>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::launch<launch_policy>(
        RAJA::LaunchParams(RAJA::Teams(NE),
                         RAJA::Threads(MPA_Q1D, MPA_Q1D, 1)),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {
          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              MASS3DEA_0_GPU

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DEA_1
                    }
                  );  // RAJA::loop<inner_x>

                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int dx) {
                      MASS3DEA_2
                    }
                  );  // RAJA::loop<inner_x>
                } // lambda (dy)
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DEA_3
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DEA_4
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DEA_5
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int d) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int q) {
                      MASS3DEA_6
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DEA_7
                    }
                  );  // RAJA::loop<inner_x
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DEA_8
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

              ctx.teamSync();

              RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DEA_9
                    }
                  );  // RAJA::loop<inner_x>
                }
              );  // RAJA::loop<inner_y>

            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch

    }  // loop over kernel reps
    stopTimer();

    break;
  }

  default: {

    getCout() << "\n MASS3DEA : Unknown Hip variant id = " << vid << std::endl;
    break;
  }
  }
}

RAJAPERF_GPU_BLOCK_SIZE_TUNING_DEFINE_BIOLERPLATE(MASS3DEA, Hip)

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP