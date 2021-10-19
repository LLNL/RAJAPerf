//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives loop unrolling
//#define USE_RAJA_UNROLL

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "common/HipDataUtils.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define MASS3DPA_DATA_SETUP_HIP                                           \
  allocAndInitHipDeviceData(B, m_B, MPA_Q1D *MPA_D1D);                    \
  allocAndInitHipDeviceData(Bt, m_Bt, MPA_Q1D *MPA_D1D);                  \
  allocAndInitHipDeviceData(D, m_D, MPA_Q1D *MPA_Q1D *MPA_Q1D *m_NE);     \
  allocAndInitHipDeviceData(X, m_X, MPA_D1D *MPA_D1D *MPA_D1D *m_NE);     \
  allocAndInitHipDeviceData(Y, m_Y, MPA_D1D *MPA_D1D *MPA_D1D *m_NE);

#define MASS3DPA_DATA_TEARDOWN_HIP                                        \
  getHipDeviceData(m_Y, Y, MPA_D1D *MPA_D1D *MPA_D1D *m_NE);              \
  deallocHipDeviceData(B);                                                \
  deallocHipDeviceData(Bt);                                               \
  deallocHipDeviceData(D);                                                \
  deallocHipDeviceData(X);                                                \
  deallocHipDeviceData(Y);

__global__ void Mass3DPA(Index_type NE, const Real_ptr B, const Real_ptr Bt,
                         const Real_ptr D, const Real_ptr X, Real_ptr Y) {

  const int e = hipBlockIdx_x;

  MASS3DPA_0_GPU

  GPU_FOREACH_THREAD(dy, y, MPA_D1D) {
    GPU_FOREACH_THREAD(dx, x, MPA_D1D){
      MASS3DPA_1
    }
    GPU_FOREACH_THREAD(dx, x, MPA_Q1D) {
      MASS3DPA_2
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(dy, y, MPA_D1D) {
    GPU_FOREACH_THREAD(qx, x, MPA_Q1D) {
      MASS3DPA_3
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, MPA_Q1D) {
    GPU_FOREACH_THREAD(qx, x, MPA_Q1D) {
      MASS3DPA_4
    }
  }
  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, MPA_Q1D) {
    GPU_FOREACH_THREAD(qx, x, MPA_Q1D) {
      MASS3DPA_5
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(d, y, MPA_D1D) {
    GPU_FOREACH_THREAD(q, x, MPA_Q1D) {
      MASS3DPA_6
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(qy, y, MPA_Q1D) {
    GPU_FOREACH_THREAD(dx, x, MPA_D1D) {
      MASS3DPA_7
    }
  }
  __syncthreads();

  GPU_FOREACH_THREAD(dy, y, MPA_D1D) {
    GPU_FOREACH_THREAD(dx, x, MPA_D1D) {
      MASS3DPA_8
    }
  }

  __syncthreads();
  GPU_FOREACH_THREAD(dy, y, MPA_D1D) {
    GPU_FOREACH_THREAD(dx, x, MPA_D1D) {
      MASS3DPA_9
    }
  }
}

void MASS3DPA::runHipVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  MASS3DPA_DATA_SETUP;

  switch (vid) {

  case Base_HIP: {

    MASS3DPA_DATA_SETUP_HIP;

    dim3 grid_size(NE);
    dim3 block_size(MPA_Q1D, MPA_Q1D, 1);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      hipLaunchKernelGGL((Mass3DPA), dim3(grid_size), dim3(block_size), 0, 0,
                         NE, B, Bt, D, X, Y);

      hipErrchk( hipGetLastError() );

    }
    stopTimer();

    MASS3DPA_DATA_TEARDOWN_HIP;

    break;
  }

  case RAJA_HIP: {

    MASS3DPA_DATA_SETUP_HIP;

    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t
                                                   ,RAJA::expt::hip_launch_t<true>
                                                   >;

    using outer_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                           ,RAJA::hip_block_x_direct
                                           >;

    using inner_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                             ,RAJA::hip_thread_x_loop
                                             >;

    using inner_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
                                             ,RAJA::hip_thread_y_loop
                                             >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<launch_policy>(
        RAJA::expt::DEVICE,
        RAJA::expt::Grid(RAJA::expt::Teams(NE),
                         RAJA::expt::Threads(MPA_Q1D, MPA_Q1D, 1)),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
          RAJA::expt::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              MASS3DPA_0_GPU

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DPA_1
                    }
                  );  // RAJA::expt::loop<inner_x> 

                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int dx) {
                      MASS3DPA_2
                    }
                  );  // RAJA::expt::loop<inner_x>
                } // lambda (dy)
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DPA_3
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DPA_4
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int qx) {
                      MASS3DPA_5
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int d) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                    [&](int q) {
                      MASS3DPA_6
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D),
                [&](int qy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DPA_7
                    }
                  );  // RAJA::expt::loop<inner_x
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DPA_8
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                [&](int dy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D),
                    [&](int dx) {
                      MASS3DPA_9
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

            }  // lambda (e)
          );  // RAJA::expt::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::expt::launch

    }  // loop over kernel reps
    stopTimer();

    MASS3DPA_DATA_TEARDOWN_HIP;

    break;
  }

  default: {

    std::cout << "\n MASS3DPA : Unknown Hip variant id = " << vid << std::endl;
    break;
  }
  }
}

} // end namespace apps
} // end namespace rajaperf

#endif // RAJA_ENABLE_HIP
