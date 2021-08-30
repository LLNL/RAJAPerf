//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

//#define USE_RAJA_UNROLL
#define RAJA_DIRECT_PRAGMA(X) _Pragma(#X)
#if defined(USE_RAJA_UNROLL)
#define RAJA_UNROLL(N) RAJA_DIRECT_PRAGMA(unroll(N))
#else
#define RAJA_UNROLL(N)
#endif
#define FOREACH_THREAD(i, k, N) for (int i = 0; i < N; i++)

void MASS3DPA::runOpenMPVariant(VariantID vid) {

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  MASS3DPA_DATA_SETUP;

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#pragma omp parallel for
      for (int e = 0; e < NE; ++e) {

        MASS3DPA_0_CPU

        FOREACH_THREAD(dy, y, D1D) {
          FOREACH_THREAD(dx, x, D1D){
            MASS3DPA_1
          }
          FOREACH_THREAD(dx, x, Q1D) {
            MASS3DPA_2
          }
        }

        FOREACH_THREAD(dy, y, D1D) {
          FOREACH_THREAD(qx, x, Q1D) {
            MASS3DPA_3
          }
        }

        FOREACH_THREAD(qy, y, Q1D) {
          FOREACH_THREAD(qx, x, Q1D) {
            MASS3DPA_4
          }
        }

        FOREACH_THREAD(qy, y, Q1D) {
          FOREACH_THREAD(qx, x, Q1D) {
            MASS3DPA_5
          }
        }

        FOREACH_THREAD(d, y, D1D) {
          FOREACH_THREAD(q, x, Q1D) {
            MASS3DPA_6
          }
        }

        FOREACH_THREAD(qy, y, Q1D) {
          FOREACH_THREAD(dx, x, D1D) {
            MASS3DPA_7
          }
        }

        FOREACH_THREAD(dy, y, D1D) {
          FOREACH_THREAD(dx, x, D1D) {
            MASS3DPA_8
          }
        }

        FOREACH_THREAD(dy, y, D1D) {
          FOREACH_THREAD(dx, x, D1D) {
            MASS3DPA_9
          }
        }

      } // element loop
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    //Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t
#if defined(RAJA_DEVICE_ACTIVE)
                                                   ,m3d_device_launch
#endif
                                                   >;

    using outer_x = RAJA::expt::LoopPolicy<RAJA::omp_for_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,m3d_gpu_block_x_policy
#endif
                                           >;

    using inner_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                             ,m3d_gpu_thread_x_policy
#endif
                                             >;

    using inner_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                             ,m3d_gpu_thread_y_policy
#endif
                                             >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //Grid is empty as the host does not need a compute grid to be specified
      RAJA::expt::launch<launch_policy>(
        RAJA::expt::HOST, RAJA::expt::Grid(),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              MASS3DPA_0_CPU

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, D1D),
                [&](int dy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, D1D),
                    [&](int dx) {
                      MASS3DPA_1
                    }
                  );  // RAJA::expt::loop<inner_x>

                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, Q1D),
                    [&](int dx) {
                      MASS3DPA_2
                    }
                  );  // RAJA::expt::loop<inner_x>
                }  // lambda (dy)
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, D1D),
                [&](int dy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, Q1D),
                    [&](int qx) {
                      MASS3DPA_3
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, Q1D),
                [&](int qy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, Q1D),
                    [&](int qx) {
                      MASS3DPA_4
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, Q1D),
                [&](int qy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, Q1D),
                    [&](int qx) {
                      MASS3DPA_5
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, D1D),
                [&](int d) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, Q1D),
                    [&](int q) {
                      MASS3DPA_6
                    }
                  );  // RAJA::expt::loop<inner_x>
                }  
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, Q1D),
                [&](int qy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, D1D),
                    [&](int dx) {
                      MASS3DPA_7
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, D1D),
                [&](int dy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, D1D),
                    [&](int dx) {
                      MASS3DPA_8
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

              ctx.teamSync();

              RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, D1D),
                [&](int dy) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, D1D),
                    [&](int dx) {
                      MASS3DPA_9
                    }
                  );  // RAJA::expt::loop<inner_x>
                }
              );  // RAJA::expt::loop<inner_y>

            }  // lambda (e) 
          );  // RAJA::expt::loop<outer_x> 

        }  // outer lambda (ctx)
      );  // // RAJA::expt::launch

    }  // loop over kernel reps
    stopTimer();

    return;
  }

  default:
    std::cout << "\n MASS3DPA : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

#else 
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf
