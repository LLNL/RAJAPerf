//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include "common/StdParUtils.hpp"
#include <algorithm>
#include <execution>

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
#define CPU_FOREACH(i, k, N) for (int i = 0; i < N; i++)

void MASS3DPA::runStdParVariant(VariantID vid, size_t tune_idx) {
#if defined(RUN_STDPAR)

  const Index_type run_reps = getRunReps();

  MASS3DPA_DATA_SETUP;

  switch (vid) {

  case Base_StdPar: {

    auto begin = counting_iterator<int>(0);
    auto end   = counting_iterator<int>((int)NE);

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      std::for_each( std::execution::par_unseq,
                     begin, end,
                     [=](int e) {

        MASS3DPA_0_CPU

        CPU_FOREACH(dy, y, MPA_D1D) {
          CPU_FOREACH(dx, x, MPA_D1D){
            MASS3DPA_1
          }
          CPU_FOREACH(dx, x, MPA_Q1D) {
            MASS3DPA_2
          }
        }

        CPU_FOREACH(dy, y, MPA_D1D) {
          CPU_FOREACH(qx, x, MPA_Q1D) {
            MASS3DPA_3
          }
        }

        CPU_FOREACH(qy, y, MPA_Q1D) {
          CPU_FOREACH(qx, x, MPA_Q1D) {
            MASS3DPA_4
          }
        }

        CPU_FOREACH(qy, y, MPA_Q1D) {
          CPU_FOREACH(qx, x, MPA_Q1D) {
            MASS3DPA_5
          }
        }

        CPU_FOREACH(d, y, MPA_D1D) {
          CPU_FOREACH(q, x, MPA_Q1D) {
            MASS3DPA_6
          }
        }

        CPU_FOREACH(qy, y, MPA_Q1D) {
          CPU_FOREACH(dx, x, MPA_D1D) {
            MASS3DPA_7
          }
        }

        CPU_FOREACH(dy, y, MPA_D1D) {
          CPU_FOREACH(dx, x, MPA_D1D) {
            MASS3DPA_8
          }
        }

        CPU_FOREACH(dy, y, MPA_D1D) {
          CPU_FOREACH(dx, x, MPA_D1D) {
            MASS3DPA_9
          }
        }

      }); // element loop
    }
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_STDPAR)
  case RAJA_StdPar: {

    //Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t
#if defined(RAJA_DEVICE_ACTIVE)
                                                   ,m3d_device_launch
#endif
                                                   >;

    using outer_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
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

      RAJA::expt::launch<launch_policy>(
        RAJA::expt::HOST, RAJA::expt::Resources(),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
            RAJA::expt::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE), [&](int e) {

                  MASS3DPA_0_CPU

                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int dx) {
                          MASS3DPA_1
                       });

                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D), [&](int dx) {
                          MASS3DPA_2
                      });
                   });

                  ctx.teamSync();

                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D), [&](int qx) {
                          MASS3DPA_3
                      });
                   });

                  ctx.teamSync();

                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D), [&](int qy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D), [&](int qx) {
                          MASS3DPA_4
                      });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D), [&](int qy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D), [&](int qx) {
                          MASS3DPA_5
                      });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int d) {
                    RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_Q1D), [&](int q) {
                        MASS3DPA_6
                     });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_Q1D), [&](int qy) {
                    RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int dx) {
                        MASS3DPA_7
                     });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int dx) {
                          MASS3DPA_8
                      });
                  });

                  ctx.teamSync();

                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int dy) {
                    RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, MPA_D1D), [&](int dx) {
                        MASS3DPA_9
                    });
                  });
              });
          });
    }
    stopTimer();

    return;
  }
#endif // RUN_RAJA_STDPAR

  default:
    std::cout << "\n MASS3DPA : Unknown StdPar variant id = " << vid << std::endl;
  }
#endif
}

} // end namespace apps
} // end namespace rajaperf
