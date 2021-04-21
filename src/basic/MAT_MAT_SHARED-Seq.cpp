//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

template <typename BODY>
inline void seq_loop(const int st, const int end, BODY const &body) {
  for (int i = st; i < end; ++i) {
    body(i);
  }
}

void MAT_MAT_SHARED::runSeqVariant(VariantID vid) {

  const Index_type run_reps = getRunReps();

  MAT_MAT_SHARED_DATA_SETUP;

  const int Nx = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);
  const int Ny = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);

  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Write Sequential variant here
      for (int by = 0; by < Ny; ++by) {
        for (int bx = 0; bx < Nx; ++bx) {

          MAT_MAT_SHARED_BODY_0

          for (int ty = 0; ty < TL_SZ; ++ty) {
            for (int tx = 0; tx < TL_SZ; ++tx) {
              MAT_MAT_SHARED_BODY_1
            }
          }

          // Sequential loop
          for (int k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

            for (int ty = 0; ty < TL_SZ; ++ty) {
              for (int tx = 0; tx < TL_SZ; ++tx) {

                MAT_MAT_SHARED_BODY_2
              }
            }

            // synchronize();
            for (int ty = 0; ty < TL_SZ; ++ty) {
              for (int tx = 0; tx < TL_SZ; ++tx) {

                MAT_MAT_SHARED_BODY_3
              }
            }

          } // Sequential loop

          for (int ty = 0; ty < TL_SZ; ++ty) {
            for (int tx = 0; tx < TL_SZ; ++tx) {
              MAT_MAT_SHARED_BODY_4
            }
          }
        }
      }

    } // number of iterations
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case Lambda_Seq: {

    startTimer();
    for (Index_type irep = 0; irep < run_reps; ++irep) {

      seq_loop(0, Ny, [&](int by) {
        seq_loop(0, Nx, [&](int bx) {
          MAT_MAT_SHARED_BODY_0

          seq_loop(0, TL_SZ, [&](int ty) {
            seq_loop(0, TL_SZ, [&](int tx) { MAT_MAT_SHARED_BODY_1 });
          });

          // Sequential loop
          for (int k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

            seq_loop(0, TL_SZ, [&](int ty) {
              seq_loop(0, TL_SZ, [&](int tx) { MAT_MAT_SHARED_BODY_2 });
            });

            // synchronize();
            seq_loop(0, TL_SZ, [&](int ty) {
              seq_loop(0, TL_SZ, [&](int tx) { MAT_MAT_SHARED_BODY_3 });
            });

          } // Sequential loop

          seq_loop(0, TL_SZ, [&](int ty) {
            seq_loop(0, TL_SZ, [&](int tx) { MAT_MAT_SHARED_BODY_4 });
          });
        });
      });
    }
    stopTimer();

    break;
  }

  case RAJA_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<launch_policy>(
          RAJA::expt::HOST,
          RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                                RAJA::expt::Threads(TL_SZ, TL_SZ)),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
            RAJA::expt::loop<teams_y>(
                ctx, RAJA::TypedRangeSegment<int>(0, Ny), [&](int by) {
                  RAJA::expt::loop<teams_x>(
                      ctx, RAJA::TypedRangeSegment<int>(0, Nx), [&](int bx) {
                        MAT_MAT_SHARED_BODY_0

                        RAJA::expt::loop<threads_y>(
                            ctx, RAJA::TypedRangeSegment<int>(0, TL_SZ),
                            [&](int ty) {
                              RAJA::expt::loop<threads_x>(
                                  ctx, RAJA::TypedRangeSegment<int>(0, TL_SZ),
                                  [&](int tx) { MAT_MAT_SHARED_BODY_1 });
                            });

                        for (int k = 0; k < (TL_SZ + N - 1) / TL_SZ; k++) {

                          RAJA::expt::loop<threads_y>(
                              ctx, RAJA::TypedRangeSegment<int>(0, TL_SZ),
                              [&](int ty) {
                                RAJA::expt::loop<threads_x>(
                                    ctx, RAJA::TypedRangeSegment<int>(0, TL_SZ),
                                    [&](int tx) { MAT_MAT_SHARED_BODY_2 });
                              });

                          ctx.teamSync();

                          RAJA::expt::loop<threads_y>(
                              ctx, RAJA::TypedRangeSegment<int>(0, TL_SZ),
                              [&](int ty) {
                                RAJA::expt::loop<threads_x>(
                                    ctx, RAJA::TypedRangeSegment<int>(0, TL_SZ),
                                    [&](int tx) { MAT_MAT_SHARED_BODY_3 });
                              });

                          ctx.teamSync();
                        }

                        RAJA::expt::loop<threads_y>(
                            ctx, RAJA::TypedRangeSegment<int>(0, TL_SZ),
                            [&](int ty) {
                              RAJA::expt::loop<threads_x>(
                                  ctx, RAJA::TypedRangeSegment<int>(0, TL_SZ),
                                  [&](int tx) { MAT_MAT_SHARED_BODY_4 });
                            });
                      });
                });
          }); // kernel
    }
    stopTimer();

    break;
  }
#endif // RUN_RAJA_SEQ

  default: {
    std::cout << "\n  MAT_MAT_SHARED : Unknown variant id = " << vid
              << std::endl;
  }
  }
}

} // end namespace basic
} // end namespace rajaperf
