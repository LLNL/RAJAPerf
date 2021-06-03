//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MASS3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define D1D 4
#define Q1D 5
#define B_(x, y) B[x + Q1D * y]
#define Bt_(x, y) Bt[x + D1D * y]
#define X_(dx, dy, dz, e)                                                      \
  X[dx + D1D * dy + D1D * D1D * dz + D1D * D1D * D1D * e]
#define Y_(dx, dy, dz, e)                                                      \
  Y[dx + D1D * dy + D1D * D1D * dz + D1D * D1D * D1D * e]
#define D_(qx, qy, qz, e)                                                      \
  D[qx + Q1D * qy + Q1D * Q1D * qz + Q1D * Q1D * Q1D * e]

#define RAJA_DIRECT_PRAGMA(X) _Pragma(#X)
#define RAJA_UNROLL(N) RAJA_DIRECT_PRAGMA(unroll(N))
#define FOREACH_THREAD(i, k, N) for (int i = 0; i < N; i++)

void MASS3DPA::runSeqVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  MASS3DPA_DATA_SETUP;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (int e = 0; e < NE; ++e) {

        MASS3DPA_0_CPU

        FOREACH_THREAD(dy, y, D1D) {
          FOREACH_THREAD(dx, x, D1D){MASS3DPA_1} FOREACH_THREAD(dx, x, Q1D) {
            MASS3DPA_2
          }
        }

        FOREACH_THREAD(dy, y, D1D) {
          FOREACH_THREAD(qx, x, Q1D) { MASS3DPA_3 }
        }

        FOREACH_THREAD(qy, y, Q1D) {
          FOREACH_THREAD(qx, x, Q1D) { MASS3DPA_4 }
        }

        FOREACH_THREAD(qy, y, Q1D) {
          FOREACH_THREAD(qx, x, Q1D) { MASS3DPA_5 }
        }

        FOREACH_THREAD(d, y, D1D) {
          FOREACH_THREAD(q, x, Q1D) { MASS3DPA_6 }
        }

        FOREACH_THREAD(qy, y, Q1D) {
          FOREACH_THREAD(dx, x, D1D) { MASS3DPA_7 }
        }

        FOREACH_THREAD(dy, y, D1D) {
          FOREACH_THREAD(dx, x, D1D) { MASS3DPA_8 }
        }

        FOREACH_THREAD(dy, y, D1D) {
          FOREACH_THREAD(dx, x, D1D) { MASS3DPA_9 }
        }

      } // element loop
    }
    stopTimer();

    break;
  }

  case RAJA_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<launch_policy>(
          RAJA::expt::HOST,
          RAJA::expt::Resources(RAJA::expt::Teams(NE),
                                RAJA::expt::Threads(Q1D, Q1D, 1)),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {
            RAJA::expt::loop<teams_x>(
                ctx, RAJA::RangeSegment(0, NE), [&](int e) {
                  constexpr int MQ1 = Q1D;
                  constexpr int MD1 = D1D;
                  constexpr int MDQ = (MQ1 > MD1) ? MQ1 : MD1;
                  double sDQ[MQ1 * MD1];
                  double(*B)[MD1] = (double(*)[MD1])sDQ;
                  double(*Bt)[MQ1] = (double(*)[MQ1])sDQ;
                  double sm0[MDQ * MDQ * MDQ];
                  double sm1[MDQ * MDQ * MDQ];
                  double(*X)[MD1][MD1] = (double(*)[MD1][MD1])sm0;
                  double(*DDQ)[MD1][MQ1] = (double(*)[MD1][MQ1])sm1;
                  double(*DQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm0;
                  double(*QQQ)[MQ1][MQ1] = (double(*)[MQ1][MQ1])sm1;
                  double(*QQD)[MQ1][MD1] = (double(*)[MQ1][MD1])sm0;
                  double(*QDD)[MD1][MD1] = (double(*)[MD1][MD1])sm1;

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, D1D), [&](int dy) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, D1D), [&](int dx) {
                              RAJA_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz) {
                              }
                            });

                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, Q1D), [&](int dx) {

                            });
                      });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, D1D), [&](int dy) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, Q1D), [&](int qx) {
                              double u[D1D];
                              RAJA_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; dz++) {
                              }
                              RAJA_UNROLL(MD1)
                              for (int dx = 0; dx < D1D; ++dx) {
                                RAJA_UNROLL(MD1)
                                for (int dz = 0; dz < D1D; ++dz) {
                                }
                              }
                              RAJA_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz) {
                              }
                            });
                      });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, Q1D), [&](int qy) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, Q1D), [&](int qx) {
                              double u[D1D];
                              RAJA_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; dz++) {
                              }
                              RAJA_UNROLL(MD1)
                              for (int dy = 0; dy < D1D; ++dy) {
                                RAJA_UNROLL(MD1)
                                for (int dz = 0; dz < D1D; dz++) {
                                }
                              }
                              RAJA_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; dz++) {
                              }
                            });
                      });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, Q1D), [&](int qy) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, Q1D), [&](int qx) {
                              RAJA_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; qz++) {
                              }
                              RAJA_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz) {
                                RAJA_UNROLL(MQ1)
                                for (int qz = 0; qz < Q1D; qz++) {
                                }
                              }
                              RAJA_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; qz++) {
                              }
                            });
                      });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, D1D), [&](int d) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, Q1D), [&](int q) {

                            });
                      });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, Q1D), [&](int qy) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, D1D), [&](int dx) {
                              double u[Q1D];
                              RAJA_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz) {
                              }
                              RAJA_UNROLL(MQ1)
                              for (int qx = 0; qx < Q1D; ++qx) {
                                RAJA_UNROLL(MQ1)
                                for (int qz = 0; qz < Q1D; ++qz) {
                                }
                              }
                              RAJA_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz) {
                              }
                            });
                      });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, D1D), [&](int dy) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, D1D), [&](int dx) {
                              RAJA_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz) {
                              }
                              RAJA_UNROLL(MQ1)
                              for (int qy = 0; qy < Q1D; ++qy) {
                                RAJA_UNROLL(MQ1)
                                for (int qz = 0; qz < Q1D; ++qz) {
                                }
                              }
                              RAJA_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz) {
                              }
                            });
                      });

                  ctx.teamSync();

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, D1D), [&](int dy) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, D1D), [&](int dx) {
                              double u[D1D];
                              RAJA_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz) {
                              }
                              RAJA_UNROLL(MQ1)
                              for (int qz = 0; qz < Q1D; ++qz) {
                                RAJA_UNROLL(MD1)
                                for (int dz = 0; dz < D1D; ++dz) {
                                }
                              }
                              RAJA_UNROLL(MD1)
                              for (int dz = 0; dz < D1D; ++dz) {
                              }
                            });
                      });
                });
          });
    }
    stopTimer();

    return;
  }

  default:
    std::cout << "\n MASS3DPA : Unknown Seq variant id = " << vid << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf
