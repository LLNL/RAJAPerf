//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT_SHARED.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

template <typename BODY>
inline void seq_loop(const int st, const int end, BODY const &body) {
  for (int i = st; i < end; ++i) {
    body(i);
  }
}

template <typename BODY>
inline void par_loop(const int st, const int end, BODY const &body) {

#pragma omp for
  for (int i = st; i < end; ++i) {
    body(i);
  }
}

void MAT_MAT_SHARED::runOpenMPVariant(VariantID vid) {
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type N = getRunSize();

  MAT_MAT_SHARED_DATA_SETUP;

  const Index_type Nx = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);
  const Index_type Ny = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#pragma omp parallel
      {
#pragma omp for
        for (int by = 0; by < Ny; ++by) {
          for (int bx = 0; bx < Nx; ++bx) {

            MAT_MAT_SHARED_BODY_0

            for (int ty = 0; ty < TL_SZ; ++ty) {
              for (int tx = 0; tx < TL_SZ; ++tx) {
                MAT_MAT_SHARED_BODY_1
              }
            }

            for (int k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

              for (int ty = 0; ty < TL_SZ; ++ty) {
                for (int tx = 0; tx < TL_SZ; ++tx) {

                  MAT_MAT_SHARED_BODY_2
                }
              }

              for (int ty = 0; ty < TL_SZ; ++ty) {
                for (int tx = 0; tx < TL_SZ; ++tx) {

                  MAT_MAT_SHARED_BODY_3
                }
              }
            }

            for (int ty = 0; ty < TL_SZ; ++ty) {
              for (int tx = 0; tx < TL_SZ; ++tx) {
                MAT_MAT_SHARED_BODY_4
              }
            }
          }
        }
      }
    }
    stopTimer();

    break;
  }

  case Lambda_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#pragma omp parallel
      {
        par_loop(0, Ny, [&](int by) {
          seq_loop(0, Nx, [&](int bx) {
            MAT_MAT_SHARED_BODY_0

            seq_loop(0, TL_SZ, [&](int ty) {
              seq_loop(0, TL_SZ, [&](int tx) { MAT_MAT_SHARED_BODY_1 });
            });

            for (int k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

              seq_loop(0, TL_SZ, [&](int ty) {
                seq_loop(0, TL_SZ, [&](int tx) { MAT_MAT_SHARED_BODY_2 });
              });

              seq_loop(0, TL_SZ, [&](int ty) {
                seq_loop(0, TL_SZ, [&](int tx) { MAT_MAT_SHARED_BODY_3 });
              });

            } // Loop across matrix

            seq_loop(0, TL_SZ, [&](int ty) {
              seq_loop(0, TL_SZ, [&](int tx) { MAT_MAT_SHARED_BODY_4 });
            });
          });
        });
      }
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      RAJA::expt::launch<omp_launch_policy>(
          RAJA::expt::HOST,
          RAJA::expt::Resources(RAJA::expt::Teams(Nx, Ny),
                                RAJA::expt::Threads(TL_SZ, TL_SZ)),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

#if 0 // TODO Enable once we update RAJA
      RAJA::expt::loop<omp_teams>(ctx, RAJA::RangeSegment(0, Ny), [&](int by) {
#else
            RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny),
                                      [&](int by) {
#endif
            RAJA::expt::loop<teams_x>(
                ctx, RAJA::RangeSegment(0, Nx), [&](int bx) {
                  MAT_MAT_SHARED_BODY_0

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, TL_SZ), [&](int ty) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, TL_SZ),
                            [&](int tx) { MAT_MAT_SHARED_BODY_1 });
                      });

                  for (int k = 0; k < (TL_SZ + N - 1) / TL_SZ; k++) {

                    RAJA::expt::loop<threads_y>(
                        ctx, RAJA::RangeSegment(0, TL_SZ),
                        [&](int ty) {
                          RAJA::expt::loop<threads_x>(
                              ctx, RAJA::RangeSegment(0, TL_SZ),
                              [&](int tx) { MAT_MAT_SHARED_BODY_2 });
                        });

                    ctx.teamSync();

                    RAJA::expt::loop<threads_y>(
                        ctx, RAJA::RangeSegment(0, TL_SZ),
                        [&](int ty) {
                          RAJA::expt::loop<threads_x>(
                              ctx, RAJA::RangeSegment(0, TL_SZ),
                              [&](int tx) { MAT_MAT_SHARED_BODY_3 });
                        });

                    ctx.teamSync();
                  }

                  RAJA::expt::loop<threads_y>(
                      ctx, RAJA::RangeSegment(0, TL_SZ), [&](int ty) {
                        RAJA::expt::loop<threads_x>(
                            ctx, RAJA::RangeSegment(0, TL_SZ),
                            [&](int tx) { MAT_MAT_SHARED_BODY_4 });
                      });
                });
          });
    }); // kernel
  }
    stopTimer();

    break;
  }

default: {
  std::cout << "\n  MAT_MAT_SHARED : Unknown variant id = " << vid << std::endl;
}
}

#endif
}

} // end namespace basic
} // end namespace rajaperf
