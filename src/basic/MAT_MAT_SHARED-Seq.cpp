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
inline void seq_loop(const Index_type st, const Index_type end, BODY const &body) {
  for (Index_type i = st; i < end; ++i) {
    body(i);
  }
}

void MAT_MAT_SHARED::runSeqVariant(VariantID vid) {

  const Index_type run_reps = getRunReps();
  const Index_type N = getRunSize();

  MAT_MAT_SHARED_DATA_SETUP;
  const Index_type Nx = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);
  const Index_type Ny = RAJA_DIVIDE_CEILING_INT(N, TL_SZ);

  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (Index_type by = 0; by < Ny; ++by) {
        for (Index_type bx = 0; bx < Nx; ++bx) {

          MAT_MAT_SHARED_BODY_0

          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
              MAT_MAT_SHARED_BODY_1
            }
          }

          for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
                MAT_MAT_SHARED_BODY_2
              }
            }

            for (Index_type ty = 0; ty < TL_SZ; ++ty) {
              for (Index_type tx = 0; tx < TL_SZ; ++tx) {
                MAT_MAT_SHARED_BODY_3
              }
            }

          } // Sequential loop

          for (Index_type ty = 0; ty < TL_SZ; ++ty) {
            for (Index_type tx = 0; tx < TL_SZ; ++tx) {
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

      seq_loop(0, Ny, [&](Index_type by) {
        seq_loop(0, Nx, [&](Index_type bx) {
          MAT_MAT_SHARED_BODY_0

          seq_loop(0, TL_SZ, [&](Index_type ty) {
            seq_loop(0, TL_SZ, [&](Index_type tx) { MAT_MAT_SHARED_BODY_1 });
          });

          for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; ++k) {

            seq_loop(0, TL_SZ, [&](Index_type ty) {
              seq_loop(0, TL_SZ, [&](Index_type tx) { MAT_MAT_SHARED_BODY_2 });
            });

            seq_loop(0, TL_SZ, [&](Index_type ty) {
              seq_loop(0, TL_SZ, [&](Index_type tx) { MAT_MAT_SHARED_BODY_3 });
            });

          }

          seq_loop(0, TL_SZ, [&](Index_type ty) {
            seq_loop(0, TL_SZ, [&](Index_type tx) { MAT_MAT_SHARED_BODY_4 });
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

            RAJA::expt::loop<teams_y>(ctx, RAJA::RangeSegment(0, Ny), [&](Index_type by) {
              RAJA::expt::loop<teams_x>(ctx, RAJA::RangeSegment(0, Nx), [&](Index_type bx) {

                        MAT_MAT_SHARED_BODY_0

              RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type ty) {
                RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type tx) {
                    MAT_MAT_SHARED_BODY_1
                });
              });

              for (Index_type k = 0; k < (TL_SZ + N - 1) / TL_SZ; k++) {

               RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type ty) {
                 RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type tx) {
                     MAT_MAT_SHARED_BODY_2
                 });
               });

               ctx.teamSync();

               RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type ty) {
                 RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type tx) {
                     MAT_MAT_SHARED_BODY_3
                       });
                 });

               ctx.teamSync();
              }

              RAJA::expt::loop<threads_y>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type ty) {
                RAJA::expt::loop<threads_x>(ctx, RAJA::RangeSegment(0, TL_SZ), [&](Index_type tx) {
                    MAT_MAT_SHARED_BODY_4
                });
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
