//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "MAT_MAT.hpp"

#include <iostream>

namespace rajaperf {
namespace basic {

void MAT_MAT::runSeqVariant(VariantID vid) {

  const Index_type run_reps = getRunReps();
  const Index_type N = m_N;

  MAT_MAT_DATA_SETUP;
  const Index_type Nx = RAJA_DIVIDE_CEILING_INT(N, MAT_MAT_TL_SZ);
  const Index_type Ny = RAJA_DIVIDE_CEILING_INT(N, MAT_MAT_TL_SZ);

  switch (vid) {

  case Base_Seq: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MAT_MAT_1");
      for (Index_type by = 0; by < Ny; ++by) {
        for (Index_type bx = 0; bx < Nx; ++bx) {

          for (Index_type ty = 0; ty < MAT_MAT_TL_SZ; ++ty) {
            for (Index_type tx = 0; tx < MAT_MAT_TL_SZ; ++tx) {

              MAT_MAT_BODY_1(MAT_MAT_TL_SZ)

              for (Index_type k = 0; k < (MAT_MAT_TL_SZ + N - 1) / MAT_MAT_TL_SZ; ++k) {
                MAT_MAT_BODY_2(MAT_MAT_TL_SZ)
              } // Sequential loop

              MAT_MAT_BODY_3(MAT_MAT_TL_SZ)
            }
          }
        }
      }
      RP_CALI_SUBKERNEL_END("MAT_MAT_1");

    } // number of iterations
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case Lambda_Seq: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MAT_MAT_1");
      auto outer_y = [&](Index_type by) {
        auto outer_x = [&](Index_type bx) {

          auto inner_y = [&](Index_type ty) {
            auto inner_x = [&](Index_type tx) {

              MAT_MAT_BODY_1(MAT_MAT_TL_SZ)

              for (Index_type k = 0; k < (MAT_MAT_TL_SZ + N - 1) / MAT_MAT_TL_SZ; ++k) {
                MAT_MAT_BODY_2(MAT_MAT_TL_SZ)
              }

              MAT_MAT_BODY_3(MAT_MAT_TL_SZ)
            };

            for (Index_type tx = 0; tx < MAT_MAT_TL_SZ; ++tx) {
              if (tx < MAT_MAT_TL_SZ)
                inner_x(tx);
            }
          };

          for (Index_type ty = 0; ty < MAT_MAT_TL_SZ; ++ty) {
            if (ty < MAT_MAT_TL_SZ)
              inner_y(ty);
          }
        }; // outer_x

        for (Index_type bx = 0; bx < Nx; ++bx) {
          outer_x(bx);
        }
      };

      for (Index_type by = 0; by < Ny; ++by) {
        outer_y(by);
      }
      RP_CALI_SUBKERNEL_END("MAT_MAT_1");
    } // irep
    stopTimer();

    break;
  }

  case RAJA_Seq: {

    auto res{getHostResource()};

    using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using outer_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MAT_MAT_1");
      //Grid is empty as the host does not need a compute grid to be specified
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_y>(ctx, RAJA::RangeSegment(0, Ny),
            [&](Index_type by) {
              RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, Nx),
                [&](Index_type bx) {

                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MAT_MAT_TL_SZ),
                    [&](Index_type ty) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MAT_MAT_TL_SZ),
                        [&](Index_type tx) {

                          MAT_MAT_BODY_1(MAT_MAT_TL_SZ)

                          for (Index_type k = 0; k < (MAT_MAT_TL_SZ + N - 1) / MAT_MAT_TL_SZ; k++) {
                            MAT_MAT_BODY_2(MAT_MAT_TL_SZ)
                          }  // for (k)

                          MAT_MAT_BODY_3(MAT_MAT_TL_SZ)
                        }
                      );  // RAJA::loop<inner_x>
                    }
                  );  // RAJA::loop<inner_y>

                }  // lambda (bx)
              );  // RAJA::loop<outer_x>
            }  // lambda (by)
          );  // RAJA::loop<outer_y>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      RP_CALI_SUBKERNEL_END("MAT_MAT_1");
    }  // loop over kernel reps
    stopTimer();

    break;
  }
#endif // RUN_RAJA_SEQ

  default: {
    getCout() << "\n  MAT_MAT : Unknown variant id = " << vid
              << std::endl;
  }
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MAT_MAT, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace basic
} // end namespace rajaperf
