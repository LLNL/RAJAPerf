//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

void MASS3DEA::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  MASS3DEA_DATA_SETUP;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
      for (Index_type e = 0; e < NE; ++e) {

        MASS3DEA_0_CPU

        CPU_FOREACH(d, x, mea::D1D) {
          CPU_FOREACH(q, y, mea::Q1D) {
            MASS3DEA_1
          }
        }

        MASS3DEA_2_CPU

        CPU_FOREACH(k1, x, mea::Q1D) {
          CPU_FOREACH(k2, y, mea::Q1D) {
            CPU_FOREACH(k3, z, mea::Q1D) {
              MASS3DEA_3
            }
          }
        }

        CPU_FOREACH(i1, x, mea::D1D) {
          CPU_FOREACH(i2, y, mea::D1D) {
            CPU_FOREACH(i3, z, mea::D1D) {
              MASS3DEA_4
            }
          }
        }

      } // element loop
      RP_CALI_SUBKERNEL_END("MASS3DEA_1");
    }
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case RAJA_Seq: {

    auto res{getHostResource()};

    // Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_z = RAJA::LoopPolicy<RAJA::seq_exec>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DEA_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

            RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
              [&](Index_type e) {

                  MASS3DEA_0

                  RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                    [&](Index_type ) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),
                        [&](Index_type d) {
                          RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                            [&](Index_type q) {
                              MASS3DEA_1
                            }
                          ); // RAJA::loop<inner_y>
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_z>

                  MASS3DEA_2

                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                    [&](Index_type k1) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                        [&](Index_type k2) {
                          RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::Q1D),
                            [&](Index_type k3) {
                              MASS3DEA_3
                            }
                          ); // RAJA::loop<inner_x>
                        }
                      ); // RAJA::loop<inner_y>
                    }
                  ); // RAJA::loop<inner_z>

                  ctx.teamSync();

                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mea::D1D),
                    [&](Index_type i1) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mea::D1D),
                        [&](Index_type i2) {
                          RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mea::D1D),
                            [&](Index_type i3) {
                              MASS3DEA_4
                            }
                          ); // RAJA::loop<inner_x>
                        }
                      ); // RAJA::loop<inner_y>
                    }
                  ); // RAJA::loop<inner_z>

                } // lambda (e)
            );    // RAJA::loop<outer_x>
          }       // outer lambda (ctx)
      );          // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("MASS3DEA_1");

    } // loop over kernel reps
    stopTimer();

    return;
  }
#endif // RUN_RAJA_SEQ

  default:
    getCout() << "\n MASS3DEA : Unknown Seq variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MASS3DEA, Seq, Base_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
