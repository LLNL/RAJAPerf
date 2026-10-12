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

#include "MASS3DPA_ATOMIC.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {


void MASS3DPA_ATOMIC::runSeqVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  MASS3DPA_ATOMIC_DATA_SETUP;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep))
    {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_ATOMIC_1");
      for (Index_type e = 0; e < NE; ++e) {

        MASS3DPA_ATOMIC_0_CPU;

        SHARED_LOOP_3D(dx, dy, dz, mpa_at::D1D, mpa_at::D1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_1;
        }

        SHARED_LOOP_2D(q, d, mpa_at::Q1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_2;
        }

        SHARED_LOOP_3D(qx, dy, dz, mpa_at::Q1D, mpa_at::D1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_3;
        }

        SHARED_LOOP_3D(qx, qy, dz, mpa_at::Q1D, mpa_at::Q1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_4;
        }

        SHARED_LOOP_3D(qx, qy, qz, mpa_at::Q1D, mpa_at::Q1D, mpa_at::Q1D) {
          MASS3DPA_ATOMIC_5;
        }

        SHARED_LOOP_3D(dx, qy, qz, mpa_at::D1D, mpa_at::Q1D, mpa_at::Q1D) {
          MASS3DPA_ATOMIC_6;
        }

        SHARED_LOOP_3D(dx, dy, qz, mpa_at::D1D, mpa_at::D1D, mpa_at::Q1D) {
          MASS3DPA_ATOMIC_7;
        }

        SHARED_LOOP_3D(dx, dy, dz, mpa_at::D1D, mpa_at::D1D, mpa_at::D1D) {
          MASS3DPA_ATOMIC_8;
          MASS3DPA_ATOMIC_9(RAJAPERF_ATOMIC_ADD_SEQ);
        }

      } // element loop
      RP_CALI_SUBKERNEL_END("MASS3DPA_ATOMIC_1");

    }
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case RAJA_Seq: {

    auto res{getHostResource()};

    //Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_z = RAJA::LoopPolicy<RAJA::seq_exec>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("MASS3DPA_ATOMIC_1");
      //clang-format off
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {


            MASS3DPA_ATOMIC_0_CPU;

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_1;
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
              [&](Index_type ) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type d) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type q) {
                      MASS3DPA_ATOMIC_2;
                      } // lambda (q)
                    ); // RAJA::loop<inner_x>
                  } // lambda (d)
                ); // RAJA::loop<inner_y>
              } // lambda ()
            ); // RAJA::loop<inner_z>
            ctx.teamSync();


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_3;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_4;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                      [&](Index_type qx) {
                      MASS3DPA_ATOMIC_5;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (qz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
                  [&](Index_type qy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_6;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (qy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();

            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::Q1D),
              [&](Index_type qz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_7;
                      } // lambda (qx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>
            ctx.teamSync();


            RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
              [&](Index_type dz) {
                RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                  [&](Index_type dy) {
                    RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, mpa_at::D1D),
                      [&](Index_type dx) {
                      MASS3DPA_ATOMIC_8;
                      MASS3DPA_ATOMIC_9(RAJAPERF_ATOMIC_ADD_RAJA_SEQ);
                      } // lambda (dx)
                    ); // RAJA::loop<inner_x>
                  } // lambda (dy)
                ); // RAJA::loop<inner_y>
              } // lambda (dz)
            ); // RAJA::loop<inner_z>


            }  // lambda (e)
          );  // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("MASS3DPA_ATOMIC_1");

    }  // loop over kernel reps
    stopTimer();

    return;
  }
#endif // RUN_RAJA_SEQ

  default:
    getCout() << "\n MASS3DPA_ATOMIC : Unknown Seq variant id = " << vid << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(MASS3DPA_ATOMIC, Seq, Base_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
