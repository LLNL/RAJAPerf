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

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

void DIFFUSION3DPA::runSeqVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("DIFFUSION3DPA_1");
      for (Index_type e = 0; e < NE; ++e) {

        DIFFUSION3DPA_0_CPU;

        CPU_FOREACH(dz,z,diff::D1D) {
          CPU_FOREACH(dy,y,diff::D1D) {
            CPU_FOREACH(dx,x,diff::D1D) {
              DIFFUSION3DPA_1
            }
          }
        }

        CPU_FOREACH(dy,y,diff::D1D) {
          CPU_FOREACH(qx,x,diff::Q1D) {
            DIFFUSION3DPA_2
          }
        }

        CPU_FOREACH(dz,z,diff::D1D) {
          CPU_FOREACH(dy,y,diff::D1D) {
            CPU_FOREACH(qx,x,diff::Q1D) {

              DIFFUSION3DPA_3
            }
          }
        }

        CPU_FOREACH(dz,z,diff::D1D) {
          CPU_FOREACH(qy,y,diff::Q1D) {
            CPU_FOREACH(qx,x,diff::Q1D) {
              DIFFUSION3DPA_4
            }
          }
        }

        CPU_FOREACH(qz,z,diff::Q1D) {
          CPU_FOREACH(qy,y,diff::Q1D) {
            CPU_FOREACH(qx,x,diff::Q1D) {
              DIFFUSION3DPA_5
            }
          }
        }

        CPU_FOREACH(dy,y,diff::D1D) {
          CPU_FOREACH(qx,x,diff::Q1D) {
            DIFFUSION3DPA_6
          }
        }

        CPU_FOREACH(qz,z,diff::Q1D) {
          CPU_FOREACH(qy,y,diff::Q1D) {
            CPU_FOREACH(dx,x,diff::D1D) {
              DIFFUSION3DPA_7
            }
          }
        }

        CPU_FOREACH(qz,z,diff::Q1D) {
          CPU_FOREACH(dy,y,diff::D1D) {
            CPU_FOREACH(dx,x,diff::D1D) {
              DIFFUSION3DPA_8
            }
          }
        }

        CPU_FOREACH(dz,z,diff::D1D) {
          CPU_FOREACH(dy,y,diff::D1D) {
            CPU_FOREACH(dx,x,diff::D1D) {
              DIFFUSION3DPA_9
            }
          }
        }

      } // element loop
      RP_CALI_SUBKERNEL_END("DIFFUSION3DPA_1");
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

      RP_CALI_SUBKERNEL_BEGIN("DIFFUSION3DPA_1");
      // Grid is empty as the host does not need a compute grid to be specified
      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

              DIFFUSION3DPA_0_CPU;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                        [&](Index_type dx) {

                          DIFFUSION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](Index_type RAJA_UNUSED_ARG(dz)) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_3;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                        [&](Index_type qx) {

                          DIFFUSION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                         DIFFUSION3DPA_5;

                       } // lambda (qx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
               [&](Index_type RAJA_UNUSED_ARG(dz)) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                       [&](Index_type qx) {

                         DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::Q1D),
                   [&](Index_type qy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_7;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::Q1D),
               [&](Index_type qz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_8;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::loop<inner_z>

             ctx.teamSync();

             RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, diff::D1D),
               [&](Index_type dz) {
                 RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, diff::D1D),
                   [&](Index_type dy) {
                     RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, diff::D1D),
                       [&](Index_type dx) {

                         DIFFUSION3DPA_9;

                       } // lambda (dx)
                     ); // RAJA::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::loop<inner_z>

            } // lambda (e)
          ); // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::launch
      //clang-format on
      RP_CALI_SUBKERNEL_END("DIFFUSION3DPA_1");
    }  // loop over kernel reps
    stopTimer();

    return;
  }
#endif // RUN_RAJA_SEQ

  default:
    getCout() << "\n DIFFUSION3DPA : Unknown Seq variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(DIFFUSION3DPA, Seq, Base_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
