//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

void CONVECTION3DPA::runSeqVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("CONVECTION3DPA_1");
      for (Index_type e = 0; e < NE; ++e) {

        CONVECTION3DPA_0_CPU;

        CPU_FOREACH(dz,z,conv::D1D)
        {
          CPU_FOREACH(dy,y,conv::D1D)
          {
            CPU_FOREACH(dx,x,conv::D1D)
            {
              CONVECTION3DPA_1;
            }
          }
        }

        CPU_FOREACH(dz,z,conv::D1D)
        {
          CPU_FOREACH(dy,y,conv::D1D)
          {
            CPU_FOREACH(qx,x,conv::Q1D)
            {
              CONVECTION3DPA_2;
            }
          }
        }

        CPU_FOREACH(dz,z,conv::D1D)
        {
          CPU_FOREACH(qx,x,conv::Q1D)
          {
            CPU_FOREACH(qy,y,conv::Q1D)
            {
              CONVECTION3DPA_3;
            }
          }
        }

        CPU_FOREACH(qx,x,conv::Q1D)
        {
          CPU_FOREACH(qy,y,conv::Q1D)
          {
            CPU_FOREACH(qz,z,conv::Q1D)
            {
              CONVECTION3DPA_4;
            }
          }
        }

        CPU_FOREACH(qz,z,conv::Q1D)
        {
          CPU_FOREACH(qy,y,conv::Q1D)
          {
            CPU_FOREACH(qx,x,conv::Q1D)
            {
              CONVECTION3DPA_5;
            }
          }
        }

        CPU_FOREACH(qx,x,conv::Q1D)
        {
          CPU_FOREACH(qy,y,conv::Q1D)
          {
            CPU_FOREACH(dz,z,conv::D1D)
            {
              CONVECTION3DPA_6;
            }
          }
        }

        CPU_FOREACH(dz,z,conv::D1D)
        {
           CPU_FOREACH(qx,x,conv::Q1D)
           {
              CPU_FOREACH(dy,y,conv::D1D)
              {
                CONVECTION3DPA_7;
             }
          }
        }

        CPU_FOREACH(dz,z,conv::D1D)
        {
          CPU_FOREACH(dy,y,conv::D1D)
          {
            CPU_FOREACH(dx,x,conv::D1D)
            {
              CONVECTION3DPA_8;
            }
          }
        }
      } // element loop
      RP_CALI_SUBKERNEL_END("CONVECTION3DPA_1");

    }
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case RAJA_Seq: {

    auto res{getHostResource()};

    using launch_policy = RAJA::LaunchPolicy<RAJA::seq_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_z = RAJA::LoopPolicy<RAJA::seq_exec>;

    startTimer();
    // Loop counter increment uses macro to quiet C++20 compiler warning
    for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

      RP_CALI_SUBKERNEL_BEGIN("CONVECTION3DPA_1");
      // Grid is empty as the host does not need a compute grid to be specified
      //clang-format off
      RAJA::launch<launch_policy>( res,
          RAJA::LaunchParams(),
          [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](Index_type e) {

             CONVECTION3DPA_0_CPU;

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dx) {

                          CONVECTION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qx) {

                          CONVECTION3DPA_2;

                        } // lambda (dx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qy) {

                          CONVECTION3DPA_3;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (dx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qz) {

                          CONVECTION3DPA_4;

                        } // lambda (qz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                        [&](Index_type qx) {

                          CONVECTION3DPA_5;

                        } // lambda (qx)
                      ); // RAJA::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                [&](Index_type qx) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qy) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dz) {

                          CONVECTION3DPA_6;

                        } // lambda (dz)
                      ); // RAJA::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::loop<inner_x>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::Q1D),
                    [&](Index_type qx) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dy)
                      ); // RAJA::loop<inner_y>
                    } // lambda (qx)
                  );  //RAJA::loop<inner_x>
                } // lambda (dz)
              );  //RAJA::loop<inner_z>

            ctx.teamSync();

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, conv::D1D),
                [&](Index_type dz) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, conv::D1D),
                    [&](Index_type dy) {
                      RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, conv::D1D),
                        [&](Index_type dx) {

                          CONVECTION3DPA_8;

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
      //clang-format off
      RP_CALI_SUBKERNEL_END("CONVECTION3DPA_1");
    }  // loop over kernel reps
    stopTimer();

    return;
  }
#endif // RUN_RAJA_SEQ

  default:
    getCout() << "\n CONVECTION3DPA : Unknown Seq variant id = " << vid
              << std::endl;
  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(CONVECTION3DPA, Seq, Base_Seq, RAJA_Seq)

} // end namespace apps
} // end namespace rajaperf
