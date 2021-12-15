//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-21, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJA_UNROLL

#include "DIFFUSION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {

#define MFEM_SYNC_THREAD

void DIFFUSION3DPA::runSeqVariant(VariantID vid) {
  const Index_type run_reps = getRunReps();

  DIFFUSION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      for (int e = 0; e < NE; ++e) {

        DIFFUSION3DPA_0_CPU;

        CPU_FOREACH(dz,z,DPA_D1D)
        {
          CPU_FOREACH(dy,y,DPA_D1D)
          {
            CPU_FOREACH(dx,x,DPA_D1D)
            {
               NEW_DIFFUSION3DPA_1;
            }
          }
        }
        //if (MFEM_THREAD_ID(z) == 0)
        //{
         CPU_FOREACH(dy,y,DPA_D1D)
         {
            CPU_FOREACH(qx,x,DPA_Q1D)
            {
              NEW_DIFFUSION3DPA_2;
            }
         }
         //}
      MFEM_SYNC_THREAD;
      CPU_FOREACH(dz,z,DPA_D1D)
      {
         CPU_FOREACH(dy,y,DPA_D1D)
         {
            CPU_FOREACH(qx,x,DPA_Q1D)
            {
              NEW_DIFFUSION3DPA_3;
            }
         }
      }
      MFEM_SYNC_THREAD;
      CPU_FOREACH(dz,z,DPA_D1D)
      {
         CPU_FOREACH(qy,y,DPA_Q1D)
         {
            CPU_FOREACH(qx,x,DPA_Q1D)
            {
              NEW_DIFFUSION3DPA_4;
            }
         }
      }
      MFEM_SYNC_THREAD;
      CPU_FOREACH(qz,z,DPA_Q1D)
      {
         CPU_FOREACH(qy,y,DPA_Q1D)
         {
            CPU_FOREACH(qx,x,DPA_Q1D)
            {
              NEW_DIFFUSION3DPA_5;
            }
         }
      }
      MFEM_SYNC_THREAD;
      //if (MFEM_THREAD_ID(z) == 0)
      //{
         CPU_FOREACH(d,y,DPA_D1D)
         {
            CPU_FOREACH(q,x,DPA_Q1D)
            {
              NEW_DIFFUSION3DPA_6;
            }
         }
         //}
      MFEM_SYNC_THREAD;
      CPU_FOREACH(qz,z,DPA_Q1D)
      {
         CPU_FOREACH(qy,y,DPA_Q1D)
         {
            CPU_FOREACH(dx,x,DPA_D1D)
            {
              NEW_DIFFUSION3DPA_7;
            }
         }
      }
      MFEM_SYNC_THREAD;
      CPU_FOREACH(qz,z,DPA_Q1D)
      {
         CPU_FOREACH(dy,y,DPA_D1D)
         {
            CPU_FOREACH(dx,x,DPA_D1D)
            {
              NEW_DIFFUSION3DPA_8;
            }
         }
      }
      MFEM_SYNC_THREAD;
      CPU_FOREACH(dz,z,DPA_D1D)
      {
         CPU_FOREACH(dy,y,DPA_D1D)
         {
            CPU_FOREACH(dx,x,DPA_D1D)
            {
              NEW_DIFFUSION3DPA_9;
            }
         }
      }

      } // element loop
    }
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case RAJA_Seq: {

    // Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t
#if defined(RAJA_DEVICE_ACTIVE)
                                                   ,
                                                   d3d_device_launch
#endif
                                                   >;

    using outer_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,
                                           d3d_gpu_block_x_policy
#endif
                                           >;

    using inner_x = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,
                                           d3d_gpu_thread_x_policy
#endif
                                           >;

    using inner_y = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,
                                           d3d_gpu_thread_y_policy
#endif
                                           >;

    using inner_z = RAJA::expt::LoopPolicy<RAJA::loop_exec
#if defined(RAJA_DEVICE_ACTIVE)
                                           ,
                                           d3d_gpu_thread_z_policy
#endif
                                           >;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //Grid is empty as the host does not need a compute grid to be specified
      RAJA::expt::launch<launch_policy>(
        RAJA::expt::HOST, RAJA::expt::Grid(),
        [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              DIFFUSION3DPA_0_CPU;

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                        [&](int dx) {

                          NEW_DIFFUSION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

              ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, 0),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          NEW_DIFFUSION3DPA_2;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

              ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          NEW_DIFFUSION3DPA_3;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

              ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                    [&](int qy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                        [&](int qx) {

                          NEW_DIFFUSION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                   [&](int qy) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                       [&](int qx) {

                         NEW_DIFFUSION3DPA_5;

                       } // lambda (qx)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, 0),
               [&](int dz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int d) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                       [&](int q) {

                         NEW_DIFFUSION3DPA_6;

                       } // lambda (q)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (d)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::expt::loop<inner_z>
 
             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
                   [&](int qy) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         NEW_DIFFUSION3DPA_7;

                       } // lambda (dx)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (qy)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_Q1D),
               [&](int qz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int dy) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         NEW_DIFFUSION3DPA_8;

                       } // lambda (dx)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (qz)
             );  //RAJA::expt::loop<inner_z>

             ctx.teamSync();

             RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, DPA_D1D),
               [&](int dz) {
                 RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                   [&](int dy) {
                     RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, DPA_D1D),
                       [&](int dx) {

                         NEW_DIFFUSION3DPA_9;

                       } // lambda (dx)
                     ); // RAJA::expt::loop<inner_x>
                   } // lambda (dy)
                 );  //RAJA::expt::loop<inner_y>
               } // lambda (dz)
             );  //RAJA::expt::loop<inner_z>

            } // lambda (e)
          ); // RAJA::expt::loop<outer_x>

        } // outer lambda (ctx)
      ); // RAJA::expt::launch
    } // loop over kernel reps
    stopTimer();

    return;
  }
#endif // RUN_RAJA_SEQ

  default:
    getCout() << "\n DIFFUSION3DPA : Unknown Seq variant id = " << vid
              << std::endl;
  }
}

} // end namespace apps
} // end namespace rajaperf
