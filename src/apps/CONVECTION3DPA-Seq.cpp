//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-22, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives for loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "CONVECTION3DPA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

#define MFEM_SHARED
#define MFEM_SYNC_THREAD

namespace rajaperf {
namespace apps {

void CONVECTION3DPA::runSeqVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {
  const Index_type run_reps = getRunReps();

  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_Seq: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {
#if 0
      for (int e = 0; e < NE; ++e) {

        constexpr int max_D1D = CPA_D1D;
        constexpr int max_Q1D = CPA_Q1D;
        constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;
        MFEM_SHARED double sm0[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm1[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm2[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm3[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm4[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm5[max_DQ*max_DQ*max_DQ];

        double (*u)[max_D1D][max_D1D] = (double (*)[max_D1D][max_D1D]) sm0;
        CPU_FOREACH(dz,z,CPA_D1D)
        {
          CPU_FOREACH(dy,y,CPA_D1D)
          {
            CPU_FOREACH(dx,x,CPA_D1D)
            {
              u[dz][dy][dx] = cpaX_(dx,dy,dz,e);
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*Bu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm1;
        double (*Gu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm2;
        CPU_FOREACH(dz,z,CPA_D1D)
        {
          CPU_FOREACH(dy,y,CPA_D1D)
          {
            CPU_FOREACH(qx,x,CPA_Q1D)
            {
              double Bu_ = 0.0;
              double Gu_ = 0.0;
              for (int dx = 0; dx < CPA_D1D; ++dx)
              {
                const double bx = cpa_B(qx,dx);
                const double gx = cpa_G(qx,dx);
                const double x = u[dz][dy][dx];
                Bu_ += bx * x;
                Gu_ += gx * x;
              }
              Bu[dz][dy][qx] = Bu_;
              Gu[dz][dy][qx] = Gu_;
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*BBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
        double (*GBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
        double (*BGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm5;
        CPU_FOREACH(dz,z,CPA_D1D)
        {
          CPU_FOREACH(qx,x,CPA_Q1D)
          {
            CPU_FOREACH(qy,y,CPA_Q1D)
            {
              double BBu_ = 0.0;
              double GBu_ = 0.0;
              double BGu_ = 0.0;
              for (int dy = 0; dy < CPA_D1D; ++dy)
              {
                const double bx = cpa_B(qy,dy);
                const double gx = cpa_G(qy,dy);
                BBu_ += bx * Bu[dz][dy][qx];
                GBu_ += gx * Bu[dz][dy][qx];
                BGu_ += bx * Gu[dz][dy][qx];
              }
              BBu[dz][qy][qx] = BBu_;
              GBu[dz][qy][qx] = GBu_;
              BGu[dz][qy][qx] = BGu_;
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*GBBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm0;
        double (*BGBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm1;
        double (*BBGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm2;
        CPU_FOREACH(qx,x,CPA_Q1D)
        {
          CPU_FOREACH(qy,y,CPA_Q1D)
          {
            CPU_FOREACH(qz,z,CPA_Q1D)
            {
              double GBBu_ = 0.0;
              double BGBu_ = 0.0;
              double BBGu_ = 0.0;
              for (int dz = 0; dz < CPA_D1D; ++dz)
              {
                const double bx = cpa_B(qz,dz);
                const double gx = cpa_G(qz,dz);
                GBBu_ += gx * BBu[dz][qy][qx];
                BGBu_ += bx * GBu[dz][qy][qx];
                BBGu_ += bx * BGu[dz][qy][qx];
              }
              GBBu[qz][qy][qx] = GBBu_;
              BGBu[qz][qy][qx] = BGBu_;
              BBGu[qz][qy][qx] = BBGu_;
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*DGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
        CPU_FOREACH(qz,z,CPA_Q1D)
        {
          CPU_FOREACH(qy,y,CPA_Q1D)
          {
            CPU_FOREACH(qx,x,CPA_Q1D)
            {
              const double O1 = cpa_op(qx,qy,qz,0,e);
              const double O2 = cpa_op(qx,qy,qz,1,e);
              const double O3 = cpa_op(qx,qy,qz,2,e);
              
              const double gradX = BBGu[qz][qy][qx];
              const double gradY = BGBu[qz][qy][qx];
              const double gradZ = GBBu[qz][qy][qx];
              
              DGu[qz][qy][qx] = (O1 * gradX) + (O2 * gradY) + (O3 * gradZ);
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*BDGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
        CPU_FOREACH(qx,x,CPA_Q1D)
        {
          CPU_FOREACH(qy,y,CPA_Q1D)
          {
            CPU_FOREACH(dz,z,CPA_D1D)
            {
               double BDGu_ = 0.0;
               for (int qz = 0; qz < CPA_Q1D; ++qz)
               {
                  const double w = cpa_Bt(dz,qz);
                  BDGu_ += w * DGu[qz][qy][qx];
               }
               BDGu[dz][qy][qx] = BDGu_;
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*BBDGu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm5;
        CPU_FOREACH(dz,z,CPA_D1D)
        {
           CPU_FOREACH(qx,x,CPA_Q1D)
           {
              CPU_FOREACH(dy,y,CPA_D1D)
              {
                 double BBDGu_ = 0.0;
                 for (int qy = 0; qy < CPA_Q1D; ++qy)
                 {
                   const double w = cpa_Bt(dy,qy);
                   BBDGu_ += w * BDGu[dz][qy][qx];
                }
                BBDGu[dz][dy][qx] = BBDGu_;
             }
          }
        }
        MFEM_SYNC_THREAD;
        CPU_FOREACH(dz,z,CPA_D1D)
        {
          CPU_FOREACH(dy,y,CPA_D1D)
          {
            CPU_FOREACH(dx,x,CPA_D1D)
            {
              double BBBDGu = 0.0;
              for (int qx = 0; qx < CPA_Q1D; ++qx)
              {
                const double w = cpa_Bt(dx,qx);
                BBBDGu += w * BBDGu[dz][dy][qx];
              }
              cpaY_(dx,dy,dz,e) += BBBDGu;
            }
          }
        }
      } // element loop
#else

      for(int e = 0; e < NE; ++e) {

        constexpr int max_D1D = CPA_D1D;
        constexpr int max_Q1D = CPA_Q1D;
        constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;
        MFEM_SHARED double sm0[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm1[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm2[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm3[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm4[max_DQ*max_DQ*max_DQ];
        MFEM_SHARED double sm5[max_DQ*max_DQ*max_DQ];

        double (*u)[max_D1D][max_D1D] = (double (*)[max_D1D][max_D1D]) sm0;
        for(int dz = 0; dz < CPA_D1D; ++dz)
        {
          for(int dy = 0; dy < CPA_D1D; ++dy)
          {
            for(int dx = 0; dx < CPA_D1D; ++dx)
            {
              u[dz][dy][dx] = cpaX_(dx,dy,dz,e);
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*Bu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm1;
        double (*Gu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm2;
        for(int dz = 0; dz < CPA_D1D; ++dz)
        {
          for(int dy = 0; dy < CPA_D1D; ++dy)
          {
            for(int qx = 0; qx < CPA_Q1D; ++qx)
            {
              double Bu_ = 0.0;
              double Gu_ = 0.0;
              for(int dx = 0; dx < CPA_D1D; ++dx)
              {
                const double bx = cpa_B(qx,dx);
                const double gx = cpa_G(qx,dx);
                const double x = u[dz][dy][dx];
                Bu_ += bx * x;
                Gu_ += gx * x;
              }
              Bu[dz][dy][qx] = Bu_;
              Gu[dz][dy][qx] = Gu_;
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*BBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
        double (*GBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
        double (*BGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm5;
        for(int dz = 0; dz < CPA_D1D; ++dz)
        {
          for(int qx = 0; qx < CPA_Q1D; ++qx)
          {
            for(int qy = 0; qy < CPA_Q1D; ++qy)
            {
              double BBu_ = 0.0;
              double GBu_ = 0.0;
              double BGu_ = 0.0;
              for(int dy = 0; dy < CPA_D1D; ++dy)
              {
                const double bx = cpa_B(qy,dy);
                const double gx = cpa_G(qy,dy);
                BBu_ += bx * Bu[dz][dy][qx];
                GBu_ += gx * Bu[dz][dy][qx];
                BGu_ += bx * Gu[dz][dy][qx];
              }
              BBu[dz][qy][qx] = BBu_;
              GBu[dz][qy][qx] = GBu_;
              BGu[dz][qy][qx] = BGu_;
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*GBBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm0;
        double (*BGBu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm1;
        double (*BBGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm2;
        for(int qx = 0; qx < CPA_Q1D; ++qx)
        {
          for(int qy = 0; qy < CPA_Q1D; ++qy)
          {
            for(int qz = 0; qz < CPA_Q1D; ++qz)
            {
              double GBBu_ = 0.0;
              double BGBu_ = 0.0;
              double BBGu_ = 0.0;
              for(int dz = 0; dz < CPA_D1D; ++dz)
              {
                const double bx = cpa_B(qz,dz);
                const double gx = cpa_G(qz,dz);
                GBBu_ += gx * BBu[dz][qy][qx];
                BGBu_ += bx * GBu[dz][qy][qx];
                BBGu_ += bx * BGu[dz][qy][qx];
              }
              GBBu[qz][qy][qx] = GBBu_;
              BGBu[qz][qy][qx] = BGBu_;
              BBGu[qz][qy][qx] = BBGu_;
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*DGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm3;
        for(int qz = 0; qz < CPA_Q1D; ++qz)
        {
          for(int qy = 0; qy < CPA_Q1D; ++qy)
          {
            for(int qx = 0; qx < CPA_Q1D; ++qx)
            {
              const double O1 = cpa_op(qx,qy,qz,0,e);
              const double O2 = cpa_op(qx,qy,qz,1,e);
              const double O3 = cpa_op(qx,qy,qz,2,e);
              
              const double gradX = BBGu[qz][qy][qx];
              const double gradY = BGBu[qz][qy][qx];
              const double gradZ = GBBu[qz][qy][qx];
              
              DGu[qz][qy][qx] = (O1 * gradX) + (O2 * gradY) + (O3 * gradZ);
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*BDGu)[max_Q1D][max_Q1D] = (double (*)[max_Q1D][max_Q1D])sm4;
        for(int qx = 0; qx < CPA_Q1D; ++qx)
        {
          for(int qy = 0; qy < CPA_Q1D; ++qy)
          {
            for(int dz = 0; dz < CPA_D1D; ++dz)
            {
               double BDGu_ = 0.0;
               for(int qz = 0; qz < CPA_Q1D; ++qz)
               {
                  const double w = cpa_Bt(dz,qz);
                  BDGu_ += w * DGu[qz][qy][qx];
               }
               BDGu[dz][qy][qx] = BDGu_;
            }
          }
        }
        MFEM_SYNC_THREAD;
        double (*BBDGu)[max_D1D][max_Q1D] = (double (*)[max_D1D][max_Q1D])sm5;
        for(int dz = 0; dz < CPA_D1D; ++dz)
        {
          for(int qx = 0; qx < CPA_Q1D; ++qx)
           {
             for(int dy = 0; dy < CPA_D1D; ++dy)
              {
                 double BBDGu_ = 0.0;
                 for(int qy = 0; qy < CPA_Q1D; ++qy)
                 {
                   const double w = cpa_Bt(dy,qy);
                   BBDGu_ += w * BDGu[dz][qy][qx];
                }
                BBDGu[dz][dy][qx] = BBDGu_;
             }
          }
        }
        MFEM_SYNC_THREAD;
        for(int dz = 0; dz < CPA_D1D; ++dz)
        {
          for(int dy = 0; dy < CPA_D1D; ++dy)
          {
            for(int dx = 0; dx < CPA_D1D; ++dx)
            {
              double BBBDGu = 0.0;
              for(int qx = 0; qx < CPA_Q1D; ++qx)
              {
                const double w = cpa_Bt(dx,qx);
                BBBDGu += w * BBDGu[dz][dy][qx];
              }
              cpaY_(dx,dy,dz,e) += BBBDGu;
            }
          }
        }
      } // element loop

#endif
    }
    stopTimer();

    break;
  }

#if defined(RUN_RAJA_SEQ)
  case RAJA_Seq: {

    // Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::seq_launch_t>;

    using outer_x = RAJA::expt::LoopPolicy<RAJA::loop_exec>;

    using inner_x = RAJA::expt::LoopPolicy<RAJA::loop_exec>;

    using inner_y = RAJA::expt::LoopPolicy<RAJA::loop_exec>;

    using inner_z = RAJA::expt::LoopPolicy<RAJA::loop_exec>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      // Grid is empty as the host does not need a compute grid to be specified
      RAJA::expt::launch<launch_policy>(
          RAJA::expt::Grid(),
          [=] RAJA_HOST_DEVICE(RAJA::expt::LaunchContext ctx) {

          RAJA::expt::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

             CONVECTION3DPA_0_CPU;
#if 1
              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dx) {

                          CONVECTION3DPA_1;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

              ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qx) {

                          CONVECTION3DPA_2;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qy) {

                          CONVECTION3DPA_3;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qx) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qz) {

                          CONVECTION3DPA_4;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                        [&](int qx) {

                          CONVECTION3DPA_5;

                        } // lambda (qx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                [&](int qx) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qy) {
                      RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dz) {

                          CONVECTION3DPA_6;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (qz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                    [&](int dy) {
                      RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dx) {

                          CONVECTION3DPA_8;

                        } // lambda (dx)
                      ); // RAJA::expt::loop<inner_x>
                    } // lambda (dy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (dz)
              );  //RAJA::expt::loop<inner_z>
#endif
            }
         );
        }  // outer lambda (ctx)
      );  // RAJA::expt::launch
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

} // end namespace apps
} // end namespace rajaperf
