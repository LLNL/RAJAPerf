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

namespace rajaperf {
namespace apps {

void CONVECTION3DPA::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  CONVECTION3DPA_DATA_SETUP;

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#pragma omp parallel for
      for (int e = 0; e < NE; ++e) {

        constexpr int max_D1D = CPA_D1D;
        constexpr int max_Q1D = CPA_Q1D;
        constexpr int max_DQ = (max_Q1D > max_D1D) ? max_Q1D : max_D1D;
        double sm0[max_DQ*max_DQ*max_DQ];
        double sm1[max_DQ*max_DQ*max_DQ];
        double sm2[max_DQ*max_DQ*max_DQ];
        double sm3[max_DQ*max_DQ*max_DQ];
        double sm4[max_DQ*max_DQ*max_DQ];
        double sm5[max_DQ*max_DQ*max_DQ];

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
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    using launch_policy = RAJA::expt::LaunchPolicy<RAJA::expt::omp_launch_t>;

    using outer_x = RAJA::expt::LoopPolicy<RAJA::omp_for_exec>;

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

                        } // lambda (dy)
                      ); // RAJA::expt::loop<inner_y>
                    } // lambda (dx)
                  );  //RAJA::expt::loop<inner_x>
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

                        } // lambda (qz)
                      ); // RAJA::expt::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::expt::loop<inner_x>

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

                        } // lambda (dz)
                      ); // RAJA::expt::loop<inner_z>
                    } // lambda (qy)
                  );  //RAJA::expt::loop<inner_y>
                } // lambda (qx)
              );  //RAJA::expt::loop<inner_x>

            ctx.teamSync();

              RAJA::expt::loop<inner_z>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                [&](int dz) {
                  RAJA::expt::loop<inner_x>(ctx, RAJA::RangeSegment(0, CPA_Q1D),
                    [&](int qx) {
                      RAJA::expt::loop<inner_y>(ctx, RAJA::RangeSegment(0, CPA_D1D),
                        [&](int dy) {

                          CONVECTION3DPA_7;

                        } // lambda (dy)
                      ); // RAJA::expt::loop<inner_y>
                    } // lambda (qx)
                  );  //RAJA::expt::loop<inner_x>
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

            } // lambda (e)
          ); // RAJA::expt::loop<outer_x>

        }  // outer lambda (ctx)
      );  // RAJA::expt::launch
    }  // loop over kernel reps
    stopTimer();

    return;
  }

  default:
    getCout() << "\n CONVECTION3DPA : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf
