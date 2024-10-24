//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

// Uncomment to add compiler directives loop unrolling
//#define USE_RAJAPERF_UNROLL

#include "MASS3DEA.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf {
namespace apps {


void MASS3DEA::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx)) {

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  MASS3DEA_DATA_SETUP;

  switch (vid) {

  case Base_OpenMP: {

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

#pragma omp parallel for
      for (int e = 0; e < NE; ++e) {

        MASS3DEA_0_CPU

        CPU_FOREACH(d, x, MEA_D1D) {
          CPU_FOREACH(q, y, MEA_Q1D) {
            MASS3DEA_1
          }
        }

        MASS3DEA_2_CPU

        CPU_FOREACH(k1, x, MEA_Q1D) {
          CPU_FOREACH(k2, y, MEA_Q1D) {
            CPU_FOREACH(k3, z, MEA_Q1D) {
              MASS3DEA_3
            }
          }
        }

        CPU_FOREACH(i1, x, MEA_D1D) {
          CPU_FOREACH(i2, y, MEA_D1D) {
            CPU_FOREACH(i3, z, MEA_D1D) {
              MASS3DEA_4
            }
          }
        }

      } // element loop
    }
    stopTimer();

    break;
  }

  case RAJA_OpenMP: {

    auto res{getHostResource()};

    //Currently Teams requires two policies if compiled with a device
    using launch_policy = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

    using outer_x = RAJA::LoopPolicy<RAJA::omp_for_exec>;

    using inner_x = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_y = RAJA::LoopPolicy<RAJA::seq_exec>;

    using inner_z = RAJA::LoopPolicy<RAJA::seq_exec>;

    startTimer();
    for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

      //Grid is empty as the host does not need a compute grid to be specified
      RAJA::launch<launch_policy>( res,
        RAJA::LaunchParams(),
        [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

          RAJA::loop<outer_x>(ctx, RAJA::RangeSegment(0, NE),
            [&](int e) {

              MASS3DEA_0

              RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, 1),
                [&](int ) {
                  RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                    [&](int d) {
                      RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                        [&](int q) {
                          MASS3DEA_1
                        }
                      ); // RAJA::loop<inner_y>
                    }
                  ); // RAJA::loop<inner_x>
                }
              ); // RAJA::loop<inner_z>

              MASS3DEA_2

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                [&](int k1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                    [&](int k2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, MEA_Q1D),
                        [&](int k3) {
                          MASS3DEA_3
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

              ctx.teamSync();

              RAJA::loop<inner_x>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                [&](int i1) {
                  RAJA::loop<inner_y>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                    [&](int i2) {
                      RAJA::loop<inner_z>(ctx, RAJA::RangeSegment(0, MEA_D1D),
                        [&](int i3) {
                          MASS3DEA_4
                        }
                      ); // RAJA::loop<inner_x>
                    }
                  ); // RAJA::loop<inner_y>
                }
              ); // RAJA::loop<inner_z>

            } // lambda (e)
          );    // RAJA::loop<outer_x>

        }  // outer lambda (ctx)
      );  // // RAJA::launch

    }  // loop over kernel reps
    stopTimer();

    return;
  }

  default:
    getCout() << "\n MASS3DEA : Unknown OpenMP variant id = " << vid
              << std::endl;
  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

} // end namespace apps
} // end namespace rajaperf
