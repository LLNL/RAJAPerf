//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "LTIMES.hpp"

#include "RAJA/RAJA.hpp"

#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

#include <iostream>

namespace rajaperf
{
namespace apps
{

using namespace ltimes_idx;

template < size_t tune_idx >
void LTIMES::runOpenMPVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  LTIMES_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
        #pragma omp parallel for
        for (RAJA::Index_type iz = 0; iz < *num_z; ++iz ) {
          IZ z(iz);
          for (IG g(0); g < num_g; ++g ) {
            for (IM m(0); m < num_m; ++m ) {
              for (ID d(0); d < num_d; ++d ) {
                LTIMES_BODY;
              }
            }
          }
        }
        RP_CALI_SUBKERNEL_END("LTIMES_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto ltimes_base_lam = [=](ID d, IZ z, IG g, IM m) {
                               LTIMES_BODY;
                             };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
        #pragma omp parallel for
        for (RAJA::Index_type iz = 0; iz < *num_z; ++iz ) {
          IZ z(iz);
          for (IG g(0); g < num_g; ++g ) {
            for (IM m(0); m < num_m; ++m ) {
              for (ID d(0); d < num_d; ++d ) {
                ltimes_base_lam(d, z, g, m);
              }
            }
          }
        }
        RP_CALI_SUBKERNEL_END("LTIMES_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      if constexpr (tune_idx == 0) {

        auto ltimes_lam = [=](ID d, IZ z, IG g, IM m) {
                            LTIMES_BODY;
                          };

        using EXEC_POL =
          RAJA::KernelPolicy<
            RAJA::statement::For<1, RAJA::omp_parallel_for_exec, // z
              RAJA::statement::For<2, RAJA::seq_exec,            // g
                RAJA::statement::For<3, RAJA::seq_exec,          // m
                  RAJA::statement::For<0, RAJA::seq_exec,        // d
                    RAJA::statement::Lambda<0>
                  >
                >
              >
            >
          >;

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
          RAJA::kernel_resource<EXEC_POL>( RAJA::make_tuple(IDRange(0, *num_d),
                                                            IZRange(0, *num_z),
                                                            IGRange(0, *num_g),
                                                            IMRange(0, *num_m)),
                                           res,
                                           ltimes_lam
                                         );
          RP_CALI_SUBKERNEL_END("LTIMES_1");

        }
        stopTimer();

      } else if constexpr (tune_idx == 1) {

        using launch_policy = RAJA::LaunchPolicy<RAJA::omp_launch_t>;

        using z_policy = RAJA::LoopPolicy<RAJA::omp_for_exec>;

        using g_policy = RAJA::LoopPolicy<RAJA::seq_exec>;

        using m_policy = RAJA::LoopPolicy<RAJA::seq_exec>;

        using d_policy = RAJA::LoopPolicy<RAJA::seq_exec>;

        startTimer();
        // Loop counter increment uses macro to quiet C++20 compiler warning
        for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

          RP_CALI_SUBKERNEL_BEGIN("LTIMES_1");
          RAJA::launch<launch_policy>( res,
              RAJA::LaunchParams(),
              [=] RAJA_HOST_DEVICE(RAJA::LaunchContext ctx) {

                RAJA::loop<z_policy>(ctx, IZRange(0, *num_z),
                  [&](IZ z) {
                    RAJA::loop<g_policy>(ctx, IGRange(0, *num_g),
                      [&](IG g) {
                        RAJA::loop<m_policy>(ctx, IMRange(0, *num_m),
                          [&](IM m) {
                            RAJA::loop<d_policy>(ctx, IDRange(0, *num_d),
                              [&](ID d) {
                                LTIMES_BODY
                              }
                            ); // RAJA::loop<d_policy>
                          }
                        ); // RAJA::loop<m_policy>
                      }
                    ); // RAJA::loop<g_policy>
                  }
                ); // RAJA::loop<z_policy>

              } // outer lambda (ctx)
          );    // RAJA::launch
          RP_CALI_SUBKERNEL_END("LTIMES_1");

        } // loop over kernel reps
        stopTimer();

      }

      break;
    }

    default : {
      getCout() << "\n LTIMES : Unknown variant id = " << vid << std::endl;
    }

  }

}

void LTIMES::defineOpenMPVariantTunings()
{

  for (VariantID vid : {Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP}) {

    if (vid == RAJA_OpenMP) {

      addVariantTuning<&LTIMES::runOpenMPVariant<0>>(
          vid, "kernel");

      addVariantTuning<&LTIMES::runOpenMPVariant<1>>(
          vid, "launch");

    } else {

      addVariantTuning<&LTIMES::runOpenMPVariant<0>>(
          vid, "default");

    }

  }

}

} // end namespace apps
} // end namespace rajaperf

#endif
