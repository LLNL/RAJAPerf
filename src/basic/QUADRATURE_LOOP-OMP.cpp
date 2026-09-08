//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "QUADRATURE_LOOP.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void QUADRATURE_LOOP::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)
  const Index_type run_reps = getRunReps();

  QUADRATURE_LOOP_DATA_SETUP;

  auto quadratureloop_lam = [=](Index_type zone, Index_type q) {
                            QUADRATURE_LOOP_BODY;
                          };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel for
        for (Index_type zone = 0; zone < num_zones; ++zone ) {
          for (Index_type q = 0; q < 27; ++q ) {
            QUADRATURE_LOOP_BODY;
          }
        }

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel for
        for (Index_type zone = 0; zone < num_zones; ++zone ) {
          for (Index_type q = 0; q < 27; ++q ) {
            quadratureloop_lam(zone, q);
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      using EXEC_POL = RAJA::fornest_basic_omp_outer_2d<RAJA::seq_exec>;

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::fornest(EXEC_POL {},
                      RAJA::range(num_zones), RAJA::range(27),
                      quadratureloop_lam);

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  QUADRATURE_LOOP : Unknown variant id = " << vid << std::endl;
    }

  }
#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(QUADRATURE_LOOP, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace basic
} // end namespace rajaperf
