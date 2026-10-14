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


void QUADRATURE_LOOP::runSeqVariant(VariantID vid)
{
  const Index_type run_reps = getRunReps();

  QUADRATURE_LOOP_DATA_SETUP;

#if defined(RUN_RAJA_SEQ)
  auto quadratureloop_lam = [=](Index_type zone, Index_type q) {
                            QUADRATURE_LOOP_BODY;
                          };
#endif

  switch ( vid ) {

    case Base_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type zone = 0; zone < num_zones; ++zone ) {
          for (Index_type q = 0; q < 27; ++q ) {
            QUADRATURE_LOOP_BODY;
          }
        }

      }
      stopTimer();

      break;
    }

#if defined(RUN_RAJA_SEQ)
    case Lambda_Seq : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        for (Index_type zone = 0; zone < num_zones; ++zone ) {
          for (Index_type q = 0; q < 27; ++q ) {
            quadratureloop_lam(zone, q);
          }
        }

      }
      stopTimer();

      break;
    }

    case RAJA_Seq : {

      using EXEC_POL = RAJA::fornest_basic_seq_2d<RAJA::seq_exec>;

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
#endif // RUN_RAJA_SEQ

    default : {
      getCout() << "\n  QUADRATURE_LOOP : Unknown variant id = " << vid << std::endl;
    }

  }
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(QUADRATURE_LOOP, Seq, Base_Seq, Lambda_Seq, RAJA_Seq)

} // end namespace basic
} // end namespace rajaperf
