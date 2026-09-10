//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "GEN_LIN_RECUR.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{


void GEN_LIN_RECUR::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();

  GEN_LIN_RECUR_DATA_SETUP;

  auto genlinrecur_lam1 = [=](Index_type k) {
                            GEN_LIN_RECUR_BODY1;
                          };
  auto genlinrecur_lam2 = [=](Index_type i) {
                            GEN_LIN_RECUR_BODY2;
                          };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
        #pragma omp parallel for
        for (Index_type k = 0; k < N; ++k ) {
          GEN_LIN_RECUR_BODY1;
        }
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
        #pragma omp parallel for
        for (Index_type i = 1; i < N+1; ++i ) {
          GEN_LIN_RECUR_BODY2;
        }
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
        #pragma omp parallel for
        for (Index_type k = 0; k < N; ++k ) {
          genlinrecur_lam1(k);
        }
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
        #pragma omp parallel for
        for (Index_type i = 1; i < N+1; ++i ) {
          genlinrecur_lam2(i);
        }
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(0, N), genlinrecur_lam1);
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_1");

        RP_CALI_SUBKERNEL_BEGIN("GEN_LIN_RECUR_2");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(1, N+1), genlinrecur_lam2);
        RP_CALI_SUBKERNEL_END("GEN_LIN_RECUR_2");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  GEN_LIN_RECUR : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(GEN_LIN_RECUR, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace lcals
} // end namespace rajaperf
