//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "PRESSURE.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{


void PRESSURE::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  PRESSURE_DATA_SETUP;

  auto pressure_lam1 = [=](Index_type i) {
                         PRESSURE_BODY1;
                       };
  auto pressure_lam2 = [=](Index_type i) {
                         PRESSURE_BODY2;
                       };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel
        {

          RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            PRESSURE_BODY1;
          }
          RP_CALI_SUBKERNEL_END("PRESSURE_1");

          RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            PRESSURE_BODY2;
          }
          RP_CALI_SUBKERNEL_END("PRESSURE_2");

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        #pragma omp parallel
        {

          RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            pressure_lam1(i);
          }
          RP_CALI_SUBKERNEL_END("PRESSURE_1");

          RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            pressure_lam2(i);
          }
          RP_CALI_SUBKERNEL_END("PRESSURE_2");

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RP_CALI_SUBKERNEL_BEGIN("PRESSURE_1");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);
          RP_CALI_SUBKERNEL_END("PRESSURE_1");

          RP_CALI_SUBKERNEL_BEGIN("PRESSURE_2");
          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);
          RP_CALI_SUBKERNEL_END("PRESSURE_2");

        }); // end omp parallel region

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  PRESSURE : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(PRESSURE, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace apps
} // end namespace rajaperf
