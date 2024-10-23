//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2017-24, Lawrence Livermore National Security, LLC
// and RAJA Performance Suite project contributors.
// See the RAJAPerf/LICENSE file for details.
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


void PRESSURE::runOpenMPVariant(VariantID vid, size_t RAJAPERF_UNUSED_ARG(tune_idx))
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
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            PRESSURE_BODY1;
          }

          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            PRESSURE_BODY2;
          }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        #pragma omp parallel
        {

          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            pressure_lam1(i);
          }

          #pragma omp for schedule(static) nowait
          for (Index_type i = ibegin; i < iend; ++i ) {
            pressure_lam2(i);
          }

        } // end omp parallel region

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      for (RepIndex_type irep = 0; irep < run_reps; ++irep) {

        RAJA::region<RAJA::omp_parallel_region>( [=]() {

          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), pressure_lam1);

          RAJA::forall< RAJA::omp_for_nowait_static_exec< > >( res,
            RAJA::RangeSegment(ibegin, iend), pressure_lam2);

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

} // end namespace apps
} // end namespace rajaperf
