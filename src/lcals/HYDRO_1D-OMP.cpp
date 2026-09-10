//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "HYDRO_1D.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace lcals
{


void HYDRO_1D::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0;
  const Index_type iend = getActualProblemSize();

  HYDRO_1D_DATA_SETUP;

  auto hydro1d_lam = [=](Index_type i) {
                       HYDRO_1D_BODY;
                     };

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_1D_1");
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          HYDRO_1D_BODY;
        }
        RP_CALI_SUBKERNEL_END("HYDRO_1D_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_1D_1");
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          hydro1d_lam(i);
        }
        RP_CALI_SUBKERNEL_END("HYDRO_1D_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("HYDRO_1D_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), hydro1d_lam);
        RP_CALI_SUBKERNEL_END("HYDRO_1D_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  HYDRO_1D : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(HYDRO_1D, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace lcals
} // end namespace rajaperf
