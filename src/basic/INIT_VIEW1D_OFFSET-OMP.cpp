//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INIT_VIEW1D_OFFSET.hpp"

#include "RAJA/RAJA.hpp"

#include <iostream>

namespace rajaperf
{
namespace basic
{


void INIT_VIEW1D_OFFSET::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 1;
  const Index_type iend = getActualProblemSize()+1;

  INIT_VIEW1D_OFFSET_DATA_SETUP;

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          INIT_VIEW1D_OFFSET_BODY;
        }
        RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      auto initview1doffset_base_lam = [=](Index_type i) {
                                         INIT_VIEW1D_OFFSET_BODY;
                                       };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
        #pragma omp parallel for
        for (Index_type i = ibegin; i < iend; ++i ) {
          initview1doffset_base_lam(i);
        }
        RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      INIT_VIEW1D_OFFSET_VIEW_RAJA;

      auto initview1doffset_lam = [=](Index_type i) {
                                    INIT_VIEW1D_OFFSET_BODY_RAJA;
                                  };

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INIT_VIEW1D_OFFSET_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), initview1doffset_lam);
        RP_CALI_SUBKERNEL_END("INIT_VIEW1D_OFFSET_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  INIT_VIEW1D_OFFSET : Unknown variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INIT_VIEW1D_OFFSET, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace basic
} // end namespace rajaperf
