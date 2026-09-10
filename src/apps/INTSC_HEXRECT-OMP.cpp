//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other 
// RAJA Project Developers. See top-level LICENSE and COPYRIGHT
// files for dates and other details. No copyright assignment is required
// to contribute to RAJA Performance Suite.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "INTSC_HEXRECT.hpp"

#include "RAJA/RAJA.hpp"

#include "AppsData.hpp"

#include <iostream>

namespace rajaperf
{
namespace apps
{

void INTSC_HEXRECT::runOpenMPVariant(VariantID vid)
{
#if defined(RAJA_ENABLE_OPENMP) && defined(RUN_OPENMP)

  //  We compute each standard intersection within a single thread
  //  to avoid collisions in vv_out hence only distribute different
  //  standard intersections among threads, iend is getActualProblemSize.
  const Index_type run_reps = getRunReps();
  const Index_type ibegin = 0 ;
  const Index_type iend = m_nrecords ;

  INTSC_HEXRECT_DATA_SETUP ;

  auto intsc_hexrect_lam = [=] ( Index_type i ) {
      INTSC_HEXRECT_OMP ( i ) ;
  } ;

  // Insert a warmup call to remove time of initialization of OpenMP
  // that affects the first call to the function.
  Bool_type const do_warmup = true ;
  if ( do_warmup ) {
#pragma omp parallel for
    for (Index_type i = ibegin ; i < iend ; ++i ) {
      INTSC_HEXRECT_OMP( i ) ;
    }
  }

  switch ( vid ) {

    case Base_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
        #pragma omp parallel for
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          INTSC_HEXRECT_OMP( i ) ;
        }
        RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");

      }
      stopTimer();

      break;
    }

    case Lambda_OpenMP : {

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
        #pragma omp parallel for
        for (Index_type i = ibegin ; i < iend ; ++i ) {
          intsc_hexrect_lam(i);
        }
        RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");

      }
      stopTimer();

      break;
    }

    case RAJA_OpenMP : {

      auto res{getHostResource()};

      startTimer();
      // Loop counter increment uses macro to quiet C++20 compiler warning
      for (RepIndex_type irep = 0; irep < run_reps; RP_REPCOUNTINC(irep)) {

        RP_CALI_SUBKERNEL_BEGIN("INTSC_HEXRECT_1");
        RAJA::forall<RAJA::omp_parallel_for_exec>( res,
          RAJA::RangeSegment(ibegin, iend), intsc_hexrect_lam);
        RP_CALI_SUBKERNEL_END("INTSC_HEXRECT_1");

      }
      stopTimer();

      break;
    }

    default : {
      getCout() << "\n  INTSC_HEXRECT : Unknown OpenMP variant id = " << vid << std::endl;
    }

  }

#else
  RAJA_UNUSED_VAR(vid);
#endif
}

RAJAPERF_DEFAULT_TUNING_DEFINE_BOILERPLATE(INTSC_HEXRECT, OpenMP, Base_OpenMP, Lambda_OpenMP, RAJA_OpenMP)

} // end namespace apps
} // end namespace rajaperf
